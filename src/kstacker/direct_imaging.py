"""Direct-imaging data preparation, noise calibration and likelihoods.

Instrument objects encapsulate their own observing epochs and photometry.
No sampler state is stored here, so additional independent data likelihoods
can be assembled alongside direct imaging at the posterior level.
"""
from __future__ import annotations
import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple, Union, Mapping, Any
from dataclasses import dataclass
import h5py
import numpy as np
from scipy.ndimage import map_coordinates, distance_transform_edt
from scipy.signal import convolve2d
from scipy.stats import t as scipy_student_t, theilslopes
from astropy.io import fits
from astropy.nddata import block_replicate
from astropy.time import Time
from photutils.aperture import CircularAperture, aperture_photometry

# MAD-to-standard-deviation conversion for Gaussian noise.
_RADIAL_RECAL_MAD_TO_SIGMA = 1.482602218505602
try:
    from .utils import Params
    from .orbital_geometry import _equinoctial_sky_tracks
    from .console import _print_section, console_progress as tqdm, console_status
except ImportError:
    from utils import Params
    from orbital_geometry import _equinoctial_sky_tracks
    from console import _print_section, console_progress as tqdm, console_status

@dataclass
class Instrument:
    """
    Per-instrument runtime context.

    Everything that is *specific* to one instrument lives here:
      - time sampling,
      - image geometry and photometry backend,
      - radial background and noise profiles,
      - geometric masks (IWA / OWA),
      - AU-to-pixel conversion.

    The MCMC log-posterior accepts a *list* of Instrument objects and sums
    their contributions.  With one instrument, only that instrument contributes to the posterior.
    """

    # Human-readable instrument label used in console output.
    name: str

    # -------------------------------------------------------------------------
    # Pixel geometry
    # -------------------------------------------------------------------------
    size: int
    """
    Native image size in pixels (the images are assumed square: size × size).
    """

    scale: float
    """
    Conversion factor [AU → native pixels].
    Computed as:  scale = 1000 [mas/arcsec] / (dist [pc] × resol [mas/px])
    where dist is the stellar distance and resol is the plate scale.
    """

    upsampling_factor: float
    """
    Upsampling factor used when producing the "convolve" images.
    Not used by the "aperture" or "paco" backends, but stored for
    completeness.
    """

    fwhm: Optional[float]
    """
    PSF full-width at half-maximum [native pixels].
    Required by the "aperture" backend (aperture radius = fwhm).
        """

    # -------------------------------------------------------------------------
    # Coronagraph masks (soft: zero photometric contribution, not hard rejection)
    # -------------------------------------------------------------------------
    r_mask: Optional[float]
    """
    Inner Working Angle (IWA) [native pixels].
    Epochs where the predicted planet position falls at r ≤ r_mask contribute
    zero flux/SNR to the likelihood (validpix zeroing).
    If None, no inner mask is applied.
    """

    r_mask_ext: Optional[float]
    """
    Outer Working Angle (OWA) [native pixels].
    Epochs where the predicted position falls at r ≥ r_mask_ext contribute
    zero flux/SNR to the likelihood.
    If None, no outer mask is applied.
    """

    # -------------------------------------------------------------------------
    # Time sampling
    # -------------------------------------------------------------------------
    t_ref: float
    """
    Reference epoch [years] used in the definition of λ0.
    In a multi-instrument run, all instruments should share the same t_ref
    so that the mean longitude λ0 has a consistent physical meaning.
    """

    ts: np.ndarray
    """
    1-D array of shape (K,) with the observation times [years] for this
    instrument.  K may differ between instruments.
    """

    # -------------------------------------------------------------------------
    # Photometry backend
    # -------------------------------------------------------------------------
    photometry_method: str
    """
    Which photometry backend to use for this instrument.
    Valid values: "convolve", "aperture", "paco".

    "convolve":
        Read the nearest pixel in the upsampled image (images_up).
        Requires images_up to be provided.

    "aperture":
        Perform circular aperture photometry on native images (images_native)
        with radius = fwhm.  Requires images_native and fwhm.

    "paco":
        Read upstream PACO alpha_hat and var_alpha.  Optional radial noise
        recalibration may inflate var_alpha by g(d)^2 before interpolation.
        No classical aperture/convolution background estimator is applied.
        Supports positive_snr_profile, signed_snr, and flux likelihoods.
    """

    images_up: Optional[np.ndarray]
    """
    Upsampled (convolved / matched-filtered) images, shape (K, H_up, W_up).
    Required when photometry_method = "convolve".
    May be None for other backends.
    """

    images_native: Optional[np.ndarray]
    """
    Native (non-upsampled) preprocessed images, shape (K, size, size).
    Required when photometry_method = "aperture".
        """

    paco_alpha_maps: Optional[np.ndarray]
    """
    PACO per-epoch flux-estimator maps, shape (K, size, size).
    Required when photometry_method = "paco"; otherwise None.
    """

    paco_var_alpha_maps: Optional[np.ndarray]
    """
    PACO per-epoch variance maps for alpha_hat.  In native/no-interpolation mode
    the shape is (K, size, size).  With PACO interpolation enabled the spatial
    dimensions are the cached oversampled grid.
    """

    # -------------------------------------------------------------------------
    # Uniform internal arrays
    # -------------------------------------------------------------------------
    xgrid: np.ndarray
    """Internal radius axis retained for a uniform Instrument interface."""

    bkg: np.ndarray
    """Internal placeholder array; classical background comes from local rings."""

    noise: np.ndarray
    """Internal placeholder array; classical noise comes from local rings."""


    bgnoise_cfg: Optional[dict] = None
    """
    Optional background/noise estimation configuration.

    Estimator
    ---------
    "local_aperture_ring":
        For each tested orbital position and each epoch,
        the code estimates the background and the noise from an annulus of
        *independent reference apertures* placed at the same separation from
        the star.

        The reference aperture that would contain the tested source is
        excluded, together with neighbouring apertures inside an exclusion
        arc-length expressed in units of FWHM.  This avoids contaminating the
        background/noise estimate with the planet signal and with the nearby
        convolved lobes or residual correlated structure.

        The exact scalar extracted in each reference aperture depends on the
        photometry backend:
          - "convolve": read the local scalar on the upsampled convolved image;
          - "aperture": perform native-image aperture photometry with the same
                        aperture radius as the science measurement.
    """

    local_bkg_maps: Optional[np.ndarray] = None
    """
    Optional per-epoch 2-D background maps used when the local aperture-ring
    estimator is precomputed on the native pixel grid.
    Shape: (K, size, size).
    """

    local_noise_maps: Optional[np.ndarray] = None
    """
    Optional per-epoch 2-D noise maps used when the local aperture-ring
    estimator is precomputed on the native pixel grid.
    Shape: (K, size, size).
    """

    paco_interpolator: str = "none"
    """
    PACO map sampling mode.  "none" preserves the  deterministic
    nearest-native-pixel behaviour.  Any other supported value means that
    alpha_hat and var_alpha were interpolated onto an oversampled cached grid.
    """

    paco_oversampling: int = 1
    """
    Spatial oversampling factor of the PACO maps used by the likelihood.
    This is forced to 1 when paco_interpolator == "none".
    """

    likelihood_mode: str = "signed_snr"
    """Statistical criterion selected for this instrument's observations."""

    noise_floor: Optional[float] = None
    """Instrument-specific lower bound on standard deviation, in its data units."""

    flux_index: Optional[int] = None
    """Column of the sampled flux for this instrument; None means fixed flux."""

    flux_fixed: float = 1.0
    """Known, fixed planet amplitude in this instrument's calibrated units."""

    flux_init: float = 1.0
    """Starting amplitude for grid scoring and optional MCMC sampling."""

    flux_bounds: Tuple[float, float] = (1.0e-4, 1.0e2)
    """Allowed interval of this instrument's sampled flux."""

    flux_prior: str = "uniform"
    """Independent prior for this instrument's sampled flux."""

    flux_spread: float = 0.2
    """Relative initial walker scatter for this instrument's flux."""

    noise_products: Optional[dict] = None
    """Locations of the cached noise products (diagnostic figures only)."""

    measurement_missing_maps: Optional[np.ndarray] = None
    """
    Per-epoch indicator of science measurements that use no-data pixels
    (`background_noise.missing_data: "zero"`), built from `images_native` by
    `configure_instrument_statistics` (see `_measurement_missing_grid`).
      - "convolve": boolean (K, H_up, W_up), on the grid read by the science
        photometry: True where the convolution disk overlaps a no-data pixel;
      - "aperture": float (K, size, size), the native no-data indicator.
    None means "unknown": the likelihood then keeps the strict bilinear rule
    of iterations 02-06 (`_bilinear_support_valid`).
    """


def _get(d: Optional[dict], key: str, default: Any) -> Any:
    """
    Safe dictionary getter that treats None values as missing.

    Returns `default` when:
      - d is None,
      - key is absent from d, or
      - d[key] is None.

    This avoids the common pitfall of YAML keys that are present but set to
    null (which YAML parses as Python None).
    """
    try:
        val = d.get(key, default)
    except AttributeError:
        return default
    return default if val is None else val


def _noise_floor_build_value(root: Mapping[str, Any]) -> float:
    """
    Return the temporary floor used while building background/noise products.

    A fixed floor can be applied immediately. In automatic mode the final floor
    is estimated only after the noise products have been loaded, so map building
    uses zero here and the science likelihood applies the resolved floor later.
    """
    cfg = root.get("noise_floor", {}) or {}
    if isinstance(cfg, (int, float)):
        return max(0.0, float(cfg))

    mode = str(cfg.get("mode", "auto") or "auto").strip().lower()
    if mode == "fixed":
        value = float(cfg.get("value", 0.0))
        if value <= 0.0:
            raise ValueError("noise_floor.value must be > 0 when mode='fixed'.")
        return value

    if mode != "auto":
        raise ValueError("noise_floor.mode must be 'auto' or 'fixed'.")

    return 0.0


def _collect_noise_samples(instruments: Sequence["Instrument"]) -> np.ndarray:
    """Collect finite positive sigma values from the supplied instruments."""
    chunks = []

    for inst in instruments:
        method = str(inst.photometry_method).lower()

        if method == "paco":
            if inst.paco_var_alpha_maps is None:
                continue
            var = np.asarray(inst.paco_var_alpha_maps, dtype=float)
            sigma = np.sqrt(var[np.isfinite(var) & (var > 0.0)])
        elif inst.local_noise_maps is not None:
            sigma = np.asarray(inst.local_noise_maps, dtype=float).ravel()
            sigma = sigma[np.isfinite(sigma) & (sigma > 0.0)]
        else:
            sigma = np.asarray(inst.noise, dtype=float).ravel()
            sigma = sigma[np.isfinite(sigma) & (sigma > 0.0)]

        if sigma.size:
            chunks.append(np.asarray(sigma, dtype=float))

    if not chunks:
        return np.empty(0, dtype=float)

    return np.concatenate(chunks)


def _resolve_noise_floor(
    root: Mapping[str, Any],
    instruments: Sequence["Instrument"],
    *,
    verbose: bool = True,
) -> float:
    """
    Resolve a minimum sigma from the supplied dataset(s).

    Modes
    -----
    fixed
        Use the exact positive value supplied by the user.

    auto
        Estimate a conservative numerical floor from the data:

            typical_sigma = median(all finite positive sigma samples)
            noise_floor   = fraction_of_median * typical_sigma

        The median is robust to a modest number of unusually large or small
        samples. The default fraction (1e-3) keeps the floor far below the
        typical measured noise while preventing nearly-zero sigma values from
        dominating a likelihood.

    The floor is applied only as a lower bound to sigma. It does not rescale the
    noise field; radial rescaling is handled separately by
    radial_noise_recalibration.
    """
    cfg = root.get("noise_floor", {}) or {}

    if isinstance(cfg, (int, float)):
        mode = "fixed"
        value = float(cfg)
        cfg = {"mode": mode, "value": value}
    else:
        mode = str(cfg.get("mode", "auto") or "auto").strip().lower()

    if mode == "fixed":
        floor = float(cfg.get("value", 0.0))
        if not np.isfinite(floor) or floor <= 0.0:
            raise ValueError("noise_floor.value must be a finite number > 0.")
        source = "user value"
        typical = None

    elif mode == "auto":
        fraction = float(cfg.get("fraction_of_median", 1.0e-3))
        min_samples = max(1, int(cfg.get("min_samples", 100)))

        if not np.isfinite(fraction) or fraction <= 0.0:
            raise ValueError("noise_floor.fraction_of_median must be > 0.")

        samples = _collect_noise_samples(instruments)
        if samples.size < min_samples:
            raise RuntimeError(
                "Automatic noise-floor estimation found only "
                f"{samples.size} finite positive sigma samples; "
                f"noise_floor.min_samples={min_samples}. "
                "Use noise_floor.mode='fixed' or provide more valid noise data."
            )

        typical = float(np.median(samples))
        if not np.isfinite(typical) or typical <= 0.0:
            raise RuntimeError("Could not estimate a positive typical noise level.")

        floor = max(
            float(np.finfo(np.float64).eps),
            fraction * typical,
        )
        source = f"{fraction:g} x median sigma"

    else:
        raise ValueError("noise_floor.mode must be 'auto' or 'fixed'.")

    if verbose:
        print()
        print("-" * 110)
        print("NOISE FLOOR")
        print("-" * 110)
        print()
        print(f"  mode                 : {mode}")
        print(f"  source               : {source}")
        if typical is not None:
            print(f"  robust median sigma  : {typical:.6g}")
        print(f"  applied sigma floor  : {floor:.6g}")
        print()
        print("  The floor only protects against unrealistically small sigma values.")
        print("  Radial noise recalibration, when enabled, is applied separately.")
        print()

    return float(floor)



def _invalid_noise_summary(instrument: "Instrument") -> str:
    """Diagnostic text: invalid sigma pixels / unusable sample area in the working annulus."""
    maps = instrument.local_noise_maps
    if maps is None or str(instrument.photometry_method).lower() == "paco":
        return ""
    maps = np.asarray(maps, dtype=float)
    n = maps.shape[1]
    yy, xx = np.mgrid[:n, :maps.shape[2]]
    r = np.hypot(xx - (maps.shape[2] - 1) / 2.0, yy - (n - 1) / 2.0)
    work = np.ones_like(r, dtype=bool)
    if instrument.r_mask is not None:
        work &= r > float(instrument.r_mask)
    if instrument.r_mask_ext is not None:
        work &= r < float(instrument.r_mask_ext)
    bad = ~(np.isfinite(maps) & (maps > 0.0))
    n_bad = np.sum(bad & work[None], axis=(1, 2))
    # A 2x2 cell is unusable for a sample if any of its corners is invalid.
    cell_bad = bad[:, :-1, :-1] | bad[:, 1:, :-1] | bad[:, :-1, 1:] | bad[:, 1:, 1:]
    rc = np.hypot(xx[:-1, :-1] + 0.5 - (maps.shape[2] - 1) / 2.0,
                  yy[:-1, :-1] + 0.5 - (n - 1) / 2.0)
    work_c = np.ones_like(rc, dtype=bool)
    if instrument.r_mask is not None:
        work_c &= rc > float(instrument.r_mask)
    if instrument.r_mask_ext is not None:
        work_c &= rc < float(instrument.r_mask_ext)
    frac = np.sum(cell_bad & work_c[None], axis=(1, 2)) / max(int(np.sum(work_c)), 1)
    text = (
        f"[noise] {instrument.name}: invalid sigma pixels (sigma = 0) in working annulus "
        f"r_mask < r < r_mask_ext ({int(np.sum(work))} px): "
        f"{int(n_bad.min())} .. {int(n_bad.max())} per epoch (total {int(n_bad.sum())}); "
    )
    rejected = _rejected_area_fractions(instrument)
    if rejected is None:
        return text + (
            f"area where samples are rejected by the strict bilinear rule: "
            f"{100 * float(frac.min()):.1f} .. {100 * float(frac.max()):.1f} % per epoch"
        )
    new_frac, strict_frac = rejected
    return text + (
        f"area where samples are rejected (measurement on no-data pixels, or no valid "
        f"bilinear neighbour): {100 * float(new_frac.min()):.1f} .. "
        f"{100 * float(new_frac.max()):.1f} % per epoch "
        f"(strict bilinear rule of iterations 02-06: {100 * float(strict_frac.min()):.1f} .. "
        f"{100 * float(strict_frac.max()):.1f} %)"
    )


def _rejected_area_fractions(instrument: "Instrument") -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    Diagnostic only: fraction of the working annulus where a sample is rejected.

    Positions are the centres of the upsampled sub-pixels (the grid read by the
    "convolve" photometry), r_mask < r < r_mask_ext.  Returns per-epoch
    fractions for the current rule (measurement on no-data pixels, or no valid
    bilinear neighbour) and for the strict rule of iterations 02-06 (any
    weighted neighbour invalid).  None when the measurement maps are absent or
    the photometry is not "convolve".
    """
    maps = instrument.local_noise_maps
    missing = instrument.measurement_missing_maps
    if (
        maps is None or missing is None
        or str(instrument.photometry_method).lower() != "convolve"
    ):
        return None
    maps = np.asarray(maps)
    missing = np.asarray(missing, dtype=bool)
    K, n, m = maps.shape
    factor = max(1, int(round(float(instrument.upsampling_factor))))
    if missing.shape != (K, n * factor, m * factor):
        return None
    # Sub-pixel centres: `_native_to_upsampled_index` maps them back exactly.
    xs = (np.arange(m * factor) + 0.5) / factor - 0.5
    ys = (np.arange(n * factor) + 0.5) / factor - 0.5
    xx, yy = np.meshgrid(xs, ys)
    r = np.hypot(xx - (m - 1) / 2.0, yy - (n - 1) / 2.0)
    work = np.ones_like(r, dtype=bool)
    if instrument.r_mask is not None:
        work &= r > float(instrument.r_mask)
    if instrument.r_mask_ext is not None:
        work &= r < float(instrument.r_mask_ext)
    x = xx[work]
    y = yy[work]
    n_work = max(int(x.size), 1)
    new_frac = np.zeros(K, dtype=float)
    strict_frac = np.zeros(K, dtype=float)
    for kk in range(K):
        k_idx = np.full(x.shape, kk, dtype=np.int64)
        support_ok, _, _ = _bilinear_interp_valid_neighbours(
            maps, maps, x, y, k_idx
        )
        meas_ok = ~missing[kk][work]
        new_frac[kk] = np.sum(~(support_ok & meas_ok)) / n_work
        strict_frac[kk] = np.sum(~_bilinear_support_valid(maps, x, y, k_idx)) / n_work
    return new_frac, strict_frac


def _build_measurement_missing_maps(instrument: "Instrument") -> Optional[np.ndarray]:
    """
    Per-epoch maps telling whether the science measurement uses no-data pixels.

    Built from the native images with the same kernel as the convolved images
    (`_measurement_missing_grid`).  With `missing_data: "none"` no pixel is
    treated as missing (all-False maps).  None for PACO, without native images
    or without precomputed local maps (the likelihood then keeps its legacy
    strict rule).
    """
    method = str(instrument.photometry_method).lower()
    if method not in ("convolve", "aperture"):
        return None
    if instrument.images_native is None or instrument.local_noise_maps is None:
        return None
    if method == "convolve" and instrument.images_up is None:
        return None
    cfg = instrument.bgnoise_cfg or {}
    mode = str(cfg.get("missing_data", "zero")).lower()
    images_native = np.asarray(instrument.images_native)
    K = int(images_native.shape[0])
    if method == "convolve":
        shape = tuple(np.shape(instrument.images_up))
    else:
        shape = tuple(images_native.shape)
    if mode != "zero":
        return np.zeros(shape, dtype=bool if method == "convolve" else float)
    if instrument.fwhm is None:
        return None
    out = np.stack([
        _measurement_missing_grid(
            photometry_method=method,
            image_native=images_native[kk],
            upsampling_factor=instrument.upsampling_factor,
            fwhm=float(instrument.fwhm),
        )
        for kk in range(K)
    ], axis=0)
    if out.shape != shape:
        raise ValueError(
            f"{instrument.name}: no-data measurement maps {out.shape} do not match "
            f"the science images {shape}."
        )
    return out


def configure_instrument_statistics(root: Mapping[str, Any], instruments: Sequence[Instrument], *, verbose: bool = True) -> list[str]:
    """Resolve independent noise floors and flux amplitudes for each dataset.

    Parameters are read from `instruments[].noise_floor` and
    `instruments[].flux`; absent entries inherit the global `noise_floor`
    and `mcmc.fp_*` defaults.  Flux dimensions follow the order of the
    flux-mode instruments with sampling enabled, starting at column seven.
    The returned labels match the extra MCMC chain columns.

    The noise floor is computed using *only* this instrument's noise pixels.
    It therefore remains meaningful if two instruments use different units.
    """
    configs = root.get("instruments", [])
    if len(configs) != len(instruments):
        raise ValueError("Instrument configuration and loaded dataset count differ")
    defaults = root.get("mcmc", {}) or {}
    labels = []
    seen_names = set()
    for i, (configuration, instrument) in enumerate(zip(configs, instruments)):
        if instrument.name in seen_names:
            raise ValueError(f"Duplicate instrument name {instrument.name!r}")
        seen_names.add(instrument.name)
        local_root = dict(root)
        local_root["noise_floor"] = configuration.get("noise_floor", root.get("noise_floor", {}))
        instrument.noise_floor = _resolve_noise_floor(local_root, [instrument], verbose=False)
        # Since iteration 07: validity of the measurement at the exact sample
        # position (rule "a" of rapports/notes_bruit_masque.md), read by the
        # likelihood.  Built here because init, mcmc and plot all call this.
        instrument.measurement_missing_maps = _build_measurement_missing_maps(instrument)
        if verbose:
            samples = _collect_noise_samples([instrument])
            median_sigma = float(np.median(samples)) if samples.size else float("nan")
            shape = (
                None if instrument.local_noise_maps is None
                else tuple(np.shape(instrument.local_noise_maps))
            )
            print(
                f"[noise] {instrument.name}: sigma floor = {instrument.noise_floor:.6g} "
                f"(median sigma of this instrument = {median_sigma:.6g}, "
                f"{samples.size} samples, noise maps shape = {shape})"
            )
            summary = _invalid_noise_summary(instrument)
            if summary:
                print(summary)

        instrument.flux_index = None
        if _normalize_likelihood_mode(instrument.likelihood_mode) != "flux":
            continue
        flux_cfg = configuration.get("flux", {}) or {}
        if not isinstance(flux_cfg, Mapping):
            raise ValueError(f"{instrument.name}: flux must be a mapping")
        bounds = tuple(map(float, flux_cfg.get("bounds", defaults.get("fp_bounds", (1e-4, 1e2)))))
        if len(bounds) != 2 or not all(np.isfinite(bounds)) or not (bounds[0] < bounds[1]):
            raise ValueError(f"{instrument.name}: flux.bounds must be two increasing finite numbers")
        prior = str(flux_cfg.get("prior", defaults.get("fp_prior", "uniform"))).lower()
        if prior not in {"uniform", "loguniform"}:
            raise ValueError(f"{instrument.name}: flux.prior must be uniform or loguniform")
        if prior == "loguniform" and bounds[0] <= 0:
            raise ValueError(f"{instrument.name}: loguniform flux requires a positive lower bound")
        initial = float(flux_cfg.get("init", defaults.get("fp_init", 1.0)))
        if not np.isfinite(initial) or not bounds[0] <= initial <= bounds[1]:
            raise ValueError(f"{instrument.name}: flux.init must lie within flux.bounds")
        spread = float(flux_cfg.get("spread", (root.get("init_spread", {}) or {}).get("fp", 0.2)))
        if not np.isfinite(spread) or spread < 0:
            raise ValueError(f"{instrument.name}: flux.spread must be finite and nonnegative")
        raw_sample = flux_cfg.get("sample", defaults.get("sample_fp", False))
        if isinstance(raw_sample, str):
            if raw_sample.strip().lower() not in {"true", "false", "yes", "no", "1", "0"}:
                raise ValueError(f"{instrument.name}: invalid flux.sample")
            sampled = raw_sample.strip().lower() in {"true", "yes", "1"}
        else:
            sampled = bool(raw_sample)
        instrument.flux_bounds = bounds
        instrument.flux_prior = prior
        instrument.flux_init = initial
        instrument.flux_fixed = initial
        instrument.flux_spread = spread
        if sampled:
            instrument.flux_index = 7 + len(labels)
            labels.append(f"fp_{instrument.name}")
        if verbose:
            status = "sampled" if sampled else "fixed"
            print(f"[flux] {instrument.name}: {status}, init={initial:g}, bounds={bounds}")
    return labels


def instrument_flux_values(theta: np.ndarray, instrument: Instrument) -> np.ndarray:
    """Return each orbit's amplitude in the instrument's own flux units."""
    samples = np.asarray(theta, dtype=float)
    if samples.ndim != 2:
        raise ValueError("theta must have shape (N, n_parameters)")
    index = instrument.flux_index
    if index is None:
        return np.full(samples.shape[0], instrument.flux_fixed, dtype=float)
    if index >= samples.shape[1]:
        raise ValueError(f"Missing sampled flux column {index} for {instrument.name}")
    return samples[:, index]


def _resolve_path_from_yaml(base_dir: Path, value: Optional[str], default: str) -> Path:
    """Resolve a path relative to the directory that contains the YAML file."""
    raw = default if value in (None, "") else str(value)
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def _pattern_to_epoch_regex(pattern: str) -> re.Pattern:
    """Convert a filename pattern containing {k} into an epoch-index regex."""
    token = "__EPOCH_INDEX__"
    pattern_tokenized = re.sub(r"\{k(?::[^}]*)?\}", token, str(pattern))
    escaped = re.escape(pattern_tokenized)
    escaped = escaped.replace(re.escape(token), r"(?P<k>\d+)")
    return re.compile(rf"^{escaped}$")


def _discover_epoch_files(directory: Path, pattern: str) -> dict[int, Path]:
    """Return {epoch_index: path} for files matching a {k} filename pattern."""
    directory = Path(directory)
    if not directory.is_dir():
        return {}
    regex = _pattern_to_epoch_regex(pattern)
    glob_pattern = re.sub(r"\{k(?::[^}]*)?\}", "*", str(pattern))
    found: dict[int, Path] = {}
    for path in sorted(directory.glob(glob_pattern)):
        match = regex.match(path.name)
        if match:
            found[int(match.group("k"))] = path
    return found


def _require_contiguous_epochs(files: Mapping[int, Path], label: str) -> list[Path]:
    """Validate 0..K-1 epoch numbering and return paths in epoch order."""
    if not files:
        raise FileNotFoundError(f"No {label} files were found.")
    indices = sorted(files)
    expected = list(range(indices[-1] + 1))
    if indices != expected:
        raise ValueError(
            f"{label} epoch indices are not contiguous from 0: found {indices}."
        )
    return [files[k] for k in expected]


def _square_fits_shape(path: Path) -> int:
    """Read a 2-D square FITS product and return its side length."""
    shape = tuple(np.asarray(fits.getdata(path, memmap=True)).shape)
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"Expected a square 2-D FITS image, got {shape} in {path}.")
    return int(shape[0])


def _candidate_parent_dirs(base_dir: Path, max_up: int = 3) -> list[Path]:
    """Directories searched for observation metadata, nearest directory first."""
    out = []
    current = Path(base_dir).resolve()
    for _ in range(max_up + 1):
        if current not in out:
            out.append(current)
        if current.parent == current:
            break
        current = current.parent
    return out


def _numeric_time_vector(value: Any, n_epochs: int) -> Optional[np.ndarray]:
    """Parse exactly one numeric time value per epoch."""
    if isinstance(value, str):
        text_value = value.strip().strip("[]()")
        tokens = [
            token
            for token in re.split(r"[+,\s;]+", text_value)
            if token
        ]
        try:
            arr = np.asarray([float(token) for token in tokens], dtype=float)
        except Exception:
            return None
    elif isinstance(value, (list, tuple, np.ndarray)):
        try:
            arr = np.asarray(value, dtype=float).ravel()
        except Exception:
            return None
    else:
        return None

    if arr.size != int(n_epochs) or not np.all(np.isfinite(arr)):
        return None
    return arr.astype(float)


def _find_time_vector_in_json_object(obj: Any, n_epochs: int) -> Optional[np.ndarray]:
    """
    Find a K-element observation-time vector in nested JSON metadata.

    The parser accepts explicit year keys and the common compact string
    representation such as "0.0+1.5+3.0+4.5".
    """
    preferred = (
        "times_years",
        "time_years",
        "observation_times_years",
        "epoch_times_years",
        "epochs_years",
        "times",
        "epoch_times",
        "time",
    )

    if isinstance(obj, dict):
        normalized = {str(k).strip().lower(): v for k, v in obj.items()}

        for key in preferred:
            if key in normalized:
                arr = _numeric_time_vector(normalized[key], n_epochs)
                if arr is not None:
                    return arr

        epochs = normalized.get("epochs", None)
        if isinstance(epochs, list) and len(epochs) == int(n_epochs):
            values = []
            for item in epochs:
                found = _find_scalar_time_in_json_object(item)
                if found is None or found[1] != "years":
                    values = []
                    break
                values.append(float(found[0]))
            if len(values) == int(n_epochs):
                return np.asarray(values, dtype=float)

        for value in obj.values():
            arr = _find_time_vector_in_json_object(value, n_epochs)
            if arr is not None:
                return arr

    elif isinstance(obj, list):
        for value in obj:
            arr = _find_time_vector_in_json_object(value, n_epochs)
            if arr is not None:
                return arr

    return None


def _times_from_observation_json(base_dir: Path, n_epochs: int) -> Optional[np.ndarray]:
    """Read observation times from the nearest observation_parameters.json."""
    for directory in _candidate_parent_dirs(base_dir):
        path = directory / "observation_parameters.json"
        if not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                obj = json.load(handle)
            arr = _find_time_vector_in_json_object(obj, n_epochs)
            if arr is not None:
                return arr
        except Exception:
            continue
    return None


def _times_from_fits_headers(paths: Sequence[Path]) -> Optional[np.ndarray]:
    """
    Read one timestamp per epoch from FITS headers.

    Relative-year keywords are used directly. MJD/JD/DATE-OBS are converted to
    years relative to the first epoch.
    """
    if not paths:
        return None

    year_keys = ("TIMEYR", "TIME_YR", "T_YEAR", "TYEAR", "EPOCH_YR", "EPOCHYR")
    mjd_keys = ("MJD-OBS", "MJD_OBS", "MJD")
    jd_keys = ("JD-OBS", "JD_OBS", "JD")
    date_keys = ("DATE-OBS", "DATE_OBS")

    headers = []
    try:
        headers = [fits.getheader(path) for path in paths]
    except Exception:
        return None

    for keys in (year_keys,):
        vals = []
        for header in headers:
            value = next((header.get(k) for k in keys if header.get(k) is not None), None)
            try:
                vals.append(float(value))
            except Exception:
                vals = []
                break
        if len(vals) == len(paths) and np.all(np.isfinite(vals)):
            return np.asarray(vals, dtype=float)

    for keys, scale in ((mjd_keys, 365.25), (jd_keys, 365.25)):
        vals = []
        for header in headers:
            value = next((header.get(k) for k in keys if header.get(k) is not None), None)
            try:
                vals.append(float(value))
            except Exception:
                vals = []
                break
        if len(vals) == len(paths) and np.all(np.isfinite(vals)):
            vals = np.asarray(vals, dtype=float)
            return (vals - vals[0]) / scale

    vals = []
    for header in headers:
        value = next((header.get(k) for k in date_keys if header.get(k) is not None), None)
        if value is None:
            vals = []
            break
        try:
            vals.append(float(Time(value).mjd))
        except Exception:
            vals = []
            break
    if len(vals) == len(paths):
        vals = np.asarray(vals, dtype=float)
        return (vals - vals[0]) / 365.25

    return None


def _times_from_exposure_table(base_dir: Path, n_epochs: int) -> Optional[np.ndarray]:
    """
    Read times from exposures_parameters.fits when it contains one row per epoch.

    Tables containing many individual exposures are intentionally ignored rather
    than guessed.
    """
    for directory in _candidate_parent_dirs(base_dir):
        path = directory / "exposures_parameters.fits"
        if not path.is_file():
            continue
        try:
            data = fits.getdata(path)
            names = list(data.dtype.names or [])
            if len(data) != int(n_epochs):
                continue
            upper = {name.upper(): name for name in names}

            for key in ("TIME_YEARS", "TIME_YEAR", "T_YEARS", "EPOCH_YEARS"):
                if key in upper:
                    arr = np.asarray(data[upper[key]], dtype=float).ravel()
                    if arr.size == n_epochs and np.all(np.isfinite(arr)):
                        return arr

            for key in ("MJD", "MJD_OBS", "MJD-OBS"):
                if key in upper:
                    arr = np.asarray(data[upper[key]], dtype=float).ravel()
                    if arr.size == n_epochs and np.all(np.isfinite(arr)):
                        return (arr - arr[0]) / 365.25

            for key in ("JD", "JD_OBS", "JD-OBS"):
                if key in upper:
                    arr = np.asarray(data[upper[key]], dtype=float).ravel()
                    if arr.size == n_epochs and np.all(np.isfinite(arr)):
                        return (arr - arr[0]) / 365.25
        except Exception:
            continue
    return None


def _infer_times_years(
    *,
    base_dir: Path,
    epoch_paths: Sequence[Path],
    n_epochs: int,
    inst_cfg: Mapping[str, Any],
) -> tuple[np.ndarray, str]:
    """
    Infer one observation time per epoch in years.

    Search order:
      1. instruments[].times_years in the YAML;
      2. a K-element vector in a nearby observation_parameters.json;
      3. timestamps in the per-epoch science FITS headers;
      4. a nearby exposure table containing one row per epoch;
      5. epoch_k/ metadata, using one representative timestamp per epoch.

    The returned vector is used directly by the Keplerian propagator.
    """
    explicit = _numeric_time_vector(inst_cfg.get("times_years", None), n_epochs)
    if explicit is not None:
        return explicit, "YAML times_years"

    arr = _times_from_observation_json(base_dir, n_epochs)
    if arr is not None:
        return arr, "observation_parameters.json"

    arr = _times_from_fits_headers(epoch_paths)
    if arr is not None:
        return arr, "FITS headers"

    arr = _times_from_exposure_table(base_dir, n_epochs)
    if arr is not None:
        return arr, "exposures_parameters.fits"

    arr = _times_from_epoch_directories(base_dir, n_epochs)
    if arr is not None:
        return arr, "epoch_k metadata"

    raise ValueError(
        "Observation times could not be inferred from the dataset. "
        "Add instruments[].times_years: [t0, t1, ...] in years."
    )


def _discover_classical_images_dir(base_dir: Path, configured: Optional[str]) -> Path:
    """Find the reduced-image directory used by convolve/aperture."""
    if configured not in (None, ""):
        path = _resolve_path_from_yaml(base_dir, configured, "images")
        if not path.is_dir():
            raise FileNotFoundError(f"Configured images_dir does not exist: {path}")
        return path

    candidates = [
        base_dir / "images",
        base_dir.parent / "images",
    ]
    candidates.extend(sorted(base_dir.glob("Ncomp_adi_*_Ncomp_sdi_*/images")))
    candidates.extend(sorted(base_dir.parent.glob("Ncomp_adi_*_Ncomp_sdi_*/images")))

    for path in candidates:
        if path.is_dir():
            return path.resolve()

    raise FileNotFoundError(
        f"Could not find a reduced-image directory near {base_dir}. "
        "Place the YAML in the reduction directory or set instruments[].images_dir."
    )


def _discover_paco_maps_dir(base_dir: Path, configured: Optional[str]) -> Path:
    """
    Find the directory containing PACO alpha_hat / var_alpha maps.

    Relative paths are resolved safely whether the YAML is placed directly in
    paco_asdi/ or one directory above it.
    """
    base_dir = Path(base_dir).resolve()

    candidates = []

    if configured not in (None, ""):
        raw = Path(str(configured)).expanduser()

        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.append(base_dir / raw)
            candidates.append(base_dir.parent / raw)

            parts = raw.parts
            if base_dir.name == "paco_asdi" and parts and parts[0] == "paco_asdi":
                stripped = Path(*parts[1:]) if len(parts) > 1 else Path(".")
                candidates.insert(0, base_dir / stripped)

    candidates.extend(
        [
            base_dir / "wpca_alpha_var",
            base_dir / "paco_asdi" / "wpca_alpha_var",
            base_dir.parent / "paco_asdi" / "wpca_alpha_var",
        ]
    )

    seen = set()
    for candidate in candidates:
        path = Path(candidate).resolve()
        if path in seen:
            continue
        seen.add(path)
        if path.is_dir():
            return path

    tried = "\n".join(f"  - {path}" for path in seen)
    raise FileNotFoundError(
        "Could not find the PACO alpha_hat / var_alpha directory.\n"
        f"Searched:\n{tried}"
    )


def _infer_convolution_factor(images_dir: Path, resampled_path: Path, native_side: Optional[int]) -> int:
    """Infer the convolve upsampling factor from FITS metadata or image dimensions."""
    header = fits.getheader(resampled_path)
    factor = header.get("FACTOR", None)
    if factor is not None:
        factor = int(round(float(factor)))
        if factor >= 1:
            return factor

    if native_side is not None:
        up_side = _square_fits_shape(resampled_path)
        ratio = up_side / float(native_side)
        rounded = int(round(ratio))
        if rounded >= 1 and np.isclose(ratio, rounded, rtol=0, atol=1e-8):
            return rounded

    raise ValueError(
        f"Could not infer the convolve upsampling factor from {resampled_path}. "
        "The FITS header should contain FACTOR."
    )


def _preprocess_native_image(raw_path: Path, preprocessed_path: Path) -> None:
    """
    Create one native preprocessed FITS image.

    The preprocessing is intentionally simple and deterministic:
      - load the raw reduced image;
      - replace NaNs by zero;
      - if needed, center-crop a rectangular image to a square;
      - save the native square image.

    Existing products are never overwritten.
    """
    raw_path = Path(raw_path)
    preprocessed_path = Path(preprocessed_path)

    image = np.asarray(fits.getdata(raw_path), dtype=float)
    image[~np.isfinite(image)] = 0.0

    ny, nx = image.shape
    if ny != nx:
        side = min(ny, nx)
        cy, cx = ny // 2, nx // 2
        half = side // 2
        if side % 2 == 0:
            image = image[cy-half:cy+half, cx-half:cx+half]
        else:
            image = image[cy-half:cy+half+1, cx-half:cx+half+1]

    preprocessed_path.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(
        preprocessed_path,
        np.asarray(image, dtype=np.float32),
        overwrite=False,
    )


def _build_resampled_image(
    preprocessed_path: Path,
    resampled_path: Path,
    *,
    aperture_radius: float,
    upsampling_factor: int,
) -> None:
    """
    Create one convolved/upsampled photometric image.

    The native image is block-replicated while conserving total flux, then
    convolved with a circular top-hat aperture kernel. The FITS header records
    the kernel radius and upsampling factor so later runs can rediscover them.

    Existing products are never overwritten.
    """
    preprocessed_path = Path(preprocessed_path)
    resampled_path = Path(resampled_path)

    image = np.asarray(fits.getdata(preprocessed_path), dtype=float)
    image[~np.isfinite(image)] = 0.0

    factor = max(1, int(upsampling_factor))
    radius = float(aperture_radius)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("A positive fwhm/aperture radius is required for image preparation.")

    replicated = block_replicate(
        image,
        factor,
        conserve_sum=True,
    )

    mask_size = (int(radius + 0.5) * 2 + 1) * factor
    xx, yy = np.mgrid[:mask_size, :mask_size] - mask_size // 2
    kernel = (
        np.hypot(xx, yy) < radius * factor
    ).astype(float)

    convolved = convolve2d(
        replicated,
        kernel,
        mode="same",
    )

    header = fits.Header(
        {
            "KERNEL": "circle",
            "RADIUS": radius,
            "FACTOR": factor,
        }
    )

    resampled_path.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(
        resampled_path,
        np.asarray(convolved, dtype=np.float32),
        header=header,
        overwrite=False,
    )


def _resolve_preprocessing_factor(
    images_dir: Path,
    root: Mapping[str, Any],
) -> int:
    """
    Resolve the upsampling factor used only when a resampled product must be made.

    Priority:
      1. FACTOR stored in any existing image_*_resampled.fits;
      2. preprocessing.upsampling_factor from the YAML;
      3. the documented default value 5.
    """
    images_dir = Path(images_dir)

    existing = sorted(images_dir.glob("image_*_resampled.fits"))
    for path in existing:
        try:
            factor = int(round(float(fits.getheader(path).get("FACTOR"))))
            if factor >= 1:
                return factor
        except Exception:
            pass

    cfg = root.get("preprocessing", {}) or {}
    factor = int(cfg.get("upsampling_factor", 5))
    if factor < 1:
        raise ValueError("preprocessing.upsampling_factor must be >= 1.")
    return factor


def _ensure_classical_image_products(
    images_dir: Path,
    *,
    fwhm: float,
    root: Mapping[str, Any],
    verbose: bool = True,
) -> tuple[int, int, int]:
    """
    Ensure image_k_preprocessed.fits and image_k_resampled.fits exist.

    Accepted input for an epoch:
      - image_k.fits
      - image_k_preprocessed.fits
      - image_k_resampled.fits

    Missing native preprocessed products are created from image_k.fits.
    Missing resampled products are created from image_k_preprocessed.fits.
    Existing products are reused unchanged.

    Returns
    -------
    n_epochs, native_size, upsampling_factor
    """
    images_dir = Path(images_dir)

    raw = _discover_epoch_files(images_dir, "image_{k}.fits")
    pre = _discover_epoch_files(images_dir, "image_{k}_preprocessed.fits")
    res = _discover_epoch_files(images_dir, "image_{k}_resampled.fits")

    indices = sorted(set(raw) | set(pre) | set(res))
    if not indices:
        raise FileNotFoundError(
            f"No image_k.fits, image_k_preprocessed.fits, or "
            f"image_k_resampled.fits files were found in {images_dir}."
        )

    expected = list(range(indices[-1] + 1))
    if indices != expected:
        raise ValueError(
            f"Image epoch indices are not contiguous from 0: found {indices}."
        )

    factor = _resolve_preprocessing_factor(images_dir, root)

    created_pre = 0
    reused_pre = 0
    created_res = 0
    reused_res = 0

    for k in tqdm(expected, desc="[images] prepare epochs", leave=True):
        raw_path = images_dir / f"image_{k}.fits"
        pre_path = images_dir / f"image_{k}_preprocessed.fits"
        res_path = images_dir / f"image_{k}_resampled.fits"

        if pre_path.is_file():
            reused_pre += 1
        else:
            if not raw_path.is_file():
                raise FileNotFoundError(
                    f"Cannot create {pre_path.name}: raw file is missing: {raw_path}"
                )
            _preprocess_native_image(raw_path, pre_path)
            created_pre += 1

        if res_path.is_file():
            reused_res += 1
        else:
            _build_resampled_image(
                pre_path,
                res_path,
                aperture_radius=float(fwhm),
                upsampling_factor=factor,
            )
            created_res += 1

    native_paths = _require_contiguous_epochs(
        _discover_epoch_files(images_dir, "image_{k}_preprocessed.fits"),
        "preprocessed image",
    )
    resampled_paths = _require_contiguous_epochs(
        _discover_epoch_files(images_dir, "image_{k}_resampled.fits"),
        "resampled image",
    )

    if len(native_paths) != len(resampled_paths):
        raise ValueError("Preprocessed and resampled epoch counts differ.")

    native_size = _square_fits_shape(native_paths[0])
    factor = _infer_convolution_factor(
        images_dir,
        resampled_paths[0],
        native_size,
    )

    if verbose:
        print()
        print("-" * 110)
        print("CLASSICAL IMAGE PRODUCTS")
        print("-" * 110)
        print()
        print(f"  images directory       : {images_dir}")
        print(f"  epochs                 : {len(native_paths)}")
        print(f"  native image size      : {native_size} x {native_size}")
        print(f"  upsampling factor      : {factor}")
        print()
        print(
            f"  preprocessed FITS      : reused={reused_pre}, created={created_pre}"
        )
        print(
            f"  resampled FITS         : reused={reused_res}, created={created_res}"
        )
        print()
        print(
            "  image_k_preprocessed.fits is the cleaned native square image "
            "used by aperture photometry and local reference measurements."
        )
        print(
            "  image_k_resampled.fits is the flux-conserving upsampled image "
            "convolved with the circular photometric kernel used by convolve."
        )
        print()

    return len(native_paths), native_size, factor


def _find_scalar_time_in_json_object(obj: Any) -> Optional[tuple[float, str]]:
    """Find one scalar time value in a JSON object."""
    direct_year_keys = {
        "time_years", "t_years", "epoch_years", "observation_time_years",
        "epoch_time_years", "time", "t",
    }
    mjd_keys = {"mjd", "mjd_obs", "mjd-obs"}
    jd_keys = {"jd", "jd_obs", "jd-obs"}

    if isinstance(obj, dict):
        normalized = {str(k).strip().lower(): v for k, v in obj.items()}

        for key in direct_year_keys:
            if key in normalized:
                try:
                    value = float(normalized[key])
                    if np.isfinite(value):
                        return value, "years"
                except Exception:
                    pass

        for key in mjd_keys:
            if key in normalized:
                try:
                    value = float(normalized[key])
                    if np.isfinite(value):
                        return value, "mjd"
                except Exception:
                    pass

        for key in jd_keys:
            if key in normalized:
                try:
                    value = float(normalized[key])
                    if np.isfinite(value):
                        return value, "jd"
                except Exception:
                    pass

        for value in obj.values():
            found = _find_scalar_time_in_json_object(value)
            if found is not None:
                return found

    elif isinstance(obj, list):
        for value in obj:
            found = _find_scalar_time_in_json_object(value)
            if found is not None:
                return found

    return None


def _find_epoch_root(base_dir: Path, n_epochs: int) -> Optional[Path]:
    """Find the nearest ancestor containing epoch_0 ... epoch_(K-1)."""
    for directory in _candidate_parent_dirs(base_dir, max_up=5):
        if all((directory / f"epoch_{k}").is_dir() for k in range(n_epochs)):
            return directory
    return None


def _epoch_time_from_exposure_table(path: Path) -> Optional[tuple[float, str]]:
    """
    Return one representative timestamp from an epoch exposure table.

    The table may contain one row per DIT. The median timestamp of the epoch is
    used as the epoch time.
    """
    path = Path(path)
    if not path.is_file():
        return None

    numeric_candidates = (
        ("TIME_YEARS", "years"),
        ("TIME_YEAR", "years"),
        ("T_YEARS", "years"),
        ("T_YEAR", "years"),
        ("EPOCH_YEARS", "years"),
        ("MJD", "mjd"),
        ("MJD_OBS", "mjd"),
        ("MJD-OBS", "mjd"),
        ("JD", "jd"),
        ("JD_OBS", "jd"),
        ("JD-OBS", "jd"),
        ("BJD", "jd"),
        ("TIME", "auto"),
        ("TIMES", "auto"),
        ("T", "auto"),
        ("TIMESTAMP", "auto"),
        ("TMID", "auto"),
    )

    try:
        with fits.open(path, memmap=True) as hdul:
            for hdu in hdul:
                data = hdu.data
                names = list(getattr(getattr(data, "dtype", None), "names", None) or [])
                if not names:
                    continue

                upper = {name.upper(): name for name in names}

                for key, kind in numeric_candidates:
                    if key not in upper:
                        continue

                    try:
                        arr = np.asarray(data[upper[key]], dtype=float).ravel()
                    except Exception:
                        continue

                    arr = arr[np.isfinite(arr)]
                    if not arr.size:
                        continue

                    value = float(np.median(arr))

                    if kind == "auto":
                        if value > 2.0e6:
                            kind_use = "jd"
                        elif value > 3.0e4:
                            kind_use = "mjd"
                        else:
                            kind_use = "years"
                    else:
                        kind_use = kind

                    return value, kind_use

                for key in ("DATE_OBS", "DATE-OBS", "DATE"):
                    if key not in upper:
                        continue
                    try:
                        values = np.asarray(data[upper[key]]).astype(str).ravel()
                        mjd = np.asarray([Time(v).mjd for v in values], dtype=float)
                        mjd = mjd[np.isfinite(mjd)]
                        if mjd.size:
                            return float(np.median(mjd)), "mjd"
                    except Exception:
                        continue

    except Exception:
        return None

    return None


def _times_from_epoch_directories(
    base_dir: Path,
    n_epochs: int,
) -> Optional[np.ndarray]:
    """
    Read one timestamp per epoch from epoch_k metadata.

    This supports datasets where the top-level metadata contains exposure
    details rather than a ready-made K-element epoch vector.
    """
    epoch_root = _find_epoch_root(base_dir, n_epochs)
    if epoch_root is None:
        return None

    values = []
    kinds = []

    for k in range(n_epochs):
        epoch_dir = epoch_root / f"epoch_{k}"
        found = None

        json_path = epoch_dir / "observation_parameters.json"
        if json_path.is_file():
            try:
                with json_path.open("r", encoding="utf-8") as handle:
                    found = _find_scalar_time_in_json_object(json.load(handle))
            except Exception:
                found = None

        if found is None:
            found = _epoch_time_from_exposure_table(
                epoch_dir / "exposures_parameters.fits"
            )

        if found is None:
            return None

        values.append(float(found[0]))
        kinds.append(str(found[1]))

    if len(set(kinds)) != 1:
        return None

    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        return None

    kind = kinds[0]
    if kind in {"mjd", "jd"}:
        return (arr - arr[0]) / 365.25

    return arr


def _prepare_runtime_config_from_files(
    yaml_path: str,
    root: Mapping[str, Any],
    *,
    verbose: bool = True,
) -> dict:
    """
    Discover dataset facts and prepare the runtime configuration.

    For PACO, epoch count and native size come from alpha_hat/var_alpha maps.
    For convolve/aperture, missing image_k_preprocessed.fits and
    image_k_resampled.fits are created automatically before data loading.

    Observation times are inferred from dataset metadata whenever possible.
    """
    base_dir = Path(yaml_path).expanduser().resolve().parent
    prepared = dict(root)

    prepared["work_dir"] = str(base_dir)
    prepared.setdefault("values_dir", "values")

    instruments_cfg = prepared.get("instruments", None)
    if not isinstance(instruments_cfg, (list, tuple)) or len(instruments_cfg) == 0:
        raise ValueError("`instruments` must contain at least one instrument block.")

    out_instruments = []

    if verbose:
        _print_section("DATASET DISCOVERY")
        print(f"  YAML directory       : {base_dir}")
        print()

    for index, original in enumerate(instruments_cfg):
        if not isinstance(original, dict):
            raise ValueError("Each instrument entry must be a dictionary.")

        inst = dict(original)
        method = str(inst.get("method", "convolve") or "convolve").strip().lower()
        if method not in {"convolve", "aperture", "paco"}:
            raise ValueError(
                "instrument.method must be 'convolve', 'aperture', or 'paco'."
            )

        if method == "paco":
            maps_dir = _discover_paco_maps_dir(
                base_dir,
                inst.get("paco_maps_dir"),
            )

            alpha_pattern = str(
                inst.get(
                    "paco_alpha_pattern",
                    "alpha_hat_epoch_{k}_north.fits",
                )
            )
            var_pattern = str(
                inst.get(
                    "paco_var_alpha_pattern",
                    "var_alpha_epoch_{k}_north.fits",
                )
            )

            alpha_paths = _require_contiguous_epochs(
                _discover_epoch_files(maps_dir, alpha_pattern),
                "PACO alpha_hat",
            )
            var_paths = _require_contiguous_epochs(
                _discover_epoch_files(maps_dir, var_pattern),
                "PACO var_alpha",
            )

            if len(alpha_paths) != len(var_paths):
                raise ValueError(
                    "PACO alpha_hat and var_alpha epoch counts differ: "
                    f"{len(alpha_paths)} vs {len(var_paths)}."
                )

            n_epochs = len(alpha_paths)
            size = _square_fits_shape(alpha_paths[0])

            for path in alpha_paths[1:] + var_paths:
                if _square_fits_shape(path) != size:
                    raise ValueError(f"PACO native map shape mismatch: {path}")

            times, time_source = _infer_times_years(
                base_dir=base_dir,
                epoch_paths=alpha_paths,
                n_epochs=n_epochs,
                inst_cfg=inst,
            )

            inst["paco_maps_dir"] = str(maps_dir)
            inst.setdefault("paco_alpha_pattern", alpha_pattern)
            inst.setdefault("paco_var_alpha_pattern", var_pattern)
            inst["upsampling_factor"] = 1

            try:
                inst["images_dir"] = str(
                    _discover_classical_images_dir(
                        base_dir,
                        inst.get("images_dir"),
                    )
                )
            except Exception:
                inst["images_dir"] = str((base_dir / "images").resolve())

        else:
            images_dir = _discover_classical_images_dir(
                base_dir,
                inst.get("images_dir"),
            )

            fwhm = inst.get("fwhm", None)
            if fwhm is None:
                probe = images_dir / "image_0_resampled.fits"
                radius = (
                    fits.getheader(probe).get("RADIUS", None)
                    if probe.is_file()
                    else None
                )
                if radius is None:
                    raise ValueError(
                        "convolve/aperture requires instruments[].fwhm when "
                        "no resampled FITS product is available to provide RADIUS."
                    )
                fwhm = float(radius)
                inst["fwhm"] = fwhm

            n_epochs, size, prepared_factor = _ensure_classical_image_products(
                images_dir,
                fwhm=float(fwhm),
                root=prepared,
                verbose=verbose,
            )

            native_paths = _require_contiguous_epochs(
                _discover_epoch_files(
                    images_dir,
                    "image_{k}_preprocessed.fits",
                ),
                "preprocessed image",
            )

            resampled_paths = _require_contiguous_epochs(
                _discover_epoch_files(
                    images_dir,
                    "image_{k}_resampled.fits",
                ),
                "resampled image",
            )

            if len(native_paths) != n_epochs or len(resampled_paths) != n_epochs:
                raise ValueError(
                    "Prepared image products do not contain the expected epochs."
                )

            times, time_source = _infer_times_years(
                base_dir=base_dir,
                epoch_paths=native_paths,
                n_epochs=n_epochs,
                inst_cfg=inst,
            )

            inst["images_dir"] = str(images_dir)
            inst["upsampling_factor"] = (
                int(prepared_factor)
                if method == "convolve"
                else 1
            )

            profile_dir = _resolve_path_from_yaml(
                base_dir,
                inst.get("profile_dir", None),
                "profiles",
            )
            profile_dir.mkdir(parents=True, exist_ok=True)
            inst["profile_dir"] = str(profile_dir)

        # Runtime metadata consumed internally by Params.
        inst["p"] = int(n_epochs)
        inst["p_prev"] = 0
        inst["total_time"] = 0
        inst["time"] = "+".join(
            f"{float(t):.15g}"
            for t in np.asarray(times, dtype=float)
        )
        inst["n"] = int(size)
        inst.setdefault("name", f"instrument_{index + 1}")
        inst.pop("times_years", None)

        if "resol" not in inst:
            raise ValueError(
                f"Instrument {index + 1}: `resol` [mas/pixel] is required."
            )

        out_instruments.append(inst)

        if verbose:
            print(f"  Instrument {index + 1}")
            print("  " + "-" * 28)
            print(f"  method               : {method}")

            if method == "paco":
                print(f"  PACO maps            : {inst['paco_maps_dir']}")
            else:
                print(f"  reduced images       : {inst['images_dir']}")

            print(f"  epochs               : {n_epochs}")
            print(f"  native image size    : {size} x {size} px")

            if method == "convolve":
                print(
                    f"  upsampling factor    : "
                    f"{inst['upsampling_factor']}"
                )

            print(
                f"  observation times    : "
                f"{np.asarray(times, dtype=float)}"
            )
            print(f"  time source          : {time_source}")
            print()

    prepared["instruments"] = out_instruments
    return prepared


def _resolve_background_noise(root: dict) -> dict:
    """
    Parse the local-aperture-ring background/noise configuration.

    For convolve and aperture photometry, background and noise are always
    estimated locally from reference apertures placed at similar stellar
    separation.

    The source aperture and nearby reference apertures are excluded from the
    reference set. The background is the median of the surviving measurements.
    The noise is either a robust MAD-based scatter or a standard deviation, with
    an optional Student small-sample correction.
    """
    cfg = root.get("background_noise", {}) or {}

    sigma_statistic = str(
        cfg.get("sigma_statistic", "mad") or "mad"
    ).strip().lower()
    if sigma_statistic not in ("mad", "std"):
        raise ValueError(
            "background_noise.sigma_statistic must be 'mad' or 'std'."
        )

    small_sample_correction = str(
        cfg.get("small_sample_correction", "student") or "student"
    ).strip().lower()
    if small_sample_correction not in ("student", "none"):
        raise ValueError(
            "background_noise.small_sample_correction must be 'student' or 'none'."
        )

    student_small_n_threshold = cfg.get(
        "student_small_n_threshold",
        None,
    )
    if student_small_n_threshold is not None:
        student_small_n_threshold = int(student_small_n_threshold)
        if student_small_n_threshold < 1:
            raise ValueError(
                "background_noise.student_small_n_threshold must be null or >= 1."
            )

    student_quantile = float(
        cfg.get(
            "student_quantile",
            0.8413447460685429,
        )
    )
    if not (0.5 < student_quantile < 1.0):
        raise ValueError(
            "background_noise.student_quantile must lie strictly between 0.5 and 1.0."
        )

    # Pixels without data.  Reduction pipelines mark them with NaN (turned
    # into 0 by the preprocessing) or directly with an exact 0.  "zero" treats
    # exact-zero / non-finite native pixels as missing: a measurement whose
    # photometric footprint touches them is neither a reference sample nor a
    # valid science sample.  "none" restores the pre-2026-11 behaviour.
    missing_data = str(cfg.get("missing_data", "zero") or "zero").strip().lower()
    if missing_data not in ("zero", "none"):
        raise ValueError("background_noise.missing_data must be 'zero' or 'none'.")

    out = dict(
        mode="local_aperture_ring",
        missing_data=missing_data,
        aperture_spacing_fwhm=float(
            cfg.get("aperture_spacing_fwhm", 2.0)
        ),
        exclusion_radius_fwhm=float(
            cfg.get("exclusion_radius_fwhm", 4.0)
        ),
        radial_samples=max(
            1,
            int(cfg.get("radial_samples", 3)),
        ),
        radial_step_fwhm=float(
            cfg.get("radial_step_fwhm", 2.0)
        ),
        min_reference_apertures=max(
            1,
            int(cfg.get("min_reference_apertures", 1)),
        ),
        sigma_statistic=sigma_statistic,
        small_sample_correction=small_sample_correction,
        student_small_n_threshold=student_small_n_threshold,
        student_quantile=student_quantile,
        local_map_mode=str(
            cfg.get("local_map_mode", "precompute_cache")
            or "precompute_cache"
        ).strip().lower(),
        precompute_maps=bool(
            cfg.get("precompute_maps", True)
        ),
        overwrite_cached_maps=bool(
            cfg.get("overwrite_cached_maps", False)
        ),
        cache_compression=bool(
            cfg.get("cache_compression", True)
        ),
        cache_dtype=str(
            cfg.get("cache_dtype", "float32")
            or "float32"
        ).strip().lower(),
    )

    print()
    print("-" * 110)
    print("LOCAL APERTURE-RING BACKGROUND / NOISE")
    print("-" * 110)
    print()
    print(f"  sigma statistic             : {sigma_statistic}")
    print(f"  aperture spacing            : {out['aperture_spacing_fwhm']:.3g} FWHM")
    print(f"  source exclusion            : {out['exclusion_radius_fwhm']:.3g} FWHM")
    print(f"  sampled rings               : {out['radial_samples']}")
    print(f"  ring spacing                : {out['radial_step_fwhm']:.3g} FWHM")
    print(f"  minimum reference apertures : {out['min_reference_apertures']}")
    print(f"  missing data (no-data px)   : {missing_data}"
          + ("  (exact 0 / NaN native pixels excluded)" if missing_data == "zero" else ""))
    print(f"  small-sample correction     : {small_sample_correction}")
    if small_sample_correction == "student":
        threshold_text = (
            "always"
            if student_small_n_threshold is None
            else str(student_small_n_threshold)
        )
        print(f"  Student correction threshold: {threshold_text}")
        print(f"  Student quantile            : {student_quantile:.12f}")
    print()

    return out


def _load_native_images(params: Params, suffix: str = "_preprocessed") -> np.ndarray:
    """
    Load native (non-upsampled) FITS images from the images_dir.

    File naming convention:
        images_dir / image_{k}{suffix}.fits   for each discovered epoch

    Parameters
    ──────────
    params : Params
        Project helper with I/O paths.
    suffix : str
        Filename suffix before ".fits".  Default is "_preprocessed".

    Returns
    ───────
    (K, size, size) float32 array.
    """
    images_dir = params.get_path("images_dir")
    nimg = int(params.p)
    imgs = []
    for k in range(nimg):
        fn = os.path.join(images_dir, f"image_{k}{suffix}.fits")
        im = fits.getdata(fn)
        imgs.append(im.astype("float32", copy=False))
    return np.asarray(imgs)


def _load_convolved_images(params: Params, suffix: str = "_resampled") -> np.ndarray:
    """
    Load the preprocessed convolved / upsampled FITS images from the images_dir.

    File naming convention:
        images_dir / image_{k}{suffix}.fits   for each discovered epoch

    These images are the science images used by the "convolve" photometry
    backend and by local aperture-ring reference measurements.
    """
    images_dir = params.get_path("images_dir")
    nimg = int(params.p)
    imgs = []
    for k in range(nimg):
        fn = os.path.join(images_dir, f"image_{k}{suffix}.fits")
        im = fits.getdata(fn)
        imgs.append(im.astype("float32", copy=False))
    return np.asarray(imgs)


def _make_placeholder_profiles(n_epochs: int, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create placeholder x/background/noise arrays.

    They keep the Instrument container shape uniform for PACO and classical
    backends. Classical background and noise are always measured with the local
    aperture-ring estimator.
    """
    nr = max(2, int(size))
    xgrid = np.arange(nr, dtype=float)
    bkg = np.zeros((int(n_epochs), nr), dtype=np.float32)
    noise = np.ones((int(n_epochs), nr), dtype=np.float32)
    return xgrid, bkg, noise


def _normalize_paco_interpolator(name: str) -> str:
    """Normalize and validate the PACO interpolation selector."""
    value = str(name or "none").strip().lower().replace("-", "_")
    aliases = {
        "off": "none",
        "native": "none",
        "no": "none",
        "false": "none",
        "linear": "bilinear",
        "cubic": "cubic_bspline",
        "bspline": "cubic_bspline",
        "catmullrom": "catmull_rom",
        "mitchell": "mitchell_netravali",
    }
    value = aliases.get(value, value)
    allowed = {
        "none", "nearest", "bilinear", "cubic_bspline",
        "lanczos4", "mitchell_netravali", "catmull_rom", "bccs",
    }
    if value not in allowed:
        raise ValueError(
            "paco_interpolator must be one of: " + ", ".join(sorted(allowed))
        )
    return value


def _normalize_radial_noise_recalibration_mode(name: str) -> str:
    """Normalize the radial-noise recalibration profile selector."""
    value = str(name or "none").strip().lower().replace("-", "_")
    aliases = {
        "off": "none", "no": "none", "false": "none",
        "raw": "empirical_radial", "g": "empirical_radial",
        "monotonic": "monotonic_inward", "monotone": "monotonic_inward",
        "extrapolation": "extrapolated_inward", "extrapolated": "extrapolated_inward",
    }
    value = aliases.get(value, value)
    allowed = {"none", "empirical_radial", "monotonic_inward", "extrapolated_inward"}
    if value not in allowed:
        raise ValueError(
            "radial_noise_recalibration.mode must be one of: "
            + ", ".join(sorted(allowed))
        )
    return value


def _resolve_radial_noise_recalibration(
    root: Mapping[str, Any],
    inst_cfg: Optional[Mapping[str, Any]] = None,
) -> dict:
    """
    Resolve global + per-instrument radial-noise recalibration settings.

    The global `radial_noise_recalibration` block gives the defaults shared by
    every instrument; any key may be overridden in
    `instruments[].radial_noise_recalibration`.

    `max_distance_px` is expressed in native pixels of the instrument.  When it
    is absent or set to "auto", it is computed for each instrument as
    `r_mask_ext - r_mask` (the full working annulus of that instrument).  The
    chosen value and its origin are returned in `max_distance_px` and
    `max_distance_origin`.
    """
    global_cfg = dict((root.get("radial_noise_recalibration", {}) or {}))
    inst_cfg = inst_cfg or {}
    local_cfg = dict((inst_cfg.get("radial_noise_recalibration", {}) or {}))
    cfg = {**global_cfg, **local_cfg}

    raw_dmax = cfg.get("max_distance_px", "auto")
    if raw_dmax is None or (isinstance(raw_dmax, str) and raw_dmax.strip().lower() == "auto"):
        r_in = inst_cfg.get("r_mask", root.get("r_mask", None))
        r_out = inst_cfg.get("r_mask_ext", root.get("r_mask_ext", None))
        size = inst_cfg.get("n", root.get("n", None))
        r_in = 0.0 if r_in is None else float(r_in)
        if r_out is None:
            if size is None:
                raise ValueError(
                    "radial_noise_recalibration.max_distance_px='auto' needs "
                    "r_mask_ext (or the image size) for each instrument."
                )
            r_out = (float(size) - 1.0) / 2.0
            origin = "computed: (n-1)/2 - r_mask (r_mask_ext undefined)"
        else:
            r_out = float(r_out)
            origin = "computed: r_mask_ext - r_mask"
        max_distance_px = r_out - r_in
        if not np.isfinite(max_distance_px) or max_distance_px <= 0.0:
            raise ValueError(
                f"Instrument {inst_cfg.get('name', '?')!r}: automatic "
                f"max_distance_px = {r_out:g} - {r_in:g} must be > 0."
            )
    else:
        max_distance_px = float(raw_dmax)
        if "max_distance_px" in local_cfg and local_cfg.get("max_distance_px") is not None:
            origin = "fixed: instrument section"
        else:
            origin = "fixed: global section (same px value for every instrument)"

    mode = _normalize_radial_noise_recalibration_mode(cfg.get("mode", "none"))
    return dict(
        mode=mode,
        bin_width_px=float(cfg.get("bin_width_px", 2.0)),
        max_distance_px=float(max_distance_px),
        max_distance_origin=origin,
        min_samples_per_bin=max(5, int(cfg.get("min_samples_per_bin", 30))),
        smoothing_bins=max(1, int(cfg.get("smoothing_bins", 3))),
        min_factor=float(cfg.get("min_factor", 1.0)),
        monotonic_extent_px=float(cfg.get("monotonic_extent_px", 4.0)),
        extrapolation_fit_min_px=float(cfg.get("extrapolation_fit_min_px", 1.0)),
        extrapolation_fit_max_px=float(cfg.get("extrapolation_fit_max_px", 5.0)),
        cache=bool(cfg.get("cache", True)),
        overwrite_cache=bool(cfg.get("overwrite_cache", False)),
    )


def _radial_recalibration_cache_token(cfg: Mapping[str, Any]) -> str:
    """Stable filesystem token for recalibration products.

    Every setting that changes g(d) is encoded, so that a cache produced with
    other settings is never reused.  The mode and the photometry method are
    encoded as sub-directories by the callers.
    """
    def num(key: str, default: float) -> str:
        return f"{float(cfg.get(key, default)):g}".replace(".", "p").replace("-", "m")

    parts = [
        f"bin{num('bin_width_px', 2.0)}px",
        f"max{num('max_distance_px', 30.0)}px",
        f"minsamp{int(cfg.get('min_samples_per_bin', 30))}",
        f"smooth{int(cfg.get('smoothing_bins', 3))}",
        f"gmin{num('min_factor', 1.0)}",
    ]
    mode = _normalize_radial_noise_recalibration_mode(cfg.get("mode", "none"))
    if mode in {"monotonic_inward", "extrapolated_inward"}:
        parts.append(f"mono{num('monotonic_extent_px', 4.0)}px")
    if mode == "extrapolated_inward":
        parts.append(
            f"fit{num('extrapolation_fit_min_px', 1.0)}"
            f"-{num('extrapolation_fit_max_px', 5.0)}px"
        )
    return "_".join(parts)


def _instrument_cache_label(instrument_name: Optional[str]) -> Optional[str]:
    """Filesystem-safe instrument label used for per-instrument cache folders."""
    if instrument_name is None:
        return None
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(instrument_name)).strip("_.")
    return label or "instrument"


def _instrument_profile_dir(profile_dir: str, instrument_name: Optional[str]) -> str:
    """Per-instrument noise-product folder: profile_dir/<instrument>/."""
    label = _instrument_cache_label(instrument_name)
    return str(profile_dir) if label is None else os.path.join(str(profile_dir), label)


def _radial_recalibration_cache_dir(
    profile_dir: str,
    instrument_name: Optional[str],
    cfg: Mapping[str, Any],
    photometry_method: str,
) -> str:
    """Folder of noise_recalibrated_epoch_k.npy and g_profile.npz."""
    mode = _normalize_radial_noise_recalibration_mode(cfg.get("mode", "none"))
    return os.path.join(
        _instrument_profile_dir(profile_dir, instrument_name),
        "radial_noise_recalibration",
        _radial_recalibration_cache_token(cfg),
        mode,
        str(photometry_method).lower(),
    )


def _robust_unit_scatter(values: np.ndarray) -> float:
    """MAD-based robust Gaussian-equivalent scatter used to estimate g."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 5:
        return np.nan
    med = float(np.median(values))
    return float(_RADIAL_RECAL_MAD_TO_SIGMA * np.median(np.abs(values - med)))


def _radial_distance_from_inner_edge(
    shape: tuple[int, int],
    valid: np.ndarray,
    *,
    center_xy: tuple[float, float],
    inner_radius: Optional[float],
) -> tuple[np.ndarray, float]:
    """
    Distance [native px] from the effective finite-map inner edge.

    For PACO, if `inner_radius` is None, the inner edge is inferred from the
    smallest radius containing valid native map pixels.  For classical
    convolve/aperture products the instrument IWA is used when supplied.
    """
    ny, nx = shape
    yy, xx = np.mgrid[:ny, :nx]
    cx, cy = map(float, center_xy)
    radius = np.hypot(xx - cx, yy - cy)
    valid = np.asarray(valid, dtype=bool)
    if inner_radius is None:
        rvals = radius[valid]
        if rvals.size == 0:
            inner = 0.0
        else:
            inner = float(np.nanmin(rvals))
    else:
        inner = float(inner_radius)
    return radius - inner, inner


def _estimate_radial_noise_profile(
    z_maps: np.ndarray,
    valid_maps: np.ndarray,
    *,
    cfg: Mapping[str, Any],
    center_xy: tuple[float, float],
    inner_radius: Optional[float],
) -> dict:
    """
    Estimate one orbit-local g(d) profile from all epochs.

    Distances are measured from EACH epoch's own finite inner edge when
    ``inner_radius`` is None (PACO).  For classical reductions, passing the
    instrument IWA uses the same reference radius at every epoch.

    The robust scatter is first estimated independently per epoch/radial bin,
    then the median across epochs defines the shared g(d) profile.
    """
    z_maps = np.asarray(z_maps, dtype=float)
    valid_maps = np.asarray(valid_maps, dtype=bool)
    if z_maps.ndim != 3 or valid_maps.shape != z_maps.shape:
        raise ValueError("z_maps and valid_maps must both have shape (epoch, y, x).")

    bw = float(cfg["bin_width_px"])
    dmax = float(cfg["max_distance_px"])
    if not np.isfinite(bw) or bw <= 0 or not np.isfinite(dmax) or dmax <= 0:
        raise ValueError("radial_noise_recalibration bin_width_px/max_distance_px must be > 0.")

    distance_maps = []
    inner_radii = []
    for k in range(z_maps.shape[0]):
        dist_k, inner_k = _radial_distance_from_inner_edge(
            z_maps.shape[1:],
            valid_maps[k],
            center_xy=center_xy,
            inner_radius=inner_radius,
        )
        distance_maps.append(dist_k)
        inner_radii.append(inner_k)
    distance_maps = np.asarray(distance_maps, dtype=float)
    inner_radii = np.asarray(inner_radii, dtype=float)

    edges = np.arange(0.0, dmax + bw + 1e-12, bw)
    centers = 0.5 * (edges[:-1] + edges[1:])
    per_epoch = np.full((z_maps.shape[0], centers.size), np.nan, dtype=float)

    for k in range(z_maps.shape[0]):
        distance = distance_maps[k]
        for b in range(centers.size):
            sel = (
                valid_maps[k]
                & (distance >= edges[b])
                & (distance < edges[b + 1])
                & np.isfinite(z_maps[k])
            )
            if int(np.count_nonzero(sel)) >= int(cfg["min_samples_per_bin"]):
                per_epoch[k, b] = _robust_unit_scatter(z_maps[k][sel])

    with np.errstate(all="ignore"):
        g_raw = np.nanmedian(per_epoch, axis=0)

    smooth_n = int(cfg["smoothing_bins"])
    g_smooth = g_raw.copy()
    if smooth_n > 1:
        half = smooth_n // 2
        for b in range(centers.size):
            lo = max(0, b - half)
            hi = min(centers.size, b + half + 1)
            finite = g_raw[lo:hi][np.isfinite(g_raw[lo:hi])]
            if finite.size:
                g_smooth[b] = float(np.median(finite))

    min_factor = float(cfg["min_factor"])
    empirical = np.where(
        np.isfinite(g_smooth),
        np.maximum(min_factor, g_smooth),
        min_factor,
    )
    mode = _normalize_radial_noise_recalibration_mode(cfg["mode"])
    corrected = empirical.copy()

    if mode in {"monotonic_inward", "extrapolated_inward"}:
        idx = np.flatnonzero(centers <= float(cfg["monotonic_extent_px"]))
        running = min_factor
        for b in idx[::-1]:
            running = max(running, float(corrected[b]))
            corrected[b] = running

    if mode == "extrapolated_inward":
        fit = (
            (centers >= float(cfg["extrapolation_fit_min_px"]))
            & (centers <= float(cfg["extrapolation_fit_max_px"]))
            & np.isfinite(empirical)
        )
        if int(np.count_nonzero(fit)) >= 2:
            slope, intercept, _, _ = theilslopes(empirical[fit], centers[fit])
            inward = centers < float(cfg["extrapolation_fit_min_px"])
            prediction = intercept + slope * centers
            corrected[inward] = np.maximum(
                corrected[inward],
                np.maximum(min_factor, prediction[inward]),
            )

    if mode == "none":
        corrected[:] = 1.0

    return dict(
        distance_centers_px=centers,
        g_raw=g_raw,
        g_empirical=empirical,
        g=corrected,
        per_epoch=per_epoch,
        distance_maps_px=distance_maps,
        inner_radii_px=inner_radii,
        # Scalar median kept for compact metadata/backward-friendly prints.
        inner_radius_px=float(np.nanmedian(inner_radii)),
    )


def _g_map_from_profile(profile: Mapping[str, Any], *, cfg: Mapping[str, Any]) -> np.ndarray:
    """
    Interpolate the shared g(d) profile onto each epoch's native distance map.

    Return shape is (epoch, y, x).  This matters for PACO because the finite
    inner edge may differ slightly from one epoch to another.
    """
    distances = np.asarray(profile["distance_maps_px"], dtype=float)
    centers = np.asarray(profile["distance_centers_px"], dtype=float)
    g = np.asarray(profile["g"], dtype=float)
    out = np.ones_like(distances, dtype=float)

    for k in range(distances.shape[0]):
        distance = distances[k]
        zone = (
            np.isfinite(distance)
            & (distance >= 0.0)
            & (distance <= float(cfg["max_distance_px"]))
        )
        if centers.size and np.any(zone):
            out[k][zone] = np.interp(
                distance[zone], centers, g, left=g[0], right=g[-1]
            )

    return np.maximum(float(cfg["min_factor"]), out)


def _normalize_paco_variance_correction(name: str) -> str:
    return _normalize_radial_noise_recalibration_mode(name)


def _paco_variance_bin_token(bin_width_px: float) -> str:
    value = float(bin_width_px)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("radial_noise_recalibration.bin_width_px must be finite and > 0.")
    return f"{value:g}".replace(".", "p")


def _effective_paco_oversampling(interpolator: str, oversampling: int) -> int:
    """Return the map oversampling actually used by the PACO likelihood."""
    method = _normalize_paco_interpolator(interpolator)
    if method == "none":
        return 1
    factor = int(oversampling)
    if factor < 1:
        raise ValueError("paco_oversampling must be >= 1.")
    return factor


def _paco_map_side(size: int, interpolator: str, oversampling: int) -> int:
    """Spatial side length of a native or cached oversampled PACO map."""
    factor = _effective_paco_oversampling(interpolator, oversampling)
    return int(size) if factor == 1 else (int(size) - 1) * factor + 1


def _cubic_cardinal_kernel(x: np.ndarray, *, psi: float, chi: float) -> np.ndarray:
    """Compact-support cubic cardinal kernel parameterised by (psi, chi)."""
    u = np.abs(np.asarray(x, dtype=np.float64))
    out = np.zeros_like(u)
    m1 = u <= 1.0
    z = u[m1]
    out[m1] = (psi - 2.0 * chi + 2.0) * z**3 + (3.0 * chi - 3.0 - psi) * z**2 + 1.0
    m2 = (u > 1.0) & (u < 2.0)
    z = 2.0 - u[m2]
    out[m2] = (-psi - 2.0 * chi) * z**3 + (psi + 3.0 * chi) * z**2
    return out


def _lanczos4_kernel(x: np.ndarray) -> np.ndarray:
    """Four-neighbour Lanczos kernel with support |x| < 2 native pixels."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    m = np.abs(x) < 2.0
    out[m] = np.sinc(x[m]) * np.sinc(x[m] / 2.0)
    return out


def _interp_axis_four_support(coords: np.ndarray, n: int, kernel) -> tuple[np.ndarray, np.ndarray]:
    """Four source indices and normalized weights for a separable kernel."""
    coords = np.asarray(coords, dtype=np.float64)
    base = np.floor(coords).astype(np.int64)
    offsets = np.asarray([-1, 0, 1, 2], dtype=np.int64)
    raw_idx = base[:, None] + offsets[None, :]
    idx = np.clip(raw_idx, 0, int(n) - 1)
    weights = np.asarray(kernel(coords[:, None] - raw_idx), dtype=np.float64)
    sums = np.sum(weights, axis=1, keepdims=True)
    good = np.abs(sums[:, 0]) > 1.0e-12
    weights[good] /= sums[good]
    weights[~good] = 0.0
    return idx, weights


def _interpolate_four_support_map(
    image: np.ndarray, valid: np.ndarray, y_out: np.ndarray, x_out: np.ndarray, kernel
) -> np.ndarray:
    """Separable four-neighbour interpolation with conservative validity."""
    image = np.asarray(image, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    ny, nx = image.shape
    x_idx, x_w = _interp_axis_four_support(x_out, nx, kernel)
    safe = np.where(valid, image, 0.0)
    temp = np.einsum("yok,ok->yo", safe[:, x_idx], x_w, optimize=True)
    temp_valid = np.all(valid[:, x_idx], axis=2)
    y_idx, y_w = _interp_axis_four_support(y_out, ny, kernel)
    out = np.einsum("okx,ok->ox", temp[y_idx, :], y_w, optimize=True)
    out_valid = np.all(temp_valid[y_idx, :], axis=1)
    out[~out_valid] = np.nan
    return out


def _interpolate_paco_map(
    image: np.ndarray, native_valid: np.ndarray, *, interpolator: str, oversampling: int
) -> np.ndarray:
    """Interpolate one native PACO map onto a regular oversampled grid.

    Native pixel centres are at integer coordinates.  The oversampled grid is
    therefore x_os / oversampling = x_native, so native centres are preserved
    exactly at indices 0, oversampling, 2*oversampling, ... .
    """
    method = _normalize_paco_interpolator(interpolator)
    factor = _effective_paco_oversampling(method, oversampling)
    image = np.asarray(image, dtype=np.float64)
    native_valid = np.asarray(native_valid, dtype=bool)
    if method == "none":
        out = image.copy()
        out[~native_valid] = np.nan
        return out

    ny, nx = image.shape
    y_out = np.arange((ny - 1) * factor + 1, dtype=np.float64) / float(factor)
    x_out = np.arange((nx - 1) * factor + 1, dtype=np.float64) / float(factor)

    if method == "nearest":
        yi = np.clip(np.floor(y_out + 0.5).astype(np.int64), 0, ny - 1)
        xi = np.clip(np.floor(x_out + 0.5).astype(np.int64), 0, nx - 1)
        out = image[yi[:, None], xi[None, :]].astype(np.float64)
        out_valid = native_valid[yi[:, None], xi[None, :]]
        out[~out_valid] = np.nan
        return out

    if method == "bilinear":
        x0 = np.floor(x_out).astype(np.int64); y0 = np.floor(y_out).astype(np.int64)
        x1 = np.clip(x0 + 1, 0, nx - 1); y1 = np.clip(y0 + 1, 0, ny - 1)
        x0 = np.clip(x0, 0, nx - 1); y0 = np.clip(y0, 0, ny - 1)
        dx = x_out - x0; dy = y_out - y0
        f00 = image[y0[:, None], x0[None, :]]; f01 = image[y0[:, None], x1[None, :]]
        f10 = image[y1[:, None], x0[None, :]]; f11 = image[y1[:, None], x1[None, :]]
        out = ((1.0-dy)[:,None]*((1.0-dx)[None,:]*f00 + dx[None,:]*f01)
               + dy[:,None]*((1.0-dx)[None,:]*f10 + dx[None,:]*f11))
        out_valid = (native_valid[y0[:,None],x0[None,:]] & native_valid[y0[:,None],x1[None,:]] &
                     native_valid[y1[:,None],x0[None,:]] & native_valid[y1[:,None],x1[None,:]])
        out[~out_valid] = np.nan
        return out

    if method == "cubic_bspline":
        if not np.any(native_valid):
            return np.full(((ny - 1) * factor + 1, (nx - 1) * factor + 1), np.nan, dtype=np.float64)
        if np.all(native_valid):
            filled = image
        else:
            nearest_indices = distance_transform_edt(~native_valid, return_distances=False, return_indices=True)
            filled = image[tuple(nearest_indices)]
        yy, xx = np.meshgrid(y_out, x_out, indexing="ij")
        out = map_coordinates(filled, [yy, xx], order=3, mode="nearest", prefilter=True)
        # Conservative 4x4 native support validity, consistent with the other cubic kernels.
        def support_indices(coords, n):
            base = np.floor(coords).astype(np.int64)
            return np.clip(base[:,None] + np.asarray([-1,0,1,2])[None,:], 0, n-1)
        yi = support_indices(y_out, ny); xi = support_indices(x_out, nx)
        out_valid = np.ones(out.shape, dtype=bool)
        for jy in range(4):
            for jx in range(4):
                out_valid &= native_valid[yi[:,jy][:,None], xi[:,jx][None,:]]
        out[~out_valid] = np.nan
        return out

    if method == "lanczos4":
        kernel = _lanczos4_kernel
    elif method == "mitchell_netravali":
        kernel = lambda x: _cubic_cardinal_kernel(x, psi=-0.5, chi=1.0/18.0)
    elif method == "catmull_rom":
        kernel = lambda x: _cubic_cardinal_kernel(x, psi=-0.5, chi=0.0)
    else:  # bccs
        kernel = lambda x: _cubic_cardinal_kernel(x, psi=-0.600, chi=-0.004)
    return _interpolate_four_support_map(image, native_valid, y_out, x_out, kernel)


def _paco_cached_paths(paco_dir: str, alpha_name: str, var_name: str, method: str, factor: int) -> tuple[str, str, str]:
    """Return cache directory and filenames for interpolated PACO products."""
    cache_dir = os.path.join(paco_dir, f"interpolated_{method}_os{factor}")
    alpha_stem, alpha_ext = os.path.splitext(alpha_name)
    var_stem, var_ext = os.path.splitext(var_name)
    alpha_out = os.path.join(cache_dir, f"{alpha_stem}_interp-{method}_os{factor}{alpha_ext or '.fits'}")
    var_out = os.path.join(cache_dir, f"{var_stem}_interp-{method}_os{factor}{var_ext or '.fits'}")
    return cache_dir, alpha_out, var_out


def _write_interpolated_paco_fits(
    path: str, data: np.ndarray, source_header, *, method: str, factor: int, native_shape: tuple[int, int]
) -> None:
    """Write one cached interpolated PACO FITS map with explicit metadata."""
    header = source_header.copy()
    header["PACOINT"] = (str(method), "PACO interpolation method")
    header["PACOSAMP"] = (int(factor), "PACO spatial oversampling factor")
    header["NATNY"] = (int(native_shape[0]), "Native PACO map Ny")
    header["NATNX"] = (int(native_shape[1]), "Native PACO map Nx")
    header["PIXSTEP"] = (1.0 / float(factor), "Sampling step in native pixels")
    for key in ("CRPIX1", "CRPIX2"):
        if key in header:
            header[key] = (float(header[key]) - 1.0) * factor + 1.0
    for key in ("CDELT1", "CDELT2", "CD1_1", "CD1_2", "CD2_1", "CD2_2"):
        if key in header:
            header[key] = float(header[key]) / float(factor)
    fits.writeto(path, np.asarray(data, dtype=np.float32), header=header, overwrite=True)


def _load_paco_maps(
    params: Params,
    *,
    paco_maps_dir: str = "wpca_alpha_var",
    alpha_pattern: str = "alpha_hat_epoch_{k}_north.fits",
    var_alpha_pattern: str = "var_alpha_epoch_{k}_north.fits",
    interpolator: str = "none",
    oversampling: int = 1,
    radial_noise_recalibration: Optional[Mapping[str, Any]] = None,
    instrument_name: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load PACO maps and, when requested, compute radial noise recalibration here.

    The correction is no longer expected to have been generated by an external
    notebook.  Starting from the original native PACO products, this function
    estimates g(d) from z=alpha_hat/sqrt(var_alpha), applies

        var_recalibrated = g(d)^2 * var_alpha,

    caches the native result, and only THEN performs optional PACO map
    interpolation.  alpha_hat is never modified by the recalibration.

    Existing caches are reused unless `overwrite_cache: true`.
    """
    images_dir = os.path.abspath(params.get_path("images_dir"))
    orbit_dir = os.path.dirname(os.path.normpath(images_dir))
    paco_path = Path(str(paco_maps_dir)).expanduser()
    paco_dir = str(paco_path if paco_path.is_absolute() else (Path(params.work_dir) / paco_path).resolve())
    n_epochs = int(params.p)
    size = int(params.n)
    center = ((size - 1.0) / 2.0, (size - 1.0) / 2.0)

    method = _normalize_paco_interpolator(interpolator)
    factor = _effective_paco_oversampling(method, oversampling)
    expected_side = _paco_map_side(size, method, factor)

    if radial_noise_recalibration is None:
        recal_cfg = _resolve_radial_noise_recalibration({}, {"n": size})
    else:
        recal_cfg = dict(radial_noise_recalibration)
        recal_cfg["mode"] = _normalize_radial_noise_recalibration_mode(
            recal_cfg.get("mode", "none")
        )

    mode = recal_cfg["mode"]

    print()
    print("=" * 110)
    print(f"PACO MAPS — {os.path.basename(orbit_dir)}")
    print("=" * 110)
    print()
    print(f"  maps directory       : {paco_dir}")
    print(f"  epochs               : {n_epochs}")
    print(f"  interpolation        : {method}" + ("" if method == "none" else f" x{factor}"))
    print()
    print("  Radial noise recalibration")
    print("  --------------------------")
    print(f"  mode                 : {mode}")
    if mode == "none":
        print("  action               : disabled; original var_alpha is used")
    else:
        print("  definition           : sigma_cal(d) = g(d) * sigma(d)")
        print("  PACO variance        : var_cal(d) = g(d)^2 * var_alpha(d)")
        print(f"  radial bin           : {float(recal_cfg['bin_width_px']):g} native px")
        print(f"  calibrated edge zone : 0 .. {float(recal_cfg['max_distance_px']):g} native px")
        print(f"  minimum g            : {float(recal_cfg['min_factor']):g}")
        print("  order                : native PACO maps -> g(d) -> optional interpolation")
    print()

    alpha_native_stack = []
    var_native_stack = []
    alpha_paths = []
    var_paths = []

    for k in range(n_epochs):
        alpha_name = str(alpha_pattern).format(k=k)
        var_name = str(var_alpha_pattern).format(k=k)
        alpha_path = os.path.join(paco_dir, alpha_name)
        var_path = os.path.join(paco_dir, var_name)

        if not os.path.isfile(alpha_path):
            raise FileNotFoundError(f"Missing PACO alpha_hat map for epoch {k}: {alpha_path}")
        if not os.path.isfile(var_path):
            raise FileNotFoundError(f"Missing PACO var_alpha map for epoch {k}: {var_path}")

        alpha_native = np.asarray(fits.getdata(alpha_path), dtype=np.float32)
        var_native = np.asarray(fits.getdata(var_path), dtype=np.float32)

        if alpha_native.shape != (size, size) or var_native.shape != (size, size):
            raise ValueError(
                f"PACO native shape mismatch at epoch {k}: "
                f"alpha={alpha_native.shape}, var={var_native.shape}, expected {(size, size)}."
            )

        alpha_native_stack.append(alpha_native)
        var_native_stack.append(var_native)
        alpha_paths.append(alpha_path)
        var_paths.append(var_path)

    alpha_native_stack = np.asarray(alpha_native_stack, dtype=np.float32)
    var_native_stack = np.asarray(var_native_stack, dtype=np.float32)

    recal_root = None
    profile = None
    if mode != "none":
        recal_parts = [paco_dir, "radial_noise_recalibration"]
        label = _instrument_cache_label(instrument_name)
        if label is not None:
            recal_parts.append(label)
        recal_root = os.path.join(
            *recal_parts,
            _radial_recalibration_cache_token(recal_cfg),
            mode,
        )
        print(f"  instrument           : {instrument_name}")
        print(f"  max_distance_px      : {float(recal_cfg['max_distance_px']):g} "
              f"({recal_cfg.get('max_distance_origin', 'n/a')})")
        print(f"  cache directory      : {recal_root}")
        os.makedirs(recal_root, exist_ok=True)
        profile_path = os.path.join(recal_root, "g_profile.npz")
        corrected_paths = [
            os.path.join(recal_root, str(var_alpha_pattern).format(k=k))
            for k in range(n_epochs)
        ]

        cache_ready = (
            bool(recal_cfg.get("cache", True))
            and not bool(recal_cfg.get("overwrite_cache", False))
            and os.path.isfile(profile_path)
            and all(os.path.isfile(path) for path in corrected_paths)
        )

        if cache_ready:
            print("  cache                : loading existing recalibrated native variance")
            loaded = np.load(profile_path)
            profile = {
                "distance_centers_px": loaded["distance_centers_px"],
                "g_raw": loaded["g_raw"],
                "g_empirical": loaded["g_empirical"],
                "g": loaded["g"],
                "per_epoch": loaded["per_epoch"],
                "inner_radius_px": float(loaded["inner_radius_px"]),
            }
            var_cal = np.asarray(
                [fits.getdata(path) for path in corrected_paths],
                dtype=np.float32,
            )
            if var_cal.shape != var_native_stack.shape:
                console_status(
                    "warning",
                    f"PACO recalibration cache {recal_root} has shape "
                    f"{var_cal.shape}, expected {var_native_stack.shape}: recomputing.",
                    tone="yellow",
                )
                cache_ready = False
        if not cache_ready:
            print("  cache                : missing/overwrite requested -> estimating g(d)")
            raw_valid = (
                np.isfinite(alpha_native_stack)
                & np.isfinite(var_native_stack)
                & (var_native_stack > 0.0)
            )
            z = np.full_like(alpha_native_stack, np.nan, dtype=np.float64)
            z[raw_valid] = (
                alpha_native_stack[raw_valid]
                / np.sqrt(var_native_stack[raw_valid])
            )

            profile = _estimate_radial_noise_profile(
                z,
                raw_valid,
                cfg=recal_cfg,
                center_xy=center,
                inner_radius=None,       # infer the actual finite PACO inner edge
            )
            g_map = _g_map_from_profile(profile, cfg=recal_cfg)
            var_cal = np.asarray(
                var_native_stack * g_map ** 2,
                dtype=np.float32,
            )

            if bool(recal_cfg.get("cache", True)):
                np.savez(
                    profile_path,
                    distance_centers_px=np.asarray(profile["distance_centers_px"]),
                    g_raw=np.asarray(profile["g_raw"]),
                    g_empirical=np.asarray(profile["g_empirical"]),
                    g=np.asarray(profile["g"]),
                    per_epoch=np.asarray(profile["per_epoch"]),
                    inner_radius_px=np.asarray(profile["inner_radius_px"]),
                )
                for k, path in enumerate(corrected_paths):
                    hdr = fits.getheader(var_paths[k], ext=0)
                    hdr.add_history(
                        "KStacker radial noise recalibration: "
                        f"var_cal=g(d)^2*var; mode={mode}"
                    )
                    fits.writeto(path, var_cal[k], header=hdr, overwrite=True)

            finite_g = np.asarray(profile["g"])[np.isfinite(profile["g"])]
            if finite_g.size:
                print(
                    f"  fitted g range       : {float(np.min(finite_g)):.3f} .. "
                    f"{float(np.max(finite_g)):.3f}"
                )
                print(f"  inferred inner edge  : r={float(profile['inner_radius_px']):.3f} px")
        print()
    else:
        var_cal = var_native_stack

    alpha_maps = []
    var_maps = []

    for k in tqdm(range(n_epochs), desc="Loading PACO likelihood maps", unit="epoch"):
        alpha_native = alpha_native_stack[k]
        var_native = var_cal[k]
        native_valid = (
            np.isfinite(alpha_native)
            & np.isfinite(var_native)
            & (var_native > 0.0)
        )
        alpha_name = str(alpha_pattern).format(k=k)
        var_name = str(var_alpha_pattern).format(k=k)

        if method == "none":
            alpha = alpha_native
            var = var_native
            source_label = "native"
        else:
            alpha_cache_dir, alpha_cache, standard_var_cache = _paco_cached_paths(
                paco_dir, alpha_name, var_name, method, factor
            )
            os.makedirs(alpha_cache_dir, exist_ok=True)

            if mode == "none":
                var_cache = standard_var_cache
            else:
                recal_interp = os.path.join(
                    str(recal_root), f"interpolated_{method}_os{factor}"
                )
                os.makedirs(recal_interp, exist_ok=True)
                stem, ext = os.path.splitext(var_name)
                var_cache = os.path.join(
                    recal_interp, f"{stem}_interp-{method}_os{factor}{ext or '.fits'}"
                )

            if os.path.isfile(alpha_cache):
                alpha = np.asarray(fits.getdata(alpha_cache), dtype=np.float32)
            else:
                alpha = np.asarray(
                    _interpolate_paco_map(
                        alpha_native, native_valid,
                        interpolator=method, oversampling=factor,
                    ),
                    dtype=np.float32,
                )
                _write_interpolated_paco_fits(
                    alpha_cache, alpha, fits.getheader(alpha_paths[k], ext=0),
                    method=method, factor=factor, native_shape=alpha_native.shape,
                )

            if os.path.isfile(var_cache) and not bool(recal_cfg.get("overwrite_cache", False)):
                var = np.asarray(fits.getdata(var_cache), dtype=np.float32)
            else:
                var_interp = _interpolate_paco_map(
                    var_native, native_valid,
                    interpolator=method, oversampling=factor,
                )
                valid_interp = (
                    np.isfinite(alpha)
                    & np.isfinite(var_interp)
                    & (var_interp > 0.0)
                )
                var_interp[~valid_interp] = np.nan
                var = np.asarray(var_interp, dtype=np.float32)
                _write_interpolated_paco_fits(
                    var_cache, var, fits.getheader(var_paths[k], ext=0),
                    method=method, factor=factor, native_shape=var_native.shape,
                )
            source_label = "cache"

        if alpha.shape != (expected_side, expected_side) or var.shape != (expected_side, expected_side):
            raise ValueError(
                f"PACO {method} map shape mismatch at epoch {k}: "
                f"alpha={alpha.shape}, var={var.shape}; expected {(expected_side, expected_side)}."
            )

        valid = np.isfinite(alpha) & np.isfinite(var) & (var > 0.0)
        n_valid = int(np.count_nonzero(valid))
        print(
            f"  epoch {k}: {source_label:>6s} | shape={alpha.shape} | "
            f"valid={n_valid}/{valid.size} ({100.0*n_valid/max(1, valid.size):.2f}%)"
        )
        alpha_maps.append(alpha)
        var_maps.append(var)

    print()
    print("-" * 110)
    print("PACO maps ready")
    print("-" * 110)
    print()

    return (
        np.asarray(alpha_maps, dtype=np.float32),
        np.asarray(var_maps, dtype=np.float32),
    )


def _aperture_sum_native(
    image_native: np.ndarray, x: float, y: float, r_ap: float
) -> float:
    """
    Circular aperture photometry at position (x, y) in a single native image.

    Parameters
    ──────────
    image_native : (H, W) float array.
    x, y         : predicted planet position in native pixels (origin lower-left).
    r_ap         : aperture radius [native pixels].

    Returns
    ───────
    float: aperture sum (same units as image pixels).
    """
    ap   = CircularAperture((x, y), r=r_ap)
    phot = aperture_photometry(image_native, ap)
    return float(np.array(phot["aperture_sum"])[0])


def _wrap_pi(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap angles to the interval [-π, π)."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def _robust_sigma(samples: np.ndarray, statistic: str) -> float:
    """
    Convert a 1-D sample into a robust σ estimate.

    "mad":
        σ ≈ 1.4826 × MAD, robust against outliers and bright residual speckles.

    "std":
        Classical sample standard deviation (ddof=1), less robust but useful
        for debugging or cross-checks.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.size == 0:
        return np.nan

    if statistic == "std":
        if samples.size < 2:
            return np.nan
        return float(np.std(samples, ddof=1))

    med = float(np.median(samples))
    mad = float(np.median(np.abs(samples - med)))
    return 1.4826 * mad


def _student_corrected_local_sigma(
    raw_sigma: float,
    n_reference: int,
    cfg: Optional[Mapping[str, Any]],
) -> float:
    """
    Apply an optional Student low-statistics correction to the local noise.

    Philosophy
    ----------
    The local background level remains the robust median of the surviving
    reference apertures. What is corrected here is the uncertainty attached to
    the local background-subtracted measurement.

    Control logic
    -------------
    - If small_sample_correction = "none", return the raw sigma unchanged.
    - If small_sample_correction = "student" and student_small_n_threshold is
      null, apply the correction whenever the local estimate succeeds.
    - If small_sample_correction = "student" and the threshold is an integer T,
      apply the correction only when N_reference <= T.

    Formula
    -------
    sigma_corrected = sigma_raw * sqrt(1 + 1 / N_reference) * t_{q, N_reference-1}

    where q is configured by background_noise.student_quantile.

    This makes the uncertainty more conservative when only a limited number of
    independent reference apertures survive on the ring.
    """
    cfg = cfg or {}

    correction_mode = str(cfg.get("small_sample_correction", "student") or "student").strip().lower()
    if correction_mode in ("none", "off", "false", "no"):
        return float(raw_sigma)

    if correction_mode != "student":
        raise ValueError(
            "background_noise.small_sample_correction must be 'student' or 'none'."
        )

    raw_sigma = float(raw_sigma)
    if (not np.isfinite(raw_sigma)) or raw_sigma <= 0.0:
        return np.nan

    n_reference = int(n_reference)
    if n_reference < 2:
        return np.nan

    threshold = cfg.get("student_small_n_threshold", None)
    if threshold is not None and n_reference > int(threshold):
        return float(raw_sigma)

    dof = n_reference - 1
    gaussian_one_sigma_quantile = 0.8413447460685429
    student_quantile = float(cfg.get("student_quantile", gaussian_one_sigma_quantile))

    if not (0.5 < student_quantile < 1.0):
        raise ValueError(
            "background_noise.student_quantile must lie strictly between 0.5 and 1.0."
        )

    finite_sample_factor = np.sqrt(1.0 + 1.0 / float(n_reference))
    student_factor = float(scipy_student_t.ppf(student_quantile, dof))

    if (not np.isfinite(student_factor)) or student_factor <= 0.0:
        return np.nan

    return float(raw_sigma * finite_sample_factor * student_factor)


def _native_to_upsampled_index(position, factor):
    """Return the nearest subpixel-grid index for a native-pixel coordinate.

    Native pixel centres are numbered 0, 1, ..., N-1.  Block replication by
    an integer factor ``f`` puts the centre of native pixel ``i`` at
    ``(i + 0.5) * f - 0.5`` in the upsampled array.  Omitting the native
    half-pixel offset displaces every convolved measurement on the detector.

    This function is shared by science photometry, noise references, radial
    calibration and Gaussian-flux sufficient statistics.
    """
    return np.floor((np.asarray(position) + 0.5) * float(factor) - 0.5).astype(np.int64)


def _extract_convolved_scalar(
    image_up: np.ndarray,
    x_native: float,
    y_native: float,
    upsampling_factor: float,
) -> float | None:
    """
    Read the same scalar as the "convolve" backend at one test position.

    The signal model for the "convolve" backend uses the *upsampled convolved
    image* as the science image and reads the nearest pixel at the predicted
    location.  The local reference measurements must therefore probe the same
    image, otherwise the background/noise estimate would not live in the same
    units as the science signal.
    """
    x_up = int(_native_to_upsampled_index(x_native, upsampling_factor))
    y_up = int(_native_to_upsampled_index(y_native, upsampling_factor))
    if x_up < 0 or x_up >= image_up.shape[1] or y_up < 0 or y_up >= image_up.shape[0]:
        return None
    return float(image_up[y_up, x_up])


def _extract_aperture_scalar(
    image_native: np.ndarray,
    x_native: float,
    y_native: float,
    r_ap: float,
) -> float | None:
    """
    Measure a reference scalar with native-image aperture photometry.

    This is used both for the science aperture backend and for the annular
    reference apertures in the local background/noise mode.
    """
    if (
        x_native < r_ap or x_native > (image_native.shape[1] - 1 - r_ap) or
        y_native < r_ap or y_native > (image_native.shape[0] - 1 - r_ap)
    ):
        return None
    return _aperture_sum_native(image_native, x_native, y_native, r_ap)


def _missing_data_native(image_native: np.ndarray) -> np.ndarray:
    """No-data native pixels: non-finite or exactly zero (NaN -> 0 at preprocessing)."""
    image = np.asarray(image_native, dtype=float)
    return (~np.isfinite(image)) | (image == 0.0)


def _measurement_missing_grid(
    *,
    photometry_method: str,
    image_native: np.ndarray,
    upsampling_factor: float,
    fwhm: float,
) -> np.ndarray:
    """
    Map telling whether the photometric scalar read at a position uses no-data pixels.

    "convolve": boolean map on the upsampled grid, True where the circular
        top-hat kernel of `_build_resampled_image` (radius = fwhm) overlaps at
        least one replicated no-data native pixel.  The convolved value there
        mixes real data with zeros that are not measurements.
    "aperture": the native no-data indicator (float); the overlap with the
        aperture is evaluated at each position by `_measurement_touches_missing`.
    """
    missing = _missing_data_native(image_native)
    method = str(photometry_method).lower()
    if method == "aperture":
        return missing.astype(float)
    if method != "convolve":
        raise ValueError("Missing-data maps require 'convolve' or 'aperture'.")
    factor = max(1, int(round(float(upsampling_factor))))
    radius = float(fwhm)
    shape_up = (missing.shape[0] * factor, missing.shape[1] * factor)
    if not np.any(missing):
        return np.zeros(shape_up, dtype=bool)
    from scipy.signal import fftconvolve

    replicated = np.kron(missing.astype(float), np.ones((factor, factor)))
    mask_size = (int(radius + 0.5) * 2 + 1) * factor
    xx, yy = np.mgrid[:mask_size, :mask_size] - mask_size // 2
    kernel = (np.hypot(xx, yy) < radius * factor).astype(float)
    # Same kernel and "same" alignment as _build_resampled_image (odd kernel);
    # counts are integers, 0.5 absorbs FFT round-off.
    counts = fftconvolve(replicated, kernel, mode="same")
    return counts > 0.5


def _measurement_touches_missing(
    missing_grid: Optional[np.ndarray],
    *,
    photometry_method: str,
    x_native: float,
    y_native: float,
    upsampling_factor: float,
    r_ap: float,
) -> bool:
    """True if the scalar measured at (x_native, y_native) uses no-data pixels."""
    if missing_grid is None:
        return False
    if photometry_method == "convolve":
        x_up = int(_native_to_upsampled_index(x_native, upsampling_factor))
        y_up = int(_native_to_upsampled_index(y_native, upsampling_factor))
        if x_up < 0 or y_up < 0 or y_up >= missing_grid.shape[0] or x_up >= missing_grid.shape[1]:
            return True
        return bool(missing_grid[y_up, x_up])
    return _aperture_sum_native(missing_grid, x_native, y_native, r_ap) > 1.0e-9


def _local_aperture_ring_stats(
    *,
    photometry_method: str,
    x_target: float,
    y_target: float,
    r_target: float,
    size: int,
    upsampling_factor: float,
    fwhm: Optional[float],
    image_up: Optional[np.ndarray],
    image_native: Optional[np.ndarray],
    cfg: Optional[dict],
    missing_grid: Optional[np.ndarray] = None,
) -> tuple[float, float, int, bool]:
    """
    Estimate local background and noise from independent reference apertures.

    Method
    ------
    1.  Build one or more thin annuli centred on the tested separation r_target.
    2.  Place reference aperture centres along each annulus with an arc-length
        spacing of `aperture_spacing_fwhm × FWHM`.
    3.  Exclude the arc around the tested source position over an arc-length
        `exclusion_radius_fwhm × FWHM`.  This masks the source and nearby lobes.
    4.  Evaluate the same photometric scalar as the science backend at every
        surviving reference centre.
    5.  Estimate:
            background = median(reference scalars)
            sigma_raw  = robust scatter of the reference scalars
       using either MAD or standard deviation.
    6.  Optionally inflate the local noise with a Student low-statistics
        correction, depending on the YAML configuration and on N_reference.

    Reference apertures whose measurement uses no-data pixels (`missing_grid`,
    see `_measurement_missing_grid`) are skipped: an exact 0 written by the
    reduction where there is no data is not a noise sample.  Keeping them
    pulls the median towards 0 and shrinks the MAD (down to sigma = 0 once
    they are a majority of the ring).

    Returns
    -------
    background, sigma, n_reference, success
    """
    cfg = cfg or {}

    if fwhm is None or not np.isfinite(fwhm) or fwhm <= 0:
        return np.nan, np.nan, 0, False
    if r_target <= 0 or not np.isfinite(r_target):
        return np.nan, np.nan, 0, False

    radial_samples = max(1, int(cfg.get("radial_samples", 1)))
    radial_step = float(cfg.get("radial_step_fwhm", 0.5)) * float(fwhm)
    spacing = max(float(cfg.get("aperture_spacing_fwhm", 1.0)) * float(fwhm), 1e-6)
    exclusion_arc = max(float(cfg.get("exclusion_radius_fwhm", 3.0)) * float(fwhm), 0.0)
    min_ref = max(1, int(cfg.get("min_reference_apertures", 12)))
    sigma_statistic = str(cfg.get("sigma_statistic", "mad")).lower()

    theta_target = float(np.arctan2(y_target - (size - 1) / 2.0, x_target - (size - 1) / 2.0))

    if radial_samples == 1:
        ring_radii = np.array([float(r_target)], dtype=float)
    else:
        offsets = np.linspace(
            -0.5 * (radial_samples - 1),
            0.5 * (radial_samples - 1),
            radial_samples,
            dtype=float,
        ) * radial_step
        ring_radii = float(r_target) + offsets
        ring_radii = ring_radii[ring_radii > 0.0]

    samples: list[float] = []
    r_ap = float(fwhm)

    for rr in ring_radii:
        n_centres = max(int(np.floor(2.0 * np.pi * rr / spacing)), 8)
        phis = np.linspace(0.0, 2.0 * np.pi, n_centres, endpoint=False, dtype=float)

        if exclusion_arc > 0.0:
            exclusion_half_angle = exclusion_arc / max(rr, 1e-6)
            keep = np.abs(_wrap_pi(phis - theta_target)) > exclusion_half_angle
        else:
            keep = np.ones_like(phis, dtype=bool)

        if not np.any(keep):
            continue

        x_ring = (size - 1) / 2.0 + rr * np.cos(phis)
        y_ring = (size - 1) / 2.0 + rr * np.sin(phis)

        for x_ref, y_ref, k_keep in zip(x_ring, y_ring, keep):
            if not k_keep:
                continue

            if photometry_method == "convolve":
                if image_up is None:
                    continue
                value = _extract_convolved_scalar(
                    image_up=image_up,
                    x_native=float(x_ref),
                    y_native=float(y_ref),
                    upsampling_factor=upsampling_factor,
                )
            elif photometry_method == "aperture":
                if image_native is None:
                    continue
                value = _extract_aperture_scalar(
                    image_native=image_native,
                    x_native=float(x_ref),
                    y_native=float(y_ref),
                    r_ap=r_ap,
                )
            else:
                value = None

            if value is None or not np.isfinite(value):
                continue
            if _measurement_touches_missing(
                missing_grid,
                photometry_method=photometry_method,
                x_native=float(x_ref),
                y_native=float(y_ref),
                upsampling_factor=upsampling_factor,
                r_ap=r_ap,
            ):
                continue
            samples.append(float(value))

    if len(samples) < min_ref:
        return np.nan, np.nan, len(samples), False

    samples_arr = np.asarray(samples, dtype=float)
    n_reference = int(samples_arr.size)

    # Pedagogical note:
    # The local background remains the robust median of the surviving
    # reference apertures. Only the local noise estimate can be inflated in
    # the low-statistics regime.
    background = float(np.median(samples_arr))
    sigma_raw = float(_robust_sigma(samples_arr, sigma_statistic))
    sigma = float(_student_corrected_local_sigma(sigma_raw, n_reference, cfg))

    if (not np.isfinite(background)) or (not np.isfinite(sigma)) or sigma <= 0.0:
        return np.nan, np.nan, n_reference, False

    return background, sigma, n_reference, True


def _sanitize_cache_token(value: Any) -> str:
    token = str(value)
    token = token.replace(".", "p")
    token = token.replace("-", "m")
    token = token.replace("/", "_")
    token = token.replace(" ", "")
    return token


def _local_aperture_ring_cache_stem(
    *,
    photometry_method: str,
    size: int,
    upsampling_factor: float,
    fwhm: float,
    cfg: Mapping[str, Any],
) -> str:
    stem = "__".join([
        "local_aperture_ring",
        f"method={_sanitize_cache_token(photometry_method)}",
        f"n={_sanitize_cache_token(size)}",
        f"up={_sanitize_cache_token(upsampling_factor)}",
        f"fwhm={_sanitize_cache_token(fwhm)}",
        f"spacing={_sanitize_cache_token(cfg.get('aperture_spacing_fwhm'))}",
        f"exclude={_sanitize_cache_token(cfg.get('exclusion_radius_fwhm'))}",
        f"radialsamples={_sanitize_cache_token(cfg.get('radial_samples'))}",
        f"radialstep={_sanitize_cache_token(cfg.get('radial_step_fwhm'))}",
        f"minref={_sanitize_cache_token(cfg.get('min_reference_apertures'))}",
        f"sigma={_sanitize_cache_token(cfg.get('sigma_statistic'))}",
        f"smallsample={_sanitize_cache_token(cfg.get('small_sample_correction', 'student'))}",
        f"studentthr={_sanitize_cache_token(cfg.get('student_small_n_threshold', 'always'))}",
        f"studentq={_sanitize_cache_token(cfg.get('student_quantile', 0.8413447460685429))}",
        # Since iteration 02 (2026-11): the no-data rule and "failed estimate
        # = sigma 0 (invalid)" belong to the map definition ("nd2" = version 2
        # of the invalid-pixel convention), so older caches are never reused.
        f"nd2={_sanitize_cache_token(cfg.get('missing_data', 'zero'))}",
    ])
    # Keep "<stem>__epoch=000__background.npy" below the 255-byte file-name
    # limit: an over-long stem keeps its head and gets a digest of the rest.
    if len(stem) > 220:
        import hashlib
        stem = stem[:180] + "__h=" + hashlib.sha1(stem.encode()).hexdigest()[:16]
    return stem


def _cache_epoch_paths(profile_dir: str, stem: str, epoch_index: int) -> tuple[str, str]:
    base = os.path.join(profile_dir, f"{stem}__epoch={epoch_index:03d}")
    return base + "__background.npy", base + "__noise.npy"


def _interp_map_bilinear(
    maps: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    k_idx: np.ndarray,
) -> np.ndarray:
    h = maps.shape[1]
    w = maps.shape[2]

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)

    dx = x - x0
    dy = y - y0

    v00 = maps[(k_idx, y0, x0)]
    v01 = maps[(k_idx, y0, x1)]
    v10 = maps[(k_idx, y1, x0)]
    v11 = maps[(k_idx, y1, x1)]

    return (
        (1.0 - dx) * (1.0 - dy) * v00
        + dx * (1.0 - dy) * v01
        + (1.0 - dx) * dy * v10
        + dx * dy * v11
    ).astype(np.float32)


def _bilinear_support_valid(
    noise_maps: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    k_idx: np.ndarray,
) -> np.ndarray:
    """
    Strict support rule for the bilinear interpolation of sigma and background.

    A sample at (x, y) is valid only if every one of the 4 surrounding pixels
    that receives a non-zero bilinear weight holds a valid estimate (finite
    sigma > 0), and (x, y) lies inside the map.  sigma = 0 means "no estimate"
    (no data under the measurement, or failed reference statistics); mixing
    it into the interpolation would invent a noise level pulled towards 0.
    Invalid samples are treated exactly like samples outside r_mask/r_mask_ext:
    they carry no information and are excluded from the combination.
    """
    h = noise_maps.shape[1]
    w = noise_maps.shape[2]
    inside = (x >= 0.0) & (x <= w - 1) & (y >= 0.0) & (y <= h - 1)

    x0 = np.clip(np.floor(x).astype(np.int32), 0, w - 1)
    y0 = np.clip(np.floor(y).astype(np.int32), 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    dx = x - x0
    dy = y - y0

    def ok(yy, xx):
        v = noise_maps[(k_idx, yy, xx)]
        return np.isfinite(v) & (v > 0.0)

    valid = inside.copy()
    valid &= ok(y0, x0) | ((1.0 - dx) * (1.0 - dy) <= 0.0)
    valid &= ok(y0, x1) | (dx * (1.0 - dy) <= 0.0)
    valid &= ok(y1, x0) | ((1.0 - dx) * dy <= 0.0)
    valid &= ok(y1, x1) | (dx * dy <= 0.0)
    return valid


def _bilinear_interp_valid_neighbours(
    bkg_maps: np.ndarray,
    noise_maps: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    k_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bilinear interpolation of background and sigma on the VALID neighbours only.

    Rule (a) of rapports/notes_bruit_masque.md, used since iteration 07.  The
    4 pixels around (x, y) receive the usual bilinear weights w_j.  A pixel is
    valid when its sigma is finite and > 0 (sigma = 0 is the "no estimate"
    code).  The interpolated values are

        sigma = sum_j w_j v_j sigma_j / sum_j w_j v_j      (v_j = 1 if valid)
        bkg   = sum_j w_j v_j bkg_j   / sum_j w_j v_j

    i.e. invalid pixels are dropped and the remaining weights renormalised.
    The sample is supported when (x, y) lies inside the map and at least one
    neighbour with a non-zero weight is valid.  When every weighted neighbour
    is valid (the only case accepted by the strict rule
    `_bilinear_support_valid`), the values are exactly those of
    `_interp_map_bilinear`.

    This function does not test whether the science measurement itself uses
    no-data pixels: see `_measurement_missing_at` (both are required).

    Returns (supported, bkg, sigma); unsupported samples get bkg = sigma = 0.
    """
    h = noise_maps.shape[1]
    w = noise_maps.shape[2]
    inside = (x >= 0.0) & (x <= w - 1) & (y >= 0.0) & (y <= h - 1)

    x0 = np.clip(np.floor(x).astype(np.int32), 0, w - 1)
    y0 = np.clip(np.floor(y).astype(np.int32), 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    dx = x - x0
    dy = y - y0

    corners = (
        (y0, x0, (1.0 - dx) * (1.0 - dy)),
        (y0, x1, dx * (1.0 - dy)),
        (y1, x0, (1.0 - dx) * dy),
        (y1, x1, dx * dy),
    )
    w_valid = np.zeros(np.shape(x), dtype=float)
    num_sig = np.zeros(np.shape(x), dtype=float)
    num_bkg = np.zeros(np.shape(x), dtype=float)
    complete = inside.copy()
    for yy, xx, wgt in corners:
        s = noise_maps[(k_idx, yy, xx)]
        b = bkg_maps[(k_idx, yy, xx)]
        good = np.isfinite(s) & (s > 0.0) & np.isfinite(b)
        weighted = wgt > 0.0
        complete &= good | ~weighted
        use = good & weighted
        w_valid += np.where(use, wgt, 0.0)
        num_sig += np.where(use, wgt * np.where(use, s, 0.0), 0.0)
        num_bkg += np.where(use, wgt * np.where(use, b, 0.0), 0.0)

    supported = inside & (w_valid > 0.0)
    partial = supported & ~complete

    sig = np.zeros(np.shape(x), dtype=np.float32)
    bkg = np.zeros(np.shape(x), dtype=np.float32)
    if np.any(complete):
        # Bit-identical to the historical interpolation where all weighted
        # neighbours are valid.
        sig_full = _interp_map_bilinear(noise_maps, x, y, k_idx)
        bkg_full = _interp_map_bilinear(bkg_maps, x, y, k_idx)
        sig[complete] = sig_full[complete]
        bkg[complete] = bkg_full[complete]
    if np.any(partial):
        sig[partial] = (num_sig[partial] / w_valid[partial]).astype(np.float32)
        bkg[partial] = (num_bkg[partial] / w_valid[partial]).astype(np.float32)
    return supported, bkg, sig


def _measurement_missing_at(
    missing_maps: np.ndarray,
    *,
    photometry_method: str,
    x_pix: np.ndarray,
    y_pix: np.ndarray,
    k_idx: np.ndarray,
    upsampling_factor: float,
    fwhm: Optional[float],
    candidates: np.ndarray,
) -> np.ndarray:
    """
    True where the science measurement at (x_pix, y_pix) uses no-data pixels.

    "convolve": the value of `missing_maps` at the same upsampled index as the
    science read-out (`_native_to_upsampled_index`), i.e. whether the
    convolution disk of THAT sample overlaps a no-data native pixel.
    "aperture": the aperture (radius fwhm) overlaps a no-data native pixel.
    Only `candidates` (already valid samples) are evaluated; out-of-map
    samples count as touching missing data.
    """
    touches = np.zeros(np.shape(x_pix), dtype=bool)
    method = str(photometry_method).lower()
    if method == "convolve":
        hu = missing_maps.shape[1]
        wu = missing_maps.shape[2]
        xu = _native_to_upsampled_index(x_pix, upsampling_factor)
        yu = _native_to_upsampled_index(y_pix, upsampling_factor)
        inside = (0 <= xu) & (xu < wu) & (0 <= yu) & (yu < hu)
        xs = np.clip(xu, 0, wu - 1)
        ys = np.clip(yu, 0, hu - 1)
        touches = np.asarray(missing_maps[(k_idx, ys, xs)], dtype=bool) | ~inside
        return touches & candidates
    if method == "aperture":
        r_ap = float(fwhm)
        for ii, kk in zip(*np.nonzero(candidates)):
            kk_map = int(np.asarray(k_idx)[ii, kk])
            touches[ii, kk] = (
                _aperture_sum_native(
                    missing_maps[kk_map], float(x_pix[ii, kk]), float(y_pix[ii, kk]), r_ap
                ) > 1.0e-9
            )
        return touches
    raise ValueError("Missing-data maps require 'convolve' or 'aperture'.")


def _compute_local_ring_maps_for_epoch(
    *,
    epoch_index: int,
    photometry_method: str,
    size: int,
    upsampling_factor: float,
    fwhm: float,
    image_up: Optional[np.ndarray],
    image_native: Optional[np.ndarray],
    cfg: Mapping[str, Any],
    noise_floor: float,
    missing_grid: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute local-aperture-ring background and noise maps for one epoch.

    A pixel is INVALID (background 0, sigma 0) when its own measurement uses
    no-data pixels (`missing_grid`) or when the reference apertures do not
    give a positive finite sigma.  sigma = 0 is the "no estimate" code read by
    the likelihood: it never enters an interpolation (see
    `_bilinear_interp_valid_neighbours`).  `noise_floor` is kept for API compatibility;
    the numerical floor is applied later, to valid samples only.
    """
    bg_map = np.zeros((size, size), dtype=np.float32)
    noise_map = np.zeros((size, size), dtype=np.float32)
    r_ap = float(fwhm) if fwhm is not None else 0.0

    row_iter = tqdm(
        range(size),
        desc=f"[bgnoise] epoch {epoch_index}",
        leave=False,
    )

    for y in row_iter:
        for x in range(size):
            r_target = float(
                np.hypot(
                    x - (size - 1) / 2.0,
                    y - (size - 1) / 2.0,
                )
            )

            if _measurement_touches_missing(
                missing_grid,
                photometry_method=photometry_method,
                x_native=float(x),
                y_native=float(y),
                upsampling_factor=upsampling_factor,
                r_ap=r_ap,
            ):
                continue  # no data under the measurement: stays invalid

            bg_loc, sig_loc, _, ok_loc = _local_aperture_ring_stats(
                photometry_method=photometry_method,
                x_target=float(x),
                y_target=float(y),
                r_target=r_target,
                size=size,
                upsampling_factor=upsampling_factor,
                fwhm=fwhm,
                image_up=image_up,
                image_native=image_native,
                cfg=cfg,
                missing_grid=missing_grid,
            )

            if ok_loc:
                bg_map[y, x] = float(bg_loc)
                noise_map[y, x] = float(sig_loc)
            else:
                bg_map[y, x] = 0.0
                noise_map[y, x] = 0.0

    return bg_map, noise_map


def _load_or_build_local_ring_maps(
    *,
    profile_dir: str,
    photometry_method: str,
    size: int,
    upsampling_factor: float,
    fwhm: float,
    images_up: Optional[np.ndarray],
    images_native: Optional[np.ndarray],
    cfg: Mapping[str, Any],
    noise_floor: float,
    instrument_name: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load cached local-ring maps or compute and cache missing epoch pairs.

    With `instrument_name`, the maps are cached in profile_dir/<instrument>/.
    A cached map whose shape differs from (size, size) is never reused.
    """
    if profile_dir is None:
        raise ValueError("profile_dir is required to cache local aperture-ring maps.")

    profile_dir = _instrument_profile_dir(profile_dir, instrument_name)
    os.makedirs(profile_dir, exist_ok=True)

    cache_dtype = (
        np.float32
        if str(cfg.get("cache_dtype", "float32")).lower() == "float32"
        else np.float64
    )

    stem = _local_aperture_ring_cache_stem(
        photometry_method=photometry_method,
        size=size,
        upsampling_factor=upsampling_factor,
        fwhm=fwhm,
        cfg=cfg,
    )

    source = images_up if images_up is not None else images_native
    if source is None:
        raise ValueError("Local aperture-ring maps need image data.")

    n_epochs = int(source.shape[0])
    overwrite = bool(cfg.get("overwrite_cached_maps", False))
    use_missing = (
        str(cfg.get("missing_data", "zero")).lower() == "zero"
        and images_native is not None
        and str(photometry_method).lower() in ("convolve", "aperture")
    )

    bg_maps = []
    noise_maps = []

    print()
    if instrument_name is not None:
        print(f"[bgnoise] instrument : {instrument_name}")
    print(f"[bgnoise] cache stem : {stem}")
    console_status("noise", f"Profile cache: {profile_dir}", tone="blue")
    print()

    for epoch_index in tqdm(
        range(n_epochs),
        desc="[bgnoise] local maps",
        leave=True,
    ):
        bg_path, noise_path = _cache_epoch_paths(
            profile_dir,
            stem,
            epoch_index,
        )

        cached = (
            not overwrite
            and os.path.exists(bg_path)
            and os.path.exists(noise_path)
        )
        if cached:
            bg_map = np.load(bg_path).astype(cache_dtype, copy=False)
            noise_map = np.load(noise_path).astype(cache_dtype, copy=False)
            if bg_map.shape != (size, size) or noise_map.shape != (size, size):
                console_status(
                    "warning",
                    f"Local-ring cache for epoch {epoch_index} has shape "
                    f"{noise_map.shape}, expected {(size, size)}: recomputing.",
                    tone="yellow",
                )
                cached = False
        if not cached:
            missing_grid = None
            if use_missing:
                missing_grid = _measurement_missing_grid(
                    photometry_method=photometry_method,
                    image_native=images_native[epoch_index],
                    upsampling_factor=upsampling_factor,
                    fwhm=fwhm,
                )
                if (
                    photometry_method == "convolve"
                    and images_up is not None
                    and missing_grid.shape != tuple(np.shape(images_up[epoch_index]))
                ):
                    raise ValueError(
                        f"{instrument_name}: no-data grid {missing_grid.shape} does not "
                        f"match the convolved image {np.shape(images_up[epoch_index])}."
                    )
            bg_map, noise_map = _compute_local_ring_maps_for_epoch(
                epoch_index=epoch_index,
                photometry_method=photometry_method,
                size=size,
                upsampling_factor=upsampling_factor,
                fwhm=fwhm,
                image_up=None if images_up is None else images_up[epoch_index],
                image_native=None if images_native is None else images_native[epoch_index],
                cfg=cfg,
                noise_floor=noise_floor,
                missing_grid=missing_grid,
            )

            bg_map = bg_map.astype(cache_dtype, copy=False)
            noise_map = noise_map.astype(cache_dtype, copy=False)

            np.save(bg_path, bg_map)
            np.save(noise_path, noise_map)

        bg_maps.append(bg_map)
        noise_maps.append(noise_map)

    noise_arr = np.asarray(noise_maps, dtype=cache_dtype)
    n_invalid = np.sum(~(np.isfinite(noise_arr) & (noise_arr > 0)), axis=(1, 2))
    print(f"[bgnoise] missing data mode   : {cfg.get('missing_data', 'zero')}")
    if images_native is not None:
        n_nodata = np.sum(_missing_data_native(images_native), axis=(1, 2))
        print(f"[bgnoise] no-data native px   : {int(n_nodata.min())} .. {int(n_nodata.max())} "
              f"per epoch (of {size * size})")
    print(f"[bgnoise] invalid map px      : {int(n_invalid.min())} .. {int(n_invalid.max())} "
          f"per epoch (sigma = 0: no data under the measurement or no estimate)")

    return (
        np.asarray(bg_maps, dtype=cache_dtype),
        noise_arr,
    )


def _classical_measurement_maps(
    *,
    photometry_method: str,
    images_up: Optional[np.ndarray],
    images_native: np.ndarray,
    size: int,
    upsampling_factor: float,
    fwhm: float,
) -> np.ndarray:
    """
    Build the scalar photometric measurement evaluated on every native pixel.

    This is used only once when radial noise recalibration is enabled.  It makes
    the calibration statistic exactly match the scalar later sampled by the
    orbit likelihood.
    """
    method = str(photometry_method).lower()
    yy, xx = np.mgrid[:int(size), :int(size)]

    if method == "convolve":
        if images_up is None:
            raise ValueError("convolve noise recalibration requires images_up.")
        factor = float(upsampling_factor)
        ix = _native_to_upsampled_index(xx, factor).astype(int)
        iy = _native_to_upsampled_index(yy, factor).astype(int)
        out = np.full((len(images_up), size, size), np.nan, dtype=np.float64)
        for k, image in enumerate(np.asarray(images_up)):
            valid = (
                (ix >= 0) & (iy >= 0)
                & (ix < image.shape[1]) & (iy < image.shape[0])
            )
            out[k][valid] = image[iy[valid], ix[valid]]
        return out

    if method == "aperture":
        if images_native is None:
            raise ValueError("aperture noise recalibration requires images_native.")
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        aperture = CircularAperture(positions, r=float(fwhm))
        out = np.empty((len(images_native), size, size), dtype=np.float64)
        for k, image in enumerate(np.asarray(images_native)):
            phot = aperture_photometry(image, aperture)
            out[k] = np.asarray(phot["aperture_sum"], dtype=float).reshape(size, size)
        return out

    raise ValueError("Classical measurement maps require 'convolve' or 'aperture'.")


def _radial_recalibration_metadata(
    *,
    instrument_name: Optional[str],
    photometry_method: str,
    size: int,
    n_epochs: int,
    upsampling_factor: float,
    fwhm: float,
    r_mask: Optional[float],
    cfg: Mapping[str, Any],
    arrays: Sequence[Optional[np.ndarray]],
) -> dict:
    """JSON-serialisable description of what a recalibration cache depends on.

    The checksum covers the local background/noise maps and the science images
    from which g(d) is estimated, so a cache built from other inputs (another
    instrument, other local-ring settings, other images) is never reused.
    """
    import hashlib

    digest = hashlib.sha1()
    for array in arrays:
        if array is None:
            digest.update(b"none")
            continue
        arr = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
        digest.update(str(arr.shape).encode())
        digest.update(arr.tobytes())
    settings = {
        key: (float(cfg[key]) if isinstance(cfg.get(key), (int, float)) and not isinstance(cfg.get(key), bool) else str(cfg.get(key)))
        for key in (
            "mode", "bin_width_px", "max_distance_px", "min_samples_per_bin",
            "smoothing_bins", "min_factor", "monotonic_extent_px",
            "extrapolation_fit_min_px", "extrapolation_fit_max_px",
        )
    }
    return dict(
        instrument=None if instrument_name is None else str(instrument_name),
        photometry_method=str(photometry_method).lower(),
        size=int(size),
        n_epochs=int(n_epochs),
        upsampling_factor=float(upsampling_factor),
        fwhm=float(fwhm),
        r_mask=None if r_mask is None else float(r_mask),
        settings=settings,
        inputs_sha1=digest.hexdigest(),
    )


def _recalibrate_classical_noise(
    *,
    photometry_method: str,
    images_up: Optional[np.ndarray],
    images_native: np.ndarray,
    size: int,
    upsampling_factor: float,
    fwhm: float,
    r_mask: Optional[float],
    xgrid: np.ndarray,
    bkg: np.ndarray,
    noise: np.ndarray,
    local_bkg_maps: Optional[np.ndarray],
    local_noise_maps: Optional[np.ndarray],
    cfg: Mapping[str, Any],
    profile_dir: str,
    instrument_name: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Recalibrate local-aperture-ring noise with a robust radial factor g(d).

    The background map is left unchanged. Only sigma is rescaled:

        z_raw = (measurement - background) / sigma_raw
        sigma_cal(d) = g(d) * sigma_raw(d)

    The recalibration is applied to the cached 2-D local-noise maps used by
    convolve/aperture likelihood evaluation.

    g(d) is estimated from the epochs passed here only, i.e. from ONE
    instrument, with d = r - r_mask of that instrument.  The cache lives in
    profile_dir/<instrument>/radial_noise_recalibration/<settings>/<mode>/<method>/
    and is reused only if its metadata (instrument, map shape, number of
    epochs, FWHM, r_mask, settings, checksum of the input maps) match.
    """
    mode = _normalize_radial_noise_recalibration_mode(cfg.get("mode", "none"))
    if mode == "none":
        return xgrid, bkg, noise, local_noise_maps

    if local_bkg_maps is None or local_noise_maps is None:
        raise RuntimeError(
            "Radial noise recalibration requires local aperture-ring background "
            "and noise maps. Enable background_noise.precompute_maps."
        )

    cache_root = _radial_recalibration_cache_dir(
        profile_dir, instrument_name, cfg, photometry_method
    )
    profile_path = os.path.join(cache_root, "g_profile.npz")
    overwrite = bool(cfg.get("overwrite_cache", False))
    use_cache = bool(cfg.get("cache", True))

    n_epochs = int(np.asarray(local_noise_maps).shape[0])
    expected_shape = (n_epochs, int(size), int(size))
    if (
        tuple(np.shape(local_noise_maps)) != expected_shape
        or tuple(np.shape(local_bkg_maps)) != expected_shape
    ):
        raise ValueError(
            f"{instrument_name}: local background/noise maps have shapes "
            f"{np.shape(local_bkg_maps)} / {np.shape(local_noise_maps)}, "
            f"expected {expected_shape}."
        )

    cached_noise_paths = [
        os.path.join(cache_root, f"noise_recalibrated_epoch_{k}.npy")
        for k in range(n_epochs)
    ]
    metadata = _radial_recalibration_metadata(
        instrument_name=instrument_name,
        photometry_method=photometry_method,
        size=size,
        n_epochs=n_epochs,
        upsampling_factor=upsampling_factor,
        fwhm=fwhm,
        r_mask=r_mask,
        cfg=cfg,
        arrays=(
            local_bkg_maps,
            local_noise_maps,
            images_up if str(photometry_method).lower() == "convolve" else images_native,
        ),
    )

    print()
    print("=" * 110)
    print(f"RADIAL NOISE RECALIBRATION — {str(photometry_method).upper()}"
          + ("" if instrument_name is None else f" — {instrument_name}"))
    print("=" * 110)
    print()
    print(f"  instrument           : {instrument_name}")
    print(f"  mode                 : {mode}")
    print("  definition           : sigma_cal(d) = g(d) * sigma_raw(d)")
    print("  standardized field   : z_raw = (measurement - background) / sigma_raw")
    print(f"  epochs used for g(d) : {n_epochs} (this instrument only)")
    print(f"  map shape            : {expected_shape[1]} x {expected_shape[2]} px")
    print(f"  inner edge r_mask    : {r_mask} px  (d = r - r_mask)")
    print(f"  radial bin           : {float(cfg['bin_width_px']):g} native px")
    print(f"  calibrated edge zone : 0 .. {float(cfg['max_distance_px']):g} native px")
    print(f"  max_distance_px      : {float(cfg['max_distance_px']):g} px "
          f"({cfg.get('max_distance_origin', 'n/a')})")
    print(f"  cache directory      : {cache_root}")
    print("  background           : unchanged")
    print()

    cache_ready = (
        use_cache
        and not overwrite
        and os.path.isfile(profile_path)
        and all(os.path.isfile(path) for path in cached_noise_paths)
    )

    loaded = None
    if cache_ready:
        problem = None
        try:
            with np.load(profile_path) as cached_profile:
                stored = (
                    json.loads(str(cached_profile["metadata"]))
                    if "metadata" in cached_profile.files
                    else None
                )
            if stored != metadata:
                problem = "metadata differ (other instrument, settings or input maps)"
            else:
                loaded = np.asarray(
                    [np.load(path) for path in cached_noise_paths],
                    dtype=float,
                )
                if loaded.shape != expected_shape:
                    problem = f"map shape {loaded.shape} != expected {expected_shape}"
        except Exception as exc:  # unreadable cache: recompute
            problem = f"unreadable cache ({exc})"
        if problem is not None:
            console_status(
                "warning",
                f"{instrument_name}: recalibration cache {cache_root} rejected: "
                f"{problem}. Recomputing g(d).",
                tone="yellow",
            )
            cache_ready = False
            loaded = None

    if cache_ready:
        print("  cache                : HIT (metadata and shape checked)")
        try:
            with np.load(profile_path) as cached_profile:
                g_cached = np.asarray(cached_profile["g"], dtype=float)
                g_cached = g_cached[np.isfinite(g_cached)]
                if g_cached.size:
                    print(f"  cached g range       : {float(np.min(g_cached)):.3f} .. "
                          f"{float(np.max(g_cached)):.3f}")
                print(f"  inner_radius_px      : {float(cached_profile['inner_radius_px']):g}")
        except Exception:
            pass
        print(f"  recalibrated maps    : {loaded.shape[1]} x {loaded.shape[2]} px, "
              f"{loaded.shape[0]} epochs")
        print("  applied to           : local aperture-ring noise maps")
        print("  status               : ready")
        print()
        print("-" * 110)
        print()
        return xgrid, bkg, noise, loaded

    print("  cache                : MISS — estimating g(d)")
    print()

    measurement = _classical_measurement_maps(
        photometry_method=photometry_method,
        images_up=images_up,
        images_native=images_native,
        size=size,
        upsampling_factor=upsampling_factor,
        fwhm=fwhm,
    )

    background_map = np.asarray(local_bkg_maps, dtype=float)
    sigma_map = np.asarray(local_noise_maps, dtype=float)

    center = ((size - 1.0) / 2.0, (size - 1.0) / 2.0)
    yy, xx = np.mgrid[:size, :size]
    radius = np.hypot(xx - center[0], yy - center[1])

    valid = (
        np.isfinite(measurement)
        & np.isfinite(background_map)
        & np.isfinite(sigma_map)
        & (sigma_map > 0.0)
    )

    if r_mask is not None:
        valid &= radius[None, :, :] > float(r_mask)

    z = np.full_like(measurement, np.nan, dtype=float)
    z[valid] = (
        measurement[valid] - background_map[valid]
    ) / sigma_map[valid]

    profile = _estimate_radial_noise_profile(
        z,
        valid,
        cfg=cfg,
        center_xy=center,
        inner_radius=None if r_mask is None else float(r_mask),
    )

    g_map = _g_map_from_profile(
        profile,
        cfg=cfg,
    )

    recalibrated = (
        np.asarray(local_noise_maps, dtype=float)
        * g_map
    )

    finite_g = np.asarray(profile["g"])[
        np.isfinite(profile["g"])
    ]
    if finite_g.size:
        print(
            f"  fitted g range       : "
            f"{float(np.min(finite_g)):.3f} .. "
            f"{float(np.max(finite_g)):.3f}"
        )
    print(f"  inner_radius_px      : {float(profile['inner_radius_px']):g}")
    print(f"  recalibrated maps    : {recalibrated.shape[1]} x {recalibrated.shape[2]} px, "
          f"{recalibrated.shape[0]} epochs")

    if use_cache:
        os.makedirs(cache_root, exist_ok=True)

        np.savez(
            profile_path,
            distance_centers_px=np.asarray(profile["distance_centers_px"]),
            g_raw=np.asarray(profile["g_raw"]),
            g_empirical=np.asarray(profile["g_empirical"]),
            g=np.asarray(profile["g"]),
            per_epoch=np.asarray(profile["per_epoch"]),
            inner_radius_px=np.asarray(profile["inner_radius_px"]),
            max_distance_px=np.asarray(float(cfg["max_distance_px"])),
            metadata=np.asarray(json.dumps(metadata)),
        )

        for k in range(recalibrated.shape[0]):
            np.save(
                cached_noise_paths[k],
                recalibrated[k],
            )

    print("  applied to           : local aperture-ring noise maps")
    print(
        "  cache saved          : yes"
        if use_cache
        else "  cache saved          : disabled"
    )
    print("  status               : ready")
    print()
    print("-" * 110)
    print()

    return xgrid, bkg, noise, recalibrated


def snr_from_hkpq(
    theta: np.ndarray,
    *,
    ts: np.ndarray,
    images: Optional[np.ndarray],
    data: dict,
    size: int,
    scale: float,
    upsampling_factor: float,
    t_ref: float = 0.0,
    r_mask: float | None = None,
    r_mask_ext: float | None = None,
    noise_floor: float = 1.0,
    weighting: str = "invvar",
    photometry_method: str = "convolve",
    images_native: Optional[np.ndarray] = None,
    fwhm: Optional[float] = None,
    paco_alpha_maps: Optional[np.ndarray] = None,
    paco_var_alpha_maps: Optional[np.ndarray] = None,
    paco_interpolator: str = "none",
    paco_oversampling: int = 1,
    background_noise_cfg: Optional[dict] = None,
    local_bkg_maps: Optional[np.ndarray] = None,
    local_noise_maps: Optional[np.ndarray] = None,
    return_epoch_snr: bool = False,
    return_epoch_valid: bool = False,
    measurement_missing_maps: Optional[np.ndarray] = None,
):
    """Compute per-walker combined signal, noise, and SNR across all epochs.

    With ``return_epoch_valid`` (diagnostics only) the per-epoch S/N and a
    boolean (W, K) array of the epochs that entered the combination are also
    returned: (signal, noise, snr, epoch_snr, epoch_valid).

    Invalid pixels of the precomputed noise maps (sigma = 0): with
    ``measurement_missing_maps`` (set on every Instrument by
    `configure_instrument_statistics`), a sample is valid when the measurement
    at its exact position uses no no-data pixel and at least one weighted
    bilinear neighbour is valid; sigma and background are interpolated on the
    valid neighbours with renormalised weights (rule "a", iteration 07).
    Without it, the strict rule of iterations 02-06 applies (any weighted
    invalid neighbour rejects the sample).

    ``convolve`` and ``aperture`` retain their existing classical photometry
    and background/noise machinery. ``paco`` reads alpha_hat and var_alpha either
    at the deterministic nearest native pixel (interpolator="none") or on a
    cached oversampled PACO interpolation grid. It performs no aperture photometry,
    convolution, or extra background/noise estimation.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.ndim == 1:
        theta = theta[None, :]
    if theta.shape[1] != 7:
        raise ValueError("theta must have 7 columns: [a, la0, m0, h, k, p, q]")

    a, la0, m0, h, k, p, q = theta.T
    W = theta.shape[0]
    r2 = p * p + q * q
    ok = np.isfinite(theta).all(axis=1) & (r2 <= 1.0) & (a > 0.0) & (m0 > 0.0)
    e = np.sqrt(np.maximum(0.0, h * h + k * k))
    ok &= (e < 1.0)
    if not np.any(ok):
        signal = np.zeros(W, dtype=float)
        noise = np.full(W, float(noise_floor), dtype=float)
        snr = signal / noise
        if return_epoch_valid:
            K0 = len(np.asarray(ts))
            return signal, noise, snr, np.zeros((W, K0), dtype=float), np.zeros((W, K0), dtype=bool)
        if return_epoch_snr:
            return signal, noise, snr, np.zeros((W, len(np.asarray(ts))), dtype=float)
        return signal, noise, snr

    idx = np.where(ok)[0]
    a_ok, la0_ok, m0_ok, h_ok, k_ok, p_ok, q_ok, e_ok = (
        a[idx], la0[idx], m0[idx], h[idx], k[idx], p[idx], q[idx], e[idx]
    )
    # Direct non-singular propagation.  λ evolves linearly in time and the
    # equinoctial Kepler equation is solved without reconstructing Ω, ω or t0.

    ts = np.asarray(ts, dtype=float)
    K = int(len(ts))
    if photometry_method == "convolve":
        if images is None or images.shape[0] != K:
            raise ValueError("convolve mode requires images with first dimension K == len(ts).")
    elif photometry_method == "aperture":
        if images_native is None:
            raise ValueError("aperture mode requires images_native=(K,size,size).")
        if fwhm is None:
            raise ValueError("aperture mode requires a finite fwhm (native px).")
        if images_native.shape != (K, size, size):
            raise ValueError("images_native must be (K, size, size).")
    elif photometry_method == "paco":
        if paco_alpha_maps is None or paco_var_alpha_maps is None:
            raise ValueError("paco mode requires paco_alpha_maps and paco_var_alpha_maps.")
        paco_method = _normalize_paco_interpolator(paco_interpolator)
        paco_factor = _effective_paco_oversampling(paco_method, paco_oversampling)
        paco_side = _paco_map_side(size, paco_method, paco_factor)
        if paco_alpha_maps.shape != (K, paco_side, paco_side):
            raise ValueError(
                f"paco_alpha_maps must be (K,{paco_side},{paco_side}) for "
                f"interpolator={paco_method!r}, oversampling={paco_factor}."
            )
        if paco_var_alpha_maps.shape != (K, paco_side, paco_side):
            raise ValueError(
                f"paco_var_alpha_maps must be (K,{paco_side},{paco_side}) for "
                f"interpolator={paco_method!r}, oversampling={paco_factor}."
            )
    else:
        raise ValueError(f"Unknown photometry_method: {photometry_method!r}")

    north_sky, east_sky = _equinoctial_sky_tracks(
        a_ok, la0_ok, m0_ok, h_ok, k_ok, p_ok, q_ok, ts, t_ref=t_ref
    )

    center = (float(size) - 1.0) / 2.0
    x_pix = (-east_sky) * scale + center
    y_pix = north_sky * scale + center
    r = np.hypot(x_pix - center, y_pix - center)

    validpix = np.ones((idx.size, K), dtype=bool)
    if r_mask is not None:
        validpix &= (r > r_mask)
    if r_mask_ext is not None:
        validpix &= (r < r_mask_ext)
    k_idx = np.broadcast_to(np.arange(K), (idx.size, K))

    if photometry_method == "convolve":
        x_up = _native_to_upsampled_index(x_pix, upsampling_factor).astype(np.int32)
        y_up = _native_to_upsampled_index(y_pix, upsampling_factor).astype(np.int32)
        validpix &= ((0 <= x_up) & (x_up < images.shape[2]) &
                     (0 <= y_up) & (y_up < images.shape[1]))
        x_up = np.clip(x_up, 0, images.shape[2] - 1)
        y_up = np.clip(y_up, 0, images.shape[1] - 1)
        flux = images[(k_idx, y_up, x_up)].astype(np.float32)
    elif photometry_method == "aperture":
        validpix &= (x_pix >= 0) & (x_pix < size) & (y_pix >= 0) & (y_pix < size)
        flux = np.zeros((idx.size, K), dtype=np.float32)
        r_ap = float(fwhm)
        for ii in range(idx.size):
            for kk in range(K):
                if validpix[ii, kk]:
                    flux[ii, kk] = _aperture_sum_native(
                        images_native[kk], float(x_pix[ii, kk]), float(y_pix[ii, kk]), r_ap
                    )
    else:  # PACO: native nearest or nearest point on the selected oversampled interpolation grid.
        paco_method = _normalize_paco_interpolator(paco_interpolator)
        paco_factor = _effective_paco_oversampling(paco_method, paco_oversampling)
        paco_side = paco_alpha_maps.shape[1]
        if paco_method == "none":
            x_map = np.floor(x_pix + 0.5).astype(np.int32)
            y_map = np.floor(y_pix + 0.5).astype(np.int32)
        else:
            x_map = np.floor(x_pix * paco_factor + 0.5).astype(np.int32)
            y_map = np.floor(y_pix * paco_factor + 0.5).astype(np.int32)
        validpix &= ((0 <= x_map) & (x_map < paco_side) & (0 <= y_map) & (y_map < paco_side))
        x_safe = np.clip(x_map, 0, paco_side - 1)
        y_safe = np.clip(y_map, 0, paco_side - 1)
        flux = paco_alpha_maps[(k_idx, y_safe, x_safe)].astype(np.float32)
        var = paco_var_alpha_maps[(k_idx, y_safe, x_safe)].astype(np.float32)
        validpix &= np.isfinite(flux) & np.isfinite(var) & (var > 0.0)
        sig = np.zeros_like(var, dtype=np.float32)
        sig[validpix] = np.sqrt(var[validpix])
        bg = np.zeros_like(flux, dtype=np.float32)

    flux[~validpix] = 0.0

    if photometry_method != "paco":
        bg_cfg = background_noise_cfg or {}
        map_mode = str(
            bg_cfg.get("local_map_mode", "precompute_cache")
        ).lower()

        use_precomputed_maps = (
            map_mode == "precompute_cache"
            and local_bkg_maps is not None
            and local_noise_maps is not None
        )

        if use_precomputed_maps:
            if measurement_missing_maps is not None:
                # Rule (a), since iteration 07: the sample is valid when the
                # measurement at the exact position uses no no-data pixel;
                # sigma and background are interpolated on the valid
                # neighbours only, with renormalised bilinear weights.
                validpix &= ~_measurement_missing_at(
                    measurement_missing_maps,
                    photometry_method=photometry_method,
                    x_pix=x_pix, y_pix=y_pix, k_idx=k_idx,
                    upsampling_factor=upsampling_factor, fwhm=fwhm,
                    candidates=validpix,
                )
                supported, bg, sig = _bilinear_interp_valid_neighbours(
                    local_bkg_maps, local_noise_maps, x_pix, y_pix, k_idx,
                )
                validpix &= supported
                flux[~validpix] = 0.0
            else:
                # Legacy strict rule (iterations 02-06, kept for low-level
                # calls without measurement maps): a sample whose
                # interpolation would use an invalid (sigma = 0) pixel is
                # itself invalid (no extrapolation).
                validpix &= _bilinear_support_valid(local_noise_maps, x_pix, y_pix, k_idx)
                flux[~validpix] = 0.0
                bg = _interp_map_bilinear(
                    local_bkg_maps,
                    x_pix,
                    y_pix,
                    k_idx,
                )
                sig = _interp_map_bilinear(
                    local_noise_maps,
                    x_pix,
                    y_pix,
                    k_idx,
                )
            bg[~validpix] = 0.0
            sig[~validpix] = 0.0
        else:
            bg = np.zeros((idx.size, K), dtype=np.float32)
            sig = np.full(
                (idx.size, K),
                float(noise_floor),
                dtype=np.float32,
            )

            for ii in range(idx.size):
                for kk in range(K):
                    if not validpix[ii, kk]:
                        continue

                    bg_loc, sig_loc, _, ok_loc = _local_aperture_ring_stats(
                        photometry_method=photometry_method,
                        x_target=float(x_pix[ii, kk]),
                        y_target=float(y_pix[ii, kk]),
                        r_target=float(r[ii, kk]),
                        size=size,
                        upsampling_factor=upsampling_factor,
                        fwhm=fwhm,
                        image_up=(
                            images[kk]
                            if photometry_method == "convolve"
                            else None
                        ),
                        image_native=(
                            images_native[kk]
                            if photometry_method == "aperture"
                            else None
                        ),
                        cfg=bg_cfg,
                    )

                    if ok_loc:
                        bg[ii, kk] = float(bg_loc)
                        sig[ii, kk] = float(sig_loc)

            bg[~validpix] = 0.0
            sig[~validpix] = 0.0

    # Apply the resolved numerical floor to every valid per-epoch sigma.
    # Invalid or masked samples remain exactly zero and therefore contribute
    # no information.
    valid_sigma = (
        validpix
        & np.isfinite(sig)
        & (sig > 0.0)
    )
    sig[valid_sigma] = np.maximum(
        sig[valid_sigma],
        float(noise_floor),
    )
    sig[~valid_sigma] = 0.0

    y = flux - bg

    # Signed mono-epoch S/N; positive-profile mode clips it only later.
    # Soft-masked, out-of-map, or invalid samples contribute exactly zero.
    epoch_snr_ok = np.zeros_like(y, dtype=float)
    epoch_valid = validpix & np.isfinite(y) & np.isfinite(sig) & (sig > 0.0)
    np.divide(y, sig, out=epoch_snr_ok, where=epoch_valid)
    epoch_snr_ok[~epoch_valid] = 0.0

    if weighting == "simple":
        signal_ok = np.sum(y, axis=1)
        noise_ok = np.sqrt(np.sum(sig * sig, axis=1))
    elif weighting == "invvar":
        wgt = np.zeros_like(sig, dtype=np.float32)
        np.divide(1.0, sig * sig, out=wgt, where=(sig > 0))
        den = np.sum(wgt, axis=1)
        signal_ok = np.divide(np.sum(y * wgt, axis=1), den,
                              out=np.zeros_like(den), where=(den > 0))
        noise_ok = np.sqrt(np.divide(1.0, den, out=np.zeros_like(den), where=(den > 0)))
    else:
        raise ValueError("weighting must be 'simple' or 'invvar'.")

    noise_ok = np.where((noise_ok > 0) & np.isfinite(noise_ok), noise_ok, float(noise_floor))
    snr_ok = signal_ok / noise_ok
    signal = np.zeros(W, dtype=float)
    noise = np.full(W, float(noise_floor), dtype=float)
    snr = np.zeros(W, dtype=float)
    signal[idx] = signal_ok; noise[idx] = noise_ok; snr[idx] = snr_ok

    if return_epoch_valid:
        epoch_snr = np.zeros((W, K), dtype=float)
        epoch_snr[idx, :] = epoch_snr_ok
        epoch_ok = np.zeros((W, K), dtype=bool)
        epoch_ok[idx, :] = epoch_valid
        return signal, noise, snr, epoch_snr, epoch_ok

    if return_epoch_snr:
        epoch_snr = np.zeros((W, K), dtype=float)
        epoch_snr[idx, :] = epoch_snr_ok
        return signal, noise, snr, epoch_snr

    return signal, noise, snr


def snr_multi_from_hkpq(
    theta: np.ndarray,
    instruments: Sequence[Instrument],
    *,
    noise_floor: float = 1.0,
    weighting: str = "invvar",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Multi-instrument SNR wrapper: calls `snr_from_hkpq` for each instrument
    and combines the results as if all frames were observed by one instrument.

    Combination rules (same as single-instrument):

    "simple":
        signal_tot = Σ_i signal_i       (sum of per-frame background-subtracted fluxes)
        noise_tot² = Σ_i noise_i²       (quadrature addition of per-frame noises)

    "invvar":
        den_i      = 1 / noise_i²       (inverse-variance weight for instrument i)
        signal_tot = (Σ_i den_i · signal_i) / (Σ_i den_i)
        noise_tot² = 1 / (Σ_i den_i)

    For each instrument, signal_i and noise_i are combined consistently across
    epochs for that instrument, and noise_i = 1.  The global combination
    then becomes:
        invvar: combined_snr = √(Σ_i snr_i²)
        simple: combined_snr proportional to Σ snr_i

    With a single instrument this function is numerically identical to calling
    `snr_from_hkpq` directly.
    """
    if not instruments:
        raise ValueError("snr_multi_from_hkpq requires at least one Instrument.")

    theta = np.asarray(theta, dtype=float)
    scalar_input = (theta.ndim == 1)
    if scalar_input:
        theta = theta[None, :]
    if theta.shape[1] != 7:
        raise ValueError("theta must have shape (W, 7).")

    W        = theta.shape[0]
    weighting = str(weighting).lower()

    if weighting == "simple":
        sum_signal = np.zeros(W, dtype=float)
        sum_var    = np.zeros(W, dtype=float)
    else:   # "invvar"
        sum_num = np.zeros(W, dtype=float)   # Σ_i (den_i · signal_i)
        sum_den = np.zeros(W, dtype=float)   # Σ_i den_i

    for inst in instruments:
        data_i = dict(x=inst.xgrid, bkg=inst.bkg, noise=inst.noise)

        signal_i, noise_i, _ = snr_from_hkpq(
            theta,
            ts=inst.ts,
            images=inst.images_up,
            data=data_i,
            size=inst.size,
            scale=inst.scale,
            upsampling_factor=inst.upsampling_factor,
            t_ref=inst.t_ref,
            r_mask=inst.r_mask,
            r_mask_ext=inst.r_mask_ext,
            noise_floor=noise_floor,
            weighting=weighting,
            photometry_method=inst.photometry_method,
            images_native=inst.images_native,
            fwhm=inst.fwhm,
            paco_alpha_maps=inst.paco_alpha_maps,
            paco_var_alpha_maps=inst.paco_var_alpha_maps,
            paco_interpolator=inst.paco_interpolator,
            paco_oversampling=inst.paco_oversampling,
            background_noise_cfg=inst.bgnoise_cfg,
            local_bkg_maps=inst.local_bkg_maps,
            local_noise_maps=inst.local_noise_maps,
            measurement_missing_maps=inst.measurement_missing_maps,
        )

        if weighting == "simple":
            sum_signal += signal_i
            sum_var    += noise_i * noise_i
        else:
            den_i   = np.divide(1.0, noise_i * noise_i,
                                 out=np.zeros_like(noise_i),
                                 where=(noise_i > 0) & np.isfinite(noise_i))
            sum_den += den_i
            sum_num += signal_i * den_i

    if weighting == "simple":
        signal = sum_signal
        noise  = np.sqrt(sum_var)
    else:
        signal = np.divide(sum_num, sum_den,
                           out=np.zeros_like(sum_den), where=(sum_den > 0))
        noise  = np.sqrt(np.divide(1.0, sum_den,
                                   out=np.zeros_like(sum_den), where=(sum_den > 0)))

    noise = np.where((noise > 0) & np.isfinite(noise), noise, float(noise_floor))
    snr   = signal / noise

    if scalar_input:
        return float(signal[0]), float(noise[0]), float(snr[0])
    return signal, noise, snr


def positive_snr_profile_multi_from_hkpq(
    theta: np.ndarray,
    instruments: Sequence[Instrument],
    *,
    noise_floor: float = 1.0,
    weighting: str = "invvar",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate the positive-S/N multi-epoch profile criterion.

    For each valid epoch/channel k:

        z_k = (F_k - bg_k) / sigma_k

    The positive-profile criterion uses a non-negative independent contribution per epoch:

        C(theta)     = sum_k max(0, z_k)^2
        log L(theta) = 0.5 * C(theta) + constant.

    PACO gives z_k = alpha_hat_k / sqrt(var_alpha_k). Soft-masked samples,
    out-of-map samples, and samples with invalid variance/noise have z_k = 0
    and therefore contribute exactly zero.

    ``profile_snr`` is sqrt(C).  Reference: Dallant et al. 2023, A&A 679 A38.
    ``signal`` and ``noise`` are retained for diagnostic outputs;
    they do not enter this positive-profile likelihood.
    """
    if not instruments:
        raise ValueError("positive_snr_profile_multi_from_hkpq requires at least one Instrument.")

    theta = np.asarray(theta, dtype=float)
    scalar_input = theta.ndim == 1
    if scalar_input:
        theta = theta[None, :]
    if theta.ndim != 2 or theta.shape[1] != 7:
        raise ValueError("theta must have shape (W, 7).")

    W = theta.shape[0]
    weighting = str(weighting).lower()
    if weighting not in ("simple", "invvar"):
        raise ValueError("weighting must be 'simple' or 'invvar'.")

    criterion = np.zeros(W, dtype=float)
    if weighting == "simple":
        sum_signal = np.zeros(W, dtype=float)
        sum_var = np.zeros(W, dtype=float)
    else:
        sum_num = np.zeros(W, dtype=float)
        sum_den = np.zeros(W, dtype=float)

    for inst in instruments:
        data_i = dict(x=inst.xgrid, bkg=inst.bkg, noise=inst.noise)
        signal_i, noise_i, _, epoch_snr_i = snr_from_hkpq(
            theta,
            ts=inst.ts,
            images=inst.images_up,
            data=data_i,
            size=inst.size,
            scale=inst.scale,
            upsampling_factor=inst.upsampling_factor,
            t_ref=inst.t_ref,
            r_mask=inst.r_mask,
            r_mask_ext=inst.r_mask_ext,
            noise_floor=noise_floor,
            weighting=weighting,
            photometry_method=inst.photometry_method,
            images_native=inst.images_native,
            fwhm=inst.fwhm,
            paco_alpha_maps=inst.paco_alpha_maps,
            paco_var_alpha_maps=inst.paco_var_alpha_maps,
            paco_interpolator=inst.paco_interpolator,
            paco_oversampling=inst.paco_oversampling,
            background_noise_cfg=inst.bgnoise_cfg,
            local_bkg_maps=inst.local_bkg_maps,
            local_noise_maps=inst.local_noise_maps,
            measurement_missing_maps=inst.measurement_missing_maps,
            return_epoch_snr=True,
        )

        positive = np.maximum(epoch_snr_i, 0.0)
        criterion += np.sum(positive * positive, axis=1, dtype=float)

        if weighting == "simple":
            sum_signal += signal_i
            sum_var += noise_i * noise_i
        else:
            den_i = np.divide(
                1.0,
                noise_i * noise_i,
                out=np.zeros_like(noise_i),
                where=(noise_i > 0.0) & np.isfinite(noise_i),
            )
            sum_den += den_i
            sum_num += signal_i * den_i

    if weighting == "simple":
        signal = sum_signal
        noise = np.sqrt(sum_var)
    else:
        signal = np.divide(sum_num, sum_den, out=np.zeros_like(sum_den), where=sum_den > 0.0)
        noise = np.sqrt(
            np.divide(1.0, sum_den, out=np.zeros_like(sum_den), where=sum_den > 0.0)
        )

    noise = np.where((noise > 0.0) & np.isfinite(noise), noise, float(noise_floor))
    profile_snr = np.sqrt(np.maximum(criterion, 0.0))

    if scalar_input:
        return float(signal[0]), float(noise[0]), float(profile_snr[0]), float(criterion[0])
    return signal, noise, profile_snr, criterion


def flux_sufficient_stats_from_hkpq(
    theta: np.ndarray,
    *,
    ts: np.ndarray,
    images: Optional[np.ndarray],
    data: dict,
    size: int,
    scale: float,
    upsampling_factor: float,
    t_ref: float = 0.0,
    r_mask: float | None = None,
    r_mask_ext: float | None = None,
    noise_floor: float = 1.0,
    photometry_method: str = "convolve",
    images_native: Optional[np.ndarray] = None,
    fwhm: Optional[float] = None,
    paco_alpha_maps: Optional[np.ndarray] = None,
    paco_var_alpha_maps: Optional[np.ndarray] = None,
    paco_interpolator: str = "none",
    paco_oversampling: int = 1,
    background_noise_cfg: Optional[dict] = None,
    local_bkg_maps: Optional[np.ndarray] = None,
    local_noise_maps: Optional[np.ndarray] = None,
    measurement_missing_maps: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Gaussian-flux sufficient statistics S1 and S2.

    For PACO, S1 = sum(1/var_alpha) and S2 = sum(alpha_hat/var_alpha)
    over valid, unmasked epochs. Classical backends retain their existing logic.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.ndim == 1:
        theta = theta[None, :]
    if theta.shape[1] != 7:
        raise ValueError("theta must have 7 columns: [a, la0, m0, h, k, p, q]")
    a, la0, m0, h, k, p, q = theta.T
    W = theta.shape[0]
    r2 = p*p + q*q
    ok = np.isfinite(theta).all(axis=1) & (r2 <= 1.0) & (a > 0.0) & (m0 > 0.0)
    e = np.sqrt(np.maximum(0.0, h*h + k*k)); ok &= (e < 1.0)
    S1 = np.zeros(W, dtype=float)
    S2 = np.zeros(W, dtype=float)
    if not np.any(ok):
        return S1, S2

    idx = np.where(ok)[0]
    a_ok, la0_ok, m0_ok, h_ok, k_ok, p_ok, q_ok, e_ok = (
        a[idx], la0[idx], m0[idx], h[idx], k[idx], p[idx], q[idx], e[idx]
    )
    # Direct EqOE propagation; no classical-angle reconstruction.
    ts = np.asarray(ts, dtype=float); K = int(len(ts))

    if photometry_method == "convolve":
        if images is None or images.shape[0] != K:
            raise ValueError("convolve mode requires images with first dimension K == len(ts).")
    elif photometry_method == "aperture":
        if images_native is None or fwhm is None:
            raise ValueError("aperture mode requires images_native and fwhm.")
        if images_native.shape != (K, size, size):
            raise ValueError("images_native must be (K, size, size).")
    elif photometry_method == "paco":
        if paco_alpha_maps is None or paco_var_alpha_maps is None:
            raise ValueError("paco mode requires paco_alpha_maps and paco_var_alpha_maps.")
        paco_method = _normalize_paco_interpolator(paco_interpolator)
        paco_factor = _effective_paco_oversampling(paco_method, paco_oversampling)
        paco_side = _paco_map_side(size, paco_method, paco_factor)
        if paco_alpha_maps.shape != (K, paco_side, paco_side) or paco_var_alpha_maps.shape != (K, paco_side, paco_side):
            raise ValueError(
                f"PACO alpha/variance maps must both be (K,{paco_side},{paco_side}) for "
                f"interpolator={paco_method!r}, oversampling={paco_factor}."
            )
    else:
        raise ValueError(f"Unknown photometry_method: {photometry_method!r}")

    north_sky, east_sky = _equinoctial_sky_tracks(
        a_ok, la0_ok, m0_ok, h_ok, k_ok, p_ok, q_ok, ts, t_ref=t_ref
    )
    center = (float(size)-1.0)/2.0
    x_pix = (-east_sky)*scale + center; y_pix = north_sky*scale + center
    r = np.hypot(x_pix-center, y_pix-center)
    validpix = np.ones((idx.size,K), dtype=bool)
    if r_mask is not None: validpix &= (r > r_mask)
    if r_mask_ext is not None: validpix &= (r < r_mask_ext)
    k_idx = np.broadcast_to(np.arange(K), (idx.size,K))

    if photometry_method == "paco":
        paco_method = _normalize_paco_interpolator(paco_interpolator)
        paco_factor = _effective_paco_oversampling(paco_method, paco_oversampling)
        paco_side = paco_alpha_maps.shape[1]
        if paco_method == "none":
            x_map = np.floor(x_pix + 0.5).astype(np.int32)
            y_map = np.floor(y_pix + 0.5).astype(np.int32)
        else:
            x_map = np.floor(x_pix * paco_factor + 0.5).astype(np.int32)
            y_map = np.floor(y_pix * paco_factor + 0.5).astype(np.int32)
        validpix &= ((0 <= x_map)&(x_map < paco_side)&(0 <= y_map)&(y_map < paco_side))
        xs = np.clip(x_map,0,paco_side-1); ys = np.clip(y_map,0,paco_side-1)
        alpha = paco_alpha_maps[(k_idx,ys,xs)].astype(np.float32)
        var = paco_var_alpha_maps[(k_idx,ys,xs)].astype(np.float32)
        validpix &= np.isfinite(alpha) & np.isfinite(var) & (var > 0.0)
        w = np.zeros_like(var, dtype=np.float32)
        np.divide(1.0, var, out=w, where=validpix)
        alpha = np.where(validpix, alpha, 0.0)
        S1_ok = np.sum(w, axis=1); S2_ok = np.sum(w*alpha, axis=1)
    else:
        if photometry_method == "convolve":
            x_up=_native_to_upsampled_index(x_pix, upsampling_factor).astype(np.int32)
            y_up=_native_to_upsampled_index(y_pix, upsampling_factor).astype(np.int32)
            validpix &= ((0<=x_up)&(x_up<images.shape[2])&(0<=y_up)&(y_up<images.shape[1]))
            xu=np.clip(x_up,0,images.shape[2]-1); yu=np.clip(y_up,0,images.shape[1]-1)
            flux=images[(k_idx,yu,xu)].astype(np.float32)
        else:
            validpix &= (x_pix>=0)&(x_pix<size)&(y_pix>=0)&(y_pix<size)
            flux=np.zeros((idx.size,K),dtype=np.float32); r_ap=float(fwhm)
            for ii in range(idx.size):
                for kk in range(K):
                    if validpix[ii,kk]:
                        flux[ii,kk]=_aperture_sum_native(images_native[kk],float(x_pix[ii,kk]),float(y_pix[ii,kk]),r_ap)
        flux[~validpix]=0.0
        bg_cfg=background_noise_cfg or {}
        map_mode=str(bg_cfg.get("local_map_mode","precompute_cache")).lower()
        use_maps=(map_mode=="precompute_cache" and local_bkg_maps is not None and local_noise_maps is not None)
        if use_maps:
            if measurement_missing_maps is not None:
                # Same rule (a) as snr_from_hkpq (since iteration 07).
                validpix &= ~_measurement_missing_at(
                    measurement_missing_maps,
                    photometry_method=photometry_method,
                    x_pix=x_pix, y_pix=y_pix, k_idx=k_idx,
                    upsampling_factor=upsampling_factor, fwhm=fwhm,
                    candidates=validpix,
                )
                supported, bg, sig = _bilinear_interp_valid_neighbours(
                    local_bkg_maps, local_noise_maps, x_pix, y_pix, k_idx,
                )
                validpix &= supported
            else:
                # Legacy strict rule, as in snr_from_hkpq.
                validpix &= _bilinear_support_valid(local_noise_maps, x_pix, y_pix, k_idx)
                bg=_interp_map_bilinear(local_bkg_maps,x_pix,y_pix,k_idx)
                sig=_interp_map_bilinear(local_noise_maps,x_pix,y_pix,k_idx)
            flux[~validpix] = 0.0
            bg[~validpix]=0.0
            sig[~validpix]=0.0
        else:
            bg=np.zeros((idx.size,K),dtype=np.float32)
            sig=np.full((idx.size,K),float(noise_floor),dtype=np.float32)
            for ii in range(idx.size):
                for kk in range(K):
                    if not validpix[ii,kk]:
                        continue
                    bg_loc,sig_loc,_,ok_loc=_local_aperture_ring_stats(
                        photometry_method=photometry_method,
                        x_target=float(x_pix[ii,kk]),
                        y_target=float(y_pix[ii,kk]),
                        r_target=float(r[ii,kk]),
                        size=size,
                        upsampling_factor=upsampling_factor,
                        fwhm=fwhm,
                        image_up=images[kk] if photometry_method=="convolve" else None,
                        image_native=images_native[kk] if photometry_method=="aperture" else None,
                        cfg=bg_cfg,
                    )
                    if ok_loc:
                        bg[ii,kk]=float(bg_loc)
                        sig[ii,kk]=float(sig_loc)
            bg[~validpix]=0.0
            sig[~validpix]=0.0

        valid_sigma = validpix & np.isfinite(sig) & (sig > 0.0)
        sig[valid_sigma] = np.maximum(sig[valid_sigma], float(noise_floor))
        sig[~valid_sigma] = 0.0

        ytilde=flux-bg; w=np.zeros_like(sig,dtype=np.float32); np.divide(1.0,sig*sig,out=w,where=(sig>0))
        S1_ok=np.sum(w,axis=1); S2_ok=np.sum(w*ytilde,axis=1)

    S1_ok=np.where((S1_ok>=0)&np.isfinite(S1_ok),S1_ok,0.0)
    S2_ok=np.where(np.isfinite(S2_ok),S2_ok,0.0)
    S1[idx]=S1_ok; S2[idx]=S2_ok
    return S1,S2


def flux_sufficient_stats_multi_from_hkpq(
    theta: np.ndarray,
    instruments: Sequence[Instrument],
    *,
    noise_floor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Multi-instrument wrapper for `flux_sufficient_stats_from_hkpq`.

    Since the instruments are independent:
        S1_total = Σ_i S1_i
        S2_total = Σ_i S2_i

    PACO is fully supported: its alpha_hat and var_alpha maps directly provide
    the sufficient statistics required by the same Gaussian flux likelihood.
    """
    if not instruments:
        raise ValueError("flux_sufficient_stats_multi_from_hkpq requires at least one Instrument.")


    theta = np.asarray(theta, dtype=float)
    scalar_input = (theta.ndim == 1)
    if scalar_input:
        theta = theta[None, :]
    if theta.shape[1] != 7:
        raise ValueError("theta must have shape (W, 7).")

    W = theta.shape[0]
    S1_total = np.zeros(W, dtype=float)
    S2_total = np.zeros(W, dtype=float)

    for inst in instruments:
        data_i = dict(x=inst.xgrid, bkg=inst.bkg, noise=inst.noise)
        S1_i, S2_i = flux_sufficient_stats_from_hkpq(
            theta,
            ts=inst.ts,
            images=inst.images_up,
            data=data_i,
            size=inst.size,
            scale=inst.scale,
            upsampling_factor=inst.upsampling_factor,
            t_ref=inst.t_ref,
            r_mask=inst.r_mask,
            r_mask_ext=inst.r_mask_ext,
            noise_floor=noise_floor,
            photometry_method=inst.photometry_method,
            images_native=inst.images_native,
            fwhm=inst.fwhm,
            paco_alpha_maps=inst.paco_alpha_maps,
            paco_var_alpha_maps=inst.paco_var_alpha_maps,
            paco_interpolator=inst.paco_interpolator,
            paco_oversampling=inst.paco_oversampling,
            background_noise_cfg=inst.bgnoise_cfg,
            local_bkg_maps=inst.local_bkg_maps,
            local_noise_maps=inst.local_noise_maps,
            measurement_missing_maps=inst.measurement_missing_maps,
        )
        S1_total += S1_i
        S2_total += S2_i

    S1_total = np.where((S1_total >= 0) & np.isfinite(S1_total),
                        S1_total, 0.0)
    S2_total = np.where(np.isfinite(S2_total), S2_total, 0.0)

    if scalar_input:
        return np.array([S1_total[0]]), np.array([S2_total[0]])
    return S1_total, S2_total


def _normalize_likelihood_mode(name: str) -> str:
    """
    Normalize user-facing likelihood names.

    Modes
    -----
    positive_snr_profile
        0.5 * sum_k max(0, z_k)^2.
        PACO maps only, because this is the positive per-epoch profile criterion
        inherited from the published multi-epoch PACO statistical construction.

    signed_snr
         Positive-amplitude common-SNR profile criterion:
            log L = 0.5 * max(SNR_common, 0)^2
        where SNR_common is the signed common-flux S/N returned by
        `snr_multi_from_hkpq`.

    flux
        Gaussian common-flux likelihood with the optional explicit fp parameter.
    """
    value = str(name or "positive_snr_profile").strip().lower().replace("-", "_")
    aliases = {
        "snr": "positive_snr_profile",
        "positive": "positive_snr_profile",
        "positive_snr": "positive_snr_profile",
        "snr_signed": "signed_snr",
        "common_snr": "signed_snr",
    }
    value = aliases.get(value, value)
    allowed = {"positive_snr_profile", "signed_snr", "flux"}
    if value not in allowed:
        raise ValueError(
            "likelihood_mode must be one of: positive_snr_profile, signed_snr, flux."
        )
    return value




def instrument_log_likelihood(
    theta: np.ndarray,
    instrument: Instrument,
    *,
    noise_floor: float,
    weighting: str,
    likelihood_mode: str | None = None,
    fp_bounds: tuple[float, float] = (1e-4, 1e2),
) -> np.ndarray | float:
    """Evaluate one instrument's relative log likelihood for a common orbital track.

    The photometric kernels, masks, sampling interpolation and noise estimates
    are exactly those used throughout this module. Only the statistical
    aggregation is performed here.

    Parameters
    ----------
    theta:
        Seven orbital coordinates, optionally followed by one sampled amplitude
        for each independently calibrated flux-mode instrument.
    instrument:
        Calibrated observation maps and epochs from one data source.
    likelihood_mode:
        `signed_snr`, `positive_snr_profile` (PACO only), or `flux`.
    fp_bounds:
        Included for API compatibility. Each flux-mode Instrument holds its
        own bounds, sampled column index and fixed amplitude.

    Returns
    -------
    Log likelihood for every supplied orbit, without applying orbital priors.
    """
    samples = np.asarray(theta, dtype=float)
    scalar = samples.ndim == 1
    if scalar:
        samples = samples[None, :]
    if samples.ndim != 2 or samples.shape[1] < 7:
        raise ValueError("theta must include the seven orbital parameters")
    mode = _normalize_likelihood_mode(likelihood_mode or instrument.likelihood_mode)
    noise_floor = float(instrument.noise_floor if instrument.noise_floor is not None else noise_floor)
    if mode == "positive_snr_profile" and instrument.photometry_method != "paco":
        raise ValueError(f"{instrument.name}: positive_snr_profile requires PACO maps")
    orbit_samples = samples[:, :7]

    if mode == "positive_snr_profile":
        _, _, _, criterion = positive_snr_profile_multi_from_hkpq(
            orbit_samples, [instrument], noise_floor=noise_floor, weighting=weighting,
        )
        result = 0.5 * np.asarray(criterion, dtype=float)
    elif mode == "signed_snr":
        _, _, signed_snr = snr_multi_from_hkpq(
            orbit_samples, [instrument], noise_floor=noise_floor, weighting=weighting,
        )
        # A non-negative source amplitude has profile MLE max(SNR, 0).
        # Negative combined SNR is not a planet detection: log LR = 0.
        # Clip AFTER combining epochs of this instrument, not per epoch.
        result = 0.5 * np.maximum(np.asarray(signed_snr, dtype=float), 0.0) ** 2
    else:
        s1, s2 = flux_sufficient_stats_multi_from_hkpq(
            orbit_samples, [instrument], noise_floor=noise_floor,
        )
        planet_flux = instrument_flux_values(samples, instrument)
        result = -0.5 * (planet_flux * planet_flux * s1 - 2.0 * planet_flux * s2)
    return float(result[0]) if scalar else result


def total_direct_imaging_log_likelihood(
    theta: np.ndarray,
    *,
    instruments: Sequence[Instrument],
    noise_floor: float = 1.0,
    weighting: str = "invvar",
    likelihood_mode: str = "signed_snr",
    fp_bounds: tuple[float, float] = (1e-4, 1e2),
) -> np.ndarray | float:
    """Sum conditionally independent direct-imaging log likelihoods.

    Each instrument may choose its own statistical criterion. Sampled flux
    parameters are independent, with each value assigned to its instrument.
    This function does not apply priors, allowing the same objective to be used
    by grid initialisation, MCMC and parallel tempering. A different numerical
    noise floor and sampled/fixed flux can be assigned to each instrument.
    """
    if not instruments:
        raise ValueError("At least one direct-imaging instrument is required")
    values = np.asarray(theta, dtype=float)
    scalar = values.ndim == 1
    if scalar:
        values = values[None, :]
    result = np.zeros(values.shape[0], dtype=float)
    for instrument in instruments:
        result += instrument_log_likelihood(
            values, instrument, noise_floor=noise_floor,
            weighting=weighting,
            likelihood_mode=(instrument.likelihood_mode or likelihood_mode),
            fp_bounds=fp_bounds,
        )
    return float(result[0]) if scalar else result


def uses_flux_likelihood(instruments: Sequence[Instrument]) -> bool:
    """Whether at least one data source requires the Gaussian flux parameter."""
    return any(_normalize_likelihood_mode(i.likelihood_mode) == "flux" for i in instruments)
