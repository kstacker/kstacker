# mcmc.py
# =============================================================================
# Keplerian orbit inference for multi-epoch direct-imaging data.
#
# Data backends
# -------------
# convolve
#     Uses reduced images convolved with a circular photometric kernel.
#
# aperture
#     Uses circular-aperture photometry on reduced native images.
#
# paco
#     Uses PACO alpha_hat and var_alpha maps directly.
#
# Likelihoods
# -----------
# positive_snr_profile   PACO only
#     log L = 0.5 * sum_k max(0, z_k)^2
#
# signed_snr             PACO / convolve / aperture
#     log L = 0.5 * SNR_common^2
#
# flux                   PACO / convolve / aperture
#     Gaussian common-flux likelihood.
#
# Sampling
# --------
# emcee
#     Affine-invariant ensemble MCMC.
#
# reddemcee
#     Adaptive parallel-tempering ensemble MCMC.
#
# Dataset metadata
# ----------------
# Epoch count, native image size, observation times, and the convolve
# upsampling factor are discovered from the files whenever possible. The YAML
# contains analysis choices rather than duplicated dataset bookkeeping.
#
# Orbital coordinates
# -------------------
# theta = (a, lambda0, m0, h, k, p, q)
#
#   h = e * sin(Omega + omega)
#   k = e * cos(Omega + omega)
#   p = sin(i/2) * sin(Omega)
#   q = sin(i/2) * cos(Omega)
#
# References
# ----------
# Flasseur et al. 2020, A&A 637, A9          (PACO ASDI)
# Dallant et al. 2023, A&A 679, A38          (multi-epoch positive-SNR profile)
# Gomez Gonzalez et al. 2017, arXiv:1705.06184 (VIP)
# Pena R. & Jenkins 2026, arXiv:2509.24870   (reddemcee)
# =============================================================================

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
import emcee
from concurrent.futures import ProcessPoolExecutor
import kepler          # must expose kepler.solve(M, e) with broadcasting
from scipy.ndimage import map_coordinates, distance_transform_edt
from scipy.signal import convolve2d
from scipy.stats import t as scipy_student_t, theilslopes
from astropy.io import fits
from astropy.nddata import block_replicate
from astropy.time import Time
from photutils.aperture import CircularAperture, aperture_photometry

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        return iterable



def _terminal_colors_enabled() -> bool:
    """Enable ANSI colors on interactive terminals; KSTACKER_COLOR can override."""
    mode = str(os.environ.get("KSTACKER_COLOR", "auto") or "auto").strip().lower()
    if mode in {"never", "0", "false", "no"}:
        return False
    if mode in {"always", "1", "true", "yes"}:
        return True
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


def _color(text: str, code: str) -> str:
    if not _terminal_colors_enabled():
        return text
    return f"\033[{code}m{text}\033[0m"


def _print_section(title: str, width: int = 110) -> None:
    """Print a readable console section heading."""
    print()
    print(_color("=" * width, "36"))
    print(_color(title, "1;36"))
    print(_color("=" * width, "36"))
    print()

# Project helpers — try relative import first (package mode), fall back to
# absolute import when the file is run standalone.
try:
    from .orbit import orbit      # must expose orbit.positions_at_multiple_times
    from .utils import Params     # YAML reader + grid + path helpers
except Exception:
    from orbit import orbit
    from utils import Params


# =============================================================================
# INSTRUMENT DATACLASS
# =============================================================================

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
    their contributions.  With a single Instrument the result is numerically
    identical to the original single-instrument code.
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


# =============================================================================
# YAML HELPERS
# =============================================================================

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




# =============================================================================
# NOISE FLOOR
# =============================================================================

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
    """Collect finite positive sigma samples from all active instruments."""
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
    Resolve the minimum allowed sigma used by all likelihoods.

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

# =============================================================================
# DATASET DISCOVERY
# =============================================================================
#
# The YAML describes analysis choices. Dataset facts are read from the files:
# number of epochs, image size, observation times, and the convolution
# upsampling factor. This keeps the configuration short and prevents duplicated
# metadata from drifting out of sync with the data on disk.
# =============================================================================

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



# =============================================================================
# CLASSICAL IMAGE PREPARATION
# =============================================================================

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


# =============================================================================
# ROBUST OBSERVATION-TIME DISCOVERY
# =============================================================================

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



def _resolve_weighting(yaml_root: dict) -> str:
    """Return the epoch-combination weighting: ``invvar`` or ``simple``."""
    value = str(yaml_root.get("weighting", "invvar") or "invvar").strip().lower()
    if value not in {"invvar", "simple"}:
        raise ValueError("`weighting` must be 'invvar' or 'simple'.")
    return value


def _resolve_bounds(
    params: Params, priors: dict
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Read the required semi-major-axis and stellar-mass prior bounds."""
    a_bounds = _get(priors, "a_bounds", None)
    m0_bounds = _get(priors, "m0_bounds", None)
    if a_bounds is None or m0_bounds is None:
        raise ValueError("`priors.a_bounds` and `priors.m0_bounds` are required.")
    return tuple(map(float, a_bounds)), tuple(map(float, m0_bounds))


def _resolve_tref(params: Params, root: dict, ts: np.ndarray) -> float:
    """
    Determine the reference epoch t_ref used in the definition of λ0.

    YAML keys
    ─────────
    t_ref_mode : "min_ts"  → use the earliest observation time across all
                             instruments (default, recommended).
                 "fixed"   → use the numeric value in `t_ref`.
    t_ref      : float, required when t_ref_mode = "fixed".

    Using "min_ts" ensures that λ0 is the mean longitude at the first
    observation, which keeps the prior on λ0 ∈ [0, 2π] well-calibrated.
    """
    mode = str(root.get("t_ref_mode", "min_ts") or "min_ts").lower()
    if mode == "fixed":
        tref = root.get("t_ref", None)
        if tref is None:
            raise ValueError("t_ref_mode='fixed' but no 't_ref' provided in YAML.")
        return float(tref)
    # "min_ts": reference epoch = earliest observation time over all instruments.
    return float(np.min(ts))



def _resolve_init_mode(root: dict) -> str:
    """
    Select the MCMC walker source.

    ``init_search`` uses the ranked explicit physical grid.
    ``manual`` uses the optional ``init`` vector in the YAML.
    """
    mode = str(root.get("init_mode", "init_search") or "init_search").lower()
    if mode not in ("manual", "init_search"):
        raise ValueError("init_mode must be 'init_search' or 'manual'.")
    return mode


def _resolve_init_vector(
    root: dict, priors: dict, m0_bounds: Tuple[float, float]
) -> np.ndarray:
    """
    Build the 7-D initial guess from the YAML `init` section.

    Each component has a sensible default so the YAML can omit fields.
    The default for a is the midpoint of `a_bounds`, and the default for
    m0 is the lower bound of `m0_bounds`.
    """
    init = root.get("init", {}) or {}
    a_init   = float(init.get("a_init",   sum(_get(priors, "a_bounds", (58.0, 58.0))) / 2.0))
    la0_init = float(init.get("la0_init", 0.0))
    m0_init  = float(init.get("m0_init",  m0_bounds[0]))
    h_init   = float(init.get("h_init",   0.0))
    k_init   = float(init.get("k_init",   0.0))
    p_init   = float(init.get("p_init",   0.0))
    q_init   = float(init.get("q_init",   0.0))
    return np.array([a_init, la0_init, m0_init, h_init, k_init, p_init, q_init], dtype=float)


def _resolve_spread(root: dict) -> dict:
    """
    Read per-parameter Gaussian jitter scales used to build the initial walker
    cloud around theta_init.

    Each walker is drawn as:
        a   ~ a_init   × (1 + N(0, spread['a']))
        la0 ~ wrap_2π(la0_init + N(0, spread['la0']))
        m0  ~ clip(m0_init × (1 + N(0, spread['m0'])), *m0_bounds)   if spread > 0
        h   ~ h_init + N(0, spread['hk'])
        k   ~ k_init + N(0, spread['hk'])
        p   ~ p_init + N(0, spread['pq'])
        q   ~ q_init + N(0, spread['pq'])

    The spreads control the initial walker cloud around each selected centre.
    """
    spread = root.get("init_spread", {}) or {}
    return dict(
        a=float(spread.get("a",   0.02)),
        la0=float(spread.get("la0", 0.2)),
        m0=float(spread.get("m0",  0.02)),
        hk=float(spread.get("hk",  0.02)),
        pq=float(spread.get("pq",  0.02)),
    )


def _resolve_mcmc(root: dict) -> dict:
    """
    Parse sampler-specific MCMC settings.

    The YAML keeps emcee and adaptive parallel tempering settings separate so
    the number of walkers and the number of steps are explicit for the selected
    sampler.
    """
    cfg = root.get("mcmc", {}) or {}
    sampler = str(cfg.get("sampler", "emcee") or "emcee").strip().lower()

    if sampler in {"apt", "pt", "adaptive_parallel_tempering"}:
        sampler = "reddemcee"

    if sampler not in {"emcee", "reddemcee"}:
        raise ValueError("mcmc.sampler must be 'emcee' or 'reddemcee'.")

    common = dict(
        sampler=sampler,
        likelihood_mode=_normalize_likelihood_mode(
            cfg.get("likelihood_mode", "positive_snr_profile")
        ),
        sample_fp=bool(cfg.get("sample_fp", False)),
        fp_bounds=tuple(map(float, cfg.get("fp_bounds", (1e-4, 1e2)))),
        fp_prior=str(cfg.get("fp_prior", "uniform")).lower(),
        fp_init=float(cfg.get("fp_init", 1.0)),
        fp_spread=float(_get(root.get("init_spread", {}), "fp", 0.2)),
        progress=_bool_from_config(cfg.get("progress", True), True),
    )

    if sampler == "emcee":
        ecfg = cfg.get("emcee", {}) or {}
        common.update(
            nwalkers=max(2, int(ecfg.get("walkers", 100))),
            burnin=max(0, int(ecfg.get("burnin_steps", 10000))),
            nsteps=max(1, int(ecfg.get("production_steps", 100000))),
            thin=max(1, int(ecfg.get("thin", 10))),
        )
        common["reddemcee"] = None
        return common

    rcfg = cfg.get("reddemcee", {}) or {}
    betas = rcfg.get("betas", None)
    if betas is not None:
        betas = tuple(float(x) for x in betas)

    common.update(
        nwalkers=max(2, int(rcfg.get("walkers", 128))),
        burnin=max(0, int(rcfg.get("burnin_sweeps", 5000))),
        nsteps=max(1, int(rcfg.get("production_sweeps", 20000))),
        thin=max(1, int(rcfg.get("thin", 20))),
    )

    common["reddemcee"] = dict(
        ntemps=max(2, int(rcfg.get("temperatures", 12))),
        beta_min=float(rcfg.get("beta_min", 1.0e-4)),
        betas=betas,
        inner_steps=max(1, int(rcfg.get("inner_steps", 1))),
        adapt_mode=str(rcfg.get("adapt_mode", "SAR") or "SAR").upper(),
        adapt_tau=float(rcfg.get("adapt_tau", 700.0)),
        adapt_nu=float(rcfg.get("adapt_nu", 0.30)),
        stretch_a=float(rcfg.get("stretch_a", 1.5)),
        smd_history=bool(rcfg.get("smd_history", True)),
        tsw_history=bool(rcfg.get("tsw_history", True)),
    )
    return common


def _resolve_parallel(root: dict) -> dict:
    """
    Parse the `parallel` section of the YAML.

    max_workers : number of worker processes used for log-posterior evaluation.
                  None / "auto" → use the CPU affinity visible to the job.
                  1             → fully sequential vectorized evaluation.

    chunk_size  : number of walkers evaluated per worker task.
                  "auto" / None / 0 is recommended for most runs: the code
                  splits each emcee batch into roughly one chunk per worker,
                  which keeps every allocated CPU busy while avoiding many tiny
                  subprocess round-trips.

                  A positive integer is still accepted for expert tuning.
                  Smaller values create more tasks; larger values create fewer
                  tasks.  For small walker ensembles, too many tiny tasks can be
                  slower than a single vectorized evaluation.
    """
    par = root.get("parallel", {}) or {}

    max_workers = par.get("max_workers", None)
    if isinstance(max_workers, str) and max_workers.strip().lower() in ("none", "null", "auto"):
        max_workers = None

    raw_chunk_size = par.get("chunk_size", "auto")
    if raw_chunk_size is None:
        chunk_size = None
    elif isinstance(raw_chunk_size, str) and raw_chunk_size.strip().lower() in ("auto", "none", "null"):
        chunk_size = None
    else:
        chunk_size = int(raw_chunk_size)
        if chunk_size <= 0:
            chunk_size = None

    return dict(
        max_workers=None if max_workers is None else int(max_workers),
        chunk_size=chunk_size,
    )


# =============================================================================
# BACKGROUND / NOISE CONFIGURATION
# =============================================================================

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

    out = dict(
        mode="local_aperture_ring",
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


# =============================================================================
# ORBITAL / ANGLE HELPERS
# =============================================================================

def wrap_2pi(x: np.ndarray | float) -> np.ndarray | float:
    """
    Wrap angles to the half-open interval [0, 2π).

    Used to keep mean longitude, mean anomaly, and argument of periapsis in
    canonical range after arithmetic operations.
    """
    return np.mod(x + 2.0 * np.pi, 2.0 * np.pi)


def mean_motion(a: np.ndarray, m0: np.ndarray) -> np.ndarray:
    """
    Keplerian mean motion: n = 2π √(m0 / a³)  [rad / year].

    With a in AU and m0 in solar masses, and using G·M_sun = 4π² AU³/yr²,
    this gives the mean motion in radians per year.
    """
    return 2.0 * np.pi * np.sqrt(m0 / (a ** 3))


def lambda_to_M0(la0: np.ndarray, h: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    Convert mean longitude at the reference epoch to a gauge-chosen mean anomaly.

    This helper is retained only for human-readable derived classical outputs.
    The MCMC likelihood itself never uses this conversion.

    For e>0, varpi = Ω+ω = atan2(h,k) and M0 = λ0-varpi.
    At e=0, varpi is undefined; for reporting we choose varpi=0, so M0=λ0.
    """
    la0 = np.asarray(la0, dtype=float)
    h = np.asarray(h, dtype=float)
    k = np.asarray(k, dtype=float)
    e = np.hypot(h, k)
    varpi = np.where(e > 0.0, np.arctan2(h, k), 0.0)
    return wrap_2pi(la0 - varpi)


def M0_to_t0(
    M0: np.ndarray, a: np.ndarray, m0: np.ndarray, t_ref: float
) -> np.ndarray:
    """Convert a gauge-chosen M0 to the corresponding reported t0."""
    n = mean_motion(a, m0)
    return t_ref - (M0 / n)


def solve_equinoctial_kepler(
    lambda_value: np.ndarray,
    h: np.ndarray,
    k: np.ndarray,
    *,
    tolerance: float = 1e-13,
    max_iterations: int = 20,
) -> np.ndarray:
    """
    Solve the equinoctial Kepler equation

        λ = F + h cos(F) - k sin(F)

    where F = E + Ω + ω is the eccentric longitude.  The solver is fully
    vectorized over walkers and epochs.  At e=0, h=k=0 and F=λ exactly.
    """
    lam = np.asarray(lambda_value, dtype=float)
    h = np.asarray(h, dtype=float)
    k = np.asarray(k, dtype=float)
    F = np.array(lam, dtype=float, copy=True)
    for _ in range(int(max_iterations)):
        sF = np.sin(F)
        cF = np.cos(F)
        residual = F + h * cF - k * sF - lam
        derivative = 1.0 - h * sF - k * cF
        step = residual / derivative
        F -= step
        if np.all(np.abs(step) < float(tolerance)):
            break
    return F


def equinoctial_xy_from_hk(
    a: np.ndarray,
    lambda_value: np.ndarray,
    h: np.ndarray,
    k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Direct orbital-plane coordinates from (a, λ, h, k), without reconstructing
    Ω, ω, M, E, or t0.

    With e²=h²+k², η=sqrt(1-e²), β=1/(1+η):

        X/a = (1-βh²)cosF + βhk sinF - k
        Y/a = βhk cosF + (1-βk²)sinF - h
    """
    a = np.asarray(a, dtype=float)
    h = np.asarray(h, dtype=float)
    k = np.asarray(k, dtype=float)
    e2 = h*h + k*k
    eta = np.sqrt(np.maximum(0.0, 1.0 - e2))
    beta = 1.0 / (1.0 + eta)
    F = solve_equinoctial_kepler(lambda_value, h, k)
    cF = np.cos(F)
    sF = np.sin(F)
    X = a * ((1.0 - beta*h*h) * cF + beta*h*k * sF - k)
    Y = a * (beta*h*k * cF + (1.0 - beta*k*k) * sF - h)
    return X, Y


def project_equinoctial_sin_i2(
    X: np.ndarray,
    Y: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Direct sky projection for the bounded inclination variables

        p = sin(i/2) sin(Ω)
        q = sin(i/2) cos(Ω).

    In the project's (North, East) convention:

        North = (1-2p²) X + 2pq Y
        East  = -2pq X - (1-2q²) Y

    No atan2, angular reconstruction, division, or square root is required.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    north = (1.0 - 2.0*p*p) * X + 2.0*p*q * Y
    east = -2.0*p*q * X - (1.0 - 2.0*q*q) * Y
    return north, east


def _equinoctial_sky_tracks(
    a: np.ndarray,
    la0: np.ndarray,
    m0: np.ndarray,
    h: np.ndarray,
    k: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    ts: np.ndarray,
    *,
    t_ref: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized EqOE propagation to North/East sky coordinates."""
    a = np.asarray(a, dtype=float)
    la0 = np.asarray(la0, dtype=float)
    m0 = np.asarray(m0, dtype=float)
    h = np.asarray(h, dtype=float)
    k = np.asarray(k, dtype=float)
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    ts = np.asarray(ts, dtype=float)

    n = mean_motion(a, m0)
    lam = la0[..., None] + n[..., None] * (ts - float(t_ref))
    X, Y = equinoctial_xy_from_hk(
        a[..., None], lam, h[..., None], k[..., None]
    )
    north, east = project_equinoctial_sin_i2(
        X, Y, p[..., None], q[..., None]
    )
    return north, east


def compute_projection_matrices_from_hkpq(
    k: np.ndarray, h: np.ndarray, p: np.ndarray, q: np.ndarray
) -> np.ndarray:
    """
    Compatibility helper returning the direct EqOE sky projection matrix for
    the NEW bounded inclination variables p=sin(i/2)sinΩ, q=sin(i/2)cosΩ.

    The matrix acts on the equinoctial (X,Y) basis, not the 
    periapsis-aligned classical (x_orb,y_orb) basis.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    r00 = 1.0 - 2.0*p*p
    r01 = 2.0*p*q
    r10 = -2.0*p*q
    r11 = -(1.0 - 2.0*q*q)
    rot = np.stack([
        np.stack([r00, r01], axis=-1),
        np.stack([r10, r11], axis=-1),
    ], axis=-2)
    return rot.astype(np.float32, copy=False)


# =============================================================================
# NATIVE IMAGE LOADERS
# =============================================================================

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



# =============================================================================
# RADIAL NOISE RECALIBRATION
# =============================================================================
#
# `radial_noise_recalibration` estimates a radial correction factor g(d):
#
#       z_raw(d) = signal(d) / sigma(d)
#       g(d)      = robust empirical scatter of z_raw in a radial bin
#       sigma_cal(d) = g(d) * sigma(d)
#
# and therefore, for a variance map:
#
#       var_cal(d) = g(d)^2 * var(d)
#
# We deliberately impose g >= 1 by default: this feature is a conservative
# noise/variance inflation, never a variance deflation.
#
# PACO:
#   signal = alpha_hat, sigma = sqrt(var_alpha).  Recalibration is performed on
#   native maps BEFORE optional PACO spatial interpolation.
#
# convolve/aperture:
#   signal = photometric measurement - estimated background and
#   sigma = the already-estimated local/radial noise.  Recalibration therefore
#   happens AFTER background/noise estimation and leaves the background and
#   measured signal untouched.
#
# Short references:
#   Flasseur et al. 2020, A&A 637 A9 (PACO ASDI)
#   Dallant et al. 2023, A&A 679 A38 (multi-epoch positive-S/N criterion)
# =============================================================================

_RADIAL_RECAL_MAD_TO_SIGMA = 1.482602218505602


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

    Instrument values override the global `radial_noise_recalibration` block.
    """
    global_cfg = dict((root.get("radial_noise_recalibration", {}) or {}))
    inst_cfg = inst_cfg or {}
    local_cfg = dict((inst_cfg.get("radial_noise_recalibration", {}) or {}))
    cfg = {**global_cfg, **local_cfg}


    mode = _normalize_radial_noise_recalibration_mode(cfg.get("mode", "none"))
    return dict(
        mode=mode,
        bin_width_px=float(cfg.get("bin_width_px", 2.0)),
        max_distance_px=float(cfg.get("max_distance_px", 30.0)),
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
    """Stable filesystem token for recalibration products."""
    bw = f"{float(cfg['bin_width_px']):g}".replace(".", "p")
    md = f"{float(cfg['max_distance_px']):g}".replace(".", "p")
    return f"bin{bw}px_max{md}px"


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


#  helper names kept only for third-party notebooks importing them.
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
        recal_cfg = _resolve_radial_noise_recalibration({}, {})
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
        recal_root = os.path.join(
            paco_dir,
            "radial_noise_recalibration",
            _radial_recalibration_cache_token(recal_cfg),
            mode,
        )
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
        else:
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


# =============================================================================
# APERTURE PHOTOMETRY HELPER
# =============================================================================

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


# =============================================================================
# LOCAL BACKGROUND / NOISE HELPERS
# =============================================================================

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
    x_up = int(np.floor(float(x_native) * float(upsampling_factor) - 0.5))
    y_up = int(np.floor(float(y_native) * float(upsampling_factor) - 0.5))
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
    return "__".join([
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
    ])


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
) -> tuple[np.ndarray, np.ndarray]:
    """Compute local-aperture-ring background and noise maps for one epoch."""
    bg_map = np.zeros((size, size), dtype=np.float32)
    noise_map = np.full((size, size), float(noise_floor), dtype=np.float32)

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
            )

            if ok_loc:
                bg_map[y, x] = float(bg_loc)
                noise_map[y, x] = float(sig_loc)
            else:
                bg_map[y, x] = 0.0
                noise_map[y, x] = float(noise_floor)

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
) -> tuple[np.ndarray, np.ndarray]:
    """Load cached local-ring maps or compute and cache missing epoch pairs."""
    if profile_dir is None:
        raise ValueError("profile_dir is required to cache local aperture-ring maps.")

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

    bg_maps = []
    noise_maps = []

    print()
    print(f"[bgnoise] cache stem : {stem}")
    print(f"[bgnoise] cache dir  : {profile_dir}")
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

        if (
            not overwrite
            and os.path.exists(bg_path)
            and os.path.exists(noise_path)
        ):
            bg_map = np.load(bg_path).astype(cache_dtype, copy=False)
            noise_map = np.load(noise_path).astype(cache_dtype, copy=False)
        else:
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
            )

            bg_map = bg_map.astype(cache_dtype, copy=False)
            noise_map = noise_map.astype(cache_dtype, copy=False)

            np.save(bg_path, bg_map)
            np.save(noise_path, noise_map)

        bg_maps.append(bg_map)
        noise_maps.append(noise_map)

    return (
        np.asarray(bg_maps, dtype=cache_dtype),
        np.asarray(noise_maps, dtype=cache_dtype),
    )



# =============================================================================
# CLASSICAL (CONVOLVE / APERTURE) RADIAL NOISE RECALIBRATION
# =============================================================================

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
        ix = np.floor(xx * factor - 0.5).astype(int)
        iy = np.floor(yy * factor - 0.5).astype(int)
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
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Recalibrate local-aperture-ring noise with a robust radial factor g(d).

    The background map is left unchanged. Only sigma is rescaled:

        z_raw = (measurement - background) / sigma_raw
        sigma_cal(d) = g(d) * sigma_raw(d)

    The recalibration is applied to the cached 2-D local-noise maps used by
    convolve/aperture likelihood evaluation.
    """
    mode = _normalize_radial_noise_recalibration_mode(cfg.get("mode", "none"))
    if mode == "none":
        return xgrid, bkg, noise, local_noise_maps

    if local_bkg_maps is None or local_noise_maps is None:
        raise RuntimeError(
            "Radial noise recalibration requires local aperture-ring background "
            "and noise maps. Enable background_noise.precompute_maps."
        )

    cache_root = os.path.join(
        str(profile_dir),
        "radial_noise_recalibration",
        _radial_recalibration_cache_token(cfg),
        mode,
        str(photometry_method).lower(),
    )
    profile_path = os.path.join(cache_root, "g_profile.npz")
    overwrite = bool(cfg.get("overwrite_cache", False))
    use_cache = bool(cfg.get("cache", True))

    cached_noise_paths = [
        os.path.join(cache_root, f"noise_recalibrated_epoch_{k}.npy")
        for k in range(np.asarray(local_noise_maps).shape[0])
    ]

    print()
    print("=" * 110)
    print(f"RADIAL NOISE RECALIBRATION — {str(photometry_method).upper()}")
    print("=" * 110)
    print()
    print(f"  mode                 : {mode}")
    print("  definition           : sigma_cal(d) = g(d) * sigma_raw(d)")
    print("  standardized field   : z_raw = (measurement - background) / sigma_raw")
    print(f"  radial bin           : {float(cfg['bin_width_px']):g} native px")
    print(f"  calibrated edge zone : 0 .. {float(cfg['max_distance_px']):g} native px")
    print("  background           : unchanged")
    print()

    cache_ready = (
        use_cache
        and not overwrite
        and os.path.isfile(profile_path)
        and all(os.path.isfile(path) for path in cached_noise_paths)
    )

    if cache_ready:
        print("  cache                : HIT")
        loaded = np.asarray(
            [np.load(path) for path in cached_noise_paths],
            dtype=float,
        )
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


# =============================================================================
# SNR CORE  (vectorized over walkers)
# =============================================================================

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
):
    """Compute per-walker combined signal, noise, and SNR across all epochs.

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
        x_up = np.floor(x_pix * upsampling_factor - 0.5).astype(np.int32)
        y_up = np.floor(y_pix * upsampling_factor - 0.5).astype(np.int32)
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
    ``signal`` and ``noise`` are retained for backward-compatible diagnostics;
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



# =============================================================================
# FLUX SUFFICIENT STATISTICS  (for "flux" likelihood only)
# =============================================================================

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
            x_up=np.floor(x_pix*upsampling_factor-0.5).astype(np.int32)
            y_up=np.floor(y_pix*upsampling_factor-0.5).astype(np.int32)
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
            bg=_interp_map_bilinear(local_bkg_maps,x_pix,y_pix,k_idx)
            sig=_interp_map_bilinear(local_noise_maps,x_pix,y_pix,k_idx)
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
        )
        S1_total += S1_i
        S2_total += S2_i

    S1_total = np.where((S1_total >= 0) & np.isfinite(S1_total),
                        S1_total, 0.0)
    S2_total = np.where(np.isfinite(S2_total), S2_total, 0.0)

    if scalar_input:
        return np.array([S1_total[0]]), np.array([S2_total[0]])
    return S1_total, S2_total


def _normalise_orbit_direction(value: str | None) -> str:
    """Normalize the optional orbital-direction prior selector."""
    direction = str(value or "any").strip().lower()
    if direction not in ("any", "prograde", "retrograde"):
        raise ValueError("orbit_direction must be 'any', 'prograde', or 'retrograde'.")
    return direction


def _direction_mask_from_pq(p: np.ndarray, q: np.ndarray, orbit_direction: str = "any") -> np.ndarray:
    """Return the direction-support mask for p=sin(i/2)sinΩ, q=sin(i/2)cosΩ."""
    direction = _normalise_orbit_direction(orbit_direction)
    r2 = np.asarray(p, dtype=float)**2 + np.asarray(q, dtype=float)**2
    if direction == "prograde":
        return r2 <= 0.5
    if direction == "retrograde":
        return r2 >= 0.5
    return np.ones_like(r2, dtype=bool)


# =============================================================================
# PRIORS
# =============================================================================

def log_prior_hkpq(
    theta: np.ndarray,
    *,
    a_bounds: Tuple[float, float],
    m0_bounds: Tuple[float, float],
    la0_bounds: Tuple[float, float] = (0.0, 2.0 * np.pi),
    e_max: float = 0.95,
    orbit_direction: str = "any",
) -> np.ndarray:
    """
    Vectorized prior for theta=(a, lambda0, m0, h, k, p, q).

    The physical priors are fixed to be uniform in eccentricity e and
    inclination i.  No alternative eccentricity or inclination prior is
    supported.

    With
        h = e sin(Omega + omega),
        k = e cos(Omega + omega),
    a uniform prior in e requires p(h,k) proportional to 1/e.

    With
        p = sin(i/2) sin(Omega),
        q = sin(i/2) cos(Omega),
    and r = sqrt(p^2+q^2) = sin(i/2), a uniform prior in i requires
    p(p,q) proportional to 1 / (r sqrt(1-r^2)).

    Optional orbit direction
    ------------------------
        "any"        : 0 <= i <= pi
        "prograde"   : 0 <= i <= pi/2
        "retrograde" : pi/2 <= i <= pi

    The directional cases are the same uniform-in-i prior conditioned on the
    selected half of the inclination interval.
    """
    theta = np.asarray(theta, dtype=float)
    scalar_input = (theta.ndim == 1)
    if scalar_input:
        theta = theta[None, :]
    if theta.shape[1] != 7:
        raise ValueError("theta must have 7 columns: [a, la0, m0, h, k, p, q]")

    a, la0, m0, h, k, p, q = theta.T
    W = theta.shape[0]
    valid = (
        np.isfinite(theta).all(axis=1) &
        (a_bounds[0] <= a) & (a <= a_bounds[1]) &
        (m0_bounds[0] <= m0) & (m0 <= m0_bounds[1]) &
        (la0_bounds[0] <= la0) & (la0 <= la0_bounds[1]) &
        (a > 0.0) & (m0 > 0.0)
    )

    e = np.hypot(h, k)
    valid &= (e <= e_max)

    r2 = p*p + q*q
    valid &= (r2 <= 1.0)
    direction = _normalise_orbit_direction(orbit_direction)
    if direction == "prograde":
        valid &= (r2 <= 0.5)
    elif direction == "retrograde":
        valid &= (r2 >= 0.5)

    # Jacobian for a prior uniform in e and in the eccentricity angle.
    eps = 1.0e-12
    e_safe = np.maximum(e, eps)
    logp_hk = -np.log(e_max) - np.log(2.0 * np.pi) - np.log(e_safe)

    # Jacobian for a prior uniform in i and Omega.
    # For orbit_direction="any", the normalized density is
    # 1 / (pi^2 r sqrt(1-r^2)).  Conditioning on one hemisphere doubles it.
    r_safe = np.maximum(np.sqrt(np.maximum(r2, 0.0)), eps)
    one_minus_r2_safe = np.maximum(1.0 - r2, eps)
    logp_pq = (
        -2.0 * np.log(np.pi)
        -np.log(r_safe)
        -0.5 * np.log(one_minus_r2_safe)
    )
    if direction != "any":
        logp_pq += np.log(2.0)

    logp = logp_hk + logp_pq
    logp = np.where(valid, logp, -np.inf)
    if scalar_input:
        return float(logp[0])
    return logp


def log_prior_fp(
    fp: np.ndarray | float,
    bounds: tuple[float, float],
    prior: str = "uniform",
) -> np.ndarray | float:
    """
    Log-prior for the planet flux parameter fp.

    "uniform":
        Flat prior over [fp_min, fp_max].
        log p = −log(fp_max − fp_min).

    "loguniform":
        Jeffreys-like prior: p(fp) ∝ 1/fp over [fp_min, fp_max].
        log p = −log(log(fp_max/fp_min)) − log(fp).
        Recommended when the dynamic range of fp is large and you have no
        strong prior on the planet-to-star flux ratio.
        Requires 0 < fp_min < fp_max.
    """
    fp = np.asarray(fp)
    lo, hi = map(float, bounds)
    inside = (fp >= lo) & (fp <= hi)

    if prior == "uniform":
        Z       = hi - lo
        logp_val = -np.log(Z) if hi > lo else -np.inf
        out     = np.full(fp.shape, logp_val, dtype=float)

    elif prior == "loguniform":
        if lo <= 0 or hi <= 0 or hi <= lo:
            raise ValueError("loguniform prior requires 0 < lo < hi.")
        Z   = np.log(hi) - np.log(lo)
        out = np.full(fp.shape, -np.inf, dtype=float)
        out[inside] = -np.log(Z) - np.log(fp[inside])

    else:
        raise ValueError("fp_prior must be 'uniform' or 'loguniform'.")

    out = np.where(inside, out, -np.inf)
    if out.ndim == 0:
        return float(out)
    return out


# =============================================================================
# LOG-POSTERIOR
# =============================================================================

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
         K-Stacker score:
            log L = 0.5 * SNR_common^2
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


def log_probability(
    theta: np.ndarray,
    *,
    instruments: Sequence[Instrument],
    a_bounds: Tuple[float, float],
    m0_bounds: Tuple[float, float],
    noise_floor: float = 1.0,
    e_max: float = 0.95,
    orbit_direction: str = "any",
    weighting: str = "invvar",
    likelihood_mode: str = "positive_snr_profile",
    fp_bounds: tuple[float, float] = (1e-3, 1e1),
    fp_prior: str = "uniform",
) -> np.ndarray | float:
    """
    Multi-instrument log posterior.

    Three likelihoods are available:

    1) ``positive_snr_profile`` — PACO only

           z_k = alpha_hat_k / sqrt(var_alpha_k)
           log L = 0.5 * sum_k max(0, z_k)^2

       This is the positive per-epoch profile criterion described for the
       multi-epoch PACO framework (Dallant et al. 2023, A&A 679 A38).
       The code deliberately uses a descriptive name instead of an algorithm
       name.

    2) ``signed_snr`` — PACO / convolve / aperture

           log L = 0.5 * SNR_common^2

       This preserves the  K-Stacker signed common-flux statistic.
       The sign of SNR_common is retained in diagnostics; squaring gives the
        likelihood score.

    3) ``flux`` — PACO / convolve / aperture

           log L = -0.5 * (fp^2 S1 - 2 fp S2)

       Gaussian common-flux likelihood.  fp may be sampled as an eighth
       parameter or held fixed.

    PACO reference: Flasseur et al. 2020, A&A 637 A9.
    """
    if instruments is None or len(instruments) == 0:
        raise ValueError("log_probability requires at least one Instrument.")

    th = np.asarray(theta, dtype=float)
    scalar_input = th.ndim == 1
    if scalar_input:
        th = th[None, :]
    if th.ndim != 2:
        raise ValueError("theta must have shape (D,) or (W,D).")
    W, D = th.shape

    mode = _normalize_likelihood_mode(likelihood_mode)

    if mode == "positive_snr_profile":
        non_paco = [
            inst.name for inst in instruments
            if str(inst.photometry_method).lower() != "paco"
        ]
        if non_paco:
            raise ValueError(
                "likelihood_mode='positive_snr_profile' is intentionally PACO-only. "
                "For convolve/aperture use 'signed_snr' or 'flux'. "
                f"Non-PACO instrument(s): {', '.join(non_paco)}"
            )

    if mode in {"positive_snr_profile", "signed_snr"}:
        if D != 7:
            raise ValueError(f"In '{mode}' mode, theta must have 7 parameters.")
        theta7 = th
        fp = None
    else:
        if D == 8:
            theta7 = th[:, :7]
            fp = th[:, 7]
        elif D == 7:
            theta7 = th
            fp = np.full(W, fp_bounds[0], dtype=float)
        else:
            raise ValueError("In 'flux' mode, theta must have 7 or 8 parameters.")

    lp_theta = log_prior_hkpq(
        theta7,
        a_bounds=a_bounds,
        m0_bounds=m0_bounds,
        la0_bounds=(0.0, 2.0 * np.pi),
        e_max=e_max,
        orbit_direction=orbit_direction,
    )

    if np.isscalar(lp_theta):
        if not np.isfinite(lp_theta):
            return -np.inf
    elif not np.any(np.isfinite(lp_theta)):
        return lp_theta

    if mode == "flux":
        lp_fp = log_prior_fp(fp, bounds=fp_bounds, prior=fp_prior)
        if np.isscalar(lp_theta):
            if not (np.isfinite(lp_theta) and np.isfinite(lp_fp)):
                return -np.inf
        else:
            valid = np.isfinite(lp_theta) & np.isfinite(lp_fp)
            if not np.any(valid):
                return np.where(valid, 0.0, -np.inf)
        lp = lp_theta + lp_fp
    else:
        lp = lp_theta

    if mode == "positive_snr_profile":
        _, _, _, criterion = positive_snr_profile_multi_from_hkpq(
            theta7,
            instruments=instruments,
            noise_floor=noise_floor,
            weighting=weighting,
        )
        logL = 0.5 * criterion

    elif mode == "signed_snr":
        _, _, signed_snr = snr_multi_from_hkpq(
            theta7,
            instruments=instruments,
            noise_floor=noise_floor,
            weighting=weighting,
        )
        logL = 0.5 * np.asarray(signed_snr, dtype=float) ** 2

    else:
        S1, S2 = flux_sufficient_stats_multi_from_hkpq(
            theta7,
            instruments=instruments,
            noise_floor=noise_floor,
        )
        logL = -0.5 * (fp * fp * S1 - 2.0 * fp * S2)

    out = lp + logL
    return float(out[0]) if scalar_input else out


# =============================================================================
# WALKER INITIALISATION
# =============================================================================



def _resolve_multistart(root: dict) -> dict:
    """
    Parse the optional YAML block controlling multi-start walker initialisation.
    """
    cfg = root.get("multistart", {}) or {}
    n_starts = max(1, int(cfg.get("n_starts", 5)))
    ratios_cfg = cfg.get("ratios", None)
    ratios = [1.0] * n_starts if ratios_cfg is None else list(ratios_cfg)
    return dict(
        enabled=bool(cfg.get("enabled", False)),
        n_starts=n_starts,
        source=str(cfg.get("source", "current_init_mode") or "current_init_mode").lower(),
        ratios=ratios,
        detrack=bool(cfg.get("detrack", True)),
        detrack_dmax_px=float(cfg.get("detrack_dmax_px", 2.0)),
        min_snr=cfg.get("min_snr", None),
        init_search_h5=str(cfg.get("init_search_h5", "mcmc_init_search.h5") or "mcmc_init_search.h5"),
    )


def _clip_theta_init_to_bounds(
    theta_init: np.ndarray,
    *,
    a_bounds: Tuple[float, float],
    m0_bounds: Tuple[float, float],
    la0_bounds: Tuple[float, float],
    e_max: float,
) -> np.ndarray:
    """
    Clip a 7-D non-singular initial vector to the supported box / disk priors.
    """
    theta_init = np.asarray(theta_init, dtype=float).copy()
    theta_init[0] = np.clip(theta_init[0], *a_bounds)
    theta_init[1] = np.clip(theta_init[1], *la0_bounds)
    theta_init[2] = np.clip(theta_init[2], *m0_bounds)

    h, k = float(theta_init[3]), float(theta_init[4])
    e = float(np.hypot(h, k))
    if e > e_max and e > 0.0:
        theta_init[3] *= e_max / e
        theta_init[4] *= e_max / e

    p, q = float(theta_init[5]), float(theta_init[6])
    r2 = p * p + q * q
    if r2 > 1.0 and r2 > 0.0:
        r = np.sqrt(r2)
        theta_init[5] /= r
        theta_init[6] /= r

    return theta_init


def _classical_row_to_theta_init(
    row: np.ndarray,
    *,
    t_ref: float,
) -> np.ndarray:
    """
    Convert one ranked classical-orbit row into the 7-D non-singular state.
    """
    a, e, t0, m0, omega, inc, theta0 = map(float, row[:7])

    n = 2.0 * np.pi * np.sqrt(m0 / (a ** 3))
    M0 = wrap_2pi(n * (t_ref - t0))
    la0 = wrap_2pi(M0 + omega + theta0)

    w_sum = omega + theta0
    h = e * np.sin(w_sum)
    k = e * np.cos(w_sum)
    sin_i_2 = np.sin(0.5 * inc)
    p = sin_i_2 * np.sin(omega)
    q = sin_i_2 * np.cos(omega)

    return np.array([a, la0, m0, h, k, p, q], dtype=float)


def _load_ranked_classical_rows_from_h5(
    h5_path: str,
    *,
    min_snr: Optional[float] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load ranked classical-orbit solutions from an HDF5 file.
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Ranked initialisation file not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        if "Best solutions" not in f:
            raise KeyError(f"Dataset 'Best solutions' not found in {h5_path}.")
        data = np.asarray(f["Best solutions"][:], dtype=float)

    if data.ndim != 2 or data.shape[1] < 7:
        raise ValueError(
            f"'Best solutions' in {h5_path} has shape {data.shape}; expected (N, >=7)."
        )

    snr = None
    if data.shape[1] >= 8:
        snr = np.asarray(data[:, -1], dtype=float)
        if min_snr is not None:
            valid = snr >= float(min_snr)
            data = data[valid]
            snr = snr[valid]

    if data.shape[0] == 0:
        raise RuntimeError(f"No ranked solutions available in {h5_path} after filtering.")

    return data, snr


def _resolve_results_h5_from_source(
    params: "Params",
    root: dict,
    *,
    init_mode: str,
    multistart_cfg: dict,
) -> Optional[str]:
    """
    Resolve which ranked HDF5 file should feed the initial centres.
    """
    source = str(multistart_cfg.get("source", "current_init_mode")).lower()
    if source == "current_init_mode":
        source = init_mode

    values_dir = params.get_path("values_dir")

    if source == "init_search":
        init_search_cfg = root.get("init_search", {}) or {}
        filename = str(
            multistart_cfg.get("init_search_h5")
            or init_search_cfg.get("output_h5", "mcmc_init_search.h5")
        )
        return os.path.join(values_dir, filename)

    if source == "manual":
        return None

    raise ValueError(
        "multistart.source must be 'current_init_mode', 'manual', or 'init_search'."
    )


def _centres_are_track_distinct(
    theta: np.ndarray,
    retained: Sequence[np.ndarray],
    *,
    track_context: Sequence[dict],
    t_ref: float,
    dmax_px: float,
) -> tuple[bool, float]:
    """
    Check whether a candidate follows a detector track distinct from every
    already retained centre.

    For each pair we compute the maximum native-pixel separation over all
    observed epochs and instruments.  A new centre is accepted only when this
    maximum separation is strictly larger than `dmax_px` for every retained
    centre.  This prevents the multi-start ensemble from spending several
    starts on numerically different parameters that trace essentially the same
    observed path.
    """
    if not retained:
        return True, float("inf")

    minimum_pair_dmax = float("inf")
    for earlier in retained:
        pair_dmax = 0.0
        for inst in track_context:
            x1, y1 = _predict_pixel_track_native(
                theta, inst["ts"], size=int(inst["size"]),
                scale=float(inst["scale"]), t_ref=float(t_ref),
            )
            x0, y0 = _predict_pixel_track_native(
                earlier, inst["ts"], size=int(inst["size"]),
                scale=float(inst["scale"]), t_ref=float(t_ref),
            )
            pair_dmax = max(pair_dmax, float(np.max(np.hypot(x1 - x0, y1 - y0))))
        minimum_pair_dmax = min(minimum_pair_dmax, pair_dmax)
        if pair_dmax <= float(dmax_px):
            return False, minimum_pair_dmax
    return True, minimum_pair_dmax


def _build_theta_centres_from_ranked_h5(
    h5_path: str,
    *,
    t_ref: float,
    n_centres: int,
    min_snr: Optional[float] = None,
    a_bounds: Tuple[float, float],
    m0_bounds: Tuple[float, float],
    la0_bounds: Tuple[float, float],
    e_max: float,
    orbit_direction: str = "any",
    track_context: Optional[Sequence[dict]] = None,
    detrack: bool = False,
    detrack_dmax_px: float = 2.0,
) -> tuple[list[np.ndarray], list[dict]]:
    """
    Read the top-ranked classical-orbit solutions from an HDF5 file and convert
    them into clipped 7-D non-singular centres suitable for walker initialisation.
    """
    data, snr = _load_ranked_classical_rows_from_h5(h5_path, min_snr=min_snr)

    centres: list[np.ndarray] = []
    meta: list[dict] = []
    direction = _normalise_orbit_direction(orbit_direction)

    for i in range(int(data.shape[0])):
        theta = _classical_row_to_theta_init(data[i, :7], t_ref=t_ref)
        if not bool(_direction_mask_from_pq(theta[5], theta[6], direction)):
            continue
        theta = _clip_theta_init_to_bounds(
            theta,
            a_bounds=a_bounds,
            m0_bounds=m0_bounds,
            la0_bounds=la0_bounds,
            e_max=e_max,
        )

        min_track_dmax_px = None
        if bool(detrack) and track_context is not None and centres:
            distinct, min_track_dmax_px = _centres_are_track_distinct(
                theta, centres, track_context=track_context, t_ref=t_ref,
                dmax_px=float(detrack_dmax_px),
            )
            if not distinct:
                continue

        centres.append(theta)
        meta.append(
            dict(
                rank=i + 1,
                snr=None if snr is None else float(snr[i]),
                classical_row=np.asarray(data[i, :7], dtype=float).tolist(),
                min_track_dmax_px=min_track_dmax_px,
            )
        )
        if len(centres) >= int(n_centres):
            break

    if not centres:
        raise RuntimeError(f"No valid initial centres could be built from {h5_path}.")
    if len(centres) < int(n_centres):
        raise RuntimeError(
            f"Requested {n_centres} initial centres but only {len(centres)} satisfy "
            f"the prior and detector-track distinctness requirements. "
            "Increase init_search.top_n_save, reduce multistart.n_starts, or reduce "
            "multistart.detrack_dmax_px."
        )

    return centres, meta


def _normalise_multistart_ratios(ratios: Sequence[float], n_starts: int) -> np.ndarray:
    """
    Normalise user-provided multi-start ratios.
    """
    ratios = list(ratios)
    if len(ratios) == 0:
        ratios = [1.0] * n_starts
    if len(ratios) < n_starts:
        ratios = ratios + [ratios[-1]] * (n_starts - len(ratios))
    ratios = np.asarray(ratios[:n_starts], dtype=float)
    ratios = np.where(np.isfinite(ratios) & (ratios > 0.0), ratios, 0.0)
    if float(np.sum(ratios)) <= 0.0:
        ratios = np.ones(n_starts, dtype=float)
    return ratios / np.sum(ratios)


def _allocate_multistart_walkers(
    nwalkers: int,
    *,
    n_starts: int,
    ratios: Sequence[float],
) -> np.ndarray:
    """
    Convert user ratios into integer walker counts per start.
    """
    if n_starts <= 0:
        raise ValueError("n_starts must be >= 1.")
    if nwalkers < n_starts:
        raise ValueError(
            f"nwalkers={nwalkers} is smaller than n_starts={n_starts}; "
            "increase nwalkers or reduce multistart.n_starts."
        )

    w = _normalise_multistart_ratios(ratios, n_starts)
    exact = nwalkers * w
    counts = np.floor(exact).astype(int)

    remainder = int(nwalkers - np.sum(counts))
    if remainder > 0:
        order = np.argsort(-(exact - counts))
        counts[order[:remainder]] += 1

    counts = np.maximum(counts, 0)
    if np.any(counts == 0):
        zero_idx = np.where(counts == 0)[0]
        for zi in zero_idx:
            donor = int(np.argmax(counts))
            if counts[donor] <= 1:
                raise RuntimeError(
                    "Could not allocate at least one walker to every start. "
                    "Increase nwalkers or reduce n_starts."
                )
            counts[donor] -= 1
            counts[zi] += 1

    return counts


def draw_walkers_multistart(
    nwalkers: int,
    theta_centres: Sequence[np.ndarray],
    *,
    ratios: Sequence[float],
    a_bounds: Tuple[float, float],
    m0_bounds: Tuple[float, float],
    e_max: float = 0.95,
    la0_bounds: Tuple[float, float] = (0.0, 2.0 * np.pi),
    spread: dict = dict(a=0.02, la0=0.2, m0=0.0, hk=0.02, pq=0.02),
    orbit_direction: str = "any",
) -> np.ndarray:
    """
    Draw the initial walker cloud around several ranked centres.
    """
    theta_centres = [np.asarray(theta, dtype=float) for theta in theta_centres]
    if len(theta_centres) == 0:
        raise ValueError("theta_centres must contain at least one centre.")

    counts = _allocate_multistart_walkers(
        nwalkers=nwalkers,
        n_starts=len(theta_centres),
        ratios=ratios,
    )

    chunks = []
    for theta_c, n_here in zip(theta_centres, counts):
        chunks.append(
            draw_walkers_around_theta_init(
                nwalkers=int(n_here),
                theta_init=np.asarray(theta_c, dtype=float),
                a_bounds=a_bounds,
                m0_bounds=m0_bounds,
                e_max=e_max,
                la0_bounds=la0_bounds,
                spread=spread,
                orbit_direction=orbit_direction,
            )
        )

    p0 = np.vstack(chunks)
    if p0.shape[0] != nwalkers:
        raise RuntimeError(
            f"draw_walkers_multistart built {p0.shape[0]} walkers, expected {nwalkers}."
        )
    return p0


def draw_walkers_around_theta_init(
    nwalkers: int,
    theta_init: np.ndarray,
    *,
    a_bounds: Tuple[float, float],
    m0_bounds: Tuple[float, float],
    e_max: float = 0.95,
    la0_bounds: Tuple[float, float] = (0.0, 2.0 * np.pi),
    spread: dict = dict(a=0.02, la0=0.2, m0=0.0, hk=0.02, pq=0.02),
    orbit_direction: str = "any",
    max_tries: int = 10000,
) -> np.ndarray:
    """
    Initialise walkers near theta_init with Gaussian jitter.

    For each walker:
      - Jitter a, la0, m0, (h,k), (p,q) with Gaussian noise (scale = spread).
      - Project (h,k) back to the disk e ≤ e_max if needed.
      - Project (p,q) back to the physical unit disk p²+q² ≤ 1 if needed.
      - Enforce orbit_direction when requested:
            prograde   -> p²+q² <= 1/2
            retrograde -> p²+q² >= 1/2
      - Reject walkers that violate box constraints on (a, m0, la0).

    The function retries until nwalkers valid samples are found, up to
    max_tries total attempts.  If this fails, raise RuntimeError.
    """
    a0, la0_0, m00, h0, k0, p0_, q0_ = np.asarray(theta_init, dtype=float)
    samples, tries = [], 0

    while len(samples) < nwalkers and tries < max_tries:
        tries += 1

        a   = a0 * (1.0 + spread["a"] * np.random.randn())
        la0 = wrap_2pi(la0_0 + spread["la0"] * np.random.randn())

        # Jitter m0 only if the prior allows variation.
        if m0_bounds[1] > m0_bounds[0] and spread.get("m0", 0.0) > 0.0:
            m0 = np.clip(m00 * (1.0 + spread["m0"] * np.random.randn()), *m0_bounds)
        else:
            m0 = m00

        h = h0 + spread["hk"] * np.random.randn()
        k = k0 + spread["hk"] * np.random.randn()
        e = np.hypot(h, k)
        if e > e_max and e > 0.0:
            h *= e_max / e
            k *= e_max / e
        elif e == 0.0:
            h = k = 0.0

        p  = p0_ + spread["pq"] * np.random.randn()
        q  = q0_ + spread["pq"] * np.random.randn()
        r2 = p * p + q * q
        if r2 > 1.0 and r2 > 0.0:
            r = np.sqrt(r2)
            p /= r
            q /= r

        direction = _normalise_orbit_direction(orbit_direction)
        r2 = p*p + q*q
        if direction == "prograde" and r2 > 0.5:
            continue
        if direction == "retrograde" and r2 < 0.5:
            continue

        if not (a_bounds[0]   <= a   <= a_bounds[1]):   continue
        if not (m0_bounds[0]  <= m0  <= m0_bounds[1]):  continue
        if not (la0_bounds[0] <= la0 <= la0_bounds[1]): continue

        samples.append([a, la0, m0, h, k, p, q])

    if len(samples) < nwalkers:
        raise RuntimeError(
            "Failed to initialise all walkers within max_tries. "
            "Try increasing max_tries or relaxing init_spread."
        )
    return np.array(samples, dtype=float)


# =============================================================================
# EMCEE DRIVER
# =============================================================================

# The multiprocessing design below is intentionally conservative and explicit.
# The log-posterior is already vectorized over walkers, so the fastest strategy
# is not to send one walker at a time to subprocesses.  Instead, each emcee call
# receives a batch of walkers, and the batch is split into a small number of
# vectorized chunks.  Each worker evaluates one chunk with NumPy vectorization.
#
# Important implementation detail:
#   The large run-time objects (instruments, images, profiles, priors, etc.) are
#   installed once in each worker when the ProcessPoolExecutor starts.  After
#   that, every submitted task sends only a small theta chunk.  This avoids
#   repeatedly serializing the full likelihood context at every MCMC step.

_WORKER_LP_KWARGS: Optional[dict] = None


def _available_cpu_count() -> int:
    """
    Return the number of CPUs visible to the current process.

    On SLURM clusters this should follow the job cpuset / CPU affinity.  This is
    more useful than the physical node CPU count because a job requesting four
    CPUs should normally see only those four CPUs in its affinity mask.
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        return max(1, os.cpu_count() or 1)


def _init_logprob_worker(lp_kwargs: dict) -> None:
    """
    Initializer executed once inside each worker process.

    The likelihood context can contain large arrays: preprocessed images,
    local background/noise maps, masks, time vectors, and instrument metadata.  Keeping this
    context in a worker-global variable means that each MCMC task only needs to
    send a small theta chunk to the worker.  This is the key optimisation for
    short, repeated MCMC likelihood calls.
    """
    global _WORKER_LP_KWARGS
    _WORKER_LP_KWARGS = lp_kwargs


def _logprob_chunk_worker(thetas_chunk: np.ndarray) -> np.ndarray:
    """
    Evaluate one vectorized chunk inside a worker process.

    The heavy likelihood context is read from the worker-global variable set by
    `_init_logprob_worker`.  Only the walker coordinates are transferred for
    each task.
    """
    if _WORKER_LP_KWARGS is None:
        raise RuntimeError(
            "Worker likelihood context is not initialised. "
            "This usually means the ProcessPoolExecutor initializer did not run."
        )
    return log_probability(thetas_chunk, **_WORKER_LP_KWARGS)


def _logprob_chunk_local(thetas_chunk: np.ndarray, lp_kwargs: dict) -> np.ndarray:
    """
    Evaluate one vectorized chunk in the main process.

    This path is used when the run is sequential, or when the current emcee batch
    is too small to benefit from subprocess scheduling.
    """
    return log_probability(thetas_chunk, **lp_kwargs)


class BatchPoolLogProb:
    """
    Callable adapter for `emcee.EnsembleSampler(vectorize=True)`.

    emcee calls this object with a batch of W walkers.  The adapter keeps the
    evaluation vectorized, but can distribute large enough batches across a
    small fixed number of worker processes.

    Parallelisation strategy
    ────────────────────────
    - Sequential mode: evaluate the full W-walker batch in the main process.
    - Parallel mode: split the W walkers into a few contiguous chunks and send
      those chunks to worker processes.
    - Each worker already owns the large likelihood context, so submitted tasks
      only contain the theta chunk.

    This is well suited to the HARMONI MCMC use case, where each emcee step is
    repeated many times and the data arrays are large compared with the walker
    coordinates.

    Parameters
    ──────────
    lp_kwargs   : keyword arguments for `log_probability` in sequential mode.
    executor    : ProcessPoolExecutor instance, or None for sequential mode.
    max_workers : resolved number of worker processes.
    chunk_size  : walkers per submitted job.  None means automatic chunking.
    """

    def __init__(
        self,
        lp_kwargs: dict,
        executor: Optional[ProcessPoolExecutor],
        *,
        max_workers: int = 1,
        chunk_size: Optional[int] = None,
    ):
        self.lp_kwargs   = lp_kwargs
        self.executor    = executor
        self.max_workers = max(1, int(max_workers))
        self.chunk_size  = None if chunk_size is None else max(1, int(chunk_size))

    def _effective_chunk_size(self, W: int) -> int:
        """
        Choose a chunk size for the current emcee batch.

        In automatic mode, the target is roughly one chunk per worker.  This is
        usually better than creating many tiny tasks, because the likelihood is
        already vectorized within each chunk.
        """
        if self.chunk_size is not None:
            return self.chunk_size
        return max(1, int(np.ceil(W / float(self.max_workers))))

    def __call__(self, thetas: np.ndarray) -> np.ndarray:
        thetas = np.asarray(thetas, dtype=float, order="C")
        W = int(thetas.shape[0])

        # If there is no executor, or the current batch is tiny, the fastest path
        # is a single vectorized evaluation in the main process.
        if self.executor is None or self.max_workers <= 1 or W <= 1:
            return _logprob_chunk_local(thetas, self.lp_kwargs)

        chunk_size = self._effective_chunk_size(W)
        slices = [(start, min(start + chunk_size, W)) for start in range(0, W, chunk_size)]

        # If the requested chunking produces only one task, avoid subprocess
        # scheduling entirely.  This keeps small red-blue emcee sub-batches fast.
        if len(slices) <= 1:
            return _logprob_chunk_local(thetas, self.lp_kwargs)

        futures = [
            self.executor.submit(_logprob_chunk_worker, thetas[start:stop])
            for start, stop in slices
        ]

        out = np.empty(W, dtype=float)
        for (start, stop), fut in zip(slices, futures):
            out[start:stop] = fut.result()
        return out


def run_emcee_hybrid(
    p0: np.ndarray,
    lp_kwargs: dict,
    *,
    nsteps: int = 4000,
    burnin: int = 1000,
    moves=None,
    max_workers: int | None = None,
    chunk_size: Optional[int] = None,
    progress: bool = True,
) -> emcee.EnsembleSampler:
    """
    Run emcee with vectorized log-probability and optional multiprocessing.

    Workflow
    ────────
    1. Resolve the number of workers from the YAML and the CPU affinity visible
       to the current job.
    2. If only one worker is requested/available, run a fully vectorized
       sequential MCMC.  This is often fastest for small ensembles.
    3. If multiple workers are available, initialise each worker once with the
       large likelihood context.
    4. During MCMC, send only theta chunks to the workers.
    5. Run burn-in, reset the sampler, run production, and return the sampler.

    Why this parallelisation is efficient
    ─────────────────────────────────────
    The expensive data arrays are not attached to every submitted task.  They
    are stored once inside each worker process.  Repeated MCMC evaluations then
    transfer only small `(n_chunk, ndim)` arrays.  This keeps inter-process
    communication low while preserving NumPy vectorisation inside every chunk.

    Parameters
    ──────────
    p0          : (nwalkers, ndim) initial positions.
    lp_kwargs   : forwarded to log_probability.
    nsteps      : production steps.
    burnin      : burn-in steps (discarded).
    moves       : emcee move objects (default: DEMove + DESnookerMove).
    max_workers : requested number of parallel processes. None means auto.
    chunk_size  : walkers per job. None means automatic chunking.
    progress    : show tqdm progress bar.

    Returns
    ───────
    emcee.EnsembleSampler containing only the production chain.
    """
    p0 = np.asarray(p0, dtype=float, order="C")
    if p0.ndim != 2:
        raise ValueError("p0 must be 2D with shape (nwalkers, ndim).")

    if moves is None:
        moves = [emcee.moves.DEMove(), emcee.moves.DESnookerMove()]

    nwalkers, ndim = p0.shape
    visible_cpus = _available_cpu_count()

    if max_workers is None:
        resolved_workers = visible_cpus
        requested_workers_label = "auto"
    else:
        resolved_workers = int(max_workers)
        requested_workers_label = str(max_workers)

    resolved_workers = max(1, min(resolved_workers, visible_cpus, nwalkers))

    if chunk_size is None:
        chunk_label = "auto"
        example_chunk = max(1, int(np.ceil(nwalkers / float(resolved_workers))))
    else:
        chunk_label = str(int(chunk_size))
        example_chunk = int(chunk_size)

    print("[MCMC parallel] Configuration")
    print(f"[MCMC parallel]   walkers / ndim          : {nwalkers} / {ndim}")
    print(f"[MCMC parallel]   visible CPUs            : {visible_cpus}")
    print(f"[MCMC parallel]   requested max_workers   : {requested_workers_label}")
    print(f"[MCMC parallel]   resolved max_workers    : {resolved_workers}")
    print(f"[MCMC parallel]   chunk_size              : {chunk_label}")
    print(f"[MCMC parallel]   example chunk size      : {example_chunk}")

    if resolved_workers <= 1:
        print("[MCMC parallel]   mode                    : sequential vectorized")
        print("[MCMC parallel]   note                    : no subprocess overhead")
        batched = BatchPoolLogProb(
            lp_kwargs,
            executor=None,
            max_workers=1,
            chunk_size=chunk_size,
        )
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, batched, vectorize=True, moves=moves
        )
        state = sampler.run_mcmc(p0, burnin, progress=progress)
        sampler.reset()
        sampler.run_mcmc(state, nsteps, progress=progress)
        return sampler

    print("[MCMC parallel]   mode                    : process pool + vectorized chunks")
    print("[MCMC parallel]   worker data transfer    : likelihood context installed once")
    print("[MCMC parallel]   per-task transfer       : theta chunk only")

    with ProcessPoolExecutor(
        max_workers=resolved_workers,
        initializer=_init_logprob_worker,
        initargs=(lp_kwargs,),
    ) as executor:
        batched = BatchPoolLogProb(
            lp_kwargs,
            executor,
            max_workers=resolved_workers,
            chunk_size=chunk_size,
        )
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, batched, vectorize=True, moves=moves
        )
        state = sampler.run_mcmc(p0, burnin, progress=progress)
        sampler.reset()   # discard burn-in steps
        sampler.run_mcmc(state, nsteps, progress=progress)

    return sampler




# =============================================================================
# ADAPTIVE PARALLEL TEMPERING (reddemcee)
# =============================================================================
#
# Reference:
#   Peña R. & Jenkins 2026, "Closing the evidence gap: reddemcee, a fast
#   adaptive parallel tempering sampler", arXiv:2509.24870.
#
# reddemcee is imported lazily so the  emcee path has no new runtime
# dependency unless mcmc.sampler: reddemcee is selected.
# =============================================================================

def _prior_only_scalar(theta: np.ndarray, lp_kwargs: Mapping[str, Any]) -> float:
    """Scalar prior callback required by reddemcee.PTSampler."""
    theta = np.asarray(theta, dtype=float)
    mode = _normalize_likelihood_mode(lp_kwargs["likelihood_mode"])

    if mode == "flux" and theta.size == 8:
        theta7 = theta[:7]
        fp = float(theta[7])
    elif theta.size == 7:
        theta7 = theta
        fp = None if mode != "flux" else float(lp_kwargs["fp_bounds"][0])
    else:
        return -np.inf

    lp = log_prior_hkpq(
        theta7,
        a_bounds=lp_kwargs["a_bounds"],
        m0_bounds=lp_kwargs["m0_bounds"],
        la0_bounds=(0.0, 2.0 * np.pi),
        e_max=lp_kwargs["e_max"],
        orbit_direction=lp_kwargs["orbit_direction"],
    )
    lp = float(lp)
    if not np.isfinite(lp):
        return -np.inf

    if mode == "flux":
        lpf = float(
            log_prior_fp(
                np.asarray([fp], dtype=float),
                bounds=lp_kwargs["fp_bounds"],
                prior=lp_kwargs["fp_prior"],
            )[0]
        )
        if not np.isfinite(lpf):
            return -np.inf
        lp += lpf

    return float(lp)


def _likelihood_only_scalar(theta: np.ndarray, lp_kwargs: Mapping[str, Any]) -> float:
    """Scalar likelihood callback required by reddemcee.PTSampler."""
    theta = np.asarray(theta, dtype=float)
    mode = _normalize_likelihood_mode(lp_kwargs["likelihood_mode"])
    theta7 = theta[:7][None, :]

    if mode == "positive_snr_profile":
        _, _, _, criterion = positive_snr_profile_multi_from_hkpq(
            theta7,
            instruments=lp_kwargs["instruments"],
            noise_floor=lp_kwargs["noise_floor"],
            weighting=lp_kwargs["weighting"],
        )
        return float(0.5 * criterion[0])

    if mode == "signed_snr":
        _, _, snr = snr_multi_from_hkpq(
            theta7,
            instruments=lp_kwargs["instruments"],
            noise_floor=lp_kwargs["noise_floor"],
            weighting=lp_kwargs["weighting"],
        )
        return float(0.5 * float(snr[0]) ** 2)

    if theta.size == 8:
        fp = float(theta[7])
    else:
        fp = float(lp_kwargs["fp_bounds"][0])
    S1, S2 = flux_sufficient_stats_multi_from_hkpq(
        theta7,
        instruments=lp_kwargs["instruments"],
        noise_floor=lp_kwargs["noise_floor"],
    )
    return float(-0.5 * (fp * fp * S1[0] - 2.0 * fp * S2[0]))


class _ReddenColdSamplerAdapter:
    """
    Expose reddemcee's cold chain through the subset of emcee's API used below.

    This deliberately keeps all temperatures in ``pt_sampler`` for advanced
    diagnostics while exposing the beta=1 chain to the standard output writer.
    """

    def __init__(
        self,
        pt_sampler: Any,
        *,
        burnin_sweeps: int,
        lp_kwargs: Mapping[str, Any],
    ) -> None:
        self.pt_sampler = pt_sampler
        self._burnin = int(burnin_sweeps)
        raw = np.asarray(pt_sampler.get_chain(flat=False), dtype=float)
        if raw.ndim != 4:
            raise ValueError(
                f"Unexpected reddemcee chain shape {raw.shape}; expected "
                "(temperature, sweep, walker, parameter)."
            )
        self._cold_chain = raw[0, self._burnin:, :, :]

        # Compute cold posterior logP exactly with the same pipeline likelihood.
        flat = self._cold_chain.reshape(-1, self._cold_chain.shape[-1])
        logp = np.asarray(log_probability(flat, **dict(lp_kwargs)), dtype=float)
        self._cold_logp = logp.reshape(self._cold_chain.shape[:2])

        acc = np.asarray(pt_sampler.acceptance_fraction, dtype=float)
        self.acceptance_fraction = acc[0] if acc.ndim > 1 else acc

    def get_chain(self, discard: int = 0, thin: int = 1, flat: bool = False) -> np.ndarray:
        chain = self._cold_chain[int(discard)::max(1, int(thin))]
        return chain.reshape(-1, chain.shape[-1]) if flat else chain

    def get_log_prob(self, discard: int = 0, thin: int = 1, flat: bool = False) -> np.ndarray:
        values = self._cold_logp[int(discard)::max(1, int(thin))]
        return values.reshape(-1) if flat else values

    def get_autocorr_time(self, discard: int = 0, thin: int = 1, tol: float = 0) -> np.ndarray:
        chain = self.get_chain(discard=discard, thin=thin, flat=False)
        return emcee.autocorr.integrated_time(chain, tol=tol, quiet=False)


def _reddemcee_betas(cfg: Mapping[str, Any]) -> np.ndarray:
    """Build an explicit temperature ladder; beta[0]=1 is the cold posterior."""
    if cfg.get("betas", None) is not None:
        betas = np.asarray(cfg["betas"], dtype=float)
    else:
        ntemps = int(cfg["ntemps"])
        beta_min = float(cfg["beta_min"])
        if not (0.0 < beta_min < 1.0):
            raise ValueError("mcmc.reddemcee.beta_min must satisfy 0 < beta_min < 1.")
        betas = np.geomspace(1.0, beta_min, ntemps)

    if betas.ndim != 1 or betas.size < 2:
        raise ValueError("reddemcee betas must be a 1-D array with >=2 temperatures.")
    if not np.isclose(betas[0], 1.0):
        raise ValueError("reddemcee beta ladder must start at beta=1.")
    if not np.all(np.diff(betas) < 0.0):
        raise ValueError("reddemcee betas must be strictly decreasing.")
    return betas


def run_reddemcee_adaptive(
    p0: np.ndarray,
    lp_kwargs: Mapping[str, Any],
    *,
    nsteps: int,
    burnin: int,
    cfg: Mapping[str, Any],
    progress: bool = True,
) -> _ReddenColdSamplerAdapter:
    """
    Run adaptive parallel tempering from the same grid/manual walker cloud.

    No quasi-random design is generated here.  The existing MCMC walker cloud is
    copied across temperatures (with a cyclic walker permutation), after which
    hot chains and swaps provide the global exploration.
    """
    try:
        import reddemcee
    except ImportError as exc:
        raise ImportError(
            "mcmc.sampler='reddemcee' requires the reddemcee package. "
            "Install it in the K-Stacker environment before launching this mode."
        ) from exc

    p0 = np.asarray(p0, dtype=float)
    if p0.ndim != 2:
        raise ValueError("p0 must have shape (nwalkers, ndim).")
    nwalkers, ndim = p0.shape
    betas = _reddemcee_betas(cfg)
    ntemps = int(betas.size)

    # Same physically valid cloud, independently ordered at each temperature.
    # This intentionally avoids quasi-random/random-prior initialisation.
    p0_pt = np.stack(
        [np.roll(p0, shift=t, axis=0) for t in range(ntemps)],
        axis=0,
    )

    def scalar_prior(theta):
        return _prior_only_scalar(theta, lp_kwargs)

    def scalar_like(theta):
        return _likelihood_only_scalar(theta, lp_kwargs)

    _print_section("MCMC SAMPLER — ADAPTIVE PARALLEL TEMPERING")
    print("  engine               : reddemcee.PTSampler")
    print("  reference            : Peña R. & Jenkins 2026, arXiv:2509.24870")
    print(f"  temperatures         : {ntemps}")
    print(f"  beta range           : {betas[0]:.3g} -> {betas[-1]:.3g}")
    print(f"  walkers / dimension  : {nwalkers} / {ndim}")
    print(f"  adaptation           : {cfg['adapt_mode']}")
    print(f"  adapt_tau / adapt_nu : {float(cfg['adapt_tau']):g} / {float(cfg['adapt_nu']):g}")
    print(f"  stretch a            : {float(cfg['stretch_a']):g}")
    print(f"  inner steps          : {int(cfg['inner_steps'])}")
    print(f"  burn-in sweeps       : {int(burnin):,}")
    print(f"  production sweeps    : {int(nsteps):,}")
    print("  initialisation       : existing grid/manual walker cloud; no quasi-random design")
    print()

    sampler = reddemcee.PTSampler(
        nwalkers,
        ndim,
        scalar_like,
        scalar_prior,
        betas=betas.copy(),
        moves=reddemcee.moves.StretchMove(a=float(cfg["stretch_a"])),
        smd_history=bool(cfg["smd_history"]),
        tsw_history=bool(cfg["tsw_history"]),
        adapt_tau=float(cfg["adapt_tau"]),
        adapt_nu=float(cfg["adapt_nu"]),
        adapt_mode=str(cfg["adapt_mode"]),
    )

    sampler.run_mcmc(
        p0_pt,
        int(burnin) + int(nsteps),
        int(cfg["inner_steps"]),
        progress=progress,
    )

    final_betas = np.asarray(sampler.betas, dtype=float)
    print()
    print("  final beta range     : "
          f"{float(final_betas[0]):.3g} -> {float(final_betas[-1]):.3g}")
    try:
        tsw = np.asarray(sampler.get_tsw(), dtype=float)
        print(f"  median swap statistic: {float(np.nanmedian(tsw)):.4f}")
    except Exception:
        pass
    print()
    print("-" * 110)
    print("Adaptive PT complete; downstream products use the cold beta=1 chain.")
    print("-" * 110)
    print()

    return _ReddenColdSamplerAdapter(
        sampler,
        burnin_sweeps=int(burnin),
        lp_kwargs=lp_kwargs,
    )


# =============================================================================
# CONSOLE LOGGING AND OUTPUT SAVE HELPERS
# =============================================================================

class Tee:
    """
    Minimal stdout/stderr tee.

    Every message written to the console is also written to a log file.  This is
    intentionally small and dependency-free so that long cluster jobs keep a
    faithful text record of what happened: configuration choices, warnings,
    diagnostics and output paths.
    """
    def __init__(self, *streams):
        self.streams = streams

    def isatty(self):
        try:
            return bool(self.streams[0].isatty())
        except Exception:
            return False

    def write(self, data):
        ansi = re.compile(r"\x1b\[[0-9;]*m")
        for stream in self.streams:
            payload = data
            try:
                if not stream.isatty():
                    payload = ansi.sub("", payload)
            except Exception:
                payload = ansi.sub("", payload)
            stream.write(payload)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _bool_from_config(value: Any, default: bool = True) -> bool:
    """Parse YAML booleans robustly, including yes/no strings."""
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _setup_console_logging_from_yaml(yaml_path: str):
    """
    Open the console log requested by the YAML and install stdout/stderr tees.

    Returns a tuple `(old_stdout, old_stderr, log_file, log_path)`.  The caller
    is responsible for restoring stdout/stderr and closing the file in `finally`.
    """
    try:
        params = Params.read(yaml_path)
        root = dict(params._params)
        logging_cfg = root.get("logging", {}) or {}
        enabled = _bool_from_config(logging_cfg.get("save_console", True), True)
        if not enabled:
            return None, None, None, None

        base_dir = Path(yaml_path).expanduser().resolve().parent
        values_dir = str(_resolve_path_from_yaml(base_dir, root.get("values_dir", None), "values"))
        os.makedirs(values_dir, exist_ok=True)
        filename = str(logging_cfg.get("filename", "mcmc_console.log") or "mcmc_console.log")
        timestamp = _bool_from_config(logging_cfg.get("timestamp_filename", False), False)
        if timestamp:
            stem, ext = os.path.splitext(filename)
            ext = ext or ".log"
            filename = f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
        log_path = os.path.join(values_dir, filename)
        log_file = open(log_path, "a", encoding="utf-8", buffering=1)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = Tee(old_stdout, log_file)
        sys.stderr = Tee(old_stderr, log_file)
        print("=" * 100)
        print(f"[logging] Console output is also being written to: {log_path}")
        print("=" * 100)
        return old_stdout, old_stderr, log_file, log_path
    except Exception as exc:
        # Do not prevent the science run from starting just because logging setup failed.
        print(f"[logging] (WARN) Could not initialise console log: {exc}")
        return None, None, None, None


def _restore_console_logging(old_stdout, old_stderr, log_file, log_path):
    """Restore stdout/stderr and close the optional console log file."""
    try:
        if log_file is not None:
            print("=" * 100)
            print(f"[logging] Closing console log: {log_path}")
            print("=" * 100)
    finally:
        if old_stdout is not None:
            sys.stdout = old_stdout
        if old_stderr is not None:
            sys.stderr = old_stderr
        if log_file is not None:
            log_file.close()


def save_mcmc_chains_and_logprob(
    *,
    params: "Params",
    sampler: emcee.EnsembleSampler,
    thin: int,
    discard: int = 0,
    filename: str = "mcmc_chain_and_logprob.h5",
) -> str:
    """
    Save the production MCMC samples and their log-probabilities.

    The file contains both the native emcee arrays and flattened/thinned arrays:
      - chain:             (nsteps, nwalkers, ndim)
      - log_prob:          (nsteps, nwalkers)
      - flat_chain:        (nflat, ndim), thinned by `thin`
      - flat_log_prob:     (nflat,), same thinning as flat_chain
      - acceptance_fraction: one value per walker

    This is the file to use later for convergence checks, posterior summaries,
    or downstream analysis without rerunning the expensive MCMC.
    """
    values_dir = params.get_path("values_dir")
    os.makedirs(values_dir, exist_ok=True)
    out_path = os.path.join(values_dir, filename)

    thin = max(1, int(thin))
    discard = max(0, int(discard))

    chain = sampler.get_chain(discard=0, flat=False)
    log_prob = sampler.get_log_prob(discard=0, flat=False)
    flat_chain = sampler.get_chain(discard=discard, thin=thin, flat=True)
    flat_log_prob = sampler.get_log_prob(discard=discard, thin=thin, flat=True)

    ndim = int(chain.shape[-1])
    parameter_names = ["a", "la0", "m0", "h", "k", "p", "q"] + (["fp"] if ndim == 8 else [])

    with h5py.File(out_path, "w") as f:
        f.create_dataset("chain", data=chain, compression="gzip", compression_opts=4)
        f.create_dataset("log_prob", data=log_prob, compression="gzip", compression_opts=4)
        f.create_dataset("flat_chain", data=flat_chain, compression="gzip", compression_opts=4)
        f.create_dataset("flat_log_prob", data=flat_log_prob, compression="gzip", compression_opts=4)
        f.create_dataset("acceptance_fraction", data=np.asarray(sampler.acceptance_fraction, dtype=float))
        f.attrs["parameter_names"] = json.dumps(parameter_names)
        f.attrs["thin"] = int(thin)
        f.attrs["discard"] = int(discard)
        f.attrs["chain_shape"] = json.dumps(list(chain.shape))
        f.attrs["log_prob_shape"] = json.dumps(list(log_prob.shape))

    print("=" * 100)
    print("[save] MCMC samples and log-probabilities saved")
    print(f"[save] HDF5 file       : {out_path}")
    print(f"[save] chain shape     : {chain.shape}  = (nsteps, nwalkers, ndim)")
    print(f"[save] log_prob shape  : {log_prob.shape}  = (nsteps, nwalkers)")
    print(f"[save] flat shape      : {flat_chain.shape}  with thin={thin}, discard={discard}")
    print(f"[save] parameters      : {parameter_names}")
    print("=" * 100)
    return out_path


def _resolve_emcee_moves(root: dict):
    """
    Build an optional weighted list of emcee moves from `mcmc.emcee.moves`.

    This is the user-facing way to change the typical jump size of a walker:
      - StretchMove: parameter `a` controls stretch amplitude (larger = wider jumps).
      - DEMove: `gamma0` controls differential-evolution jump scale; `sigma` adds noise.
      - DESnookerMove: `gammas` controls snooker jump scale.

    If `mcmc.emcee.moves` is absent or empty, run_emcee_hybrid keeps its robust default:
    50/50 DEMove + DESnookerMove.
    """
    mcmc = root.get("mcmc", {}) or {}
    moves_cfg = (mcmc.get("emcee", {}) or {}).get("moves", None)
    if not moves_cfg:
        return None
    if not isinstance(moves_cfg, (list, tuple)):
        raise ValueError("mcmc.emcee.moves must be a list of move definitions.")

    moves = []
    print("[MCMC] Custom emcee moves requested from YAML:")
    for item in moves_cfg:
        if not isinstance(item, dict):
            raise ValueError("Each entry in mcmc.emcee.moves must be a dictionary.")
        name = str(item.get("name", "")).strip().lower()
        weight = float(item.get("weight", 1.0))

        if name in ("stretch", "stretchmove"):
            a = float(item.get("a", 2.0))
            move = emcee.moves.StretchMove(a=a)
            print(f"  - StretchMove(weight={weight:g}, a={a:g})")
        elif name in ("de", "demove"):
            kwargs = {}
            if item.get("sigma", None) is not None:
                kwargs["sigma"] = float(item.get("sigma"))
            if item.get("gamma0", None) is not None:
                kwargs["gamma0"] = float(item.get("gamma0"))
            move = emcee.moves.DEMove(**kwargs)
            print(f"  - DEMove(weight={weight:g}, kwargs={kwargs})")
        elif name in ("desnooker", "desnookermove"):
            kwargs = {}
            if item.get("gammas", None) is not None:
                kwargs["gammas"] = float(item.get("gammas"))
            move = emcee.moves.DESnookerMove(**kwargs)
            print(f"  - DESnookerMove(weight={weight:g}, kwargs={kwargs})")
        elif name in ("walk", "walkmove"):
            s = int(item.get("s", 2))
            move = emcee.moves.WalkMove(s=s)
            print(f"  - WalkMove(weight={weight:g}, s={s})")
        else:
            raise ValueError(
                f"Unknown emcee move name {name!r}. Supported: stretch, de, desnooker, walk."
            )
        moves.append((move, weight))

    return moves


# =============================================================================
# DERIVED QUANTITIES & CHAIN AUGMENTATION
# =============================================================================

def derive_physical_from_chain(a, la0, m0, h, k, p, q, t_ref):
    """
    Convert the new non-singular state to gauge-chosen classical quantities for
    human-readable tables only. The likelihood never performs this inverse conversion.

    New definitions:
        h=e sin(Ω+ω), k=e cos(Ω+ω)
        p=sin(i/2) sinΩ, q=sin(i/2) cosΩ

    Gauges at singular limits:
        i=0  -> choose Ω=0
        e=0  -> choose ω=0 and varpi=Ω, preserving λ and therefore the orbit.
    """
    e = np.hypot(h, k)
    r = np.sqrt(np.clip(p*p + q*q, 0.0, 1.0))
    inc = 2.0 * np.arcsin(r)

    Omega_raw = np.arctan2(p, q)
    Omega = np.where(r > 1e-15, Omega_raw, 0.0)

    varpi_raw = np.arctan2(h, k)
    varpi = np.where(e > 1e-15, varpi_raw, Omega)
    theta0 = wrap_2pi(varpi - Omega)  # argument of periapsis ω
    omega = wrap_2pi(Omega)           # ascending node Ω

    M0 = wrap_2pi(la0 - varpi)
    n = mean_motion(a, m0)
    t0 = t_ref - (M0 / n)
    return M0, t0, e, inc, omega, theta0


def augment_chain_with_physical(flat: np.ndarray, t_ref: float):
    """
    Augment a 7-D chain with 6 derived physical quantities.

    Returns
    ───────
    flat_aug  : (Nsamples, 13) array.
    labels_aug: list of 13 column names.
    """
    a, la0, m0, h, k, p, q = flat.T
    M0, t0, e, inc, omg, th0 = derive_physical_from_chain(
        a, la0, m0, h, k, p, q, t_ref
    )
    flat_aug   = np.column_stack([a, la0, m0, h, k, p, q, M0, t0, e, inc, omg, th0])
    labels_aug = ["a", "la0", "m0", "h", "k", "p", "q", "M0", "t0", "e", "i", "omega", "theta0"]
    return flat_aug, labels_aug


def credible_interval(
    arr: np.ndarray, q: Sequence[float] = (16, 50, 84)
) -> Tuple[float, float, float, float, float]:
    """
    Median and 16th/84th percentile credible interval.

    Returns (median, minus_err, plus_err, q16, q84).
    """
    p16, p50, p84 = np.percentile(arr, q)
    return p50, (p50 - p16), (p84 - p50), p16, p84












# =============================================================================
# NATIVE-PIXEL PREDICTION & STACKING
# =============================================================================

def _predict_pixel_track_native(theta, ts, *, size, scale, t_ref):
    """
    Predict one native-pixel track directly from the EqOE state.

    No conversion to classical Ω, ω, M0 or t0 occurs here.
    """
    a, la0, m0, h, k, p, q = map(float, theta)
    north_sky, east_sky = _equinoctial_sky_tracks(
        np.array([a]), np.array([la0]), np.array([m0]),
        np.array([h]), np.array([k]), np.array([p]), np.array([q]),
        np.asarray(ts, dtype=float), t_ref=t_ref,
    )
    north_sky = north_sky[0]
    east_sky = east_sky[0]
    cx = cy = (size - 1) / 2.0
    return (-east_sky) * scale + cx, north_sky * scale + cy


def _predict_tracks_native_chunk(thetas_chunk, ts, *, size, scale, t_ref):
    """Vectorized direct-EqOE native pixel tracks for a walker chunk."""
    th = np.asarray(thetas_chunk, float)
    a, la0, m0, h, k, p, q = th.T
    north_sky, east_sky = _equinoctial_sky_tracks(
        a, la0, m0, h, k, p, q, np.asarray(ts, dtype=float), t_ref=t_ref
    )
    # helper returns (W,K); this public chunk helper returns (K,W)
    x_pix = (-east_sky).T * scale + ((size - 1) / 2.0)
    y_pix = north_sky.T * scale + ((size - 1) / 2.0)
    return x_pix, y_pix

















# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def _run_mcmc_from_yaml_impl(yaml_path: str):
    """
    End-to-end MCMC runner driven by a YAML configuration file.

    Pipeline
    ────────
    1.  Read YAML → Params object.
    2.  For each instrument block: load images, optional profiles, and the time vector.
        PACO maps are loaded only when method is "paco".
    3.  Combine all time vectors → resolve global t_ref.
    4.  Resolve priors, eccentricity prior, and bounds.
    5.  Resolve likelihood mode without backend-specific forcing.
    6.  Build θ_init from the grid search or manual centre and draw the walker cloud.
    7.  Optionally add the fp dimension (flux mode with sample_fp=true).
    8.  Build Instrument objects and lp_kwargs.
    9.  Run burn-in + production MCMC via run_emcee_hybrid.
    10. Extract thinned flattened chain and log-probability arrays.
    11. Print diagnostics (acceptance fraction, IAT).
    12. Save chain and log-probability outputs.

    Parameters
    ──────────
    yaml_path : str
        Path to the YAML configuration file.

    Returns
    ───────
    sampler : emcee.EnsembleSampler
    flat    : (Nsamples, D) posterior samples.
    """
    # ── (1) Load YAML ──
    params = Params.read(yaml_path)
    root = _prepare_runtime_config_from_files(yaml_path, params._params, verbose=True)
    params._params = root
    params.work_dir = str(Path(yaml_path).expanduser().resolve().parent)
    base_root = dict(root)

    # ── (2) Load per-instrument data ──

    instruments_cfg = root.get("instruments", None)
    if not isinstance(instruments_cfg, (list, tuple)):
        raise ValueError("`instruments` must be a list of instrument configs.")

    per_inst_data: list = []
    all_ts:        list = []
    background_noise_cfg = _resolve_background_noise(root)

    for inst_cfg in instruments_cfg:
        if not isinstance(inst_cfg, dict):
            raise ValueError("Each entry in `instruments` must be a dict.")

        # Merge global config with instrument-specific overrides.
        tmp_root = dict(base_root)
        tmp_root.update(inst_cfg)
        params._params = tmp_root

        # Decide which photometry backend this instrument uses.
        photometry_method = str(getattr(params, "method", "convolve") or "convolve").lower()
        if photometry_method not in ("convolve", "aperture", "paco"):
            raise ValueError(
                f"Instrument {inst_cfg.get('name', 'INST')!r}: method must be "
                "'convolve', 'aperture', or 'paco'."
            )

        ts_i = params.get_ts(use_p_prev=True)
        size_i = params.n
        scale_i = params.scale
        upsampling_factor = getattr(params, "upsampling_factor", 1)
        r_mask = getattr(params, "r_mask", None)
        r_mask_ext = getattr(params, "r_mask_ext", None)
        fwhm = float(getattr(params, "fwhm", _get(tmp_root, "fwhm", 3.0)))
        inst_name = inst_cfg.get("name", inst_cfg.get("instrument_name", "INST"))
        radial_recal_cfg = _resolve_radial_noise_recalibration(base_root, inst_cfg)

        images_up = None
        images_native = None
        paco_alpha_maps = None
        paco_var_alpha_maps = None
        paco_interpolator = "none"
        paco_oversampling = 1
        local_bkg_maps = None
        local_noise_maps = None

        if photometry_method == "paco":
            # PACO likelihood evaluation is deliberately lazy and independent:
            # do not load classical images, profiles, or local-ring products.
            paco_interpolator = _normalize_paco_interpolator(
                str(inst_cfg.get("paco_interpolator", "none"))
            )
            paco_oversampling = _effective_paco_oversampling(
                paco_interpolator, int(inst_cfg.get("paco_oversampling", 1))
            )
            paco_alpha_maps, paco_var_alpha_maps = _load_paco_maps(
                params,
                paco_maps_dir=str(inst_cfg.get("paco_maps_dir", "wpca_alpha_var")),
                alpha_pattern=str(inst_cfg.get("paco_alpha_pattern", "alpha_hat_epoch_{k}_north.fits")),
                var_alpha_pattern=str(inst_cfg.get("paco_var_alpha_pattern", "var_alpha_epoch_{k}_north.fits")),
                interpolator=paco_interpolator,
                oversampling=paco_oversampling,
                radial_noise_recalibration=radial_recal_cfg,
            )
            xgrid, bkg, noise = _make_placeholder_profiles(len(ts_i), size_i)

            # PACO likelihood evaluation only needs alpha_hat and var_alpha.
            images_native = None
        else:
            use_local_ring = background_noise_cfg["mode"] == "local_aperture_ring"
            # Radial noise recalibration needs an explicit 2-D sigma field in
            # local-ring mode.  Therefore it automatically builds/reuses the
            # existing local-map cache even if ordinary likelihood evaluation
            # would otherwise have used on-the-fly local rings.
            use_precomputed_local_maps = (
                use_local_ring
                and (
                    (
                        str(background_noise_cfg.get("local_map_mode", "on_the_fly")).lower()
                        == "precompute_cache"
                        and bool(background_noise_cfg.get("precompute_maps", False))
                    )
                    or radial_recal_cfg["mode"] != "none"
                    or _noise_floor_build_value(root) == 0.0
                )
            )
            profile_dir = params.get_path("profile_dir")
            images_native = _load_native_images(
                params, suffix=str(inst_cfg.get("native_images_suffix", "_preprocessed"))
            )
            if photometry_method == "convolve":
                images_up = _load_convolved_images(params)

            # The classical likelihood uses local aperture-ring statistics only.
            # Placeholder arrays keep the Instrument container uniform; they are
            # not used to estimate background or noise.
            xgrid, bkg, noise = _make_placeholder_profiles(
                len(ts_i),
                size_i,
            )

            if use_precomputed_local_maps:
                local_bkg_maps, local_noise_maps = _load_or_build_local_ring_maps(
                    profile_dir=profile_dir,
                    photometry_method=photometry_method,
                    size=size_i,
                    upsampling_factor=upsampling_factor,
                    fwhm=fwhm,
                    images_up=images_up,
                    images_native=images_native,
                    cfg=background_noise_cfg,
                    noise_floor=_noise_floor_build_value(root),
                )

            if radial_recal_cfg["mode"] != "none":
                xgrid, bkg, noise, local_noise_maps = _recalibrate_classical_noise(
                    photometry_method=photometry_method,
                    images_up=images_up,
                    images_native=images_native,
                    size=int(size_i),
                    upsampling_factor=float(upsampling_factor),
                    fwhm=float(fwhm),
                    r_mask=r_mask,
                    xgrid=xgrid,
                    bkg=bkg,
                    noise=noise,
                    local_bkg_maps=local_bkg_maps,
                    local_noise_maps=local_noise_maps,
                    cfg=radial_recal_cfg,
                    profile_dir=profile_dir,
                )

        per_inst_data.append(dict(
            name=inst_name,
            size=size_i,
            scale=scale_i,
            upsampling_factor=upsampling_factor,
            fwhm=fwhm,
            r_mask=r_mask,
            r_mask_ext=r_mask_ext,
            ts=np.asarray(ts_i, float),
            photometry_method=photometry_method,
            images_up=images_up,
            images_native=images_native,
            paco_alpha_maps=paco_alpha_maps,
            paco_var_alpha_maps=paco_var_alpha_maps,
            paco_interpolator=paco_interpolator,
            paco_oversampling=paco_oversampling,
            xgrid=xgrid,
            bkg=bkg,
            noise=noise,
            local_bkg_maps=local_bkg_maps,
            local_noise_maps=local_noise_maps,
        ))
        all_ts.append(np.asarray(ts_i, float))

    # Restore the global config in Params.
    params._params = base_root

    ts_global = np.concatenate(all_ts) if all_ts else np.array([], float)

    # ── (3 / 4) Global knobs and priors ──
    weighting = _resolve_weighting(root)

    priors = root.get("priors", {}) or {}
    la0_bounds = tuple(map(float, _get(priors, "la0_bounds", (0.0, 2.0 * np.pi))))
    e_max      = float(_get(priors, "e_max", 0.95))
    orbit_direction = _normalise_orbit_direction(_get(priors, "orbit_direction", "any"))
    print(f"[priors] orbit_direction = {orbit_direction}")
    a_bounds, m0_bounds = _resolve_bounds(params, priors)

    t_ref = _resolve_tref(params, root, ts_global)
    params._params["t_ref"] = float(t_ref)
    init_mode = _resolve_init_mode(root)
    multistart_cfg = _resolve_multistart(root)
    mconf     = _resolve_mcmc(root)

    # ── (5/6) Build θ_init and draw walker cloud ──
    if init_mode == "init_search":
        init_search_h5 = os.path.join(
            params.get_path("values_dir"),
            str((root.get("init_search", {}) or {}).get("output_h5", "mcmc_init_search.h5")),
        )
        theta_centres_init_search, meta_init_search = _build_theta_centres_from_ranked_h5(
            init_search_h5,
            t_ref=t_ref,
            n_centres=1,
            min_snr=None,
            a_bounds=a_bounds,
            m0_bounds=m0_bounds,
            la0_bounds=la0_bounds,
            e_max=e_max,
            orbit_direction=orbit_direction,
        )
        theta_init = np.asarray(theta_centres_init_search[0], dtype=float)
        print("[init] mode='init_search' — θ_init taken from best ranked init-search solution")
        if meta_init_search and meta_init_search[0].get("snr") is not None:
            print(f"[init] init-search best SNR = {meta_init_search[0]['snr']:.6f}")
    else:
        theta_init = _resolve_init_vector(root, priors, m0_bounds)
        print("[init] mode='manual' — using YAML 'init'")

    theta_init = _clip_theta_init_to_bounds(
        theta_init,
        a_bounds=a_bounds,
        m0_bounds=m0_bounds,
        la0_bounds=la0_bounds,
        e_max=e_max,
    )

    spread = _resolve_spread(root)
    print(f"[init] θ_init = {theta_init!r}  (a, λ0, m0, h, k, p, q)")
    print(f"[init] spreads = {spread}")

    if bool(multistart_cfg["enabled"]):
        ranked_h5 = _resolve_results_h5_from_source(
            params,
            root,
            init_mode=init_mode,
            multistart_cfg=multistart_cfg,
        )

        if ranked_h5 is None:
            theta_centres = [np.asarray(theta_init, dtype=float)]
            centres_meta = [dict(rank=1, snr=None, source="manual")]
        else:
            theta_centres, centres_meta = _build_theta_centres_from_ranked_h5(
                ranked_h5,
                t_ref=t_ref,
                n_centres=multistart_cfg["n_starts"],
                min_snr=multistart_cfg["min_snr"],
                a_bounds=a_bounds,
                m0_bounds=m0_bounds,
                la0_bounds=la0_bounds,
                e_max=e_max,
                orbit_direction=orbit_direction,
                track_context=per_inst_data,
                detrack=multistart_cfg["detrack"],
                detrack_dmax_px=multistart_cfg["detrack_dmax_px"],
            )

        print("[init] multi-start initialisation enabled")
        print(f"[init] source          : {multistart_cfg['source']}")
        print(f"[init] n_starts        : {len(theta_centres)}")
        print(f"[init] ratios          : {multistart_cfg['ratios']}")
        if ranked_h5 is not None:
            print(f"[init] ranked file     : {ranked_h5}")
        print(f"[init] detector-track distinctness: {multistart_cfg['detrack']} "
              f"(dmax threshold = {multistart_cfg['detrack_dmax_px']:.3f} px)")
        print("[init] Selected MCMC starting centres BEFORE walker jitter:")
        for idx_c, (theta_c, meta_c) in enumerate(zip(theta_centres, centres_meta), start=1):
            row_c = meta_c.get("classical_row", None)
            snr_text = "n/a" if meta_c.get("snr") is None else f"{meta_c['snr']:.6f}"
            track_text = meta_c.get("min_track_dmax_px", None)
            track_text = "first centre" if track_text is None else f"{track_text:.3f} px"
            print(f"[init]   centre {idx_c}: rank={meta_c.get('rank')} | SNR={snr_text} | nearest retained-track dmax={track_text}")
            if row_c is not None:
                a_c, e_c, t0_c, m0_c, Omega_c, inc_c, argperi_c = map(float, row_c[:7])
                n_c = 2.0 * np.pi * np.sqrt(m0_c / a_c**3)
                M0_c = wrap_2pi(-n_c * t0_c)
                print(
                    "[init]       physical: "
                    f"a={a_c:.6f} AU, e={e_c:.6f}, i={np.degrees(inc_c):.6f} deg, "
                    f"M0={np.degrees(M0_c):.6f} deg, theta0={np.degrees(argperi_c)%360.0:.6f} deg, "
                    f"omega={np.degrees(Omega_c)%360.0:.6f} deg, m0={m0_c:.6f} Msun"
                )
            print(
                "[init]       nonsingular: "
                f"a={theta_c[0]:.6f}, lambda_ref={theta_c[1]:.6f}, m0={theta_c[2]:.6f}, "
                f"h={theta_c[3]:+.6f}, k={theta_c[4]:+.6f}, p={theta_c[5]:+.6f}, q={theta_c[6]:+.6f}"
            )

        p0 = draw_walkers_multistart(
            nwalkers=mconf["nwalkers"],
            theta_centres=theta_centres,
            ratios=multistart_cfg["ratios"],
            a_bounds=a_bounds,
            m0_bounds=m0_bounds,
            e_max=e_max,
            la0_bounds=la0_bounds,
            spread=spread,
            orbit_direction=orbit_direction,
        )
    else:
        p0 = draw_walkers_around_theta_init(
            nwalkers=mconf["nwalkers"],
            theta_init=theta_init,
            a_bounds=a_bounds,
            m0_bounds=m0_bounds,
            e_max=e_max,
            la0_bounds=la0_bounds,
            spread=spread,
            orbit_direction=orbit_direction,
        )

    # ── (7) Optional fp dimension ──
    if mconf["likelihood_mode"] == "flux" and bool(mconf["sample_fp"]):
        lo_, hi_ = mconf["fp_bounds"]
        fp0  = float(mconf["fp_init"])
        s_   = float(mconf["fp_spread"])
        if mconf["fp_prior"] == "loguniform":
            log_fp0  = np.log(max(fp0, lo_))
            fp_cloud = np.exp(log_fp0 + s_ * np.random.randn(p0.shape[0]))
        else:
            fp_cloud = fp0 * (1.0 + s_ * np.random.randn(p0.shape[0]))
        fp_cloud = np.clip(fp_cloud, lo_, hi_)
        p0 = np.column_stack([p0, fp_cloud])
        print(f"[init] fp sampling enabled — fp_init={fp0:.3g}, spread={s_:.3g}, prior='{mconf['fp_prior']}'")

    # ── (8) Build Instrument objects ──
    instruments: list = []
    for d in per_inst_data:
        instruments.append(Instrument(
            name=d["name"],
            size=d["size"],
            scale=d["scale"],
            upsampling_factor=d["upsampling_factor"],
            fwhm=d["fwhm"],
            r_mask=d["r_mask"],
            r_mask_ext=d["r_mask_ext"],
            t_ref=t_ref,
            ts=d["ts"],
            photometry_method=d["photometry_method"],
            images_up=d["images_up"],
            images_native=d["images_native"],
            paco_alpha_maps=d["paco_alpha_maps"],
            paco_var_alpha_maps=d["paco_var_alpha_maps"],
            paco_interpolator=d["paco_interpolator"],
            paco_oversampling=d["paco_oversampling"],
            xgrid=d["xgrid"],
            bkg=d["bkg"],
            noise=d["noise"],
            bgnoise_cfg=background_noise_cfg,
            local_bkg_maps=d.get("local_bkg_maps", None),
            local_noise_maps=d.get("local_noise_maps", None),
        ))

    # Resolve one common numerical sigma floor from the loaded noise products.
    noise_floor = _resolve_noise_floor(
        root,
        instruments,
        verbose=True,
    )

    lp_kwargs = dict(
        instruments=instruments,
        a_bounds=a_bounds,
        m0_bounds=m0_bounds,
        noise_floor=noise_floor,
        e_max=e_max,
        orbit_direction=orbit_direction,
        weighting=weighting,
        likelihood_mode=mconf["likelihood_mode"],
        fp_bounds=mconf["fp_bounds"],
        fp_prior=mconf["fp_prior"],
    )


    # ── (9) Parallelization + RNG seed ──
    par  = _resolve_parallel(root)
    seed = root.get("random_seed", None)
    if seed is not None:
        np.random.seed(int(seed))

    # ── (10) Run MCMC ──
    _print_section("MCMC CONFIGURATION")
    print(f"  sampler              : {mconf['sampler']}")
    print(f"  likelihood           : {mconf['likelihood_mode']}")
    print(f"  walkers              : {mconf['nwalkers']}")
    print(f"  burn-in              : {mconf['burnin']:,}")
    print(f"  production           : {mconf['nsteps']:,}")
    print(f"  thinning             : {mconf['thin']}")
    print()

    if mconf["sampler"] == "emcee":
        moves = _resolve_emcee_moves(root)
        sampler = run_emcee_hybrid(
            p0, lp_kwargs,
            nsteps=mconf["nsteps"],
            burnin=mconf["burnin"],
            moves=moves,
            max_workers=par["max_workers"],
            chunk_size=par["chunk_size"],
            progress=mconf["progress"],
        )
    else:
        sampler = run_reddemcee_adaptive(
            p0,
            lp_kwargs,
            nsteps=mconf["nsteps"],
            burnin=mconf["burnin"],
            cfg=mconf["reddemcee"],
            progress=mconf["progress"],
        )

    # ── (11) Extract thinned flattened products and save chain ──
    flat = sampler.get_chain(discard=0, thin=mconf["thin"], flat=True)

    save_cfg = root.get("outputs", {}) or {}
    if _bool_from_config(save_cfg.get("save_chain", True), True):
        save_mcmc_chains_and_logprob(
            params=params, sampler=sampler, thin=mconf["thin"], discard=0,
            filename=str(save_cfg.get("chain_filename", "mcmc_chain_and_logprob.h5")),
        )

    # ── (12) Diagnostics ──
    acc = float(np.mean(sampler.acceptance_fraction))
    print(f"[MCMC] mean acceptance fraction = {acc:.3f}")
    try:
        tau   = sampler.get_autocorr_time(discard=0, thin=mconf["thin"], tol=0)
        names = ["a", "la0", "m0", "h", "k", "p", "q"] + (["fp"] if flat.shape[1] == 8 else [])
        print("[MCMC] integrated autocorrelation time (steps):")
        for name_, t_ in zip(names, tau):
            print(f"  {name_:>4s}: {t_:.1f}")
    except Exception as err:
        print("[MCMC] IAT not reliable:", err)

    # ── (13) Finish ──
    print()
    print("-" * 110)
    print("MCMC complete. Chain products are ready in values/.")
    print("-" * 110)
    print()

    return sampler, flat



def run_mcmc_from_yaml(yaml_path: str):
    """
    Public entry point with automatic console logging.

    The implementation is kept in `_run_mcmc_from_yaml_impl`; this wrapper only
    installs the optional tee logger requested by the YAML, then restores the
    terminal state even if the run fails.
    """
    old_stdout, old_stderr, log_file, log_path = _setup_console_logging_from_yaml(yaml_path)
    try:
        return _run_mcmc_from_yaml_impl(yaml_path)
    finally:
        _restore_console_logging(old_stdout, old_stderr, log_file, log_path)


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run KStacker MCMC from a YAML config file.")
    ap.add_argument("yaml", help="Path to the YAML parameter file.")
    args = ap.parse_args()

    sampler, flat = run_mcmc_from_yaml(args.yaml)
    print("[MCMC] done. flat chain shape:", flat.shape)
