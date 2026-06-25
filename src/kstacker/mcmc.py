# mcmc.py
# =======
#
# End-to-end MCMC driver for direct-imaging orbit inference.
#
# This module is configured entirely by a YAML file (see parameters_harmoni_mcmc.yml)
# and relies on a project-specific `Params` helper for I/O.
#
# ─────────────────────────────────────────────────────────────────────────────
# HIGH-LEVEL OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
#
# The module estimates the posterior distribution over a 7-D (or 8-D) orbital
# parameter vector by running an affine-invariant ensemble MCMC sampler (emcee).
#
# Orbital parameterization
# ─────────────────────────
# We use a *non-singular* parameterization to avoid coordinate singularities:
#
#   θ = (a, λ0, m0, h, k, p, q)
#
#   a   : semi-major axis [AU]
#   λ0  : mean longitude at reference epoch [rad] — avoids the M0/ω singularity
#   m0  : stellar mass [solar masses] — controls the orbital period
#   h   = e · sin(Ω + ω)             — eccentricity/orientation component
#   k   = e · cos(Ω + ω)             — eccentricity/orientation component
#   p   = sin(i/2) · cos(Ω - ω)      — inclination/orientation component
#   q   = sin(i/2) · sin(Ω - ω)      — inclination/orientation component
#
# IMPORTANT ANGLE CONVENTION USED EVERYWHERE IN THIS FILE
# ───────────────────────────────────────────────────────
#   omega  = Ω = longitude of the ascending node [rad]
#   theta0 = ω = argument of periapsis [rad]
#
# From (h, k) you can recover:  e = √(h²+k²),  Ω+ω = atan2(h, k)
# From (p, q) you can recover:  i = 2·arcsin(√(p²+q²)),  Ω−ω = atan2(q, p)
#
# Optional 8th parameter:
#   fp  : planet flux [native units]  — only when likelihood_mode = "flux"
#
# ─────────────────────────────────────────────────────────────────────────────
# PHOTOMETRY BACKENDS
# ─────────────────────────────────────────────────────────────────────────────
#
# For each walker θ and each epoch k, we predict the planet's on-sky position
# in the sky frame (North, West), then convert it to native image pixels with
# x = West and y = North before extracting a photometric scalar from the
# images.  Three backends are supported:
#
# 1) "convolve"  (default)
#    ─────────────────────
#    Uses *upsampled* images (preprocessed by a matched-filter / convolution).
#    Reads the single nearest pixel at the predicted sub-pixel location.
#    Background and noise are interpolated from a radial profile at radius r_k.
#    This is the legacy KStacker approach.
#
# 2) "aperture"
#    ──────────
#    Uses *native* (non-upsampled) preprocessed images.
#    Performs circular aperture photometry with radius = fwhm [native pixels].
#    The aperture sum replaces the single-pixel readout from the "convolve" mode.
#
# 3) "snr_map"  
#    ─────────────────
#    Uses pre-computed SNR maps (one per epoch), where each pixel already
#    contains an SNR value produced by the preprocessing pipeline.
#    The photometric scalar at (x_k, y_k) is simply the SNR at that pixel.
#    No background subtraction, no noise division — the map is already in SNR
#    units, so the combination across epochs is a direct quadratic sum (or
#    simple sum, depending on `weighting`).
#
#    IMPORTANT: when this backend is active, `likelihood_mode` is automatically
#    forced to "snr" (flux mode is meaningless with pre-computed SNR values).
#    The `flux` likelihood requires raw photometric counts and their per-pixel
#    noise; that information is no longer available once the SNR map is formed.
#
# ─────────────────────────────────────────────────────────────────────────────
# LIKELIHOOD MODES
# ─────────────────────────────────────────────────────────────────────────────
#
# "snr"  (default / recommended with SNR maps)
#    The log-likelihood surrogate is:
#        log L ≈ snr_scale × SNR_total(θ)
#    where SNR_total is the multi-epoch, multi-instrument combined SNR.
#    SNR is combined across epochs using either simple or inverse-variance
#    weighting.
#
# "flux"  (requires raw images + profiles; INCOMPATIBLE with snr_map backend)
#    A Gaussian model in which the planet contributes flux fp to each epoch.
#    The log-likelihood is:
#        log L = -0.5 × (fp² S1 - 2 fp S2)
#    where S1 = Σ_k (1/σ_k²) and S2 = Σ_k ((F_k - bg_k)/σ_k²) are
#    *sufficient statistics* that can be precomputed from the images.
#    fp can be sampled as an 8th parameter (sample_fp=true) or held fixed.
#
# ─────────────────────────────────────────────────────────────────────────────
# MULTI-INSTRUMENT SUPPORT
# ─────────────────────────────────────────────────────────────────────────────
#
# Each instrument entry in the YAML defines its own:
#   - time sampling (possibly different epochs),
#   - image set (images_dir),
#   - photometry backend (method, fwhm, snr_maps_suffix),
#   - geometry (n, fwhm, resol),
#   - masks (r_mask, r_mask_ext).
#
# All instruments share the *same* orbital parameters θ.  The log-posterior
# is the sum of all per-instrument contributions:
#
#   log p(θ | data) = log prior(θ) + Σ_i log L_i(θ | data_i)
#
# ─────────────────────────────────────────────────────────────────────────────
# MASK HANDLING (IWA / OWA)
# ─────────────────────────────────────────────────────────────────────────────
#
# r_mask (IWA) and r_mask_ext (OWA) are *soft* masks: they zero-out the
# photometric contribution of epochs where the planet falls inside IWA or
# outside OWA. This allows the MCMC to explore orbits that transit masked
# regions at some epochs, only the unmasked epochs contribute to the
# likelihood

from __future__ import annotations

import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple, Union, Mapping, Any
from dataclasses import dataclass
import h5py

import numpy as np
import emcee
from concurrent.futures import ProcessPoolExecutor
import kepler          # must expose kepler.solve(M, e) with broadcasting
import matplotlib.pyplot as plt
import corner
from matplotlib.patches import Circle
from scipy.ndimage import shift as ndi_shift
from scipy.stats import beta as scipy_beta
from astropy.io import fits
from photutils.aperture import CircularAperture, aperture_photometry

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

    # Human-readable label used in plot filenames and console output.
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
    Not used by the "aperture" or "snr_map" backends, but stored for
    completeness.
    """

    fwhm: Optional[float]
    """
    PSF full-width at half-maximum [native pixels].
    Required by the "aperture" backend (aperture radius = fwhm).
    Also used by the GLRT off-tracks to set the default offset scale.
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
    Valid values: "convolve", "aperture", "snr_map".

    "convolve":
        Read the nearest pixel in the upsampled image (images_up).
        Requires images_up to be provided.

    "aperture":
        Perform circular aperture photometry on native images (images_native)
        with radius = fwhm.  Requires images_native and fwhm.

    "snr_map":
        Read the pre-computed SNR value at the predicted pixel in snr_maps.
        No background subtraction or noise division — the SNR is already
        encoded in the map.  Requires snr_maps.
        Forces likelihood_mode = "snr" globally.
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
    Also used by the coadd, orbit overlay, and logprob-map plots regardless
    of the photometry backend.
    """

    snr_maps: Optional[np.ndarray]
    """
    Pre-computed per-epoch SNR maps, shape (K, size, size).
    Each pixel [k, y, x] contains the local SNR value at that location for
    epoch k, as produced by the preprocessing pipeline.

    Required when photometry_method = "snr_map".
    May be None for other backends.

    IMPORTANT:
      * The SNR values in these maps must already account for background and
        noise — they must be dimensionless signal-to-noise ratios, NOT raw
        photon counts.
      * Using snr_maps forces likelihood_mode to "snr".  The "flux" mode
        is incompatible because it needs raw counts and per-pixel noise.
      * The maps must be in native-pixel resolution (same shape as
        images_native), not upsampled.
    """

    # -------------------------------------------------------------------------
    # Radial profiles (for "convolve" and "aperture" backends only)
    # -------------------------------------------------------------------------
    xgrid: np.ndarray
    """
    1-D radius grid [native pixels], shape (R,), on which background and
    noise profiles are tabulated.
    Used to interpolate background and noise at any predicted planet radius.
    """

    bkg: np.ndarray
    """
    Per-epoch azimuthally averaged background profile, shape (K, R).
    bkg[k, r_idx] = mean background at radius xgrid[r_idx] for epoch k.
    """

    noise: np.ndarray
    """
    Per-epoch azimuthally averaged noise (σ) profile, shape (K, R).
    noise[k, r_idx] = σ at radius xgrid[r_idx] for epoch k.
    Used for inverse-variance weighting and as the denominator in SNR.
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


def _resolve_weighting(yaml_root: dict) -> str:
    """
    Determine the epoch-combination weighting scheme from the YAML.

    Priority:
      1. The `weighting` key (string: "invvar" or "simple").
      2. Legacy boolean `invvar_weight` (True → "invvar", False → "simple").

    "invvar":
        Inverse-variance weighting. Each epoch k is weighted by w_k = 1/σ_k².
        This gives more weight to epochs observed under good conditions and
        is the recommended setting.

    "simple":
        Uniform weighting. All epochs contribute equally.
        Equivalent to summing the per-epoch scalars.
    """
    w = yaml_root.get("weighting", None)
    if isinstance(w, str):
        w = w.strip().lower()
        if w in ("invvar", "simple"):
            return w
    invvar = bool(yaml_root.get("invvar_weight", False))
    return "invvar" if invvar else "simple"


def _resolve_bounds(
    params: Params, priors: dict
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Resolve box prior supports for semi-major axis (a) and stellar mass (m0).

    Uses the YAML `priors.a_bounds` / `priors.m0_bounds` when available,
    falling back to the ranges stored inside the Params object.
    """
    a_bounds = _get(priors, "a_bounds", None)
    m0_bounds = _get(priors, "m0_bounds", None)
    if a_bounds is None:
        a_bounds = (params["a"]["min"], params["a"]["max"])
    if m0_bounds is None:
        m0_bounds = (params["m0"]["min"], params["m0"]["max"])
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
    Decide how to initialise the MCMC walkers.

    "manual"     : Use the explicit `init` + `init_spread` values from the YAML.
    "bruteforce" : Seed the walkers from the best solution in the brute-force
                   grid (res_grid.h5 inside values_dir).

    Use "bruteforce" when you have already run the K-Stacker brute-force
    search and want the MCMC to refine the best solutions.
    Use "manual" when you have a good prior guess for the orbital parameters.
    """
    mode = str(root.get("init_mode", "manual") or "manual").lower()
    if mode not in ("manual", "bruteforce"):
        raise ValueError("init_mode must be 'manual' or 'bruteforce'.")
    return mode


def _build_init_from_bruteforce(
    params: "Params",
    root: dict,
    t_ref: float,
) -> np.ndarray:
    """
    Build the MCMC initial parameter vector from the brute-force best solution.

    Reads the HDF5 file `values_dir/res_grid.h5` and extracts the top-ranked
    solution (assumed sorted by decreasing SNR).

    The brute-force solution is in the classical parameterization
    (a, e, t0, m0, omega, i, theta0), with this convention:

        omega  = Ω = longitude of the ascending node
        theta0 = ω = argument of periapsis

    The function converts it to the non-singular sampling space
    (a, λ0, m0, h, k, p, q).

    Conversion formulae
    ───────────────────
        n     = 2π √(m0 / a³)
        M0    = wrap_2π(n · (t_ref − t0))
        λ0    = wrap_2π(M0 + Ω + ω)
        h     = e · sin(Ω + ω)
        k     = e · cos(Ω + ω)
        p     = sin(i/2) · cos(Ω − ω)
        q     = sin(i/2) · sin(Ω − ω)

    The brute-force solution is read directly in the same geometric
    convention used throughout this file, so no detector-axis swap or angle
    remapping is applied before the conversion.
    """
    grid_dir = params.get_path("values_dir")
    h5_path = os.path.join(grid_dir, "res_grid.h5")

    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"init_mode='bruteforce' but brute-force file not found: {h5_path}"
        )

    with h5py.File(h5_path, "r") as f:
        if "Best solutions" not in f:
            raise KeyError(
                f"Dataset 'Best solutions' not found in {h5_path}."
            )
        # Dataset is assumed sorted by decreasing SNR.
        data = np.asarray(f["Best solutions"][:], dtype=float)

    if data.ndim != 2 or data.shape[1] < 7:
        raise ValueError(
            f"'Best solutions' in {h5_path} has shape {data.shape}; "
            "expected (N, ≥7) with columns (a, e, t0, m0, omega=Ω, i, theta0=ω)."
        )

    # Optional SNR filtering.
    snr_best = None
    if data.shape[1] >= 8:
        snr = data[:, -1]
        min_snr = root.get("min_snr", None)
        if min_snr is not None:
            min_snr = float(min_snr)
            valid = snr >= min_snr
            if not np.any(valid):
                raise RuntimeError(
                    f"No brute-force solutions pass min_snr={min_snr}."
                )
            data = data[valid]
            snr = snr[valid]
        snr_best = float(snr[0])

    if data.shape[0] == 0:
        raise RuntimeError("Brute-force 'Best solutions' dataset is empty.")

    row = data[0]
    a, e, t0, m0, omega, inc, theta0 = row[:7]


    if not (np.isfinite(a) and np.isfinite(e) and np.isfinite(t0) and np.isfinite(m0)):
        raise RuntimeError("Best brute-force solution has non-finite parameters.")
    if a <= 0.0 or m0 <= 0.0:
        raise RuntimeError("Best brute-force solution has non-physical a or m0.")

    # Convert classical → non-singular parameterization.
    n = 2.0 * np.pi * np.sqrt(m0 / (a ** 3))
    M0 = wrap_2pi(n * (t_ref - t0))
    la0 = wrap_2pi(M0 + omega + theta0)

    # With the convention used by the simulator and by the projection matrix:
    #   omega  = Ω = longitude of ascending node
    #   theta0 = ω = argument of periapsis
    # The non-singular variables encode Ω+ω and Ω−ω.
    w_sum = omega + theta0
    Delta = omega - theta0
    h = e * np.sin(w_sum)
    k = e * np.cos(w_sum)
    sin_i_2 = np.sin(0.5 * inc)
    p = sin_i_2 * np.cos(Delta)
    q = sin_i_2 * np.sin(Delta)

    theta_init = np.array([a, la0, m0, h, k, p, q], dtype=float)

    print(f"[init] bruteforce: using best solution from '{h5_path}'")
    if snr_best is not None:
        print(f"[init] best solution SNR = {snr_best:.3f}")
    print(f"[init] θ_init (bruteforce) = {theta_init!r}  (a, λ0, m0, h, k, p, q)")
    return theta_init


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

    The spreads should be small (e.g., 0.01–0.05) for "bruteforce" init so
    that walkers start near the best solution.  For "manual" init they can
    be larger to explore the prior.
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
    Parse the `mcmc` section of the YAML into a typed configuration dict.

    Key fields
    ──────────
    nwalkers         : number of emcee walkers (must be ≥ 2 × ndim, even).
    burnin           : burn-in steps (discarded after sampler.reset()).
    nsteps           : production steps kept in the chain.
    thin             : thinning factor applied when extracting the flat chain.
    progress         : show tqdm progress bars.
    likelihood_mode  : "snr" or "flux" (see module docstring).
    sample_fp        : if True, fp is a sampled 8th parameter (flux mode only).
    fp_bounds        : [fp_min, fp_max] for fp prior / walker initialisation.
    fp_prior         : "uniform" or "loguniform".
    fp_init          : centre of the fp walker cloud.
    fp_spread        : width of the fp walker cloud (read from init_spread.fp).
    """
    mcmc = root.get("mcmc", {}) or {}
    return dict(
        nwalkers   =int(mcmc.get("nwalkers", 100)),
        burnin     =int(mcmc.get("burnin",   10000)),
        nsteps     =int(mcmc.get("nsteps",   100000)),
        thin       =int(mcmc.get("thin",     10)),
        progress   =bool(str(mcmc.get("progress", "yes")).lower() in ("1", "true", "yes", "y")),
        likelihood_mode=str(mcmc.get("likelihood_mode", "snr")).lower(),
        sample_fp  =bool(mcmc.get("sample_fp", False)),
        fp_bounds  =tuple(map(float, mcmc.get("fp_bounds", (1e-3, 1e1)))),
        fp_prior   =str(mcmc.get("fp_prior", "loguniform")).lower(),
        fp_init    =float(mcmc.get("fp_init", 1.0)),
        fp_spread  =float(_get(root.get("init_spread", {}), "fp", 0.2)),
    )


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
    Convert mean longitude at reference epoch (λ0) to mean anomaly (M0).

    Relations, with omega = Ω and theta0 = ω:
        Ω + ω = atan2(h, k)
        M0    = wrap_2π(λ0 − (Ω + ω))
    """
    w = np.arctan2(h, k)   # Ω + ω
    return wrap_2pi(la0 - w)


def M0_to_t0(
    M0: np.ndarray, a: np.ndarray, m0: np.ndarray, t_ref: float
) -> np.ndarray:
    """
    Convert mean anomaly at reference epoch (M0) to periastron epoch (t0).

    t0 = t_ref − M0 / n
    """
    n = mean_motion(a, m0)
    return t_ref - (M0 / n)


def compute_projection_matrices_from_hkpq(
    k: np.ndarray, h: np.ndarray, p: np.ndarray, q: np.ndarray
) -> np.ndarray:
    """
    Build 2×2 sky-projection matrices from non-singular elements (h, k, p, q).

    The matrix R maps (x_orb, y_orb) in the orbital plane [AU] to
    (North, West) in the sky plane [AU].  Native image pixels use the same
    convention as the plotting code: x = West, y = North.

    Derivation
    ──────────
        omega  = Ω = (atan2(h,k) + atan2(q,p)) / 2    (ascending node)
        theta0 = ω = (atan2(h,k) − atan2(q,p)) / 2    (argument of periapsis)
        cos(i) = 1 − 2(p²+q²)                         (inclination)

    Returns
    ───────
    Array of shape (W, 2, 2) — one 2×2 matrix per walker.
    """
    # Correct convention:
    #   omega  = Ω = longitude of the ascending node
    #   theta0 = ω = argument of periapsis
    # Therefore w = Ω+ω and Delta = Ω−ω.  The projection matrix below is
    # the standard sky projection for (North, West) and is converted to image
    # pixels later with x = West and y = North.
    w     = np.arctan2(h, k)   # Ω + ω
    Delta = np.arctan2(q, p)   # Ω − ω

    omega  = 0.5 * (w + Delta)    # Ω
    theta0 = 0.5 * (w - Delta)    # ω

    cos_omega,  sin_omega  = np.cos(omega),  np.sin(omega)
    cos_theta0, sin_theta0 = np.cos(theta0), np.sin(theta0)

    r2    = np.clip(p * p + q * q, 0.0, 1.0)   # sin²(i/2)
    cos_i = 1.0 - 2.0 * r2                      # cos(i) ∈ [−1, 1]

    # Build the 2×2 rotation matrix for each walker simultaneously using
    # numpy broadcasting; then move the walker axis to position 0.
    rot = np.array([
        [cos_omega * cos_theta0 - sin_omega * sin_theta0 * cos_i,
         -cos_omega * sin_theta0 - sin_omega * cos_theta0 * cos_i],
        [sin_omega * cos_theta0 + cos_omega * sin_theta0 * cos_i,
         -sin_omega * sin_theta0 + cos_omega * cos_theta0 * cos_i],
    ], dtype=np.float32)
    return np.rollaxis(rot, 2)   # (W, 2, 2)


# =============================================================================
# RADIAL PROFILE INTERPOLATION
# =============================================================================

def _interp_profiles_vectorized(
    xgrid: np.ndarray,
    profiles: np.ndarray,
    r: np.ndarray,
    k_idx_broadcast: np.ndarray,
) -> np.ndarray:
    """
    Vectorized linear interpolation of per-epoch radial profiles.

    For each walker w and epoch k, interpolates profiles[k, ·] at radius r[w,k].

    Parameters
    ──────────
    xgrid           : (R,) radius grid [native pixels].
    profiles        : (K, R) per-epoch profiles (background or noise).
    r               : (W, K) radii [native pixels] for each walker and epoch.
    k_idx_broadcast : (W, K) int array of epoch indices (broadcast).

    Returns
    ───────
    (W, K) array of interpolated values.
    """
    R  = xgrid.size
    i1 = np.searchsorted(xgrid, r, side="right")
    i1 = np.clip(i1, 1, R - 1)
    i0 = i1 - 1

    x0 = xgrid[i0]
    x1 = xgrid[i1]
    t  = (r - x0) / (x1 - x0)   # fractional position in [0, 1]

    P0 = profiles[(k_idx_broadcast, i0)]
    P1 = profiles[(k_idx_broadcast, i1)]
    return P0 + t * (P1 - P0)


# =============================================================================
# NATIVE IMAGE LOADERS
# =============================================================================

def _load_native_images(params: Params, suffix: str = "_preprocessed") -> np.ndarray:
    """
    Load native (non-upsampled) FITS images from the images_dir.

    File naming convention:
        images_dir / image_{k}{suffix}.fits   for k in [0, p_prev + p)

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
    nimg = params.p + params.p_prev
    imgs = []
    for k in range(nimg):
        fn = os.path.join(images_dir, f"image_{k}{suffix}.fits")
        im = fits.getdata(fn)
        imgs.append(im.astype("float32", copy=False))
    return np.asarray(imgs)


def _load_snr_maps(params: Params, suffix: str) -> np.ndarray:
    """
    Load pre-computed per-epoch SNR maps from the images_dir.

    The SNR maps must be in native-pixel resolution (same size as the
    preprocessed images) and contain dimensionless signal-to-noise ratios.

    File naming convention:
        images_dir / image_{k}{suffix}.fits   for k in [0, p_prev + p)

    Parameters
    ──────────
    params : Params
        Project helper with I/O paths.
    suffix : str
        Filename suffix before ".fits" (e.g. "_snr_map").
        Set via `snr_maps_suffix` in the instrument YAML block.

    Returns
    ───────
    (K, size, size) float32 array.

    Notes
    ─────
    - The pixel values must be SNR values, not raw counts or flux.
    - The maps are combined across epochs by the "snr_map" photometry
      backend using simple quadratic addition (invvar) or linear sum
      (simple weighting), without any further background subtraction.
    - These maps are distinct from `images_native` (which contain flux).
    """
    images_dir = params.get_path("images_dir")
    nimg = params.p + params.p_prev
    maps = []
    for k in range(nimg):
        fn = os.path.join(images_dir, f"image_{k}{suffix}.fits")
        m = fits.getdata(fn)
        maps.append(m.astype("float32", copy=False))
    return np.asarray(maps)   # (K, size, size)


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
# SNR CORE  (vectorized over walkers)
# =============================================================================

def snr_from_hkpq(
    theta: np.ndarray,
    *,
    ts: np.ndarray,
    images: np.ndarray,              # upsampled images for "convolve" mode
    data: dict,                      # {"x": xgrid, "bkg": bkg, "noise": noise}
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
    snr_maps: Optional[np.ndarray] = None,   
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-walker (signal, noise, SNR) across all epochs.

    For each walker θ:
      1. Predict the planet track (x_k, y_k) in native pixels for each epoch k.
      2. Extract a photometric scalar at each epoch using the chosen backend.
      3. Combine the per-epoch scalars into a single (signal, noise, SNR).

    Three photometry backends
    ─────────────────────────
    "convolve":
        Read the nearest pixel in the upsampled image at the predicted
        sub-pixel location.  Background and noise are interpolated from
        radial profiles.

    "aperture":
        Perform circular aperture photometry on the native image with
        radius = fwhm [native pixels].  Background and noise come from
        radial profiles.

    "snr_map": 
        Read the pre-computed SNR value at the nearest native pixel.
        No background subtraction, no noise division.
        The "signal" returned is the SNR value itself; "noise" is 1.
        Epoch combination:
            invvar: combined_snr = √(Σ snr_k²)    (quadratic addition)
            simple: combined_snr = Σ snr_k / √K   (mean-normalised sum)

    Mask handling (soft)
    ────────────────────
    Epochs where the predicted position falls inside IWA (r ≤ r_mask) or
    outside OWA (r ≥ r_mask_ext) are zeroed out via the `validpix` mask.
    They contribute neither signal nor noise.  The walker is not rejected
    outright; only its masked epochs are silenced.

    Invalid walkers
    ───────────────
    Walkers with non-physical parameters (e ≥ 1, a ≤ 0, m0 ≤ 0, p²+q² > 1)
    are detected early and assigned SNR = 0 to keep the log-posterior finite.

    Parameters
    ──────────
    theta              : (W, 7) walker array in (a, λ0, m0, h, k, p, q).
    ts                 : (K,) observation times [years].
    images             : (K, H_up, W_up) upsampled images for "convolve".
    data               : dict with keys "x", "bkg", "noise" (radial profiles).
    size               : native image size [pixels].
    scale              : AU → native pixels conversion.
    upsampling_factor  : for "convolve" pixel-coordinate conversion.
    t_ref              : reference epoch [years].
    r_mask             : IWA [native pixels], or None.
    r_mask_ext         : OWA [native pixels], or None.
    noise_floor        : numerical floor on noise to avoid division by zero.
    weighting          : "invvar" or "simple".
    photometry_method  : "convolve", "aperture", or "snr_map".
    images_native      : (K, size, size) for "aperture" backend.
    fwhm               : aperture radius for "aperture" backend.
    snr_maps           : (K, size, size) pre-computed SNR maps for "snr_map".

    Returns
    ───────
    (signal, noise, snr) : three (W,) arrays.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.ndim == 1:
        theta = theta[None, :]
    if theta.shape[1] != 7:
        raise ValueError("theta must have 7 columns: [a, la0, m0, h, k, p, q]")

    a, la0, m0, h, k, p, q = theta.T
    W = theta.shape[0]

    # ── Validity check: physical constraints on the orbital parameters ──
    r2 = p * p + q * q
    ok = np.isfinite(theta).all(axis=1) & (r2 <= 1.0) & (a > 0.0) & (m0 > 0.0)
    e  = np.sqrt(np.maximum(0.0, h * h + k * k))
    ok &= (e < 1.0)   # strictly elliptic

    if not np.any(ok):
        signal = np.zeros(W, dtype=float)
        noise  = np.full(W, float(noise_floor), dtype=float)
        return signal, noise, signal / noise

    # Work only on the valid subset to avoid propagating NaNs.
    idx = np.where(ok)[0]
    a_ok, la0_ok, m0_ok, h_ok, k_ok, p_ok, q_ok, e_ok = (
        a[idx], la0[idx], m0[idx], h[idx], k[idx], p[idx], q[idx], e[idx]
    )

    # ── Keplerian orbit: λ0 → M0 → t0 → eccentric anomaly E ──
    w_ok  = np.arctan2(h_ok, k_ok)                            # Ω + ω
    M0_ok = (la0_ok - w_ok + 2.0 * np.pi) % (2.0 * np.pi)
    n_ok  = 2.0 * np.pi * np.sqrt(m0_ok / (a_ok ** 3))       # mean motion
    t0_ok = t_ref - (M0_ok / n_ok)                            # periastron epoch

    ts = np.asarray(ts, dtype=float)
    K  = int(len(ts))

    # Validate image dimensions against K before the Kepler solve.
    if photometry_method == "convolve":
        if images.shape[0] != K:
            raise ValueError("images (upsampled) must have first dimension K == len(ts).")
    elif photometry_method == "aperture":
        if images_native is None:
            raise ValueError("aperture mode requires images_native=(K,size,size).")
        if fwhm is None:
            raise ValueError("aperture mode requires a finite fwhm (native px).")
        Kn, H, Wn = images_native.shape
        if Kn != K or H != size or Wn != size:
            raise ValueError("images_native must be (K, size, size).")
    elif photometry_method == "snr_map":
        # SNR-map mode: validate the pre-computed maps.
        if snr_maps is None:
            raise ValueError("snr_map mode requires snr_maps=(K,size,size).")
        Km, Hm, Wm = snr_maps.shape
        if Km != K or Hm != size or Wm != size:
            raise ValueError("snr_maps must be (K, size, size).")

    M = n_ok[:, None] * (ts[None, :] - t0_ok[:, None])        # mean anomaly (W_ok, K)
    E = kepler.solve(M, e_ok[:, None])                         # eccentric anomaly (W_ok, K)

    # ── Orbital-plane coordinates in AU ──
    cosfac = np.sqrt(np.maximum(0.0, 1.0 - e_ok * e_ok))
    x_orb  = a_ok[:, None] * (np.cos(E) - e_ok[:, None])      # (W_ok, K)
    y_orb  = a_ok[:, None] * (cosfac[:, None] * np.sin(E))    # (W_ok, K)

    # ── Sky projection (AU → native pixels) ──
    rot = compute_projection_matrices_from_hkpq(k_ok, h_ok, p_ok, q_ok).astype(np.float32)
    r00 = rot[:, 0, 0][:, None]; r01 = rot[:, 0, 1][:, None]
    r10 = rot[:, 1, 0][:, None]; r11 = rot[:, 1, 1][:, None]
    north_sky = x_orb * r00 + y_orb * r01
    west_sky  = x_orb * r10 + y_orb * r11

    cx = cy = size // 2
    x_pix = west_sky * scale + cx    # (W_ok, K) native pixel x = West
    y_pix = north_sky * scale + cy   # (W_ok, K) native pixel y = North
    r     = np.hypot(x_pix - cx, y_pix - cy)   # (W_ok, K) radius [native px]

    # ── Build the validpix mask (soft IWA/OWA + image bounds) ──
    validpix = np.ones((idx.size, K), dtype=bool)
    if r_mask is not None:
        validpix &= (r > r_mask)        # IWA: exclude pixels inside the mask
    if r_mask_ext is not None:
        validpix &= (r < r_mask_ext)    # OWA: exclude pixels outside the mask

    k_idx = np.broadcast_to(np.arange(K), (idx.size, K))

    # ── Extract photometric scalars per epoch ──
    if photometry_method == "convolve":
        # Convert native pixel coords to upsampled grid coords.
        x_up = np.floor(x_pix * upsampling_factor - 0.5).astype(np.int32)
        y_up = np.floor(y_pix * upsampling_factor - 0.5).astype(np.int32)
        validpix &= (
            (0 <= x_up) & (x_up < images.shape[2]) &
            (0 <= y_up) & (y_up < images.shape[1])
        )
        x_up  = np.clip(x_up, 0, images.shape[2] - 1)
        y_up  = np.clip(y_up, 0, images.shape[1] - 1)
        flux  = images[(k_idx, y_up, x_up)].astype(np.float32)

    elif photometry_method == "aperture":
        # Aperture photometry on native images.
        validpix &= (x_pix >= 0) & (x_pix < size) & (y_pix >= 0) & (y_pix < size)
        flux = np.zeros((idx.size, K), dtype=np.float32)
        r_ap = float(fwhm)
        for ii in range(idx.size):
            for kk in range(K):
                if validpix[ii, kk]:
                    flux[ii, kk] = _aperture_sum_native(
                        images_native[kk],
                        float(x_pix[ii, kk]), float(y_pix[ii, kk]),
                        r_ap,
                    )

    elif photometry_method == "snr_map":
        # ── SNR-map backend ──────────────────────────────────────────────────
        # Round to nearest native pixel (no upsampling needed).
        x_nat = np.round(x_pix).astype(np.int32)
        y_nat = np.round(y_pix).astype(np.int32)
        # Exclude pixels outside the native image bounds.
        validpix &= (
            (0 <= x_nat) & (x_nat < size) &
            (0 <= y_nat) & (y_nat < size)
        )
        x_nat = np.clip(x_nat, 0, size - 1)
        y_nat = np.clip(y_nat, 0, size - 1)
        # Read SNR value directly from the pre-computed map.
        # snr_maps is indexed as [epoch, y, x] (row = y, col = x).
        flux = snr_maps[(k_idx, y_nat, x_nat)].astype(np.float32)
        # ─────────────────────────────────────────────────────────────────────

    else:
        raise ValueError(f"Unknown photometry_method: {photometry_method!r}")

    # Zero out masked / out-of-bounds epochs.
    flux[~validpix] = 0.0

    # ── Combine epochs into a single scalar per walker ──
    if photometry_method == "snr_map":
        # In SNR-map mode the values are already in SNR units.
        # No background subtraction; no noise division needed.
        if weighting == "simple":
            # Simple average: divide by √K so the combined SNR has the right
            # scaling (equivalent to matched-filter for equal-noise epochs).
            n_valid   = np.maximum(validpix.sum(axis=1).astype(float), 1.0)
            signal_ok = np.sum(flux, axis=1) / np.sqrt(n_valid)
            noise_ok  = np.ones(idx.size, dtype=float)
        else:   # "invvar"
            # Quadratic addition: combined_snr = √(Σ snr_k²).
            # This is exact for independent Gaussian SNR measurements.
            signal_ok = np.sqrt(np.sum(flux * flux, axis=1))
            noise_ok  = np.ones(idx.size, dtype=float)

    else:
        # ── Standard flux-based combination ──
        xgrid = np.asarray(data["x"],   dtype=float)
        bkg   = np.asarray(data["bkg"], dtype=np.float32)
        noise = np.asarray(data["noise"],dtype=np.float32)

        # Interpolate background and noise at the predicted radii.
        bg  = _interp_profiles_vectorized(xgrid, bkg,  r, k_idx)
        sig = _interp_profiles_vectorized(xgrid, noise, r, k_idx)
        bg[~validpix]  = 0.0
        sig[~validpix] = 0.0

        y = flux - bg   # background-subtracted flux

        if weighting == "simple":
            signal_ok = np.sum(y, axis=1)
            noise_ok  = np.sqrt(np.sum(sig * sig, axis=1))
        elif weighting == "invvar":
            wgt = np.zeros_like(sig, dtype=np.float32)
            np.divide(1.0, sig * sig, out=wgt, where=(sig > 0))
            den       = np.sum(wgt, axis=1)
            signal_ok = np.divide(np.sum(y * wgt, axis=1), den,
                                  out=np.zeros_like(den), where=(den > 0))
            noise_ok  = np.sqrt(np.divide(1.0, den,
                                          out=np.zeros_like(den), where=(den > 0)))
        else:
            raise ValueError("weighting must be 'simple' or 'invvar'.")

    # Apply noise floor.
    noise_ok = np.where((noise_ok > 0) & np.isfinite(noise_ok),
                        noise_ok, float(noise_floor))
    snr_ok   = signal_ok / noise_ok

    # Scatter valid results back to the full (W,) arrays.
    signal = np.zeros(W, dtype=float)
    noise  = np.full(W, float(noise_floor), dtype=float)
    snr    = np.zeros(W, dtype=float)
    signal[idx] = signal_ok
    noise[idx]  = noise_ok
    snr[idx]    = snr_ok
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

    For "snr_map" instruments, signal_i is already the combined SNR across
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
            snr_maps=inst.snr_maps,
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


# =============================================================================
# FLUX SUFFICIENT STATISTICS  (for "flux" likelihood only)
# =============================================================================

def flux_sufficient_stats_from_hkpq(
    theta: np.ndarray,
    *,
    ts: np.ndarray,
    images: np.ndarray,
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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute sufficient statistics (S1, S2) for the Gaussian flux likelihood.

    The flux likelihood is:
        log L(fp | θ) = -0.5 (fp² S1 − 2 fp S2) + const

    where:
        S1 = Σ_k  w_k           w_k = 1 / σ_k²
        S2 = Σ_k  w_k ỹ_k      ỹ_k = (F_k − bg_k) / 1     (background-subtracted)

    These statistics are computed once per θ, and the optimal fp (MAP estimate)
    is simply fp* = S2 / S1.

    This function is only valid for "convolve" and "aperture" backends.
    Do NOT call it when photometry_method = "snr_map" — the SNR-map backend
    does not have the raw photometric counts needed to form S1 and S2.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.ndim == 1:
        theta = theta[None, :]
    if theta.shape[1] != 7:
        raise ValueError("theta must have 7 columns: [a, la0, m0, h, k, p, q]")

    a, la0, m0, h, k, p, q = theta.T
    W = theta.shape[0]

    r2  = p * p + q * q
    ok  = np.isfinite(theta).all(axis=1) & (r2 <= 1.0) & (a > 0.0) & (m0 > 0.0)
    e   = np.sqrt(np.maximum(0.0, h * h + k * k))
    ok &= (e < 1.0)

    # Default: invalid walkers get S1 = 1/noise_floor² (finite weight sum) and S2 = 0.
    S1 = np.full(W, 1.0 / float(noise_floor) ** 2, dtype=float)
    S2 = np.zeros(W, dtype=float)
    if not np.any(ok):
        return S1, S2

    idx = np.where(ok)[0]
    a_ok, la0_ok, m0_ok, h_ok, k_ok, p_ok, q_ok, e_ok = (
        a[idx], la0[idx], m0[idx], h[idx], k[idx], p[idx], q[idx], e[idx]
    )

    w_ok  = np.arctan2(h_ok, k_ok)
    M0_ok = (la0_ok - w_ok + 2.0 * np.pi) % (2.0 * np.pi)
    n_ok  = 2.0 * np.pi * np.sqrt(m0_ok / (a_ok ** 3))
    t0_ok = t_ref - (M0_ok / n_ok)

    ts = np.asarray(ts, dtype=float)
    K  = int(len(ts))

    if photometry_method == "convolve":
        if images.shape[0] != K:
            raise ValueError("images must have first dimension K == len(ts).")
    else:
        if images_native is None:
            raise ValueError("aperture mode requires images_native=(K,size,size).")
        if fwhm is None:
            raise ValueError("aperture mode requires a finite fwhm (native px).")
        Kn, H, Wn = images_native.shape
        if Kn != K or H != size or Wn != size:
            raise ValueError("images_native must be (K, size, size).")

    M = n_ok[:, None] * (ts[None, :] - t0_ok[:, None])
    E = kepler.solve(M, e_ok[:, None])

    cosfac = np.sqrt(np.maximum(0.0, 1.0 - e_ok * e_ok))
    x_orb  = a_ok[:, None] * (np.cos(E) - e_ok[:, None])
    y_orb  = a_ok[:, None] * (cosfac[:, None] * np.sin(E))

    rot  = compute_projection_matrices_from_hkpq(k_ok, h_ok, p_ok, q_ok).astype(np.float32)
    r00  = rot[:, 0, 0][:, None]; r01 = rot[:, 0, 1][:, None]
    r10  = rot[:, 1, 0][:, None]; r11 = rot[:, 1, 1][:, None]
    north_sky = x_orb * r00 + y_orb * r01
    west_sky  = x_orb * r10 + y_orb * r11

    cx = cy = size // 2
    x_pix = west_sky * scale + cx
    y_pix = north_sky * scale + cy
    r     = np.hypot(x_pix - cx, y_pix - cy)

    validpix = np.ones((idx.size, K), dtype=bool)
    if r_mask is not None:
        validpix &= (r > r_mask)
    if r_mask_ext is not None:
        validpix &= (r < r_mask_ext)

    k_idx = np.broadcast_to(np.arange(K), (idx.size, K))

    if photometry_method == "convolve":
        x_up  = np.floor(x_pix * upsampling_factor - 0.5).astype(np.int32)
        y_up  = np.floor(y_pix * upsampling_factor - 0.5).astype(np.int32)
        validpix &= (
            (0 <= x_up) & (x_up < images.shape[2]) &
            (0 <= y_up) & (y_up < images.shape[1])
        )
        x_up  = np.clip(x_up, 0, images.shape[2] - 1)
        y_up  = np.clip(y_up, 0, images.shape[1] - 1)
        flux  = images[(k_idx, y_up, x_up)].astype(np.float32)
    else:
        validpix &= (x_pix >= 0) & (x_pix < size) & (y_pix >= 0) & (y_pix < size)
        flux = np.zeros((idx.size, K), dtype=np.float32)
        r_ap = float(fwhm)
        for ii in range(idx.size):
            for kk in range(K):
                if validpix[ii, kk]:
                    flux[ii, kk] = _aperture_sum_native(
                        images_native[kk],
                        float(x_pix[ii, kk]), float(y_pix[ii, kk]),
                        r_ap,
                    )
    flux[~validpix] = 0.0

    xgrid = np.asarray(data["x"],    dtype=float)
    bkg   = np.asarray(data["bkg"],  dtype=np.float32)
    noise = np.asarray(data["noise"],dtype=np.float32)
    bg    = _interp_profiles_vectorized(xgrid, bkg,  r, k_idx)
    sig   = _interp_profiles_vectorized(xgrid, noise, r, k_idx)
    bg[~validpix]  = 0.0
    sig[~validpix] = 0.0

    ytilde = flux - bg

    w = np.zeros_like(sig, dtype=np.float32)
    np.divide(1.0, sig * sig, out=w, where=(sig > 0))

    S1_ok = np.sum(w, axis=1)
    S2_ok = np.sum(w * ytilde, axis=1)

    S1_ok = np.where((S1_ok > 0) & np.isfinite(S1_ok),
                     S1_ok, 1.0 / float(noise_floor) ** 2)
    S2_ok = np.where(np.isfinite(S2_ok), S2_ok, 0.0)

    S1[idx] = S1_ok
    S2[idx] = S2_ok
    return S1, S2


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

    IMPORTANT: this function cannot be called when any instrument uses the
    "snr_map" backend.  SNR maps do not carry the raw photometric information
    (flux and per-pixel noise) needed to compute the Gaussian sufficient
    statistics.  If you have SNR-map instruments, you must use
    likelihood_mode = "snr".
    """
    if not instruments:
        raise ValueError("flux_sufficient_stats_multi_from_hkpq requires at least one Instrument.")

    # Guard against accidental use with snr_map instruments.
    for inst in instruments:
        if inst.photometry_method == "snr_map":
            raise ValueError(
                f"Instrument '{inst.name}' uses photometry_method='snr_map', which is "
                "incompatible with the 'flux' likelihood.  Set likelihood_mode='snr' "
                "in the YAML mcmc section when using pre-computed SNR maps."
            )

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
        )
        S1_total += S1_i
        S2_total += S2_i

    S1_total = np.where((S1_total > 0) & np.isfinite(S1_total),
                        S1_total, 1.0 / float(noise_floor) ** 2)
    S2_total = np.where(np.isfinite(S2_total), S2_total, 0.0)

    if scalar_input:
        return np.array([S1_total[0]]), np.array([S2_total[0]])
    return S1_total, S2_total


# =============================================================================
# PRIORS
# =============================================================================

def log_prior_hkpq(
    theta: np.ndarray,
    *,
    a_bounds: Tuple[float, float],
    m0_bounds: Tuple[float, float],
    la0_bounds: Tuple[float, float] = (0.0, 2.0 * np.pi),
    ecc_prior: str = "kipping",
    ecc_beta_a: float = 0.867,
    ecc_beta_b: float = 3.03,
    e_max: float = 0.95,
    isotropic_orientation: bool = True,
    pq_prior: str = "none",
) -> np.ndarray:
    """
    Vectorized log-prior for the orbital parameters (a, λ0, m0, h, k, p, q).

    Box priors
    ──────────
    - a   ∈ a_bounds   (uniform)
    - m0  ∈ m0_bounds  (uniform)
    - λ0  ∈ la0_bounds (uniform, usually [0, 2π])

    Eccentricity prior in (h, k) space (three choices)
    ───────────────────────────────────────────────────
    "kipping":
        Beta(a=0.867, b=3.03) prior on e, transformed to (h,k) via
        the Jacobian pdf_{h,k}(e) = Beta(e) / (2π e).
        This prior is peaked near e=0 and has a soft upper cutoff at e_max.
        Recommended for physically motivated eccentricity inference.

    "uniform_e":
        Uniform in e ∈ [0, e_max], with Jacobian in (h,k):
        log p = −log(e_max) − log(2π) − log(e).
        Penalises near-circular orbits but treats all non-zero eccentricities
        equally.

    "uniform_disk":
        Uniform density inside the disk e ≤ e_max in the (h,k) plane:
        log p = −log(π e_max²).
        Equivalent to a uniform prior in (h,k) restricted to the disk.

    Orientation prior in (p, q) space
    ──────────────────────────────────
    isotropic_orientation = True  (recommended):
        Uniform area over the unit disk p²+q² ≤ 1.
        Corresponds to a uniform distribution over orbital inclinations and
        position angles on the sphere.
        log p = −log(π).

    isotropic_orientation = False (discouraged):
        Flat prior on the square [−1,1]² in (p,q).
        Not physically motivated; provided for legacy compatibility only.

    Orbital rotation direction prior (pq_prior)
    ─────────────────────────────────────────
    Only applicable when isotropic_orientation = True.
    Restricts the orbital inclination range to favor a specific rotation direction:
      - "none"          : No additional constraint (default).
      - "clockwise"     : p² + q² < 0.5  → corresponds to 0 < i < π/2 (clockwise orbits)
      - "counterclockwise": p² + q² > 0.5  → corresponds to π/2 < i < π (counter-clockwise orbits)
    Note: p² + q² = sin²(i/2), so the threshold 0.5 corresponds to i = π/2.
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
        (a_bounds[0]   <= a)   & (a   <= a_bounds[1])   &
        (m0_bounds[0]  <= m0)  & (m0  <= m0_bounds[1])  &
        (la0_bounds[0] <= la0) & (la0 <= la0_bounds[1]) &
        (a > 0.0) & (m0 > 0.0)
    )

    if isotropic_orientation:
        r2     = p * p + q * q
        valid &= (r2 <= 1.0)
        # Apply rotation direction prior
        if pq_prior == "clockwise":
            valid &= (r2 < 0.5)
            logp_pq = np.log(2) - np.log(np.pi)
        elif pq_prior == "counterclockwise":
            valid &= (r2 > 0.5)
            logp_pq = np.log(2) - np.log(np.pi)
        elif pq_prior == "none":
            logp_pq = -np.log(np.pi)
        else:
            raise ValueError("pq_prior must be 'none', 'clockwise', or 'counterclockwise'.")
    else:
        valid &= (np.abs(p) <= 1.0) & (np.abs(q) <= 1.0)
        logp_pq = -np.log(4.0)
        # pq_prior is incompatible with isotropic_orientation=False
        if pq_prior != "none":
            raise ValueError(
                "pq_prior requires isotropic_orientation=True. "
                "Set isotropic_orientation: true in the YAML."
            )

    e      = np.sqrt(np.maximum(0.0, h * h + k * k))
    valid &= (e <= e_max)

    if ecc_prior == "uniform_disk":
        logp_hk = -np.log(np.pi * e_max ** 2)
        logp    = np.full(W, logp_pq + logp_hk, dtype=float)

    elif ecc_prior == "kipping":
        eps           = 1e-12
        e_clip        = np.where(e < eps, eps, e)
        log_beta_pdf  = scipy_beta.logpdf(e_clip, ecc_beta_a, ecc_beta_b)
        logp          = log_beta_pdf - np.log(2.0 * np.pi) - np.log(e_clip)
        logp          = logp + logp_pq

    elif ecc_prior == "uniform_e":
        eps    = 1e-12
        e_clip = np.where(e < eps, eps, e)
        logp   = -np.log(e_max) - np.log(2.0 * np.pi) - np.log(e_clip)
        logp   = logp + logp_pq

    else:
        raise ValueError("ecc_prior must be 'kipping', 'uniform_disk', or 'uniform_e'.")

    logp = np.where(valid, logp, -np.inf)
    if scalar_input:
        return float(logp[0])
    return logp


def log_prior_fp(
    fp: np.ndarray | float,
    bounds: tuple[float, float],
    prior: str = "loguniform",
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

def log_probability(
    theta: np.ndarray,
    *,
    instruments: Sequence[Instrument],
    a_bounds: Tuple[float, float],
    m0_bounds: Tuple[float, float],
    noise_floor: float = 1.0,
    ecc_prior: str = "kipping",
    ecc_beta_a: float = 0.867,
    ecc_beta_b: float = 3.03,
    e_max: float = 0.95,
    orientation_isotropic: bool = True,
    pq_prior: str = "none",
    snr_scale: float = 1.0,
    weighting: str = "invvar",
    likelihood_mode: str = "snr",
    fp_bounds: tuple[float, float] = (1e-3, 1e1),
    fp_prior: str = "loguniform",
) -> np.ndarray | float:
    """
    Multi-instrument log-posterior for the planetary orbit (and optional flux).

    log p(θ | data) = log prior(θ) + log L(θ | data)

    The likelihood can be:

    "snr" mode:
        log L ≈ snr_scale × SNR_total(θ)
        where SNR_total is computed by `snr_multi_from_hkpq`.
        Works with all three photometry backends ("convolve", "aperture",
        "snr_map").  Recommended default.

    "flux" mode:
        log L = Σ_k [-0.5 (fp² S1 − 2 fp S2)]
        where (S1, S2) are sufficient statistics from
        `flux_sufficient_stats_multi_from_hkpq`.
        INCOMPATIBLE with the "snr_map" backend.
        Requires either D=8 (fp sampled) or D=7 (fp fixed at fp_bounds[0]).

    Parameters
    ──────────
    theta              : (D,) or (W, D) walker array.
                         D=7 for snr mode; D=7 or 8 for flux mode.
    instruments        : list of Instrument objects (at least one).
    a_bounds           : (a_min, a_max) [AU].
    m0_bounds          : (m0_min, m0_max) [solar masses].
    noise_floor        : numerical floor on σ to prevent division by zero.
    ecc_prior          : eccentricity prior type (see log_prior_hkpq).
    ecc_beta_a/b       : Beta prior hyperparameters (used when ecc_prior="kipping").
    e_max              : hard upper limit on eccentricity.
    orientation_isotropic: whether to use the isotropic orientation prior.
    pq_prior           : rotation direction prior: "none", "clockwise", or "counterclockwise".
    snr_scale          : multiplicative scaling of SNR in log-likelihood.
    weighting          : "invvar" or "simple".
    likelihood_mode    : "snr" or "flux".
    fp_bounds          : support for the flux parameter.
    fp_prior           : "uniform" or "loguniform".

    Returns
    ───────
    float or (W,) array of log-posterior values.
    """
    if instruments is None or len(instruments) == 0:
        raise ValueError("log_probability requires at least one Instrument.")

    th           = np.asarray(theta, dtype=float)
    scalar_input = (th.ndim == 1)
    if scalar_input:
        th = th[None, :]
    W, D = th.shape

    mode = str(likelihood_mode).lower()
    if mode not in ("snr", "flux"):
        raise ValueError("likelihood_mode must be 'snr' or 'flux'.")

    # ── Disentangle orbital parameters from optional flux parameter ──
    if mode == "snr":
        if D != 7:
            raise ValueError("In 'snr' mode, theta must have 7 parameters.")
        theta7 = th
        fp     = None

    else:   # "flux"
        if D == 8:
            theta7 = th[:, :7]
            fp     = th[:, 7]
        elif D == 7:
            # Fixed-flux mode: fp held at fp_bounds[0].
            theta7 = th
            fp     = np.full(W, fp_bounds[0], dtype=float)
        else:
            raise ValueError("In 'flux' mode, theta must have 7 or 8 parameters.")

    # ── Orbital prior ──
    lp_theta = log_prior_hkpq(
        theta7,
        a_bounds=a_bounds,
        m0_bounds=m0_bounds,
        la0_bounds=(0.0, 2.0 * np.pi),
        ecc_prior=ecc_prior,
        ecc_beta_a=ecc_beta_a,
        ecc_beta_b=ecc_beta_b,
        e_max=e_max,
        isotropic_orientation=orientation_isotropic,
        pq_prior=pq_prior,
    )

    # Early exit if all walkers are already out of prior support.
    if np.isscalar(lp_theta):
        if not np.isfinite(lp_theta):
            return -np.inf
    else:
        if not np.any(np.isfinite(lp_theta)):
            return lp_theta

    # ── Flux prior (flux mode only) ──
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

    # ── Likelihood ──
    if mode == "snr":
        _, _, snr = snr_multi_from_hkpq(
            theta7,
            instruments=instruments,
            noise_floor=noise_floor,
            weighting=weighting,
        )
        out = lp + snr_scale * snr

    else:   # "flux"
        S1, S2 = flux_sufficient_stats_multi_from_hkpq(
            theta7,
            instruments=instruments,
            noise_floor=noise_floor,
        )
        # Gaussian log-likelihood: -0.5 (fp² S1 − 2 fp S2)
        logL = -0.5 * (fp * fp * S1 - 2.0 * fp * S2)
        out  = lp + logL

    if scalar_input:
        return float(out[0])
    return out


# =============================================================================
# WALKER INITIALISATION
# =============================================================================

def draw_walkers_around_theta_init(
    nwalkers: int,
    theta_init: np.ndarray,
    *,
    a_bounds: Tuple[float, float],
    m0_bounds: Tuple[float, float],
    e_max: float = 0.95,
    la0_bounds: Tuple[float, float] = (0.0, 2.0 * np.pi),
    spread: dict = dict(a=0.02, la0=0.2, m0=0.0, hk=0.02, pq=0.02),
    max_tries: int = 10000,
) -> np.ndarray:
    """
    Initialise walkers near theta_init with Gaussian jitter.

    For each walker:
      - Jitter a, la0, m0, (h,k), (p,q) with Gaussian noise (scale = spread).
      - Project (h,k) back to the disk e ≤ e_max if needed.
      - Project (p,q) back to the unit disk p²+q² ≤ 1 if needed.
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
    radial profiles, masks, time vectors, and instrument metadata.  Keeping this
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
# CONSOLE LOGGING AND OUTPUT SAVE HELPERS
# =============================================================================

class Tee:
    """
    Minimal stdout/stderr tee.

    Every message written to the console is also written to a log file.  This is
    intentionally small and dependency-free so that long cluster jobs keep a
    faithful text record of what happened: configuration choices, warnings,
    diagnostics, output paths, and GLRT summaries.
    """
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
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
        root = params._params
        logging_cfg = root.get("logging", {}) or {}
        enabled = _bool_from_config(logging_cfg.get("save_console", True), True)
        if not enabled:
            return None, None, None, None

        values_dir = params.get_path("values_dir")
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
    or re-plotting without rerunning the expensive MCMC.
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
    Build an optional weighted list of emcee moves from `mcmc.moves`.

    This is the user-facing way to change the typical jump size of a walker:
      - StretchMove: parameter `a` controls stretch amplitude (larger = wider jumps).
      - DEMove: `gamma0` controls differential-evolution jump scale; `sigma` adds noise.
      - DESnookerMove: `gammas` controls snooker jump scale.

    If `mcmc.moves` is absent or empty, run_emcee_hybrid keeps its robust default:
    50/50 DEMove + DESnookerMove.
    """
    mcmc = root.get("mcmc", {}) or {}
    moves_cfg = mcmc.get("moves", None)
    if not moves_cfg:
        return None
    if not isinstance(moves_cfg, (list, tuple)):
        raise ValueError("mcmc.moves must be a list of move definitions.")

    moves = []
    print("[MCMC] Custom emcee moves requested from YAML:")
    for item in moves_cfg:
        if not isinstance(item, dict):
            raise ValueError("Each entry in mcmc.moves must be a dictionary.")
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
    Convert non-singular sampling parameters to conventional orbital elements.

    Returns (M0, t0, e, inc, omega, theta0) for each sample, with:
      omega  = Ω = longitude of the ascending node,
      theta0 = ω = argument of periapsis.
    """
    w      = np.arctan2(h, k)        # Ω + ω
    Delta  = np.arctan2(q, p)        # Ω − ω
    omega  = 0.5 * (w + Delta)      # Ω
    theta0 = 0.5 * (w - Delta)      # ω
    e      = np.hypot(h, k)
    sin_i_2 = np.sqrt(np.clip(p * p + q * q, 0.0, 1.0))
    inc    = 2.0 * np.arcsin(sin_i_2)
    M0     = wrap_2pi(la0 - w)
    n      = mean_motion(a, m0)
    t0     = t_ref - (M0 / n)
    return M0, t0, e, inc, omega, theta0


def augment_chain_with_physical(flat: np.ndarray, t_ref: float):
    """
    Augment a 7-D chain with 6 derived quantities for plotting.

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


# =============================================================================
# PLOTTING UTILITIES
# =============================================================================

def credible_interval(
    arr: np.ndarray, q: Sequence[float] = (16, 50, 84)
) -> Tuple[float, float, float, float, float]:
    """
    Median and 16th/84th percentile credible interval.

    Returns (median, minus_err, plus_err, q16, q84).
    """
    p16, p50, p84 = np.percentile(arr, q)
    return p50, (p50 - p16), (p84 - p50), p16, p84


def _safe_range(col: np.ndarray) -> Tuple[float, float]:
    """
    Robust (lo, hi) range for plotting that handles constant or NaN arrays.
    """
    lo = float(np.nanmin(col)); hi = float(np.nanmax(col))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return -1.0, 1.0
    if lo == hi:
        pad = 1e-6 if hi == 0.0 else 1e-3 * abs(hi)
        return lo - pad, hi + pad
    return lo, hi


def make_corner_plot(flat: np.ndarray, labels: List[str], out_path: str) -> str:
    """
    Render and save a corner plot with robust axis ranges.
    """
    ranges = [_safe_range(flat[:, j]) for j in range(flat.shape[1])]
    fig = corner.corner(
        flat, labels=labels, range=ranges, show_titles=True,
        quantiles=[0.16, 0.50, 0.84], title_fmt=".3f",
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_posterior_histograms(
    samples: np.ndarray,
    labels: List[str],
    *,
    bins: int = 30,
    figsize: Tuple[float, float] = (14, 8),
    density: bool = True,
    show_band: bool = True,
    savepath: Optional[str] = None,
) -> Optional[str]:
    """
    Grid of 1-D posterior histograms with median and 16/84% credible band.
    """
    samples = np.asarray(samples)
    nparams = samples.shape[1]
    ncols   = 3
    nrows   = int(np.ceil(nparams / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    for j, name in enumerate(labels):
        ax  = axes[j]
        col = samples[:, j]
        med, mlo, mhi, q16, q84 = credible_interval(col)
        lo, hi = _safe_range(col)
        edges  = np.linspace(lo, hi, bins + 1)
        ax.hist(col, bins=edges, density=density, alpha=0.6, edgecolor="none")
        ax.axvline(med, linestyle="--", linewidth=1.8)
        ax.axvline(q16, linestyle=":",  linewidth=1.2)
        ax.axvline(q84, linestyle=":",  linewidth=1.2)
        if show_band:
            ax.axvspan(q16, q84, alpha=0.15)
        ax.set_xlabel(name)
        ax.set_ylabel("density" if density else "count")
        ax.set_title(f"{name}: {med:.3g}  -{mlo:.3g}  +{mhi:.3g}")

    for k_ in range(nparams, len(axes)):
        axes[k_].axis("off")

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return savepath
    plt.close(fig)
    return None


def plot_augmented_posteriors(
    flat_chain: np.ndarray,
    t_ref: float,
    labels_to_plot: Optional[List[str]] = None,
    *,
    bins: int = 30,
    figsize: Tuple[float, float] = (14, 8),
    density: bool = True,
    show_band: bool = True,
    savepath: Optional[str] = None,
) -> Optional[str]:
    """
    Augment 7-D chain with derived quantities and plot selected posteriors.
    """
    flat_aug, labels_aug = augment_chain_with_physical(flat_chain, t_ref)
    if labels_to_plot is None:
        labels_to_plot = ["a", "la0", "m0", "h", "k", "p", "q",
                          "e", "i", "M0", "t0", "omega", "theta0"]
    idx_ = [labels_aug.index(n) for n in labels_to_plot]
    return plot_posterior_histograms(
        flat_aug[:, idx_], [labels_aug[i] for i in idx_],
        bins=bins, figsize=figsize, density=density,
        show_band=show_band, savepath=savepath,
    )


def plot_augmented_posteriors_with_optional_fp(
    flat_chain7: np.ndarray,
    t_ref: float,
    fp: Optional[np.ndarray] = None,
    labels_to_plot: Optional[List[str]] = None,
    *,
    bins: int = 30,
    figsize: Tuple[float, float] = (14, 8),
    density: bool = True,
    show_band: bool = True,
    savepath: Optional[str] = None,
) -> Optional[str]:
    """
    Augment 7-D chain and optionally append fp as an extra column.
    """
    flat_aug, labels_aug = augment_chain_with_physical(flat_chain7, t_ref)

    if fp is not None:
        fp = np.asarray(fp).reshape(-1, 1)
        flat_aug   = np.column_stack([flat_aug, fp])
        labels_aug = labels_aug + ["fp"]

    if labels_to_plot is None:
        labels_to_plot = ["a", "la0", "m0", "h", "k", "p", "q",
                          "e", "i", "M0", "t0", "omega", "theta0"]
        if "fp" in labels_aug:
            labels_to_plot = labels_to_plot + ["fp"]
    else:
        labels_to_plot = [lab for lab in labels_to_plot if lab in labels_aug]

    idx_ = [labels_aug.index(n) for n in labels_to_plot]
    return plot_posterior_histograms(
        flat_aug[:, idx_], [labels_aug[i] for i in idx_],
        bins=bins, figsize=figsize, density=density,
        show_band=show_band, savepath=savepath,
    )


# =============================================================================
# NATIVE-PIXEL PREDICTION & STACKING
# =============================================================================

def _predict_pixel_track_native(theta, ts, *, size, scale, t_ref):
    """
    Predict the planet's native-pixel track for a single 7-D orbital θ.

    Returns (x_pix, y_pix), each a (K,) array of native pixel coordinates.
    """
    a, la0, m0, h, k, p, q = map(float, theta)
    M0 = lambda_to_M0(la0, h, k)
    t0 = M0_to_t0(M0, a, m0, t_ref)
    x_orb, y_orb = orbit.positions_at_multiple_times(
        np.asarray(ts),
        np.array([[a, np.hypot(h, k), t0, m0]]),
    )
    x_orb = x_orb[:, 0]
    y_orb = y_orb[:, 0]
    proj  = compute_projection_matrices_from_hkpq(
        np.array([k], float), np.array([h], float),
        np.array([p], float), np.array([q], float),
    )[0]
    north_sky = x_orb * proj[0, 0] + y_orb * proj[0, 1]
    west_sky  = x_orb * proj[1, 0] + y_orb * proj[1, 1]
    cx = cy = size // 2
    return west_sky * scale + cx, north_sky * scale + cy


def _predict_tracks_native_chunk(thetas_chunk, ts, *, size, scale, t_ref):
    """
    Vectorized prediction of native pixel tracks for a chunk of walkers.

    Returns (x_pix, y_pix) of shape (K, Wc), with x = West and y = North.
    """
    th = np.asarray(thetas_chunk, float)
    a, la0, m0, h, k, p, q = th.T
    M0 = lambda_to_M0(la0, h, k)
    n  = mean_motion(a, m0)
    t0 = t_ref - (M0 / n)

    x_orb, y_orb = orbit.positions_at_multiple_times(
        np.asarray(ts),
        np.column_stack([a, np.hypot(h, k), t0, m0]),
    )   # (K, Wc)

    R     = compute_projection_matrices_from_hkpq(k, h, p, q)   # (Wc, 2, 2)
    north_sky = x_orb * R[None, :, 0, 0] + y_orb * R[None, :, 0, 1]
    west_sky  = x_orb * R[None, :, 1, 0] + y_orb * R[None, :, 1, 1]

    x_pix = west_sky * scale + (size // 2)
    y_pix = north_sky * scale + (size // 2)
    return x_pix, y_pix


def _invvar_weights_from_profiles_native(ts, x_pix, y_pix, data, *, size, eps=1e-12):
    """
    Per-epoch inverse-variance weights from radial noise profiles.

    Used by `stack_planet_from_posterior_native` for invvar coadd combination.
    """
    cx = cy = size / 2.0
    r      = np.hypot(x_pix - cx, y_pix - cy)
    xgrid  = np.asarray(data["x"])
    noise  = np.asarray(data["noise"])
    sig    = np.array([np.interp(r[k_], xgrid, noise[k_]) for k_ in range(len(ts))])
    w      = 1.0 / np.maximum(sig, eps) ** 2
    w[~np.isfinite(w)] = 0.0
    return w


def stack_planet_from_posterior_native(
    instruments: Sequence[Instrument],
    flat_samples: np.ndarray,
    *,
    theta: Optional[np.ndarray] = None,
    combine: str = "invvar",
    interpolation_order: int = 1,
    align_to: Union[str, Tuple[float, float]] = "first",
    sampler: Optional[emcee.EnsembleSampler] = None,
    thin: Optional[int] = None,
    discard: int = 0,
    logprob_kwargs: Optional[dict] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
    """
    Build a native coadded image per instrument using a single orbital θ.

    If theta is None, selects the MAP sample (highest log-posterior) from
    flat_samples.  Otherwise uses the provided theta directly.

    For each instrument:
      1. Predict the planet's native-pixel track over that instrument's epochs.
      2. Shift each epoch's native image so the planet lands at the anchor (x0, y0).
      3. Combine shifted frames using invvar or simple averaging.

    Returns
    ───────
    coadds : {inst_name: (size, size) array}
    meta   : {inst_name: {"theta", "x0", "y0", "index"}}
    """
    if not instruments:
        raise ValueError("At least one Instrument must be provided.")

    fs = np.asarray(flat_samples, dtype=float)
    if fs.ndim != 2 or fs.shape[1] != 7:
        raise ValueError("flat_samples must have shape (Ns, 7).")

    for inst in instruments:
        if inst.images_native is None:
            raise ValueError(
                f"Instrument '{inst.name}' has no images_native; cannot build coadd."
            )

    def _one_instrument_one_theta(theta_: np.ndarray, inst: Instrument):
        imgs = np.asarray(inst.images_native)
        K, H, W = imgs.shape

        x_pix, y_pix = _predict_pixel_track_native(
            theta_, inst.ts, size=inst.size, scale=inst.scale, t_ref=inst.t_ref
        )

        if align_to == "first":
            x0, y0 = float(x_pix[0]), float(y_pix[0])
        elif isinstance(align_to, (list, tuple)) and len(align_to) == 2:
            x0, y0 = map(float, align_to)
        else:
            x0 = y0 = float(inst.size // 2)

        dx = x0 - x_pix
        dy = y0 - y_pix

        if combine == "invvar":
            data_i = dict(x=inst.xgrid, noise=inst.noise)
            w = _invvar_weights_from_profiles_native(
                inst.ts, x_pix, y_pix, data_i, size=inst.size
            )
        else:
            w = np.ones(K, dtype=float)

        num = np.zeros((H, W), dtype=float)
        den = np.zeros((H, W), dtype=float) if combine == "invvar" else None

        for k_ in range(K):
            shifted = ndi_shift(
                imgs[k_], shift=(dy[k_], dx[k_]),
                order=interpolation_order, mode="nearest",
            )
            if combine == "invvar":
                wk = float(w[k_])
                if wk > 0.0 and np.isfinite(wk):
                    num += wk * shifted
                    den += wk
            else:
                num += shifted

        if combine == "invvar":
            coadd_inst = np.divide(num, den, out=np.zeros_like(num), where=(den > 0))
        elif combine == "sum":
            coadd_inst = num / max(K, 1)
        else:
            raise ValueError("combine must be 'invvar' or 'sum'.")

        return coadd_inst, dict(theta=np.asarray(theta_, float), x0=x0, y0=y0)

    if theta is None:
        top1_idx, _ = _select_top_by_logprob(
            fs, 1, sampler=sampler, logprob_kwargs=logprob_kwargs,
            thin=thin, discard=discard,
        )
        theta_used  = fs[top1_idx[0]]
        theta_index = int(top1_idx[0])
    else:
        theta_used  = np.asarray(theta, dtype=float)
        theta_index = None

    coadds: Dict[str, np.ndarray] = {}
    meta:   Dict[str, dict]       = {}
    for inst in instruments:
        coadd_inst, meta_inst = _one_instrument_one_theta(theta_used, inst)
        meta_inst["index"]    = theta_index
        coadds[inst.name]     = coadd_inst
        meta[inst.name]       = meta_inst
    return coadds, meta


def show_coadd_native(
    coadds: Mapping[str, np.ndarray],
    meta: Mapping[str, dict],
    *,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: str = "Posterior-aligned coadd (native)",
    cmap: str = "gray",
    save_path: Optional[str] = None,
) -> None:
    """
    Display (and optionally save) per-instrument native coadds.

    One figure is created per instrument, with the instrument name appended
    to the base title and base filename.
    """
    if not isinstance(coadds, Mapping) or len(coadds) == 0:
        raise ValueError("coadds must be a non-empty mapping.")

    for inst_name, img in coadds.items():
        arr = np.asarray(img)
        if arr.ndim != 2:
            raise ValueError(f"Coadd for '{inst_name}' must be 2D, got {arr.shape}.")

        vmin_local = np.percentile(arr, 1)  if vmin is None else vmin
        vmax_local = np.percentile(arr, 99) if vmax is None else vmax

        plt.figure(figsize=(6, 6))
        plt.imshow(arr, origin="lower", cmap=cmap, vmin=vmin_local, vmax=vmax_local)

        x0 = y0 = None
        if isinstance(meta, Mapping) and inst_name in meta:
            m  = meta[inst_name]
            x0 = m.get("x0", None)
            y0 = m.get("y0", None)
        if x0 is not None and y0 is not None:
            plt.plot([x0], [y0], marker="o", ms=18, mfc="none", mec="r", mew=1.5)

        plt.title(f"{title} [{inst_name}]")
        plt.xlabel("x [px]"); plt.ylabel("y [px]")
        plt.tight_layout()

        if save_path:
            base, ext  = os.path.splitext(save_path)
            inst_path  = f"{base}_{inst_name}{ext or '.png'}"
            plt.savefig(inst_path, dpi=200)
        plt.close()


def _select_top_by_logprob(
    flat_samples: np.ndarray,
    n_top: int,
    *,
    sampler: Optional[emcee.EnsembleSampler] = None,
    logprob_kwargs: Optional[dict] = None,
    thin: Optional[int] = None,
    discard: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return indices of the top-N samples by log-posterior value.

    Uses the sampler's stored log_prob when available (fast).
    Falls back to re-evaluating log_probability for each sample otherwise.
    """
    fs = np.asarray(flat_samples)
    lp = None
    if sampler is not None:
        lp_try = sampler.get_log_prob(flat=True, thin=thin, discard=discard)
        if lp_try.shape[0] == fs.shape[0]:
            lp = lp_try
    if lp is None:
        if logprob_kwargs is None:
            raise ValueError("Need logprob_kwargs to recompute log_prob.")
        lp = np.array([log_probability(fs[i], **logprob_kwargs) for i in range(fs.shape[0])])
    idx_sorted = np.argsort(lp)[::-1]
    return idx_sorted[:int(n_top)], lp


def annotate_top_orbits_by_logprob_native(
    instruments: Sequence[Instrument],
    flat_samples: np.ndarray,
    n_top: int,
    *,
    sampler: Optional[emcee.EnsembleSampler] = None,
    logprob_kwargs: Optional[dict] = None,
    thin: Optional[int] = None,
    discard: int = 0,
    ncols: int = 6,
    circle_radius: float = 4.0,
    top_color: str = "C0",
    top_alpha: float = 0.5,
    top_lw: float = 1.0,
    cmap: str = "gray",
    title: str = "Top-N posterior orbits (by log-probability)",
    save_path: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> dict:
    """
    Overlay predicted positions of the top-N posterior samples on native images.

    One figure per instrument; one panel per epoch.  Circles mark the predicted
    planet position for each of the n_top highest-probability samples.

    When the instrument uses the "snr_map" backend, images_native is still used
    for the overlay (the SNR map is not shown here; only the native image is
    used as the background).  This is intentional: the orbit overlay is purely
    a visual diagnostic of the orbital solution, independent of the backend.
    """
    if not instruments:
        raise ValueError("At least one Instrument must be provided.")

    fs = np.asarray(flat_samples, dtype=float)
    if fs.ndim != 2 or fs.shape[1] != 7:
        raise ValueError("flat_samples must have shape (Ns, 7).")

    top_idx, lp = _select_top_by_logprob(
        fs, n_top, sampler=sampler, logprob_kwargs=logprob_kwargs,
        thin=thin, discard=discard,
    )
    fs_top = fs[top_idx]

    per_inst_meta: Dict[str, dict] = {}

    for inst in instruments:
        imgs = np.asarray(inst.images_native)
        if imgs.ndim != 3:
            raise ValueError(f"Instrument '{inst.name}' images_native must be 3D.")
        K, H, W = imgs.shape

        x_list, y_list = [], []
        for theta in fs_top:
            xr, yr = _predict_pixel_track_native(
                theta, inst.ts, size=inst.size, scale=inst.scale, t_ref=inst.t_ref
            )
            x_list.append(xr); y_list.append(yr)

        x_top = np.stack(x_list, axis=0)   # (n_top, K)
        y_top = np.stack(y_list, axis=0)
        per_inst_meta[inst.name] = {"x_pix_top": x_top, "y_pix_top": y_top}

        vmin_local = np.percentile(imgs, 1)  if vmin is None else vmin
        vmax_local = np.percentile(imgs, 99) if vmax is None else vmax

        ncols_eff = max(1, int(ncols))
        nrows     = int(np.ceil(K / ncols_eff))
        fig, axes = plt.subplots(
            nrows, ncols_eff,
            figsize=(3.2 * ncols_eff, 3.2 * nrows), squeeze=False,
        )
        axes = axes.ravel()

        for i, ax in enumerate(axes):
            ax.set_xticks([]); ax.set_yticks([])
            if i < K:
                ax.imshow(imgs[i], origin="lower", cmap=cmap,
                          vmin=vmin_local, vmax=vmax_local)
                for r_ in range(fs_top.shape[0]):
                    ax.add_patch(Circle(
                        (x_top[r_, i], y_top[r_, i]),
                        radius=circle_radius, fill=False,
                        ec=top_color, lw=top_lw, alpha=top_alpha,
                    ))
                ax.set_title(f"Epoch {i} | top-{fs_top.shape[0]}")
            else:
                ax.axis("off")

        if title:
            fig.suptitle(f"{title} [{inst.name}]", fontsize=14)
        fig.tight_layout()

        if save_path:
            base, ext = os.path.splitext(save_path)
            fig.savefig(f"{base}_{inst.name}{ext or '.png'}", dpi=200)
        plt.close(fig)

    return {
        "top_indices": top_idx,
        "top_logprob": lp[top_idx],
        "per_instrument": per_inst_meta,
    }


def plot_image_and_logprob_maps_per_epoch_fast(
    instruments: Sequence[Instrument],
    flat_samples: np.ndarray,
    *,
    sampler: Optional[emcee.EnsembleSampler] = None,
    thin: Optional[int] = None,
    discard: int = 0,
    logprob_kwargs: Optional[dict] = None,
    bins: Union[int, Tuple[int, int]] = 64,
    chunk_size: int = 50000,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap_image: str = "gray",
    cmap_map: str = "viridis",
    save_dir: Optional[str] = None,
    dpi: int = 200,
) -> dict:
    """
    For each instrument and epoch, plot the native image alongside a 2-D map
    of the maximum log-probability of posterior samples that predict the planet
    at each (x, y) bin.

    This is a model-posterior diagnostic: it shows *where* the posterior places
    the planet in each frame, allowing you to visually check that the MCMC
    samples are consistent with a real signal.
    """
    if not instruments:
        raise ValueError("At least one Instrument is required.")

    fs = np.asarray(flat_samples, dtype=float)
    if fs.ndim != 2 or fs.shape[1] != 7:
        raise ValueError("flat_samples must have shape (Ns, 7).")

    # Retrieve log-probabilities.
    lp = None
    if sampler is not None:
        lp_try = sampler.get_log_prob(flat=True, thin=thin, discard=discard)
        if lp_try.shape[0] == fs.shape[0]:
            lp = lp_try
    if lp is None:
        if logprob_kwargs is None:
            raise ValueError("Need logprob_kwargs to recompute log_prob.")
        lp = np.array([log_probability(fs[i], **logprob_kwargs) for i in range(fs.shape[0])])

    results = {}

    for inst in instruments:
        imgs = np.asarray(inst.images_native)
        if imgs is None or imgs.ndim != 3:
            print(f"[logprob_maps] Skipping {inst.name}: no native images.")
            continue

        K    = int(len(inst.ts))
        size = inst.size

        if isinstance(bins, int):
            nbins_x = nbins_y = bins
        else:
            nbins_x, nbins_y = int(bins[0]), int(bins[1])

        bin_edges_x = np.linspace(0, size, nbins_x + 1)
        bin_edges_y = np.linspace(0, size, nbins_y + 1)

        # For each epoch, build a 2-D max-logprob map in pixel space.
        lp_maps = np.full((K, nbins_y, nbins_x), -np.inf, dtype=float)

        # Process in chunks to limit memory usage.
        Ns = fs.shape[0]
        for start in range(0, Ns, chunk_size):
            stop        = min(start + chunk_size, Ns)
            theta_chunk = fs[start:stop]
            lp_chunk    = lp[start:stop]

            x_pix_c, y_pix_c = _predict_tracks_native_chunk(
                theta_chunk, inst.ts, size=size, scale=inst.scale, t_ref=inst.t_ref
            )   # (K, chunk)

            for k_ in range(K):
                ix = np.searchsorted(bin_edges_x, x_pix_c[k_], side="right") - 1
                iy = np.searchsorted(bin_edges_y, y_pix_c[k_], side="right") - 1
                ix = np.clip(ix, 0, nbins_x - 1)
                iy = np.clip(iy, 0, nbins_y - 1)

                for j_ in range(len(lp_chunk)):
                    bx = int(ix[j_]); by = int(iy[j_])
                    if lp_chunk[j_] > lp_maps[k_, by, bx]:
                        lp_maps[k_, by, bx] = lp_chunk[j_]

        inst_results = {"lp_maps": lp_maps}
        results[inst.name] = inst_results

        # Plot and optionally save.
        ncols_eff = min(K, 4)
        nrows     = int(np.ceil(K / ncols_eff))
        fig, axes = plt.subplots(
            nrows, ncols_eff * 2,
            figsize=(3.0 * ncols_eff * 2, 3.0 * nrows), squeeze=False,
        )

        for k_ in range(K):
            row = k_ // ncols_eff
            col = (k_ % ncols_eff) * 2

            # Native image panel.
            vmin_i = np.percentile(imgs[k_], 1)  if vmin is None else vmin
            vmax_i = np.percentile(imgs[k_], 99) if vmax is None else vmax
            axes[row, col].imshow(
                imgs[k_], origin="lower", cmap=cmap_image,
                vmin=vmin_i, vmax=vmax_i,
            )
            axes[row, col].set_title(f"{inst.name} epoch {k_} (image)")
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])

            # Log-prob map panel.
            lp_k = lp_maps[k_].copy()
            finite_mask = np.isfinite(lp_k)
            if np.any(finite_mask):
                lp_k[~finite_mask] = lp_k[finite_mask].min()
            axes[row, col + 1].imshow(
                lp_k, origin="lower", cmap=cmap_map,
                extent=[0, size, 0, size],
            )
            axes[row, col + 1].set_title(f"max log-prob map")
            axes[row, col + 1].set_xticks([]); axes[row, col + 1].set_yticks([])

        # Hide unused panels.
        for k_ in range(K, nrows * ncols_eff):
            row = k_ // ncols_eff
            col = (k_ % ncols_eff) * 2
            if row < axes.shape[0]:
                axes[row, col].axis("off")
                axes[row, col + 1].axis("off")

        fig.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(
                os.path.join(save_dir, f"logprob_map_{inst.name}.png"), dpi=dpi
            )
        plt.close(fig)

    return results


def plot_sum_with_top_orbits_and_epoch_marks(
    instruments: Sequence[Instrument],
    flat_samples: np.ndarray,
    n_top: int,
    *,
    sampler: Optional[emcee.EnsembleSampler] = None,
    thin: Optional[int] = None,
    discard: int = 0,
    logprob_kwargs: Optional[dict] = None,
    orbit_time_pad: float = 0.1,
    min_periods: float = 1.0,
    orbit_T: int = 1000,
    orbit_color: str = "r",
    orbit_alpha: float = 0.25,
    orbit_lw: float = 0.6,
    cross_color: str = "b",
    cross_ms: float = 6.0,
    cross_mew: float = 1.2,
    zoom_margin: float = 10.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: str = "Sum of images + top-N orbits and epoch positions",
    save_path: Optional[str] = None,
    dpi: int = 200,
) -> dict:
    """
    For each instrument, plot the sum of all native images with the top-N
    posterior orbits overlaid as smooth curves, and epoch marks as crosses.

    A zoom panel is added alongside the main panel, centred around the region
    where the predicted positions cluster.

    The summed image uses native frames (images_native), so the orbital overlay
    is always shown on a real-space representation of the data regardless of
    the photometry backend used for the likelihood.
    """
    if not instruments:
        raise ValueError("At least one Instrument is required.")

    fs = np.asarray(flat_samples, dtype=float)
    if fs.ndim != 2 or fs.shape[1] != 7:
        raise ValueError("flat_samples must have shape (Ns, 7).")

    top_idx, lp = _select_top_by_logprob(
        fs, n_top, sampler=sampler, logprob_kwargs=logprob_kwargs,
        thin=thin, discard=discard,
    )
    thetas_top = fs[top_idx]   # (n_top, 7)

    for inst in instruments:
        imgs = np.asarray(inst.images_native)
        if imgs is None or imgs.ndim != 3:
            print(f"[sum_orbits] Skipping {inst.name}: no native images.")
            continue

        ts       = np.asarray(inst.ts)
        summed   = np.sum(imgs, axis=0)

        vmin_local = np.percentile(summed, 1)  if vmin is None else vmin
        vmax_local = np.percentile(summed, 99) if vmax is None else vmax

        t_min    = float(np.min(ts))
        t_max    = float(np.max(ts))
        t_mid    = 0.5 * (t_min + t_max)
        span_obs = max(t_max - t_min, 0.0)
        span_obs_padded = span_obs * (1.0 + 2.0 * orbit_time_pad)

        fig, (ax_main, ax_zoom) = plt.subplots(
            1, 2, figsize=(11, 6), gridspec_kw=dict(width_ratios=[3, 2])
        )

        ax_main.imshow(summed, origin="lower", cmap="gray",
                       vmin=vmin_local, vmax=vmax_local)
        ax_main.set_xlabel("x [px]"); ax_main.set_ylabel("y [px]")
        ax_main.set_title(f"{title} [{inst.name}]")

        all_xk, all_yk = [], []

        for theta in thetas_top:
            a_, la0_, m0_, h_, k_, p_, q_ = map(float, theta)
            n_ = mean_motion(a_, m0_)
            P_ = 2.0 * np.pi / n_ if n_ > 0 else 0.0

            span_target = max(span_obs_padded, min_periods * P_) or (P_ if P_ > 0 else 1.0)
            t_grid = np.linspace(t_mid - 0.5 * span_target, t_mid + 0.5 * span_target, orbit_T)

            xo, yo = _predict_pixel_track_native(
                theta, t_grid, size=inst.size, scale=inst.scale, t_ref=inst.t_ref
            )
            ax_main.plot(xo, yo, "-", color=orbit_color,
                         alpha=orbit_alpha, linewidth=orbit_lw, solid_capstyle="round", zorder=10)

            xk, yk = _predict_pixel_track_native(
                theta, ts, size=inst.size, scale=inst.scale, t_ref=inst.t_ref
            )
            ax_main.plot(xk, yk, "+", ls="none", mec=cross_color, mfc="none",
                         ms=cross_ms, mew=cross_mew, zorder=11)
            all_xk.append(xk); all_yk.append(yk)

        ax_main.set_xlim(0, inst.size); ax_main.set_ylim(0, inst.size)

        all_xk = np.concatenate(all_xk) if all_xk else np.array([inst.size / 2.0])
        all_yk = np.concatenate(all_yk) if all_yk else np.array([inst.size / 2.0])

        x_min = max(0.0, float(np.min(all_xk)) - zoom_margin)
        x_max = min(float(inst.size), float(np.max(all_xk)) + zoom_margin)
        y_min = max(0.0, float(np.min(all_yk)) - zoom_margin)
        y_max = min(float(inst.size), float(np.max(all_yk)) + zoom_margin)

        ax_zoom.imshow(summed, origin="lower", cmap="gray", vmin=vmin_local, vmax=vmax_local)
        ax_zoom.set_xlim(x_min, x_max); ax_zoom.set_ylim(y_min, y_max)
        ax_zoom.set_title(f"{inst.name} — zoom on discrete epochs")
        ax_zoom.set_xlabel("x [px]"); ax_zoom.set_ylabel("y [px]")

        for theta in thetas_top:
            a_, la0_, m0_, h_, k_, p_, q_ = map(float, theta)
            n_ = mean_motion(a_, m0_)
            P_ = 2.0 * np.pi / n_ if n_ > 0 else 0.0
            span_target = max(span_obs_padded, min_periods * P_) or (P_ if P_ > 0 else 1.0)
            t_grid = np.linspace(t_mid - 0.5 * span_target, t_mid + 0.5 * span_target, orbit_T)

            xo, yo = _predict_pixel_track_native(
                theta, t_grid, size=inst.size, scale=inst.scale, t_ref=inst.t_ref
            )
            ax_zoom.plot(xo, yo, "-", color=orbit_color,
                         alpha=orbit_alpha, linewidth=orbit_lw, solid_capstyle="round", zorder=10)

            xk, yk = _predict_pixel_track_native(
                theta, ts, size=inst.size, scale=inst.scale, t_ref=inst.t_ref
            )
            ax_zoom.plot(xk, yk, "+", ls="none", mec=cross_color, mfc="none",
                         ms=cross_ms, mew=cross_mew, zorder=11)

        fig.tight_layout()
        if save_path:
            base, ext = os.path.splitext(save_path)
            fig.savefig(f"{base}_{inst.name}{ext or '.png'}", dpi=dpi)
        plt.close(fig)

    return {"top_indices": top_idx, "top_logprob": lp[top_idx]}


# =============================================================================
# GLRT + OFF-TRACKS
# =============================================================================
#
# The Generalised Likelihood Ratio Test (GLRT) quantifies how significant the
# orbital detection is, by comparing the observed test statistic at the best-fit
# orbit θ* against a null distribution built from "off-track" trials.
#
# OFF-TRACKS NULL DISTRIBUTION
# ─────────────────────────────
# For each trial, random angular offsets are applied to each predicted epoch
# position, breaking orbital coherence while preserving the local noise and
# background structure.  The same random pattern is applied coherently across
# all instruments in a multi-instrument run, so that the off-track test is as
# conservative as possible.
#
# The off-tracks support the "snr_map" backend transparently: the photometric
# scalar is read from the SNR map at the offset position, exactly as it is
# during the likelihood evaluation.  The FAP estimate is thus consistent with
# the actual MCMC metric.

def _eval_snr_or_fluxZ_for_theta(
    theta7: np.ndarray,
    *,
    mode: str,
    instruments: Sequence[Instrument],
    noise_floor: float = 1.0,
    weighting: str = "invvar",
) -> tuple[Dict[str, float], Dict[str, float], float, float]:
    """
    Evaluate the GLRT test statistic (Z and Z²) for a single orbit θ.

    Returns (Z_by_inst, Z2_by_inst, Z_global, Z2_global).
    """
    theta7 = np.asarray(theta7, dtype=float).ravel()
    if theta7.size != 7:
        raise ValueError("theta7 must have 7 elements.")

    like = str(mode).lower()
    Z_by_inst:  Dict[str, float] = {}
    Z2_by_inst: Dict[str, float] = {}

    for inst in instruments:
        data_i = dict(x=inst.xgrid, bkg=inst.bkg, noise=inst.noise)

        if like == "flux":
            S1_i, S2_i = flux_sufficient_stats_from_hkpq(
                theta7[None, :], ts=inst.ts, images=inst.images_up,
                data=data_i, size=inst.size, scale=inst.scale,
                upsampling_factor=inst.upsampling_factor, t_ref=inst.t_ref,
                r_mask=inst.r_mask, r_mask_ext=inst.r_mask_ext,
                noise_floor=noise_floor,
                photometry_method=inst.photometry_method,
                images_native=inst.images_native, fwhm=inst.fwhm,
            )
            Zi = float(S2_i[0] / np.sqrt(S1_i[0])) if (S1_i[0] > 0 and np.isfinite(S1_i[0])) else 0.0
        else:
            _, _, snr_i = snr_from_hkpq(
                theta7[None, :], ts=inst.ts, images=inst.images_up,
                data=data_i, size=inst.size, scale=inst.scale,
                upsampling_factor=inst.upsampling_factor, t_ref=inst.t_ref,
                r_mask=inst.r_mask, r_mask_ext=inst.r_mask_ext,
                noise_floor=noise_floor, weighting=weighting,
                photometry_method=inst.photometry_method,
                images_native=inst.images_native, fwhm=inst.fwhm,
                snr_maps=inst.snr_maps,
            )
            Zi = float(snr_i[0]) if np.isfinite(snr_i[0]) else 0.0

        Z_by_inst[inst.name]  = Zi
        Z2_by_inst[inst.name] = Zi * Zi

    if like == "flux":
        S1_tot, S2_tot = flux_sufficient_stats_multi_from_hkpq(
            theta7[None, :], instruments=instruments, noise_floor=noise_floor
        )
        Z_global = float(S2_tot[0] / np.sqrt(S1_tot[0])) if (S1_tot[0] > 0 and np.isfinite(S1_tot[0])) else 0.0
    else:
        _, _, snr_tot = snr_multi_from_hkpq(
            theta7[None, :], instruments=instruments,
            noise_floor=noise_floor, weighting=weighting,
        )
        Z_global = float(snr_tot) if np.isfinite(snr_tot) else 0.0

    return Z_by_inst, Z2_by_inst, Z_global, Z_global * Z_global


def _photometric_scalar_at_epoch_for_glrt(
    inst: Instrument,
    k_epoch: int,
    x_pix: float,
    y_pix: float,
) -> float:
    """
    Read a photometric scalar at (x_pix, y_pix) for epoch k_epoch.

    Supports all three backends:
      - "convolve": read nearest upsampled pixel.
      - "aperture": circular aperture on native image.
      - "snr_map":  read nearest pixel in SNR map.

    This is used by the off-tracks null distribution to evaluate the
    photometric scalar at perturbed positions.
    """
    if inst.photometry_method == "snr_map":
        # SNR-map backend: read the SNR value at the nearest native pixel.
        if inst.snr_maps is None:
            return 0.0
        x_nat = int(round(x_pix))
        y_nat = int(round(y_pix))
        if not (0 <= x_nat < inst.size and 0 <= y_nat < inst.size):
            return 0.0
        val = float(inst.snr_maps[k_epoch, y_nat, x_nat])
        return val if np.isfinite(val) else 0.0

    elif inst.photometry_method == "convolve":
        if inst.images_up is None:
            return 0.0
        x_up = int(np.floor(x_pix * inst.upsampling_factor - 0.5))
        y_up = int(np.floor(y_pix * inst.upsampling_factor - 0.5))
        if not (0 <= x_up < inst.images_up.shape[2] and 0 <= y_up < inst.images_up.shape[1]):
            return 0.0
        val = float(inst.images_up[k_epoch, y_up, x_up])
        return val if np.isfinite(val) else 0.0

    else:   # "aperture"
        if inst.images_native is None or inst.fwhm is None or inst.fwhm <= 0:
            return 0.0
        if not (0.0 <= x_pix < float(inst.size) and 0.0 <= y_pix < float(inst.size)):
            return 0.0
        ap   = CircularAperture((x_pix, y_pix), r=float(inst.fwhm))
        phot = aperture_photometry(inst.images_native[k_epoch], ap)
        val  = float(np.array(phot["aperture_sum"])[0])
        return val if np.isfinite(val) else 0.0


def _offtrack_stat_for_theta(
    theta7: np.ndarray,
    *,
    mode: str,
    instruments: Sequence[Instrument],
    noise_floor: float,
    weighting: str,
    n_trials: int = 200,
    dr_min_px: Optional[float] = None,
    dr_max_px: Optional[float] = None,
    dr_min_fwhm: Optional[float] = None,
    dr_max_fwhm: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Build the off-tracks null distribution for the GLRT.

    For each trial, a shared random angular offset pattern is applied coherently
    across all instruments.  The same photometry backend that is used in the
    MCMC likelihood is used to read the photometric scalar at the offset positions.

    Offset scale convention
    ───────────────────────
    Preferred user convention: dr_min_fwhm/dr_max_fwhm are multipliers of each
    instrument FWHM.  For example dr_min_fwhm=2 means an offset of at least
    2 × fwhm for every instrument.  Absolute dr_min_px/dr_max_px are retained
    only for expert/backward-compatible use.
    """
    theta7 = np.asarray(theta7, dtype=float).ravel()
    if theta7.size != 7:
        raise ValueError("theta7 must have 7 elements.")

    like      = str(mode).lower()
    weighting = str(weighting).lower()
    if rng is None:
        rng = np.random.default_rng()

    # Precompute the on-orbit positions for each instrument.
    a, la0, m0, h, k, p, q = map(float, theta7)
    M0 = lambda_to_M0(la0, h, k)
    t0 = M0_to_t0(M0, a, m0, instruments[0].t_ref)

    base_tracks: Dict[str, tuple] = {}
    centers:     Dict[str, tuple] = {}
    max_K = 0

    for inst in instruments:
        ts_i    = np.asarray(inst.ts, dtype=float)
        x_orb_, y_orb_ = orbit.positions_at_multiple_times(
            ts_i, np.array([[a, np.hypot(h, k), t0, m0]])
        )
        x_orb_ = x_orb_[:, 0]; y_orb_ = y_orb_[:, 0]
        proj = compute_projection_matrices_from_hkpq(
            np.array([k], float), np.array([h], float),
            np.array([p], float), np.array([q], float),
        )[0]
        north_sky_ = x_orb_ * proj[0, 0] + y_orb_ * proj[0, 1]
        west_sky_  = x_orb_ * proj[1, 0] + y_orb_ * proj[1, 1]
        cx = cy = float(inst.size // 2)
        base_tracks[inst.name] = (west_sky_ * inst.scale + cx, north_sky_ * inst.scale + cy)
        centers[inst.name]     = (cx, cy)
        max_K = max(max_K, len(ts_i))

    use_abs = (dr_min_px is not None and dr_max_px is not None)
    use_fwhm_range = (dr_min_fwhm is not None and dr_max_fwhm is not None)
    base_scale: Dict[str, float] = {}
    for inst in instruments:
        base_scale[inst.name] = float(inst.fwhm) if (inst.fwhm and inst.fwhm > 0) else 1.0

    if use_abs:
        dr_min_px = float(dr_min_px)
        dr_max_px = float(dr_max_px)
    elif use_fwhm_range:
        dr_min_fwhm = float(dr_min_fwhm)
        dr_max_fwhm = float(dr_max_fwhm)
    else:
        # Historical default: random offsets between 1 and 5 FWHM.
        dr_min_fwhm = 1.0
        dr_max_fwhm = 5.0
        use_fwhm_range = True

    off_by_inst: Dict[str, list] = {inst.name: [] for inst in instruments}
    off_global:  list            = []

    for _ in range(int(n_trials)):
        ang_global = rng.uniform(0.0, 2.0 * np.pi, size=max_K)
        if use_abs:
            dr_global = rng.uniform(dr_min_px, dr_max_px, size=max_K)
            u_global  = None
        else:
            # u_global is dimensionless.  The actual offset for each instrument is
            # u_global × instrument.fwhm, so a value of 2 means 2 × FWHM.
            u_global  = rng.uniform(dr_min_fwhm, dr_max_fwhm, size=max_K)
            dr_global = None

        if like == "flux":
            S1_global = 0.0; S2_global = 0.0
        else:
            num_global = 0.0; den_global = 0.0

        per_inst_flux    = {}
        per_inst_invvar  = {}
        per_inst_simple  = {}

        for inst in instruments:
            x0_arr, y0_arr = base_tracks[inst.name]
            cx, cy         = centers[inst.name]
            K_i            = int(len(inst.ts))

            ang = ang_global[:K_i]
            dr  = dr_global[:K_i] if use_abs else base_scale[inst.name] * u_global[:K_i]
            dx  = dr * np.cos(ang)
            dy  = dr * np.sin(ang)

            xgrid = np.asarray(inst.xgrid, dtype=float)
            bkg   = np.asarray(inst.bkg,   dtype=np.float32)
            noise = np.asarray(inst.noise,  dtype=np.float32)

            if like == "flux":
                S1_i = 0.0; S2_i = 0.0
            else:
                num_i = 0.0; den_i = 0.0; sum_y_i = 0.0; sum_var_i = 0.0

            for kk in range(K_i):
                xk   = float(x0_arr[kk] + dx[kk])
                yk   = float(y0_arr[kk] + dy[kk])
                rnat = float(np.hypot(xk - cx, yk - cy))

                # Soft mask: skip if outside the photometric annulus.
                if (inst.r_mask is not None and rnat <= float(inst.r_mask)) or \
                   (inst.r_mask_ext is not None and rnat >= float(inst.r_mask_ext)):
                    continue

                F = _photometric_scalar_at_epoch_for_glrt(inst, kk, xk, yk)

                if inst.photometry_method == "snr_map":
                    # SNR-map backend: F is already a SNR value.
                    # Accumulate SNR² for the invvar combination or SNR for simple.
                    if like == "flux":
                        pass   # SNR map + flux mode is not allowed; guarded upstream.
                    else:
                        if weighting == "invvar":
                            num_i      += F * F   # SNR²
                            den_i      += 1.0     # weight = 1 (SNR already normalised)
                            num_global += F * F
                            den_global += 1.0
                        else:
                            sum_y_i   += F
                            sum_var_i += 1.0
                            num_global += F
                            den_global += 1.0
                    continue

                # Standard flux-based backends.
                bg  = float(np.interp(rnat, xgrid, bkg[kk]))
                sig = float(np.interp(rnat, xgrid, noise[kk]))
                if not np.isfinite(sig) or sig <= 0.0:
                    continue
                y = F - bg

                if like == "flux":
                    w = 1.0 / (sig * sig)
                    S1_i      += w; S2_i      += w * y
                    S1_global += w; S2_global += w * y
                else:
                    if weighting == "invvar":
                        w = 1.0 / (sig * sig)
                        num_i += w * y; den_i += w
                        num_global += w * y; den_global += w
                    else:
                        sum_y_i   += y; sum_var_i += sig * sig
                        num_global += y; den_global += sig * sig

            # Store accumulators.
            if like == "flux":
                per_inst_flux[inst.name] = (S1_i, S2_i)
            else:
                if weighting == "invvar":
                    per_inst_invvar[inst.name] = (num_i, den_i)
                else:
                    per_inst_simple[inst.name] = (sum_y_i, sum_var_i)

        # Convert accumulators to Z² per instrument and globally.
        if like == "flux":
            for inst in instruments:
                S1_i, S2_i = per_inst_flux.get(inst.name, (0.0, 0.0))
                Zi2 = float((S2_i * S2_i) / S1_i) if (S1_i > 0 and np.isfinite(S1_i)) else 0.0
                off_by_inst[inst.name].append(Zi2)
            Z2g = float((S2_global * S2_global) / S1_global) if (S1_global > 0 and np.isfinite(S1_global)) else 0.0
        else:
            for inst in instruments:
                if weighting == "invvar":
                    num_i, den_i = per_inst_invvar.get(inst.name, (0.0, 0.0))
                    if den_i > 0 and np.isfinite(den_i):
                        Zi = float(num_i / np.sqrt(den_i))
                    else:
                        Zi = 0.0
                else:
                    sum_y_i, sum_var_i = per_inst_simple.get(inst.name, (0.0, 0.0))
                    if sum_var_i > 0 and np.isfinite(sum_var_i):
                        Zi = float(sum_y_i / np.sqrt(sum_var_i))
                    else:
                        Zi = 0.0
                off_by_inst[inst.name].append(Zi * Zi)

            if den_global > 0 and np.isfinite(den_global):
                Z2g = float((num_global / np.sqrt(den_global)) ** 2)
            else:
                Z2g = 0.0

        off_global.append(Z2g)

    return (
        {k_: np.asarray(v, dtype=float) for k_, v in off_by_inst.items()},
        np.asarray(off_global, dtype=float),
    )


def run_glrt_offtracks_and_save(
    *,
    params: "Params",
    mode: str,
    sampler: emcee.EnsembleSampler,
    flat: np.ndarray,
    lp_kwargs: dict,
    instruments: Sequence[Instrument],
    noise_floor: float,
    weighting: str,
    n_off: int = 200,
    dr_min_px: Optional[float] = None,
    dr_max_px: Optional[float] = None,
    dr_min_fwhm: Optional[float] = None,
    dr_max_fwhm: Optional[float] = None,
    seed: Optional[int] = None,
) -> dict:
    """
    Compute and save the GLRT detection significance via off-tracks.

    Selects the MAP orbit from the MCMC chain, computes the observed test
    statistic, builds a null distribution with n_off off-track trials,
    estimates the false-alarm probability (FAP), and writes results to
    values_dir/glrt_offtracks.json.
    """
    like = str(mode).lower()
    flat = np.asarray(flat)
    if flat.ndim != 2 or flat.shape[1] < 7:
        raise ValueError("flat must be (Nsamples, ≥7).")

    top_idx, _ = _select_top_by_logprob(
        flat[:, :7], 1, sampler=sampler, logprob_kwargs=lp_kwargs,
        thin=int(params._params.get("mcmc", {}).get("thin", 10)), discard=0,
    )
    theta_best = np.asarray(flat[top_idx[0], :7], dtype=float)

    Z_by_inst, Z2_by_inst, Z_global, Z2_global = _eval_snr_or_fluxZ_for_theta(
        theta_best, mode=mode, instruments=instruments,
        noise_floor=noise_floor, weighting=weighting,
    )

    rng = np.random.default_rng(None if seed is None else int(seed))
    off_by_inst, off_global = _offtrack_stat_for_theta(
        theta_best, mode=mode, instruments=instruments,
        noise_floor=noise_floor, weighting=weighting,
        n_trials=int(n_off),
        dr_min_px=dr_min_px, dr_max_px=dr_max_px,
        dr_min_fwhm=dr_min_fwhm, dr_max_fwhm=dr_max_fwhm,
        rng=rng,
    )

    per_inst_out: Dict[str, dict] = {}
    for inst in instruments:
        name     = inst.name
        off_vals = off_by_inst[name]
        Z2_obs_i = Z2_by_inst[name]
        if off_vals.size:
            fap_i = float(np.mean(off_vals >= Z2_obs_i))
            q95_i = float(np.quantile(off_vals, 0.95))
            q99_i = float(np.quantile(off_vals, 0.99))
        else:
            fap_i = q95_i = q99_i = float("nan")
        per_inst_out[name] = dict(
            Z_obs=float(Z_by_inst[name]), GLRT_stat=float(Z2_obs_i),
            fap=fap_i, off_quantiles=dict(q95=q95_i, q99=q99_i),
        )

    if off_global.size:
        fap_g = float(np.mean(off_global >= Z2_global))
        q95_g = float(np.quantile(off_global, 0.95))
        q99_g = float(np.quantile(off_global, 0.99))
    else:
        fap_g = q95_g = q99_g = float("nan")

    global_out = dict(
        Z_obs=float(Z_global), GLRT_stat=float(Z2_global),
        fap=fap_g, off_quantiles=dict(q95=q95_g, q99=q99_g),
    )

    out = dict(
        mode=str(mode), theta_best=theta_best.tolist(), n_off=int(n_off),
        offtrack_offsets=dict(
            dr_min_px=None if dr_min_px is None else float(dr_min_px),
            dr_max_px=None if dr_max_px is None else float(dr_max_px),
            dr_min_fwhm=None if dr_min_fwhm is None else float(dr_min_fwhm),
            dr_max_fwhm=None if dr_max_fwhm is None else float(dr_max_fwhm),
            note=(
                "dr_min_fwhm/dr_max_fwhm are dimensionless multipliers of each "
                "instrument FWHM; e.g. 2 means 2×FWHM. Absolute dr_min_px/"
                "dr_max_px are used only if explicitly provided."
            ),
        ),
        per_instrument=per_inst_out, global_stat=global_out,
    )

    values_dir = params.get_path("values_dir")
    os.makedirs(values_dir, exist_ok=True)
    out_path = os.path.join(values_dir, "glrt_offtracks.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    except Exception as e:
        print(f"[GLRT] (WARN) Failed to write JSON: {e}")

    def _fmt_fap(fap_val, ntrials):
        if np.isfinite(fap_val) and fap_val > 0.0:
            return f"{np.log10(fap_val):.4g}"
        elif np.isfinite(fap_val):
            return f"< {np.log10(1.0 / (ntrials + 1.0)):.4g}"
        return "nan"

    print(f"[GLRT] mode = {mode}")
    if dr_min_px is not None and dr_max_px is not None:
        print(f"[GLRT] off-track offset range = [{dr_min_px}, {dr_max_px}] absolute native pixels")
    else:
        print(f"[GLRT] off-track offset range = [{dr_min_fwhm}, {dr_max_fwhm}] × FWHM")
    print("[GLRT] Per-instrument results:")
    for name, res in per_inst_out.items():
        print(
            f"  - {name}: Z_obs={res['Z_obs']:.3f}, "
            f"Z²={res['GLRT_stat']:.3f}, "
            f"FAP≈{res['fap']:.4g} (log10={_fmt_fap(res['fap'], n_off)}, "
            f"q95={res['off_quantiles']['q95']:.3f}, q99={res['off_quantiles']['q99']:.3f})"
        )
    print("[GLRT] Global (all instruments combined):")
    print(
        f"  Z_obs={global_out['Z_obs']:.3f}, Z²={global_out['GLRT_stat']:.3f}, "
        f"FAP≈{global_out['fap']:.4g} (log10={_fmt_fap(global_out['fap'], n_off)}, "
        f"q95={global_out['off_quantiles']['q95']:.3f}, "
        f"q99={global_out['off_quantiles']['q99']:.3f})"
    )
    print(f"[GLRT] Results written to: {out_path}")
    return out



def plot_autocorrelation_diagnostics(
    sampler: emcee.EnsembleSampler,
    *,
    names: Sequence[str],
    save_path: str,
    step_grid: Optional[Sequence[int]] = None,
) -> Optional[str]:
    """
    Plot integrated autocorrelation-time estimates versus chain length.

    Interpretation for users:
      - A parameter is more trustworthy when its curve becomes approximately flat.
      - A common rule of thumb is chain_length ≳ 50 × tau for each parameter.
      - If the curves are still rising strongly at the final step, the chain is
        probably too short and `nsteps` should be increased.
    """
    chain = sampler.get_chain(discard=0, flat=False)
    nsteps = int(chain.shape[0])
    if nsteps < 20:
        print("[plots] autocorrelation: chain too short to estimate tau robustly.")
        return None

    if step_grid is None:
        # Use a modest number of points so the diagnostic is informative but cheap.
        n_grid = min(30, max(6, nsteps // 50))
        step_grid = np.unique(np.linspace(max(20, nsteps // 20), nsteps, n_grid).astype(int))

    taus_by_step = []
    valid_steps = []
    for step in step_grid:
        if step < 10:
            continue
        try:
            tau = emcee.autocorr.integrated_time(chain[:step], tol=0, quiet=True)
            tau = np.asarray(tau, dtype=float)
            if tau.size == len(names):
                taus_by_step.append(tau)
                valid_steps.append(int(step))
        except Exception:
            continue

    if not taus_by_step:
        print("[plots] autocorrelation: no finite tau estimate could be computed.")
        return None

    taus = np.vstack(taus_by_step)
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    for j, name in enumerate(names):
        ax.plot(valid_steps, taus[:, j], marker="o", ms=3, lw=1.2, label=name)
    ax.plot(valid_steps, np.asarray(valid_steps, dtype=float) / 50.0, ls="--", lw=1.0, label="N/50 reference")
    ax.set_xlabel("Production chain length used [steps]")
    ax.set_ylabel("Integrated autocorrelation time τ [steps]")
    ax.set_title("MCMC autocorrelation diagnostic by parameter")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] autocorrelation diagnostic saved to '{save_path}'")
    return save_path


# =============================================================================
# PLOTTING ORCHESTRATION (from YAML)
# =============================================================================

def _append_suffix_to_filename(filename: str, suffix: str) -> str:
    """
    Insert a suffix before the file extension.
    E.g. "coadd_native.png" + "bruteforce_best" → "coadd_native_bruteforce_best.png"
    """
    base = str(filename or "").strip() or "plot.png"
    root_, ext = os.path.splitext(base)
    if not ext:
        ext = ".png"
    return f"{root_}_{suffix}{ext}"


def _run_bruteforce_best_plots(
    params: "Params",
    theta_init: np.ndarray,
    instruments: Sequence["Instrument"],
    lp_kwargs: dict,
) -> None:
    """
    Pre-MCMC diagnostic: plot the coadd and orbit overlay for the best
    brute-force solution, before running the MCMC sampler.
    """
    plots_cfg = params._params.get("plots", {}) or {}
    if not plots_cfg.get("enable", True):
        return

    values_dir = params.get_path("values_dir")
    plots_dir  = os.path.join(values_dir, params._params.get("plots_dir", "plots"))
    os.makedirs(plots_dir, exist_ok=True)

    theta_init = np.asarray(theta_init, float).ravel()
    if theta_init.size < 7:
        raise ValueError("theta_init must contain at least 7 parameters.")
    theta7    = theta_init[:7]
    flat_best = theta7[None, :]

    print(f"[plots] bruteforce-best: θ_init = {theta7!r}  (a, λ0, m0, h, k, p, q)")

    if plots_cfg.get("coadd_native", {}).get("enable", False):
        cfg = plots_cfg["coadd_native"]
        coadds, meta = stack_planet_from_posterior_native(
            instruments, flat_best, theta=theta7,
            combine=cfg.get("combine", "invvar"),
            align_to=cfg.get("align_to", "first"),
            sampler=None, thin=None, discard=0, logprob_kwargs=None,
        )
        fname    = _append_suffix_to_filename(cfg.get("filename", "coadd_native.png"), "bruteforce_best")
        out_path = os.path.join(plots_dir, fname)
        show_coadd_native(coadds, meta, title="Coadd (native) — best brute-force orbit", save_path=out_path)
        print(f"[plots] bruteforce-best: coadd_native saved to '{out_path}'")

    if plots_cfg.get("top_orbits_overlay", {}).get("enable", False):
        cfg      = plots_cfg["top_orbits_overlay"]
        fname    = _append_suffix_to_filename(cfg.get("filename", "images_with_top_orbits.png"), "bruteforce_best")
        out_path = os.path.join(plots_dir, fname)
        annotate_top_orbits_by_logprob_native(
            instruments, flat_best, n_top=1,
            sampler=None, logprob_kwargs=lp_kwargs, thin=None, discard=0,
            ncols=int(cfg.get("ncols", 6)),
            circle_radius=float(cfg.get("circle_radius", 4.0)),
            top_alpha=float(cfg.get("alpha", 0.35)),
            top_color=cfg.get("top_color", "C0"),
            top_lw=float(cfg.get("top_lw", 1.0)),
            cmap=cfg.get("cmap", "gray"),
            title="Brute-force best orbit",
            save_path=out_path,
        )
        print(f"[plots] bruteforce-best: orbit overlay saved to '{out_path}'")


def _run_plots_from_yaml(
    params: "Params",
    sampler: emcee.EnsembleSampler,
    flat: np.ndarray,
    instruments: Sequence[Instrument],
    lp_kwargs: dict,
) -> None:
    """
    Run the full plotting suite as configured in the YAML `plots` section.
    """
    plots_cfg = params._params.get("plots", {}) or {}
    if not plots_cfg.get("enable", True):
        return

    values_dir = params.get_path("values_dir")
    plots_dir  = os.path.join(values_dir, params._params.get("plots_dir", "plots"))
    os.makedirs(plots_dir, exist_ok=True)

    thin = int(params._params.get("mcmc", {}).get("thin", 10))

    # ── 0) Autocorrelation/convergence diagnostic ──
    if plots_cfg.get("autocorrelation", {}).get("enable", False):
        cfg = plots_cfg["autocorrelation"]
        ndim = sampler.get_chain().shape[-1]
        names = ["a", "la0", "m0", "h", "k", "p", "q"] + (["fp"] if ndim == 8 else [])
        plot_autocorrelation_diagnostics(
            sampler,
            names=names,
            save_path=os.path.join(plots_dir, cfg.get("filename", "autocorrelation.png")),
        )

    # ── 1) Corner plot ──
    if plots_cfg.get("corner", {}).get("enable", False):
        cfg        = plots_cfg["corner"]
        flat7      = flat[:, :7]
        flat_aug, labels_aug = augment_chain_with_physical(flat7, params.t_ref)
        fname = cfg.get("filename", "corner.png")
        if flat.shape[1] == 8:
            fname = _append_suffix_to_filename(fname, "with_fp")
        make_corner_plot(flat_aug, labels_aug, os.path.join(plots_dir, fname))

    # ── 2) 1-D posterior histograms ──
    if plots_cfg.get("hist", {}).get("enable", False):
        cfg          = plots_cfg["hist"]
        labels_user  = cfg.get("labels", None)

        if flat.shape[1] == 8:
            fp_samples = flat[:, 7]
            plot_augmented_posteriors_with_optional_fp(
                flat[:, :7], params.t_ref, fp=fp_samples,
                labels_to_plot=labels_user,
                bins=int(cfg.get("bins", 40)),
                savepath=os.path.join(plots_dir, cfg.get("filename", "posteriors.png")),
            )
        else:
            plot_augmented_posteriors(
                flat[:, :7], params.t_ref,
                labels_to_plot=labels_user,
                bins=int(cfg.get("bins", 40)),
                savepath=os.path.join(plots_dir, cfg.get("filename", "posteriors.png")),
            )

    # ── 3) Native coadd per instrument ──
    if plots_cfg.get("coadd_native", {}).get("enable", False):
        cfg = plots_cfg["coadd_native"]
        coadds, meta = stack_planet_from_posterior_native(
            instruments, flat[:, :7], theta=None,
            combine=cfg.get("combine", "invvar"),
            align_to=cfg.get("align_to", "first"),
            sampler=sampler, thin=thin, discard=0, logprob_kwargs=lp_kwargs,
        )
        show_coadd_native(
            coadds, meta,
            title="Coadd (native, aligned to first-epoch predicted position)",
            save_path=os.path.join(plots_dir, cfg.get("filename", "coadd_native.png")),
        )

    # ── 4) Top-N orbit overlay per epoch, per instrument ──
    if plots_cfg.get("top_orbits_overlay", {}).get("enable", False):
        cfg = plots_cfg["top_orbits_overlay"]
        annotate_top_orbits_by_logprob_native(
            instruments, flat[:, :7],
            n_top=int(cfg.get("n_top", 100)),
            sampler=sampler, logprob_kwargs=lp_kwargs, thin=thin, discard=0,
            ncols=int(cfg.get("ncols", 6)),
            circle_radius=float(cfg.get("circle_radius", 4.0)),
            top_alpha=float(cfg.get("alpha", 0.35)),
            top_color=cfg.get("top_color", "C0"),
            top_lw=float(cfg.get("top_lw", 1.0)),
            cmap=cfg.get("cmap", "gray"),
            title=cfg.get("title", "Top-N posterior orbits (by log-probability)"),
            save_path=os.path.join(plots_dir, cfg.get("filename", "images_with_top_orbits.png")),
        )

    # ── 5) Log-probability maps per epoch, per instrument ──
    if plots_cfg.get("logprob_maps", {}).get("enable", False):
        cfg     = plots_cfg["logprob_maps"]
        out_dir = os.path.join(plots_dir, cfg.get("out_dir", "logprob_maps"))
        plot_image_and_logprob_maps_per_epoch_fast(
            instruments, flat[:, :7],
            sampler=sampler, thin=thin, discard=0, logprob_kwargs=lp_kwargs,
            bins=int(cfg.get("bins", 128)),
            chunk_size=int(cfg.get("chunk_size", 50000)),
            vmin=cfg.get("vmin", None), vmax=cfg.get("vmax", None),
            cmap_image=cfg.get("cmap_image", "gray"),
            cmap_map=cfg.get("cmap_map", "viridis"),
            save_dir=out_dir, dpi=int(cfg.get("dpi", 200)),
        )

    # ── 6) Summed image + top-N orbit curves + epoch marks ──
    if plots_cfg.get("sum_with_top_orbits", {}).get("enable", False):
        cfg = plots_cfg["sum_with_top_orbits"]
        plot_sum_with_top_orbits_and_epoch_marks(
            instruments, flat[:, :7],
            n_top=int(cfg.get("n_top", 1000)),
            sampler=sampler, thin=thin, discard=0, logprob_kwargs=lp_kwargs,
            orbit_time_pad=float(cfg.get("orbit_time_pad", 0.1)),
            min_periods=float(cfg.get("min_periods", 1.0)),
            orbit_T=int(cfg.get("orbit_T", 1000)),
            orbit_color=cfg.get("orbit_color", "r"),
            orbit_alpha=float(cfg.get("orbit_alpha", 0.25)),
            orbit_lw=float(cfg.get("orbit_lw", 0.6)),
            cross_color=cfg.get("cross_color", "b"),
            cross_ms=float(cfg.get("cross_ms", 6.0)),
            cross_mew=float(cfg.get("cross_mew", 1.2)),
            zoom_margin=float(cfg.get("zoom_margin", 10.0)),
            vmin=cfg.get("vmin", None), vmax=cfg.get("vmax", None),
            title=cfg.get("title", "Sum of images + top-N orbits and epoch positions"),
            save_path=os.path.join(plots_dir, cfg.get("filename", "sum_top_orbits.png")),
            dpi=int(cfg.get("dpi", 200)),
        )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def _run_mcmc_from_yaml_impl(yaml_path: str):
    """
    End-to-end MCMC runner driven by a YAML configuration file.

    Pipeline
    ────────
    1.  Read YAML → Params object.
    2.  For each instrument block: load images, profiles, time vector.
        If snr_maps_suffix is present, load SNR maps and set backend to "snr_map".
    3.  Combine all time vectors → resolve global t_ref.
    4.  Resolve priors, eccentricity prior, and bounds.
    5.  If any instrument uses "snr_map", force likelihood_mode = "snr".
    6.  Build θ_init (manual or bruteforce) and draw the walker cloud.
    7.  Optionally add the fp dimension (flux mode with sample_fp=true).
    8.  Build Instrument objects and lp_kwargs.
    9.  Run burn-in + production MCMC via run_emcee_hybrid.
    10. Extract flattened chain.
    11. Print diagnostics (acceptance fraction, IAT).
    12. Run the plotting suite.
    13. Run GLRT + off-tracks.

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
    params    = Params.read(yaml_path)
    root      = params._params
    base_root = dict(root)

    # ── (2) Load per-instrument data ──
    instruments_cfg = root.get("instruments", None)
    if not isinstance(instruments_cfg, (list, tuple)):
        raise ValueError("`instruments` must be a list of instrument configs.")

    per_inst_data: list = []
    all_ts:        list = []

    for inst_cfg in instruments_cfg:
        if not isinstance(inst_cfg, dict):
            raise ValueError("Each entry in `instruments` must be a dict.")

        # Merge global config with instrument-specific overrides.
        tmp_root = dict(base_root)
        tmp_root.update(inst_cfg)
        params._params = tmp_root

        # Decide which photometry backend this instrument uses.
        # "snr_maps_suffix" takes precedence and forces "snr_map" backend.
        snr_maps_suffix = inst_cfg.get("snr_maps_suffix", None)
        if snr_maps_suffix is not None:
            # SNR-map mode: the instrument provides pre-computed SNR maps.
            photometry_method = "snr_map"
        else:
            # Standard backends: "convolve" or "aperture".
            photometry_method = str(getattr(params, "method", "convolve")).lower()
            if photometry_method not in ("convolve", "aperture"):
                photometry_method = "convolve"

        # Load image data and radial profiles via the Params helper.
        data_io     = params.load_data(method=photometry_method if photometry_method != "snr_map" else "convolve")
        ts_i        = params.get_ts(use_p_prev=True)
        images_up   = data_io["images"]    # upsampled images (or equivalent)
        xgrid       = data_io["x"]
        bkg         = data_io["bkg"]
        noise       = data_io["noise"]
        size_i      = params.n
        scale_i     = params.scale
        upsampling_factor = params.upsampling_factor
        r_mask      = getattr(params, "r_mask",     None)
        r_mask_ext  = getattr(params, "r_mask_ext", None)
        fwhm        = float(getattr(params, "fwhm", _get(tmp_root, "fwhm", 3.0)))
        inst_name   = inst_cfg.get("name", inst_cfg.get("instrument_name", "INST"))

        # Always load native images (used by coadd and orbit overlay plots).
        images_native = _load_native_images(params)

        # Load SNR maps when the "snr_map" backend is requested.
        if photometry_method == "snr_map":
            snr_maps = _load_snr_maps(params, suffix=snr_maps_suffix)
            print(
                f"[load] Instrument '{inst_name}': loaded {snr_maps.shape[0]} SNR maps "
                f"(suffix='{snr_maps_suffix}', backend='snr_map')."
            )
        else:
            snr_maps = None

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
            snr_maps=snr_maps,
            xgrid=xgrid,
            bkg=bkg,
            noise=noise,
        ))
        all_ts.append(np.asarray(ts_i, float))

    # Restore the global config in Params.
    params._params = base_root

    ts_global = np.concatenate(all_ts) if all_ts else np.array([], float)

    # ── (3 / 4) Global knobs and priors ──
    noise_floor = float(root.get("noise_floor", 1.0))
    snr_scale   = float(root.get("snr_scale",   1.0))
    weighting   = _resolve_weighting(root)

    priors = root.get("priors", {}) or {}
    la0_bounds = tuple(map(float, _get(priors, "la0_bounds", (0.0, 2.0 * np.pi))))
    ecc_prior  = str(_get(priors, "ecc_prior", "kipping")).lower()
    ecc_beta_a = float(_get(priors, "ecc_beta_a", 0.867))
    ecc_beta_b = float(_get(priors, "ecc_beta_b", 3.03))
    e_max      = float(_get(priors, "e_max", 0.95))
    orientation_isotropic = bool(
        str(_get(priors, "orientation_isotropic", "yes")).lower()
        in ("1", "true", "yes", "y")
    )
    pq_prior = str(_get(priors, "pq_prior", "none")).lower()
    print(f"orientation_isotropic: {orientation_isotropic}")
    print(f"pq_prior: {pq_prior}")
    # Validate pq_prior compatibility with orientation_isotropic
    if pq_prior in ("clockwise", "counterclockwise") and not orientation_isotropic:
        raise ValueError(
            "pq_prior ('clockwise' or 'counterclockwise') requires "
            "orientation_isotropic: true. Set orientation_isotropic: true in YAML."
        )
    if pq_prior not in ("none", "clockwise", "counterclockwise"):
        raise ValueError(
            f"Invalid pq_prior value '{pq_prior}'. "
            "Must be 'none', 'clockwise', or 'counterclockwise'."
        )
    a_bounds, m0_bounds = _resolve_bounds(params, priors)

    t_ref     = _resolve_tref(params, root, ts_global)
    init_mode = _resolve_init_mode(root)
    mconf     = _resolve_mcmc(root)

    # ── (5) Force likelihood_mode = "snr" when any instrument uses snr_maps ──
    any_snr_map = any(d["photometry_method"] == "snr_map" for d in per_inst_data)
    if any_snr_map:
        if mconf["likelihood_mode"] != "snr":
            print(
                "[WARN] At least one instrument uses photometry_method='snr_map'. "
                "This is incompatible with likelihood_mode='flux'. "
                "Forcing likelihood_mode='snr'."
            )
        mconf["likelihood_mode"] = "snr"
        # fp sampling also makes no sense with SNR maps.
        if mconf["sample_fp"]:
            print("[WARN] sample_fp=true is incompatible with snr_map backend. Disabling.")
            mconf["sample_fp"] = False

    # ── (6) Build θ_init and draw walker cloud ──
    if init_mode == "bruteforce":
        theta_init = _build_init_from_bruteforce(params=params, root=root, t_ref=t_ref)
        print("[init] mode='bruteforce' — θ_init taken from best brute-force solution")

        a_lo, a_hi = a_bounds; m_lo, m_hi = m0_bounds; la_lo, la_hi = la0_bounds
        theta_raw  = theta_init.copy()
        theta_init[0] = np.clip(theta_init[0], a_lo, a_hi)
        theta_init[1] = np.clip(theta_init[1], la_lo, la_hi)
        theta_init[2] = np.clip(theta_init[2], m_lo, m_hi)
        for pname, pidx, (lo, hi) in [("a", 0, (a_lo, a_hi)), ("la0", 1, (la_lo, la_hi)), ("m0", 2, (m_lo, m_hi))]:
            if theta_raw[pidx] < lo or theta_raw[pidx] > hi:
                print(
                    f"[init] WARNING: bruteforce {pname}_init={theta_raw[pidx]:.8g} "
                    f"was outside [{lo:.8g}, {hi:.8g}]; clipped to {theta_init[pidx]:.8g}."
                )
    else:
        theta_init = _resolve_init_vector(root, priors, m0_bounds)
        print("[init] mode='manual' — using YAML 'init'")

    spread = _resolve_spread(root)
    print(f"[init] θ_init = {theta_init!r}  (a, λ0, m0, h, k, p, q)")
    print(f"[init] spreads = {spread}")

    p0 = draw_walkers_around_theta_init(
        nwalkers=mconf["nwalkers"],
        theta_init=theta_init,
        a_bounds=a_bounds,
        m0_bounds=m0_bounds,
        e_max=e_max,
        la0_bounds=la0_bounds,
        spread=spread,
    )

    # ── (7) Optional fp dimension ──
    if bool(mconf["sample_fp"]):
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
            snr_maps=d["snr_maps"],
            xgrid=d["xgrid"],
            bkg=d["bkg"],
            noise=d["noise"],
        ))

    lp_kwargs = dict(
        instruments=instruments,
        a_bounds=a_bounds,
        m0_bounds=m0_bounds,
        noise_floor=noise_floor,
        ecc_prior=ecc_prior,
        ecc_beta_a=ecc_beta_a,
        ecc_beta_b=ecc_beta_b,
        e_max=e_max,
        orientation_isotropic=orientation_isotropic,
        pq_prior=pq_prior,
        snr_scale=snr_scale,
        weighting=weighting,
        likelihood_mode=mconf["likelihood_mode"],
        fp_bounds=mconf["fp_bounds"],
        fp_prior=mconf["fp_prior"],
    )

    # Optional pre-MCMC diagnostic plots (brute-force best solution).
    if init_mode == "bruteforce":
        try:
            _run_bruteforce_best_plots(
                params=params, theta_init=theta_init,
                instruments=instruments, lp_kwargs=lp_kwargs,
            )
        except Exception as e:
            print(f"[plots] (WARN) Brute-force diagnostic plots failed: {e}")

    # ── (9) Parallelization + RNG seed ──
    par  = _resolve_parallel(root)
    seed = root.get("random_seed", None)
    if seed is not None:
        np.random.seed(int(seed))

    # ── (10) Run MCMC ──
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

    # ── (11) Extract and save chain ──
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
    except emcee.autocorr.AutocorrError as err:
        print("[MCMC] IAT not reliable:", err)

    # ── (13) Plots ──
    try:
        _run_plots_from_yaml(
            params=params, sampler=sampler, flat=flat,
            instruments=instruments, lp_kwargs=lp_kwargs,
        )
    except Exception as plot_err:
        print(f"[plots] Skipped due to error: {plot_err}")

    # ── (14) GLRT + Off-tracks ──
    try:
        glrt_cfg = root.get("glrt", {}) or {}

        # User-facing convention: dr_min_px/dr_max_px are dimensionless FWHM
        # multipliers in this configuration file.  Thus dr_min_px=2 means
        # an off-track displacement of at least 2 × fwhm for every instrument.
        # Expert absolute-pixel offsets remain available as dr_min_abs_px/dr_max_abs_px.
        dr_min_abs_px = glrt_cfg.get("dr_min_abs_px", None)
        dr_max_abs_px = glrt_cfg.get("dr_max_abs_px", None)
        dr_min_fwhm = glrt_cfg.get("dr_min_fwhm", glrt_cfg.get("dr_min_px", 1.0))
        dr_max_fwhm = glrt_cfg.get("dr_max_fwhm", glrt_cfg.get("dr_max_px", 5.0))

        run_glrt_offtracks_and_save(
            params=params,
            mode=mconf["likelihood_mode"],
            sampler=sampler,
            flat=flat,
            lp_kwargs=lp_kwargs,
            instruments=instruments,
            noise_floor=noise_floor,
            weighting=weighting,
            n_off=int(glrt_cfg.get("n_off", 200)),
            dr_min_px=None if dr_min_abs_px is None else float(dr_min_abs_px),
            dr_max_px=None if dr_max_abs_px is None else float(dr_max_abs_px),
            dr_min_fwhm=None if dr_min_fwhm is None else float(dr_min_fwhm),
            dr_max_fwhm=None if dr_max_fwhm is None else float(dr_max_fwhm),
            seed=root.get("random_seed", None),
        )
    except Exception as glrt_err:
        print(f"[GLRT] Skipped due to error: {glrt_err}")

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