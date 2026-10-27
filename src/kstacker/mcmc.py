"""MCMC posterior assembly and sampling for Keplerian orbit inference.

Independent observing techniques contribute their own log likelihood. The
orbit prior is applied exactly once, regardless of the instrument count.
"""
from __future__ import annotations
import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple, Union, Mapping, Any
import h5py
import numpy as np
import emcee
from concurrent.futures import ProcessPoolExecutor

# Initialized once per process for vectorized parallel posterior evaluation.
_WORKER_LP_KWARGS: Optional[dict] = None
from tqdm import tqdm
try:
    from .utils import Params
    from .console import _color, _print_section, ProgressReporter, console_progress, interactive_progress, console_status
    from .orbital_geometry import wrap_2pi, mean_motion, _equinoctial_sky_tracks
    from .direct_imaging import (
        Instrument,
        _get,
        _noise_floor_build_value,
        _resolve_noise_floor,
        configure_instrument_statistics,
        _resolve_path_from_yaml,
        _prepare_runtime_config_from_files,
        _resolve_background_noise,
        _load_native_images,
        _load_convolved_images,
        _make_placeholder_profiles,
        _normalize_paco_interpolator,
        _resolve_radial_noise_recalibration,
        _effective_paco_oversampling,
        _load_paco_maps,
        _load_or_build_local_ring_maps,
        _recalibrate_classical_noise,
        _normalize_likelihood_mode,
        total_direct_imaging_log_likelihood,
        uses_flux_likelihood,
    )
except ImportError:
    from utils import Params
    from console import _color, _print_section, ProgressReporter, console_progress, interactive_progress, console_status
    from orbital_geometry import wrap_2pi, mean_motion, _equinoctial_sky_tracks
    from direct_imaging import (
        Instrument,
        _get,
        _noise_floor_build_value,
        _resolve_noise_floor,
        configure_instrument_statistics,
        _resolve_path_from_yaml,
        _prepare_runtime_config_from_files,
        _resolve_background_noise,
        _load_native_images,
        _load_convolved_images,
        _make_placeholder_profiles,
        _normalize_paco_interpolator,
        _resolve_radial_noise_recalibration,
        _effective_paco_oversampling,
        _load_paco_maps,
        _load_or_build_local_ring_maps,
        _recalibrate_classical_noise,
        _normalize_likelihood_mode,
        total_direct_imaging_log_likelihood,
        uses_flux_likelihood,
    )

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
    a_support = tuple(map(float, a_bounds))
    m_support = tuple(map(float, m0_bounds))
    if a_support[1] <= a_support[0]:
        raise ValueError("priors.a_bounds must have strictly increasing limits")
    if m_support[1] <= m_support[0]:
        raise ValueError("priors.m0_bounds must have strictly increasing limits: "
                         "the seven-parameter ensemble sampler cannot use a "
                         "zero-width stellar-mass prior. Use a physically "
                         "justified narrow interval if the mass is well known.")
    return a_support, m_support


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


def _flux_log_priors(samples: np.ndarray, instruments: Sequence[Instrument]) -> np.ndarray:
    """Sum independent flux priors, each once, for sampled amplitudes only."""
    out = np.zeros(len(samples), dtype=float)
    for inst in instruments:
        if inst.flux_index is not None:
            out += log_prior_fp(samples[:, inst.flux_index], inst.flux_bounds, inst.flux_prior)
    return out


def _flux_parameter_names(instruments: Sequence[Instrument]) -> list[str]:
    """Return extra parameter names in the exact MCMC-column order."""
    return [f"fp_{inst.name}" for inst in instruments if inst.flux_index is not None]


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
    likelihood_mode: str = "signed_snr",
    fp_bounds: tuple[float, float] = (1e-4, 1e2),
    fp_prior: str = "uniform",
) -> np.ndarray | float:
    """Evaluate one orbital prior and an independent likelihood per instrument.

    The first seven coordinates are shared Keplerian parameters. Subsequent
    coordinates are independently calibrated fluxes, ordered by instrument,
    but only for flux-mode instruments with `flux.sample: true`.

    A fixed instrument flux is not an MCMC coordinate and has no sampled
    prior. The same function is evaluated during emcee and tempering.
    """
    samples = np.asarray(theta, dtype=float)
    scalar = samples.ndim == 1
    if scalar:
        samples = samples[None, :]
    expected = 7 + len(_flux_parameter_names(instruments))
    if samples.ndim != 2 or samples.shape[1] != expected:
        raise ValueError(f"theta must have {expected} columns (7 orbital + {expected-7} flux)")
    prior = np.asarray(log_prior_hkpq(
        samples[:, :7], a_bounds=a_bounds, m0_bounds=m0_bounds,
        la0_bounds=(0.0, 2.0 * np.pi), e_max=e_max,
        orbit_direction=orbit_direction,
    ), dtype=float)
    prior += _flux_log_priors(samples, instruments)
    valid = np.isfinite(prior)
    answer = np.full(samples.shape[0], -np.inf)
    if np.any(valid):
        answer[valid] = prior[valid] + total_direct_imaging_log_likelihood(
            samples[valid], instruments=instruments,
            noise_floor=noise_floor, weighting=weighting,
            likelihood_mode=likelihood_mode, fp_bounds=fp_bounds,
        )
    return float(answer[0]) if scalar else answer


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
        min_log_likelihood=cfg.get("min_log_likelihood", None),
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
    min_log_likelihood: Optional[float] = None,
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

    log_likelihood = None
    if data.shape[1] >= 8:
        log_likelihood = np.asarray(data[:, -1], dtype=float)
        if min_log_likelihood is not None:
            valid = log_likelihood >= float(min_log_likelihood)
            data = data[valid]
            log_likelihood = log_likelihood[valid]

    if data.shape[0] == 0:
        raise RuntimeError(f"No ranked solutions available in {h5_path} after filtering.")

    return data, log_likelihood


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
    min_log_likelihood: Optional[float] = None,
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
    data, log_likelihood = _load_ranked_classical_rows_from_h5(h5_path, min_log_likelihood=min_log_likelihood)

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
                log_likelihood=None if log_likelihood is None else float(log_likelihood[i]),
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


def _run_emcee_phase(sampler, initial_state, steps: int, label: str, progress: bool):
    """Advance an emcee phase through the shared progress interface.

    Interactive terminals receive tqdm bars; redirected batch logs receive
    occasional full-line progress updates. The sampler's state updates are
    identical in both modes. This function does not retain burn-in samples
    after the production sampler is reset by its caller.
    """
    if steps <= 0:
        return initial_state
    stream = console_progress(
        sampler.sample(initial_state, iterations=int(steps)),
        desc=label, total=int(steps), disable=not bool(progress),
    )
    final_state = initial_state
    try:
        for final_state in stream:
            pass
    finally:
        stream.close()
    return final_state


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
        state = _run_emcee_phase(sampler, p0, burnin, "Burn-in", progress)
        sampler.reset()
        _run_emcee_phase(sampler, state, nsteps, "Production", progress)
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
        state = _run_emcee_phase(sampler, p0, burnin, "Burn-in", progress)
        sampler.reset()   # discard burn-in steps
        _run_emcee_phase(sampler, state, nsteps, "Production", progress)

    return sampler


def _prior_only_scalar(theta: np.ndarray, lp_kwargs: Mapping[str, Any]) -> float:
    """The exact same prior term used by the cold posterior in emcee."""
    samples = np.asarray(theta, dtype=float)
    instruments = lp_kwargs["instruments"]
    if samples.ndim != 1 or samples.size != 7 + len(_flux_parameter_names(instruments)):
        return -np.inf
    prior = float(log_prior_hkpq(
        samples[:7], a_bounds=lp_kwargs["a_bounds"],
        m0_bounds=lp_kwargs["m0_bounds"],
        la0_bounds=(0.0, 2.0*np.pi), e_max=lp_kwargs["e_max"],
        orbit_direction=lp_kwargs["orbit_direction"],
    ))
    if not np.isfinite(prior):
        return -np.inf
    return prior + float(_flux_log_priors(samples[None, :], instruments)[0])


def _likelihood_only_scalar(theta: np.ndarray, lp_kwargs: Mapping[str, Any]) -> float:
    """The likelihood terms used by adaptive parallel tempering."""
    return float(total_direct_imaging_log_likelihood(
        np.asarray(theta, dtype=float), instruments=lp_kwargs["instruments"],
        noise_floor=lp_kwargs["noise_floor"], weighting=lp_kwargs["weighting"],
        likelihood_mode=lp_kwargs["likelihood_mode"], fp_bounds=lp_kwargs["fp_bounds"],
    ))


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


class Tee:
    """Mirror science logs and render batch tqdm updates as occasional lines.

    In a TTY, native carriage-return progress remains interactive. For a shell
    job writing into a file, percentage updates are retained every 5% so that
    `tail -f` displays clear, compact progress without thousands of redraws.
    """

    def __init__(self, *streams):
        self.streams = streams
        self._last_percentage = -5

    def isatty(self):
        try:
            return bool(self.streams[0].isatty())
        except Exception:
            return False

    def write(self, data):
        # reddemcee may use its own tqdm rather than console_progress.
        if not self.isatty() and "\r" in data:
            match = re.search(r"(?<!\d)(\d{1,3})%", data)
            if match:
                percentage = int(match.group(1))
                if percentage < self._last_percentage:
                    self._last_percentage = -5
                if percentage < self._last_percentage + 5 and percentage != 100:
                    return
                self._last_percentage = percentage
                data = data.replace("\r", "").strip() + "\n"
            else:
                data = data.replace("\r", "\n")
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
    parameter_names: Optional[Sequence[str]] = None,
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
    if parameter_names is None:
        parameter_names = ["a", "la0", "m0", "h", "k", "p", "q"] + [f"fp_{i+1}" for i in range(ndim - 7)]
    if len(parameter_names) != ndim:
        raise ValueError("parameter_names count must equal chain dimension")

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
        if hasattr(sampler, "pt_sampler"):
            pt = sampler.pt_sampler
            f.attrs["sampler"] = "reddemcee"
            f.create_dataset("pt_betas_final", data=np.asarray(pt.betas, dtype=float))
            # Full temperature chains can be large: store them only on request.
            if bool(getattr(params, "save_tempered_chains", False)):
                f.create_dataset("pt_chain", data=np.asarray(pt.get_chain(flat=False)), compression="gzip")
            for key, getter in (("pt_swap_history", "get_tsw"), ("pt_smd_history", "get_smd")):
                try:
                    dataset = np.asarray(getattr(pt, getter)(), dtype=float)
                    if dataset.size:
                        f.create_dataset(key, data=dataset, compression="gzip")
                except (AttributeError, ValueError, TypeError):
                    pass
        else:
            f.attrs["sampler"] = "emcee"

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


def derive_physical_from_chain(a, la0, m0, h, k, p, q, t_ref):
    """
    Convert the non-singular state to gauge-chosen classical quantities for
    human-readable tables only. The likelihood never performs this inverse conversion.

    Coordinate definitions:
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
    7.  Build Instrument objects; assign noise floor and flux for each dataset.
    8.  Optionally append one flux dimension per flux-sampling instrument.
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
    initial_logging = params._params.get("logging", {}) or {}
    if "color" in initial_logging:
        os.environ["KSTACKER_COLOR"] = str(initial_logging["color"])
    if "progress_mode" in initial_logging:
        os.environ["KSTACKER_PROGRESS"] = str(initial_logging["progress_mode"])
    root = _prepare_runtime_config_from_files(yaml_path, params._params, verbose=True)
    params._params = root
    params.work_dir = str(Path(yaml_path).expanduser().resolve().parent)
    base_root = dict(root)
    console_cfg = root.get("logging", {}) or {}
    if "progress_mode" in console_cfg:
        os.environ["KSTACKER_PROGRESS"] = str(console_cfg["progress_mode"])
    if "color" in console_cfg:
        os.environ["KSTACKER_COLOR"] = str(console_cfg["color"])
    base_root["save_tempered_chains"] = bool((root.get("outputs", {}) or {}).get("save_tempered_chains", False))

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
                    or _noise_floor_build_value(tmp_root) == 0.0
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
                    noise_floor=_noise_floor_build_value(tmp_root),
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
            likelihood_mode=_normalize_likelihood_mode(inst_cfg.get("likelihood_mode", (root.get("mcmc", {}) or {}).get("likelihood_mode", "signed_snr"))),
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

    # Seed before any walker/flux drawing to make full initialisation reproducible.
    seed = root.get("random_seed", None)
    if seed is not None:
        np.random.seed(int(seed))

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
            min_log_likelihood=None,
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
            centres_meta = [dict(rank=1, log_likelihood=None, source="manual")]
        else:
            theta_centres, centres_meta = _build_theta_centres_from_ranked_h5(
                ranked_h5,
                t_ref=t_ref,
                n_centres=multistart_cfg["n_starts"],
                min_log_likelihood=multistart_cfg["min_log_likelihood"],
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
            score_text = "n/a" if meta_c.get("log_likelihood") is None else f"{meta_c['log_likelihood']:.6f}"
            track_text = meta_c.get("min_track_dmax_px", None)
            track_text = "first centre" if track_text is None else f"{track_text:.3f} px"
            print(f"[init]   centre {idx_c}: rank={meta_c.get('rank')} | logL={score_text} | nearest retained-track dmax={track_text}")
            if row_c is not None:
                a_c, e_c, t0_c, m0_c, Omega_c, inc_c, argperi_c = map(float, row_c[:7])
                n_c = 2.0 * np.pi * np.sqrt(m0_c / a_c**3)
                M0_c = wrap_2pi(n_c * (t_ref - t0_c))
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

    # ── (8) Build Instrument objects ──
    instruments: list = []
    for d in per_inst_data:
        instruments.append(Instrument(
            name=d["name"],
            likelihood_mode=d["likelihood_mode"],
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

    # Each detector owns its sigma floor and its optional sampled planet flux.
    # Shared orbital coordinates occupy columns 0:7; sampled amplitudes follow.
    flux_names = configure_instrument_statistics(root, instruments, verbose=True)
    if flux_names:
        columns = []
        for inst in instruments:
            if inst.flux_index is None:
                continue
            if inst.flux_prior == "loguniform":
                values = np.exp(np.log(inst.flux_init) + inst.flux_spread * np.random.randn(len(p0)))
            else:
                scale = max(abs(inst.flux_init), 1.0e-8)
                values = inst.flux_init + scale * inst.flux_spread * np.random.randn(len(p0))
            low, high = inst.flux_bounds
            margin = (high - low) * 1.0e-9
            columns.append(np.clip(values, low + margin, high - margin))
        p0 = np.column_stack([p0] + columns)
        print("[init] sampled instrument fluxes: " + ", ".join(flux_names))
    # The entry-point likelihood always resolves the floor from its Instrument.
    # This fallback is used only by lower-level calls without instrument metadata.
    noise_floor = min(float(inst.noise_floor) for inst in instruments)

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
    # ── (10) Run MCMC ──
    _print_section("MCMC CONFIGURATION")
    print(f"  sampler              : {mconf['sampler']}")
    print("  likelihoods          : " + ", ".join(f"{d['name']}={d['likelihood_mode']}" for d in per_inst_data))
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
            parameter_names=["a", "la0", "m0", "h", "k", "p", "q"] + flux_names,
        )

    # ── (12) Scientific and convergence plots ──
    plot_config = root.get("plots", {}) or {}
    if plot_config.get("enabled", True):
        try:
            try:
                from .scientific_plots import create_scientific_figures
                from .mcmc_diagnostics import create_mcmc_diagnostics
            except ImportError:
                from scientific_plots import create_scientific_figures
                from mcmc_diagnostics import create_mcmc_diagnostics
            plot_dir = Path(params.get_path("values_dir")) / str(plot_config.get("directory", "figures"))
            plot_dir.mkdir(parents=True, exist_ok=True)
            # Draw corners from the configurable thinned visualization sample,
            # but always include the true highest-posterior production state.
            # Otherwise a MAP annotation can depend on the YAML thinning stride.
            full_log_probability = sampler.get_log_prob(flat=False)
            map_flat_index = int(np.nanargmax(full_log_probability))
            map_step, map_walker = np.unravel_index(
                map_flat_index, full_log_probability.shape
            )
            map_state = sampler.get_chain(flat=False)[map_step, map_walker]
            display_states = np.vstack([flat, map_state[None, :]])
            display_scores = np.concatenate([
                sampler.get_log_prob(thin=mconf["thin"], flat=True),
                np.array([full_log_probability[map_step, map_walker]]),
            ])
            create_scientific_figures(
                instruments, display_states, display_scores,
                output_dir=plot_dir / "science", noise_floor=noise_floor,
                max_posterior_samples=int(plot_config.get("max_posterior_samples", 12000)),
                max_orbit_tracks=int(plot_config.get("max_orbit_tracks", 900)),
                dpi=int(plot_config.get("dpi", 200)),
                mask_to_working_region=bool(plot_config.get("mask_to_working_region", True)),
                align_to_first_epoch=bool(plot_config.get("align_to_first_epoch", True)),
            )
            create_mcmc_diagnostics(
                sampler, output_dir=plot_dir / "convergence", thin=mconf["thin"],
                dpi=int(plot_config.get("dpi", 200)),
                parameter_names=["a", "la0", "m0", "h", "k", "p", "q"] + flux_names,
            )

        except Exception as exc:
            console_status("plots", f"Figure export failed: {exc}", tone="red")
            print("[plots] The saved chain can be plotted separately with the plot command.")

    # ── (13) Sampler summary ──
    acc = float(np.mean(sampler.acceptance_fraction))
    print(f"[MCMC] mean acceptance fraction = {acc:.3f}")
    try:
        tau   = sampler.get_autocorr_time(discard=0, thin=mconf["thin"], tol=0)
        names = ["a", "la0", "m0", "h", "k", "p", "q"] + flux_names
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


