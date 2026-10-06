#!/usr/bin/env python3
# =============================================================================
# mcmc_init_search.py
#
# Explicit physical-grid search used to initialise orbital MCMC sampling.
#
# The search scans:
#   a [AU], e, i [deg], M0 [deg], omega [deg], theta0 [deg]
#
# Every physical orbit is converted to the non-singular MCMC coordinates
#
#   theta = (a, lambda0, m0, h, k, p, q)
#
# before it is scored. The same photometry, noise model, radial noise
# recalibration, masks, and orbit propagation as mcmc.py are used.
#
# Supported data backends:
#   - convolve
#   - aperture
#   - paco
#
# The search is a deterministic grid. It does not use Sobol points or another
# quasi-random global design.
#
# Outputs are written in the YAML directory under values/ by default:
#   values/mcmc_init_search.h5
#   values/mcmc_init_search_top.csv
# =============================================================================

from __future__ import annotations

import os
import json
from pathlib import Path
from itertools import product
from typing import Any, Iterator, List, Optional, Sequence

import h5py
import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        return iterable

try:
    from .mcmc import (
        Instrument,
        Params,
        _prepare_runtime_config_from_files,
        _resolve_background_noise,
        _resolve_bounds,
        _resolve_tref,
        _resolve_weighting,
        _noise_floor_build_value,
        _resolve_noise_floor,
        _print_section,
        _resolve_radial_noise_recalibration,
        _recalibrate_classical_noise,
        _load_convolved_images,
        _load_native_images,
        _load_or_build_local_ring_maps,
        _load_paco_maps,
        _normalize_paco_interpolator,
        _effective_paco_oversampling,
        _make_placeholder_profiles,
        positive_snr_profile_multi_from_hkpq,
        snr_multi_from_hkpq,
        wrap_2pi,
    )
except Exception:
    from mcmc import (
        Instrument,
        Params,
        _prepare_runtime_config_from_files,
        _resolve_background_noise,
        _resolve_bounds,
        _resolve_tref,
        _resolve_weighting,
        _noise_floor_build_value,
        _resolve_noise_floor,
        _print_section,
        _resolve_radial_noise_recalibration,
        _recalibrate_classical_noise,
        _load_convolved_images,
        _load_native_images,
        _load_or_build_local_ring_maps,
        _load_paco_maps,
        _normalize_paco_interpolator,
        _effective_paco_oversampling,
        _make_placeholder_profiles,
        positive_snr_profile_multi_from_hkpq,
        snr_multi_from_hkpq,
        wrap_2pi,
    )


# =============================================================================
# CONFIG HELPERS
# =============================================================================

def _resolve_init_search_config(root: dict) -> dict:
    """
    Parse the YAML `init_search` section.

    The pre-MCMC search is intentionally grid-only.  The user edits each
    physical axis with three explicit numbers: `min`, `max`, and `step`.
    The explored points are therefore explicit and directly reproducible.

    The six scanned orbital coordinates are:
        a [AU], e, i [deg], M0 [deg], theta0=omega_argperi [deg],
        omega=Omega_node [deg].

    Stellar mass is fixed during the grid by `init_search.stellar_mass_solar`.
    The resulting physical orbits are converted exactly to the non-singular
    MCMC coordinates (a, lambda_ref, m0, h, k, p, q) before S/N evaluation.
    """
    cfg = root.get("init_search", {}) or {}
    grid = cfg.get("physical_grid", {}) or {}
    priors = root.get("priors", {}) or {}
    mass_bounds = priors.get("m0_bounds", (1.0, 1.0))
    default_mass = 0.5 * (float(mass_bounds[0]) + float(mass_bounds[1]))

    def axis_cfg(name: str, default_min: float, default_max: float, default_step: float) -> dict:
        item = grid.get(name, {}) or {}
        return dict(
            min=float(item.get("min", default_min)),
            max=float(item.get("max", default_max)),
            step=float(item.get("step", default_step)),
        )

    return dict(
        enabled=bool(cfg.get("enabled", True)),
        method="grid",
        search_backend=str(cfg.get("search_backend", "auto") or "auto").lower(),
        weighting=str(cfg.get("weighting", "auto") or "auto").lower(),
        score_mode=str(cfg.get("score_mode", "auto") or "auto").lower(),
        top_n_save=max(1, int(cfg.get("top_n_save", 500))),
        output_h5=str(cfg.get("output_h5", "mcmc_init_search.h5") or "mcmc_init_search.h5"),
        output_csv=str(cfg.get("output_csv", "mcmc_init_search_top.csv") or "mcmc_init_search_top.csv"),
        grid_batch_size=max(1, int(cfg.get("grid_batch_size", 20000))),
        m0_solar=float(cfg.get("stellar_mass_solar", default_mass)),
        physical_grid=dict(
            a_au=axis_cfg("a_au", 8.0, 30.0, 2.0),
            e=axis_cfg("e", 0.0, 0.9, 0.1),
            i_deg=axis_cfg("i_deg", 0.0, 180.0, 10.0),
            M0_deg=axis_cfg("M0_deg", 0.0, 350.0, 10.0),
            theta0_deg=axis_cfg("theta0_deg", 0.0, 350.0, 10.0),
            omega_deg=axis_cfg("omega_deg", 0.0, 350.0, 10.0),
        ),
    )


def _resolve_parameter_supports(root: dict, params: "Params", init_cfg: dict) -> dict:
    """Resolve the MCMC prior support used to validate the physical grid."""
    priors = root.get("priors", {}) or {}
    a_bounds, m0_bounds = _resolve_bounds(params, priors)
    la0_bounds = tuple(map(float, priors.get("la0_bounds", (0.0, 2.0 * np.pi))))
    e_max = float(priors.get("e_max", 0.95))
    orbit_direction = str(priors.get("orbit_direction", "any") or "any").strip().lower()
    if orbit_direction not in ("any", "prograde", "retrograde"):
        raise ValueError("priors.orbit_direction must be 'any', 'prograde', or 'retrograde'.")
    return dict(
        a_bounds=a_bounds, m0_bounds=m0_bounds, la0_bounds=la0_bounds,
        e_max=e_max, orbit_direction=orbit_direction,
    )


# =============================================================================
# INSTRUMENT LOADING
# =============================================================================


def _load_instruments_for_search(
    params: "Params",
    root: dict,
    *,
    t_ref: float,
    search_backend: str,
    instrument_ts: Optional[Sequence[np.ndarray]] = None,
) -> List[Instrument]:
    """
    Build the same Instrument objects used by the MCMC, but for the backend
    requested by the initialisation search.

    Backend policy
    --------------
    "auto":
        follow each instrument YAML block.

    explicit backend:
        force every instrument to use the same backend, which is useful for
        controlled comparisons between:
          - convolve
          - aperture
          - paco

    instrument_ts:
        Optional precomputed per-instrument time vectors. When provided,
        this avoids calling params.get_ts() a second time and prevents the
        duplicated "time vector" console printouts.
    """
    instruments_cfg = root.get("instruments", None)
    if not isinstance(instruments_cfg, (list, tuple)):
        raise ValueError("`instruments` must be a list of instrument configs.")

    base_root = dict(root)
    instruments: List[Instrument] = []
    background_noise_cfg = _resolve_background_noise(root)

    print()
    print("[init_search] local aperture-ring background/noise")
    print(
        "[init_search] small_sample_correction = "
        f"{background_noise_cfg.get('small_sample_correction')}"
    )
    print(
        "[init_search] student_small_n_threshold = "
        f"{background_noise_cfg.get('student_small_n_threshold')}"
    )
    print(
        "[init_search] min_reference_apertures = "
        f"{background_noise_cfg.get('min_reference_apertures')}"
    )
    print()

    for inst_cfg in instruments_cfg:
        if not isinstance(inst_cfg, dict):
            raise ValueError("Each instrument entry must be a dictionary.")

        tmp_root = dict(base_root)
        tmp_root.update(inst_cfg)
        params._params = tmp_root
        radial_recal_cfg = _resolve_radial_noise_recalibration(base_root, inst_cfg)

        if search_backend == "auto":
            photometry_method = str(getattr(params, "method", "convolve") or "convolve").lower()
        else:
            photometry_method = str(search_backend).lower()
        if photometry_method not in ("convolve", "aperture", "paco"):
            raise ValueError(
                "init_search.search_backend must be 'auto', 'convolve', 'aperture', or 'paco', "
                "and each instrument method must resolve to one of those three backends."
            )

        if instrument_ts is not None:
            ts_i = np.asarray(instrument_ts[len(instruments)], dtype=float)
        else:
            ts_i = np.asarray(params.get_ts(use_p_prev=True), dtype=float)

        size_i = params.n
        scale_i = params.scale
        upsampling_factor_i = getattr(params, "upsampling_factor", 1)
        fwhm_i = getattr(params, "fwhm", None)
        r_mask_i = getattr(params, "r_mask", None)
        r_mask_ext_i = getattr(params, "r_mask_ext", None)

        images_up = None
        images_native = None
        paco_alpha_maps = None
        paco_var_alpha_maps = None
        paco_interpolator = "none"
        paco_oversampling = 1
        local_bkg_maps = None
        local_noise_maps = None

        if photometry_method == "paco":
            # Use the exact PACO loader and likelihood helper shared with mcmc.py.
            # No classical profiles, local-ring maps, or likelihood images are loaded.
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
            # The init search does not need native images for its objective.
            images_native = None
        else:
            use_local_ring = background_noise_cfg["mode"] == "local_aperture_ring"
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
            images_native_suffix = str(inst_cfg.get("native_images_suffix", "_preprocessed"))
            images_native = _load_native_images(params, suffix=images_native_suffix)
            if photometry_method == "convolve":
                images_up = _load_convolved_images(params)

            # Classical background and noise are estimated only with local
            # reference apertures. These arrays keep the Instrument interface
            # uniform and are not used as science noise estimates.
            xgrid, bkg, noise = _make_placeholder_profiles(
                len(ts_i),
                size_i,
            )

            if use_precomputed_local_maps:
                local_bkg_maps, local_noise_maps = _load_or_build_local_ring_maps(
                    profile_dir=profile_dir,
                    photometry_method=photometry_method,
                    size=size_i,
                    upsampling_factor=upsampling_factor_i,
                    fwhm=fwhm_i,
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
                    upsampling_factor=float(upsampling_factor_i),
                    fwhm=float(fwhm_i),
                    r_mask=r_mask_i,
                    xgrid=xgrid,
                    bkg=bkg,
                    noise=noise,
                    local_bkg_maps=local_bkg_maps,
                    local_noise_maps=local_noise_maps,
                    cfg=radial_recal_cfg,
                    profile_dir=profile_dir,
                )

        inst_name = str(inst_cfg.get("name", f"instrument_{len(instruments)}"))

        instruments.append(
            Instrument(
                name=inst_name,
                size=size_i,
                scale=scale_i,
                upsampling_factor=upsampling_factor_i,
                fwhm=fwhm_i,
                r_mask=r_mask_i,
                r_mask_ext=r_mask_ext_i,
                t_ref=t_ref,
                ts=np.asarray(ts_i, dtype=float),
                photometry_method=photometry_method,
                images_up=None if images_up is None else np.asarray(images_up),
                images_native=None if images_native is None else np.asarray(images_native),
                paco_alpha_maps=None if paco_alpha_maps is None else np.asarray(paco_alpha_maps),
                paco_var_alpha_maps=None if paco_var_alpha_maps is None else np.asarray(paco_var_alpha_maps),
                paco_interpolator=paco_interpolator,
                paco_oversampling=paco_oversampling,
                xgrid=np.asarray(xgrid, dtype=float),
                bkg=np.asarray(bkg, dtype=float),
                noise=np.asarray(noise, dtype=float),
                bgnoise_cfg=background_noise_cfg,
                local_bkg_maps=None if local_bkg_maps is None else np.asarray(local_bkg_maps),
                local_noise_maps=None if local_noise_maps is None else np.asarray(local_noise_maps),
            )
        )

    params._params = base_root
    return instruments


# =============================================================================
# PARAMETER CONVERSIONS
# =============================================================================

def _hkpq_to_classical(
    theta7: np.ndarray,
    *,
    t_ref: float,
) -> np.ndarray:
    """
    Convert the NEW non-singular coordinates to gauge-chosen classical rows for
    human-readable ranked output only.

    Sampling definitions:
        h = e sin(Ω+ω),  k = e cos(Ω+ω)
        p = sin(i/2) sinΩ, q = sin(i/2) cosΩ

    Gauge choices at singular limits preserve the physical longitude λ:
        i=0 -> Ω=0
        e=0 -> ω=0 and varpi=Ω
    """
    theta7 = np.asarray(theta7, dtype=float)
    if theta7.ndim != 2 or theta7.shape[1] != 7:
        raise ValueError("theta7 must have shape (N, 7).")

    a, la0, m0, h, k, p, q = theta7.T
    e = np.hypot(h, k)
    r = np.sqrt(np.clip(p*p + q*q, 0.0, 1.0))
    inc = 2.0 * np.arcsin(r)

    Omega_raw = np.arctan2(p, q)
    Omega = np.where(r > 1e-15, Omega_raw, 0.0)
    varpi_raw = np.arctan2(h, k)
    varpi = np.where(e > 1e-15, varpi_raw, Omega)
    argperi = wrap_2pi(varpi - Omega)
    Omega = wrap_2pi(Omega)

    M0 = wrap_2pi(la0 - varpi)
    n = 2.0 * np.pi * np.sqrt(m0 / (a ** 3))
    t0 = float(t_ref) - (M0 / n)

    return np.column_stack([a, e, t0, m0, Omega, inc, argperi])


# =============================================================================
# GRID SEARCH ENGINE
# =============================================================================

def _stepped_axis(low: float, high: float, step: float) -> np.ndarray:
    """Return an inclusive fixed-step axis without appending an irregular endpoint."""
    low, high, step = float(low), float(high), float(step)
    if step <= 0.0:
        raise ValueError("Grid steps must be strictly positive.")
    if high < low:
        raise ValueError(f"Grid maximum {high} is smaller than minimum {low}.")
    n = int(np.floor((high - low) / step + 1.0e-12))
    values = low + step * np.arange(n + 1, dtype=float)
    return values[values <= high + 1.0e-10]


def _physical_grid_axes(init_cfg: dict) -> dict:
    """Build the six user-facing physical axes from min/max/step settings."""
    axes = {}
    for name, spec in init_cfg["physical_grid"].items():
        axes[name] = _stepped_axis(spec["min"], spec["max"], spec["step"])
    return axes


def _physical_batch_to_theta(arr: np.ndarray, *, m0_solar: float, t_ref: float) -> np.ndarray:
    """Convert [a,e,i_deg,M0_deg,theta0_deg,omega_deg] rows to MCMC coordinates."""
    arr = np.asarray(arr, dtype=float)
    a, e, i_deg, M0_deg, theta0_deg, omega_deg = arr.T
    inc = np.radians(i_deg)
    M0 = np.radians(M0_deg)
    theta0 = np.radians(theta0_deg)
    Omega = np.radians(omega_deg)
    m0 = np.full_like(a, float(m0_solar), dtype=float)

    n = 2.0 * np.pi * np.sqrt(m0 / a**3)
    M_ref = wrap_2pi(M0 + n * float(t_ref))
    varpi = Omega + theta0
    la0 = wrap_2pi(M_ref + varpi)
    h = e * np.sin(varpi)
    k = e * np.cos(varpi)
    s = np.sin(0.5 * inc)
    p = s * np.sin(Omega)
    q = s * np.cos(Omega)
    return np.column_stack([a, la0, m0, h, k, p, q]).astype(float)


def _iter_grid_theta_batches(*, init_cfg: dict, support: dict, t_ref: float) -> Iterator[np.ndarray]:
    """
    Yield batches from the explicit physical grid.

    The singular limits follow the diagnostic grid convention:
      - e = 0       -> theta0 is not independently scanned;
      - i = 0/180   -> omega is not independently scanned.
    """
    axes = _physical_grid_axes(init_cfg)
    batch_size = int(init_cfg["grid_batch_size"])
    m0_solar = float(init_cfg["m0_solar"])

    if not (support["m0_bounds"][0] <= m0_solar <= support["m0_bounds"][1]):
        raise ValueError(
            f"init_search.stellar_mass_solar={m0_solar} lies outside "
            f"priors.m0_bounds={support['m0_bounds']}."
        )

    rows = []
    for a in axes["a_au"]:
        for e in axes["e"]:
            if not (support["a_bounds"][0] <= a <= support["a_bounds"][1]):
                continue
            if not (0.0 <= e <= support["e_max"]):
                continue
            theta0_values = np.asarray([0.0]) if np.isclose(e, 0.0, atol=1e-12) else axes["theta0_deg"]
            for inc in axes["i_deg"]:
                direction = support.get("orbit_direction", "any")
                if direction == "prograde" and inc > 90.0 + 1e-12:
                    continue
                if direction == "retrograde" and inc < 90.0 - 1e-12:
                    continue
                face_on = np.isclose(inc, 0.0, atol=1e-12) or np.isclose(inc, 180.0, atol=1e-12)
                omega_values = np.asarray([0.0]) if face_on else axes["omega_deg"]
                for M0, theta0, omega in product(axes["M0_deg"], theta0_values, omega_values):
                    rows.append((a, e, inc, M0, theta0, omega))
                    if len(rows) >= batch_size:
                        yield _physical_batch_to_theta(np.asarray(rows), m0_solar=m0_solar, t_ref=t_ref)
                        rows = []
    if rows:
        yield _physical_batch_to_theta(np.asarray(rows), m0_solar=m0_solar, t_ref=t_ref)


def _count_physical_grid(init_cfg: dict, support: dict) -> int:
    """Count the exact number of valid physical grid orbits, including singular reductions."""
    axes = _physical_grid_axes(init_cfg)
    total = 0
    for a in axes["a_au"]:
        if not (support["a_bounds"][0] <= a <= support["a_bounds"][1]):
            continue
        for e in axes["e"]:
            if not (0.0 <= e <= support["e_max"]):
                continue
            n_theta0 = 1 if np.isclose(e, 0.0, atol=1e-12) else len(axes["theta0_deg"])
            for inc in axes["i_deg"]:
                direction = support.get("orbit_direction", "any")
                if direction == "prograde" and inc > 90.0 + 1e-12:
                    continue
                if direction == "retrograde" and inc < 90.0 - 1e-12:
                    continue
                n_omega = 1 if (np.isclose(inc, 0.0, atol=1e-12) or np.isclose(inc, 180.0, atol=1e-12)) else len(axes["omega_deg"])
                total += len(axes["M0_deg"]) * n_theta0 * n_omega
    return int(total)


# =============================================================================
# OUTPUT KEEPERS
# =============================================================================

def _keep_top_rows(existing: Optional[np.ndarray], new_rows: np.ndarray, *, top_n: int) -> np.ndarray:
    """
    Keep only the highest-SNR rows seen so far.

    The last column is assumed to be the ranking score.
    """
    if existing is None or existing.size == 0:
        merged = np.asarray(new_rows, dtype=float)
    else:
        merged = np.vstack([existing, np.asarray(new_rows, dtype=float)])

    if merged.shape[0] <= int(top_n):
        order = np.argsort(np.abs(merged[:, -1]))[::-1]
        return merged[order]

    idx = np.argpartition(np.abs(merged[:, -1]), -int(top_n))[-int(top_n):]
    top = merged[idx]
    order = np.argsort(np.abs(top[:, -1]))[::-1]
    return top[order]


def _save_ranked_results(
    *,
    values_dir: str,
    output_h5: str,
    output_csv: str,
    classical_rows: np.ndarray,
    nonsingular_rows: np.ndarray,
    meta: dict,
) -> tuple[str, str]:
    """
    Save ranked results in both HDF5 and CSV form.
    """
    os.makedirs(values_dir, exist_ok=True)

    h5_path = os.path.join(values_dir, output_h5)
    csv_path = os.path.join(values_dir, output_csv)

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("Best solutions", data=np.asarray(classical_rows, dtype=float))
        f.create_dataset("Best solutions nonsingular", data=np.asarray(nonsingular_rows, dtype=float))
        for key, value in meta.items():
            if isinstance(value, (dict, list, tuple)):
                f.attrs[key] = json.dumps(value)
            else:
                f.attrs[key] = value

    header = ",".join([
        "a_au", "e", "t0_years", "m0_solar", "omega_rad", "inc_rad", "theta0_rad",
        "signal", "noise", "snr",
        "a_au_ns", "lambda0_rad_ns", "m0_solar_ns", "h", "k", "p", "q",
    ])
    merged = np.hstack([classical_rows, nonsingular_rows[:, :7]])
    np.savetxt(csv_path, merged, delimiter=",", header=header, comments="")

    return h5_path, csv_path


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

def search_mcmc_initial_parameters_from_yaml(yaml_path: str):
    """
    Run the dedicated ranked initialisation search from a YAML file.

    Important design choice
    -----------------------
    The search is GRID ONLY.  `score_mode` chooses either:
      - positive_snr_profile: sqrt(sum max(0,z_k)^2), PACO only;
      - signed_snr:  signed common-flux S/N, ranked by |S/N|.

    No quasi-random or random global design is used.
    """
    params = Params.read(yaml_path)
    root = _prepare_runtime_config_from_files(yaml_path, params._params, verbose=True)
    params._params = root
    params.work_dir = str(Path(yaml_path).expanduser().resolve().parent)

    init_cfg = _resolve_init_search_config(root)

    weighting = _resolve_weighting(root) if init_cfg["weighting"] == "auto" else str(init_cfg["weighting"]).lower()
    if weighting not in ("invvar", "simple"):
        raise ValueError("init_search.weighting must be 'auto', 'invvar', or 'simple'.")

    support = _resolve_parameter_supports(root, params, init_cfg)

    instruments_cfg = root.get("instruments", None)
    if not isinstance(instruments_cfg, (list, tuple)):
        raise ValueError("`instruments` must be a list.")

    instrument_ts = []
    base_root = dict(root)
    for inst_cfg in instruments_cfg:
        tmp_root = dict(base_root)
        tmp_root.update(inst_cfg)
        params._params = tmp_root
        instrument_ts.append(np.asarray(params.get_ts(use_p_prev=True), dtype=float))
    params._params = base_root

    if not instrument_ts:
        raise RuntimeError("No instrument time vectors found in the YAML.")

    t_ref = _resolve_tref(params, root, np.concatenate(instrument_ts))
    search_backend = init_cfg["search_backend"]
    instruments = _load_instruments_for_search(
        params,
        root,
        t_ref=t_ref,
        search_backend=search_backend,
        instrument_ts=instrument_ts,
    )

    requested_score_mode = str(init_cfg["score_mode"]).lower()
    if requested_score_mode == "auto":
        active_like = str((root.get("mcmc", {}) or {}).get(
            "likelihood_mode", "positive_snr_profile"
        )).lower()
        if active_like in {"snr", "positive", "positive_snr", "positive_snr_profile"} and all(
            inst.photometry_method == "paco" for inst in instruments
        ):
            score_mode = "positive_snr_profile"
        else:
            score_mode = "signed_snr"
    else:
        aliases = {
            "snr": "positive_snr_profile",
            "positive": "positive_snr_profile",
            "snr_signed": "signed_snr",
        }
        score_mode = aliases.get(requested_score_mode, requested_score_mode)

    if score_mode not in {"positive_snr_profile", "signed_snr"}:
        raise ValueError(
            "init_search.score_mode must be 'auto', 'positive_snr_profile', or 'signed_snr'."
        )
    if score_mode == "positive_snr_profile" and any(
        inst.photometry_method != "paco" for inst in instruments
    ):
        raise ValueError(
            "init_search.score_mode='positive_snr_profile' is PACO-only. "
            "Use 'signed_snr' for convolve/aperture."
        )

    noise_floor = _resolve_noise_floor(root, instruments, verbose=True)
    values_dir = params.get_path("values_dir")

    _print_section("MCMC INITIALISATION SEARCH", width=100)
    print(f"[search] yaml_path                : {yaml_path}")
    print(f"[search] method                  : {init_cfg['method']}")
    print(f"[search] search_backend          : {search_backend}")
    print(f"[search] score_mode              : {score_mode}")
    print(f"[search] weighting               : {weighting}")
    print(f"[search] top_n_save              : {init_cfg['top_n_save']}")
    print(f"[search] values_dir              : {values_dir}")
    print(f"[search] t_ref                   : {t_ref:.6f}")
    print(f"[search] a_bounds (prior)        : {support['a_bounds']}")
    print(f"[search] m0_bounds (prior)       : {support['m0_bounds']}")
    print(f"[search] e_max (prior)           : {support['e_max']}")
    print(f"[search] fixed m0_solar          : {init_cfg['m0_solar']}")
    print("[search] physical grid (min, max, step):")
    for grid_name, grid_spec in init_cfg["physical_grid"].items():
        print(f"[search]   - {grid_name:12s}: {grid_spec['min']:g}, {grid_spec['max']:g}, {grid_spec['step']:g}")
    print(f"[search] exact grid size         : {_count_physical_grid(init_cfg, support):,}")
    print(f"[search] orbit_direction         : {support['orbit_direction']}")
    print(f"[search] n_instruments           : {len(instruments)}")
    for inst in instruments:
        print(
            f"[search]   - {inst.name}: method={inst.photometry_method}, "
            f"n_epochs={len(inst.ts)}, size={inst.size}, fwhm={inst.fwhm}"
            + (f", paco_interpolator={inst.paco_interpolator}, paco_oversampling={inst.paco_oversampling}"
               if inst.photometry_method == "paco" else "")
        )
    print("=" * 100)

    top_classical = None
    top_nonsingular = None
    n_tested = 0
    n_kept = 0

    exact_grid_size = _count_physical_grid(init_cfg, support)
    batch_size = int(init_cfg["grid_batch_size"])
    n_batches_total = max(1, int(np.ceil(exact_grid_size / float(batch_size))))

    iterator = _iter_grid_theta_batches(
        init_cfg=init_cfg,
        support=support,
        t_ref=t_ref,
    )

    progress_bar = tqdm(
        iterator,
        total=n_batches_total,
        desc="Initial grid search",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
    )

    for batch_index, theta_batch in enumerate(progress_bar, start=1):
        theta_batch = np.asarray(theta_batch, dtype=float)
        if theta_batch.ndim != 2 or theta_batch.shape[1] != 7:
            raise ValueError("Internal error: theta batch must have shape (N, 7).")

        if score_mode == "positive_snr_profile":
            signal, noise, snr, criterion = positive_snr_profile_multi_from_hkpq(
                theta_batch,
                instruments,
                noise_floor=noise_floor,
                weighting=weighting,
            )
        else:
            signal, noise, snr = snr_multi_from_hkpq(
                theta_batch,
                instruments,
                noise_floor=noise_floor,
                weighting=weighting,
            )
            criterion = np.asarray(snr, dtype=float) ** 2

        classical = _hkpq_to_classical(theta_batch, t_ref=t_ref)

        classical_rows = np.column_stack([classical, signal, noise, snr])
        nonsingular_rows = np.column_stack([theta_batch, signal, noise, snr])

        valid = np.isfinite(classical_rows).all(axis=1) & np.isfinite(nonsingular_rows).all(axis=1)
        valid &= np.isfinite(snr)

        classical_rows = classical_rows[valid]
        nonsingular_rows = nonsingular_rows[valid]

        n_tested += int(theta_batch.shape[0])
        n_kept += int(np.sum(valid))

        top_classical = _keep_top_rows(
            top_classical,
            classical_rows,
            top_n=init_cfg["top_n_save"],
        )
        top_nonsingular = _keep_top_rows(
            top_nonsingular,
            nonsingular_rows,
            top_n=init_cfg["top_n_save"],
        )

        best_snr_here = (
            float(top_classical[0, -1])
            if top_classical is not None and top_classical.size
            else float("nan")
        )
        try:
            progress_bar.set_postfix(
                tested=f"{n_tested:,}",
                kept=f"{n_kept:,}",
                best=f"{best_snr_here:.3f}",
                refresh=False,
            )
        except Exception:
            pass

    if top_classical is None or top_classical.size == 0:
        raise RuntimeError("The initialisation search did not produce any valid candidate.")

    meta = dict(
        yaml_path=str(yaml_path),
        method=init_cfg["method"],
        search_backend=search_backend,
        weighting=weighting,
        score_mode=score_mode,
        n_tested=int(n_tested),
        n_kept=int(n_kept),
        t_ref=float(t_ref),
        a_bounds=list(map(float, support["a_bounds"])),
        la0_bounds=list(map(float, support["la0_bounds"])),
        m0_bounds=list(map(float, support["m0_bounds"])),
        e_max=float(support["e_max"]),
        orbit_direction=str(support["orbit_direction"]),
        fixed_m0_solar=float(init_cfg["m0_solar"]),
        physical_grid=init_cfg["physical_grid"],
        instruments=[
            dict(
                name=inst.name,
                method=inst.photometry_method,
                n_epochs=int(len(inst.ts)),
                size=int(inst.size),
                fwhm=None if inst.fwhm is None else float(inst.fwhm),
            )
            for inst in instruments
        ],
    )

    h5_path, csv_path = _save_ranked_results(
        values_dir=values_dir,
        output_h5=init_cfg["output_h5"],
        output_csv=init_cfg["output_csv"],
        classical_rows=np.asarray(top_classical, dtype=float),
        nonsingular_rows=np.asarray(top_nonsingular, dtype=float),
        meta=meta,
    )

    _print_section("INITIALISATION SEARCH COMPLETE", width=100)
    print(f"[search] tested candidates       : {n_tested}")
    print(f"[search] kept candidates         : {n_kept}")
    print(f"[search] best search S/N         : {float(top_classical[0, -1]):.6f}")
    print(f"[search] output HDF5             : {h5_path}")
    print(f"[search] output CSV              : {csv_path}")
    print("=" * 100)

    return dict(
        h5_path=h5_path,
        csv_path=csv_path,
        best_classical=np.asarray(top_classical[0], dtype=float),
        best_nonsingular=np.asarray(top_nonsingular[0], dtype=float),
        top_classical=np.asarray(top_classical, dtype=float),
        top_nonsingular=np.asarray(top_nonsingular, dtype=float),
        meta=meta,
    )


def run_mcmc_initial_search_from_yaml(yaml_path: str):
    """
    Small alias kept for readability in notebooks and batch scripts.
    """
    return search_mcmc_initial_parameters_from_yaml(yaml_path)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Run the ranked pre-MCMC initialisation search from a YAML file."
    )
    ap.add_argument("yaml", help="Path to the YAML parameter file.")
    args = ap.parse_args()

    out = search_mcmc_initial_parameters_from_yaml(args.yaml)
    print("[search] best classical row shape:", out["best_classical"].shape)
