#!/usr/bin/env python3
# =============================================================================
# mcmc_init_search.py
#
# Ranked pre-MCMC initialisation search for the direct-imaging orbit inference
# pipeline.
#
# Why this file exists
# --------------------
# The historical pipeline could initialise the MCMC from only one source:
# the best brute-force orbit stored in res_grid.h5.  In practice, however, the
# MCMC often benefits from a dedicated pre-search performed directly in the
# non-singular MCMC variables:
#
#     θ = (a, λ0, m0, h, k, p, q)
#
# This file provides that search.  It can:
#   - explore the MCMC parameter space with Sobol or a regular grid,
#   - score each candidate with the same photometric backends as mcmc.py,
#   - combine several instruments exactly like the MCMC,
#   - write a ranked HDF5 file to values_dir for later single-start or
#     multi-start initialisation.
#
# Supported photometry backends
# -----------------------------
#   - "convolve"
#   - "aperture"
#   - "snr_map"
#   - "auto"  -> follow each instrument YAML block
#
# Output files
# ------------
# HDF5
#   dataset "Best solutions"
#       classical coordinates, sorted by decreasing SNR:
#           (a, e, t0, m0, omega=Ω, inc=i, theta0=ω, signal, noise, snr)
#
#   dataset "Best solutions nonsingular"
#       matching MCMC coordinates:
#           (a, λ0, m0, h, k, p, q, signal, noise, snr)
#
# CSV
#   a human-readable top-candidates export for quick inspection.
# =============================================================================

from __future__ import annotations

import os
import json
from itertools import product
from typing import Any, Iterator, List, Optional, Sequence

import h5py
import numpy as np
from scipy.stats import qmc

try:
    from .mcmc import (
        Instrument,
        Params,
        _resolve_background_noise,
        _resolve_bounds,
        _resolve_tref,
        _resolve_weighting,
        _load_convolved_images,
        _load_native_images,
        _load_or_build_local_ring_maps,
        _load_snr_maps,
        _make_placeholder_profiles,
        snr_multi_from_hkpq,
        wrap_2pi,
    )
except Exception:
    from mcmc import (
        Instrument,
        Params,
        _resolve_background_noise,
        _resolve_bounds,
        _resolve_tref,
        _resolve_weighting,
        _load_convolved_images,
        _load_native_images,
        _load_or_build_local_ring_maps,
        _load_snr_maps,
        _make_placeholder_profiles,
        snr_multi_from_hkpq,
        wrap_2pi,
    )


# =============================================================================
# CONFIG HELPERS
# =============================================================================

def _resolve_init_search_config(root: dict) -> dict:
    """
    Parse the YAML `init_search` section.

    Main idea
    ---------
    One YAML file should control both:
      1. how we search for promising initial orbital modes,
      2. how the MCMC later consumes those modes.

    Important fields
    ----------------
    method:
        "sobol" or "grid"

    search_backend:
        "auto", "convolve", "aperture", or "snr_map"

    weighting:
        "auto", "invvar", or "simple"

    top_n_save:
        Number of best-ranked candidates kept on disk.

    output_h5 / output_csv:
        Output files written inside values_dir.

    random_seed:
        Controls Sobol scrambling reproducibility.

    sobol_power:
        Number of points per Sobol scramble = 2 ** sobol_power.

    sobol_scrambles:
        Number of independent scrambled Sobol passes.

    grid_counts:
        Regular grid counts in the user-facing variables:
            a, λ0, m0, e, phi_hk, sin(i/2), phi_pq

    grid_batch_size:
        How many grid points are scored per Python batch.

    Parameter supports:
        Optional overrides of the same supports already used by the MCMC priors.
    """
    cfg = root.get("init_search", {}) or {}
    return dict(
        enabled=bool(cfg.get("enabled", True)),
        method=str(cfg.get("method", "sobol") or "sobol").lower(),
        search_backend=str(cfg.get("search_backend", "auto") or "auto").lower(),
        weighting=str(cfg.get("weighting", "auto") or "auto").lower(),
        top_n_save=max(1, int(cfg.get("top_n_save", 500))),
        output_h5=str(cfg.get("output_h5", "mcmc_init_search.h5") or "mcmc_init_search.h5"),
        output_csv=str(cfg.get("output_csv", "mcmc_init_search_top.csv") or "mcmc_init_search_top.csv"),
        random_seed=cfg.get("random_seed", None),
        sobol_power=max(1, int(cfg.get("sobol_power", 15))),
        sobol_scrambles=max(1, int(cfg.get("sobol_scrambles", 8))),
        grid_batch_size=max(1, int(cfg.get("grid_batch_size", 20000))),
        lambda0_bounds=tuple(map(float, cfg.get("lambda0_bounds", [0.0, 2.0 * np.pi]))),
        m0_bounds_override=cfg.get("m0_bounds", None),
        a_bounds_override=cfg.get("a_bounds", None),
        e_max_override=cfg.get("e_max", None),
        grid_counts=dict(cfg.get("grid_counts", {
            "a_au": 8,
            "lambda0_rad": 8,
            "m0_solar": 3,
            "e": 5,
            "phi_hk": 12,
            "sin_i_over_2": 10,
            "phi_pq": 12,
        })),
    )


def _resolve_parameter_supports(root: dict, params: "Params", init_cfg: dict) -> dict:
    """
    Resolve the numerical support of the initialisation search.

    Priority
    --------
    1. explicit init_search overrides,
    2. MCMC priors,
    3. Params defaults already used by mcmc.py.
    """
    priors = root.get("priors", {}) or {}
    a_bounds, m0_bounds = _resolve_bounds(params, priors)

    if init_cfg["a_bounds_override"] is not None:
        a_bounds = tuple(map(float, init_cfg["a_bounds_override"]))
    if init_cfg["m0_bounds_override"] is not None:
        m0_bounds = tuple(map(float, init_cfg["m0_bounds_override"]))

    la0_bounds = tuple(map(float, priors.get("la0_bounds", init_cfg["lambda0_bounds"])))
    if init_cfg["lambda0_bounds"] is not None:
        la0_bounds = tuple(map(float, init_cfg["lambda0_bounds"]))

    e_max = float(priors.get("e_max", 0.95))
    if init_cfg["e_max_override"] is not None:
        e_max = float(init_cfg["e_max_override"])

    return dict(
        a_bounds=a_bounds,
        m0_bounds=m0_bounds,
        la0_bounds=la0_bounds,
        e_max=e_max,
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
          - snr_map

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

    for inst_cfg in instruments_cfg:
        if not isinstance(inst_cfg, dict):
            raise ValueError("Each instrument entry must be a dictionary.")

        tmp_root = dict(base_root)
        tmp_root.update(inst_cfg)
        params._params = tmp_root

        declared_snr_maps_suffix = inst_cfg.get("snr_maps_suffix", None)

        if search_backend == "auto":
            if declared_snr_maps_suffix is not None:
                photometry_method = "snr_map"
            else:
                photometry_method = str(getattr(params, "method", "convolve")).lower()
                if photometry_method not in ("convolve", "aperture"):
                    photometry_method = "convolve"
        else:
            photometry_method = search_backend

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
        snr_maps = None
        local_bkg_maps = None
        local_noise_maps = None

        if photometry_method == "snr_map":
            # Pure VIP SNR-map workflow: do not call params.load_data("convolve"),
            # because that helper expects resampled images.  We load only the
            # native SNR maps needed by the snr_map backend and create harmless
            # placeholder profiles for downstream helper signatures.
            snr_maps_suffix = inst_cfg.get("snr_maps_suffix", "_snr_map")
            snr_maps = _load_snr_maps(params, suffix=str(snr_maps_suffix))
            images_up = None
            xgrid, bkg, noise = _make_placeholder_profiles(len(ts_i), size_i)

            images_native_suffix = str(inst_cfg.get("native_images_suffix", "_preprocessed"))
            try:
                images_native = _load_native_images(params, suffix=images_native_suffix)
            except Exception:
                images_native = np.asarray(snr_maps, dtype=np.float32)
        else:
            use_local_ring = (
                background_noise_cfg["mode"] == "local_aperture_ring"
            )
            use_precomputed_local_maps = (
                use_local_ring
                and str(background_noise_cfg.get("local_map_mode", "on_the_fly")).lower() == "precompute_cache"
                and bool(background_noise_cfg.get("precompute_maps", False))
            )

            profile_dir = params.get_path("profile_dir")

            # Native images are always useful for plots and are required by the
            # "aperture" backend.
            images_native_suffix = str(inst_cfg.get("native_images_suffix", "_preprocessed"))
            images_native = _load_native_images(params, suffix=images_native_suffix)

            if photometry_method == "convolve":
                images_up = _load_convolved_images(params)
            else:
                images_up = None

            snr_maps = None

            need_radial_profiles = (
                background_noise_cfg["mode"] == "radial_profile"
                or bool(background_noise_cfg.get("fallback_to_radial_profile", True))
            )

            if need_radial_profiles:
                data_io = params.load_data(method=photometry_method)
                xgrid = data_io["x"]
                bkg = data_io["bkg"]
                noise = data_io["noise"]

                if photometry_method == "convolve":
                    images_up = data_io["images"]
                elif photometry_method == "aperture":
                    images_up = None
            else:
                xgrid, bkg, noise = _make_placeholder_profiles(len(ts_i), size_i)

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
                    radial_xgrid=xgrid if need_radial_profiles else None,
                    radial_bkg=bkg if need_radial_profiles else None,
                    radial_noise=noise if need_radial_profiles else None,
                    noise_floor=float(root.get("noise_floor", 1.0)),
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
                snr_maps=None if snr_maps is None else np.asarray(snr_maps),
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
    Convert non-singular MCMC coordinates to the classical-orbit convention used
    by the ranked output files:
        (a, e, t0, m0, omega=Ω, inc=i, theta0=ω)
    """
    theta7 = np.asarray(theta7, dtype=float)
    if theta7.ndim != 2 or theta7.shape[1] != 7:
        raise ValueError("theta7 must have shape (N, 7).")

    a = theta7[:, 0]
    la0 = theta7[:, 1]
    m0 = theta7[:, 2]
    h = theta7[:, 3]
    k = theta7[:, 4]
    p = theta7[:, 5]
    q = theta7[:, 6]

    e = np.sqrt(h * h + k * k)

    w_sum = np.arctan2(h, k)
    delta = np.arctan2(q, p)

    omega = 0.5 * (w_sum + delta)
    theta0 = 0.5 * (w_sum - delta)

    sin_i_2 = np.sqrt(np.clip(p * p + q * q, 0.0, 1.0))
    inc = 2.0 * np.arcsin(np.clip(sin_i_2, 0.0, 1.0))

    M0 = wrap_2pi(la0 - w_sum)
    n = 2.0 * np.pi * np.sqrt(m0 / (a ** 3))
    t0 = float(t_ref) - (M0 / n)

    return np.column_stack([a, e, t0, m0, wrap_2pi(omega), inc, wrap_2pi(theta0)])


# =============================================================================
# SEARCH ENGINES
# =============================================================================

def _draw_sobol_theta_batches(
    *,
    support: dict,
    sobol_power: int,
    sobol_scrambles: int,
    random_seed: Optional[int],
) -> Iterator[np.ndarray]:
    """
    Yield Sobol batches already mapped to the 7-D MCMC parameterisation.
    """
    a_lo, a_hi = support["a_bounds"]
    la_lo, la_hi = support["la0_bounds"]
    m0_lo, m0_hi = support["m0_bounds"]
    e_max = float(support["e_max"])

    base_seed = None if random_seed is None else int(random_seed)

    for scramble_id in range(int(sobol_scrambles)):
        seed_here = None if base_seed is None else base_seed + scramble_id
        sampler = qmc.Sobol(d=7, scramble=True, seed=seed_here)
        u = sampler.random_base2(m=int(sobol_power))

        a = a_lo + (a_hi - a_lo) * u[:, 0]
        la0 = la_lo + (la_hi - la_lo) * u[:, 1]
        m0 = m0_lo + (m0_hi - m0_lo) * u[:, 2]

        e = e_max * u[:, 3]
        phi_hk = 2.0 * np.pi * u[:, 4]
        s = u[:, 5]
        phi_pq = 2.0 * np.pi * u[:, 6]

        h = e * np.sin(phi_hk)
        k = e * np.cos(phi_hk)
        p = s * np.cos(phi_pq)
        q = s * np.sin(phi_pq)

        yield np.column_stack([a, la0, m0, h, k, p, q]).astype(float)


def _iter_grid_theta_batches(
    *,
    support: dict,
    grid_counts: dict,
    batch_size: int,
) -> Iterator[np.ndarray]:
    """
    Yield grid-search batches already mapped to the 7-D MCMC parameterisation.

    The grid is expressed in user-facing coordinates:
      - a
      - λ0
      - m0
      - e
      - angle in (h,k)
      - sin(i/2)
      - angle in (p,q)

    This keeps the search explicit and physically interpretable while still
    landing directly in the MCMC variable space.
    """
    a_lo, a_hi = support["a_bounds"]
    la_lo, la_hi = support["la0_bounds"]
    m0_lo, m0_hi = support["m0_bounds"]
    e_max = float(support["e_max"])

    axes = dict(
        a_au=np.linspace(a_lo, a_hi, int(grid_counts["a_au"]), dtype=float),
        lambda0_rad=np.linspace(la_lo, la_hi, int(grid_counts["lambda0_rad"]), endpoint=False, dtype=float),
        m0_solar=np.linspace(m0_lo, m0_hi, int(grid_counts["m0_solar"]), dtype=float),
        e=np.linspace(0.0, e_max, int(grid_counts["e"]), dtype=float),
        phi_hk=np.linspace(0.0, 2.0 * np.pi, int(grid_counts["phi_hk"]), endpoint=False, dtype=float),
        sin_i_over_2=np.linspace(0.0, 1.0, int(grid_counts["sin_i_over_2"]), dtype=float),
        phi_pq=np.linspace(0.0, 2.0 * np.pi, int(grid_counts["phi_pq"]), endpoint=False, dtype=float),
    )

    keys = list(axes.keys())
    value_lists = [axes[k] for k in keys]
    batch_rows: List[tuple[float, ...]] = []

    for values in product(*value_lists):
        batch_rows.append(values)
        if len(batch_rows) >= int(batch_size):
            arr = np.asarray(batch_rows, dtype=float)
            a = arr[:, 0]
            la0 = arr[:, 1]
            m0 = arr[:, 2]
            e = arr[:, 3]
            phi_hk = arr[:, 4]
            s = arr[:, 5]
            phi_pq = arr[:, 6]

            h = e * np.sin(phi_hk)
            k = e * np.cos(phi_hk)
            p = s * np.cos(phi_pq)
            q = s * np.sin(phi_pq)

            yield np.column_stack([a, la0, m0, h, k, p, q]).astype(float)
            batch_rows = []

    if batch_rows:
        arr = np.asarray(batch_rows, dtype=float)
        a = arr[:, 0]
        la0 = arr[:, 1]
        m0 = arr[:, 2]
        e = arr[:, 3]
        phi_hk = arr[:, 4]
        s = arr[:, 5]
        phi_pq = arr[:, 6]

        h = e * np.sin(phi_hk)
        k = e * np.cos(phi_hk)
        p = s * np.cos(phi_pq)
        q = s * np.sin(phi_pq)

        yield np.column_stack([a, la0, m0, h, k, p, q]).astype(float)


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
        order = np.argsort(merged[:, -1])[::-1]
        return merged[order]

    idx = np.argpartition(merged[:, -1], -int(top_n))[-int(top_n):]
    top = merged[idx]
    order = np.argsort(top[:, -1])[::-1]
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
    The ranking objective is intentionally the same MCMC SNR objective,
    because the point of this file is to initialise the MCMC in the same
    landscape that the sampler will subsequently refine.
    """
    params = Params.read(yaml_path)
    root = params._params

    init_cfg = _resolve_init_search_config(root)
    if init_cfg["method"] not in ("sobol", "grid"):
        raise ValueError("init_search.method must be 'sobol' or 'grid'.")

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

    noise_floor = float(root.get("noise_floor", 1.0))
    values_dir = params.get_path("values_dir")

    print("=" * 100)
    print("MCMC INITIALISATION SEARCH")
    print("=" * 100)
    print(f"[search] yaml_path                : {yaml_path}")
    print(f"[search] method                  : {init_cfg['method']}")
    print(f"[search] search_backend          : {search_backend}")
    print(f"[search] weighting               : {weighting}")
    print(f"[search] top_n_save              : {init_cfg['top_n_save']}")
    print(f"[search] values_dir              : {values_dir}")
    print(f"[search] t_ref                   : {t_ref:.6f}")
    print(f"[search] a_bounds                : {support['a_bounds']}")
    print(f"[search] la0_bounds              : {support['la0_bounds']}")
    print(f"[search] m0_bounds               : {support['m0_bounds']}")
    print(f"[search] e_max                   : {support['e_max']}")
    print(f"[search] n_instruments           : {len(instruments)}")
    for inst in instruments:
        print(
            f"[search]   - {inst.name}: method={inst.photometry_method}, "
            f"n_epochs={len(inst.ts)}, size={inst.size}, fwhm={inst.fwhm}"
        )
    print("=" * 100)

    top_classical = None
    top_nonsingular = None
    n_tested = 0
    n_kept = 0

    if init_cfg["method"] == "sobol":
        iterator = _draw_sobol_theta_batches(
            support=support,
            sobol_power=init_cfg["sobol_power"],
            sobol_scrambles=init_cfg["sobol_scrambles"],
            random_seed=init_cfg["random_seed"],
        )
        n_batches_total = init_cfg["sobol_scrambles"]
    else:
        iterator = _iter_grid_theta_batches(
            support=support,
            grid_counts=init_cfg["grid_counts"],
            batch_size=init_cfg["grid_batch_size"],
        )
        n_batches_total = None

    for batch_index, theta_batch in enumerate(iterator, start=1):
        theta_batch = np.asarray(theta_batch, dtype=float)
        if theta_batch.ndim != 2 or theta_batch.shape[1] != 7:
            raise ValueError("Internal error: theta batch must have shape (N, 7).")

        signal, noise, snr = snr_multi_from_hkpq(
            theta_batch,
            instruments,
            noise_floor=noise_floor,
            weighting=weighting,
        )

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

        best_snr_here = float(top_classical[0, -1]) if top_classical is not None and top_classical.size else float("nan")
        if n_batches_total is None:
            print(
                f"[search] batch {batch_index:04d} | tested={n_tested} | kept={n_kept} | "
                f"current_best_snr={best_snr_here:.6f}"
            )
        else:
            print(
                f"[search] batch {batch_index:04d}/{n_batches_total:04d} | tested={n_tested} | kept={n_kept} | "
                f"current_best_snr={best_snr_here:.6f}"
            )

    if top_classical is None or top_classical.size == 0:
        raise RuntimeError("The initialisation search did not produce any valid candidate.")

    meta = dict(
        yaml_path=str(yaml_path),
        method=init_cfg["method"],
        search_backend=search_backend,
        weighting=weighting,
        n_tested=int(n_tested),
        n_kept=int(n_kept),
        t_ref=float(t_ref),
        a_bounds=list(map(float, support["a_bounds"])),
        la0_bounds=list(map(float, support["la0_bounds"])),
        m0_bounds=list(map(float, support["m0_bounds"])),
        e_max=float(support["e_max"]),
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

    print("=" * 100)
    print("[search] DONE")
    print(f"[search] tested candidates       : {n_tested}")
    print(f"[search] kept candidates         : {n_kept}")
    print(f"[search] best SNR                : {float(top_classical[0, -1]):.6f}")
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
