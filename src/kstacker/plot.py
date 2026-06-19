# plot.py
# =======
#
# Standalone plotting script for MCMC results.
#
# This module provides functions to regenerate the sum_top_orbits and
# images_with_top_orbits plots from an existing MCMC HDF5 file, with
# additional features:
#   - Selection of orbits by log_prob rank (N to P)
#   - Color coding from red (highest log_prob) to green (lowest in selection)
#   - Transparency for overlapping orbits
#
# Usage: kstacker plot nom_du_yml.yml --orbits-range N,P
#
# The plots are saved in a 'custom_postprocessing_plots' directory at the same level
# as the 'plots' directory created by mcmc.py.

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Project helpers — try relative import first (package mode), fall back to
# absolute import when the file is run standalone.
try:
    from .orbit import orbit
    from .utils import Params
    from . import mcmc as mcmc_module
    # Import specific functions and classes from mcmc
    mean_motion = mcmc_module.mean_motion
    lambda_to_M0 = mcmc_module.lambda_to_M0
    M0_to_t0 = mcmc_module.M0_to_t0
    compute_projection_matrices_from_hkpq = mcmc_module.compute_projection_matrices_from_hkpq
    wrap_2pi = mcmc_module.wrap_2pi
    Instrument = mcmc_module.Instrument
    _select_top_by_logprob = mcmc_module._select_top_by_logprob
    log_probability = mcmc_module.log_probability
    stack_planet_from_posterior_native = mcmc_module.stack_planet_from_posterior_native
except Exception:
    from orbit import orbit
    from utils import Params
    import mcmc as mcmc_module
    mean_motion = mcmc_module.mean_motion
    lambda_to_M0 = mcmc_module.lambda_to_M0
    M0_to_t0 = mcmc_module.M0_to_t0
    compute_projection_matrices_from_hkpq = mcmc_module.compute_projection_matrices_from_hkpq
    wrap_2pi = mcmc_module.wrap_2pi
    Instrument = mcmc_module.Instrument
    _select_top_by_logprob = mcmc_module._select_top_by_logprob
    log_probability = mcmc_module.log_probability
    stack_planet_from_posterior_native = mcmc_module.stack_planet_from_posterior_native


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
    proj = compute_projection_matrices_from_hkpq(
        np.array([k], float), np.array([h], float),
        np.array([p], float), np.array([q], float),
    )[0]
    north_sky = x_orb * proj[0, 0] + y_orb * proj[0, 1]
    west_sky = x_orb * proj[1, 0] + y_orb * proj[1, 1]
    cx = cy = size // 2
    return west_sky * scale + cx, north_sky * scale + cy


def _build_instrument_from_yaml_config(
    params: Params, inst_cfg: dict, base_root: dict, global_t_ref: Optional[float] = None
) -> Instrument:
    """
    Build an Instrument object from a YAML instrument configuration block.

    This mirrors the instrument-building logic in mcmc.py's _run_mcmc_from_yaml_impl.
    All instruments share the same t_ref (global reference epoch).
    """
    # Merge global config with instrument-specific overrides.
    tmp_root = dict(base_root)
    tmp_root.update(inst_cfg)
    params._params = tmp_root

    # Decide which photometry backend this instrument uses.
    snr_maps_suffix = inst_cfg.get("snr_maps_suffix", None)
    if snr_maps_suffix is not None:
        photometry_method = "snr_map"
    else:
        photometry_method = str(inst_cfg.get("method", "convolve")).lower()
        if photometry_method not in ("convolve", "aperture", "snr_map"):
            photometry_method = "convolve"

    # Instrument name
    inst_name = inst_cfg.get("name", inst_cfg.get("instrument_name", "INST"))

    # Time sampling
    # t_ref is provided globally and shared by all instruments
    t_ref = global_t_ref
    if t_ref is None:
        # Fallback: compute from YAML t_ref_mode
        t_ref_mode = str(tmp_root.get("t_ref_mode", "min_ts")).lower()
        if t_ref_mode == "fixed":
            t_ref = float(tmp_root.get("t_ref", 0.0))

    # Time vector for this instrument
    time_str = tmp_root.get("time", "")
    if isinstance(time_str, str) and "+" in time_str:
        ts = np.array([float(t.strip()) for t in time_str.split("+")])
    else:
        ts = np.array([float(time_str)])

    p = int(tmp_root.get("p", 0))
    p_prev = int(tmp_root.get("p_prev", 0))
    nimg = p + p_prev

    # If time is a single value, replicate it for all images
    if len(ts) == 1 and nimg > 1:
        ts = np.full(nimg, ts[0])
    elif len(ts) != nimg:
        raise ValueError(
            f"Instrument '{inst_name}': len(time)={len(ts)} but nimg={nimg}."
        )

    # Pixel geometry
    size = int(tmp_root.get("n", 150))
    dist = float(tmp_root.get("dist", 1.0))  # parsec
    resol = float(tmp_root.get("resol", 1.0))  # mas/pixel
    scale = 1000.0 / (dist * resol)  # AU -> native pixels

    upsampling_factor = float(tmp_root.get("upsampling_factor", 1.0))
    fwhm = tmp_root.get("fwhm", None)
    if fwhm is not None:
        fwhm = float(fwhm)

    # Coronagraph masks
    r_mask = tmp_root.get("r_mask", None)
    if r_mask is not None:
        r_mask = float(r_mask)
    r_mask_ext = tmp_root.get("r_mask_ext", None)
    if r_mask_ext is not None:
        r_mask_ext = float(r_mask_ext)

    # Load images
    images_dir = params.get_path("images_dir")
    images_native = None
    images_up = None
    snr_maps = None

    try:
        if photometry_method in ("convolve", "aperture") or images_dir:
            # Try to load native images
            imgs = []
            for k in range(nimg):
                try:
                    from astropy.io import fits
                    fn = os.path.join(images_dir, f"image_{k}_preprocessed.fits")
                    im = fits.getdata(fn)
                    imgs.append(im.astype("float32", copy=False))
                except Exception:
                    pass
            if imgs:
                images_native = np.asarray(imgs)

        if photometry_method == "convolve":
            # Try to load upsampled images
            imgs_up = []
            for k in range(nimg):
                try:
                    from astropy.io import fits
                    fn = os.path.join(images_dir, f"image_{k}_upsampled.fits")
                    im = fits.getdata(fn)
                    imgs_up.append(im.astype("float32", copy=False))
                except Exception:
                    pass
            if imgs_up:
                images_up = np.asarray(imgs_up)

        if photometry_method == "snr_map" and snr_maps_suffix:
            # Load SNR maps
            maps = []
            for k in range(nimg):
                try:
                    from astropy.io import fits
                    fn = os.path.join(images_dir, f"image_{k}{snr_maps_suffix}.fits")
                    m = fits.getdata(fn)
                    maps.append(m.astype("float32", copy=False))
                except Exception:
                    pass
            if maps:
                snr_maps = np.asarray(maps)

    except Exception as e:
        print(f"[plot] Warning: could not load images for {inst_name}: {e}")

    # Load radial profiles if available (using params.load_data like mcmc.py)
    xgrid = None
    bkg = None
    noise = None

    try:
        # Temporarily set params._params to this instrument's config
        original_params = dict(params._params)
        params._params = tmp_root
        
        data_io = params.load_data(method=photometry_method if photometry_method != "snr_map" else "convolve")
        xgrid = data_io["x"]
        bkg = data_io["bkg"]
        noise = data_io["noise"]
        
        # Restore original params
        params._params = original_params
    except Exception as e:
        print(f"[plot] Warning: could not load radial profiles for {inst_name}: {e}")
        # Continue without profiles - they're not strictly needed for orbit plots

    # Final fallback for t_ref if still None
    if t_ref is None:
        t_ref = float(np.min(ts)) if len(ts) > 0 else 0.0

    return Instrument(
        name=inst_name,
        size=size,
        scale=scale,
        upsampling_factor=upsampling_factor,
        fwhm=fwhm,
        r_mask=r_mask,
        r_mask_ext=r_mask_ext,
        t_ref=t_ref,
        ts=ts,
        photometry_method=photometry_method,
        images_up=images_up,
        images_native=images_native,
        snr_maps=snr_maps,
        xgrid=xgrid,
        bkg=bkg,
        noise=noise,
    )


def load_instruments_from_yaml(yaml_path: str) -> List[Instrument]:
    """
    Load and build Instrument objects from a YAML configuration file.

    All instruments share the same t_ref (global reference epoch) to ensure
    consistent orbital phase definition across instruments.

    Parameters
    ----------
    yaml_path : str
        Path to the YAML configuration file.

    Returns
    -------
    List[Instrument]
        List of Instrument objects, one per instrument block in the YAML.
    """
    params = Params.read(yaml_path)
    root = params._params
    base_root = dict(root)

    instruments_cfg = root.get("instruments", None)
    if not isinstance(instruments_cfg, (list, tuple)):
        # Single instrument mode: wrap in a list
        instruments_cfg = [base_root]

    # First pass: collect all time vectors to compute global t_ref
    all_ts = []
    for inst_cfg in instruments_cfg:
        if not isinstance(inst_cfg, dict):
            continue
        # Merge config
        tmp_root = dict(base_root)
        tmp_root.update(inst_cfg)
        
        # Get time vector
        time_str = tmp_root.get("time", "")
        if isinstance(time_str, str) and "+" in time_str:
            ts = np.array([float(t.strip()) for t in time_str.split("+")])
        else:
            ts = np.array([float(time_str)])
        
        p = int(tmp_root.get("p", 0))
        p_prev = int(tmp_root.get("p_prev", 0))
        nimg = p + p_prev
        
        # Replicate if needed
        if len(ts) == 1 and nimg > 1:
            ts = np.full(nimg, ts[0])
        elif len(ts) != nimg:
            # Try to match
            if len(ts) > 0:
                ts = ts[:nimg] if len(ts) >= nimg else np.tile(ts, nimg // len(ts) + 1)[:nimg]
        
        all_ts.append(ts)

    # Compute global t_ref (shared by all instruments)
    ts_global = np.concatenate(all_ts) if all_ts else np.array([], float)
    t_ref_mode = str(root.get("t_ref_mode", "min_ts")).lower()
    if t_ref_mode == "fixed":
        global_t_ref = float(root.get("t_ref", 0.0))
    else:
        # "min_ts": use earliest observation time across all instruments
        global_t_ref = float(np.min(ts_global)) if len(ts_global) > 0 else 0.0

    # Second pass: build instruments with global t_ref
    instruments = []
    for inst_cfg in instruments_cfg:
        if not isinstance(inst_cfg, dict):
            continue
        try:
            inst = _build_instrument_from_yaml_config(
                params, inst_cfg, base_root, global_t_ref=global_t_ref
            )
            instruments.append(inst)
        except Exception as e:
            print(f"[plot] Warning: skipping instrument config {inst_cfg}: {e}")
            continue

    # If no instruments from list, try single instrument from root
    if not instruments and isinstance(base_root, dict):
        try:
            inst = _build_instrument_from_yaml_config(
                params, base_root, base_root, global_t_ref=global_t_ref
            )
            instruments.append(inst)
        except Exception as e:
            print(f"[plot] Warning: could not build instrument from root config: {e}")

    return instruments


def plot_sum_with_top_orbits_colored(
    instruments: Sequence[Instrument],
    flat_samples: np.ndarray,
    flat_log_prob: np.ndarray,
    top_indices: np.ndarray,
    *,
    orbit_time_pad: float = 0.1,
    min_periods: float = 1.0,
    orbit_T: int = 1000,
    orbit_alpha: float = 0.25,
    orbit_lw: float = 0.6,
    cross_color: str = "b",
    cross_ms: float = 6.0,
    cross_mew: float = 1.2,
    zoom_margin: float = 10.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: str = "Sum of images + top orbits (colored by log_prob)",
    save_path: Optional[str] = None,
    dpi: int = 200,
) -> dict:
    """
    For each instrument, plot the sum of all native images with the selected
    orbits overlaid as smooth curves with color coding by log_prob rank,
    and epoch marks as crosses.

    Colors range from red (highest log_prob in selection) to green (lowest).
    Transparency is maintained to show overlapping orbits (darker = more overlap).

    Parameters
    ----------
    instruments : Sequence[Instrument]
        List of Instrument objects.
    flat_samples : np.ndarray
        (Ns, 7) array of posterior samples.
    flat_log_prob : np.ndarray
        (Ns,) array of log probabilities.
    top_indices : np.ndarray
        Indices of selected samples from flat_samples/flat_log_prob.
    orbit_time_pad : float
        Fractional padding around the time span for orbit drawing.
    min_periods : float
        Minimum number of orbital periods to span when drawing orbits.
    orbit_T : int
        Number of points in the orbit curve.
    orbit_alpha : float
        Transparency for orbit curves.
    orbit_lw : float
        Line width for orbit curves.
    cross_color : str
        Color for epoch marks.
    cross_ms : float
        Marker size for epoch marks.
    cross_mew : float
        Marker edge width for epoch marks.
    zoom_margin : float
        Margin in pixels for the zoom panel.
    vmin, vmax : Optional[float]
        Image value range.
    title : str
        Plot title.
    save_path : Optional[str]
        Base path for saving figures. If provided, figures are saved as
        {save_path}_{inst.name}.png
    dpi : int
        DPI for saved figures.

    Returns
    -------
    dict
        Metadata about the plots.
    """
    if not instruments:
        raise ValueError("At least one Instrument is required.")

    fs = np.asarray(flat_samples, dtype=float)
    lp = np.asarray(flat_log_prob, dtype=float)
    top_idx = np.asarray(top_indices, dtype=int)

    # Create colormap from red (best) to green (worst in selection)
    n_top = len(top_idx)
    if n_top == 0:
        raise ValueError("No top indices provided.")

    # Sort top indices by log_prob descending for color assignment
    lp_top = lp[top_idx]
    sorted_order = np.argsort(lp_top)[::-1]  # Best first
    
    # Create normalized color values: 0 = best (red), 1 = worst in selection (green)
    color_values = np.linspace(0, 1, n_top)
    cmap = plt.cm.RdYlGn  # Red -> Yellow -> Green
    
    results = {}

    for inst in instruments:
        imgs = np.asarray(inst.images_native)
        if imgs is None or imgs.ndim != 3:
            print(f"[sum_orbits] Skipping {inst.name}: no native images.")
            continue

        ts = np.asarray(inst.ts)
        summed = np.sum(imgs, axis=0)

        vmin_local = np.percentile(summed, 1) if vmin is None else vmin
        vmax_local = np.percentile(summed, 99) if vmax is None else vmax

        t_min = float(np.min(ts))
        t_max = float(np.max(ts))
        t_mid = 0.5 * (t_min + t_max)
        span_obs = max(t_max - t_min, 0.0)
        span_obs_padded = span_obs * (1.0 + 2.0 * orbit_time_pad)

        fig, (ax_main, ax_zoom) = plt.subplots(
            1, 2, figsize=(11, 6), gridspec_kw=dict(width_ratios=[3, 2])
        )

        ax_main.imshow(summed, origin="lower", cmap="gray",
                       vmin=vmin_local, vmax=vmax_local)
        ax_main.set_xlabel("x [px]")
        ax_main.set_ylabel("y [px]")
        ax_main.set_title(f"{title} [{inst.name}]")

        all_xk, all_yk = [], []

        # Draw orbits for each selected sample, colored by log_prob rank
        for rank, idx_in_top in enumerate(sorted_order):
            actual_idx = top_idx[idx_in_top]
            theta = fs[actual_idx]
            
            # Get color from colormap based on rank
            color_val = color_values[rank]
            orbit_color = cmap(color_val)
            
            a_, la0_, m0_, h_, k_, p_, q_ = map(float, theta)
            n_ = mean_motion(a_, m0_)
            P_ = 2.0 * np.pi / n_ if n_ > 0 else 0.0

            span_target = max(span_obs_padded, min_periods * P_) or (P_ if P_ > 0 else 1.0)
            t_grid = np.linspace(t_mid - 0.5 * span_target, t_mid + 0.5 * span_target, orbit_T)

            xo, yo = _predict_pixel_track_native(
                theta, t_grid, size=inst.size, scale=inst.scale, t_ref=inst.t_ref
            )
            # Draw orbit with color based on log_prob rank
            ax_main.plot(xo, yo, "-", color=orbit_color,
                         alpha=orbit_alpha, linewidth=orbit_lw, 
                         solid_capstyle="round", zorder=10)

            xk, yk = _predict_pixel_track_native(
                theta, ts, size=inst.size, scale=inst.scale, t_ref=inst.t_ref
            )
            # Use same color as orbit for cross markers
            ax_main.plot(xk, yk, "+", ls="none", mec=orbit_color, mfc="none",
                         ms=cross_ms, mew=cross_mew, zorder=11)
            all_xk.append(xk)
            all_yk.append(yk)

        ax_main.set_xlim(0, inst.size)
        ax_main.set_ylim(0, inst.size)

        all_xk = np.concatenate(all_xk) if all_xk else np.array([inst.size / 2.0])
        all_yk = np.concatenate(all_yk) if all_yk else np.array([inst.size / 2.0])

        x_min = max(0.0, float(np.min(all_xk)) - zoom_margin)
        x_max = min(float(inst.size), float(np.max(all_xk)) + zoom_margin)
        y_min = max(0.0, float(np.min(all_yk)) - zoom_margin)
        y_max = min(float(inst.size), float(np.max(all_yk)) + zoom_margin)

        ax_zoom.imshow(summed, origin="lower", cmap="gray", vmin=vmin_local, vmax=vmax_local)
        ax_zoom.set_xlim(x_min, x_max)
        ax_zoom.set_ylim(y_min, y_max)
        ax_zoom.set_title(f"{inst.name} — zoom on discrete epochs")
        ax_zoom.set_xlabel("x [px]")
        ax_zoom.set_ylabel("y [px]")

        # Draw orbits in zoom panel too
        for rank, idx_in_top in enumerate(sorted_order):
            actual_idx = top_idx[idx_in_top]
            theta = fs[actual_idx]
            color_val = color_values[rank]
            orbit_color = cmap(color_val)
            
            a_, la0_, m0_, h_, k_, p_, q_ = map(float, theta)
            n_ = mean_motion(a_, m0_)
            P_ = 2.0 * np.pi / n_ if n_ > 0 else 0.0
            span_target = max(span_obs_padded, min_periods * P_) or (P_ if P_ > 0 else 1.0)
            t_grid = np.linspace(t_mid - 0.5 * span_target, t_mid + 0.5 * span_target, orbit_T)

            xo, yo = _predict_pixel_track_native(
                theta, t_grid, size=inst.size, scale=inst.scale, t_ref=inst.t_ref
            )
            ax_zoom.plot(xo, yo, "-", color=orbit_color,
                         alpha=orbit_alpha, linewidth=orbit_lw, 
                         solid_capstyle="round", zorder=10)

            xk, yk = _predict_pixel_track_native(
                theta, ts, size=inst.size, scale=inst.scale, t_ref=inst.t_ref
            )
            # Use same color as orbit for cross markers in zoom panel too
            ax_zoom.plot(xk, yk, "+", ls="none", mec=orbit_color, mfc="none",
                         ms=cross_ms, mew=cross_mew, zorder=11)

        fig.tight_layout()
        if save_path:
            base, ext = os.path.splitext(save_path)
            fig.savefig(f"{base}_{inst.name}{ext or '.png'}", dpi=dpi)
        plt.close(fig)

        results[inst.name] = {
            "top_indices": top_idx,
            "top_logprob": lp[top_idx],
        }

    return results


def plot_images_with_top_orbits_colored(
    instruments: Sequence[Instrument],
    flat_samples: np.ndarray,
    flat_log_prob: np.ndarray,
    top_indices: np.ndarray,
    *,
    ncols: int = 6,
    circle_radius: float = 4.0,
    top_alpha: float = 0.5,
    top_lw: float = 1.0,
    cmap_image: str = "gray",
    title: str = "Images with top orbits (colored by log_prob)",
    save_path: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    dpi: int = 200,
) -> dict:
    """
    Overlay predicted positions of the selected samples on native images.

    One figure per instrument; one panel per epoch. Circles mark the predicted
    planet position for each selected sample, with color coding by log_prob rank.

    Parameters
    ----------
    instruments : Sequence[Instrument]
        List of Instrument objects.
    flat_samples : np.ndarray
        (Ns, 7) array of posterior samples.
    flat_log_prob : np.ndarray
        (Ns,) array of log probabilities.
    top_indices : np.ndarray
        Indices of selected samples from flat_samples/flat_log_prob.
    ncols : int
        Number of columns in the epoch grid.
    circle_radius : float
        Radius of the circle markers in pixels.
    top_alpha : float
        Transparency for circle markers.
    top_lw : float
        Line width for circle markers.
    cmap_image : str
        Colormap for the image display.
    title : str
        Plot title.
    save_path : Optional[str]
        Base path for saving figures. If provided, figures are saved as
        {save_path}_{inst.name}.png
    vmin, vmax : Optional[float]
        Image value range.
    dpi : int
        DPI for saved figures.

    Returns
    -------
    dict
        Metadata about the plots including per-instrument position data.
    """
    if not instruments:
        raise ValueError("At least one Instrument is required.")

    fs = np.asarray(flat_samples, dtype=float)
    lp = np.asarray(flat_log_prob, dtype=float)
    top_idx = np.asarray(top_indices, dtype=int)

    # Create colormap from red (best) to green (worst in selection)
    n_top = len(top_idx)
    if n_top == 0:
        raise ValueError("No top indices provided.")

    # Sort top indices by log_prob descending for color assignment
    lp_top = lp[top_idx]
    sorted_order = np.argsort(lp_top)[::-1]  # Best first
    
    # Create normalized color values: 0 = best (red), 1 = worst in selection (green)
    color_values = np.linspace(0, 1, n_top)
    cmap = plt.cm.RdYlGn  # Red -> Yellow -> Green

    results = {}
    per_inst_meta: Dict[str, dict] = {}

    for inst in instruments:
        imgs = np.asarray(inst.images_native)
        if imgs.ndim != 3:
            raise ValueError(f"Instrument '{inst.name}' images_native must be 3D.")

        K, H, W = imgs.shape

        # Predict positions for all top samples
        x_list, y_list = [], []
        thetas_top = fs[top_idx]
        
        for theta in thetas_top:
            xr, yr = _predict_pixel_track_native(
                theta, inst.ts, size=inst.size, scale=inst.scale, t_ref=inst.t_ref
            )
            x_list.append(xr)
            y_list.append(yr)

        x_top = np.stack(x_list, axis=0)  # (n_top, K)
        y_top = np.stack(y_list, axis=0)
        per_inst_meta[inst.name] = {"x_pix_top": x_top, "y_pix_top": y_top}

        vmin_local = np.percentile(imgs, 1) if vmin is None else vmin
        vmax_local = np.percentile(imgs, 99) if vmax is None else vmax

        ncols_eff = max(1, int(ncols))
        nrows = int(np.ceil(K / ncols_eff))
        fig, axes = plt.subplots(
            nrows, ncols_eff,
            figsize=(3.2 * ncols_eff, 3.2 * nrows), squeeze=False,
        )
        axes = axes.ravel()

        # Draw circles for each selected sample, colored by log_prob rank
        for i, ax in enumerate(axes):
            ax.set_xticks([])
            ax.set_yticks([])
            if i < K:
                ax.imshow(imgs[i], origin="lower", cmap=cmap_image,
                          vmin=vmin_local, vmax=vmax_local)
                
                # Draw circles for all top samples, colored by rank
                for rank, idx_in_top in enumerate(sorted_order):
                    color_val = color_values[rank]
                    circle_color = cmap(color_val)
                    
                    ax.add_patch(Circle(
                        (x_top[idx_in_top, i], y_top[idx_in_top, i]),
                        radius=circle_radius, fill=False,
                        ec=circle_color, lw=top_lw, alpha=top_alpha,
                    ))
                ax.set_title(f"Epoch {i} | {n_top} orbits")
            else:
                ax.axis("off")

        if title:
            fig.suptitle(f"{title} [{inst.name}]", fontsize=14)
        fig.tight_layout()

        if save_path:
            base, ext = os.path.splitext(save_path)
            fig.savefig(f"{base}_{inst.name}{ext or '.png'}", dpi=dpi)
        plt.close(fig)

    results["top_indices"] = top_idx
    results["top_logprob"] = lp[top_idx]
    results["per_instrument"] = per_inst_meta

    return results


def run_plot_from_yaml(
    yaml_path: str,
    orbits_range: Optional[str] = None,
) -> dict:
    """
    Main entry point for the plot command.

    Loads the MCMC results from the HDF5 file specified in the YAML,
    builds the instruments, and generates the plots.

    Parameters
    ----------
    yaml_path : str
        Path to the YAML configuration file.
    orbits_range : Optional[str]
        String of the form "N,P" specifying the range of orbits to plot.
        N and P are 1-indexed ranks by log_prob (1 = highest).
        If None, plots all orbits (same as mcmc.py behavior).

    Returns
    -------
    dict
        Metadata about the generated plots.
    """
    import os

    print(f"[plot] Loading parameters from: {yaml_path}")

    # Read YAML to get values_dir
    params = Params.read(yaml_path)
    values_dir = params.get_path("values_dir")

    # Path to HDF5 file
    h5_path = os.path.join(values_dir, "mcmc_chain_and_logprob.h5")

    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"MCMC HDF5 file not found: {h5_path}"
        )

    print(f"[plot] Loading MCMC results from: {h5_path}")

    # Load HDF5 data
    with h5py.File(h5_path, "r") as f:
        # Load flat_chain if available, otherwise reshape chain
        if "flat_chain" in f:
            flat_chain = np.array(f["flat_chain"], dtype=float)
        elif "chain" in f:
            chain = np.array(f["chain"], dtype=float)
            flat_chain = chain.reshape(-1, chain.shape[-1])
        else:
            raise KeyError("Neither 'flat_chain' nor 'chain' found in HDF5 file")
        
        # Load flat_log_prob if available, otherwise reshape log_prob
        if "flat_log_prob" in f:
            flat_log_prob = np.array(f["flat_log_prob"], dtype=float)
        elif "log_prob" in f:
            log_prob = np.array(f["log_prob"], dtype=float)
            flat_log_prob = log_prob.reshape(-1)
        else:
            raise KeyError("Neither 'flat_log_prob' nor 'log_prob' found in HDF5 file")

    # Ensure flat_log_prob matches flat_chain
    if flat_log_prob.shape[0] != flat_chain.shape[0]:
        raise ValueError(
            f"Shape mismatch: flat_chain={flat_chain.shape}, "
            f"flat_log_prob={flat_log_prob.shape}. "
            f"The flattened chain and log_prob must have the same number of samples."
        )

    # Ensure we only use 7-D parameters (orbital only, no flux)
    if flat_chain.shape[1] > 7:
        flat_chain = flat_chain[:, :7]
    elif flat_chain.shape[1] < 7:
        raise ValueError(
            f"Expected at least 7 parameters, got {flat_chain.shape[1]}"
        )

    print(f"[plot] Loaded {flat_chain.shape[0]} samples with {flat_chain.shape[1]} parameters")

    # Load instruments
    instruments = load_instruments_from_yaml(yaml_path)
    print(f"[plot] Loaded {len(instruments)} instrument(s): " + 
          ", ".join(inst.name for inst in instruments))

    # Parse orbits_range
    if orbits_range:
        try:
            parts = orbits_range.strip().split(",")
            if len(parts) != 2:
                raise ValueError("orbits_range must be of the form 'N,P'")
            n_start = int(parts[0].strip())  # 1-indexed
            n_end = int(parts[1].strip())    # 1-indexed
            if n_start < 1 or n_end < n_start:
                raise ValueError(f"Invalid range: {n_start}, {n_end}. Must have 1 <= N <= P")
        except Exception as e:
            raise ValueError(f"Invalid orbits_range '{orbits_range}': {e}")
        
        # Get all indices sorted by log_prob descending
        all_sorted_idx = np.argsort(flat_log_prob)[::-1]
        # Select from N-1 to P-1 (converting to 0-indexed)
        top_indices = all_sorted_idx[n_start-1:n_end]
        print(f"[plot] Selecting orbits ranked {n_start} to {n_end} by log_prob " +
              f"({len(top_indices)} orbits)")
    else:
        # Use default: top 1000 orbits (like mcmc.py default for images_with_top_orbits)
        # Get all indices sorted by log_prob descending
        all_sorted_idx = np.argsort(flat_log_prob)[::-1]
        top_indices = all_sorted_idx[:1000]  # Default: top 1000
        print(f"[plot] Using default: top 1000 orbits")

    # Create plots directory
    custom_postprocessing_plots_dir = os.path.join(values_dir, "custom_postprocessing_plots")
    os.makedirs(custom_postprocessing_plots_dir, exist_ok=True)
    print(f"[plot] Saving plots to: {custom_postprocessing_plots_dir}")

    # Generate plots
    results = {}

    # 1. sum_top_orbits plots
    sum_save_path = os.path.join(custom_postprocessing_plots_dir, "sum_top_orbits")
    sum_results = plot_sum_with_top_orbits_colored(
        instruments, flat_chain, flat_log_prob, top_indices,
        orbit_time_pad=0.1,
        min_periods=1.0,
        orbit_T=1000,
        orbit_alpha=0.25,
        orbit_lw=0.6,
        cross_color="b",
        cross_ms=6.0,
        cross_mew=1.2,
        zoom_margin=10.0,
        vmin=None,
        vmax=None,
        title="Sum of images + top orbits (colored by log_prob rank)",
        save_path=sum_save_path,
        dpi=200,
    )
    results["sum_top_orbits"] = sum_results
    print(f"[plot] Saved sum_top_orbits plots")

    # 2. images_with_top_orbits plots
    images_save_path = os.path.join(custom_postprocessing_plots_dir, "images_with_top_orbits")
    images_results = plot_images_with_top_orbits_colored(
        instruments, flat_chain, flat_log_prob, top_indices,
        ncols=6,
        circle_radius=4.0,
        top_alpha=0.5,
        top_lw=1.0,
        cmap_image="gray",
        title="Images with top orbits (colored by log_prob rank)",
        save_path=images_save_path,
        vmin=None,
        vmax=None,
        dpi=200,
    )
    results["images_with_top_orbits"] = images_results
    print(f"[plot] Saved images_with_top_orbits plots")

    # 3. Create coadd_native plots for the best orbit in the selection
    try:
        coadd_results, best_theta, best_log_prob, num_orbit_sorted = create_coadd_native_plots(
            instruments, flat_chain, flat_log_prob, top_indices,
            save_dir=custom_postprocessing_plots_dir,
            combine="invvar",
            interpolation_order=1,
            align_to="first",
            dpi=200,
        )
        results["coadd_native"] = coadd_results
    except Exception as e:
        print(f"[plot] Warning: could not create coadd_native plots: {e}")
        best_theta = flat_chain[top_indices[0]]  # Fallback: use first in selection
        best_log_prob = flat_log_prob[top_indices[0]]
        all_sorted_idx = np.argsort(flat_log_prob)[::-1]
        num_orbit_sorted = int(np.where(all_sorted_idx == top_indices[0])[0][0]) + 1

    # 4. Write best_orbit.txt with parameters
    try:
        best_orbit_path = write_best_orbit_info(
            best_theta=best_theta,
            best_log_prob=best_log_prob,
            num_orbit_sorted_log_prob=num_orbit_sorted,
            save_dir=custom_postprocessing_plots_dir,
        )
        results["best_orbit"] = best_orbit_path
    except Exception as e:
        print(f"[plot] Warning: could not write best_orbit.txt: {e}")

    print(f"[plot] Done! All plots and files saved in {custom_postprocessing_plots_dir}")
    return results


def create_coadd_native_plots(
    instruments: Sequence[Instrument],
    flat_samples: np.ndarray,
    flat_log_prob: np.ndarray,
    top_indices: np.ndarray,
    save_dir: str,
    *,
    combine: str = "invvar",
    interpolation_order: int = 1,
    align_to: str = "first",
    dpi: int = 200,
) -> dict:
    """
    Create coadd_native plots for the orbit with the highest log_prob in the selection.
    
    This uses stack_planet_from_posterior_native and show_coadd_native from mcmc.py
    to create plots identical to those generated by mcmc.py.
    
    Parameters
    ----------
    instruments : Sequence[Instrument]
        List of Instrument objects.
    flat_samples : np.ndarray
        (Ns, 7) array of posterior samples.
    flat_log_prob : np.ndarray
        (Ns,) array of log probabilities.
    top_indices : np.ndarray
        Indices of selected samples.
    save_dir : str
        Directory where to save the plots.
    combine : str
        How to combine epochs: "invvar" or "sum".
    interpolation_order : int
        Interpolation order for image shifting.
    align_to : str
        Alignment point: "first" or (x0, y0).
    dpi : int
        DPI for saved figures.
    
    Returns
    -------
    dict
        Metadata about the created plots.
    """
    if not instruments:
        raise ValueError("At least one Instrument is required.")
    
    fs = np.asarray(flat_samples, dtype=float)
    lp = np.asarray(flat_log_prob, dtype=float)
    top_idx = np.asarray(top_indices, dtype=int)
    
    if len(top_idx) == 0:
        raise ValueError("No top indices provided.")
    
    # Find the orbit with the highest log_prob in the selection
    lp_top = lp[top_idx]
    best_idx_in_top = np.argmax(lp_top)  # Index within top_idx
    best_global_idx = top_idx[best_idx_in_top]  # Index in flat_samples
    best_theta = fs[best_global_idx]  # Theta with highest log_prob in selection
    best_log_prob = lp[best_global_idx]
    
    # Use the rank in the FULL sorted list (1-indexed)
    all_sorted_idx = np.argsort(lp)[::-1]  # All samples sorted by log_prob descending
    num_orbit_sorted_log_prob = int(np.where(all_sorted_idx == best_global_idx)[0][0]) + 1  # 1-indexed
    
    print(f"[plot] Creating coadd_native plots for orbit ranked #{num_orbit_sorted_log_prob} (log_prob={best_log_prob:.6f})")
    
    # Create coadds using the best theta
    coadds, meta = stack_planet_from_posterior_native(
        instruments, fs, theta=best_theta, combine=combine,
        interpolation_order=interpolation_order, align_to=align_to,
    )
    
    # Save each coadd plot
    os.makedirs(save_dir, exist_ok=True)
    results = {}
    
    for inst in instruments:
        if inst.name not in coadds:
            continue
            
        coadd_img = coadds[inst.name]
        
        vmin = np.percentile(coadd_img, 1) if coadd_img.size > 0 else 0
        vmax = np.percentile(coadd_img, 99) if coadd_img.size > 0 else 1
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(coadd_img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px]")
        ax.set_title(f"Coadd (native, aligned to first-epoch predicted position) — orbit #{num_orbit_sorted_log_prob} [{inst.name}]")
        
        # Add red circle at recombination point (x0, y0)
        x0 = meta.get(inst.name, {}).get("x0", None)
        y0 = meta.get(inst.name, {}).get("y0", None)
        if x0 is not None and y0 is not None:
            ax.plot([x0], [y0], marker="o", ms=18, mfc="none", mec="r", mew=1.5)
        
        save_path = os.path.join(save_dir, f"coadd_native_{inst.name}.png")
        fig.savefig(save_path, dpi=dpi)
        plt.close(fig)
        
        print(f"[plot] Saved coadd_native plot for {inst.name}: {save_path}")
        results[inst.name] = {"theta": best_theta, "log_prob": best_log_prob, 
                            "num_orbit": num_orbit_sorted_log_prob, "save_path": save_path}
    
    return results, best_theta, best_log_prob, num_orbit_sorted_log_prob


def write_best_orbit_info(
    best_theta: np.ndarray,
    best_log_prob: float,
    num_orbit_sorted_log_prob: int,
    save_dir: str,
) -> str:
    """
    Write best_orbit.txt file with orbit parameters.
    Format: parameter descriptions as comments, then values.
    """
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "best_orbit.txt")
    
    a, la0, m0, h, k, p, q = best_theta
    
    # Calculate additional orbital parameters
    e = np.sqrt(h**2 + k**2)
    sin_i_2 = np.sqrt(p**2 + q**2)
    i = 2.0 * np.arcsin(sin_i_2)  # inclination
    w_sum = np.arctan2(h, k)      # Omega + omega
    Delta = np.arctan2(q, p)      # Omega - omega
    Omega = 0.5 * (w_sum + Delta)  # Longitude of ascending node
    omega = 0.5 * (w_sum - Delta)  # Argument of periapsis
    
    with open(out_path, "w") as f:
        f.write("# Parameter descriptions:\n")
        f.write("# Rank of orbit in sorted log_prob list (1=highest)\n")
        f.write("# a [AU] : semi-major axis\n")
        f.write("# lambda0 [rad] : mean longitude at reference epoch\n")
        f.write("# m0 [solar masses] : stellar mass\n")
        f.write("# h : e*sin(Omega+omega)\n")
        f.write("# k : e*cos(Omega+omega)\n")
        f.write("# p : sin(i/2)*cos(Omega-omega)\n")
        f.write("# q : sin(i/2)*sin(Omega-omega)\n")
        f.write("# log_prob : log probability\n")
        f.write("# e : eccentricity\n")
        f.write("# i [rad] : inclination\n")
        f.write("# Omega [rad] : longitude of ascending node\n")
        f.write("# omega [rad] : argument of periapsis\n\n")
        
        f.write("# Values:\n")
        f.write(f"num_orbit_sorted_log_prob: {num_orbit_sorted_log_prob}\n")
        f.write(f"a: {a}\n")
        f.write(f"lambda0: {la0}\n")
        f.write(f"m0: {m0}\n")
        f.write(f"h: {h}\n")
        f.write(f"k: {k}\n")
        f.write(f"p: {p}\n")
        f.write(f"q: {q}\n")
        f.write(f"log_prob: {best_log_prob}\n")
        f.write(f"e: {e}\n")
        f.write(f"i: {i}\n")
        f.write(f"Omega: {Omega}\n")
        f.write(f"omega: {omega}\n")
    
    print(f"[plot] Wrote best_orbit info to: {out_path}")
    return out_path


def run_plot_from_yaml_cli(yaml_path: str, orbits_range: Optional[str] = None):
    """
    CLI wrapper for run_plot_from_yaml.

    This function is called from cli.py and handles the command-line interface.

    Parameters
    ----------
    yaml_path : str
        Path to the YAML configuration file.
    orbits_range : Optional[str]
        String of the form "N,P" specifying the orbit range.
    """
    print("=" * 70)
    print("K-Stacker Plot Command")
    print("=" * 70)
    
    try:
        run_plot_from_yaml(yaml_path, orbits_range=orbits_range)
    except Exception as e:
        print(f"[plot] ERROR: {e}")
        raise
