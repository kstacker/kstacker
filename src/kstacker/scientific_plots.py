"""Publication-oriented images, sky orbits and projected posterior displays.

Plots are computed from posterior samples without changing model evaluation.
Every image epoch is drawn in its native detector frame, with on-sky East/North
axis coordinates in AU. Instrument-specific images are never silently combined
in uncalibrated flux units.
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import Sequence
import numpy as np
try:
    from .orbital_geometry import _equinoctial_sky_tracks
except ImportError:
    from orbital_geometry import _equinoctial_sky_tracks


def _matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value)).strip("_").lower()


def _save(fig, filename: Path, dpi: int) -> None:
    """Save vector PDF and print-resolution PNG with consistent typography."""
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(filename.with_suffix(".png"), bbox_inches="tight", dpi=dpi)
    _matplotlib().close(fig)


def _tracks(samples: np.ndarray, times: np.ndarray, reference: float):
    """Return arrays (n_sample, n_time) of North and East positions in AU."""
    a, longitude, mass, h, k, p, q = np.asarray(samples, float)[:, :7].T
    return _equinoctial_sky_tracks(a, longitude, mass, h, k, p, q,
                                    np.asarray(times, float), t_ref=reference)



def _symmetric_image_limits(data: np.ndarray) -> tuple[float, float]:
    """Exact symmetric limits about zero, computed from finite image pixels.

    A single outlying pixel can make the display less contrasty; that is
    intentional here because the requested colour scale includes the full
    observed range. The fallback is only for zero-valued or empty arrays.
    """
    finite = np.asarray(data, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return -1.0, 1.0
    alpha = max(abs(float(finite.min())), abs(float(finite.max())))
    return (-alpha, alpha) if alpha > 0 else (-1.0, 1.0)


def _map_position(ax, east: float, north: float, *, scale: float,
                  color: str = "#ffdf5b", radius_px: float = 4.0,
                  label: bool = True) -> None:
    """Annotate the MAP planet position with a readable open circle.

    The location is given in (East, North) AU and is evaluated with the exact
    same orbital propagator as the likelihood. The marker is deliberately an
    *outline only*: it highlights the location while leaving the underlying
    image visible. ``radius_px`` is a display radius in native detector pixels;
    it is not a photometric aperture and does not represent an uncertainty.
    """
    from matplotlib.patheffects import Stroke, Normal
    plt = _matplotlib()
    radius = float(radius_px) / float(scale)
    if not np.isfinite([east, north, radius]).all() or radius <= 0:
        return
    ring = plt.Circle((float(east), float(north)), radius, fill=False,
                      edgecolor=color, linewidth=1.55, zorder=12)
    ring.set_path_effects([Stroke(linewidth=2.8, foreground="#12273a"), Normal()])
    ax.add_patch(ring)
    if label:
        ax.annotate("MAP", (east, north), xytext=(7, 8),
                    textcoords="offset points", color="#183a55", fontsize=10,
                    weight="bold", zorder=13,
                    path_effects=[Stroke(linewidth=2.2, foreground="white"), Normal()])


def _image_frame(instrument, epoch: int, *, noise_floor: float) -> tuple[np.ndarray, float, str]:
    """Return a display-only map, spatial factor and quantity label.

    PACO maps use alpha_hat/sqrt(var_alpha). Classical likelihood products are
    shown in their stored pixel values; no visual conversion is used by MCMC.
    """
    noise_floor = float(instrument.noise_floor if instrument.noise_floor is not None else noise_floor)
    if instrument.photometry_method == "paco":
        with np.errstate(divide="ignore", invalid="ignore"):
            variance = np.asarray(instrument.paco_var_alpha_maps[epoch], float)
            signal = np.asarray(instrument.paco_alpha_maps[epoch], float)
            display = np.divide(signal, np.sqrt(np.maximum(variance, noise_floor**2)),
                                out=np.full_like(signal, np.nan), where=variance > 0)
        return display, float(instrument.paco_oversampling), "PACO $\\hat\\alpha/\\sigma_\\alpha$"
    if instrument.photometry_method == "convolve" and instrument.images_up is not None:
        return (np.asarray(instrument.images_up[epoch], float),
                float(instrument.upsampling_factor), "Convolved image [image units]")
    if instrument.images_native is not None:
        return np.asarray(instrument.images_native[epoch], float), 1.0, "Image [native units]"
    raise ValueError(f"No displayable image for {instrument.name}")


def _image_extent(instrument) -> tuple[float, float, float, float]:
    """Detector image extent, East increasing leftwards and North upwards."""
    # imshow(extent=...) expects pixel EDGES, not the coordinates of the
    # outermost pixel centres. An N-pixel detector has edges at +/-N/2 px.
    # Using (N-1)/2 shifts every plotted image pixel by half a native pixel.
    radius_au = float(instrument.size) / (2.0 * float(instrument.scale))
    return (+radius_au, -radius_au, -radius_au, +radius_au)


def _style():
    """Use a clean vector-exportable figure style with readable fonts."""
    plt = _matplotlib()
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.labelsize": 12, "axes.titlesize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    return plt


def _probability_contours(ax, east: np.ndarray, north: np.ndarray, color: str) -> None:
    """Draw approximate 68/95% highest-density regions on the sky.

    Contours use sample occupancy, not the maximum likelihood at a location.
    They can be multimodal and are therefore preferable to symmetric error bars.
    """
    if len(east) < 40:
        return
    valid = np.isfinite(east) & np.isfinite(north)
    if valid.sum() < 40:
        return
    e, n = east[valid], north[valid]
    extent = np.nanpercentile(np.r_[e, n], [0.5, 99.5])
    if not np.isfinite(extent).all() or extent[1] <= extent[0]:
        return
    # Use a range shared by both sky axes to avoid degenerate histograms.
    h, eb, nb = np.histogram2d(e, n, bins=48, range=[extent, extent])
    positive = np.sort(h[h > 0])[::-1]
    if positive.size < 2:
        return
    cumulative = np.cumsum(positive) / positive.sum()
    thresholds = [float(positive[min(len(positive)-1, np.searchsorted(cumulative, q))])
                  for q in (0.95, 0.68)]
    levels = sorted(set(thresholds))
    if levels and np.max(h) > min(levels):
        x = (eb[:-1] + eb[1:])/2
        y = (nb[:-1] + nb[1:])/2
        ax.contour(x, y, h.T, levels=levels, colors=[color]*len(levels),
                   linewidths=[0.9]*len(levels), alpha=0.9)


def _complete_orbit_tracks(samples: np.ndarray, phase: np.ndarray):
    """Propagate one full Keplerian period per posterior orbit sample.

    ``phase`` ranges from 0 to 1.  Each orbit has its own period, so a
    shared physical-time grid would not close the trajectories.  The same
    equinoctial Kepler solver and sky projection as the MCMC are used.
    """
    try:
        from .orbital_geometry import equinoctial_xy_from_hk, project_equinoctial_sin_i2
    except ImportError:
        from orbital_geometry import equinoctial_xy_from_hk, project_equinoctial_sin_i2
    state = np.asarray(samples, float)
    phase = np.asarray(phase, float)
    if state.ndim != 2 or state.shape[1] < 7 or phase.ndim != 1:
        raise ValueError("Expected samples (N, >=7) and a 1-D orbital phase grid")
    a, longitude, _, h, k, p, q = state[:, :7].T
    lam = longitude[:, None] + (2.0 * np.pi) * phase[None, :]
    X, Y = equinoctial_xy_from_hk(a[:, None], lam, h[:, None], k[:, None])
    return project_equinoctial_sin_i2(X, Y, p[:, None], q[:, None])


def plot_joint_sky_orbits(instruments: Sequence, samples: np.ndarray,
                          log_probability: np.ndarray, output_dir: Path, dpi: int,
                          max_orbit_tracks: int = 900) -> None:
    """Show the full MAP orbit over pale full-period posterior orbit draws.

    Every trajectory is produced with the same equinoctial geometry as the
    image likelihood. The MAP orbit is highlighted with its own colour so it
    remains immediately identifiable, while the remaining posterior draws are
    shown in a paler common hue. Epoch annotations are open circles evaluated
    from the MAP state at the instrument observing times.
    """
    plt = _style()
    fig, ax = plt.subplots(figsize=(8.2, 7.4), constrained_layout=True)
    t_ref = float(instruments[0].t_ref)
    map_idx = int(np.nanargmax(log_probability))
    state = np.asarray(samples[map_idx:map_idx + 1], float)
    max_orbit_tracks = max(1, int(max_orbit_tracks))
    rng = np.random.default_rng(2813)
    draw_count = min(len(samples), max_orbit_tracks)
    chosen = rng.choice(len(samples), size=draw_count, replace=False)
    phase = np.linspace(0, 1, 361)

    posterior_color = "#4a90c2"
    map_color = "#c43c39"
    north, east = _complete_orbit_tracks(samples[chosen], phase)
    for e, n in zip(east, north):
        ax.plot(e, n, lw=0.65, color=posterior_color, alpha=0.08,
                zorder=1, rasterized=True)
    map_n, map_e = _complete_orbit_tracks(state, phase)
    ax.plot(map_e[0], map_n[0], color=map_color, lw=2.6,
            label="MAP orbit", zorder=5)
    ax.plot([], [], color=posterior_color, lw=2.2, alpha=0.45,
            label=f"Posterior orbits (n={draw_count:,})")

    colors = ("#d45b00", "#248b73", "#8e56b0", "#bb3975")
    for j, inst in enumerate(instruments):
        color = colors[j % len(colors)]
        epoch_n, epoch_e = _tracks(state, inst.ts, t_ref)
        # Circle radius = 2 x FWHM of THIS instrument, converted from native
        # pixels to AU with its own plate scale (scale is in px/AU).
        fwhm_px = float(getattr(inst, "fwhm", None) or 1.5)
        radius_px = 2.0 * fwhm_px
        radius_au = radius_px / float(inst.scale)
        for k, (e, n) in enumerate(zip(epoch_e[0], epoch_n[0]), 1):
            _map_position(ax, e, n, scale=float(inst.scale), color=color,
                          radius_px=radius_px, label=False)
            ax.annotate(str(k), (e, n), xytext=(7, 6),
                        textcoords="offset points", fontsize=10,
                        color=color, weight="bold")
        ax.plot([], [], ls="none", marker="o", markersize=9, markerfacecolor="none",
                markeredgecolor=color, markeredgewidth=1.6,
                label=f"{inst.name} epochs (circle r = 2 FWHM = {radius_au:.2f} AU)")

    ax.scatter([0], [0], s=70, marker="*", color="#e2a449",
               edgecolors="#1b3550", linewidths=0.7, zorder=10,
               label="Host star")
    ax.set(xlabel="East offset [AU]", ylabel="North offset [AU]",
           title="Projected Keplerian orbits")
    ax.invert_xaxis()
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=10.5, frameon=False, loc="best")
    _save(fig, output_dir / "projected_orbits", dpi)


def _align_instrument_images(inst, map_state: np.ndarray, noise_floor: float,
                             *, mask_to_working_region: bool = True):
    """Shift one instrument's epochs onto its first MAP planet position.

    The same orbit solution and sky convention as the photometry define the
    shifts. x_native = centre - East * scale, y_native = centre + North * scale.
    Consequently the shift (in image pixels) is
      dx = (East_epoch - East_ref) * scale * upsampling,
      dy = (North_ref - North_epoch) * scale * upsampling.

    NaNs and the native coronagraphic mask are transported with each exposure;
    only valid overlapping pixels enter the unweighted display mean. No
    inverse-variance or cross-instrument pooling is performed, and the stack is
    never used in the inference likelihood.

    Returns (reference_image, aligned_mean, contributing_epoch_count, extent,
             unit_label, reference_east, reference_north).
    """
    from scipy.ndimage import shift as image_shift

    sample = np.asarray(map_state, dtype=float).reshape(1, -1)
    north, east = _tracks(sample, np.asarray(inst.ts, float), float(inst.t_ref))
    north, east = north[0], east[0]
    frames = []
    factor = None
    unit_label = None
    for epoch in range(len(inst.ts)):
        image, frame_factor, frame_label = _image_frame(inst, epoch,
                                                       noise_floor=noise_floor)
        image = np.asarray(image, dtype=float)
        if factor is None:
            factor, unit_label = float(frame_factor), frame_label
        elif image.shape != frames[0].shape or frame_factor != factor:
            raise ValueError(f"Inconsistent frame sizes or scales in {inst.name}")
        frames.append(_working_region_image(image, inst, factor=factor)
                      if mask_to_working_region else image)

    ref = frames[0]
    numerator = np.zeros_like(ref, dtype=float)
    coverage = np.zeros_like(ref, dtype=float)
    for k, frame in enumerate(frames):
        # A valid-pixel mask is interpolated using the same transformation as
        # the signal, then used to normalize boundary/NaN transitions.
        shift_xy = ((north[0] - north[k]) * float(inst.scale) * factor,
                    (east[k] - east[0]) * float(inst.scale) * factor)
        valid = np.isfinite(frame)
        values = np.where(valid, frame, 0.0)
        shifted_values = image_shift(values, shift_xy, order=1,
                                     mode="constant", cval=0.0,
                                     prefilter=False)
        shifted_weights = image_shift(valid.astype(float), shift_xy, order=1,
                                      mode="constant", cval=0.0,
                                      prefilter=False)
        good = shifted_weights > 1e-8
        numerator[good] += shifted_values[good] / shifted_weights[good]
        coverage[good] += 1.0
    aligned_mean = np.full_like(ref, np.nan, dtype=float)
    good = coverage > 0
    aligned_mean[good] = numerator[good] / coverage[good]
    return ref, aligned_mean, coverage, _image_extent(inst), unit_label, east[0], north[0]


def plot_aligned_instrument_stack(inst, map_state: np.ndarray, output_dir: Path,
                                  dpi: int, noise_floor: float,
                                  mask_to_working_region: bool = True) -> None:
    """Display the MAP-aligned mean image for one instrument.

    All epochs of the instrument are shifted so that their MAP-predicted planet
    positions coincide with the position at the first observing epoch. Only the
    final aligned mean is shown: the reference image is deliberately omitted to
    keep the figure focused on the registered stack.

    The stack is for interpretation only. It is not a detection statistic and
    is never fed back into the MCMC likelihood.
    """
    plt = _style()
    reference, aligned, coverage, extent, label, east, north = _align_instrument_images(
        inst, map_state, noise_floor,
        mask_to_working_region=mask_to_working_region,
    )
    fig, ax = plt.subplots(1, 1, figsize=(5.3, 4.8), constrained_layout=True)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#e8edf1")
    low, high = _symmetric_image_limits(aligned)
    im = ax.imshow(aligned, origin="lower", extent=extent, cmap=cmap,
                   vmin=low, vmax=high, interpolation="nearest", aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.047, pad=0.03, label=label)
    _map_position(ax, east, north, scale=float(inst.scale),
                  radius_px=4.0 * float(getattr(inst, "fwhm", 1.5)),
                  color="#f6b638", label=True)
    contributing = int(np.nanmax(coverage)) if np.isfinite(coverage).any() else 0
    ax.set(xlabel="East [AU]", ylabel="North [AU]",
           title=f"MAP-aligned mean ({contributing} epochs)")
    fig.suptitle(f"{inst.name} | Aligned observation stack", fontsize=11)
    _save(fig, output_dir / f"{_slug(inst.name)}_aligned_stack", dpi)


def _working_region_image(image: np.ndarray, inst, *, factor: float) -> np.ndarray:
    """Mask pixels excluded by the likelihood's inner/outer working angles.

    Coordinates of the pixel centres are expressed in native pixels; the
    center corresponds to (N-1)/2 for a native even-sized array. Retaining
    NaNs rather than assigning zero prevents masked pixels from controlling
    the signed display color scale. No image data or likelihood is modified.
    """
    array = np.asarray(image, dtype=float)
    yy, xx = np.indices(array.shape, dtype=float)
    rr = np.hypot(yy - (array.shape[0]-1)/2.0,
                  xx - (array.shape[1]-1)/2.0) / float(factor)
    good = np.isfinite(array)
    if getattr(inst, "r_mask", None) is not None:
        good &= rr > float(inst.r_mask)
    if getattr(inst, "r_mask_ext", None) is not None:
        good &= rr < float(inst.r_mask_ext)
    return np.where(good, array, np.nan)


def plot_epoch_images(inst, samples: np.ndarray, log_probability: np.ndarray,
                      output_dir: Path, dpi: int, noise_floor: float,
                      mask_to_working_region: bool = True) -> None:
    """Display each epoch with independent symmetric limits and the MAP sample.

    Display maps preserve the raw photometric units. By default the central
    mask and exterior excluded pixels are shown in pale grey, and the colorbar
    uses the exact finite min/max of the displayed valid working region.
    Set ``mask_to_working_region=False`` to inspect the full image instead.
    Neither option changes the likelihood or the posterior samples.
    """
    plt = _style()
    ix_best = int(np.nanargmax(log_probability))
    ns, es = _tracks(samples, inst.ts, float(inst.t_ref))
    nbest, ebest = ns[ix_best], es[ix_best]
    extent = _image_extent(inst)
    n_epochs = len(inst.ts)
    for page_start in range(0, n_epochs, 9):
        selected = list(range(page_start, min(page_start + 9, n_epochs)))
        ncols = min(3, len(selected)); nrows = int(np.ceil(len(selected)/ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(4.0*ncols, 3.7*nrows),
                                squeeze=False, constrained_layout=True)
        axs_flat = axs.ravel()
        for panel, k in enumerate(selected):
            ax = axs_flat[panel]
            try:
                image, factor, label = _image_frame(inst, k, noise_floor=noise_floor)
            except ValueError:
                ax.set_visible(False)
                continue
            # Only valid science pixels set the display color limits. The
            # full image can be requested by setting plots.mask_to_working_region
            # to false; this does not change the inference masks.
            shown = (_working_region_image(image, inst, factor=factor)
                     if mask_to_working_region else np.asarray(image, float))
            low, high = _symmetric_image_limits(shown)
            cmap = plt.get_cmap("RdBu_r").copy()
            cmap.set_bad("#e9eef2")
            artist = ax.imshow(shown, origin="lower", cmap=cmap, vmin=low, vmax=high,
                               extent=extent, interpolation="nearest", aspect="equal")
            fig.colorbar(artist, ax=ax, fraction=0.045, pad=0.03, label=label)
            _map_position(ax, ebest[k], nbest[k], scale=float(inst.scale),
                          color="#ffdf5b", radius_px=4.0 * float(getattr(inst, "fwhm", 1.5)), label=True)
            if inst.r_mask is not None:
                inner = float(inst.r_mask) / float(inst.scale)
                ax.add_patch(plt.Circle((0,0), inner, color="white", fill=False,
                                        ls=":", lw=0.8, alpha=0.75))
            ax.set_title(f"Epoch {k+1}  ·  t = {inst.ts[k]:.3f} yr", fontsize=10)
            ax.set_xlabel("East [AU]")
            ax.set_ylabel("North [AU]")
            # `extent=(+east,-east,-north,+north)` already points East left.
            # Inverting again would mirror the FITS image relative to its orbit.
        for ax in axs_flat[len(selected):]:
            ax.set_visible(False)
        fig.suptitle(f"{inst.name} | Observation images and MAP positions", fontsize=11)
        _save(fig, output_dir / f"{_slug(inst.name)}_observations_{page_start//9+1:02d}", dpi)


def plot_position_log_density(inst, samples: np.ndarray, output_dir: Path, dpi: int) -> None:
    """Map projected log10 posterior position density at every observing epoch.

    The plotted scalar is log10 of spatial sample occupancy. It is NOT the
    logarithm of the pixel-level image likelihood: all epochs jointly constrain
    every orbit sample.
    """
    plt = _style()
    ns, es = _tracks(samples, inst.ts, float(inst.t_ref))
    for first in range(0, len(inst.ts), 9):
        chosen = list(range(first, min(first+9, len(inst.ts))))
        ncols=min(3, len(chosen)); nrows=int(np.ceil(len(chosen)/ncols))
        fig, axes=plt.subplots(nrows,ncols,figsize=(4*ncols,3.5*nrows),squeeze=False,
                              constrained_layout=True)
        image_artist = None
        visible_axes = []
        for ax,k in zip(axes.ravel(),chosen):
            radius=float(inst.size)/(2*float(inst.scale))
            limits=[-radius,radius]
            hist,xe,ye=np.histogram2d(es[:,k],ns[:,k],bins=90,range=[limits,limits])
            hist=hist/max(1, hist.sum())
            map_log=np.where(hist.T>0,np.log10(np.maximum(hist.T,1e-300)),np.nan)
            # Retain the plotted image rather than relying on the final axis:
            # the last grid cell can be an unused, invisible subplot.
            image_artist = ax.imshow(map_log,origin="lower",cmap="magma",extent=[*limits,*limits],
                                     interpolation="nearest",aspect="equal",vmin=-5,vmax=-1)
            visible_axes.append(ax)
            ax.set(title=f"Epoch {k+1} · t={inst.ts[k]:.3f} yr",
                   xlabel="East [AU]",ylabel="North [AU]")
            ax.invert_xaxis()
        for ax in axes.ravel()[len(chosen):]:ax.set_visible(False)
        if image_artist is not None:
            fig.colorbar(image_artist, ax=visible_axes,
                         label="log$_{10}$ position probability per sky bin",
                         fraction=0.022, pad=0.02)
        fig.suptitle(f"{inst.name} | Projected position probability", fontsize=11)
        _save(fig,output_dir/f"{_slug(inst.name)}_position_probability_{first//9+1:02d}",dpi)


def _draw_corner(samples: np.ndarray, names: list[str], title: str,
                 destination: Path, dpi: int) -> None:
    """Render readable marginal and joint posterior distributions.

    Each diagonal includes the median and central 68% quantiles. Joint panels
    use a two-dimensional binned density with logarithmic occupancy scaling,
    rather than a forest of overplotted scatter markers. Binning is a display
    operation only and does not change saved samples or inferred probabilities.
    """
    from matplotlib.colors import LogNorm

    plt = _style()
    data = np.asarray(samples, dtype=float)
    ndim = len(names)
    if data.ndim != 2 or data.shape[1] != ndim:
        raise ValueError("Corner plot columns must match parameter labels")
    if ndim < 2:
        raise ValueError("At least two coordinates are required")
    if len(data) < 2:
        raise ValueError("At least two posterior samples are required")
    fig, axes = plt.subplots(ndim, ndim, figsize=(2.8 * ndim, 2.8 * ndim),
                             squeeze=False, constrained_layout=True)
    bounds = []
    for dim in range(ndim):
        finite = data[:, dim][np.isfinite(data[:, dim])]
        if not finite.size:
            raise ValueError("Cannot plot a coordinate without finite samples")
        lo, hi = np.min(finite), np.max(finite)
        if hi - lo < max(1e-12, 1e-12 * abs(lo)):
            pad = 0.05 * max(abs(lo), 1.0)
        else:
            pad = 0.035 * (hi - lo)
        bounds.append((lo - pad, hi + pad))
    for row in range(ndim):
        for col in range(ndim):
            ax = axes[row, col]
            if col > row:
                ax.axis("off")
                continue
            ax.set_xlim(bounds[col])
            if row == col:
                finite = data[:, row][np.isfinite(data[:, row])]
                low, median, high = np.quantile(finite, [0.16, 0.5, 0.84])
                ax.hist(finite, bins=50, range=bounds[col], density=True,
                        color="#3487a7", alpha=0.84, edgecolor="white", linewidth=0.3)
                ax.axvline(median, color="#143f59", lw=1.5)
                ax.axvline(low, color="#c57d3c", lw=1, ls="--")
                ax.axvline(high, color="#c57d3c", lw=1, ls="--")
                ax.set_title(f"{median:.3g} (+{high-median:.2g}/-{median-low:.2g})",
                             fontsize=9, pad=3)
            else:
                hist, xedges, yedges = np.histogram2d(data[:, col], data[:, row],
                    bins=48, range=[bounds[col], bounds[row]])
                positive = np.ma.masked_less_equal(hist.T, 0)
                ax.pcolormesh(xedges, yedges, positive,
                              cmap="viridis", norm=LogNorm(vmin=1,
                                  vmax=max(2, float(np.max(hist)))),
                              rasterized=True, shading="flat")
                ax.set_ylim(bounds[row])
            ax.tick_params(axis="both", labelsize=8.5, length=2.8,
                           direction="in", pad=2.5)
            if row == ndim - 1:
                ax.set_xlabel(names[col], fontsize=11, labelpad=6)
                ax.tick_params(axis="x", labelrotation=30)
            else:
                ax.tick_params(labelbottom=False)
            if row > col and col == 0:
                ax.set_ylabel(names[row], fontsize=11, labelpad=5)
            elif col > 0:
                ax.tick_params(labelleft=False)
    fig.suptitle(title, fontsize=15)
    _save(fig, destination, dpi)


def plot_orbital_corner(samples: np.ndarray, output_dir: Path, dpi: int,
                        flux_labels: Sequence[str] = ()) -> None:
    """Display posterior projections in the MCMC's regular orbital coordinates."""
    names = [r"$a$ [AU]", r"$\lambda_0$ [rad]", r"$M_\star$ [$M_\odot$]",
             r"$h$", r"$k$", r"$p$", r"$q$"]
    names.extend(flux_labels or [rf"$f_{{p,{j+1}}}$" for j in range(samples.shape[1] - 7)])
    _draw_corner(samples, names, "Keplerian posterior — sampling coordinates",
                 output_dir / "orbital_parameters", dpi)


def plot_physical_orbital_corner(samples: np.ndarray, output_dir: Path, dpi: int,
                                 flux_labels: Sequence[str] = ()) -> None:
    """Display derived physical elements with a documented singularity gauge.

    At e=0 the argument of periapsis is undefined and fixed to zero; at i=0
    the ascending node is undefined and fixed to zero. These elements are
    presentation coordinates only and do not enter likelihood evaluation.
    """
    a, lam, mass, h, k, p, q = np.asarray(samples, float)[:, :7].T
    eccentricity = np.hypot(h, k)
    inclination = 2 * np.arcsin(np.sqrt(np.clip(p*p + q*q, 0, 1)))
    ascending_node = np.where(np.hypot(p, q) > 1e-15, np.arctan2(p, q), 0.0)
    peri_longitude = np.where(eccentricity > 1e-15, np.arctan2(h, k), ascending_node)
    arg_periapsis = np.mod(peri_longitude - ascending_node, 2*np.pi)
    mean_anomaly = np.mod(lam - peri_longitude, 2*np.pi)
    physical = np.column_stack([
        a, eccentricity, np.degrees(inclination),
        np.degrees(np.mod(ascending_node, 2*np.pi)),
        np.degrees(arg_periapsis), np.degrees(mean_anomaly), mass,
    ])
    names = [r"$a$ [AU]", r"$e$", r"$i$ [deg]", r"$\Omega$ [deg]",
             r"$\omega$ [deg]", r"$M_0$ [deg]", r"$M_\star$ [$M_\odot$]"]
    if samples.shape[1] > 7:
        physical = np.column_stack([physical, samples[:, 7:]])
        names.extend(flux_labels or [rf"$f_{{p,{j+1}}}$" for j in range(samples.shape[1] - 7)])
    _draw_corner(physical, names, "Keplerian posterior — physical orbital elements",
                 output_dir / "physical_orbit", dpi)


def create_scientific_figures(instruments: Sequence, samples: np.ndarray,
                              log_probability: np.ndarray, *, output_dir,
                              noise_floor: float, max_posterior_samples: int=12000,
                              max_orbit_tracks: int=900,
                              dpi: int=200,
                              mask_to_working_region: bool=True,
                              align_to_first_epoch: bool=True) -> None:
    """Export physical plots and all-epoch posterior overlays as PDF and PNG."""
    if not instruments:return
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    samples=np.asarray(samples,float); log_probability=np.asarray(log_probability,float)
    valid=np.isfinite(samples).all(axis=1)&np.isfinite(log_probability)
    samples=samples[valid]; log_probability=log_probability[valid]
    if len(samples)==0:
        raise ValueError("No finite posterior samples for physical figures")
    max_posterior_samples = max(1, int(max_posterior_samples))
    if len(samples) > max_posterior_samples:
        # A fixed stride through an emcee chain flattened in (step,walker)
        # order can alias the walker dimension and omit whole modes. Draw
        # uniformly without replacement across ALL retained walker states.
        # Always retain the best-scoring state for the reference-arc overlay.
        rng = np.random.default_rng(2813)
        best = int(np.argmax(log_probability))
        if max_posterior_samples == 1:
            ids = np.array([best], dtype=int)
        else:
            # Draw only from non-MAP states, then include MAP exactly once.
            pool = np.arange(len(samples), dtype=int)
            pool = pool[pool != best]
            ids = np.sort(np.r_[rng.choice(pool, size=max_posterior_samples - 1,
                                          replace=False), best])
        samples = samples[ids]
        log_probability = log_probability[ids]
    print(f"[plots] scientific figures -> {output_dir}",flush=True)
    flux_labels = [f"Flux: {inst.name}" for inst in instruments if inst.flux_index is not None]
    if samples.shape[1] != 7 + len(flux_labels):
        raise ValueError("The chain does not match the configured instrument flux parameters")
    plot_joint_sky_orbits(instruments,samples,log_probability,output_dir,dpi,
                          max_orbit_tracks=max_orbit_tracks)
    plot_orbital_corner(samples,output_dir,dpi,flux_labels)
    plot_physical_orbital_corner(samples,output_dir,dpi,flux_labels)
    for inst in instruments:
        plot_epoch_images(inst,samples,log_probability,output_dir,dpi,noise_floor,
                          mask_to_working_region=mask_to_working_region)
        plot_position_log_density(inst,samples,output_dir,dpi)
        if align_to_first_epoch and len(inst.ts) > 1:
            plot_aligned_instrument_stack(
                inst, samples[int(np.nanargmax(log_probability))], output_dir,
                dpi, noise_floor, mask_to_working_region=mask_to_working_region,
            )

class _StoredChainSampler:
    """Read saved cold-chain arrays using the interface expected by diagnostics."""

    def __init__(self, h5_file):
        self.chain = np.asarray(h5_file["chain"], dtype=float)
        self.logp = np.asarray(h5_file["log_prob"], dtype=float)
        self.acceptance_fraction = np.asarray(h5_file["acceptance_fraction"], dtype=float)
        if "pt_betas_final" in h5_file:
            self.pt_sampler = _StoredTemperingDiagnostics(h5_file)
            # Absent from files written before the burn-in fix: nothing is cut.
            self.burnin_sweeps = int(h5_file.attrs.get("pt_burnin_sweeps", 0))
            self.lines_per_sweep = int(h5_file.attrs.get("pt_lines_per_sweep", 1))
            # Since iteration 08: `chain` keeps one production line in
            # `chain_stride`; absent attributes = every line (older files).
            self.chain_stride = int(h5_file.attrs.get("chain_stride", 1))
            if "pt_chain_burnin_rows" in h5_file.attrs:
                self.pt_chain_burnin_rows = int(h5_file.attrs["pt_chain_burnin_rows"])

    def get_chain(self, discard=0, thin=1, flat=False):
        values = self.chain[int(discard)::max(1, int(thin))]
        return values.reshape(-1, values.shape[-1]) if flat else values

    def get_log_prob(self, discard=0, thin=1, flat=False):
        values = self.logp[int(discard)::max(1, int(thin))]
        return values.reshape(-1) if flat else values


class _StoredTemperingDiagnostics:
    """Expose optional APT histories saved in the chain HDF5 file."""

    def __init__(self, handle):
        self.betas = np.asarray(handle["pt_betas_final"], dtype=float)
        self._chain = np.asarray(handle["pt_chain"], dtype=float) if "pt_chain" in handle else None
        self._swap = np.asarray(handle["pt_swap_history"], dtype=float) if "pt_swap_history" in handle else None
        self._smd = np.asarray(handle["pt_smd_history"], dtype=float) if "pt_smd_history" in handle else None

    def get_chain(self, flat=False):
        if self._chain is None:
            raise AttributeError("The full temperature chain was not stored")
        return self._chain

    def get_tsw(self):
        if self._swap is None:
            raise AttributeError("Swap history was not stored")
        return self._swap

    def get_smd(self):
        if self._smd is None:
            raise AttributeError("Adaptation history was not stored")
        return self._smd


def _explicit_plot_overrides_disabled(plots_settings) -> bool:
    """Return True: the explicit `kstacker plot` command always writes figures.

    When `plots.enabled` is false, print that the setting is ignored for this
    command only (it is still respected at the end of `kstacker init/mcmc`).
    """
    if not bool((plots_settings or {}).get("enabled", True)):
        try:
            from .console import console_status
        except ImportError:
            from console import console_status
        console_status(
            "plots",
            "plots.enabled is false in the YAML: ignored by the explicit 'kstacker plot' "
            "command, figures are written anyway (the setting is still respected at the "
            "end of 'kstacker mcmc' and 'kstacker init').",
            tone="yellow",
        )
    return True


def create_figures_from_yaml(yaml_path: str) -> None:
    """Regenerate physical and convergence figures from a saved MCMC chain.

    The YAML resolves image products, time vectors and instrument geometry;
    saved HDF5 arrays supply the posterior without repeating any sampling.
    """
    from pathlib import Path
    import h5py
    try:
        from .utils import Params
        from .mcmc import _prepare_runtime_config_from_files, _resolve_tref, configure_instrument_statistics
        from .mcmc_init_search import _load_instruments_for_search
        from .mcmc_diagnostics import create_mcmc_diagnostics
    except ImportError:
        from utils import Params
        from mcmc import _prepare_runtime_config_from_files, _resolve_tref, configure_instrument_statistics
        from mcmc_init_search import _load_instruments_for_search
        from mcmc_diagnostics import create_mcmc_diagnostics
    yaml_path = str(Path(yaml_path).expanduser().resolve())
    params = Params.read(yaml_path)
    root = _prepare_runtime_config_from_files(yaml_path, params._params, verbose=False)
    params._params = root
    params.work_dir = str(Path(yaml_path).parent)
    images_config = root.get("instruments", [])
    inst_times = []
    for instrument_cfg in images_config:
        combined = dict(root)
        combined.update(instrument_cfg)
        params._params = combined
        inst_times.append(np.asarray(params.get_ts(use_p_prev=True), dtype=float))
    if not inst_times:
        raise ValueError("No instruments available for plotting")
    params._params = root
    t_ref = _resolve_tref(params, root, np.concatenate(inst_times))
    instruments = _load_instruments_for_search(
        params, root, t_ref=t_ref, search_backend="auto", instrument_ts=inst_times,
    )
    configure_instrument_statistics(root, instruments, verbose=False)
    noise_floor = min(float(inst.noise_floor) for inst in instruments)
    params._params = root
    output_settings = root.get("outputs", {}) or {}
    plots_settings = root.get("plots", {}) or {}
    chain_file = Path(params.get_path("values_dir")) / str(
        output_settings.get("chain_filename", "mcmc_chain_and_logprob.h5")
    )
    with h5py.File(chain_file, "r") as handle:
        parameter_names = __import__("json").loads(handle.attrs["parameter_names"]) if "parameter_names" in handle.attrs else None
        chain = _StoredChainSampler(handle)
        # The native HDF5 chain contains production samples only. Use all
        # walkers, including those omitted from an optional thinned export.
        samples = chain.get_chain(flat=True)
        scores = chain.get_log_prob(flat=True)
        # reddemcee files since iteration 08 keep one line in `chain_stride`;
        # add the MAP over every production line, as `kstacker mcmc` does.
        if "map_state" in handle and "map_log_prob" in handle.attrs:
            samples = np.vstack([samples, np.asarray(handle["map_state"], dtype=float)[None, :]])
            scores = np.concatenate([scores, [float(handle.attrs["map_log_prob"])]])
    figure_dir = Path(params.get_path("values_dir")) / str(plots_settings.get("directory", "figures"))
    # `kstacker plot` is an explicit request for figures: `plots.enabled`
    # only controls the automatic figures at the end of `kstacker init/mcmc`.
    if _explicit_plot_overrides_disabled(plots_settings):
        create_scientific_figures(
            instruments, samples, scores, output_dir=figure_dir / "science",
            noise_floor=noise_floor,
            max_posterior_samples=int(plots_settings.get("max_posterior_samples", 12000)),
            max_orbit_tracks=int(plots_settings.get("max_orbit_tracks", 900)),
            dpi=int(plots_settings.get("dpi", 200)),
            mask_to_working_region=bool(plots_settings.get("mask_to_working_region", True)),
            align_to_first_epoch=bool(plots_settings.get("align_to_first_epoch", True)),
        )
        create_mcmc_diagnostics(
            chain, output_dir=figure_dir / "convergence",
            dpi=int(plots_settings.get("dpi", 200)),
            parameter_names=parameter_names,
        )
