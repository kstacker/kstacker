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
                  color: str = "#ffdf5b", radius_px: float = 1.25,
                  label: bool = True) -> None:
    """Annotate the MAP planet position with a small, open circle.

    The location is in (East, North) AU, from the SAME equinoctial orbit
    propagator as the likelihood. The radius is a display size in native
    detector pixels and does not represent a confidence or photometric region.
    The open centre leaves the underlying FITS signal visible.
    """
    from matplotlib.patheffects import Stroke, Normal
    plt = _matplotlib()
    radius = float(radius_px) / float(scale)
    if not np.isfinite([east, north, radius]).all() or radius <= 0:
        return
    ring = plt.Circle((float(east), float(north)), radius, fill=False,
                      edgecolor=color, linewidth=1.8, zorder=12)
    ring.set_path_effects([Stroke(linewidth=3.1, foreground="#102b40"), Normal()])
    ax.add_patch(ring)
    if label:
        ax.annotate("MAP", (east, north), xytext=(5, 6),
                    textcoords="offset points", color="#164363", fontsize=7,
                    weight="bold", zorder=13,
                    path_effects=[Stroke(linewidth=2.0, foreground="white"), Normal()])


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
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelsize": 10, "axes.titlesize": 11,
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
    """Plot complete posterior orbits and the maximum-score observed arc.

    Random, reproducible posterior draws reveal multimodal geometry without
    selecting only neighbouring points of the flattened MCMC chain.  The
    highlighted arc is evaluated ONLY over the observed time interval; the
    faint curves each cover a complete (orbit-specific) Keplerian period.
    The small MAP circles are display annotations, not position-confidence regions.
    """
    plt = _style()
    fig, ax = plt.subplots(figsize=(8.1, 7.3), constrained_layout=True)
    t_ref = float(instruments[0].t_ref)
    epochs = np.concatenate([np.asarray(i.ts, float) for i in instruments])
    arc_times = np.linspace(float(epochs.min()), float(epochs.max()), 400)
    ix_best = int(np.nanargmax(log_probability))

    # Include more independent, uniformly drawn posterior samples, capped to
    # keep vector PDF output tractable. Draw indices are reproducible.
    max_orbit_tracks = max(1, int(max_orbit_tracks))
    rng = np.random.default_rng(2813)
    choose = rng.choice(len(samples), size=min(len(samples), max_orbit_tracks), replace=False)
    phase = np.linspace(0.0, 1.0, 361)
    all_n, all_e = _complete_orbit_tracks(samples[choose], phase)
    for east_curve, north_curve in zip(all_e, all_n):
        ax.plot(east_curve, north_curve, lw=0.45, color="#367a9c",
                alpha=0.045, zorder=1, rasterized=True)

    n_arc, e_arc = _tracks(samples[ix_best:ix_best+1], arc_times, t_ref)
    ax.plot(e_arc[0], n_arc[0], color="#142c4c", lw=2.3,
            label="Highest-scoring observed arc", zorder=5)
    colors = ("#006fba", "#d65b00", "#0b826c", "#8c52a1")
    for j, inst in enumerate(instruments):
        color = colors[j % len(colors)]
        n, e = _tracks(samples[ix_best:ix_best+1], inst.ts, t_ref)
        for k, (east_k, north_k) in enumerate(zip(e[0], n[0]), 1):
            _map_position(ax, east_k, north_k, scale=float(inst.scale),
                          color=color, radius_px=1.3, label=False)
            ax.annotate(str(k), (east_k, north_k), xytext=(8, 8),
                        textcoords="offset points", fontsize=8, color=color)
        ax.plot([], [], color=color, linewidth=2.5, label=f"{inst.name} epochs")
        if inst.r_mask is not None:
            inner = float(inst.r_mask) / float(inst.scale)
            ax.add_patch(plt.Circle((0, 0), inner, fill=False, ls=":", lw=0.8,
                                    color=color, alpha=0.50))
    # Symbol size is for legibility, not the physical radius of the star.
    ax.add_patch(plt.Circle((0, 0), 2.5 / max(i.scale for i in instruments),
                            facecolor="#ecb149", edgecolor="#172e44", lw=0.9,
                            zorder=6, label="Host star (symbol)"))
    ax.set(xlabel="East offset [AU]", ylabel="North offset [AU]",
           title="Posterior Keplerian orbits and observing epochs")
    ax.invert_xaxis()
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.12, linewidth=0.5)
    ax.legend(loc="best", fontsize=8, frameon=False, ncol=2)
    _save(fig, output_dir / "projected_orbits", dpi)


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
                          color="#ffdf5b", radius_px=1.2, label=True)
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
    """Render a publication-ready lower-triangle posterior projection."""
    plt = _style()
    columns = np.asarray(samples, float)
    ndim = len(names)
    fig, axes = plt.subplots(ndim, ndim, figsize=(1.55*ndim, 1.55*ndim),
                             constrained_layout=True)
    for row in range(ndim):
        for col in range(ndim):
            ax = axes[row, col]
            if row < col:
                ax.axis("off")
                continue
            if row == col:
                ax.hist(columns[:, col], bins=36, color="#3e7895",
                        histtype="stepfilled", alpha=0.75)
            else:
                ax.scatter(columns[:, col], columns[:, row], s=1,
                           rasterized=True, color="#31637b", alpha=0.09)
            ax.tick_params(axis="both", labelsize=6, length=2)
            if row == ndim-1:
                ax.set_xlabel(names[col], fontsize=8)
            else:
                ax.tick_params(labelbottom=False)
            if row > col and col == 0:
                ax.set_ylabel(names[row], fontsize=8)
            elif col > 0:
                ax.tick_params(labelleft=False)
    fig.suptitle(title, fontsize=13)
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
                              mask_to_working_region: bool=True) -> None:
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

class _StoredChainSampler:
    """Read saved cold-chain arrays using the interface expected by diagnostics."""

    def __init__(self, h5_file):
        self.chain = np.asarray(h5_file["chain"], dtype=float)
        self.logp = np.asarray(h5_file["log_prob"], dtype=float)
        self.acceptance_fraction = np.asarray(h5_file["acceptance_fraction"], dtype=float)
        if "pt_betas_final" in h5_file:
            self.pt_sampler = _StoredTemperingDiagnostics(h5_file)

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
    figure_dir = Path(params.get_path("values_dir")) / str(plots_settings.get("directory", "figures"))
    if plots_settings.get("enabled", True):
        create_scientific_figures(
            instruments, samples, scores, output_dir=figure_dir / "science",
            noise_floor=noise_floor,
            max_posterior_samples=int(plots_settings.get("max_posterior_samples", 12000)),
            max_orbit_tracks=int(plots_settings.get("max_orbit_tracks", 900)),
            dpi=int(plots_settings.get("dpi", 200)),
            mask_to_working_region=bool(plots_settings.get("mask_to_working_region", True)),
        )
        create_mcmc_diagnostics(
            chain, output_dir=figure_dir / "convergence",
            dpi=int(plots_settings.get("dpi", 200)),
            parameter_names=parameter_names,
        )
