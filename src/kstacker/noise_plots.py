"""Diagnostic PDF figures of the per-instrument noise products.

These figures are written by ``kstacker init`` from the cached files only:

  - local aperture-ring noise maps      (profile_dir/<instrument>/local_aperture_ring__...__noise.npy)
  - local aperture-ring background maps (profile_dir/<instrument>/local_aperture_ring__...__background.npy)
  - radially recalibrated noise maps    (.../radial_noise_recalibration/.../noise_recalibrated_epoch_k.npy)
  - radial recalibration profile g(d)   (.../g_profile.npz)

They never modify the arrays used by the likelihood (files are re-read from
disk) and do not use any random generator.  Output folder:
``values/figures/noise/<instrument>/``.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# Maximum number of per-epoch g(d) curves drawn on a single figure.
MAX_EPOCH_CURVES_PER_FIGURE = 8


def _label(name: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_.")
    return label or "instrument"


def _stats_text(values: np.ndarray) -> str:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return "no valid pixel"
    return (f"min {np.min(finite):.3g} / median {np.median(finite):.3g} / "
            f"max {np.max(finite):.3g}")


def _map_pages(pdf_path: Path, maps: Sequence[np.ndarray],
               invalid_masks: Sequence[np.ndarray], *, instrument: str,
               kind: str, unit: str, r_mask: Optional[float],
               r_mask_ext: Optional[float], diverging: bool) -> None:
    """One page per epoch; invalid pixels are drawn in magenta."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        for k, (data, invalid) in enumerate(zip(maps, invalid_masks)):
            data = np.array(data, dtype=float, copy=True)
            invalid = np.asarray(invalid, dtype=bool) | ~np.isfinite(data)
            shown = np.ma.array(data, mask=invalid)
            valid_values = data[~invalid]
            fig, ax = plt.subplots(figsize=(6.6, 5.8), constrained_layout=True)
            cmap = plt.get_cmap("RdBu_r" if diverging else "viridis").with_extremes(bad="magenta")
            if valid_values.size:
                if diverging:
                    amp = float(np.percentile(np.abs(valid_values), 99.5)) or 1.0
                    vmin, vmax = -amp, amp
                else:
                    vmin, vmax = np.percentile(valid_values, [0.5, 99.5])
                    if not vmax > vmin:
                        vmin, vmax = float(np.min(valid_values)), float(np.max(valid_values)) + 1e-30
            else:
                vmin, vmax = 0.0, 1.0
            image = ax.imshow(shown, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                              interpolation="nearest")
            fig.colorbar(image, ax=ax, extend="both",
                         label=f"{kind} [{unit}] (colour range: 0.5-99.5 %)")
            n = data.shape[0]
            centre = ((data.shape[1] - 1) / 2.0, (n - 1) / 2.0)
            for radius, style, name in ((r_mask, "--", "r_mask"),
                                        (r_mask_ext, ":", "r_mask_ext")):
                if radius is not None:
                    ax.add_patch(plt.Circle(centre, float(radius), fill=False,
                                            color="white", ls=style, lw=1.2))
                    ax.plot([], [], color="black", ls=style, label=f"{name} = {float(radius):g} px")
            yy, xx = np.mgrid[:n, :data.shape[1]]
            rr = np.hypot(xx - centre[0], yy - centre[1])
            work = np.ones_like(rr, dtype=bool)
            if r_mask is not None:
                work &= rr > float(r_mask)
            if r_mask_ext is not None:
                work &= rr < float(r_mask_ext)
            ax.plot([], [], marker="s", ls="none", color="magenta",
                    label=f"invalid (sigma = 0 or NaN): {int(np.count_nonzero(invalid))} px, "
                          f"{int(np.count_nonzero(invalid & work))} of {int(np.count_nonzero(work))} "
                          f"in r_mask < r < r_mask_ext")
            ax.legend(loc="upper right", fontsize=7, framealpha=0.8)
            ax.set_xlabel("x [native px]")
            ax.set_ylabel("y [native px]")
            ax.set_title(f"{instrument} | epoch {k} | {kind}\n{_stats_text(valid_values)}",
                         fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)


def _g_profile_figures(npz_path: Path, output_dir: Path, *, instrument: str,
                       label: str) -> list[Path]:
    """g(d) curves, at most MAX_EPOCH_CURVES_PER_FIGURE epochs per PDF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(npz_path) as data:
        d = np.array(data["distance_centers_px"], dtype=float)
        per_epoch = np.array(data["per_epoch"], dtype=float)
        g_raw = np.array(data["g_raw"], dtype=float)
        g = np.array(data["g"], dtype=float)
        inner = float(data["inner_radius_px"])
        dmax = float(data["max_distance_px"]) if "max_distance_px" in data.files else float("nan")

    n_epochs = per_epoch.shape[0]
    groups = [list(range(i, min(i + MAX_EPOCH_CURVES_PER_FIGURE, n_epochs)))
              for i in range(0, max(n_epochs, 1), MAX_EPOCH_CURVES_PER_FIGURE)]
    written = []
    for part, epochs in enumerate(groups, start=1):
        fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
        cmap = plt.get_cmap("tab10")
        for j, k in enumerate(epochs):
            ax.plot(d, per_epoch[k], marker="o", ms=3, lw=1.0, color=cmap(j % 10),
                    alpha=0.85, label=f"epoch {k}")
        ax.plot(d, g_raw, color="black", lw=2.6, ls="--", label="g_raw (median of epochs)")
        ax.plot(d, g, color="crimson", lw=2.8, label="g (applied)")
        ax.axhline(1.0, color="grey", lw=0.8, ls=":")
        ax.set_xlabel("d = r - r_mask [native px]")
        ax.set_ylabel("robust scatter of z  (g)")
        suffix = "" if len(groups) == 1 else f" (part {part}/{len(groups)})"
        ax.set_title(f"{instrument} | radial noise recalibration g(d){suffix}\n"
                     f"r_mask (inner_radius_px) = {inner:g} px, max_distance_px = {dmax:g} px",
                     fontsize=10)
        ax.legend(fontsize=7, ncol=2, framealpha=0.8)
        name = f"{label}_g_profile.pdf" if part == 1 else f"{label}_g_profile_part{part}.pdf"
        path = output_dir / name
        fig.savefig(path)
        plt.close(fig)
        written.append(path)
    return written


def plot_instrument_noise_products(inst, output_dir: Path) -> list[Path]:
    """Write the noise diagnostic PDFs of one instrument; return written paths."""
    products = getattr(inst, "noise_products", None) or {}
    if not products:
        return []
    label = _label(inst.name)
    out = Path(output_dir) / label
    out.mkdir(parents=True, exist_ok=True)
    n_epochs = len(inst.ts)
    written: list[Path] = []

    ring_dir = products.get("local_ring_dir")
    stem = products.get("local_ring_stem")
    noise_maps, bg_maps = [], []
    for k in range(n_epochs):
        base = os.path.join(ring_dir, f"{stem}__epoch={k:03d}")
        noise_maps.append(np.load(base + "__noise.npy"))
        bg_maps.append(np.load(base + "__background.npy"))
    invalid = [~(np.isfinite(m) & (m > 0)) for m in noise_maps]
    unit = "convolved image units" if inst.photometry_method == "convolve" else "aperture-sum units"
    common = dict(instrument=inst.name, r_mask=inst.r_mask, r_mask_ext=inst.r_mask_ext)

    path = out / f"{label}_local_noise.pdf"
    _map_pages(path, noise_maps, invalid, kind="local noise sigma_local", unit=unit,
               diverging=False, **common)
    written.append(path)
    path = out / f"{label}_local_background.pdf"
    _map_pages(path, bg_maps, invalid, kind="local background", unit=unit,
               diverging=True, **common)
    written.append(path)

    recal_dir = products.get("recalibration_dir")
    if recal_dir is not None:
        recal_paths = [os.path.join(recal_dir, f"noise_recalibrated_epoch_{k}.npy")
                       for k in range(n_epochs)]
        if all(os.path.isfile(p) for p in recal_paths):
            recal = [np.load(p) for p in recal_paths]
            path = out / f"{label}_recalibrated_noise.pdf"
            _map_pages(path, recal, [~(np.isfinite(m) & (m > 0)) for m in recal],
                       kind="recalibrated noise sigma_cal", unit=unit,
                       diverging=False, **common)
            written.append(path)
        else:
            print(f"[noise-plots] {inst.name}: recalibrated maps not cached in {recal_dir}; skipped.")
        npz = Path(recal_dir) / "g_profile.npz"
        if npz.is_file():
            written.extend(_g_profile_figures(npz, out, instrument=inst.name, label=label))
        else:
            print(f"[noise-plots] {inst.name}: {npz} not found; g(d) figure skipped.")
    return written


def create_noise_figures(instruments: Sequence, output_dir) -> list[Path]:
    """Write noise diagnostics for every instrument; failures never propagate."""
    written: list[Path] = []
    for inst in instruments:
        try:
            paths = plot_instrument_noise_products(inst, Path(output_dir))
            written.extend(paths)
            if paths:
                print(f"[noise-plots] {inst.name}: {len(paths)} PDF -> {Path(output_dir) / _label(inst.name)}")
        except Exception as exc:  # diagnostics must never stop the computation
            print(f"[noise-plots] WARNING: {inst.name}: noise figures failed: {exc}")
    return written
