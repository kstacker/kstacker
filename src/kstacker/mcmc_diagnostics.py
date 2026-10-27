"""Sampler convergence checks kept separate from physical orbital figures.

The plots help evaluate stationarity, mixing and parallel-temperature exchange.
No individual diagnostic constitutes proof of convergence.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np


def _plotter():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "savefig.facecolor": "white",
    })
    return plt


def _save(fig, path: Path, dpi: int):
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=dpi)
    _plotter().close(fig)


def _labels(dim: int, parameter_names=None) -> list[str]:
    """Label every sampled dimension, including detector-specific amplitudes."""
    if parameter_names is not None:
        if len(parameter_names) != dim:
            raise ValueError("parameter_names must match chain dimension")
        return list(parameter_names)
    return ["a [AU]", "lambda_ref [rad]", "stellar mass [M_sun]", "h", "k", "p", "q"] + [
        f"Flux {index + 1}" for index in range(dim - 7)
    ]


def _finite_scale(array):
    good=np.asarray(array,float)
    return good[np.isfinite(good)]


def plot_cold_chain_trace(chain: np.ndarray, output_dir: Path, dpi: int,
                          parameter_names=None):
    """Plot every post-burn-in state from every walker in each parameter.

    A rasterized polyline per walker keeps PDF outputs manageable. No
    decimation of time or walkers is applied, and stored samples are untouched.
    Trace proximity alone does not demonstrate posterior convergence.
    """
    plt = _plotter()
    steps, nwalkers, ndim = chain.shape
    if not steps or not nwalkers:
        return
    names = _labels(ndim, parameter_names)
    fig, axes = plt.subplots(ndim, 1, figsize=(12.5, 1.95 * ndim),
                             sharex=True, constrained_layout=True)
    colors = plt.get_cmap("tab10")
    x = np.arange(steps)
    for j, ax in enumerate(np.atleast_1d(axes)):
        # Every production iteration and walker is plotted (no thinning).
        for walker in range(nwalkers):
            ax.plot(x, chain[:, walker, j], color=colors(j % 10),
                    lw=0.30, alpha=min(0.22, 7.0 / nwalkers),
                    rasterized=True)
        ax.plot(x, np.nanmedian(chain[:, :, j], axis=1), color="#183a55",
                lw=0.95, label="Walker median" if j == 0 else None,
                rasterized=True)
        ax.set_ylabel(names[j], fontsize=10)
        ax.tick_params(labelsize=8)
    axes[-1].set_xlabel("Production iteration (burn-in excluded)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"Orbital parameter traces — all {nwalkers} walkers",
                 fontsize=14)
    _save(fig, output_dir / "parameter_traces", dpi)


def plot_sampler_overview(chain, log_probability, acceptance, output_dir, dpi):
    """Display every walker and production step of the log posterior.

    Top-left shows all complete walker trajectories; bottom-left includes an
    exact time-by-walker heatmap. The heatmap colour range is percentile-based
    for visual contrast but never excludes any cell; extreme values saturate.
    The orange median curve is descriptive, not a convergence criterion.
    """
    plt = _plotter()
    logp = np.asarray(log_probability, dtype=float)
    steps, nwalkers = logp.shape
    fig, axs = plt.subplots(2, 2, figsize=(14, 9),
                            constrained_layout=True,
                            gridspec_kw={"width_ratios": [2.1, 1.0]})
    x = np.arange(steps)
    for walker in range(nwalkers):
        axs[0, 0].plot(x, logp[:, walker], color="#237aab", lw=0.32,
                       alpha=min(0.30, 8.0 / nwalkers), rasterized=True)
    median = np.nanmedian(logp, axis=1)
    axs[0, 0].plot(x, median, color="#eb743a", lw=1.15,
                   label="Walker median", rasterized=True)
    axs[0, 0].set(xlabel="Production iteration", ylabel="Log posterior",
                  title=f"Full log-posterior traces ({nwalkers} walkers)")
    axs[0, 0].legend(fontsize=9, frameon=False)

    # Every matrix element corresponds to an actual walker at an actual step.
    finite = logp[np.isfinite(logp)]
    if finite.size:
        vmin, vmax = np.percentile(finite, [2, 98])
        if vmax <= vmin:
            vmin, vmax = float(np.min(finite)), float(np.max(finite))
        if vmax <= vmin:
            vmin -= 0.5
            vmax += 0.5
    else:
        vmin, vmax = -1, 1
    im = axs[1, 0].imshow(logp.T, origin="lower", aspect="auto",
                          interpolation="nearest", cmap="viridis",
                          extent=(-0.5, steps - 0.5, -0.5, nwalkers - 0.5),
                          vmin=vmin, vmax=vmax, rasterized=True)
    fig.colorbar(im, ax=axs[1, 0], fraction=0.03, pad=0.015,
                 label="Log posterior (2–98% contrast)")
    axs[1, 0].set(xlabel="Production iteration", ylabel="Walker index",
                  title="All walker states (no thinning)")
    acceptance = np.asarray(acceptance, dtype=float)
    accepted = acceptance[np.isfinite(acceptance)]
    if accepted.size:
        axs[0, 1].hist(accepted, bins=min(28, max(8, int(np.sqrt(len(accepted))))),
                       color="#42a48b", edgecolor="white")
    axs[0, 1].set(xlabel="Acceptance fraction", ylabel="Walkers",
                  title="Acceptance across all walkers")

    # Separate within-walker and between-walker variations: prolonged
    # horizontal bands in the heatmap can indicate walkers trapped in modes.
    ranges = np.full(nwalkers, np.nan, dtype=float)
    for walker in range(nwalkers):
        trajectory = logp[:, walker]
        good = trajectory[np.isfinite(trajectory)]
        if good.size > 1:
            q10, q90 = np.quantile(good, [0.1, 0.9])
            ranges[walker] = q90 - q10
    axs[1, 1].bar(np.arange(nwalkers), ranges, color="#7464b5", width=0.9)
    axs[1, 1].set(xlabel="Walker index", ylabel="90% − 10% log-posterior range",
                  title="Within-walker variation")
    if accepted.size:
        axs[0, 1].text(0.96, 0.94, f"Median: {np.median(accepted):.3f}",
                       ha="right", va="top", transform=axs[0, 1].transAxes,
                       fontsize=9, color="#284659")
    for ax in axs.flat:
        ax.tick_params(labelsize=9)
    fig.suptitle("MCMC production diagnostics — complete walker histories",
                 fontsize=14)
    _save(fig, output_dir / "sampling_diagnostics", dpi)


def plot_apt_diagnostics(pt_sampler, output_dir: Path, dpi: int):
    """Plot adaptive-tempering temperature coverage, swaps and hot-chain mobility.

    `reddemcee` exposes diagnostics via its sampler-specific methods. When a
    specific history is not stored by that installation, the figure omits it
    rather than inventing exchange or adaptation rates.
    """
    plt=_plotter()
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    betas=np.asarray(pt_sampler.betas,float)
    axes[0,0].plot(np.arange(len(betas)),betas,marker="o",color="#307f9e")
    axes[0,0].set(xlabel="Temperature index",ylabel="Inverse temperature (beta)",
                  title="Final temperature ladder")
    axes[0,0].set_yscale("log")
    # Full temperature chains are optional, to avoid holding a second large copy
    # during ordinary plotting runs.
    try:
        chain=np.asarray(pt_sampler.get_chain(flat=False),float)
        if chain.ndim==4:
            subset=np.arange(0,chain.shape[1],max(1,chain.shape[1]//800))
            for t in range(chain.shape[0]):
                trace=np.nanmedian(chain[t,subset,:,0],axis=1)
                axes[0,1].plot(subset,trace,lw=.6,alpha=.8,
                               label=f"beta={betas[t]:.2g}")
            if chain.shape[0]<=16:axes[0,1].legend(fontsize=7,frameon=False,ncol=2)
    except (AttributeError,ValueError,TypeError) as error:
        axes[0,1].text(.1,.5,"Temperature traces unavailable",transform=axes[0,1].transAxes)
    axes[0,1].set(xlabel="Sweep",ylabel="Median a [AU]",title="Mobility across temperatures")

    for ax,method,title,color in [
        (axes[1,0],"get_tsw","Temperature swap history","#d08b3f"),
        (axes[1,1],"get_smd","Adaptation / SMD history","#8b69a6"),
    ]:
        try:
            values=np.asarray(getattr(pt_sampler,method)(),float)
            if values.size==0:raise ValueError("empty history")
            if values.ndim==1:
                ax.plot(values,lw=.9,color=color)
            else:
                # Plot each channel's mean across leading stored axes, keeping
                # the sweep-like trailing axis without assuming a fixed schema.
                flattened=values.reshape(-1,values.shape[-1])
                for line in flattened[:18]:ax.plot(line,lw=.6,color=color,alpha=.5)
                ax.plot(np.nanmedian(flattened,axis=0),lw=1.5,color="#263747")
        except (AttributeError,ValueError,TypeError):
            ax.text(.1,.5,"History not available",transform=ax.transAxes)
        ax.set(xlabel="Stored history index",ylabel="Sampler statistic",title=title)
    fig.suptitle("Adaptive parallel tempering — temperature exploration")
    _save(fig,output_dir/"temperature_diagnostics",dpi)



def _autocorrelation_1d(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Estimate the normalized autocorrelation function of a 1-D series.

    The estimator uses FFT-based convolution after mean subtraction. The
    returned sequence starts at lag zero and is normalized by the zero-lag
    variance. Invalid or constant series yield NaNs except at lag zero.
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.full(max_lag + 1, np.nan, dtype=float)
    x = x - np.mean(x)
    var = np.dot(x, x)
    if not np.isfinite(var) or var <= 0:
        acf = np.full(max_lag + 1, np.nan, dtype=float)
        acf[0] = 1.0
        return acf
    n = x.size
    nfft = 1 << int(np.ceil(np.log2(2 * n - 1)))
    fx = np.fft.rfft(x, n=nfft)
    acov = np.fft.irfft(fx * np.conjugate(fx), n=nfft)[:n]
    acov = acov / np.arange(n, 0, -1)
    acf = acov[:max_lag + 1] / acov[0]
    if acf.size < max_lag + 1:
        acf = np.pad(acf, (0, max_lag + 1 - acf.size), constant_values=np.nan)
    return acf


def plot_autocorrelation_summary(chain: np.ndarray, output_dir: Path, dpi: int,
                                 parameter_names=None, max_lag: int | None = None) -> None:
    """Summarize walker-wise autocorrelation functions for every parameter.

    One panel is produced per sampled parameter. For each lag, the thick line is
    the median ACF over walkers and the shaded region spans the 16th to 84th
    percentile range across walkers. This keeps the figure readable while still
    showing between-walker differences in mixing behaviour.
    """
    plt = _plotter()
    steps, nwalkers, ndim = chain.shape
    if steps < 3 or nwalkers < 1:
        return
    names = _labels(ndim, parameter_names)
    if max_lag is None:
        max_lag = min(max(20, steps // 5), 400)
    lags = np.arange(max_lag + 1)
    fig, axes = plt.subplots(ndim, 1, figsize=(11.5, 2.2 * ndim),
                             sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for j, ax in enumerate(axes):
        walker_acf = []
        for walker in range(nwalkers):
            walker_acf.append(_autocorrelation_1d(chain[:, walker, j], max_lag))
        walker_acf = np.asarray(walker_acf, dtype=float)
        q16 = np.nanpercentile(walker_acf, 16, axis=0)
        q50 = np.nanpercentile(walker_acf, 50, axis=0)
        q84 = np.nanpercentile(walker_acf, 84, axis=0)
        ax.fill_between(lags, q16, q84, color="#a6cee3", alpha=0.45, linewidth=0.0)
        ax.plot(lags, q50, color="#1f78b4", lw=1.8)
        ax.axhline(0.0, color="#6b7280", lw=0.8)
        ax.set_ylabel(names[j])
        ax.set_ylim(-0.2, 1.05)
        ax.tick_params(labelsize=10)
    axes[-1].set_xlabel("Lag [saved production steps]")
    fig.suptitle("Autocorrelation functions", fontsize=15)
    _save(fig, output_dir / "autocorrelation_functions", dpi)

def create_mcmc_diagnostics(sampler, *, output_dir, thin: int=1, dpi: int=200,
                            parameter_names=None) -> None:
    """Save convergence-oriented plots for either emcee or adaptive PT."""
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    chain=np.asarray(sampler.get_chain(flat=False),float)
    logp=np.asarray(sampler.get_log_prob(flat=False),float)
    if chain.ndim!=3 or logp.shape!=chain.shape[:2]:
        raise ValueError("Sampler must expose aligned (step, walker, parameter) arrays")
    print(f"[plots] convergence diagnostics -> {output_dir}",flush=True)
    if chain.size==0:return
    plot_cold_chain_trace(chain,output_dir,dpi,parameter_names)
    plot_sampler_overview(chain,logp,np.asarray(sampler.acceptance_fraction,float),output_dir,dpi)
    plot_autocorrelation_summary(chain, output_dir, dpi, parameter_names)
    if hasattr(sampler,"pt_sampler"):
        plot_apt_diagnostics(sampler.pt_sampler,output_dir,dpi)
