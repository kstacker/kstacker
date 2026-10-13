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
        "font.family": "DejaVu Sans", "font.size": 10,
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


def plot_cold_chain_trace(chain: np.ndarray, output_dir: Path, dpi: int, parameter_names=None):
    """Visualize individual walkers and across-walker median for each parameter."""
    plt=_plotter(); steps,nwalkers,ndim=chain.shape
    fig, axes=plt.subplots(ndim,1,figsize=(10,1.65*ndim),sharex=True,
                          constrained_layout=True)
    colors=plt.get_cmap("tab10")
    stride=max(1,int(np.ceil(steps/16000)))
    ix=np.arange(0,steps,stride)
    for j,ax in enumerate(np.atleast_1d(axes)):
        for walker in range(nwalkers):
            ax.plot(ix,chain[ix,walker,j],color=colors(j%10),lw=.28,
                    alpha=min(.22,8.0/nwalkers),rasterized=True)
        ax.plot(ix,np.nanmedian(chain[ix,:,j],axis=1),color="#21334a",lw=1,
                label="Walker median" if j==0 else None)
        ax.set_ylabel(_labels(ndim,parameter_names)[j],fontsize=9)
        ax.grid(alpha=.12)
    axes[-1].set_xlabel("Production iteration")
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle("Cold-chain parameter traces — mixing and stationarity")
    _save(fig,output_dir/"parameter_traces",dpi)


def plot_sampler_overview(chain, log_probability, acceptance, output_dir, dpi):
    """Show the log posterior of every walker after burn-in.

    Every walker is retained. For very long runs the time axis is sampled
    uniformly to limit plot size; this display simplification NEVER modifies
    stored samples, statistics, or posterior inference.
    """
    plt = _plotter()
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.3), constrained_layout=True)
    steps, nwalkers = log_probability.shape
    stride = max(1, int(np.ceil(steps / 16000)))
    ix = np.arange(0, steps, stride)
    timeline = np.asarray(log_probability[ix], dtype=float)
    for walker in range(nwalkers):
        axs[0, 0].plot(ix, timeline[:, walker], color="#226ca2", lw=.32,
                        alpha=min(.32, 9.0 / nwalkers), rasterized=True)
    median = np.nanmedian(timeline, axis=1)
    axs[0, 0].plot(ix, median, color="#d44d43", lw=1.2, label="Walker median")
    axs[0, 0].legend(fontsize=8, frameon=False)
    axs[0, 0].set(xlabel="Production step (burn-in excluded)", ylabel="Log posterior",
                  title=f"All {nwalkers} walker log-posterior traces")
    axs[0, 0].grid(alpha=.13)

    finite = _finite_scale(acceptance)
    if finite.size:
        axs[0, 1].hist(finite, bins=min(24, max(7, int(np.sqrt(finite.size)))),
                        color="#26a69a", edgecolor="white")
    axs[0, 1].set(xlabel="Acceptance fraction", ylabel="Walkers",
                  title="Within-walker acceptance")

    # Descriptive ACF of the median is NOT the integrated autocorrelation
    # of individual chains and does not provide an ESS estimate.
    series = np.nanmedian(np.asarray(chain[:, :, 0], float), axis=1)
    if len(series) > 12 and np.isfinite(series).all():
        series = series - np.mean(series)
        denom = np.dot(series, series)
        lags = min(250, max(5, len(series) // 5))
        if denom > 0:
            acf = [float(np.dot(series[:len(series)-lag], series[lag:]) / denom)
                   for lag in range(lags)]
            axs[1, 0].plot(np.arange(lags), acf, color="#8955b5", lw=1.1)
    axs[1, 0].axhline(0, color="grey", lw=.7)
    axs[1, 0].set(xlabel="Lag [steps]", ylabel="Autocorrelation",
                  title="Median-trace ACF (descriptive only)")
    axs[1, 1].axis("off")
    mean_accept = float(np.mean(finite)) if finite.size else float("nan")
    axs[1, 1].text(.04, .94, f"Production steps: {steps:,}\n"
                   f"All walkers: {nwalkers}\n"
                   f"Parameter dimensions: {chain.shape[-1]}\n"
                   f"Mean acceptance: {mean_accept:.3f}\n"
                   f"Plot time stride: {stride}\n\n"
                   "Burn-in samples are not present.\n"
                   "ACF of median is not ESS.", va="top", fontsize=11,
                   color="#23374f")
    fig.suptitle("MCMC production diagnostics")
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
    if hasattr(sampler,"pt_sampler"):
        plot_apt_diagnostics(sampler.pt_sampler,output_dir,dpi)
