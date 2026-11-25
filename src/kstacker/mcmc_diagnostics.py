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


def _step_axis_label(chain_stride: int=1, lines_per_sweep=None) -> str:
    """Name the x unit of plotted chain rows.

    Rows are multiplied by `chain_stride` before plotting, so the axis is in
    production steps (emcee) or production lines (reddemcee inner steps) even
    when only one line in `chain_stride` is stored.
    """
    if lines_per_sweep is None:
        label = "Production step (burn-in excluded)"
    else:
        label = (f"Production line = inner step (1 sweep = {int(lines_per_sweep)} lines, "
                 "burn-in excluded)")
    if int(chain_stride) > 1:
        label += f"\n1 stored row every {int(chain_stride)} lines"
    return label


def _nanpercentile_columns(values: np.ndarray, q) -> np.ndarray:
    """Column-wise `np.nanpercentile` returning NaN for all-NaN columns.

    Avoids the "All-NaN slice" RuntimeWarning (e.g. lags longer than a very
    short stored chain, or constant series) without silencing other warnings.
    """
    values = np.asarray(values, dtype=float)
    out = np.full((len(np.atleast_1d(q)),) + values.shape[1:], np.nan, dtype=float)
    good = np.any(np.isfinite(values), axis=0)
    if np.any(good):
        out[:, good] = np.nanpercentile(values[:, good], np.atleast_1d(q), axis=0)
    return out if np.ndim(q) else out[0]


def _finite_scale(array):
    good=np.asarray(array,float)
    return good[np.isfinite(good)]


def plot_cold_chain_trace(chain: np.ndarray, output_dir: Path, dpi: int,
                          parameter_names=None, chain_stride: int=1,
                          lines_per_sweep=None):
    """Plot every stored post-burn-in state from every walker in each parameter.

    A rasterized polyline per walker keeps PDF outputs manageable. No further
    decimation of stored rows or walkers is applied, and stored samples are
    untouched. Rows are placed at their production step / line (row index
    times `chain_stride`). Trace proximity alone does not demonstrate
    posterior convergence.
    """
    plt = _plotter()
    steps, nwalkers, ndim = chain.shape
    if not steps or not nwalkers:
        return
    names = _labels(ndim, parameter_names)
    fig, axes = plt.subplots(ndim, 1, figsize=(12.5, 1.95 * ndim),
                             sharex=True, constrained_layout=True)
    colors = plt.get_cmap("tab10")
    x = np.arange(steps) * max(1, int(chain_stride))
    for j, ax in enumerate(np.atleast_1d(axes)):
        # Every stored row and walker is plotted (no further decimation).
        for walker in range(nwalkers):
            ax.plot(x, chain[:, walker, j], color=colors(j % 10),
                    lw=0.30, alpha=min(0.22, 7.0 / nwalkers),
                    rasterized=True)
        ax.plot(x, np.nanmedian(chain[:, :, j], axis=1), color="#183a55",
                lw=0.95, label="Walker median" if j == 0 else None,
                rasterized=True)
        ax.set_ylabel(names[j], fontsize=10)
        ax.tick_params(labelsize=8)
    axes[-1].set_xlabel(_step_axis_label(chain_stride, lines_per_sweep))
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"Orbital parameter traces — all {nwalkers} walkers",
                 fontsize=14)
    _save(fig, output_dir / "parameter_traces", dpi)


def plot_sampler_overview(chain, log_probability, acceptance, output_dir, dpi,
                          chain_stride: int=1, lines_per_sweep=None):
    """Display every walker and stored production row of the log posterior.

    Top-left shows all stored walker trajectories; bottom-left includes an
    exact stored-row-by-walker heatmap (rows placed at row x `chain_stride`). The heatmap colour range is percentile-based
    for visual contrast but never excludes any cell; extreme values saturate.
    The orange median curve is descriptive, not a convergence criterion.
    """
    plt = _plotter()
    logp = np.asarray(log_probability, dtype=float)
    steps, nwalkers = logp.shape
    fig, axs = plt.subplots(2, 2, figsize=(14, 9),
                            constrained_layout=True,
                            gridspec_kw={"width_ratios": [2.1, 1.0]})
    stride = max(1, int(chain_stride))
    xlabel = _step_axis_label(stride, lines_per_sweep)
    x = np.arange(steps) * stride
    for walker in range(nwalkers):
        axs[0, 0].plot(x, logp[:, walker], color="#237aab", lw=0.32,
                       alpha=min(0.30, 8.0 / nwalkers), rasterized=True)
    median = np.nanmedian(logp, axis=1)
    axs[0, 0].plot(x, median, color="#eb743a", lw=1.15,
                   label="Walker median", rasterized=True)
    axs[0, 0].set(xlabel=xlabel, ylabel="Log posterior",
                  title=f"Log-posterior traces, all stored rows ({nwalkers} walkers)")
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
                          extent=(-0.5 * stride, (steps - 0.5) * stride,
                                  -0.5, nwalkers - 0.5),
                          vmin=vmin, vmax=vmax, rasterized=True)
    fig.colorbar(im, ax=axs[1, 0], fraction=0.03, pad=0.015,
                 label="Log posterior (2–98% contrast)")
    axs[1, 0].set(xlabel=xlabel, ylabel="Walker index",
                  title=("All stored walker states" if stride == 1 else
                         f"All stored walker states (1 row every {stride})"))
    acceptance = np.asarray(acceptance, dtype=float)
    accepted = acceptance[np.isfinite(acceptance)]
    if accepted.size:
        axs[0, 1].hist(accepted, bins=min(28, max(8, int(np.sqrt(len(accepted))))),
                       color="#42a48b", edgecolor="white")
    axs[0, 1].set(xlabel="Acceptance fraction (every production step)",
                  ylabel="Walkers", title="Acceptance across all walkers")

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
    fig.suptitle("MCMC production diagnostics — stored walker histories",
                 fontsize=14)
    _save(fig, output_dir / "sampling_diagnostics", dpi)


def plot_apt_diagnostics(pt_sampler, output_dir: Path, dpi: int,
                         burnin_sweeps: int=0, lines_per_sweep: int=1,
                         chain_burnin_rows=None, chain_stride: int=1,
                         cold_chain=None):
    """Plot adaptive-tempering temperature coverage, swaps and hot-chain mobility.

    `reddemcee` exposes diagnostics via its sampler-specific methods. When a
    specific history is not stored by that installation, the figure omits it
    rather than inventing exchange or adaptation rates. Histories are restricted
    to production: chains hold `lines_per_sweep` lines per sweep, the swap and
    adaptation histories one row per sweep.  When only one line in
    `chain_stride` is stored (reddemcee, iteration 08), `chain_burnin_rows`
    stored rows belong to the burn-in.

    Upper-right panel: walker median of the first sampled coordinate, the
    semi-major axis a in AU (sampled linearly, uniform prior), per stored
    production row at each temperature, placed at production line = row x
    `chain_stride`.  The tempered chains (beta < 1) are only available while
    `kstacker mcmc` runs or when `outputs.save_tempered_chains` is true;
    otherwise the cold chain `cold_chain` (beta = 1, production rows) is
    shown with its 16-84 % walker range.  Lower panels: per adjacent
    temperature pair (cold -> hot), the distribution over production sweeps
    of the swap acceptance and of reddemcee's swap mean distance.
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
    stride=max(1,int(chain_stride))
    xlabel=_step_axis_label(stride,lines_per_sweep)
    ylabel="Semi-major axis a [AU]\n(median over walkers)"
    try:
        chain=np.asarray(pt_sampler.get_chain(flat=False),float)
        if chain.ndim!=4:
            raise ValueError("tempered chains must be (temperature, row, walker, parameter)")
        cut=(int(burnin_sweeps)*int(lines_per_sweep) if chain_burnin_rows is None
             else int(chain_burnin_rows))
        chain=chain[:,cut:]
        if chain.shape[1]==0:
            raise ValueError("no production rows in the tempered chains")
        subset=np.arange(0,chain.shape[1],max(1,chain.shape[1]//800))
        for t in range(chain.shape[0]):
            trace=np.nanmedian(chain[t,subset,:,0],axis=1)
            axes[0,1].plot(subset*stride,trace,lw=.6,alpha=.8,
                           label=f"beta={betas[t]:.2g}")
        if chain.shape[0]<=16:
            axes[0,1].legend(fontsize=7,frameon=False,ncol=2,title="final beta",
                             title_fontsize=7)
        title="Walker median of a at each temperature"
    except (AttributeError,ValueError,TypeError,IndexError):
        title="Walker median of a, cold chain only"
        cold=None if cold_chain is None else np.asarray(cold_chain,float)
        if cold is not None and cold.ndim==3 and cold.shape[0] and cold.shape[1]:
            subset=np.arange(0,cold.shape[0],max(1,cold.shape[0]//800))
            band=_nanpercentile_columns(cold[subset,:,0].T,[16,50,84])
            axes[0,1].fill_between(subset*stride,band[0],band[2],color="#a6cee3",
                                   alpha=.5,linewidth=0,label="16-84 % of walkers")
            axes[0,1].plot(subset*stride,band[1],lw=.8,color="#1f78b4",
                           label="median, beta = 1")
            axes[0,1].legend(fontsize=7,frameon=False,loc="upper right")
            axes[0,1].text(.02,.03,"Tempered chains not saved\n"
                           "(outputs.save_tempered_chains: false)",
                           transform=axes[0,1].transAxes,fontsize=7,color="#555555")
        else:
            axes[0,1].text(.1,.5,"Temperature traces unavailable",transform=axes[0,1].transAxes)
    axes[0,1].set(xlabel=xlabel,ylabel=ylabel,title=title)
    from matplotlib.ticker import EngFormatter
    axes[0,1].xaxis.set_major_formatter(EngFormatter(sep=""))  # 300000 -> 300k

    for ax,method,title,ylabel,color in [
        (axes[1,0],"get_tsw","Temperature swap history",
         "Swap acceptance per sweep\n(fraction of walkers)","#d08b3f"),
        (axes[1,1],"get_smd","Adaptation / SMD history",
         "Swap mean distance per sweep\n(reddemcee SMD)","#8b69a6"),
    ]:
        xlabel="Production sweep"
        try:
            values=np.asarray(getattr(pt_sampler,method)(),float)[int(burnin_sweeps):]
            if values.size==0:raise ValueError("empty history")
            if values.ndim==1:
                ax.plot(values,lw=.9,color=color)
            else:
                # reddemcee stores one row per sweep and one column per adjacent
                # temperature pair (cold -> hot): summarise each pair over the
                # production sweeps.
                flattened=values.reshape(values.shape[0],-1)
                pairs=np.arange(flattened.shape[1])
                band=_nanpercentile_columns(flattened,[16,50,84])
                ax.fill_between(pairs,band[0],band[2],color=color,alpha=.35,
                                linewidth=0,label="16-84 % of sweeps")
                ax.plot(pairs,band[1],lw=1.5,color="#263747",marker="o",ms=3,
                        label="median over sweeps")
                ax.plot(pairs,np.nanmean(flattened,axis=0),lw=.9,color=color,
                        ls="--",label="mean over sweeps")
                ax.legend(fontsize=7,frameon=False)
                xlabel=f"Adjacent temperature pair (t, t+1), cold -> hot\n{flattened.shape[0]} production sweeps"
        except (AttributeError,ValueError,TypeError):
            ax.text(.1,.5,"History not available",transform=ax.transAxes)
        ax.set(xlabel=xlabel,ylabel=ylabel,title=title)
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


def _default_acf_max_lag(chain: np.ndarray, factor: float = 3.0,
                         minimum: int = 20) -> tuple[int, float]:
    """Default ACF lag range, in stored rows: about `factor` x tau_max.

    tau_max is the largest integrated autocorrelation time over the sampled
    parameters, estimated on the stored rows exactly as `kstacker mcmc` does
    for its log (`emcee.autocorr.integrated_time`, tol=0); the plotted lag is
    capped at the stored chain length minus one.  Returns (max_lag in stored
    rows, tau_max in stored rows; NaN when tau is not available, in which case
    the former default min(max(20, steps // 5), 400) is used).
    """
    steps = int(chain.shape[0])
    tau_max = float("nan")
    try:
        import emcee
        with np.errstate(all="ignore"):
            tau = np.asarray(emcee.autocorr.integrated_time(
                np.asarray(chain, dtype=float), tol=0, quiet=True), dtype=float)
        tau = tau[np.isfinite(tau)]
        if tau.size:
            tau_max = float(np.max(tau))
    except Exception:
        tau_max = float("nan")
    if np.isfinite(tau_max) and tau_max > 0:
        max_lag = max(int(minimum), int(np.ceil(factor * tau_max)))
    else:
        max_lag = min(max(int(minimum), steps // 5), 400)
    return min(max_lag, steps - 1), tau_max


def plot_autocorrelation_summary(chain: np.ndarray, output_dir: Path, dpi: int,
                                 parameter_names=None, max_lag: int | None = None,
                                 chain_stride: int=1, lines_per_sweep=None) -> None:
    """Summarize walker-wise autocorrelation functions for every parameter.

    One panel is produced per sampled parameter. For each lag, the thick line is
    the median ACF over walkers and the shaded region spans the 16th to 84th
    percentile range across walkers. This keeps the figure readable while still
    showing between-walker differences in mixing behaviour. Lags are computed
    on stored rows and plotted in production steps / lines (lag x
    `chain_stride`). By default lags run up to about 3 tau_max (largest
    integrated autocorrelation time over parameters, `_default_acf_max_lag`);
    lags beyond the stored chain length are not plotted.
    """
    plt = _plotter()
    steps, nwalkers, ndim = chain.shape
    if steps < 3 or nwalkers < 1:
        return
    names = _labels(ndim, parameter_names)
    tau_max = float("nan")
    if max_lag is None:
        # About 3 tau_max (stored rows), capped at the chain length.
        max_lag, tau_max = _default_acf_max_lag(chain)
    max_lag = min(int(max_lag), steps - 1)
    stride = max(1, int(chain_stride))
    lags = np.arange(max_lag + 1) * stride
    fig, axes = plt.subplots(ndim, 1, figsize=(11.5, 2.2 * ndim),
                             sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for j, ax in enumerate(axes):
        walker_acf = []
        for walker in range(nwalkers):
            walker_acf.append(_autocorrelation_1d(chain[:, walker, j], max_lag))
        walker_acf = np.asarray(walker_acf, dtype=float)
        q16, q50, q84 = _nanpercentile_columns(walker_acf, [16, 50, 84])
        ax.fill_between(lags, q16, q84, color="#a6cee3", alpha=0.45, linewidth=0.0)
        ax.plot(lags, q50, color="#1f78b4", lw=1.8)
        ax.axhline(0.0, color="#6b7280", lw=0.8)
        ax.set_ylabel(names[j])
        ax.set_ylim(-0.2, 1.05)
        ax.tick_params(labelsize=10)
    unit = "production steps" if lines_per_sweep is None else "production lines (inner steps)"
    axes[-1].set_xlabel(f"Lag [{unit}]" + (f" — computed on stored rows, 1 every {stride}"
                                          if stride > 1 else ""))
    from matplotlib.ticker import EngFormatter
    axes[-1].xaxis.set_major_formatter(EngFormatter(sep=""))
    title = "Autocorrelation functions"
    if np.isfinite(tau_max):
        word = "lines" if lines_per_sweep is not None else "steps"
        capped = max_lag < int(np.ceil(3.0 * tau_max))
        title += (f"\nmax lag = {max_lag * stride} {word} "
                  + ("(capped at chain length; " if capped else "~ 3 tau_max (")
                  + f"tau_max = {tau_max * stride:.0f} {word}, largest over parameters)")
    fig.suptitle(title, fontsize=15 if not np.isfinite(tau_max) else 12)
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
    # Rows of `chain` are production steps (emcee) or one production line in
    # `chain_stride` (reddemcee, iteration 08); axes are drawn in those units.
    stride=int(getattr(sampler,"chain_stride",1) or 1)
    lps=int(getattr(sampler,"lines_per_sweep",1)) if hasattr(sampler,"pt_sampler") else None
    plot_cold_chain_trace(chain,output_dir,dpi,parameter_names,
                          chain_stride=stride,lines_per_sweep=lps)
    plot_sampler_overview(chain,logp,np.asarray(sampler.acceptance_fraction,float),output_dir,dpi,
                          chain_stride=stride,lines_per_sweep=lps)
    plot_autocorrelation_summary(chain, output_dir, dpi, parameter_names,
                                 chain_stride=stride, lines_per_sweep=lps)
    if hasattr(sampler,"pt_sampler"):
        plot_apt_diagnostics(sampler.pt_sampler,output_dir,dpi,
                             burnin_sweeps=int(getattr(sampler,"burnin_sweeps",0)),
                             lines_per_sweep=int(getattr(sampler,"lines_per_sweep",1)),
                             chain_burnin_rows=getattr(sampler,"pt_chain_burnin_rows",None),
                             chain_stride=stride,cold_chain=chain)
