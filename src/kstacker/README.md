# Keplerian orbital inference

Multi-epoch, multi-instrument orbit inference from direct-imaging observations. The software loads FITS products, propagates a seven-parameter Keplerian state, evaluates an orbital prior plus independent data log-likelihoods, and samples the posterior using `emcee` or adaptive parallel tempering (`reddemcee`).

## Modules

| Module | Responsibility |
|---|---|
| `orbital_geometry.py` | Equinoctial Kepler propagation and North/East projection |
| `direct_imaging.py` | Instrument contexts, FITS loading, noise maps, PACO, photometry, image log likelihoods |
| `mcmc.py` | Orbital priors, joint posterior, emcee and reddemcee, chain HDF5 outputs |
| `mcmc_init_search.py` | Physical grid and total-log-likelihood ranking |
| `scientific_plots.py` | Sky orbits, epoch images, posterior sky density, orbit-aligned visualization |
| `mcmc_diagnostics.py` | Chain, acceptance, autocorrelation, and parallel-tempering diagnostics |
| `console.py` | ANSI sections, terminal progress and line-based progress for redirected logs |
| `utils.py`, `cli.py` | Configuration and command-line entry points |

`__init__.py` and `version.py` complete the Python package.

## Mathematical model

A common orbital state is

\[
\theta=(a,\lambda_0,M_\star,h,k,p,q),\quad
h=e\sin(\Omega+\omega),\quad k=e\cos(\Omega+\omega),\quad
p=\sin(i/2)\sin\Omega,\quad q=\sin(i/2)\cos\Omega.
\]

The posterior is

\[
\log p(\theta\mid D)=\log p(\theta)+\sum_{j=1}^{N_{\rm inst}}
\log L_j(D_j\mid\theta),
\]

assuming **conditionally independent data**. A single instrument can contain multiple epochs. Its `method` selects the data backend (`convolve`, `aperture`, `paco`); its `likelihood_mode` selects the statistical model:

- **`signed_snr`**: \(\log L_j = \tfrac12 \mathrm{SNR}_j^2\) for the **signed common-flux S/N of that instrument's epochs**. Each instrument has a separately profiled amplitude; negative and positive signed S/N of equal magnitude yield equal scores by construction.
- **`positive_snr_profile`**: \(\log L_j=\tfrac12\sum_k\max(0,z_{jk})^2\), available for PACO products. This profiles positive contributions separately across epochs.
- **`flux`**: \(\log L_j=-\tfrac12(f_p^2S_{1j}-2f_pS_{2j})\). `sample_fp: true` adds a **single common flux coordinate** to all `flux` instruments. The calibration/physical units must support that assumption. With no sampled amplitude, `fp_bounds[0]` is the fixed amplitude in this statistical definition.

Independent per-instrument profiling and one global profile are **not statistically equivalent**. Mixed data from the same raw exposures must not be treated as independent twice. The statistical criteria are relative log-likelihood expressions (additive data-only constants omitted); do not interpret these outputs as fully normalized model evidences.

## Install and run

Python 3.10+ with NumPy, SciPy, Astropy, Photutils, PyYAML, h5py, matplotlib, tqdm and emcee is required. Install `reddemcee` to select adaptive parallel tempering. Install these dependencies in the scientific environment used to run the pipeline.

Copy `parameters_harmoni_mcmc.yml` beside the observation products. Paths to images, PACO products, epochs and instrument geometry are resolved using the YAML and observation files. You can use multiple `instruments` entries, each with its own `method`, `likelihood_mode` and `times_years`.

From the directory **containing** the `keplerian_inference` package:

```bash
python -m keplerian_inference.cli init /path/to/parameters_harmoni_mcmc.yml
python -m keplerian_inference.cli mcmc /path/to/parameters_harmoni_mcmc.yml
# Regenerate figures without sampling again:
python -m keplerian_inference.cli plot /path/to/parameters_harmoni_mcmc.yml
```

Alternatively call `run_mcmc_from_yaml` and `search_mcmc_initial_parameters_from_yaml` from their respective modules.

The initialization grid evaluates **the sum of the instrument log likelihoods**, not a joint S/N. An explicit sampled flux dimension is evaluated at `mcmc.fp_init` during the physical grid search; with a fixed flux it uses `mcmc.fp_bounds[0]`. The saved files in `values/` contain `total_log_likelihood` as the ranking column. The MCMC driver uses the saved orbital centres and evaluates the same instrument likelihoods.

For a shell-script job with readable, line-oriented progress:

```bash
export PYTHONUNBUFFERED=1
export KSTACKER_PROGRESS=log
export KSTACKER_COLOR=never
python -m keplerian_inference.cli init parameters_harmoni_mcmc.yml > init_job.log 2>&1
python -m keplerian_inference.cli mcmc parameters_harmoni_mcmc.yml > sampler_job.log 2>&1
# In another terminal:
tail -f sampler_job.log
```

For a batch script, run:

```bash
bash keplerian_inference/run_orbit_inference.sh parameters_harmoni_mcmc.yml all > orbit_job.log 2>&1
# In a second shell: tail -f orbit_job.log
```

`KSTACKER_COLOR=always` forces ANSI colors if the log viewer supports them; the YAML `logging.color` setting normally controls automatic behavior. Set `logging.progress_mode: terminal` for live TTY bars or `log` for occasional full-line job-log progress. Console output may also be mirrored to `values/mcmc_console.log`.

## Outputs

```
values/
  mcmc_init_search.h5
  mcmc_init_search_top.csv
  mcmc_chain_and_logprob.h5
  mcmc_console.log
  figures/
    science/
      joint_sky_orbits.pdf
      orbital_posterior_corner.pdf
      physical_orbital_corner.pdf
      <instrument>_epoch_images_01.pdf
      <instrument>_position_log_posterior_01.pdf
      orbit_aligned_display_stack.pdf     # when more than one instrument
    convergence/
      cold_chain_traces.pdf
      sampler_overview.pdf
      adaptive_tempering_diagnostics.pdf  # when reddemcee is selected
```

Each figure is also exported as PNG. Additional numbered pages cover **every epoch** if an instrument has more than nine observations.

**Sky-coordinate convention:** x is East [AU], increasing leftward, and y is North [AU], increasing upward. The instrument overlay marks the maximum-posterior orbit and two-dimensional positional credible contours. Posterior position maps are *projected posterior sample density* (log10 of probability mass in display bins), not pixel-by-pixel instrument log-likelihood maps. The full orbit produces a position at every exposure time.

The orbit-aligned standardized stack resamples the images to a common sky plane, shifts them along the orbit at the maximum posterior sample and combines a *display-only* standardized proxy. Instrument units and noise correlations are not assumed interchangeable. **Do not report its displayed values as physical detection significance, independent evidence or a substitute for the joint log-likelihood.** Use native calibrated per-instrument images and the likelihood outputs for quantitative analysis.

For sampler interpretation, see [`CONVERGENCE_GUIDE.md`](CONVERGENCE_GUIDE.md).
