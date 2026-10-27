# K-Stacker: multi-epoch orbit inference

K-Stacker infers a common Keplerian orbit from one or more direct-imaging instruments. The package covers image preparation, local background/noise estimation, deterministic orbital-grid initialization, MCMC sampling, and publication-oriented figures for the resulting posterior.

## What the code does

The workflow is organised around a small number of clear modules:

- `src/kstacker/orbital_geometry.py` implements the equinoctial orbital parameterisation and the sky projection used everywhere else in the code.
- `src/kstacker/direct_imaging.py` handles FITS loading, convolve/aperture/PACO photometry, masks, and instrument-specific noise products.
- `src/kstacker/mcmc_init_search.py` runs the deterministic initial grid search used to find good starting points.
- `src/kstacker/mcmc.py` builds the posterior, prepares walkers, and runs emcee or adaptive parallel tempering.
- `src/kstacker/scientific_plots.py` generates observation maps, aligned stacks, projected orbits, posterior sky densities and corner plots.
- `src/kstacker/mcmc_diagnostics.py` generates trace plots, log-posterior diagnostics and tempering diagnostics.
- `src/kstacker/console.py` provides concise, colour-aware terminal messages and consistent progress bars.

An example dataset is provided in `example/` with four FITS frames and a YAML file ready to run.

## Installation

Use Python 3.10 or newer (Python 3.11 is recommended):

```bash
python -m pip install -e .
kstacker --help
```

## Basic workflow

From the package root:

```bash
cd example
kstacker init parameters_harmoni_mcmc.yml
kstacker mcmc parameters_harmoni_mcmc.yml
kstacker plot parameters_harmoni_mcmc.yml
```

For a quick end-to-end check with reduced settings, run:

```bash
python tests/smoke_dataset.py
```

This smoke test is only a pipeline check. It does not establish convergence or detection significance.

## Observation model

Each instrument contributes its own likelihood term, noise model, and optional flux parameter. The shared orbital state is

`theta = (a, lambda_ref, Mstar, h, k, p, q)`

and the posterior factorises as

`log P(theta | D) = log P_orbit(theta) + sum_j log L_j(D_j | theta)`

with additional independent flux-prior terms when flux amplitudes are sampled.

The currently implemented likelihood modes are:

- `signed_snr`
  - one-sided common-SNR score
  - `log L_j = 0.5 * max(SNR_common,j, 0)^2`
- `positive_snr_profile`
  - PACO-only positive standardized profile
  - `log L_j = 0.5 * sum_k max(0, z_k)^2`
- `flux`
  - Gaussian common-flux likelihood
  - optionally with one flux parameter per instrument

The one-sided SNR score is convenient for direct-imaging searches, but it is still a modelling choice: convergence and robustness should always be checked with care.

## Figures

### Observation maps

Each epoch is displayed with its own symmetric colour scale `[-A, A]`, where `A = max(abs(min), abs(max))` over the finite displayed pixels. By default, the displayed region is restricted to the working annulus defined by the IWA and OWA. The MAP position is shown with a large **open** circle so the underlying signal remains visible.

### Aligned stacks

For each instrument, all epochs are shifted so that the MAP-predicted planet location coincides with the location at the first epoch. The figure shows only the resulting aligned mean image. This plot is a visual aid; it is not a likelihood product and it is never used by the inference.

### Projected orbits

The projected-orbit figure contains one clearly highlighted MAP orbit and a set of paler posterior draws. Observation times are shown as open circles computed from the MAP state at the instrument epochs.

### Corner plots and diagnostics

Corner plots use readable one-dimensional histograms and binned two-dimensional occupancy maps. The MCMC diagnostics show every saved production step and every walker. The HDF5 file remains the authoritative record; the plots are there to help interpret it.

## Outputs

The main output files are written under `values/`:

- `mcmc_init_search.h5` and `mcmc_init_search_top.csv`: ranked initial-grid-search results
- `mcmc_chain_and_logprob.h5`: saved production chains and log posterior values
- `figures/science/`: science figures (observations, aligned stacks, orbits, posterior sky densities, corner plots)
- `figures/convergence/`: sampler diagnostics

For convolve/aperture workflows, cached local noise products are written under `profiles/`. These are background/noise products, not stacked observation images.

## Scientific checks

The package includes tests for geometry, plotting conventions, console behaviour and likelihood assembly. They are useful sanity checks, but they are not a substitute for a scientific validation of a run. For real analyses, check:

- convergence across independent starts
- autocorrelation time and effective sample size
- consistency of the noise model
- sensitivity to prior choices
- robustness of the inference when a subset of epochs is removed

The highest-posterior sample is a convenient reference state, but it should not be interpreted as a full summary of the posterior on its own.


## Convergence assessment

A practical guide to sampler diagnostics is available in `CONVERGENCE_GUIDE.md`. The plotting command writes trace plots, log-posterior diagnostics, adaptive-tempering diagnostics when relevant, and autocorrelation summaries.
