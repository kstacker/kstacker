# Validation record

## Source and inputs

The code in this package was built from the supplied K-Stacker modules. The example FITS data are the four uploaded 124 × 124 float32 images, stored in `example/images`.

## Checks executed

- Compiled every Python source file with Python's bytecode compiler.
- Parsed the complete example YAML using PyYAML.
- Ran nine automated geometry/plot tests: full-period closure, common sky-track propagator, exact display extent, signed symmetric limits, MAP-ring size, and four-epoch PDF/PNG creation.
- Ran three progress-console tests for text-mode progress, disabled output and ANSI color handling.
- Exercised the numerical likelihood with synthetic NumPy-backed PACO arrays and import-only stubs: positive one-sided SNR, suppression of negative combined SNR, instrument likelihood aggregation and consistency between log prior + likelihood and log posterior.
- Ran two FITS payload tests: four 124 x 124 native images, finite annulus, lack of timing metadata, and an extreme negative central pixel masked by the IWA.
- Installed the local package in editable mode with `--no-deps --no-build-isolation` to check packaging and entry-point metadata. This does not validate scientific dependencies.

## Limits

This runtime does not have `astropy`, `photutils` or `emcee`, and network package installation was unavailable. Full FITS preprocessing, local noise-map generation, a real emcee run, and an end-to-end test with these four images could not be executed here. `tests/smoke_dataset.py` is provided for this validation in a scientific Python environment. A smoke test does not establish scientific convergence.

## Important observation

The fourth FITS image has an extreme negative pixel value near -0.0841 (versus a positive maximum near +0.00746). That extreme falls within the excluded 26-pixel inner working angle. By default the scientific map masks this excluded region for display and uses strict symmetric limits on the remaining valid pixels. The original data and likelihood remain unchanged. Full-frame display remains available with `plots.mask_to_working_region: false`.

## Inference safeguards

- The input images include approximately 144 NaN pixels each near the stellar core; they are finite in the configured science annulus. Image preparation converts invalid pixels to zero for the convolve kernel.
- The four observation times are assumed from the supplied configuration, because the FITS headers do not provide a reliable observing-time series.
- `signed_snr` is an instrument-level positive-amplitude profile: `0.5 * max(SNR_common, 0)^2`.
- `weighting: simple` follows the supplied non-inverse-variance setting.
- Multi-start is configured with five starting centres, selected from the initial grid.
- The narrow nonzero stellar-mass prior is a pragmatic approximation to the supplied 1.59 solar-mass fixed value, because the sampler has a stellar-mass coordinate.
- Reuse of noise cache products depends on the current cache-key and overwrite settings. Regenerate noise maps when changing image centering, mask geometry, upsampling, noise calibration or photometric kernels.

## Posterior-visualization and instrument-registration checks

- Ran 16 local tests for image orientation, exact symmetric color limits, orbit closure, open MAP circle annotations, independent instrument registration, missing-pixel handling, full-walker diagnostic exports, binned corner exports and per-instrument aligned stack creation. Tests use synthetic array-valued instruments, not the full FITS/photometry pipeline.
- Confirmed the saved HDF5 `chain` and `log_prob` datasets are obtained with `thin=1` from every production walker. The YAML `thin` is applied to optional `flat_chain` and `flat_log_prob` only.
- Confirmed the aligned-stack shift direction with detector coordinates `x = centre - East * scale`, `y = centre + North * scale`, including separate instrument payloads.
- The aligned-mean maps are illustrative: their independently averaged, correlated image pixels are not a calibrated S/N detection map.
- The full scientific dependency test remains pending, because `astropy`, `photutils` and `emcee` are not available and network access for installing them failed. Run `python tests/smoke_dataset.py` in a scientific Conda environment. A short smoke test checks execution paths, not MCMC convergence.
