import h5py
import numpy as np
import os


def log_prior(params):
    # Define your prior here
    return 0

def log_likelihood(params, data):
    # Define your log-likelihood function here
    return 0

def log_posterior(params, data):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, data)

def reoptimize_mcmc(params, n_jobs=1, n_orbits=None, n_walkers=100, n_steps=1000):
    # We sort the results in several directories
    values_dir = params.get_path("values_dir")
    os.makedirs(f"{values_dir}/fin_fits", exist_ok=True)
    os.makedirs(f"{values_dir}/fin_tiff", exist_ok=True)
    os.makedirs(f"{values_dir}/orbites", exist_ok=True)
    os.makedirs(f"{values_dir}/single", exist_ok=True)
    # os.makedirs(f"{values_dir}/pla", exist_ok=True)

    ts = params.get_ts()  # time of observations (years)
    size = params.n  # number of pixels
    data = params.load_data(method="aperture")

    # define bounds
    bounds = params.grid.bounds()

    # Computation on the q best SNR
    args = (
        ts,
        size,
        params.scale,
        params.fwhm,
        data,
        params.invvar_weight,
    )

    with h5py.File(f"{values_dir}/res_grid.h5") as f:
        # note: results are already sorted by decreasing SNR
        results = f["Best solutions"]

    # Define initial positions for MCMC walkers
    ndim = len(params)
    p0 = np.zeros((n_walkers, ndim))
    for i in range(ndim):
        p0[:, i] = results

    import emcee
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior, args=[data])

    # Run MCMC
    pos, _, _ = sampler.run_mcmc(p0, n_steps)

    # Get best fit
    best_idx = np.argmax(sampler.get_log_prob())
    best_params = pos[best_idx]

    # Sort values with the recomputed SNR
