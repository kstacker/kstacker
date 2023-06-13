import os
from .snr import compute_snr
import emcee

def log_prior(pos, params):
    """
    Calculate the logarithm of the prior probability for the given parameters.

    Parameters:
    -----------
    params : array_like
        Array of model parameters: [a, e, t0, m0, omega, theta_0, i]
    grid : dict
        Dictionary containing the grid limits for each parameter.

    Returns:
    --------
    float
        The logarithm of the prior probability.

    note perso pour brancher la fonction sur notre code:
    ____________________________________________________

    a, e, t0, m0, omega, theta_0, i = results[1][:7]
    (verifier l'ordre des paramétres)
    """
    # Unpack the parameter
    a, e, t0, m0, omega, theta_0, i = pos

    # Get the parameter limits from the grid
    a_min, a_max, a_step = params.grid.limits("a")
    e_min, e_max, e_step = params.grid.limits("e")
    t0_min, t0_max, t0_step = params.grid.limits("t0")
    m0_min, m0_max, m0_step = params.grid.limits("m0")
    omega_min, omega_max, omega_step = params.grid.limits("omega")
    theta_0_min, theta_0_max, theta_0_step = params.grid.limits("theta_0")
    i_min, i_max, i_step = params.grid.limits("i")

    # Check if the parameters are within the allowed ranges
    if (a_min <= a <= a_max) and (e_min <= e <= e_max) and (t0_min <= t0 <= t0_max) and \
            (m0_min <= m0 <= m0_max) and (omega_min <= omega <= omega_max) and \
            (theta_0_min <= theta_0 <= theta_0_max) and (i_min <= i <= i_max):

        prior = 0.
        return prior
    else:
        # If any parameter is outside the allowed range, return -inf
        return -np.inf

def compute_snr_objfun(pos, ts, size, scale, fwhm, data, invvar_weighted, r_mask):
    """returns SNR."""

    return compute_snr(
        pos,
        ts,
        size,
        scale,
        fwhm,
        data,
        invvar_weighted=invvar_weighted,
        exclude_source=True,
        exclude_lobes=True,
        r_mask=r_mask)

def log_likelihood(pos, params):

    ts = params.get_ts()  # time of observations (years)
    size = params.n  # number of pixels
    scale = params.scale
    fwhm = params.fwhm
    invar_weighted = params.invar_weight
    r_mask = params.r_mask
    data = params.load_data(method="aperture")

    return compute_snr_objfun(pos, ts, size, scale, fwhm, data, invar_weighted, r_mask)

def log_posterior(pos, params):

    lp = log_prior(pos, params)

    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(pos, params)


def reoptimize_mcmc(params, n_walkers=2, n_steps=1000):
    # We sort the results in several directories
    values_dir = params.get_path("values_dir")
    os.makedirs(f"{values_dir}/fin_fits", exist_ok=True)
    os.makedirs(f"{values_dir}/fin_tiff", exist_ok=True)
    os.makedirs(f"{values_dir}/orbites", exist_ok=True)
    os.makedirs(f"{values_dir}/single", exist_ok=True)

    # Sort the results by decreasing SNR
    with h5py.File(f"{values_dir}/res_grid.h5") as f:
        results = f["Best solutions"][:n_walkers]

    ndim = 7

    pos = results[:n_walkers, :ndim]

    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior, args=[params], threads=n_walkers)

    # Burn-in phase
    # pos, prob, state = sampler.run_mcmc(pos, n_steps, progress=True)
    # sampler.reset()

    # Production phase
    pos, prob, state = sampler.run_mcmc(pos, n_steps, progress=True, Processes=n_walkers)

    print(f"\nOptimizing orbits")

    # Select the best solution as the last one
    best_sol_idx = np.argmax(prob)
    best_sol = sampler.flatchain[best_sol_idx]

    # Save the best solution
    with h5py.File(f"{values_dir}/single/best_mcmc.h5", "w") as f:
        f.create_dataset("Best solution", data=best_sol)

    # Save the MCMC chain and the posterior probability
    with h5py.File(f"{values_dir}/orbites/mcmc.h5", "w") as f:
        f.create_dataset("Chain", data=sampler.chain)
        f.create_dataset("Posterior probability", data=sampler.lnprobability)

    print(f"Best solution: {best_sol}")
