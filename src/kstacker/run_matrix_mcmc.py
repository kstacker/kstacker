from kstacker.orbit import corner_plots_mcmc
from kstacker.mcmc_reoptimization import make_plots
import kstacker.Matrix_Likelihood as ML

from pathlib import Path
import numpy as np
import emcee
import h5py
from joblib import Parallel, delayed
from astropy.io import ascii
import time
from multiprocessing import Pool

def log_posterior(orbital_params):
    """

    Parameters
    ----------
    orbital_params : list
        a list containing a, e, t0, m0, omega, i, theta_0 value given by emcee.
    args : list
        a list containing ts, size, scale, fwhm, bounds, fast, treated_image.
    state : kstacker.Matrix_Likelihood.MCMCState
        manage the state of the sampler value.

    Returns
    -------
    float
       log likelihood and log posterior value for these x values.

    """
    global ts, size, scale, fwhm, bounds, fast, treated_image, state
    state.walker_index = (state.walker_index + 1) % state.n_walkers # index update
    if fast:
        if state.sampler.iteration > 1:
            chain = state.sampler.get_chain()
            accepted = np.any(chain[-1] != chain[-2], axis=1)[state.walker_index]
            if accepted:
                state.cached_terms[state.walker_index][0] = state.cached_terms[state.walker_index][1]
    lp = log_prior(orbital_params,bounds)
    if np.isinf(lp):
        return -np.inf
    log_likelihood_value = ML.log_likelihood(orbital_params, ts, size, scale, fwhm, fast, state, treated_image)
    if np.isinf(log_likelihood_value):
        return -np.inf
    else:
        return lp + log_likelihood_value

def log_prior(orbital_params,bounds):
    if not all(bound[0] <= param <= bound[1] for param, bound in zip(orbital_params, bounds)):
        return -np.inf
    return 0

def set_globals(ts_, size_, scale_, fwhm_, bounds_, fast_, treated_image_, state_):
    global ts, size, scale, fwhm, bounds, fast, treated_image, state
    ts = ts_
    size = size_
    scale = scale_
    fwhm = fwhm_
    bounds = bounds_
    fast = fast_
    treated_image = treated_image_
    state = state_

def compute_mcmc_matrix(params, n_jobs=1, n_walkers=28, n_steps=100000, n_orbits=1000, n_check=1000, fixed_params=None):
    profile_dir = params.get_path("profile_dir")
    values_dir = params.get_path("values_dir")
    ts = np.array(params.get_ts())
    size = params.n
    scale = params.scale
    fwhm = params.fwhm
    data = params.load_data(method="aperture")
    images = data['images']
    N,M,_ = np.shape(images)
    
    treated_image = ML.extract_preteated_image(profile_dir,N,M)
    
    bounds = params.grid.bounds()
    
    with h5py.File(f"{values_dir}/res_grid.h5") as f:
        # note: results are already sorted by decreasing SNR
        results = f["Best solutions"][:]
        
    n_walkers = min(n_walkers, results.shape[0])
    
    # Define search range
    nbr_psf = 1.
    param_names = ["a", "e", "t0", "m0", "omega", "i", "theta_0"]
    if fixed_params == None: 
        final_param_names = param_names
        delta_param = {key: None for key in final_param_names}
        
        unfixed_param_indices = np.linspace(0,6,7,dtype=int)
    else :
        final_param_names = [key for key in param_names if key not in fixed_params]
        delta_param = {key: None for key in final_param_names}
        
        unfixed_param_indices = [index for index, keys in enumerate(param_names) if keys not in fixed_params]
    
    p0 = results[:n_walkers, unfixed_param_indices].copy()
    
    for i in range(len(delta_param)):
        index = param_names.index(final_param_names[i])
        delta_param[final_param_names[i]] = nbr_psf * (bounds[index][1]-bounds[index][0]) / params.grid.limits(final_param_names[i])[2]
    
    # Loop over each walker and each parameter to add
    # a small random perturbation to create independence between walkers
    for walker in range(n_walkers):
        for i in range(len(delta_param)):
            # Set up the initial flag for checking bounds
            in_bounds = False
    
            # Loop until the random perturbation is within the bounds
            while not in_bounds:
                # Generate random factor in range [-1, 1]
                random_factor = (np.random.rand() - 0.5) * 2
                perturbation = random_factor * delta_param[final_param_names[i]]
    
                # Add the perturbation to the parameter
                new_value = p0[5, i] + perturbation # This line use only one of the best output of  brute-force+gradiant (line 5)
                # new_value = p0[walker, param_index] + perturbation # This line use all the outputs of brute-force+gradiant (Doesn't work! before using this line, take into account modulos pi on Omega and omega)

                # Check if new values are in the bounds
                if bounds[param_names.index(final_param_names[i])][0] <= new_value <= bounds[param_names.index(final_param_names[i])][1]:
                    p0[walker, i] = new_value
                    in_bounds = True
    
    # mcmc configuration
    
    # Calculate the mean for each column
    means = np.mean(p0, axis=0)
    bounds = []
    for  i in range(len(delta_param)):
        index = param_names.index(final_param_names[i])
        bounds.append((means[i] - delta_param[final_param_names[i]], means[i] + delta_param[final_param_names[i]]))
    
    pos = p0
    
    pos = np.array(pos)
    
    state = ML.MCMCState(n_walkers)
    
    state.fixed_params = fixed_params
    
    state.r_vals, state.j0_vals = ML.precompute_bessel_lookup()
    
    start = time.time()
    
    ndim = len(bounds)
    
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior)
    set_globals(ts, size, scale, fwhm, bounds, True, treated_image, state)
    state.set_sampler(sampler)
    
    log_path = Path(f"{values_dir}/mcmc_log.txt")
    log_path.write_text("")  # vide le fichier au début (équivalent à "w")
    
    with Pool(processes=n_jobs) as pool:
        sampler.pool = pool
        try:
            for i in range(0, n_steps, n_check):
                pos, _, _ = sampler.run_mcmc(pos, n_check, progress=True)

                if sampler.iteration > 2*n_check:
                    tau = sampler.get_autocorr_time(tol=0)
                    with open(log_path, "a") as f:
                        f.write(f"Step {sampler.iteration}: Autocorrelation time = {tau}")
                        f.write(f"Step {sampler.iteration}: tau*50/iter = {(tau * 50)/sampler.iteration}\n")
                        f.write(f"Step {sampler.iteration}: mean acceptance = {np.mean(sampler.acceptance_fraction)}\n")

                        if np.all((tau * 50)/sampler.iteration < 1):
                            end = time.time()
                            f.write("Convergence criteria met\n")
                            f.write(f"Time taken : {end-start}\n")
                            break
            
            with open(log_path, "a") as f:
                end = time.time()
                f.write("Convergence criteria not met\n")
                f.write(f"Time taken : {end-start}\n")

        except Exception as e:
            with open(log_path, "a") as f:
                f.write(f"An error occurred during MCMC execution: {e}\n")
        
            try:
            
                # Get the final chain of parameters
                samples = sampler.get_chain(flat=True)  # shape: (n_steps * n_walkers, n_params)
                log_probs = sampler.get_log_prob(flat=True)  # shape: (n_steps * n_walkers,)
            
                # Remove invalid values from log_probs
                unique_samples, unique_indices = np.unique(samples, axis=0, return_index=True)
                unique_log_probs = log_probs[unique_indices]
                
                valid_indices = np.isfinite(unique_log_probs)
                filtered_samples = unique_samples[valid_indices]
                filtered_log_probs = unique_log_probs[valid_indices]
                
                # Tri décroissant selon log_prob
                sorted_indices = np.argsort(-filtered_log_probs)
                final_samples = filtered_samples[sorted_indices]
                final_log_probs = filtered_log_probs[sorted_indices]
                
                # Prepare an array to store the top 100 results
                reopt_mcmc = []
                for idx in sorted_indices:
                    # Extract parameter values for each of the top 100 samples
                    a, e, t0, m0, omega, i, theta_0 = final_samples[idx]
                    log_prob = final_log_probs[idx]
                    reopt_mcmc.append([idx, log_prob, a, e, t0, m0, omega, i, theta_0])
                
                reopt_mcmc = np.array(reopt_mcmc[:1000])
                # Add index column
                reopt_mcmc = np.concatenate([np.arange(reopt_mcmc.shape[0])[:, None], reopt_mcmc], axis=1)
                # Save results
                names = ("image_number", "best_indice", "log_prob", "a", "e", "t0", "m0", "omega", "i", "theta_0")
                ascii.write(
                    reopt_mcmc,
                    f"{values_dir}/results_mcmc.txt",
                    names=names,
                    format="fixed_width_two_line",
                    formats={"image_number": "%d"},
                    overwrite=True,
                )
             
                # Plots results
                Parallel(n_jobs=n_jobs)(
                    delayed(make_plots)(
                        reopt_mcmc[k, 3:], k, params, data["images"], ts, values_dir)        
                    for k in range(min(n_orbits, 100))
                )
                
                corner_plots_mcmc(params, nbins=5)
            
            except ValueError as e:
                with open(log_path, "a") as f: f.write(f"ValueError: {e}\n")
            
            except IOError as e:
                with open(log_path, "a") as f: f.write(f"File error: {e}\n")
            
            except Exception as e:
                with open(log_path, "a") as f: f.write(f"Unexpected error: {e}\n")