import os

import h5py
import numpy as np
import emcee
import matplotlib.pyplot as plt
from astropy.io import ascii, fits
from astropy.visualization import ZScaleInterval
from joblib import Parallel, delayed
from pathlib import Path
from multiprocessing import Pool

from .imagerie import recombine_images
from .orbit import orbit, plot_ontop, plot_orbites
from .likelihood import compute_log_likelihood
from .mcmc_starting_pos import read_starting_file
import time
from kstacker.orbit import plot_converge_points_map
from kstacker.PSF_shape_mcmc import precompute_bessel_lookup

def plot_coadd(idx, coadded, x, params, outdir):
    a, e, t0, m0, omega, i, theta_0 = x
    # plot the corresponding image and save it as a png (for quick view)
    plt.figure()
    vmin, vmax = ZScaleInterval().get_limits(coadded)
    plt.imshow(
        coadded, origin="lower", interpolation="none", cmap="gray", vmin=vmin, vmax=vmax
    )
    plt.colorbar()
    xa, ya = orbit.project_position_full(t0, a, e, t0, m0, omega, i, theta_0)
    xpix = params.n // 2 + params.scale * xa
    ypix = params.n // 2 + params.scale * ya
    # comment this line if you don't want to see where the planet is recombined:
    # decalage 2 fwhm
    plt.scatter(ypix - 2 * params.fwhm, xpix, color="b", marker=">")
    plt.savefig(f"{outdir}/fin_png/fin_{idx}.png")
    plt.close()

    fits.writeto(f"{outdir}/fin_fits/fin_{idx}.fits", coadded, overwrite=True)

def make_plots(x_best, k, params, images, ts, values_dir):
    print(f"Make plots for solution {k+1}")
    # create combined images (for the q eme best SNR)
    coadded = recombine_images(images, ts, params.scale, *x_best)
    plot_coadd(k, coadded, x_best, params, values_dir)

    # plot the orbits
    ax = [params.xmin, params.xmax, params.ymin, params.ymax]
    plot_orbites(ts, x_best, ax, f"{values_dir}/orbites/orbites{k}")

    # If single_plot=='yes' a cross is ploted on each image where the
    # planet is found (by default no);
    if params.single_plot == "yes":
        for l in range(len(ts)):
            plot_ontop(
                x_best,
                params.dist,
                [ts[l]],
                params.resol,
                images[l],
                f"{values_dir}/single/single_{k}fin_{l}",
            )

def log_prior(orbital_params, bounds, prior_info=None):
    # Check limits for each parameter
    if not all(bound[0] <= param <= bound[1] for param, bound in zip(orbital_params, bounds)):
        return -np.inf
    else:
        return 0.

    # Compute priors
    log_prior_value = 0
    if prior_info is not None:
        # For priors
        for i, (param, info) in enumerate(zip(orbital_params, prior_info)):
            if info is not None:
                mean, sigma = info
                log_prior_value += -0.5 * ((param - mean) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))
    return log_prior_value


def log_likelihood(x, ts, size, scale, fwhm, data):
    # log-likelihood function
    param_names = ["a", "e", "t0", "m0", "omega", "i", "theta_0"] # all the name of the parameters
    x_complete = [0] * 7 # initialise the final paramters variable
    
    if fixed_params is None: # all the parameters are free
        unfixed_param_indices = range(7)
        x_complete = list(x)
    else:
        unfixed_param_indices = [i for i, name in enumerate(param_names) if name not in fixed_params] # get the indices of the unfixed variables
        for x_val, i in zip(x, unfixed_param_indices): # save the unfixed variable into the final orbital parameters variable
            x_complete[i] = x_val
        for name, val in fixed_params.items(): # save the fixed variable into the final orbital parameters variable
            x_complete[param_names.index(name)] = val
            
    global r_vals, j0_vals
            
    loglikelihood = compute_log_likelihood(x_complete,
    ts,
    size,
    scale,
    fwhm,
    data,
    r_vals, 
    j0_vals,
    exclude_source=True,
    exclude_lobes=True,
    method="aperture",
    upsampling_factor=None,
    use_interp_bgnoise=False,
    r_mask=r_mask,
    r_mask_ext=r_mask_ext,
    return_all=False)

    return loglikelihood


def log_posterior(orbital_params):
    global ts, size, scale, fwhm, data, bounds
    # Check if parameters are within bounds
    if not all(bound[0] <= param <= bound[1] for param, bound in zip(orbital_params, bounds)):
        return -np.inf
    #lp = log_prior(orbital_params, bounds)
    #if not np.isfinite(lp):
    #    return -np.inf
    lp = 0.
    return lp + log_likelihood(orbital_params, ts, size, scale, fwhm, data)

def set_globals(ts_, size_, scale_, fwhm_, data_, bounds_, fixed_params_, r_vals_, j0_vals_, r_mask_=None, r_mask_ext_=None):
    global ts, size, scale, fwhm, data, bounds, fixed_params, r_mask, r_mask_ext, r_vals, j0_vals
    ts = ts_
    size = size_
    scale = scale_
    fwhm = fwhm_
    data = data_
    bounds = bounds_
    fixed_params = fixed_params_
    r_mask=r_mask_
    r_mask_ext=r_mask_ext_
    r_vals = r_vals_
    j0_vals = j0_vals_

def reoptimize_mcmc(params, n_jobs=1, n_walkers=28, n_steps=100000, n_orbits=1000, n_check=1000, fixed_params=None, nbr_psf=1., init_pos_precomputed=False):
    # We sort the results in several directories
    values_dir = params.get_path("values_dir")
    os.makedirs(f"{values_dir}/fin_fits", exist_ok=True)
    os.makedirs(f"{values_dir}/fin_png", exist_ok=True)
    os.makedirs(f"{values_dir}/orbites", exist_ok=True)
    os.makedirs(f"{values_dir}/single", exist_ok=True)
    # os.makedirs(f"{values_dir}/pla", exist_ok=True)

    ts = params.get_ts()  # time of observations (years)
    size = params.n  # number of pixels
    data = params.load_data(method="aperture")
    
    if init_pos_precomputed:
        p0, bounds = read_starting_file(params)
    
    elif not init_pos_precomputed:
        bounds = params.grid.bounds()
        
        with h5py.File(f"{values_dir}/res_grid.h5") as f:
            # note: results are already sorted by decreasing SNR
            results = f["Best solutions"][:]
        
        n_walkers = min(n_walkers, results.shape[0])
        
        # Define search range
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
        
    pos = np.array(p0)
    ndim = len(bounds)
    
    r_mask = 30
    r_mask_ext = size//2
    r_vals, j0_vals = precompute_bessel_lookup()
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior)
    set_globals(ts, size, params.scale, params.fwhm, data, bounds, fixed_params, r_vals, j0_vals, r_mask_=r_mask, r_mask_ext_=r_mask_ext)
    
    log_path = Path(f"{values_dir}/mcmc_log.txt")
    log_path.write_text("")
    
    start = time.time()
    
    with Pool(processes=n_jobs) as pool:
        sampler.pool = pool
        sampler.run_mcmc(pos, n_check, progress=True)
        
    samples = sampler.get_chain(flat=True)  # shape: (n_steps * n_walkers, n_params)
    log_probs = sampler.get_log_prob(flat=True)  # shape: (n_steps * n_walkers,)

    unique_samples, unique_indices = np.unique(samples, axis=0, return_index=True)
    unique_log_probs = log_probs[unique_indices]

    # Remove invalid values from log_probs
    valid_indices = np.isfinite(unique_log_probs)  # True for finite values, False for -inf
    if not np.any(valid_indices):
        raise ValueError("All values in log_probs are invalid (e.g., -inf or NaN)")

    filtered_log_probs = unique_log_probs[valid_indices]
    filtered_samples = unique_samples[valid_indices]
    
    plot_converge_points_map(data["images"],ts,scale,4,124,filtered_log_probs,filtered_samples,values_dir)
    
    plt.hist(filtered_samples.T[0])
    
    # plot_converge_points_map(data["images"],ts,scale,4,124,filtered_log_probs,filtered_samples,values_dir)
    #     try:
    #         for i in range(0, n_steps, n_check):
    #             pos, _, _ = sampler.run_mcmc(pos, n_check, progress=True)

    #             if sampler.iteration > 1:#6*n_check:
    #                 tau = sampler.get_autocorr_time(tol=0)
    #                 with open(log_path, "a") as f:
    #                     f.write(f"Step {sampler.iteration}: Autocorrelation time = {tau}")
    #                     f.write(f"Step {sampler.iteration}: tau*50/iter = {(tau * 50)/sampler.iteration}\n")
    #                     f.write(f"Step {sampler.iteration}: mean acceptance = {np.mean(sampler.acceptance_fraction)}\n")
                        
    #                     if np.all((tau * 50)/sampler.iteration < 1):
    #                         end = time.time()
    #                         written = True
    #                         f.write("Convergence criteria met\n")
    #                         f.write(f"Time taken : {end-start}\n")
    #                         break
                        
    #         with open(log_path, "a") as f:
    #             if not written:
    #                 end = time.time()
    #                 f.write("Convergence criteria not met\n")
    #                 f.write(f"Time taken : {end-start}\n")

    #     except Exception as e:
    #         with open(log_path, "a") as f:
    #             f.write(f"An error occurred during MCMC execution: {e}\n")

    # try:
    #     # Get the final chain of parameters
    #     samples = sampler.get_chain(flat=True)  # shape: (n_steps * n_walkers, n_params)
    #     log_probs = sampler.get_log_prob(flat=True)  # shape: (n_steps * n_walkers,)

    #     unique_samples, unique_indices = np.unique(samples, axis=0, return_index=True)
    #     unique_log_probs = log_probs[unique_indices]

    #     # Remove invalid values from log_probs
    #     valid_indices = np.isfinite(unique_log_probs)  # True for finite values, False for -inf
    #     if not np.any(valid_indices):
    #         raise ValueError("All values in log_probs are invalid (e.g., -inf or NaN)")

    #     filtered_log_probs = unique_log_probs[valid_indices]
    #     filtered_samples = unique_samples[valid_indices]

    #     #print(f"Debug: n_orbits={n_orbits} (type={type(n_orbits)}), len(filtered_log_probs)={len(filtered_log_probs)}")

    #     n_orbits = int(n_orbits)

    #     if len(filtered_log_probs) < n_orbits:
    #         print(
    #             f"Warning: n_orbits ({n_orbits}) exceeds the number of valid samples ({len(filtered_log_probs)}). Adjusting n_orbits to {len(filtered_log_probs)}.")
    #         n_orbits = len(filtered_log_probs)


    #     # Sort valid log_probs and get best indices
    #     best_indices = np.argsort(filtered_log_probs)[-int(n_orbits):][::-1]

    #     # Prepare an array to store the top 100 results
    #     reopt_mcmc = []
    #     for idx in best_indices:
    #         # Extract parameter values for each of the top 100 samples
    #         a, e, t0, m0, omega, i, theta_0 = filtered_samples[idx]
    #         log_prob = filtered_log_probs[idx]
    #         reopt_mcmc.append([idx, log_prob, a, e, t0, m0, omega, i, theta_0])

    #     reopt_mcmc = np.array(reopt_mcmc)
    #     # Add index column
    #     reopt_mcmc = np.concatenate([np.arange(reopt_mcmc.shape[0])[:, None], reopt_mcmc], axis=1)
    #     # Save results
    #     names = ("image_number", "best_indice", "log_prob", "a", "e", "t0", "m0", "omega", "i", "theta_0")
    #     ascii.write(
    #         reopt_mcmc,
    #         f"{values_dir}/results_mcmc.txt",
    #         names=names,
    #         format="fixed_width_two_line",
    #         formats={"image_number": "%d"},
    #         overwrite=True,
    #     )
     
    #     # Plots results
    #     Parallel(n_jobs=n_jobs)(
    #         delayed(make_plots)(
    #             reopt_mcmc[k, 3:], k, params, data["images"], ts, values_dir)        
    #         for k in range(min(n_orbits, 100))
    #     )

    #     print("Done!")
        

    # except ValueError as e:
    #     with open(log_path, "a") as f: f.write(f"ValueError: {e}\n")
    
    # except IOError as e:
    #     with open(log_path, "a") as f: f.write(f"File error: {e}\n")
    
    # except Exception as e:
    #     with open(log_path, "a") as f: f.write(f"Unexpected error: {e}\n")
