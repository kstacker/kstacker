import os

import h5py
import numpy as np
import emcee
import matplotlib.pyplot as plt
from astropy.io import ascii, fits
from astropy.visualization import ZScaleInterval
from joblib import Parallel, delayed

from .imagerie import recombine_images
from .orbit import orbit, plot_ontop, plot_orbites
from .likelihood import compute_log_likelihood

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
    loglikelihood = compute_log_likelihood(x,
    ts,
    size,
    scale,
    fwhm,
    data,
    exclude_source=True,
    exclude_lobes=True,
    method="aperture",
    upsampling_factor=None,
    use_interp_bgnoise=False,
    r_mask=None,
    r_mask_ext=None,
    return_all=False,)

    return loglikelihood


def log_posterior(orbital_params, args):
    ts, size, scale, fwhm, data, bounds = args
    # Check if parameters are within bounds
    if not all(bound[0] <= param <= bound[1] for param, bound in zip(orbital_params, bounds)):
        return -np.inf
    #lp = log_prior(orbital_params, bounds)
    #if not np.isfinite(lp):
    #    return -np.inf
    lp = 0.
    return lp + log_likelihood(orbital_params, ts, size, scale, fwhm, data)


def reoptimize_mcmc(params, n_jobs=1, n_walkers=14, n_steps=150000, n_orbits=1000):
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
    bounds = params.grid.bounds()

    with h5py.File(f"{values_dir}/res_grid.h5") as f:
        # note: results are already sorted by decreasing SNR
        results = f["Best solutions"][:]

    n_walkers = min(n_walkers, results.shape[0])
    p0 = results[:n_walkers, 0:7].copy()
    ndim = p0.shape[1]

    # Define search range
    nbr_psf = 3.
    delta_a = nbr_psf * (bounds[0][1]-bounds[0][0]) / params.grid.limits('a')[2]
    delta_e = nbr_psf * (bounds[1][1]-bounds[1][0]) / params.grid.limits('e')[2]
    delta_t0 = nbr_psf * (bounds[2][1] - bounds[2][0]) / params.grid.limits('t0')[2]
    delta_m0 = (bounds[3][1] - bounds[3][0]) / params.grid.limits('m0')[2]
    delta_omega = nbr_psf * (bounds[4][1] - bounds[4][0]) / params.grid.limits('omega')[2]
    delta_i = nbr_psf * (bounds[5][1] - bounds[5][0]) / params.grid.limits('i')[2]
    delta_theta0 = nbr_psf * (bounds[6][1] - bounds[6][0]) / params.grid.limits('theta_0')[2]

    # Loop over each walker and each parameter to add
    # a small random perturbation to create independence between walkers
    for walker in range(n_walkers):
        for param_index, (delta, bound) in enumerate(zip(
                [delta_a, delta_e, delta_t0, delta_m0, delta_omega, delta_i, delta_theta0], bounds)):

            # Set up the initial flag for checking bounds
            in_bounds = False

            # Loop until the random perturbation is within the bounds
            while not in_bounds:
                # Generate random factor in range [-1, 1]
                random_factor = (np.random.rand() - 0.5) * 2
                perturbation = random_factor * delta

                # Add the perturbation to the parameter
                new_value = p0[5, param_index] + perturbation # This line use only one of the best output of  brute-force+gradiant (line 5)

                #new_value = p0[walker, param_index] + perturbation # This line use all the outputs of brute-force+gradiant (Doesn't work! before using this line, take into account modulos pi on Omega and omega)

                # Check if new values are in the bounds
                if bound[0] <= new_value <= bound[1]:
                    p0[walker, param_index] = new_value
                    in_bounds = True

    # mcmc configuration

    # Calculate the mean for each column
    means = np.mean(p0, axis=0)
    a_moy, e_moy, t0_moy, m0_moy, omega_moy, i_moy, theta0_moy = means
    bounds_mcmc = [
        (a_moy - delta_a, a_moy + delta_a),
        (e_moy - delta_e, e_moy + delta_e),
        (t0_moy - delta_t0, t0_moy + delta_t0),
        (m0_moy - delta_m0, m0_moy + delta_m0),
        (omega_moy - delta_omega, omega_moy + delta_omega),
        (max(0, i_moy - delta_i), i_moy + delta_i),
        (theta0_moy - delta_theta0, theta0_moy + delta_theta0)
    ]
    print('Mean orbit of the walkers:', means)
    print('bounds for the walkers:', bounds_mcmc)

    args_for_mcmc = (
        ts,
        size,
        params.scale,
        params.fwhm,
        data,
        bounds_mcmc,
    )

    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior, args=(args_for_mcmc,))  #emcee.EnsembleSampler is passing args to log_posterior

    # Run MCMC in one time
    # pos, _, _ = sampler.run_mcmc(p0, n_steps)

    # Initialize pos with p0 for the first iteration
    pos = p0
    n_check = 100
    # Open a log file to record convergence information (optional)
    with open("mcmc_log.txt", "w") as log_file:
        try:
            for i in range(0, n_steps, n_check):
                # Run MCMC for a batch of n_check steps
                pos, _, _ = sampler.run_mcmc(pos, n_check, progress=True)

                # Calculate the autocorrelation time if the iteration count is sufficient
                if sampler.iteration > 100:  # Wait until the sampler has enough samples
                    tau = sampler.get_autocorr_time(tol=0)
                    print(f"Step {sampler.iteration}: Autocorrelation time = {tau}")
                    log_file.write(f"Step {sampler.iteration}: Autocorrelation time = {tau}\n")

                    # Check if convergence criterion is met
                    if np.all(tau * 50 < sampler.iteration):
                        print("Convergence criteria met")
                        log_file.write("Convergence criteria met\n")
                        break
        except Exception as e:
            print(f"An error occurred during MCMC execution: {e}")
            log_file.write(f"An error occurred during MCMC execution: {e}\n")

    try:
        # Get the final chain of parameters
        samples = sampler.get_chain(flat=True)  # shape: (n_steps * n_walkers, n_params)
        log_probs = sampler.get_log_prob(flat=True)  # shape: (n_steps * n_walkers,)

        # Remove invalid values from log_probs
        valid_indices = np.isfinite(log_probs)  # True for finite values, False for -inf
        if not np.any(valid_indices):
            raise ValueError("All values in log_probs are invalid (e.g., -inf or NaN)")

        filtered_log_probs = log_probs[valid_indices]
        filtered_samples = samples[valid_indices]

        #print(f"Debug: n_orbits={n_orbits} (type={type(n_orbits)}), len(filtered_log_probs)={len(filtered_log_probs)}")

        n_orbits = int(n_orbits)

        if len(filtered_log_probs) < n_orbits:
            print(
                f"Warning: n_orbits ({n_orbits}) exceeds the number of valid samples ({len(filtered_log_probs)}). Adjusting n_orbits to {len(filtered_log_probs)}.")
            n_orbits = len(filtered_log_probs)


        # Sort valid log_probs and get best indices
        best_indices = np.argsort(filtered_log_probs)[-int(n_orbits):][::-1]

        # Prepare an array to store the top 100 results
        reopt_mcmc = []
        for idx in best_indices:
            # Extract parameter values for each of the top 100 samples
            a, e, t0, m0, omega, i, theta_0 = filtered_samples[idx]
            log_prob = filtered_log_probs[idx]
            reopt_mcmc.append([idx, log_prob, a, e, t0, m0, omega, i, theta_0])

        reopt_mcmc = np.array(reopt_mcmc)
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

        print("Done!")


    except ValueError as e:
        print(f"ValueError: {e}")
        log_file.write(f"ValueError: {e}\n")

    except IOError as e:
        print(f"File error: {e}")
        log_file.write(f"File error: {e}\n")

    except Exception as e:
        print(f"Unexpected error: {e}")
        log_file.write(f"Unexpected error: {e}\n")
