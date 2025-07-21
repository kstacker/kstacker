import numpy as np
import os
import emcee
import h5py
import time

from pathlib import Path
from astropy.io import ascii, fits
from multiprocessing import Pool
from joblib import Parallel, delayed

from kstacker.orbit import corner_plots_mcmc
from kstacker.mcmc_reoptimization import make_plots
from kstacker.mcmc_starting_pos import read_starting_file
from kstacker.PSF_shape_mcmc import precompute_bessel_lookup
from kstacker.Matrix_Likelihood import log_likelihood, compute_log_likelihood
from kstacker.orbit import plot_converge_points_map

class MCMCCstData:
    """
    Class to manage the constante parameters used in the MCMC.
    ts : list
        the 
    size : int
        numbers of images in the data set
    scale : float
        scale of one pixel
    fwhm : float
        radius of the aperture studied
    bounds : list
        searching bounds define on all 7 orbital param
    treated_image : numpy.ndarray
        all 5 pretreated images
    r_mask : float
        value of the inner radius of the mask 
    r_mask_ext: float
        value of the outer radius of the mask 
    r_vals : list
        x value for the bessel shape.
    j0_vals : list
        y value for the bessel shape.
    fixed_params : dict 
        fixe some orbital params, typical shape is {"a":50,"e":0.1}, the orbital param names must be : ["a", "e", "t0", "m0", "omega", "i", "theta_0"]. The default is None.
    cste_part_Likelihood : float
        store the constant component of the likelihood.
    
    Attributes
    ----------
    """
    def __init__(self):
        self.ts = None
        self.size = None
        self.scale = None
        self.fwhm = None
        self.bounds = None
        self.treated_image = None
        self.r_mask = None
        self.r_mask_ext = None
        self.r_vals = None
        self.j0_vals = None
        self.fixed_params = None
        self.PSF_shape = None
        self.cste_part_Likelihood = None
        self.images = None
        
def extract_preteated_image(profile_dir, N, M):
    """
    Parameters
    ----------
    profile_dir : str
        Path to the directory containing pre-treated FITS files.
    N : int
        Total number of time steps.
    M : int
        Number of pixels on each side of the square image.
        
    Returns
    -------
    one_over_var : np.ndarray
        Shape (N, M, M) - 1/σ² maps.
    Signal : np.ndarray
        Shape (N, M, M) - images minus background.
    Signal_over_var : np.ndarray
        Shape (N, M, M) - (image-background)/σ².
    Signal_2_over_var : np.ndarray
        Shape (N, M, M) - (image-background)²/σ².
    log_sigma : np.ndarray
        Shape (N, M, M) - log(σ).
    """
    one_over_var = np.zeros((N, M, M))
    Signal = np.zeros((N, M, M))
    Signal_over_var = np.zeros((N, M, M))
    Signal_2_over_var = np.zeros((N, M, M))
    log_sigma = np.zeros((N, M, M))
    
    for k in range(N):
        one_over_var[k] = fits.getdata(f"{profile_dir}/one_over_var_{k}.fits")
        Signal[k] = fits.getdata(f"{profile_dir}/Signal_prof_{k}.fits")
        Signal_over_var[k] = fits.getdata(f"{profile_dir}/Signal_over_variance_prof_{k}.fits")
        Signal_2_over_var[k] = fits.getdata(f"{profile_dir}/Signal_square_over_variance_prof_{k}.fits")
        log_sigma[k] = fits.getdata(f"{profile_dir}/log_sigma_prof_{k}.fits")
        
    return one_over_var, Signal, Signal_over_var, Signal_2_over_var, log_sigma

def log_posterior(orbital_params):
    """

    Parameters
    ----------
    orbital_params : list
        a list containing a, e, t0, m0, omega, i, theta_0 value given by emcee.

    Returns
    -------
    float
       log posterior value for these x values.

    """
    # get the prior value
    lp = log_prior(orbital_params,CstData.bounds)
    if np.isinf(lp):
        return -np.inf
    # get the likelihood value
    log_likelihood_value = log_likelihood(orbital_params, CstData)
    if np.isinf(log_likelihood_value):
        return -np.inf
    else:
        return lp + log_likelihood_value

def log_prior(orbital_params,bounds):
    """

    Parameters
    ----------
    orbital_params : list
        the value of the 7 orbital params ["a", "e", "t0", "m0", "omega", "i", "theta_0"].
    bounds : list
        the value of the 7 orbital params bounds in the shape [(lower 'a' bound, upper 'a' bound),...,((lower 'theta_0' bound, upper 'theta_0' bound)].

    Returns
    -------
    float
        log prior values value.

    """
    if not all(bound[0] <= param <= bound[1] for param, bound in zip(orbital_params, bounds)):
        return -np.inf
    return 0

def compute_mcmc_matrix(params, n_jobs=1, n_walkers=28, n_steps=100000, n_orbits=1000, n_check=1000, fixed_params=None, nbr_psf=1., init_pos_precomputed=False, PSF_shape="Bessel"):
    """
    
    Parameters
    ----------
    params : kstacker.utils.Params
        get information on the data set studied.
    n_jobs : int, optional
        fixe the number of core assigned to the task. The default is 1.
    n_walkers : int, optional
        fixe the emcee walkers assigned to the task. The default is 28.
    n_steps : int, optional
        fixe the max emcee iteration allowed. The default is 100000.
    n_orbits : int, optional
        numbers of orbit output in the results file. The default is 1000.
    n_check : int, optional
        fixe the numbers of step between each autocorrelation check. The default is 1000.
    fixed_params : dict, optional
        fixe some orbital params, typical shape is {"a":50,"e":0.1}, the orbital param names must be : ["a", "e", "t0", "m0", "omega", "i", "theta_0"]. The default is None.
    nbr_psf : float, optional
        fixe the searching range of the walkers, the bigger is this value the larger is the searching area. The default is 1..
    init_pos_precomputed : bool, optional
        If True use precomputed intial position and bound, must be estimate before. The default is False.
    PSF_shape : string, optional
        name of the used PSF, possible value "Bessel", "Circle". The default is "Bessel".
        
    Returns
    -------
    None.

    """
    # init directories
    profile_dir = params.get_path("profile_dir")
    values_dir = params.get_path("values_dir")
    os.makedirs(f"{values_dir}/fin_fits", exist_ok=True)
    os.makedirs(f"{values_dir}/fin_png", exist_ok=True)
    os.makedirs(f"{values_dir}/orbites", exist_ok=True)
    os.makedirs(f"{values_dir}/single", exist_ok=True)
    
    # get data from the the yml file
    ts = np.array(params.get_ts())
    size = params.n
    scale = params.scale
    fwhm = params.fwhm
    data = params.load_data(method="aperture")
    images = data['images']
    N,M,_ = np.shape(images)
    
    # get the data images petreated
    treated_image = extract_preteated_image(profile_dir,N,M)
    
    if init_pos_precomputed:
        # us pre-initialized initial position and bounds
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
            # delete from unfixed parmaters table the fixed values
            final_param_names = [key for key in param_names if key not in fixed_params]
            delta_param = {key: None for key in final_param_names}
            
            unfixed_param_indices = [index for index, keys in enumerate(param_names) if keys not in fixed_params]
        
        p0 = results[:n_walkers, unfixed_param_indices].copy()
        # initial walkers positions
        
        for i in range(len(delta_param)):
            # define the the bounds for the research of the MCMC, could be deleted when after the reparameterization of Kepler's equations
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
                    new_value = p0[3, i] + perturbation # This line use only one of the best output of  brute-force+gradiant (line 5)
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
            # initialize the real bound values for each walkers. Could be deleted when after the reparameterization of Kepler's equations
            index = param_names.index(final_param_names[i])
            bounds.append((means[i] - delta_param[final_param_names[i]], means[i] + delta_param[final_param_names[i]]))
        
    pos = np.array(p0)
    ndim = len(bounds)
    
    # pre run the PSF shape (bessel) to note get to run it every time and just interpolate value
    r_vals, j0_vals = precompute_bessel_lookup()
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior)
    r_mask = 30
    r_mask_ext = size//2
    
    # initialize the globals value passed to emcee (global values are necessary to allow multiprocessing)
    global CstData
    
    CstData = MCMCCstData()
    CstData.ts = ts
    CstData.size = size
    CstData.scale = scale
    CstData.fwhm = fwhm
    CstData.bounds = bounds
    CstData.treated_image = treated_image
    CstData.r_mask = r_mask
    CstData.r_mask_ext = r_mask_ext
    CstData.r_vals = r_vals
    CstData.j0_vals = j0_vals
    CstData.fixed_params = fixed_params
    CstData.PSF_shape = PSF_shape
    CstData.images = images
    
    written = False
    
    start = time.time()
    
    # log file to get evolution of the autocorrelation time trough execution
    log_path = Path(f"{values_dir}/mcmc_log.txt")
    log_path.write_text("")
    
    # execution of the mcmc with multiprocessing method
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
        # try:
        #     for i in range(0, n_steps, n_check):
        #         pos, _, _ = sampler.run_mcmc(pos, n_check, progress=True)

        #         if sampler.iteration > 6*n_check:
        #             tau = sampler.get_autocorr_time(tol=0)
        #             with open(log_path, "a") as f:
        #                 f.write(f"Step {sampler.iteration}: Autocorrelation time = {tau}")
        #                 f.write(f"Step {sampler.iteration}: tau*50/iter = {(tau * 50)/sampler.iteration}\n")
        #                 f.write(f"Step {sampler.iteration}: mean acceptance = {np.mean(sampler.acceptance_fraction)}\n")
                        
        #                 if np.all((tau * 50)/sampler.iteration < 1):
        #                     end = time.time()
        #                     written = True
        #                     f.write("Convergence criteria met\n")
        #                     f.write(f"Time taken : {end-start}\n")
        #                     break
                        
        #     with open(log_path, "a") as f:
        #         if not written:
        #             end = time.time()
        #             f.write("Convergence criteria not met\n")
        #             f.write(f"Time taken : {end-start}\n")

        # except Exception as e:
        #     with open(log_path, "a") as f:
        #         f.write(f"An error occurred during MCMC execution: {e}\n")
        
        # try:
        #     # Get the final chain of parameters
        #     samples = sampler.get_chain(flat=True)
        #     log_probs = sampler.get_log_prob(flat=True)
        
        #     # Remove invalid values from log_probs, get only the unique values
        #     unique_samples, unique_indices = np.unique(samples, axis=0, return_index=True)
        #     unique_log_probs = log_probs[unique_indices]
            
        #     # Remove invalid values from log_probs, delete the non finite result values 
        #     valid_indices = np.isfinite(unique_log_probs)
        #     filtered_samples = unique_samples[valid_indices]
        #     filtered_log_probs = unique_log_probs[valid_indices]
            
        #     # sort the results by Likelihood values
        #     sorted_indices = np.argsort(-filtered_log_probs)
        #     final_samples = filtered_samples[sorted_indices]
        #     final_log_probs = filtered_log_probs[sorted_indices]
            
        #     # Prepare an array to store the top 1000 results
        #     reopt_mcmc = []
        #     for idx in sorted_indices:
        #         # Extract parameter values for each of the top 1000 samples
        #         a, e, t0, m0, omega, i, theta_0 = final_samples[idx]
        #         log_prob = final_log_probs[idx]
        #         reopt_mcmc.append([idx, log_prob, a, e, t0, m0, omega, i, theta_0])
            
        #     reopt_mcmc = np.array(reopt_mcmc[:n_orbits])
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
            
        #     corner_plots_mcmc(params, nbins=5)
        
        # except ValueError as e:
        #     with open(log_path, "a") as f: f.write(f"ValueError: {e}\n")
        
        # except IOError as e:
        #     with open(log_path, "a") as f: f.write(f"File error: {e}\n")
        
        # except Exception as e:
        #     with open(log_path, "a") as f: f.write(f"Unexpected error: {e}\n")