import numpy as np
from .orbit import orbit
from kstacker.PSF_shape_mcmc import aperture, PSF

def planet_flux_and_model(N, Signal_over_var, one_over_var, g_values, all_pixel_indices, all_mask):
    """

    Parameters
    ----------
    N : int
        total number of time steps.
    Signal_over_var : numpy.ndarray
        an array of shape (N,M,M), precomputed images for each N time step of (images-background)/sigma².
    one_over_var : numpy.ndarray
        an array of shape (N,M,M), precomputed images for each N time step of 1/sigma².
    g_values : list
        an N sized list of array, each array is a 2D zero order bessel function, centered, with the border values
        close to zero, the value is computed for all the non zero value in the all_mask variable.
    all_pixel_indices : list
        an N sized list of array, each list contain 4 array, with respectively the two first array defining the 
        y and x coordinates in the M sized image and, the two last y and x coordinates in the aperture matrice.
    all_mask : list
        an N sized list of array, each value of the array is the weigth added to the aperture mask by photutils.
        
    Description
    -----------
    compute a bessel shapped matrixs, on non zeros values of the aperture mask contained in all_mask variable
    for each N time step.
    
    Returns
    -------
    float
        planet flux factor.

    """
    numerator = 0.0
    denominator = 0.0

    for k in range(N):
        if not all_pixel_indices[k] is None:
            (y_im, x_im), (y_ap, x_ap) = all_pixel_indices[k]
    
            S_over_var = Signal_over_var[k][y_im, x_im]
            G = g_values[k][y_ap, x_ap]
            W = one_over_var[k][y_im, x_im]
            M = all_mask[k][y_ap, x_ap]
    
            numerator += np.sum(S_over_var * G * M)
            denominator += np.sum(G**2 * W * M)
    
    if numerator / denominator < 0:
        return -np.inf
    
    return numerator / denominator
 
def compute_log_likelihood(x, CstData):
    """

    Parameters
    ----------
    x : list
        a list containing a, e, t0, m0, omega, i, theta_0 value given by emcee.
    CstData : kstacker.Matrix_Likelihood.MCMCCstData
        manage the constante values needed.
        
    Description
    -----------
    for each it compute all the three value of log_likelihood_res.

    Returns
    -------
    log_likelihood : float
        log likelihood value for these x values.
    
    """
    a, e, t0, m0, omega, i, theta_0 = x
    x_kepler = orbit.project_position_full(CstData.ts, a, e, t0, m0, omega, i, theta_0)
    x_kepler *= CstData.scale
    temp_d = np.hypot(x_kepler[:, 0], x_kepler[:, 1])
    x_kepler += CstData.size // 2
    N,M = len(CstData.ts),CstData.size
    # orbitals parameter are translated to cartesians coordinates and translate to suite the matrix formatilsm
    
    one_over_var, Signal, Signal_over_var, Signal_2_over_var, log_sigma = CstData.treated_image
    all_mask, all_pixel_indices = aperture(x_kepler,N,M,CstData.fwhm,CstData.PSF_shape)
    g_values = PSF(x_kepler,N,M,CstData,all_pixel_indices,all_mask)
    # function named g(x_j - x_kepler) in the mathematical formalis
    planet_flux_value = planet_flux_and_model(N,Signal_over_var,one_over_var,g_values,all_pixel_indices,all_mask)
    # variable named f_p in the mathematical formalism
    
    if (np.any(np.array(temp_d) <= CstData.r_mask)) or (np.any(np.array(temp_d) >= CstData.r_mask_ext) or np.isinf(planet_flux_value)):
        return -np.inf
        # negative flux can't relate to the presence of a planet = no planet here
        # can't mesure into the mask
    else:
        som = 0
        for k in range(N):
            if all_mask[k] is None:
                som += 0
            else:
                # Extract the indices for optimization
                image_size_y, image_size_x = all_pixel_indices[k][0]
                resize_y, resize_x = all_pixel_indices[k][1]
                
                S = Signal[k][image_size_y, image_size_x]
                G = g_values[k][resize_y, resize_x]
                Mask = all_mask[k][resize_y, resize_x]
                W = one_over_var[k][image_size_y, image_size_x]
            
                # Calculate som (the sum of squared differences)
                numerator = (planet_flux_value * G)**2 - 2 * S * planet_flux_value * G
                som += np.sum(numerator * W * Mask)
        
        if CstData.cste_part_Likelihood is None :
            cst_part = -N * M**2 / 2 * np.log(2 * np.pi) - np.nansum(log_sigma) - 0.5*np.nansum(Signal_2_over_var)
            CstData.cste_part_Likelihood = cst_part
        else:
            cst_part = CstData.cste_part_Likelihood
            
        log_likelihood = cst_part -.5*som
        
        return log_likelihood

def log_likelihood(x, CstData):
    """

    Parameters
    ----------
    x : list
        a list containing a, e, t0, m0, omega, i, theta_0 value given by emcee.
    CstData : kstacker.Matrix_Likelihood.MCMCCstData
        manage the constante values needed.
    Returns
    -------
    log_p : float
        log likelihood value for these x values.

    """
    param_names = ["a", "e", "t0", "m0", "omega", "i", "theta_0"] # all the name of the parameters
    x_complete = [0] * 7 # initialise the final paramters variable
    
    if CstData.fixed_params is None: # all the parameters are free
        unfixed_param_indices = range(7)
        x_complete = list(x)
    else:
        unfixed_param_indices = [i for i, name in enumerate(param_names) if name not in CstData.fixed_params] # get the indices of the unfixed variables
        for x_val, i in zip(x, unfixed_param_indices): # save the unfixed variable into the final orbital parameters variable
            x_complete[i] = x_val
        for name, val in CstData.fixed_params.items(): # save the fixed variable into the final orbital parameters variable
            x_complete[param_names.index(name)] = val

    log_p = compute_log_likelihood(x_complete, CstData)
    return log_p