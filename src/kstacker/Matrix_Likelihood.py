import numpy as np
from .orbit import orbit
from kstacker.PSF_shape_mcmc import aperture, PSF

def planet_flux(N, Signal_over_var, one_over_var, g_values, all_pixel_indices, all_apperture_mask):
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
        close to zero, the value is computed for all the non zero value in the all_apperture_mask variable.
    all_pixel_indices : list
        an N sized list of array, each list contain 4 array, with respectively the two first array defining the 
        y and x coordinates in the M sized image and, the two last y and x coordinates in the aperture matrice.
    all_apperture_mask : list
        an N sized list of array, each value of the array is the weigth added to the aperture mask by photutils.
        
    Description
    -----------
    compute a bessel shapped matrixs, on non zeros values of the aperture mask contained in all_apperture_mask variable
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
            image_y, image_x = all_pixel_indices[k][0]
            apperture_y, apperture_x = all_pixel_indices[k][1]
    
            S_over_var = Signal_over_var[k][image_y, image_x]
            G = g_values[k][apperture_y, apperture_x]
            inv_var = one_over_var[k][image_y, image_x]
            apperture_mask = all_apperture_mask[k][apperture_y, apperture_x]
            # apperture_mask is photutils mask, it is needed to extract the flux
    
            numerator += np.sum(S_over_var * G * apperture_mask)
            denominator += np.sum(G**2 * inv_var * apperture_mask)
    
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
    all_apperture_mask, all_pixel_indices = aperture(x_kepler,N,M,CstData.fwhm,CstData.PSF_shape)
    g_values = PSF(x_kepler,N,M,CstData,all_pixel_indices,all_apperture_mask)
    # function named g(x_j - x_kepler) in the mathematical formalis
    planet_flux_value = planet_flux(N,Signal_over_var,one_over_var,g_values,all_pixel_indices,all_apperture_mask)
    # variable named f_p in the mathematical formalism
    
    if (np.any(np.array(temp_d) <= CstData.r_mask)) or (np.any(np.array(temp_d) >= CstData.r_mask_ext)):
        return -np.inf
        # can't mesure into the mask
    
    if  np.isinf(planet_flux_value):
        return -np.inf
        # negative flux can't relate to the presence of a planet = no planet here
    # T_1 is the constant term, T_2 is the non-constant term
    else:
        T_2 = 0
        for k in range(N):
            if all_apperture_mask[k] is None:
                T_2 += 0
            else:
                # Extract the indices for optimization
                image_y, image_x = all_pixel_indices[k][0]
                apperture_y, apperture_x = all_pixel_indices[k][1]
                
                S = Signal[k][image_y, image_x]
                G = g_values[k][apperture_y, apperture_x]
                apperture_mask = all_apperture_mask[k][apperture_y, apperture_x]
                # apperture_mask is photutils mask, it is needed to extract the flux
                inv_var = one_over_var[k][image_y, image_x]
            
                # Calculate som (the sum of squared differences)
                numerator = (planet_flux_value * G)**2 - 2 * S * planet_flux_value * G
                T_2 += -.5*np.sum(numerator * inv_var * apperture_mask)
        
        if CstData.cste_part_Likelihood is None :
            T_1 = -N * M**2 / 2 * np.log(2 * np.pi) - np.nansum(log_sigma) - 0.5*np.nansum(Signal_2_over_var)
            CstData.cste_part_Likelihood = T_1
        else:
            T_1 = CstData.cste_part_Likelihood
            
        log_likelihood = T_1 + T_2
        
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
    log_L : float
        log likelihood value for these x values.

    """
    param_names = ["a", "e", "t0", "m0", "omega", "i", "theta_0"] # all the name of the parameters
    x_complete = [0] * 7 # initialise the final paramters variable
    
    if CstData.fixed_params is None: # all the parameters are free
        x_complete = list(x)
    else:
        unfixed_param_indices = [i for i, name in enumerate(param_names) if name not in CstData.fixed_params] # get the indices of the unfixed variables
        for x_val, i in zip(x, unfixed_param_indices): # save the unfixed variable into the final orbital parameters variable
            x_complete[i] = x_val
        for name, val in CstData.fixed_params.items(): # save the fixed variable into the final orbital parameters variable
            x_complete[param_names.index(name)] = val

    log_L = compute_log_likelihood(x_complete, CstData)
    return log_L