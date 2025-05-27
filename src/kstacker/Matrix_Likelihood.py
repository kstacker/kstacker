import numpy as np
from .orbit import orbit
from photutils import CircularAperture
from scipy.special import j0
from astropy.io import fits

class MCMCState:
    """
    Class to manage the state of the MCMC walkers inside of the probabilities function.

    Attributes
    ----------
    cached_terms : list of list
        Stores cached likelihood terms for each walker, standard shape will be [log_likelihood_res, all_mask, all_pixel_indices]
        with log_likelihood_res a table with three part of the log likelihood somme, all_mask a table for each time of the mask 
        aperture matrix defined with photutils, all_pixel_indices for the coordinate of all non zero pixel in all_mask and the
        same pixel into the studied image. the index 0 values for each walkers index is modified each step that the mcmc goes
        to this walkers, the 1 index is on the oposite the save of the last accepted value for the walker, if no value were
        accepted the it the first value for this walker.
    walker_index : int
        Tracks the current walker index.
    n_walkers : int
        Number of walkers in the MCMC sampler.
    sampler : emcee.EnsembleSampler or None
        The MCMC sampler, associated after initialization.

    Methods
    -------
    set_sampler(sampler)
        Associates the sampler after its creation with emcee.EnsembleSampler.
    """
    def __init__(self, n_walkers):
        """
        Parameters
        ----------
        n_walkers : int
            The number of walkers in the MCMC sampler.
        """
        self.cached_terms = [[None, None] for _ in range(n_walkers)]
        self.walker_index = -1
        self.n_walkers = n_walkers
        self.sampler = None
        self.log_likelihood_rest = [[None, None] for _ in range(n_walkers)] # je pense que c'est pas utilisé
        self.r_vals = None
        self.j0_vals = None
        self.fixed_params = None

    def set_sampler(self, sampler):
        """
        Associates the sampler after its creation.

        Parameters
        ----------
        sampler : emcee.EnsembleSampler
            The MCMC sampler that will be linked to this state.
        """
        self.sampler = sampler
        
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

def photometry_extract_matrice(position,N,M,radius):
    """

    Parameters
    ----------
    position : numpy.ndarray
        table of pair value of x,y position, with a shape equal to (N,2).
    N : int
        total number of time steps.
    M : int
        number of pixel on each side of the matrix.
    radius : int
        radius of the studied apeture, defined by fwhm/2 value.
        
    Description
    -----------
    for each N step, compute a mask aperture on a M*M sized sized array, return this mask and the coordinates
    values of this mask.

    Returns
    -------
    all_mask : list
        an N sized list of array, each value of the array is the weigth added to the aperture mask by photutils.
    all_pixel_indices : list
        an N sized list of array, each list contain 4 array, with respectively the two first array defining the 
        y and x coordinates in the M sized image and, the two last y and x coordinates in the aperture matrice.

    """
    all_mask = []
    all_pixel_indices = []
    # initialize the output values
    for k in range (N):
        ypix, xpix  = position[k]
    
        xpix = xpix - 0.5
        ypix = ypix - 0.5
        # shift of xpix and ypix due to photutils 
        
        aperture = CircularAperture([xpix, ypix], r=radius)
        mask = aperture.to_mask(method='exact')
        # output mask matrix
        
        # Generate the mask image cropped to MxM
        image_mask = mask.to_image((M, M))
        
        if image_mask is None:
            all_mask.append(None)
            all_pixel_indices.append(None)
            
        else:
            # Get non-zero indices in the image (cropped version)
            y_img, x_img = np.nonzero(image_mask)
    
            # Get corresponding indices in mask.data
            # mask.bbox gives the (xmin, xmax, ymin, ymax) of mask placement on image
            bbox = mask.bbox
            y0, x0 = bbox.iymin, bbox.ixmin  # top-left corner of mask in image
            y_mask = y_img - y0
            x_mask = x_img - x0
    
            # Save data
            all_mask.append(mask.data)
            all_pixel_indices.append(((y_img, x_img), (y_mask, x_mask)))
    
    return all_mask, all_pixel_indices

def precompute_bessel_lookup(r_max=2.41, num_points=1000000):
    """Precompute Bessel function lookup table"""
    r_vals = np.linspace(0, r_max, num_points)
    j0_vals = j0(r_vals)
    return r_vals, j0_vals

def bessel_aperture(x_kepler, N, M, fwhm, all_pixel_indices, all_mask, r_vals, j0_vals):
    """

    Parameters
    ----------
    x_kepler : numpy.ndarray
        table of pair value of x,y position, with a shape equal to (N,2).
    N : int
        total number of time steps.
    M : int
        number of pixel on each side of the matrix.
    fwhm : float
        diameter of the studied aperture.
    all_mask : list
        an N sized list of array, each value of the array is the weigth added to the aperture mask by photutils.
    all_pixel_indices : list
        an N sized list of array, each list contain 4 array, with respectively the two first array defining the 
        y and x coordinates in the M sized image and, the two last y and x coordinates in the aperture matrice.
        
    Description
    -----------
    compute a bessel shapped matrixs, on non zeros values of the aperture mask contained in all_mask variable
    for each N time step.

    Returns
    -------
    all_bessel_value : list
        an N sized list of array, each array is a 2D zero order bessel function, centered, with the border values
        close to zero, the value is computed for all the non zero value in the all_mask variable.
    
    """
    all_bessel_value = []
    factor = 2.41 / fwhm

    for k in range(N):
        bessel_value = np.zeros_like(all_mask[k])
        
        if all_mask[k] is None:
            all_bessel_value.append(None)
            
        else:
            # Récupère les indices (y, x) dans la matrice de l'ouverture
            y_ap, x_ap = all_pixel_indices[k][1]
            Y_size, X_size = all_mask[k].shape
    
            # Coordonnées centrées (subpixel shift inclus : +0.5)
            dx = x_ap - X_size / 2 + 0.5
            dy = y_ap - Y_size / 2 + 0.5
    
            r = np.sqrt(dx**2 + dy**2) * factor
    
            # Interpolation rapide via np.interp
            bessel_vals = np.interp(r, r_vals, j0_vals)
    
            # Remplissage direct dans le tableau
            bessel_value[y_ap, x_ap] = bessel_vals

            all_bessel_value.append(bessel_value)

    return all_bessel_value

def planet_flux_and_model(
        N,
        Signal_over_var,
        one_over_var,
        g_values,
        all_pixel_indices,
        all_mask):
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

    if numerator == 0:
        return 0
    
    if numerator / denominator < 0:
        return -np.inf
    
    return numerator / denominator
 
def compute_log_likelihood(
        x,
        state,
        ts,
        size,
        scale,
        fwhm,
        treated_image):
    """

    Parameters
    ----------
    x : list
        a list containing a, e, t0, m0, omega, i, theta_0 value given by emcee.
    ts : list
        all capture time given by the initiating files.
    size : int
        size of each images.
    scale : float
        scale factor.
    fwhm : float
        diameter of the studied aperture.
    treated_image : numpy.ndarray
        pretreated images by noise_profile.compute_mcmc_noise_signal_profil and loaded with extract_preteated_image.
        
    Description
    -----------
    for each it compute all the three value of log_likelihood_res.

    Returns
    -------
    log_likelihood_res : list
        the three part of log likelihood somme, with log_likelihood_res[0] the constante part, log_likelihood_res[1]
        the sigma part without R(theta) and log_likelihood_res[2] the R(theta) part.
    float
        log likelihood value for these x values.
    all_mask : list
        an N sized list of array, each value of the array is the weigth added to the aperture mask by photutils.
    all_pixel_indices : list
        an N sized list of array, each list contain 4 array, with respectively the two first array defining the 
        y and x coordinates in the M sized image and, the two last y and x coordinates in the aperture matrice.
    
    """
    a, e, t0, m0, omega, i, theta_0 = x
    x_kepler = orbit.project_position_full(ts, a, e, t0, m0, omega, i, theta_0)
    x_kepler *= scale
    x_kepler += size // 2
    # orbitals parameter are translated to cartesians coordinates and translate to suite the matrix formatilsm
    
    N,M = len(ts),size
        
    one_over_var, Signal, Signal_over_var, Signal_2_over_var, log_sigma = treated_image
    
    log_likelihood_res = [-N * M**2 / 2 * np.log(2 * np.pi) - np.nansum(log_sigma),-np.inf, -np.inf]
    # first value of the table of likelihood
    
    all_mask, all_pixel_indices = photometry_extract_matrice(x_kepler,N,M,fwhm)
    
    g_values = bessel_aperture(x,N,M,fwhm,all_pixel_indices,all_mask, r_vals=state.r_vals, j0_vals=state.j0_vals)
    # function named g(x_j - x_kepler) in the mathematical formalism

    planet_flux_value = planet_flux_and_model(N,Signal_over_var,one_over_var,g_values,all_pixel_indices,all_mask)
    # variable named f_p in the mathematical formalism
    
    if np.isinf(planet_flux_value):
        return log_likelihood_res,-np.inf,all_mask,all_pixel_indices
        # negative lux can't relate to the presence of a planet = no planet here
    
    else:
        som_2 = 0
        som_3 = 0
        
        matrice_wo_aperture = np.ones((N, M, M))
        
        for k in range(N):
            if all_mask[k] is None:
                som_2 += 0
                som_3 += np.nansum(Signal_2_over_var[k])
            
            else:
                mask_k = all_mask[k]
                signal_k = Signal[k]
                g_values_k = g_values[k]
                one_over_var_k = one_over_var[k]
                signal_2_over_var_k = Signal_2_over_var[k]
                
                # Extract the indices for optimization
                resize_y = all_pixel_indices[k][1][0]
                resize_x = all_pixel_indices[k][1][1]
                image_size_y = all_pixel_indices[k][0][0]
                image_size_x = all_pixel_indices[k][0][1]
            
                # Calculate som_2 (the sum of squared differences)
                delta = signal_k[image_size_y, image_size_x] - planet_flux_value * g_values_k[resize_y, resize_x]
                som_2 += np.sum(delta**2 * one_over_var_k[image_size_y, image_size_x] * mask_k[resize_y, resize_x])
            
                # Update matrice_wo_aperture
                matrice_wo_aperture[k, image_size_y, image_size_x] -= mask_k[resize_y, resize_x]
                
                # Calculate som_3 (the sum of Signal_2_over_var multiplied by matrice_wo_aperture)
                som_3 += np.nansum(signal_2_over_var_k * matrice_wo_aperture[k])
        
        log_likelihood_res[1] = -.5*som_2
        log_likelihood_res[2] = -.5*som_3
        return log_likelihood_res,np.sum(log_likelihood_res),all_mask,all_pixel_indices
    
def R_theta(prev_all_mask,prev_all_pixel_indices,all_mask,all_pixel_indices,Signal_2_over_var):
    """

    Parameters
    ----------
    prev_all_mask : list
        same as all_mask for previous mcmc time step.
    prev_all_pixel_indices : list
        same as all_pixel_indices for previous mcmc time step.
    all_mask : list
        an N sized list of array, each value of the array is the weigth added to the aperture mask by photutils.
    all_pixel_indices : list
        an N sized list of array, each list contain 4 array, with respectively the two first array defining the 
        y and x coordinates in the M sized image and, the two last y and x coordinates in the aperture matrice.
    Signal_2_over_var : numpy.ndarray
        an array of shape (N,M,M), precomputed images for each N time step of (images-background)²/sigma².
        
    Description
    -----------
    compute the somme over each apeture to get R(theta+1) from R(theta).

    Returns
    -------
    new_step_rest : float
        present mcmc compute step of (images-background)²/sigma² into the associated mask.
    old_step_rest : float
        last accepted mcmc compute step of (images-background)²/sigma² into the associated mask.

    """
    new_step_rest = 0
    old_step_rest = 0
    for k in range(len(Signal_2_over_var)):
        # New
        if all_mask[k] is None:
            new_step_rest += 0
        else:
            (y_im_new, x_im_new), (y_ap_new, x_ap_new) = all_pixel_indices[k]
            new_step_rest += np.dot(
                Signal_2_over_var[k][y_im_new, x_im_new] ,
                all_mask[k][y_ap_new, x_ap_new]
            )
        
        #Old
        if prev_all_mask[k] is None:
            old_step_rest += 0
        else:
            (y_im_old, x_im_old), (y_ap_old, x_ap_old) = prev_all_pixel_indices[k]
            old_step_rest += np.dot(
                Signal_2_over_var[k][y_im_old, x_im_old] ,
                prev_all_mask[k][y_ap_old, x_ap_old]
            )
    return new_step_rest,old_step_rest
    
def recompute_log_likelihood(
        x,
        state,
        ts,
        size,
        scale,
        fwhm,
        treated_image):
    """

    Parameters
    ----------
    x : list
        a list containing a, e, t0, m0, omega, i, theta_0 value given by emcee.
    state : kstacker.Matrix_Likelihood.MCMCState
        manage the state of the sampler value.
    ts : list
        all capture time given by the initiating files.
    size : int
        size of each images.
    scale : float
        scale factor.
    fwhm : float
        diameter of the studied aperture.
    treated_image : numpy.ndarray
        pretreated images by noise_profile.compute_mcmc_noise_signal_profil and loaded with extract_preteated_image.
        
    Description
    -----------
    for each it compute all the three value of log_likelihood_res[1] and log_likelihood_res[2], log_likelihood_res[0]
    already computed in a previous step.
        
    Returns
    -------
    log_likelihood_res : list
        the three part of log likelihood somme, with log_likelihood_res[0] the constante part, log_likelihood_res[1]
        the sigma part without R(theta) and log_likelihood_res[2] the R(theta) part.
    float
        log likelihood value for these x values.
    all_mask : list
        an N sized list of array, each value of the array is the weigth added to the aperture mask by photutils.
    all_pixel_indices : list
        an N sized list of array, each list contain 4 array, with respectively the two first array defining the 
        y and x coordinates in the M sized image and, the two last y and x coordinates in the aperture matrice.
    

    """
    cached_items = state.cached_terms
    walker_id = state.walker_index
    
    a, e, t0, m0, omega, i, theta_0 = x
    x_kepler = orbit.project_position_full(ts, a, e, t0, m0, omega, i, theta_0)
    x_kepler *= scale
    x_kepler += size // 2
    
    log_likelihood_res, prev_all_mask, prev_all_pixel_indices = cached_items[walker_id][1]
    
    N,M = len(ts),size
        
    one_over_var, Signal, Signal_over_var, Signal_2_over_var, _ = treated_image
    
    all_mask, all_pixel_indices = photometry_extract_matrice(x_kepler,N,M,fwhm)
    
    g_values = bessel_aperture(x,N,M,fwhm,all_pixel_indices,all_mask, r_vals=state.r_vals, j0_vals=state.j0_vals)
        
    planet_flux_value = planet_flux_and_model(N,Signal_over_var,one_over_var,g_values,all_pixel_indices,all_mask)
    
    if np.isinf(planet_flux_value):
        return log_likelihood_res,-np.inf,prev_all_mask,prev_all_pixel_indices
    
    else:
        som_2 = 0
        som_2 = 0.0
        for k in range(N):
            if all_mask[k] is None:
                som_2 += 0
            
            else:
                image_size_y, image_size_x = all_pixel_indices[k][0]
                resize_y, resize_x = all_pixel_indices[k][1]
                
                S = Signal[k][image_size_y, image_size_x]
                G = g_values[k][resize_y, resize_x]
                M = all_mask[k][resize_y, resize_x]
                W = one_over_var[k][image_size_y, image_size_x]
                
                delta = S - planet_flux_value * G
                som_2 += np.sum(delta**2 * W * M)
        
        new_step_rest,old_step_rest = R_theta(prev_all_mask, prev_all_pixel_indices, all_mask, all_pixel_indices, Signal_2_over_var)    
        
        # C, L, R = log_likelihood_res
        
        log_likelihood_res[1] = -.5*som_2
        log_likelihood_res[2] = -.5*(-2*log_likelihood_res[2]+old_step_rest-new_step_rest)
        
        # A = R - log_likelihood_res[2]
        # B = log_likelihood_res[1] - L
        
        # if abs(A/abs(A+B))>1:
        #     print(A,B)
        #     print(C,L,R)
        #     print(log_likelihood_res[0],log_likelihood_res[1],log_likelihood_res[2])
        #     print(A/abs(A+B))
        return log_likelihood_res,np.sum(log_likelihood_res),all_mask,all_pixel_indices

def log_likelihood(
        x, 
        ts, 
        size, 
        scale, 
        fwhm, 
        fast, 
        state, 
        treated_image):
    """

    Parameters
    ----------
    x : list
        a list containing a, e, t0, m0, omega, i, theta_0 value given by emcee.
    ts : list
        all capture time given by the initiating files.
    size : int
        size of each images.
    scale : float
        scale factor.
    fwhm : float
        diameter of the studied aperture.
    fast : bool
        Boolean value, if True the log likelihood will be used the mathematical trick.
    state : kstacker.Matrix_Likelihood.MCMCState
        manage the state of the sampler value.
    treated_image : numpy.ndarray
        pretreated images by noise_profile.compute_mcmc_noise_signal_profil and loaded with extract_preteated_image.

    Returns
    -------
    log_p : float
        log likelihood value for these x values.

    """
    param_names = ["a", "e", "t0", "m0", "omega", "i", "theta_0"] # all the name of the parameters
    x_complete = [0] * 7 # initialise the final paramters variable
    
    fixed_params = state.fixed_params # info about the fixed variable
    
    if fixed_params is None: # all the parameters are free
        unfixed_param_indices = range(7)
        x_complete = list(x)
    else:
        unfixed_param_indices = [i for i, name in enumerate(param_names) if name not in fixed_params] # get the indices of the unfixed variables
        for x_val, i in zip(x, unfixed_param_indices): # save the unfixed variable into the final orbital parameters variable
            x_complete[i] = x_val
        for name, val in fixed_params.items(): # save the fixed variable into the final orbital parameters variable
            x_complete[param_names.index(name)] = val

    walker_id = state.walker_index
    if fast:
        if (state.cached_terms[walker_id][0] is None):
            log_likelihood_res, log_p, all_mask, all_pixel_indices = compute_log_likelihood(x_complete, state, ts, size, scale, fwhm, treated_image)
            if not np.isinf(log_p):
                state.cached_terms[walker_id][0] = [log_likelihood_res, all_mask, all_pixel_indices]
                state.cached_terms[walker_id][1] = [log_likelihood_res, all_mask, all_pixel_indices]
        else :
            log_likelihood_res, log_p, all_mask, all_pixel_indices = recompute_log_likelihood(x_complete, state, ts, size, scale, fwhm, treated_image)
            state.cached_terms[walker_id][1] = [log_likelihood_res, all_mask, all_pixel_indices]
    else:
        _, log_p, _, _ = compute_log_likelihood(x_complete, state, ts, size, scale, fwhm, treated_image)
    return log_p