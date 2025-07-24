import numpy as np
from photutils.aperture import CircularAperture
from scipy.special import j0

def precompute_bessel_lookup(r_max=2, num_points=1000000):
    """
    
    Parameters
    ----------
    r_max : float, optional
        max value of the x value for the bessel shape. The default is 2.
    num_points : int, optional
        total number of points used to estimate the bessel shape. The default is 1000000.
        
    Description
    -----------
    Precompute Bessel function lookup table

    Returns
    -------
    r_vals : list
        x value for the bessel shape.
    j0_vals : list
        y value for the bessel shape.

    """
    r_vals = np.linspace(0, r_max, num_points)
    j0_vals = j0(r_vals)
    j0_vals = j0_vals / ((r_vals[1]-r_vals[0]) * np.sum(j0_vals * r_vals)) *1 / (2*np.pi)
    return r_vals, j0_vals

def aperture(position,N,M,radius,PSF_shape):
    """

    Parameters
    ----------
    position : numpy.ndarray
        table of pair value of x,y position, with a shape equal to (N,2).
    N : int
        total number of time steps.
    M : int
        number of pixel on each side of the matrix.
    radius : float
        radius of the studied apeture, defined by fwhm/2 value.
        
    Description
    -----------
    choose the appropriate aperture for each PSF shape.

    Returns
    -------
    all_apperture_mask : list
        an N sized list of array, each value of the array is the weigth added to the aperture mask by photutils.
    all_pixel_indices : list
        an N sized list of array, each list contain 4 array, with respectively the two first array defining the 
        y and x coordinates in the M sized image and, the two last y and x coordinates in the aperture matrice.

    """
    if (PSF_shape == "Bessel" or "Circle"):
        all_apperture_mask, all_pixel_indices = circular_aperture_mask(position,N,M,radius)
    
    return all_apperture_mask, all_pixel_indices

def circular_aperture_mask(position,N,M,radius):
    """

    Parameters
    ----------
    position : numpy.ndarray
        table of pair value of x,y position, with a shape equal to (N,2).
    N : int
        total number of time steps.
    M : int
        number of pixel on each side of the matrix.
    radius : float
        radius of the studied apeture, defined by fwhm/2 value.
        
    Description
    -----------
    for each N step, compute a mask aperture on a M*M sized sized array, return this mask and the coordinates
    values of this mask.

    Returns
    -------
    all_apperture_mask : list
        an N sized list of array, each value of the array is the weigth added to the aperture mask by photutils.
    all_pixel_indices : list
        an N sized list of array, each list contain 4 array, with respectively the two first array defining the 
        y and x coordinates in the M sized image and, the two last y and x coordinates in the aperture matrice.

    """
    all_apperture_mask = []
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
            all_apperture_mask.append(None)
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
            all_apperture_mask.append(mask.data)
            all_pixel_indices.append(((y_img, x_img), (y_mask, x_mask)))
    
    return all_apperture_mask, all_pixel_indices

def PSF(x_kepler, N, M, CstData, all_pixel_indices, all_apperture_mask):
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
    all_apperture_mask : list
        an N sized list of array, each value of the array is the weigth added to the aperture mask by photutils.
    all_pixel_indices : list
        an N sized list of array, each list contain 4 array, with respectively the two first array defining the 
        y and x coordinates in the M sized image and, the two last y and x coordinates in the aperture matrice.
        
    Description
    -----------
    choose the appropriate PSF shape.

    Returns
    -------
    all_psf_shape_matrix : list
        an N sized list of array, each array is a 2D zero order bessel function, centered, with the border values
        close to zero, the value is computed for all the non zero value in the all_apperture_mask variable.
    
    """
    if (CstData.PSF_shape == "Bessel"):
        all_psf_shape_matrix = bessel_PSF(x_kepler, N, M, CstData.fwhm, all_pixel_indices, all_apperture_mask, CstData.r_vals, CstData.j0_vals)
    
    if (CstData.PSF_shape == "Circle"):
        all_psf_shape_matrix = circle_PSF(x_kepler, N, M, CstData.fwhm, all_pixel_indices, all_apperture_mask)
    
    return all_psf_shape_matrix
    

def circle_PSF(x_kepler, N, M, fwhm, all_pixel_indices, all_apperture_mask):
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
    all_apperture_mask : list
        an N sized list of array, each value of the array is the weigth added to the aperture mask by photutils.
    all_pixel_indices : list
        an N sized list of array, each list contain 4 array, with respectively the two first array defining the 
        y and x coordinates in the M sized image and, the two last y and x coordinates in the aperture matrice.
        
    Description
    -----------
    classical extracting method in a circle

    Returns
    -------
    all_psf_shape_matrix : list
        an N sized list of array, each array contain a circle of one centered on the position.
    
    """
    all_psf_shape_matrix = []
    
    for k in range(N):
        circle_psf = np.zeros_like(all_apperture_mask[k])
        
        if all_apperture_mask[k] is None:
            all_psf_shape_matrix.append(None)
            
        else:
            # Get the (y, x) indices in the aperture matrix
            apperture_y, apperture_x = all_pixel_indices[k][1]
            
            circle_psf[apperture_y, apperture_x] = 1
    
            all_psf_shape_matrix.append(circle_psf)
        
    return all_psf_shape_matrix
    

def bessel_PSF(x_kepler, N, M, fwhm, all_pixel_indices, all_apperture_mask, r_vals, j0_vals):
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
    all_apperture_mask : list
        an N sized list of array, each value of the array is the weigth added to the aperture mask by photutils.
    all_pixel_indices : list
        an N sized list of array, each list contain 4 array, with respectively the two first array defining the 
        y and x coordinates in the M sized image and, the two last y and x coordinates in the aperture matrice.
    r_vals : list
        x value for the bessel shape.
    j0_vals : list
        y value for the bessel shape.
        
    Description
    -----------
    compute a bessel shapped matrixs, on non zeros values of the aperture mask contained in all_apperture_mask variable
    for each N time step.

    Returns
    -------
    all_psf_shape_matrix : list
        an N sized list of array, each array is a 2D zero order bessel function, centered, with the border values
        close to zero, the value is computed for all the non zero value in the all_apperture_mask variable.
    
    """
    all_psf_shape_matrix = []
    factor = 2 / fwhm
    
    for k in range(N):
        bessel_value = np.zeros_like(all_apperture_mask[k])
        
        if all_apperture_mask[k] is None:
            all_psf_shape_matrix.append(None)
            
        else:
            # Get the (y, x) indices in the aperture matrix
            image_y, image_x = all_pixel_indices[k][0]
            apperture_y, apperture_x = all_pixel_indices[k][1]
            Y_size, X_size = all_apperture_mask[k].shape
    
            dx = image_x - x_kepler[k][1] + 0.5
            dy = image_y - x_kepler[k][0] + 0.5
    
            r = np.sqrt(dx**2 + dy**2) * factor
    
            # Fast interpolation using np.interp
            bessel_vals = np.interp(r, r_vals, j0_vals)
            
            # Direct fill into the array
            bessel_value[apperture_y, apperture_x] = bessel_vals
    
            # Ensure all values are non-negative
            bessel_value[bessel_value < 0] = 0
    
            all_psf_shape_matrix.append(bessel_value)
        
    return all_psf_shape_matrix