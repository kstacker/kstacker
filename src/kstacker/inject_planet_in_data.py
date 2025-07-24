from kstacker.orbit import orbit
from kstacker.PSF_shape_mcmc import precompute_bessel_lookup, aperture, bessel_PSF

from astropy.io import fits

import numpy as np
import os
import re

import matplotlib.pyplot as plt

def read_orbital_param_injected_planet(params):
    """

    Parameters
    ----------
    params : kstacker.utils.Params
        get information on the data set studied.

    Returns
    -------
    coord : list
        orbital parameters for all injected planet.

    """
    
    # initialisation
    work_dir = params.get_path("work_dir")
    coord = []
    planet_dict = {}
    
    # read file
    with open(f"{work_dir}/orbital_param_injected_planet.txt") as file:
        for line in file:
            line = line.strip()
            if line!='':
                key, value = line.split(' = ')
                try:
                    planet_dict[key] = eval(value)
                except:
                    planet_dict[key] = value
            else:
                # save orbital parameters for each injected planet
                coord.append(planet_dict)
                planet_dict = {}
    if planet_dict:
        coord.append(planet_dict)
        
    return coord

def inject_planet(params):
    """

    Parameters
    ----------
    params : kstacker.utils.Params
        get information on the data set studied.

    Returns
    -------
    None

    """
    # read file
    coord = read_orbital_param_injected_planet(params)
    images_dir = params.get_path("images_dir")
    images = []
    
    ts = np.array(params.get_ts())
    
    # get all the original images
    files = os.listdir(images_dir)
    pattern = re.compile(r'^image_(\d+)\.fits$')
    dico = {int(pattern.match(s).group(1)): s for s in files if pattern.match(s)}
    
    for a in range(len(dico)):
        fits_image_filename = os.path.join(images_dir, dico[a])
        hdul = fits.open(fits_image_filename)
        images.append(hdul[0].data)
        hdul.close()
        
    N,M,_ = np.shape(np.array(images))
    
    for i in range(len(coord)):
        fwhm = 2.44*(coord[i]['wave_length']*10**(-9))/coord[i]['Telescope_D']*((206265*10**(3))/(params.resol))
        print(fwhm)
        # planet projected positions
        position = orbit.project_position_full(ts,coord[i]['a'],coord[i]['e'],coord[i]['t0'],
                                    coord[i]['m0'],coord[i]['omega'],coord[i]['i'],coord[i]['theta_0'])
        position *= params.scale
        position += params.n // 2
        
        # take the aperture shape from the MCMC method 
        all_mask, all_pixel_indices = aperture(position,N,M,fwhm,coord[i]['PSF'])
        
        # define the PSF shape 
        if coord[i]['PSF']=='Bessel':
            r_vals, j0_vals = precompute_bessel_lookup()
            j0_vals = j0_vals/max(j0_vals)*coord[i]['intensity']
            PSF = bessel_PSF(position,N,M,fwhm,all_pixel_indices,all_mask, r_vals, j0_vals)
        
        if coord[i]['PSF']=='Circle':
            PSF = all_mask/np.max(all_mask)*coord[0]['intensity']
        
        # the planet on each images
        for k in range(len(dico)):
            image_size_y, image_size_x = all_pixel_indices[k][0]
            resize_y, resize_x = all_pixel_indices[k][1]
            
            images[k][image_size_y, image_size_x] = images[k][image_size_y, image_size_x] + PSF[k][resize_y, resize_x]
    
    for k in range(len(dico)):
        fits_image_filename = os.path.join(images_dir, dico[k])
        
        with fits.open(fits_image_filename, mode='update') as hdul:
            hdul[0].data = images[k]
            hdul.flush()