"""
Functions used to generate and add noise to images
"""


import math

import numpy as np


def gaussian_noise(n, stddev):
    """
    A simple function used to generate a random white noise image
    @param int n: size (in pixes) of the returned image
    @param float stddev: standard deviation for the gaussain noise
    @return float[n, n]: image containing only a pixel to pixel gaussain noise
    """
    image = np.zeros([n, n])
    for k in range(n):
        for l in range(n):
            image[k, l] = np.random.normal(0, stddev)
    return image


def add_readout_noise(image, noise_level):
    """
    This simple function adds a readout noise to the given image
    @param float[n, n] image: intensity values representing the image on which readout noise should be added
    @param float noise_level stdandard deviation for the gaussian readout noise
    @return float[n, n]: image+readout noise
    """
    n = image.shape[0]
    noise = gaussian_noise(n, noise_level)
    return image + noise


def add_photon_noise(image):
    """
    This function is used to add photon noise to any given image
    @param float[n, n] intensity values of the image
    @return float[n, n]: sama image with photon noise added
    """
    n = image.shape(image)[0]
    for k in range(n):
        for l in range(n):
            image[k, l] = np.random.normal(image[k, l], math.sqrt(image[k, l]))
    return image
