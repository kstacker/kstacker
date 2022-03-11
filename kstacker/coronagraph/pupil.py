"""
This package contains all the functions required to generate and manipulate (i.e. apodize) the pupil wavefront
"""

__author__ = "Mathias Nowak"
__email__ = "mathias.nowak@ens-cachan.fr"
__status__ = "Testing"


import math

import numpy as np


def circular(n, d, m0, n0):
    """
    Function used to generate a circular aperture of diameter d (in cm), sampled on a n*n grid
    @param int n: size of the returned array
    @param int d: diameter of the circular aperture
    @param int m0: x position of the center of the aperture
    @param int n0: y position of the center of the aperture
    @return complex[n, n]: array representing the aperture (1 inside, 0 outside; unit of the grid: cm)
    """
    # intialize matrix
    aperture = np.zeros([n, n]) * 1j

    # create aperture
    for k in range(0, n):
        for l in range(0, n):
            if (k - m0) ** 2 + (l - n0) ** 2 <= (d // 2) ** 2:
                aperture[k, l] = 1

    return aperture


def apodize(aperture, apod_filename, diam, center=None):
    """
    Function used to apodize an aperture. Different types of apodization can be used (sin, guyon for example). Each type
    relates on a two column file (first column is the x axis, from EXACTLY 0 to EXACTLY 1, and second column is the y axis for the values)
    To add an option, one must add a specific file in this package, and then edit this function.
    @param int[n, n] aperture: an array representing the aperture
    @param string apod_type: filename for the apodization function to be applied
    @param float diam: diameter of the apodization window
    @param int[2] center: optional argument that gives the position of the center of the apodization (useful when aperture is not centered in the array)
                          if not given, the program will assume that the aperture is centered in the array
    @return int[n, n]: same array as aperture, but apodizated
    """
    # size of aperture array (assumes square)
    n = aperture.shape[0]

    # check if center argument has been given. if not, assume that the aperture is centered
    if center is None:
        center = [n // 2, n // 2]
        [m0, n0] = center

    # load the apodization function. apod_axis represents the axis on which is defined the function. it should always range from 0 to 1
    # apod_values represent tha actual apodization values, sampled on the apod_axis. It should range from 1 where apod_axis=0 (normalization
    # to 0 where apod_axis=1
    apod_data = np.loadtxt(apod_filename)
    apod_axis = apod_data[:, 0]
    apod_values = apod_data[:, 1]

    # apodize the aperture
    for k in range(0, n):
        for l in range(0, n):
            if (k - m0) ** 2 + (l - n0) ** 2 <= (
                diam // 2
            ) ** 2:  # pixel has to be in the apodization window
                aperture[k, l] = aperture[k, l] * np.interp(
                    abs(math.sqrt((k - m0) ** 2 + (l - n0) ** 2) / (diam / 2.0)),
                    apod_axis,
                    apod_values,
                )
            else:  # if not in the window, apodization function is 0
                aperture[k, l] = 0.0

    return aperture


def star(n, amp, x, y, wav):
    """
    Function that generates a wavefront corresponding to a point source with a given amplitude and a given position
    wavefront(k, l)=amp*exp(2*i*pi/wav*(k*x+l*y)
    @param int n: size of the matrix sampling the wavefront
    @param float amp: amplitude for the wave (related to the brightness of the object)
    @param float x: angular position of the star (in arcsec, on the x axis)
    @param float y: angular position of the star (in arcsec, on the y axis)
    @param float wav: wavelength (in cm)
    @return complex[n, n]: wavefront corresponding to the star, sampled on a n*n grid (unit of the grid: cm)
    """
    # initialize matrix
    wavefront = np.zeros([n, n]) * 1j

    # convert x and y in radians
    x_angle = x / 3600.0 / 360.0 * 2 * math.pi
    y_angle = y / 3600.0 / 360.0 * 2 * math.pi

    # create wavefront
    for k in range(n):
        for l in range(n):
            wavefront[k, l] = amp * math.exp(
                -2 * math.pi * 1j / wav * (k * x_angle + l * y_angle)
            )

    return wavefront
