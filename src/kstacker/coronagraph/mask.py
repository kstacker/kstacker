"""
This module contains all the functions that are used to generate coronagraphic masks
"""


import numpy as np


def lyot(n, d, center=None):
    """
    Function to create a simple circular Lyot mask (0 inside a disk of radius d)
    @param int n: size of the array
    @param int d: diameter of the mask (given in pixel)
    @param int[2] center: optional argument for the position of the mask (if None, mask is centered in the array
    @return complex[n, n]: complex array (actually 0 or 1 values) corresponding to the coronagraphic mask in the focal plane (no specific unit for the grid)
    """
    # check if center argument has been given and process it
    if center is None:
        center = [n // 2, n // 2]
    m0, n0 = center

    # initialize matrix
    mask = np.zeros([n, n]) * 1j

    # create simple circular lyot mask
    for k in range(0, n):
        for l in range(0, n):
            if (k - m0) ** 2 + (l - n0) ** 2 > (d // 2) ** 2:
                mask[k, l] = 1

    return mask
