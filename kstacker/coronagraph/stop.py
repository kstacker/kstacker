"""
This module contains all the functions used in the lyot plane (lyot stop functions)
"""

__author__ = "Mathias Nowak"
__email__ = "mathias.nowak@ens-cachan.fr"
__status__ = "Testing"


import numpy as np


def circular_stop(n, d, center=None):
    """
    This function is used to create a simple circular Lyot stop of diameter d.
    @param int n: size of the returned array
    @param int d: diameter of the lyot stop
    @param int[2] center: optional argument that gives the position of the center of the stop (is None, the lyot stop is assumed to be centered in the array)
    @return complex[n, n]: complex array representing the lyot stop
    """
    # check if center argument has been given and process it
    if center is None:
        center = [n / 2, n / 2]
    m0, n0 = center

    # initialize matrix
    stop = np.zeros([n, n]) * 1j

    # create simple circular lyot mask
    for k in range(0, n):
        for l in range(0, n):
            if (k - m0) ** 2 + (l - n0) ** 2 <= (d / 2) ** 2:
                stop[k, l] = 1

    return stop
