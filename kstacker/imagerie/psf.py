"""
Function to create a PSF of a given fwhm
"""

__author__ = "Mathias Nowak"
__email__ = "mathias.nowak@ens-cachan.fr"
__status__ = "Working"

import numpy as np
import scipy.integrate
import scipy.special
from scipy.ndimage.interpolation import shift


def psf(x, y, fwhm):
    """
    Function to compute the value of at position (x, y) of a PSF centered at (0, 0) and of given fwhm
    @param float x: x-axis position where the PSF value should be computed, in arbitrary unit
    @param float y: y-axis position where the PSF value should be computed, in abritrary unit
    @param float fwhm: Full Width at Half Maximum of the PSF, in arbitrary unit
    @return float: Value of the PSF at (x, y), normalized to a pic value of 1.
    """
    res = 0.0

    # compute radial coordinate, in fwhm unit
    r = np.sqrt(x**2 + y**2) * (2 * 1.61633) / fwhm

    # compute PSF from J1 function
    res = (2 * scipy.special.j1(r) / r) ** 2

    # check if result is nan (r=0 above). In this case, PSF is 1.
    if np.isnan(res):
        res = 1.0

    return res


def int_psf(xmin, xmax, ymin, ymax, fwhm):
    """
    A simple function that integrate the flux of a PSF of given fwhm within a given squared-box
    @param float xmin: min x-value of the integration box (arbitrary unit)
    @param float xmax: max x-value of the integration box (arbitrary unit)
    @param float ymin: min y-value of the integration box (arbitrary unit)
    @param float ymax: max y-value of the integration box (arbitrary unit)
    @param float fwhm: Full Width at Half Maximum of the PSF (arbitrary unit)
    @return float: flux of the PSF within the squared-box [xmin, xmax]X[ymin, ymax]
    """
    return scipy.integrate.dblquad(
        lambda x, y: psf(x, y, fwhm), xmin, xmax, lambda x: ymin, lambda x: ymax
    )[0]


def image_psf(center, fwhm, flux, n):
    """
    Create a n x n pixel image containing only a PSF of a given fwhm, intensity, and central position.
    @param float[2] center: x, y position of the center of the PSF (in pixel)
    @param float fwhm: Full Width at Half Maximum of the PSF (in pixel)
    @param float flux: Total flux of the PSF
    @param int n: size of the image
    @return float[n, n]: Image containing only the PSF
    """
    # Initialize
    [x, y] = center
    image = np.zeros([n, n])

    # For each pixel within 5 fwhm, compute the flux of the PSF within the
    # square box corresponding to the pixel
    for k in range(n):
        for l in range(n):
            if ((k - x) ** 2 + (l - y) ** 2) < (5 * fwhm) ** 2:
                image[k, l] = int_psf(k - x, k + 1 - x, l - y, l + 1 - y, fwhm)

    # Normalizing to a given total flux
    image = image / np.sum(image) * flux

    return image


def image_psf_from_file(filename, center, flux):
    """
    This function uses a txt file containing the correct normalized psf, shifts it to the requested position, and multiply it by the flux. It is extremely useful to
    compute complex PSF (ex: guyon apodized) for which there is no closed form.
    @param string filename: name of the file containing the PSF
    @param float[2] center: xy position (in pixel, from bottom left) of the PSF
    @param float flux: total flux (energy) in the PSF
    @return float[n, n]: intensity values of the PSF. Size is the same than from the file
    """
    # load image
    image = np.loadtxt(filename)

    # mutliply to get correct total energy
    image = image * flux

    # create shift vector (counted from center)
    n = image.shape[0]
    [x, y] = center
    shift_vect = [x - n // 2, y - n // 2]

    # shift psf
    shift_image = shift(image, shift_vect, order=3, mode="constant", cval=0.0)

    return shift_image
