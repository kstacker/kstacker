"""
Function wich is automatically loaded with the module, and which can be used to create a noisy image containing only one PSF (the planet)
"""

__author__ = "Mathias Nowak"
__email__ = "mathias.nowak@ens-cachan.fr"
__status__ = "Testing"

from .psf import image_psf_from_file


def create_noisy_image(n, position, psf_filename, flux, background_image):
    """
    Function which creates an image with a planet PSF ontop of a background image
    @param int n: size (in pixels) of the image
    @param float[2] position: xy position (in pixels) of the planet PSF (counted from the center)
    @param string psf_filename: name of the file fro which the PSF shall be loaded
    @param float flux: total flux in the planet PSF
    @param string background_filename: name of the txt file containing the background noise image
    """
    m = background_image.shape[0]
    psf = image_psf_from_file(psf_filename, position, flux)
    return (
        psf[0:n, 0:n]
        + background_image[(m - n) // 2 : (m + n) // 2, (m - n) // 2 : (m + n) // 2]
    )
