"""
A set of functions used to analyze images produced by the rest of the code
(compute photometry, Signal to Noise Ratios, etc.)
"""

__author__ = "Mathias Nowak, Dimitri Estevez"
__email__ = "mathias.nowak@ens-cachan.fr"
__status__ = "Testing"

import math

import numpy as np
import scipy.ndimage as ndi
from photutils import CircularAperture, aperture_photometry

from ..orbit import orbit as orb
from ._photometry import photometry_preprocessed


def photometry(image, position, diameter):
    """
    This function is used to compute the flux within a circular aperture in a given image.
    @param float[n, n] image: image within wich the flux shall be computed
    @param float[2] position: xy position (in pixels) of the center of the photometry box
    @param float diameter: diameter of the circular photometry box (in pixels)
    @return float: flux contained in the circular photometry box
    """
    xpix, ypix = position

    # grid for photutils is centered on pixels hence the - 0.5
    xpix = xpix - 0.5
    ypix = ypix - 0.5

    # CircularAperture works the other way around compared to numpy hence the x<->y reversal
    if isinstance(xpix, np.ndarray):
        position = list(zip(ypix, xpix))
    else:
        position = (ypix, xpix)

    aperture = CircularAperture(position, r=diameter / 2.0)
    phot = aperture_photometry(image, aperture)
    res = np.array(phot["aperture_sum"])
    return res[0] if res.size == 1 else res


def total_photometry(images, positions, diameter):
    """
    For each image images[k], a photometric box is placed at position[k], and the total flux in these box is returned. This function is used to compute the "signal" in
    the recombined image.
    @param float[n, n, m] images: a serie of k images of size n (intensity values)
    @param float[2, k] positions: a serie of xy positions (in pixels)
    @param float diameter: diameter of the circular photometric box
    @return float: total flux in all box
    """
    m = images.shape[2]
    signal = 0.0

    for k in range(m):
        signal = signal + photometry(images[:, :, k], positions[:, k], diameter)

    return signal


def derotate(image, t, scale, a, e, t0, m, omega, i, theta_0):
    """
    This function is used to de-rotate and translate an image so that the planet is placed over its position at perihelion.
    @param float[n, n] image: input intensity array for the image
    @param float t: time corresponding to the image
    @param float scale: scale of the image (in a.u. per pixel)
    @param float a: semi-major axis of the orbit (in a.u.)
    @param float e: eccentricity
    @param float t0: time at perihelion (year)
    @param float m: ass of the central star (in solar masses)
    @param float omega: longitude of asc. node (rad)
    @param float i: inclination (rad)
    @param float theta_0: argument of the perihelion (rad)
    @return float[n, n]: the rotated and translated image where the planet ends up on its perihelion position
    """
    # Compute position at perihelion and position at time t
    x0, y0 = orb.project_position(orb.position(t0, a, e, t0, m), omega, i, theta_0)
    x, y = orb.project_position(orb.position(t, a, e, t0, m), omega, i, theta_0)

    # Compute angle between perihelion vector and position at time t
    cos_alpha = np.dot([x0, y0], [x, y])
    sin_alpha = np.cross([x0, y0], [x, y])
    alpha = np.arctan2(sin_alpha, cos_alpha)  # in rad
    alpha = alpha / (2 * math.pi) * 360  # conversion to degrees

    # Rotate the image to align planet with the perihelion vector
    rot_image = ndi.rotate(
        image, -alpha, reshape=False, order=3, mode="constant", cval=0.0
    )

    # Now, compute the translation distance and convert it to pixels
    dist = np.linalg.norm([x0, y0]) - np.linalg.norm([x, y])
    dist = scale * dist

    # Compute unit vector corresponding to perihelion direction and multiply
    # by distance to get shift vector to apply to the image
    unit_vect = [x0, y0] / np.linalg.norm([x0, y0])
    shift_vect = dist * unit_vect

    # shift the image
    shift_image = ndi.shift(rot_image, shift_vect, order=3, mode="constant", cval=0.0)

    return shift_image


def recombine_images(images, ts, scale, a, e, t0, m, omega, i, theta_0):
    """
    This function is used to compute the coadded image form orbital parameters. Each image of the serie is rotated and shifted to put the planet on its perihelion
    position, and the results are coadded to give the final result.
    @param ndarray[n, n, p] images: serie of intensity arrays corresponding to the images
    @param ndarray[p] ts: time serie corresponding to the image
    @param float scale: scale of the image (in a.u. per pixel)
    @param float a: semi-major axis of the orbit (in a.u.)
    @param float e: eccentricity
    @param float t0: time at perihelion (year)
    @param float m: ass of the central star (in solar masses)
    @param float omega: longitude of asc. node (rad)
    @param float i: inclination (rad)
    @param float theta_0: argument of perihelion (rad)
    @return float[n, n]: coadded result image
    """
    rot_images = []
    for k in range(len(images)):
        im = derotate(images[k], ts[k], scale, a, e, t0, m, omega, i, theta_0)
        rot_images.append(im)

    return np.mean(rot_images, axis=0)


def noise(image, r_int, r_ext):
    """
    A simple function that computes the standard deviation on n annulus between
    r_int and r_ext in the image.

    @param float[n, n] image: the image on which the stddev shall be computed
    @param float r_int: internal radius of the annulus at which the stddev shall be computed
    @param float r_ext: external radius of the annulus at which the stddev shall be computed
    @return float: stddev
    """
    n = image.shape[0]
    values = []
    for k in range(n):
        for l in range(n):
            idx = (k - n // 2) ** 2 + (l - n // 2) ** 2
            if idx < r_ext**2 and idx > r_int**2:
                values.append(image[k, l])
    return np.std(values)


def snr(image, center, position, fwhm):
    """
    A function to compute the snr, define as the ratio between a given pixel
    value (the signal) to the stadard deviation in the ring of same radius (noise).

    @param float[n,n] image: intensity values of the image
    @param float[n] position: x, y position of the 'signal' pixel (given in pixel from the center)
    @return float: snr
    """
    x, y = position
    x0, y0 = center
    n = image.shape[0]

    signal = photometry(image, [x + n // 2, y + n // 2], 2 * fwhm) / (
        math.pi * fwhm**2
    )  # mean value inside the photometric box

    # start by computing noise level in an annulus of width fwhm, excluding the
    # area where psf is
    r = math.sqrt(x**2 + y**2)
    r_int = r - fwhm
    r_ext = r + fwhm
    values = []

    for k in range(n):
        for l in range(n):
            idx = (k - x0) ** 2 + (l - y0) ** 2
            if idx < r_ext**2 and idx > r_int**2:
                if (k - x - x0) ** 2 + (l - y - y0) ** 2 > (2 * fwhm) ** 2:
                    # exclude area of 2*fwhm around psf central position
                    values.append(image[k, l])

    #    return image[n/2+x, n/2+y]/np.std(values)
    return signal / np.std(values)


def snr_boite(image, position, size, fwhm):
    """
    Function used to compute the Signal to Noise Ratio at a given position,
    using a box around this position to estimate a local noise level.

    The signal is computed as the mean value inside a disk of radius 2*fwhm.
    Noise is stddev in the box.

    @param float[n, n]: intensity values of the image
    @param float[2]: x y position where the snr shall be computed (in pixel, from the center)
    @param float size: sie of the box to compute the noise (in pixel, typically 3*fwhm)
    @param float fwhm: fwhm in pixels
    @return float: snr value
    """
    n = image.shape[0]
    x, y = position
    x = x + n // 2  # convert from-center-position to real position
    y = y + n // 2

    # mean value inside the photometric box
    signal = photometry(image, [x, y], 2 * fwhm) / (math.pi * (fwhm) ** 2)

    values = []  # a list that will contain all the noise pixels
    for k in range(int(x - size // 2), int(x + size // 2 + 1)):
        for l in range(int(y - size // 2), int(y + size // 2 + 1)):
            if (k - x) ** 2 + (l - y) ** 2 > (1.5 * fwhm) ** 2:
                # for each pixel in the noise box around position, check if
                # this pixel is inside the photometric box
                values.append(image[k, l])  # if not, add it to the list

    # noise level is simply the stddev of the list of noise pixels
    noise = np.std(values)
    bg = np.mean(values)

    #    plt.hist(values, 15)
    #    plt.show()
    return signal - bg, noise, (signal - bg) / noise


def snr_tot(image, position, fwhm):
    """
    Function used to compute the Signal to Noise Ratio at a given position,
    using the noise on the total image

    The signal is computed as the mean value inside a disk of radius 2*fwhm.
    Noise is stddev of image.

    @param float[n, n]: intensity values of the image
    @param float[2]: x y position where the snr shall be computed (in pixel, from the center)
    @param float size: sie of the box to compute the noise (in pixel, typically 3*fwhm)
    @param float fwhm: fwhm in pixels
    @return float: snr value
    """
    n = image.shape[0]
    [x, y] = position
    x = x + n // 2  # convert from-center-position to real position
    y = y + n // 2

    # mean value inside the photometric box
    signal = photometry(image, [x, y], 2 * fwhm) / (math.pi * (fwhm) ** 2)

    # noise level is simply the stddev of the list of noise pixels
    noise = np.std(image[np.where(image != 0)])

    return signal / noise


def radial_profile(image, fwhm, center=None):
    n = image.shape[0]
    if center is None:
        center = [n // 2, n // 2]

    [x, y] = center

    profile = np.zeros(n // 2)
    for p in range(n // 2):
        values = []
        r_ext = p + fwhm
        r_int = p - fwhm
        for k in range(n):
            for l in range(n):
                idx = (k - x) ** 2 + (l - y) ** 2
                if idx < r_ext**2 and idx > r_int**2:
                    values.append(image[k, l])
        profile[p] = np.mean(values)

    return profile


def monte_carlo_noise(image, npix, radius, fwhm, upsampling_factor, method="convolve"):
    """
    Another function to estimate the noise in a given image by throwing disks
    at random positions and estimatin gthe deviation of the flux inside them.

    @param float[npix, npix] image: the image in which the noise shall be computed
    @return float noise_level: noise level (stddev of the distribution of flux in the airy disks)
    """
    p = 1000  # number of disks to generate
    theta = np.random.uniform(-math.pi, math.pi, size=p)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    position = [x + npix // 2, y + npix // 2]

    if method == "convolve":
        fluxes = photometry_preprocessed(image, position, upsampling_factor)
    elif method == "aperture":
        fluxes = photometry(image, position, 2 * fwhm)
    else:
        raise ValueError(f"invalid method {method}")

    return np.mean(fluxes), np.std(fluxes)


def monte_carlo_noise_remove_planet(
    image, npix, radius, planet, remove_box, fwhm, upsampling_factor, method="convolve"
):
    """
    Author : Justin Bec-Canet

    Another function to estimate the noise in a given image by throwing disks
    at random positions and estimating the deviation of the flux inside them.

    It is excluding from the calculation the position of the planet.

    @param float[npix, npix] image: the image in which the noise shall be computed
    @param float planet : couple of coordinates for the planet (x_planet,y_planet)
    @param float[4] remove_box : size of removal box (default = [10,10,10,10])
    @return float noise_level: noise level (stddev of the distribution of flux in the airy disks)
    """

    p = 1000  # number of disks to generate
    realcount = 0
    count = 0
    fluxes = np.zeros(p)
    while realcount < p:
        theta = np.random.uniform(-math.pi, math.pi)
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        # the positions are stored in (xtest,ytest) in pixels
        xtest, ytest = (x + npix // 2, y + npix // 2)
        if (
            (planet[0] - remove_box[0] <= xtest)
            and (xtest <= planet[0] + remove_box[1])
        ) and (
            (planet[1] - remove_box[2] <= ytest)
            and (ytest <= planet[1] + remove_box[3])
        ):  # the exclusion zone can be changed and adapted to the image.
            count += 1  # counter
        else:
            position = [xtest, ytest]

            if method == "convolve":
                fluxes[realcount] = photometry_preprocessed(
                    image, position, upsampling_factor
                )
            elif method == "aperture":
                fluxes[realcount] = photometry(image, position, 2 * fwhm)
            else:
                raise ValueError(f"invalid method {method}")

            realcount += 1
        if count > 1e5:
            print("Exclusion box must be too large, convergence error")
    return np.mean(fluxes), np.std(fluxes)


def monte_carlo_profiles(image, npix, fwhm, upsampling_factor, method="convolve"):
    noise_profile = np.zeros(npix // 2)
    background_profile = np.zeros(npix // 2)
    for k in range(npix // 2):
        bg, noise = monte_carlo_noise(
            image, npix, k, fwhm, upsampling_factor, method=method
        )
        noise_profile[k] = noise
        background_profile[k] = bg

    return background_profile, noise_profile


def monte_carlo_profiles_remove_planet(
    image, npix, planet_coord, remove_box, fwhm, upsampling_factor, method="convolve"
):
    """Author : Justin Bec-Canet
    using almost the same function as above. I just implemented the parameter
    "planet_coord" to be able to exclude the planet
    (via its position in pixels) in the calculation of the noise and background profiles.
    @param float[4] remove_box : size of removal box (default = [10,10,10,10])
    """
    noise_profile = np.zeros(npix // 2)
    background_profile = np.zeros(npix // 2)
    for k in range(npix // 2):
        bg, noise = monte_carlo_noise_remove_planet(
            image,
            npix,
            k,
            planet_coord,
            remove_box,
            fwhm,
            upsampling_factor,
            method=method,
        )
        noise_profile[k] = noise
        background_profile[k] = bg

    return background_profile, noise_profile


def gauss_function(x_model, a, x0, sigma):
    return a * np.exp(-((x_model - x0) ** 2) / (2 * sigma**2))


def snr_annulus(image, position, fwhm):
    """
    A function that computes signal, noise level, and snr in a given image.
    @param float[n, n] image: array that represents the image
    @param float[2] position: xy position (in pixels) of the center of the photometry box
    @param float fwhm: fwhm of the airy disk (photometry box is a disk of radius=fwhm
    @return float[3]: [signal, noise, signal/noise]. signal is the total flux in the photometry box, minus the continuum level, and divided by nimber of pixel;
                      noise is computed as the stddev on the annulus.
    """
    n = image.shape[0]

    # Start by taking all the pixels in the annulus
    [x, y] = position
    d = np.sqrt(x**2 + y**2)
    r_int = d - fwhm
    r_ext = d + fwhm

    x = x + n // 2
    y = y + n // 2

    values = []
    for k in range(n):
        for l in range(n):
            ind = (k - n // 2) ** 2 + (l - n // 2) ** 2
            if ind < r_ext**2 and ind > r_int**2:
                if (k - x) ** 2 + (l - y) ** 2 > 2 * fwhm:
                    values.append(image[k, l])

    # stddev is noise level; mean is background continuum
    bg = np.mean(values)
    noise_level = np.std(values)

    # compute photometry and remove background
    s = photometry(image, [x, y], 2 * fwhm)
    s = s / (math.pi * fwhm**2)
    s = s - bg

    return [-s, noise_level, -s / noise_level]


def noise_profile(image, fwhm):
    """
    A simple function that computes the standard deviation on n annulus between
    r_int and r_ext in the image

    @param float[n, n] image: the image on which the stddev shall be computed
    @param float r_int: internal radius of the annulus at which the stddev shall be computed
    @param float r_ext: external radius of the annulus at which the stddev shall be computed
    @return float: stddev
    """

    n = image.shape[0]
    noise_levels = np.zeros(n // 2)

    for k in range(n // 2):
        noise_levels[k] = noise(image, k - fwhm, k + fwhm)

    return noise_levels
