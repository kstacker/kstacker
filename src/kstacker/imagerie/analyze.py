"""
A set of functions used to analyze images produced by the rest of the code
(compute photometry, Signal to Noise Ratios, etc.)
"""


import math

import numpy as np
import scipy.ndimage as ndi
from photutils import CircularAperture, aperture_photometry

from .._utils import photometry_preprocessed
from ..orbit import orbit as orb


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
    x0, y0 = orb.project_position_full(t0, a, e, t0, m, omega, i, theta_0)
    x, y = orb.project_position_full(t, a, e, t0, m, omega, i, theta_0)

    # Compute angle between perihelion vector and position at time t
    cos_alpha = np.dot([x0, y0], [x, y])
    sin_alpha = np.cross([x0, y0], [x, y])
    alpha = np.rad2deg(np.arctan2(sin_alpha, cos_alpha))

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

    return np.nanmean(rot_images, axis=0)


def monte_carlo_noise(image, npix, radius, fwhm, upsampling_factor, method="convolve"):
    """
    Another function to estimate the noise in a given image by throwing disks
    at random positions and estimatin gthe deviation of the flux inside them.

    @param float[npix, npix] image: the image in which the noise shall be computed
    @return float noise_level: noise level (stddev of the distribution of flux in the airy disks)
    """
    p = 1000  # number of disks to generate
    theta = np.random.uniform(-math.pi, math.pi, size=p)
    x = radius * np.cos(theta) + npix // 2
    y = radius * np.sin(theta) + npix // 2

    if method == "convolve":
        fluxes = photometry_preprocessed(image, x, y, upsampling_factor)
    elif method == "aperture":
        fluxes = photometry(image, [x, y], 2 * fwhm)
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
                    image, position[0], position[1], upsampling_factor
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


def compute_noise_apertures(
    img,
    x,
    y,
    aperture_radius,
    mask=None,
    exclude_source=False,
    exclude_lobes=False,
    return_apertures=False,
):
    """Compute noise and bkg value at position x, y.

    Using apertures on a disk for the radius.
    """

    center = img.shape[0] // 2
    xc = x - center
    yc = y - center
    radius = np.hypot(xc, yc)

    aperture_angle = np.arcsin(aperture_radius / radius) * 2

    if np.isnan(aperture_angle):
        # happens e.g. if too close to the center
        return np.nan, np.nan, 0

    n_aper = int(np.floor(2 * np.pi / aperture_angle))

    angles = np.linspace(0, 2 * np.pi, n_aper, endpoint=False)
    xx = center + np.cos(angles) * xc + np.sin(angles) * yc
    yy = center + np.cos(angles) * yc - np.sin(angles) * xc

    pos = np.array([xx, yy]).T
    # exclude x,y
    pos = pos[1:]
    if exclude_lobes:
        pos = pos[1:-1]

    apertures = CircularAperture(pos, r=aperture_radius)
    fluxes = aperture_photometry(img, apertures, mask=mask)["aperture_sum"]
    # Remove apertures that fall on masked data
    fluxes = fluxes[fluxes != 0]

    if fluxes.size == 0:
        res = 0, 0, 0
    else:
        std = np.std(fluxes, ddof=1) * np.sqrt(1 + (1 / n_aper))
        res = np.mean(fluxes), std, n_aper

    if return_apertures:
        return *res, apertures
    else:
        return res


def compute_noise_profile_apertures(img, aperture_radius, mask_apertures=None):
    """Compute noise and bkg profiles with apertures on a disk for each radius."""
    if mask_apertures is not None:
        mask = np.zeros(img.shape, dtype=bool)
        for x, y, r in mask_apertures:
            mask_planet = CircularAperture((y, x), r=r).to_mask(method="center")
            mask |= mask_planet.to_image(img.shape, dtype=bool)
    else:
        mask = None

    res = []
    center = img.shape[0] // 2
    start = int(np.ceil(aperture_radius))
    res = [
        compute_noise_apertures(img, center, center + r, aperture_radius, mask=mask)
        for r in range(start, img.shape[0] // 2)
    ]
    res = [(np.nan, np.nan, 0)] * start + res
    bg, noise, n_aper = np.array(res).T
    return bg, noise, n_aper
