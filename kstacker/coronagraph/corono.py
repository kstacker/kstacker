"""
This contains the main function for simulating the coronagraph. It calls all
the necessary functions, and perform the required Fourier Transform in proper
order to get the image as given by a coronagraphic system
"""


import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

from . import mask, phase, pupil, stop


def corono(
    n,
    wav,
    d,
    dpix,
    d_stop,
    d_mask,
    t_exp,
    t_ao,
    apod_type="none",
    plot="rien",
    turb="y",
    turb_mask="none",
    wind=10.0,
    seeing=0.8,
    mag_gs=8,
    static="y",
    static_mask=None,
    image_mask="n",
    d_image_mask=10,
    d_image_lyot=120,
    parfait="n",
):
    """
    This function returns a coronagraphic image of a star, that takes into account phase error due to atmospheric turbulence and AO correction, as well as static phase    errors induced by optical defects.
    @param int n: pixel size of all the array used (including the returned image)
    @param float wav: wavelength of observation (in m)
    @param float d: diameter of the primary mirror (in m)
    @param int dpix: diameter of the primarry mirror (in pixels)
    @param int d_stop: diameter of the lyot stop (in pixels, usually 0.9 to 1 times dpix)
    @param int d_mask: diameter of the coronagraphic mask (in pixels)
    @param float t_exp: exposure time (in s).
    @param float t_ao: decorrelation time for the AO (in s). This parameter is used to compute the number of independant images (given by texp/t_ao) required
                       to get the total exposure time.
    @param (optional) string apod_type: apodization to be used on the primary mirror. Shall be 'none' or shall correspond to a txt file located in coronagraph/apod
                                        (see coronagraph.pupil)
    @param (optional) string plot: shall be one of "rien" (don't plot anything) "images" (plot focal plane and coronagraphic images), or "tout" (plot everything,
                                   including all phase masks and fields in different planes)
    @param (optional) string turb: add atmospheric+AO corrected phase error? ('y' or 'n')
    @param (optional) string turb_mask: name of the turbulence mask to be used (if 'none', a new one is generated)
    @param (optional) float wind: wind speed, used to move atmospheric phase mask between single images, to average the speckeles over the entire exposure.
                                  The AO loop time is supposed to be 10 ms, so a total number of t_exp/0.010 images will be generated and summed (in intesity)
    @param (optinal) float seeing: seeing value (in arcsecond, at 0.5 microns)
    @param (optinal) float mag_gs: magnitude of the guide star for the AO system (for wafefront sensor band)
    @param (optional) string static: add static phase error? ('y' or 'n')
    @param (optional) float[n, n] static_mask: static mask to be used (in radians; if 'none', a new one is automatically generated)
    @param (optional) string image_mask: apply a circular mask on the center of the final image? (shall be one of 'y' or 'n')
    @param (optional) int d_image_mask: diameter of the simple circular mask to be applied on the final image
    @param (optional) int d_image_lyot: diameter of the external circular mask to be applied on the final image
    @param (optional) string parfait: use a perfect coronagraph ('y' or 'n')?
    @return float[n, n]: coronagraphic image (intensity in arbitrary unit)
    """

    m0 = n // 2
    n0 = n // 2  # coordinates of the center of the aperture

    n_im = int(t_exp / t_ao)  # number of images required to get total exposure time
    image = np.zeros([n, n]) * 1j

    norm_factor = 0.0

    # DEBUT DU PROGRAMME

    # define wavefront
    wavefront = pupil.star(
        n, 1, 0, 0, wav
    )  # correspond to a single star at the center of the image

    # define telescope aperture with apodization
    aperture = pupil.circular(n, dpix, m0, n0)

    if apod_type == "guyon":
        aperture = pupil.apodize(aperture, "coronagraph/apod/guyon.txt", dpix)
    if apod_type == "apo1_384":
        aperture = pupil.apodize(aperture, "coronagraph/apod/apo1_384.txt", dpix)
    if apod_type == "apo2_384":
        aperture = pupil.apodize(aperture, "coronagraph/apod/apo2_384.txt", dpix)

    # define coronagraphic mask
    mask_corono = mask.lyot(n, d_mask)

    # define lyot stop
    lyot_stop = stop.circular_stop(n, d_stop)

    # create and load static phase error mask
    if static == "y":
        if static_mask is None:
            # FIXME: wfe is undefined!
            static_phase_mask = phase.my_static(wfe, n, wav)  # noqa
        else:
            static_phase_mask = static_mask  # mask is passed as a parameter

    # compute fields for each image (atmospheric mask is moving according to wind speed)
    for l in range(n_im):
        print(l)

        # create and load atmospheric+ao corrected phase mask
        if turb == "y":
            if turb_mask == "none":
                phase.create_ao_mask(
                    n, d, dpix, wav, t_ao, wind, seeing, mag_gs, "atm" + str(l)
                )
                ao_phase_mask = phase.get_mask("atm" + str(l))
            else:
                ao_phase_mask = phase.get_mask(turb_mask + str(l))

        # Compute field in the pupil plane:

        pupil_field = aperture * wavefront

        if static == "y":
            pupil_field = pupil_field * np.exp(1j * static_phase_mask)

        if turb == "y":
            angle = np.random.uniform(0, 360)
            ao_phase_mask = scipy.ndimage.interpolation.rotate(
                ao_phase_mask, angle, reshape=False
            )
            pupil_field = pupil_field * np.exp(1j * ao_phase_mask)

        # Compute the FFT to get the field in the focal plane

        focal_field = np.fft.fft2(pupil_field)
        focal_field = np.fft.fftshift(focal_field)

        focal_image = abs(focal_field) ** 2
        norm_factor = norm_factor + np.max(
            np.fft.fft2(aperture) ** 2
        )  # save the central peak value for normalization

        # print("Strehl="+str(np.abs(np.max(focal_image/norm_factor))))

        # Case of a perfect coronagraph (Fusco2004)
        if parfait == "y":
            sigma = np.std(pupil_field[np.where(aperture != 0)])
            e = np.exp(-(sigma**2))
            pupil_field = pupil_field - np.sqrt(e) * aperture
            image_l = abs(np.fft.fft2(pupil_field) ** 2)
            image_l = np.fft.fftshift(image_l)
            image = image + image_l

        # Case of the not perfect ALC
        else:
            masked_field = focal_field * mask_corono
            lyot_field = np.fft.fft2(masked_field)
            lyot_field_stopped = lyot_field * lyot_stop
            image_field = np.fft.ifft2(lyot_field_stopped)
            image = image + abs(image_field**2)

    # If final image masking has been requested, apply it
    if image_mask == "y":
        image = image * mask.lyot(n, d_image_mask)  # lyot is a simple circular mask
        image = image * stop.circular_stop(
            n, d_image_lyot
        )  # external stop to mask non AO corrected area

    # plot everything (or almost for last image)

    plt.figure(2)
    plt.gray()

    if plot == "tout":
        plt.figure(1)
        plt.title("aperture")
        plt.imshow(
            abs(pupil_field), origin="lower", extent=[-n // 2, n // 2, -n // 2, n // 2]
        )

    if plot == "tout" or plot == "images":
        plt.figure(2)
        plt.title("focal plane image")
        plt.imshow(
            np.sqrt(abs(focal_image / norm_factor)),
            origin="lower",
            interpolation="none",
        )
        plt.colorbar()

    if plot == "tout":
        plt.figure(3)
        plt.title("coronographic mask")
        plt.imshow(abs(mask_corono), origin="lower")

    if plot == "tout":
        plt.figure(4)
        plt.title("masked field")
        plt.imshow(abs(masked_field), origin="lower")
    #        plt.savefig("focal.png")

    if plot == "tout":
        plt.figure(5)
        plt.title("lyot field")
        plt.imshow(abs(lyot_field), origin="lower")
    #        plt.savefig("lyot_amont.png")

    if plot == "tout":
        plt.figure(6)
        plt.title("stopped field")
        plt.imshow(abs(lyot_field_stopped), origin="lower")
    #        plt.savefig("lyot_aval.png")

    if plot == "tout" or plot == "images":
        plt.figure(7)
        plt.title("image")
        plt.imshow(abs(image / norm_factor), origin="lower", interpolation="none")
        plt.colorbar()
    #        plt.savefig("image.png")

    if plot == "tout" or plot == "images":
        if turb == "y":
            plt.figure(8)
            plt.title("ao_phase_mask")
            plt.imshow(ao_phase_mask, origin="lower")
            plt.colorbar()

        if static == "y":
            plt.figure(9)
            plt.title("static_phase_mask")
            plt.imshow(static_phase_mask, origin="lower")
            plt.colorbar()

    if plot == "tout" or plot == "images":
        plt.show()

    return abs(image / norm_factor)
