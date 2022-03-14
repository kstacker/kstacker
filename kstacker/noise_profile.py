"""
Compute the noise and background profiles. Then, compute SNR_K-Stacker profiles
and can remove the noisy images.  We remove the images: average_noise_image
> reject_coef * total_average_noise_images Use the values in parameters.sh

__author__="Herve Le Coroller"
__mail__="herve.lecoroller@lam.fr"
__status="initial Development"

"""

import math
import os
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from . import coronagraph
from .imagerie.analyze import (
    monte_carlo_profiles,
    monte_carlo_profiles_remove_planet,
    photometry,
)
from .orbit import orbit as orb


def pre_process_image(
    image_filename,
    output_filename=None,
    size=None,
    mask_diameter_int=None,
    mask_diameter_ext=None,
    mask_value=None,
    plot=False,
):
    """
    Pre-processing of an image before running KStacker.

    The image will be loaded, masked, and cut to a given size.

    @param str image_filename: full path to the image. must be a .fits file
    @param (optional) str output_filename: the full path to the output file, without extension. If not given, 'preprocessed' is append to input file.
    @param (optional) int n: the final size (in pixels) of the pre-process image. If not given, initial size is kept
    @param (optional) int mask_diameter_int: diameter (in pixels) of the central internal mask to be applied. If not given, no internal mask is applied
    @param (optional) int mask_diameter_ext: diameter (in pixels) of the external mask to be applied. If not given, no external mask is applied
    @param (optional) float mask_value: the value to put in pixel masked. Default is 0.0.
    @param (optional) bool plot: true to also get a png as output of the program

    """
    filename, extension = os.path.splitext(image_filename)
    if extension not in (".fits", ".FITS", ".Fits"):
        raise Exception("Image must be a .fits file!")

    if output_filename is None:
        output_filename = f"{filename}_preprocessed"

    image = fits.getdata(image_filename)
    nx, ny = np.shape(image)
    if nx != ny:
        print("Warning: image is not a square. Will be cut to a square")
        m = min(nx, ny) // 2
        cx, cy = nx // 2, ny // 2
        image = image[cx - m : cx + m, cy - m : cy + m]

    size = image.shape[0]
    mask = 0 * image + 1

    # We put Zero values at the mask position
    if mask_diameter_int is not None:
        if mask_diameter_int < 0 or mask_diameter_int > size:
            raise Exception("Internal mask diameter must be > 0 and smaller than image")
        # mask = mask * np.abs(coronagraph.mask.lyot(size, mask_diameter_int))

    if mask_diameter_ext is not None:
        if mask_diameter_ext < 0 or mask_diameter_ext > size:
            raise Exception("External mask diameter must be > 0 and smaller than image")
        # mask = mask * np.abs(coronagraph.stop.circular_stop(size, mask_diameter_ext))

    # image = image * mask

    if mask_value is None:
        # If mask_value is None, we put the pixel values at zero in the masks
        # (internal, external)
        mask = mask * np.abs(coronagraph.mask.lyot(size, mask_diameter_int))
        mask = mask * np.abs(coronagraph.stop.circular_stop(size, mask_diameter_ext))
        image = image * mask
    else:
        # In the mask (internal, external), we put the pixel values at 'mask_value'
        # FIXME: optimize this ?
        for k in range(0, size):
            for l in range(0, size):
                if (
                    (k - size / 2.0) ** 2.0 + (l - size / 2.0) ** 2.0
                    < (mask_diameter_int / 2.0) ** 2.0
                ) or (
                    (k - size / 2.0) ** 2.0 + (l - size / 2.0) ** 2.0
                    > (mask_diameter_ext / 2.0) ** 2.0
                ):
                    image[k, l] = mask_value

    image[np.isnan(image)] = 0

    # FIXME: already done above ?
    center, radius = size // 2, size // 2
    image_cut = image[
        center - radius : center + radius, center - radius : center + radius
    ]
    fits.writeto(f"{output_filename}.fits", image_cut, overwrite=True)

    if plot:
        plt.imshow(image_cut.T, origin="lower", interpolation="none", cmap="gray")
        plt.savefig(f"{output_filename}.png")
        plt.close()


def plot_noise(q, reject_coef, profile_dir, output_snrdir):
    """
    Plot the noise profile of each images
    :param q: int number of images
    :param reject_coef: float (default=1.4)
    :param profile_dir: directory where the noise and background profiles are stored
    :param output_snrdir: directory where snr plots are stored
    """

    dict_params = dict()
    total_average_noise_images = 0

    for k in range(q):
        noise = np.load(f"{profile_dir}/noise_prof{str(k)}.npy")

        # Computation of the sum of average noise
        average_noise_image = np.average(noise)
        total_average_noise_images += average_noise_image

        plt.figure(f"Noise Plot {str(k)}")
        plt.title(f"Noise Plot {str(k)}")
        plt.plot(noise)
        plt.xlabel("radius in pixel")
        plt.ylabel("standart deviation of the noise")
        plt.savefig(f"{output_snrdir}/Plot_noise{str(k)}.png")
        plt.close()

    total_average_noise_images = total_average_noise_images / q

    image_removed = []

    for k in range(q):
        noise = np.load(f"{profile_dir}/noise_prof{str(k)}.npy")
        average_noise_image = np.average(noise)
        if average_noise_image > reject_coef * total_average_noise_images:
            # print 'Retirer image_' + str(k)
            image_removed.append(k)
            dict_params["num of bad images"] = k

    with open(f"{output_snrdir}/bad_images.txt", "w") as file:
        for key, val in dict_params.items():
            file.write(f"{key}: {val}\n")

    return image_removed


def compute_signal(x, ts, m0, n, scale, images, fwhm, x_profile, bkg_profiles):
    """define signal"""
    nimg = len(images)
    a, e, t0, omega, i, theta_0 = x
    res = np.zeros(nimg)  # res will contain signal for each image
    for k in range(nimg):
        # compute position
        x, y = orb.project_position(
            orb.position(ts[k], a, e, t0, m0), omega, i, theta_0
        )
        # convert position into pixel in the image
        position = [scale * x + n // 2, scale * y + n // 2]
        temp_d = np.sqrt(x**2 + y**2) * scale  # get the distance to the center
        # compute the signal by integrating flux on a PSF, and correct it for
        # background (using pre-computed background profile)
        res[k] = photometry(images[k], position, 2 * fwhm) - np.interp(
            temp_d, x_profile, bkg_profiles[k]
        )
        if math.isnan(res[k]):
            # if the value of signal is nan outside the image, consider it to be 0
            res[k] = 0.0
    return np.sum(res)  # return sn


def compute_noise(x, ts, m0, scale, x_profile, noise_profiles):
    """define noise"""
    nimg = len(noise_profiles)
    a, e, t0, omega, i, theta_0 = x
    res = np.zeros(nimg)  # res will contain noise for each image

    for k in range(nimg):
        # compute position
        x, y = orb.project_position(
            orb.position(ts[k], a, e, t0, m0), omega, i, theta_0
        )
        temp_d = np.sqrt(x**2 + y**2) * scale  # get the distance to the center
        # get noise at position using pre-computed radial noise profil
        res[k] = np.interp(temp_d, x_profile, noise_profiles[k])
        if math.isnan(res[k]):
            # if the value of signal is nan outside the image, consider it to be 0.
            res[k] = 0.0

    noise = np.sqrt(np.sum(res**2))
    if noise == 0:
        # if the value of total noise is 0 (i.e. all values of noise are 0,
        # i.e. the orbit is completely out of the image) then snr=0
        noise = 1
    return noise


def compute_signal_and_noise(
    x,
    ts,
    m0,
    n,
    scale,
    images,
    fwhm,
    x_profile,
    bkg_profiles,
    noise_profiles,
):
    nimg = len(images)
    a, e, t0, omega, i, theta_0 = x
    signal = np.zeros(nimg)
    noise = np.zeros(nimg)

    for k in range(nimg):
        # compute position
        x, y = orb.project_position(
            orb.position(ts[k], a, e, t0, m0), omega, i, theta_0
        )
        # convert position into pixel in the image
        position = [scale * x + n // 2, scale * y + n // 2]
        temp_d = np.sqrt(x**2 + y**2) * scale  # get the distance to the center

        # compute the signal by integrating flux on a PSF, and correct it for
        # background (using pre-computed background profile)
        signal[k] = photometry(images[k], position, 2 * fwhm) - np.interp(
            temp_d, x_profile, bkg_profiles[k]
        )

        # get noise at position using pre-computed radial noise profil
        noise[k] = np.interp(temp_d, x_profile, noise_profiles[k])

    # if the value of signal is nan outside the image, consider it to be 0
    signal[np.isnan(signal)] = 0
    noise[np.isnan(noise)] = 0

    noise = np.sqrt(np.sum(noise**2))
    if noise == 0:
        # if the value of total noise is 0 (i.e. all values of noise are 0,
        # i.e. the orbit is completely out of the image) then snr=0
        noise = 1

    return np.sum(signal), noise


def get_path(params, key):
    return os.path.join(os.path.expanduser(params["work_dir"]), params[key])


def create_output_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def compute_noise_profiles(params):
    # Main Path definitions
    images_dir = get_path(params, "images_dir")
    profile_dir = get_path(params, "profile_dir")

    # Optical parameters
    resol = params["resol"]  # mas / pixels
    wav = float(params["wav"])
    diam = params["d"]

    # images characteristic (size, masks size, number of images, etc.) for K-Stacker
    nimg = params["p"]  # number of images
    size = params["n"]  # to keep the initial size n
    r_mask = params["r_mask"]
    mask_diameter_int = 2 * r_mask
    r_mask_ext = float(params["r_mask_ext"])
    mask_diameter_ext = 2 * r_mask_ext

    # Parameters used in the SNR plot part of the program:

    # Parameters to reject very bad images (due to bad seeing, AO problems, etc.)
    #  'no' or 'weakly' or 'strongly'; Default: weakly (=1.4)
    remove_noisy = params["remove_noisy"]

    if remove_noisy == "weakly":
        print("We remove the noisy images [average_noise > 1.4 * total_average_noise]")
        # Default: 1.4 for number of images large ; See demo of herve logbook
        reject_coef = 1.4
    else:
        if remove_noisy == "strongly":
            print("We remove strongly the noisy images (reject_coef = 1) ")
            reject_coef = 1
        else:
            print("We don't remove noisy images")
            reject_coef = 999999

    # The epochs of observation are put in an array of time
    total_time = float(params["total_time"])
    # total time for all the observations in years for simulations;
    # 0 for real observations (time will be used)

    if total_time == 0:
        ts = [float(x) for x in params["time"].split("+")]
        print("time_vector used: ", ts)
    else:
        ts = np.linspace(0, total_time, nimg)
        # put nimg + p_prev, if later we use the p_prev option
        print("Creation of a regular time vector")

    # Star parameters
    dist = float(params["dist"])
    m0 = float(params["m0"])

    # scale factor used to convert pixel to astronomical unit (in pixel/a.u.)
    scale = 1.0 / (dist * (resol / 1000.0))

    # Reference orbit used to plot the SNRs (can be the orbit of a planet)
    a_init = float(params["a_init"])
    e_init = float(params["e_init"])
    t0_init = float(params["t0_init"])
    omega_init = float(params["omega_init"])
    i_init = float(params["i_init"])
    theta_0_init = float(params["theta_0_init"])

    # range of parameters where the SNR function will be plotted
    a_min = float(params["a_min"])
    a_max = float(params["a_max"])
    e_min = float(params["e_min"])
    e_max = float(params["e_max"])
    t0_min = float(params["t0_min"])
    t0_max = float(params["t0_max"])
    omega_min = float(params["omega_min"])
    omega_max = float(params["omega_max"])
    i_min = float(params["i_min"])
    i_max = float(params["i_max"])
    theta_0_min = float(params["theta_0_min"])
    theta_0_max = float(params["theta_0_max"])

    # This variables are used to determine the part of the software that will be run
    noise_prof = params["noise_prof"]  # yes or no
    snr_plot = params["snr_plot"]  # yes or no

    # Value of the intern and extern masks
    mask_value = float(params["mask_value"])

    # This section check if the program must remove the planet from noise and
    # background (Justin Bec-canet)
    remove_planet = params["remove_planet"]  # yes or no

    if remove_planet == "yes":
        # The coordinates of the planet are put into an array
        planet_coord = params["planet_coord"]
        remove_box = params["remove_box"]
        remove_box = [float(x) for x in remove_box.split("+")]

        print(planet_coord)
        print(remove_box)
        # planet coordinates is put in a numpy python format (1 Dim array)

        planet_coord_elem = planet_coord.split("+")  # splits coordinates a first time
        # splits a second time, to replace ':' by ','
        planet_coord_elem = [x.split(":") for x in planet_coord_elem]
        planet_coord = []
        for k in range(len(planet_coord_elem)):
            # for each element, we reassemble and evaluate as tuples the
            # different coordinates
            planet_coord.append(
                eval(f"{planet_coord_elem[k][0]},{planet_coord_elem[k][1]}")
            )  # gives a list of tuples with floats inside

    # Directories where the noise profiles will be stored
    if noise_prof == "yes":
        create_output_dir(profile_dir)

    # Directories where the snr plots will be stored
    output_snrdir = (
        f"{profile_dir}/snr_plot_steps_remove_noise_{remove_noisy}_{reject_coef}"
    )
    output_snrgraph = f"{output_snrdir}/snr_graph"

    if snr_plot == "yes":
        create_output_dir(output_snrdir)
        create_output_dir(output_snrgraph)

    ###################################
    # Main noise and background program
    ###################################

    # initialization
    fwhm = (1.028 * wav / diam) * (180.0 / np.pi) * 3600 / (resol / 1000.0)

    if noise_prof == "yes":
        # preparation of the images (cuts, add of masks at zero or mask_value, etc.)
        t0 = time.time()
        for k in range(nimg):
            pre_process_image(
                f"{images_dir}/image_{k}.fits",
                None,
                size,
                mask_diameter_int,
                mask_diameter_ext,
                mask_value=mask_value,
                plot=True,
            )
        print(f"preprocess: took {time.time() - t0:.2f} sec.")

        print("The images have been adjusted in size, masked and saved")

        # load the images and estimate the noise level assuming a radial profile
        t0 = time.time()
        for k in range(nimg):
            img = fits.getdata(f"{images_dir}/image_{k}_preprocessed.fits")
            if remove_planet == "yes":
                # uses a function to remove the planet in background calculations
                bg_prof, n_prof = monte_carlo_profiles_remove_planet(
                    img, fwhm, planet_coord[k], remove_box
                )
            else:
                bg_prof, n_prof = monte_carlo_profiles(img, fwhm)

            np.save(f"{profile_dir}/background_prof{k}.npy", bg_prof)
            np.save(f"{profile_dir}/noise_prof{k}.npy", n_prof)
            print(f"{k} sur {nimg - 1}")

        print(f"profiles: took {time.time() - t0:.2f} sec.")
        print("Background and noise profile Done.")

    ###################################
    # Main SNRs plot program
    ###################################

    if snr_plot == "yes":
        print("Coeff of rejection for noisy images: ", reject_coef)

        t0 = time.time()
        image_removed = plot_noise(nimg, reject_coef, profile_dir, output_snrdir)
        print(f"plot_noise: took {time.time() - t0:.2f} sec.")
        print("Images removed because too noisy:", image_removed)

        # initialization for the second part of this software for 'nimg' images
        x_profile = np.linspace(0, size // 2 - 1, size // 2)

        # load the images .fits and the noise profiles
        used, images, bkg_profiles, noise_profiles, ts_selected = [], [], [], [], []
        for k in range(nimg):
            if k in image_removed:
                # remove noisy images
                continue
            used.append(k)
            ts_selected.append(ts[k])
            images.append(fits.getdata(f"{images_dir}/image_{k}_preprocessed.fits"))
            bkg_profiles.append(np.load(f"{profile_dir}/background_prof{k}.npy"))
            noise_profiles.append(np.load(f"{profile_dir}/noise_prof{k}.npy"))

        nimg = len(images)
        print("List of images used for the SNR computation:", used)

        # Definition of parameters that will be ploted function of the SNR
        parameters = ("a_j", "e_j", "t0_j", "omega_j", "i_j", "theta_0_j")

        for param in parameters:
            print(f"Computing SNR for param {param}")
            tstart = time.time()

            # Orbital parameters initialisation
            a = a_init
            e = e_init
            t0 = t0_init
            omega = omega_init
            i = i_init
            theta_0 = theta_0_init

            # parameters of the brute force search grid.
            # Format is [min value, max value, number of points].
            if param == "a_j":
                a = np.linspace(a_min, a_max, 3000)  # (A.U)
                param_vect = a
                name_Xaxis = "semi major a in au"

            if param == "e_j":
                e = np.linspace(e_min, e_max, 2000)  # default (0,0.8,500)
                param_vect = e
                name_Xaxis = "excenticity"

            if param == "t0_j":
                t0 = np.linspace(t0_min, t0_max, 2000)
                param_vect = t0
                name_Xaxis = "t0"

            if param == "omega_j":
                # (rad) default (-3.14,3.14,1000)
                omega = np.linspace(omega_min, omega_max, 3000)
                param_vect = omega
                name_Xaxis = "omega"

            if param == "i_j":
                i = np.linspace(i_min, i_max, 3000)  # (rad) default (0, PI,3000)
                param_vect = i
                name_Xaxis = "i"

            if param == "theta_0_j":
                # (rad) default (-3.14,3.14,1000)
                theta_0 = np.linspace(theta_0_min, theta_0_max, 3000)
                param_vect = theta_0
                name_Xaxis = "theta_0"

            signal = np.zeros(len(param_vect))
            noise = np.zeros(len(param_vect))

            args = (
                ts_selected,
                m0,
                size,
                scale,
                images,
                fwhm,
                x_profile,
                bkg_profiles,
                noise_profiles,
            )

            # Run the loop on the orbital parameter param to compute the snr:
            if param == "a_j":
                for k in range(len(param_vect)):
                    x = [a[k], e, t0, omega, i, theta_0]
                    signal[k], noise[k] = compute_signal_and_noise(x, *args)

            if param == "e_j":
                for k in range(len(param_vect)):
                    x = [a, e[k], t0, omega, i, theta_0]
                    signal[k], noise[k] = compute_signal_and_noise(x, *args)

            if param == "t0_j":
                for k in range(len(param_vect)):
                    x = [a, e, t0[k], omega, i, theta_0]
                    signal[k], noise[k] = compute_signal_and_noise(x, *args)

            if param == "omega_j":
                for k in range(len(param_vect)):
                    x = [a, e, t0, omega[k], i, theta_0]
                    signal[k], noise[k] = compute_signal_and_noise(x, *args)

            if param == "i_j":
                for k in range(len(param_vect)):
                    x = [a, e, t0, omega, i[k], theta_0]
                    signal[k], noise[k] = compute_signal_and_noise(x, *args)

            if param == "theta_0_j":
                for k in range(len(param_vect)):
                    x = [a, e, t0, omega, i, theta_0[k]]
                    signal[k], noise[k] = compute_signal_and_noise(x, *args)

            snr = signal / noise

            plt.plot(param_vect, snr)
            plt.xlabel(name_Xaxis)
            plt.ylabel("SNR")
            suffix = f"{param}_{np.min(param_vect)}-{np.max(param_vect)}"
            plt.savefig(f"{output_snrgraph}/steps_{suffix}.png")
            plt.close()

            print(f"compute snr: took {time.time() - tstart:.2f} sec.")
