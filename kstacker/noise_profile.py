"""
Compute the noise and background profiles. Then, compute SNR_K-Stacker profiles
and can remove the noisy images.  We remove the images: average_noise_image
> reject_coef * total_average_noise_images Use the values in parameters.sh

"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.nddata import block_replicate
from scipy.signal import convolve2d

from . import coronagraph
from .imagerie import monte_carlo_profiles, monte_carlo_profiles_remove_planet
from .utils import compute_signal_and_noise_grid, create_output_dir

__author__ = "Herve Le Coroller"
__mail__ = "herve.lecoroller@lam.fr"
__status = "initial Development"


def pre_process_image(
    image_filename,
    aperture_radius,
    size=None,
    r_mask=None,
    r_mask_ext=None,
    mask_value=None,
    upsampling_factor=1,
    plot=False,
):
    """
    Pre-processing of an image before running KStacker.

    The image will be loaded, masked, and cut to a given size.

    Parameters
    ----------
    image_filename : str
        full path to the image. must be a .fits file
    size : int
        the final size (in pixels) of the pre-process image.
        If not given, initial size is kept.
    r_mask : int
        radius (in pixels) of the central internal mask to be applied.
        If not given, no internal mask is applied.
    r_mask_ext : int
        radius (in pixels) of the external mask to be applied.
        If not given, no external mask is applied.
    mask_value : float
        the value to put in pixel masked. Default is 0.0.
    upsampling_factor : int
        upsampling factor to precompute the aperture fluxes.
    plot : bool
        true to also get a png as output of the program

    """
    filename, extension = os.path.splitext(image_filename)
    if extension not in (".fits", ".FITS", ".Fits"):
        raise Exception("Image must be a .fits file!")

    image = fits.getdata(image_filename)
    nx, ny = np.shape(image)
    if nx != ny:
        print("Warning: image is not a square. Will be cut to a square")
        m = min(nx, ny) // 2
        cx, cy = nx // 2, ny // 2
        image = image[cx - m : cx + m, cy - m : cy + m]

    npix = image.shape[0]

    if r_mask < 0 or r_mask > npix / 2:
        raise Exception("Internal mask diameter must be > 0 and smaller than image")

    if r_mask_ext < 0 or r_mask_ext > npix / 2:
        raise Exception("External mask diameter must be > 0 and smaller than image")

    if mask_value is None:
        # If mask_value is None, we put the pixel values at zero in the masks
        # (internal, external)
        mask = 0 * image + 1
        mask = mask * np.abs(coronagraph.mask.lyot(npix, r_mask * 2))
        mask = mask * np.abs(coronagraph.stop.circular_stop(npix, r_mask_ext * 2))
        image = image * mask
    else:
        # In the mask (internal, external), we put the pixel values at 'mask_value'
        x, y = np.mgrid[:npix, :npix] - npix / 2
        dist2 = x**2 + y**2
        mask = (dist2 < r_mask**2) | (dist2 > r_mask_ext**2)
        image[mask] = mask_value

    image[np.isnan(image)] = 0
    hdr = fits.Header({"RMASK": r_mask, "RMASKEXT": r_mask_ext})
    fits.writeto(f"{filename}_preprocessed.fits", image, header=hdr, overwrite=True)

    if size is not None and size != npix:
        center, rad = size // 2, size // 2
        image = image[center - rad : center + rad, center - rad : center + rad]

    # Replicate the image for the upsampling
    imrepl = block_replicate(image, upsampling_factor, conserve_sum=True)

    # Compute the aperture mask
    masksize = (int(aperture_radius + 0.5) * 2 + 1) * upsampling_factor
    xx, yy = np.mgrid[:masksize, :masksize] - masksize // 2
    mask = (np.hypot(xx, yy) < aperture_radius * upsampling_factor).astype(int)

    # Convolve by the mask to compute the signal value
    imconv = convolve2d(imrepl, mask, mode="same")

    hdr.update(
        {"KERNEL": "circle", "RADIUS": aperture_radius, "FACTOR": upsampling_factor}
    )
    fits.writeto(f"{filename}_resampled.fits", imconv, overwrite=True)

    if plot:
        plt.imshow(image.T, origin="lower", interpolation="none", cmap="gray")
        plt.savefig(f"{filename}_preprocessed.png")
        plt.close()
        plt.imshow(imconv.T, origin="lower", interpolation="none", cmap="gray")
        plt.savefig(f"{filename}_resampled.png")
        plt.close()


def plot_noise(q, reject_coef, profile_dir, output_snrdir):
    """
    Plot the noise profile of each images
    :param q: int number of images
    :param reject_coef: float (default=1.4)
    :param profile_dir: directory where the noise and background profiles are stored
    :param output_snrdir: directory where snr plots are stored
    """

    total_average_noise_images = 0

    for k in range(q):
        noise = np.load(f"{profile_dir}/noise_prof{k}.npy")

        # Computation of the sum of average noise
        average_noise_image = np.average(noise)
        total_average_noise_images += average_noise_image

        plt.figure(f"Noise Plot {k}")
        plt.title(f"Noise Plot {k}")
        plt.plot(noise)
        plt.xlabel("radius in pixel")
        plt.ylabel("standart deviation of the noise")
        plt.savefig(f"{output_snrdir}/Plot_noise{k}.pdf")
        plt.close()

    total_average_noise_images = total_average_noise_images / q

    dict_params = {}
    image_removed = []

    for k in range(q):
        noise = np.load(f"{profile_dir}/noise_prof{k}.npy")
        average_noise_image = np.average(noise)
        if average_noise_image > reject_coef * total_average_noise_images:
            image_removed.append(k)
            dict_params["num of bad images"] = k

    with open(f"{output_snrdir}/bad_images.txt", "w") as file:
        for key, val in dict_params.items():
            file.write(f"{key}: {val}\n")

    return image_removed


def compute_noise_profiles(params):
    # Main Path definitions
    images_dir = params.get_path("images_dir")
    # Directories where the noise profiles will be stored
    profile_dir = params.get_path("profile_dir", remove_if_exist=True)

    # images characteristic (size, masks size, number of images, etc.) for K-Stacker
    nimg = params.p  # number of images
    size = params.n  # to keep the initial size n
    upsampling_factor = params.upsampling_factor

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
    ts = params.get_ts()

    # This variables are used to determine the part of the software that will be run
    noise_prof = params["noise_prof"]  # yes or no
    snr_plot = params["snr_plot"]  # yes or no

    # This section check if the program must remove the planet from noise and
    # background (Justin Bec-canet)
    remove_planet = params["remove_planet"]  # yes or no
    if remove_planet == "yes":
        # The coordinates of the planet are put into an array
        remove_box = [float(x) for x in params.remove_box.split("+")]

        print(params.planet_coord)
        print(remove_box)
        # planet coordinates is put in a numpy python format (1 Dim array)
        # splits coordinates, replace ':' by ','
        planet_coord_elem = [x.split(":") for x in params.planet_coord.split("+")]
        # for each element, we reassemble and evaluate as tuples the
        # different coordinates
        planet_coord = [eval(f"{elem[0]},{elem[1]}") for elem in planet_coord_elem]

    ###################################
    # Main noise and background program
    ###################################

    if noise_prof == "yes":
        # preparation of the images (cuts, add of masks at zero or mask_value, etc.)
        t0 = time.time()
        for k in range(nimg):
            pre_process_image(
                f"{images_dir}/image_{k}.fits",
                params.fwhm,
                size=size,
                r_mask=params.r_mask,
                r_mask_ext=params.r_mask_ext,
                mask_value=params.mask_value,
                upsampling_factor=upsampling_factor,
                plot=True,
            )
        print(f"preprocess: took {time.time() - t0:.2f} sec.")
        print("The images have been adjusted in size, masked and saved")

        # load the images and estimate the noise level assuming a radial profile
        t0 = time.time()
        img_suffix = params.get_image_suffix()
        for k in range(nimg):
            img = fits.getdata(f"{images_dir}/image_{k}{img_suffix}.fits")
            img = img.astype(float)
            if remove_planet == "yes":
                # uses a function to remove the planet in background calculations
                bg_prof, n_prof = monte_carlo_profiles_remove_planet(
                    img,
                    size,
                    planet_coord[k],
                    remove_box,
                    params.fwhm,
                    upsampling_factor,
                    method=params.method,
                )
            else:
                bg_prof, n_prof = monte_carlo_profiles(
                    img, size, params.fwhm, upsampling_factor, method=params.method
                )

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

        # Directories where the snr plots will be stored
        output_snrdir = (
            f"{profile_dir}/snr_plot_steps_remove_noise_{remove_noisy}_{reject_coef}"
        )
        output_snrgraph = f"{output_snrdir}/snr_graph"

        create_output_dir(output_snrdir)
        create_output_dir(output_snrgraph)

        t0 = time.time()
        image_removed = plot_noise(nimg, reject_coef, profile_dir, output_snrdir)
        print(f"plot_noise: took {time.time() - t0:.2f} sec.")
        print("Images removed because too noisy:", image_removed)

        x_profile = np.linspace(0, size // 2 - 1, size // 2)

        # load the images .fits and the noise profiles
        img_suffix = params.get_image_suffix()
        used, images, bkg_profiles, noise_profiles, ts_selected = [], [], [], [], []
        for k in range(nimg):
            if k in image_removed:
                # remove noisy images
                continue
            used.append(k)
            ts_selected.append(ts[k])
            images.append(fits.getdata(f"{images_dir}/image_{k}{img_suffix}.fits"))
            bkg_profiles.append(np.load(f"{profile_dir}/background_prof{k}.npy"))
            noise_profiles.append(np.load(f"{profile_dir}/noise_prof{k}.npy"))

        nimg = len(images)
        print("List of images used for the SNR computation:", used)

        images = np.array(images)
        bkg_profiles = np.array(bkg_profiles)
        noise_profiles = np.array(noise_profiles)

        # Definition of parameters that will be ploted function of the SNR
        parameters = ("a_j", "e_j", "t0_j", "omega_j", "i_j", "theta_0_j")

        for param in parameters:
            print(f"Computing SNR for param {param}")
            tstart = time.time()

            # parameters of the brute force search grid.
            if param == "a_j":
                param_vect = np.linspace(params.a_min, params.a_max, 3000)  # (A.U)
                name_Xaxis = "semi major a in au"

            if param == "e_j":
                # default (0,0.8,500)
                param_vect = np.linspace(params.e_min, params.e_max, 2000)
                name_Xaxis = "eccentricity"

            if param == "t0_j":
                param_vect = np.linspace(params.t0_min, params.t0_max, 2000)
                name_Xaxis = "t0"

            if param == "omega_j":
                # (rad) default (-3.14,3.14,1000)
                param_vect = np.linspace(params.omega_min, params.omega_max, 3000)
                name_Xaxis = "omega"

            if param == "i_j":
                # (rad) default (0, PI,3000)
                param_vect = np.linspace(params.i_min, params.i_max, 3000)
                name_Xaxis = "i"

            if param == "theta_0_j":
                # (rad) default (-3.14,3.14,1000)
                param_vect = np.linspace(params.theta_0_min, params.theta_0_max, 3000)
                name_Xaxis = "theta_0"

            # Orbital parameters initialisation
            x = [
                params.a_init,
                params.e_init,
                params.t0_init,
                params.omega_init,
                params.i_init,
                params.theta_0_init,
            ]
            x = np.tile(x, (param_vect.size, 1))
            param_idx = parameters.index(param)
            x[:, param_idx] = param_vect

            args = (
                ts_selected,
                params.m0,
                size,
                params.scale,
                params.fwhm,
                images,
                x_profile,
                bkg_profiles,
                noise_profiles,
                upsampling_factor,
                None,
                params.method,
            )

            signal, noise = compute_signal_and_noise_grid(x, *args)
            snr = signal / noise

            plt.plot(param_vect, snr, linewidth=1)
            plt.xlabel(name_Xaxis)
            plt.ylabel("SNR")
            suffix = f"{param}_{np.min(param_vect)}-{np.max(param_vect)}"
            plt.savefig(f"{output_snrgraph}/steps_{suffix}.pdf")
            plt.close()

            print(f"compute snr: took {time.time() - tstart:.2f} sec.")
