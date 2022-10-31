"""
Compute the noise and background profiles.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.nddata import block_replicate
from scipy.signal import convolve2d

from .imagerie import monte_carlo_profiles, monte_carlo_profiles_remove_planet
from .utils import compute_signal_and_noise_grid, create_output_dir


def pre_process_image(
    image_filename,
    aperture_radius,
    size=None,
    r_mask=None,
    r_mask_ext=None,
    mask_value=0,
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
        raise ValueError("Image must be a .fits file!")

    image = fits.getdata(image_filename)
    nx, ny = np.shape(image)
    if nx != ny:
        print("Warning: image is not a square. Will be cut to a square")
        m = min(nx, ny) // 2
        cx, cy = nx // 2, ny // 2
        image = image[cx - m : cx + m, cy - m : cy + m]

    npix = image.shape[0]

    if r_mask < 0 or r_mask > npix / 2:
        raise ValueError("Internal mask diameter must be > 0 and smaller than image")

    if r_mask_ext < 0 or r_mask_ext > npix / 2:
        raise ValueError("External mask diameter must be > 0 and smaller than image")

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


def compute_noise_profiles(params):
    # preparation of the images (cuts, add of masks at zero or mask_value, etc.)
    t0 = time.time()
    nimg = params.p  # number of images
    size = params.n  # to keep the initial size n

    # This section check if the program must remove the planet from noise and
    # background (Justin Bec-canet)
    if params.remove_planet == "yes":
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

    images_dir = params.get_path("images_dir")
    for k in range(nimg):
        pre_process_image(
            f"{images_dir}/image_{k}.fits",
            params.fwhm,
            size=size,
            r_mask=params.r_mask,
            r_mask_ext=params.r_mask_ext,
            mask_value=params.mask_value,
            upsampling_factor=params.upsampling_factor,
            plot=True,
        )
    print(f"preprocess: took {time.time() - t0:.2f} sec.")
    print("The images have been adjusted in size, masked and saved")

    # load the images and estimate the noise level assuming a radial profile
    t0 = time.time()
    img_suffix = params.get_image_suffix()
    profile_dir = params.get_path("profile_dir", remove_if_exist=True)
    output_snrdir = f"{profile_dir}/snr_plot_steps"
    create_output_dir(output_snrdir)

    for k in range(nimg):
        img = fits.getdata(f"{images_dir}/image_{k}{img_suffix}.fits")
        img = img.astype(float)
        if params.remove_planet == "yes":
            # uses a function to remove the planet in background calculations
            bg_prof, n_prof = monte_carlo_profiles_remove_planet(
                img,
                size,
                planet_coord[k],
                remove_box,
                params.fwhm,
                params.upsampling_factor,
                method=params.method,
            )
        else:
            bg_prof, n_prof = monte_carlo_profiles(
                img,
                size,
                params.fwhm,
                params.upsampling_factor,
                method=params.method,
            )

        np.save(f"{profile_dir}/background_prof{k}.npy", bg_prof)
        np.save(f"{profile_dir}/noise_prof{k}.npy", n_prof)
        print(f"{k} sur {nimg - 1}")

        fig, ax = plt.subplots()
        ax.plot(n_prof)
        ax.set(
            title=f"Noise Plot {k}",
            xlabel="radius in pixel",
            ylabel="standart deviation of the noise",
        )
        plt.savefig(f"{output_snrdir}/Plot_noise{k}.pdf")
        plt.close()

    print(f"profiles: took {time.time() - t0:.2f} sec.")
    print("Background and noise profile Done.")


def compute_snr_plots(params):
    """Create SNR plot for each grid parameter."""

    # Directories where the snr plots will be stored
    profile_dir = params.get_path("profile_dir")
    outdir = f"{profile_dir}/snr_plot_steps/snr_graph"
    create_output_dir(outdir)

    # load the images and the noise/background profiles
    data = params.load_data()
    ts = params.get_ts()
    size = params.n  # to keep the initial size n

    args = (
        ts,
        size,
        params.scale,
        params.fwhm,
        data,
        params.upsampling_factor,
        None,
        params.method,
    )

    # Definition of parameters that will be ploted function of the SNR
    parameters = params.grid.grid_params
    for name in parameters:
        print(f"Computing SNR for param {name}")
        tstart = time.time()
        param = params[name]

        if param["Ninit"] <= 1:
            print("skipping this parameter")
            continue

        # Orbital parameters initialisation: fixed value for all
        # parameters except for the one being processed in the loop
        param_vect = np.linspace(param["min"], param["max"], param["Ninit"])
        x = [params[key]["init"] for key in parameters]
        x = np.tile(x, (param_vect.size, 1))
        param_idx = parameters.index(name)
        x[:, param_idx] = param_vect

        signal, noise = compute_signal_and_noise_grid(x, *args)
        snr = signal / noise
        np.save(f"{outdir}/snr_{name}.npy", snr)

        fig, ax = plt.subplots()
        ax.plot(param_vect, snr, linewidth=1)
        ax.set(xlabel=param["label"], ylabel="SNR")
        plt.savefig(f"{outdir}/steps_{name}_{param_vect.min()}-{param_vect.max()}.pdf")
        plt.close()

        print(f"compute snr: took {time.time() - tstart:.2f} sec.")
