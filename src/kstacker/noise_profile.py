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

from .imagerie import compute_noise_profile_apertures
from .snr import compute_signal_and_noise_grid
from .utils import create_output_dir


def pre_process_image(
    image_filename,
    aperture_radius,
    size=None,
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
    upsampling_factor : int
        upsampling factor to precompute the aperture fluxes.
    plot : bool
        true to also get a png as output of the program

    """
    filename, extension = os.path.splitext(image_filename)
    if extension not in (".fits", ".FITS", ".Fits"):
        raise ValueError("Image must be a .fits file!")

    image = fits.getdata(image_filename)
    image[np.isnan(image)] = 0

    nx, ny = np.shape(image)
    if nx != ny:
        print("Warning: image is not a square. Will be cut to a square")
        m = min(nx, ny) // 2
        cx, cy = nx // 2, ny // 2
        image = image[cx - m : cx + m, cy - m : cy + m]

    fits.writeto(f"{filename}_preprocessed.fits", image, overwrite=True)

    if size is not None and size != image.shape[0]:
        print(f"Cutting image to {size}x{size} pixels")
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

    hdr = fits.Header(
        {"KERNEL": "circle", "RADIUS": aperture_radius, "FACTOR": upsampling_factor}
    )
    fits.writeto(f"{filename}_resampled.fits", imconv, header=hdr, overwrite=True)

    if plot:
        plt.imshow(image.T, origin="lower", interpolation="none", cmap="gray")
        plt.savefig(f"{filename}_preprocessed.png")
        plt.close()
        plt.imshow(imconv.T, origin="lower", interpolation="none", cmap="gray")
        plt.savefig(f"{filename}_resampled.png")
        plt.close()


def compute_noise_profiles(params):
    # preparation of the images
    t0 = time.time()
    nimg = params.p  # number of images
    size = params.n  # to keep the initial size n

    images_dir = params.get_path("images_dir")
    for k in range(nimg):
        pre_process_image(
            f"{images_dir}/image_{k}.fits",
            params.fwhm,
            size=size,
            upsampling_factor=params.upsampling_factor,
            plot=True,
        )
    print(f"preprocess: took {time.time() - t0:.2f} sec.")
    print("The images have been adjusted in size, masked and saved")

    # load the images and estimate the noise level assuming a radial profile
    t0 = time.time()
    img_suffix = params.get_image_suffix(method="aperture")
    profile_dir = params.get_path("profile_dir", remove_if_exist=True)
    output_snrdir = f"{profile_dir}/snr_plot_steps"
    create_output_dir(output_snrdir)

    for k in range(nimg):
        img = fits.getdata(f"{images_dir}/image_{k}{img_suffix}.fits")

        try:
            mask_apertures = params.remove_planet[k]
        except (AttributeError, IndexError):
            mask_apertures = None
        bg_prof, noise_prof, n_aper = compute_noise_profile_apertures(
            img, aperture_radius=params.fwhm, mask_apertures=mask_apertures
        )
        np.save(f"{profile_dir}/background_prof{k}.npy", bg_prof)
        np.save(f"{profile_dir}/noise_prof{k}.npy", noise_prof)
        print(f"{k} sur {nimg - 1}")

        fig, ax = plt.subplots()
        ax.plot(noise_prof)
        ax.set(
            yscale="log",
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

def compute_mcmc_noise_signal_profil (params):
    """

    Parameters
    ----------
    params : kstacker.utils.Params
        point to the destination.
        
    Create an image of the 1/variance, log(sigma), (Signal-background)^2/variance, (Signal-background)/variance and Signal-background
    
    """
    def pixel_circle_std(N,M,images):
        """

        Parameters
        ----------
        N : float
            number of time step.
        M : float
            image length.
        images : numpy.ndarray
            all the images for every time step.

        Returns
        -------
        std : numpy.ndarray
            standard deviation for every time step.
        image : numpy.ndarray
            Signal-background for every time step.
    
        """
        radii = np.hypot(*np.meshgrid(np.arange(M)-M//2+0.5, np.arange(M)-M//2+0.5))
        # build an image of every radii value for every pixele
        
        std = np.empty((N,M,M,))
        std[:] = np.nan
        image = np.empty((N,M,M,))
        image[:] = np.nan
        # initialise the output image.
        
        for k in range(N):
            interp = RectBivariateSpline(np.arange(M), np.arange(M), images[k], kx=3, ky=3)
            # initialize the interpolation method for the k image
            for x in range(M):
                for y in range(M):
                    r = radii[x, y]
                    x0 = M / 2 + 0.5
                    y0 = M / 2 + 0.5
                    dx = x - x0
                    dy = y - y0
                    angle_theta_pixel = np.arctan2(dy, dx)
                    # acces to the theta angle of the point studied
                    
                    theta = np.random.uniform(angle_theta_pixel-np.pi/4, angle_theta_pixel+np.pi/4, 10000)
                    # generat 10000 random angles value on one quarter of a circle centered on the studied point on k,x,y value
                    
                    x_c = M//2 +0.5 + r * np.cos(theta)
                    y_c = M//2 +0.5 + r * np.sin(theta)
                    value = interp(x_c,y_c, grid=False)
                    # for every theta angles interpol the value of the hypothetical pixel on the quarter circle studied
                    
                    std[k,x,y] = 1.4826*np.median(abs(value-np.median(value)))* np.sqrt(1 + (1 / (2*np.pi*r/4)))
                    # build the standard deviation value has the the median absolute deviation correction with student over the quarter circle
                    
                    bg = np.mean(value)
                    image[k,x,y] = images[k,x,y] - bg
                    # build the background value and substract it to the signal value
            print(f"Time step number {k+1} done")
        return std, image
    
    data = params.load_data(method="aperture")
    images = data['images']
    N, M, _ = np.shape(data['images'])
    profile_dir = params.get_path("profile_dir")
    
    one_over_var = np.empty((N,M,M,))
    one_over_var[:] = np.nan
    Signal = np.empty((N,M,M,))
    Signal[:] = np.nan
    Signal_over_var = np.empty((N,M,M,))
    Signal_over_var[:] = np.nan
    Signal_2_over_var = np.empty((N,M,M,))
    Signal_2_over_var[:] = np.nan
    log_sigma = np.empty((N,M,M,))
    log_sigma[:] = np.nan
    # initialise the output image.
        
    std, image = pixel_circle_std(N,M,images)
    # build the std and signal images
    
    for k in range(N):
        for x in range(M):
            for y in range(M):
                one_over_var[k][x][y] = 1/std[k][y][x]**2
                Signal[k][x][y] = image[k][y][x]
                Signal_over_var[k][x][y] = image[k][y][x]/std[k][y][x]**2
                Signal_2_over_var[k][x][y] = image[k][y][x]**2/std[k][y][x]**2
                log_sigma[k][x][y] = np.log(std[k][y][x])
    
    for k in range(N):
        fits.writeto(f"{profile_dir}/one_over_var_{k}.fits", one_over_var[k], overwrite=True)
        fits.writeto(f"{profile_dir}/Signal_prof_{k}.fits", Signal[k], overwrite=True)
        fits.writeto(f"{profile_dir}/Signal_over_variance_prof_{k}.fits", Signal_over_var[k], overwrite=True)
        fits.writeto(f"{profile_dir}/Signal_square_over_variance_prof_{k}.fits", Signal_2_over_var[k], overwrite=True)
        fits.writeto(f"{profile_dir}/log_sigma_prof_{k}.fits", log_sigma[k], overwrite=True)
