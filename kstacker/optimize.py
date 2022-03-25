"""
Core script used to compute the values of the signal and the noise on a given
part of the total grid (will be run on several nodes). A brute force algorithm
is used.
"""

import numpy as np
from astropy.io import fits

from .utils import compute_signal_and_noise_grid, create_output_dir

__author__ = "Mathias Nowak, Dimitri Estevez"
__email__ = "mathias.nowak@ens-cachan.fr, destevez@lam.fr"
__status__ = "Development"


def brute_force(params):
    # name of the directory where one loads and saves the images and values
    images_dir = params.get_path("images_dir")
    profile_dir = params.get_path("profile_dir")
    grid_dir = params.get_path("grid_dir")

    # total time of the observation (years)
    total_time = float(params["total_time"])
    nimg = params["p"]  # number of timesteps
    p_prev = params["p_prev"]

    if total_time == 0:
        ts = [float(x) for x in params["time"].split("+")]
        ts = ts[p_prev:]
    else:
        ts = np.linspace(0, total_time, nimg + p_prev)
    print("time_vector used: ", ts)

    size = int(params["n"])  # number of pixels in one direction
    x_profile = np.linspace(0, size // 2 - 1, size // 2)

    # load the images .fits or .txt and the noise profiles
    images, bkg_profiles, noise_profiles = [], [], []
    for k in range(nimg):
        i = k + p_prev
        images.append(fits.getdata(f"{images_dir}/image_{i}_preprocessed.fits"))
        bkg_profiles.append(np.load(f"{profile_dir}/background_prof{i}.npy"))
        noise_profiles.append(np.load(f"{profile_dir}/noise_prof{i}.npy"))

    # grid on which the brute force algorithm will be computed on one node/core
    print(repr(params.grid))

    # brute force
    args = (
        ts,
        params.m0,
        size,
        params.scale,
        images,
        params.fwhm,
        x_profile,
        bkg_profiles,
        noise_profiles,
        params.r_mask,
    )
    create_output_dir(grid_dir)

    grid, (s_values, n_values) = params.grid.evaluate(
        compute_signal_and_noise_grid, args=args, nchunks=2
    )

    if params.adding == "no":
        np.save(f"{grid_dir}/s_values.npy", s_values)
        np.save(f"{grid_dir}/grid.npy", grid)
        np.save(f"{grid_dir}/n_values.npy", n_values)
    elif params.adding == "yes":
        np.save(f"{grid_dir}/s_values_add.npy", s_values)
        np.save(f"{grid_dir}/n_values_add.npy", n_values)

        s_values_prev = np.load(f"{grid_dir}/s_values.npy")
        n_values_prev = np.load(f"{grid_dir}/n_values.npy")
        s_values_new = s_values_prev + s_values
        n_values_new = np.sqrt(n_values_prev**2 + n_values**2)
        np.save(f"{grid_dir}/s_values_new.npy", s_values_new)
        np.save(f"{grid_dir}/n_values_new.npy", n_values_new)
