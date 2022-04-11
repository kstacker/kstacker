"""
Core script used to compute the values of the signal and the noise on a given
part of the total grid (will be run on several nodes). A brute force algorithm
is used.
"""

import numpy as np
from astropy.io import fits

from .utils import compute_signal_and_noise_grid, create_output_dir, get_image_suffix

__author__ = "Mathias Nowak, Dimitri Estevez"
__email__ = "mathias.nowak@ens-cachan.fr, destevez@lam.fr"
__status__ = "Development"


def brute_force(params):
    # name of the directory where one loads and saves the images and values
    images_dir = params.get_path("images_dir")
    profile_dir = params.get_path("profile_dir")
    grid_dir = params.get_path("grid_dir")
    values_dir = params.get_path("values_dir")
    create_output_dir(grid_dir)

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
    img_suffix = get_image_suffix(params.method)
    for k in range(nimg):
        i = k + p_prev
        images.append(fits.getdata(f"{images_dir}/image_{i}{img_suffix}.fits"))
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
        params.fwhm,
        images,
        x_profile,
        bkg_profiles,
        noise_profiles,
        params.upsampling_factor,
        params.r_mask,
        params.method,
    )

    res = params.grid.evaluate(
        compute_signal_and_noise_grid, args=args, nchunks=params.nchunks
    )

    if params.adding == "no":
        np.save(f"{grid_dir}/res.npy", res)
    elif params.adding == "yes":
        np.save(f"{grid_dir}/res_add.npy", res)
        prev = np.load(f"{grid_dir}/res.npy")
        res[:, 6] += prev[:, 6]  # signal
        res[:, 7] = np.sqrt(res[:, 6]**2 + prev[:, 6]**2)  # noise
        res[:, 8] = - res[:, 6] / res[:, 7]  # recompute SNR
        np.save(f"{grid_dir}/res_new.npy", res)

    # Sort on the SNR column and store the q best results
    ind = np.argsort(res[:, 8])
    best = res[ind[: params.q]]
    # put SNR as the first column and remove signal & noise
    best = np.concatenate([best[:, 8:], best[:, :6]], axis=1)
    np.save(f"{values_dir}/res_grid.npy", best)
