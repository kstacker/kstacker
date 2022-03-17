"""
Core script used to compute the values of the signal and the noise on a given
part of the total grid (will be run on several nodes). A brute force algorithm
is used.
"""

import os
import sys
import time

import numpy as np
import scipy.optimize
from astropy.io import fits

from .imagerie.analyze import photometry
from .orbit import orbit as orb
from .utils import create_output_dir

__author__ = "Mathias Nowak, Dimitri Estevez"
__email__ = "mathias.nowak@ens-cachan.fr, destevez@lam.fr"
__status__ = "Development"


def compute_signal(x, ts, m0, n, scale, images, fwhm, x_profile, bkg_profiles, r_mask):
    """define signal"""
    # tstart = time.time()
    nimg = len(images)
    a, e, t0, omega, i, theta_0 = x
    res = np.zeros(nimg)  # res will contain signal for each image
    for k in range(nimg):
        x, y = orb.project_position(
            orb.position(ts[k], a, e, t0, m0), omega, i, theta_0
        )
        # convert position into pixel in the image
        position = [scale * x + n // 2, scale * y + n // 2]
        temp_d = np.sqrt(x**2 + y**2) * scale  # get the distance to the center

        if temp_d <= r_mask:
            res[k] = 0.0
        else:
            # compute the signal by integrating flux on a PSF, and correct it for
            # background (using pre-computed background profile)
            res[k] = photometry(images[k], position, 2 * fwhm) - np.interp(
                temp_d, x_profile, bkg_profiles[k]
            )
    # if the value of signal is nan outside the image, consider it to be 0
    res[np.isnan(res)] = 0.0
    # print(f"compute_signal: {time.time()-tstart:.2f} sec.")
    return np.sum(res), 0  # return sn


def compute_noise(x, ts, m0, scale, x_profile, noise_profiles):
    """define noise"""
    # tstart = time.time()
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

    # if the value of signal is nan outside the image, consider it to be 0
    res[np.isnan(res)] = 0.0

    noise = np.sqrt(np.sum(res**2))
    if noise == 0:
        # if the value of total noise is 0 (i.e. all values of noise are 0,
        # i.e. the orbit is completely out of the image) then snr=0
        noise = 1
    # print(f"compute_noise: {time.time()-tstart:.2f} sec.")
    return noise


def brute_force(params):
    # name of the directory where one loads and saves the images and values
    images_dir = params.get_path("images_dir")
    profile_dir = params.get_path("profile_dir")
    grid_dir = params.get_path("grid_dir")
    # values_dir = params.get_path("values_dir")

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
    ranges = params.grid.ranges()

    # brute force
    args_signal = (
        ts,
        params.m0,
        size,
        params.scale,
        images,
        params.fwhm,
        x_profile,
        bkg_profiles,
        params.r_mask,
    )
    args_noise = (ts, params.m0, params.scale, x_profile, noise_profiles)
    create_output_dir(grid_dir)

    adding = params["adding"]  # addition of images to a previous run
    id_number = params["id_number"]
    temporary_files = params["Temporary"]
    restart = params["restart"]

    if adding == "no":
        if temporary_files == "no":
            opt_result = scipy.optimize.brute(
                compute_signal,
                ranges=ranges,
                args=args_signal,
                full_output=True,
                finish=None,
                disp=True,
            )
            grid = opt_result[2]  # get the values of the grid
            print(np.shape(grid))
            s_values = opt_result[3]  # get the values of compute_signal on the grid
            print(np.shape(s_values))
            opt_result = scipy.optimize.brute(
                compute_noise,
                ranges=ranges,
                args=args_noise,
                full_output=True,
                finish=None,
                disp=True,
            )
            n_values = opt_result[3]  # get the values of noise on the grid

            np.save(f"{grid_dir}/s_values{id_number}.npy", s_values)
            np.save(f"{grid_dir}/grid{id_number}.npy", grid)
            np.save(f"{grid_dir}/n_values{id_number}.npy", n_values)
        elif temporary_files == "yes":
            path = f"{grid_dir}/core_{id_number}"

            if restart == "no":  # creation of the table
                table = params.grid.split_ranges()
                os.mkdir(path)
                np.save(f"{path}/Table.npy", table)
                deb = 0
            elif restart == "yes":
                # if there is a restart : load of the table and get the last line computed
                list_files = os.listdir(path)
                nb_file = len(list_files)
                table = np.load(f"{path}/Table.npy")
                # at each line there is 3 files created signal,noise and
                # grid-- number_of_files/3 and -1 to begin by the last compute for security
                deb = nb_file / 3 - 1

            for k in range(deb, np.shape(table)[0]):
                ranges = table[k, :]
                print(k)
                opt_result = scipy.optimize.brute(
                    compute_signal,
                    ranges=ranges,
                    args=args_signal,
                    full_output=True,
                    finish=None,
                    disp=True,
                )
                grid = opt_result[2]
                s_values = opt_result[3]
                opt_result = scipy.optimize.brute(
                    compute_noise,
                    ranges=ranges,
                    args=args_noise,
                    full_output=True,
                    finish=None,
                    disp=True,
                )
                n_values = opt_result[3]  # get the values of noise on the grid
                np.save(f"{path}/s_values{id_number}_{k}.npy", s_values)
                np.save(f"{path}/grid{id_number}_{k}.npy", grid)
                np.save(f"{path}/n_values{id_number}_{k}.npy", n_values)

    if adding == "yes":
        if temporary_files == "no":
            opt_result_add = scipy.optimize.brute(
                compute_signal,
                ranges=ranges,
                args=args_signal,
                full_output=True,
                finish=None,
            )
            # get the values of compute_signal on the grid
            s_values_add = opt_result_add[3]
            opt_result_add = scipy.optimize.brute(
                compute_noise,
                ranges=ranges,
                args=args_noise,
                full_output=True,
                finish=None,
            )
            n_values_add = opt_result_add[3]  # get the values of noise on the grid
            np.save(f"{grid_dir}/s_values_add{id_number}.npy", s_values_add)
            np.save(f"{grid_dir}/n_values_add{id_number}.npy", n_values_add)

            s_values_prev = np.load(f"{grid_dir}/s_values{id_number}.npy")
            n_values_prev = np.load(f"{grid_dir}/n_values{id_number}.npy")
            s_values_new = s_values_prev + s_values_add
            n_values_new = np.sqrt(n_values_prev**2 + n_values_add**2)
            np.save(f"{grid_dir}/s_values_new{id_number}.npy", s_values_new)
            np.save(f"{grid_dir}/n_values_new{id_number}.npy", n_values_new)
        elif temporary_files == "yes":
            print("Adding mode not developed for Temporary files mode")
