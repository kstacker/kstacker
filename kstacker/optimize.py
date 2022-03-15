"""
Core script used to compute the values of the signal and the noise on a given
part of the total grid (will be run on several nodes). A brute force algorithm
is used.
"""

import math
import os
import sys

import numpy as np
import scipy.optimize
from astropy.io import fits

from .imagerie.analyze import photometry
from .orbit import orbit as orb
from .utils import get_path

__author__ = "Mathias Nowak, Dimitri Estevez"
__email__ = "mathias.nowak@ens-cachan.fr, destevez@lam.fr"
__status__ = "Development"


def compute_signal(x, ts, m0, n, scale, images, fwhm, x_profile, bkg_profiles, r_mask):
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

        if temp_d <= r_mask:
            res[k] = 0.0
        else:
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


def split_ranges(val_min, val_max, N_origin, split):
    """
    Split all the range in sub ranges
    :param val_min: min of the range
    :param val_max: max of the range
    :param N_origin: original number of steps
    :param split: Number of split wanted
    :return: a list of slice objects, that are the sub_ranges for the brut force
    """
    n = N_origin / split

    if N_origin == 1:
        return np.array([slice(val_min, val_max, val_max - val_min)], dtype=object)
    if n < 2.0:
        print("Number of values in each splited list under 2 : ABORTING")
        sys.exit()
    elif int(n) != n:
        print("n not int")
        if n - int(n) >= 0.5:
            n = float(int(n)) + 1.0
        elif n - int(n) < 0.5:
            n = float(int(n))
    elif int(n) == n:
        print("N_origin/split is int")
    print("N_origin/split = ", n)
    print(val_min, val_max, N_origin, n, split)
    table = np.zeros(int(split), dtype=object)
    delta = (val_max - val_min) / split
    print(delta)
    x = val_min
    for k in range(0, int(split)):
        # Dans version Antoine (bug):
        # if x==0.2 or delta==0.2 and x==0.4 or delta==0.4:
        # Pyhon decid that 0.2+0.4=6.000000001 and not 6.0
        # --> Split version is not working with this ><'
        #        x_max=0.6
        # else :
        #        x_max=x+delta
        x_max = round(x + delta, 12)
        print(x_max)
        table[k] = slice(x, x_max, delta / n)
        x = x_max
    return table


def Table(a_list, e_list, t0_list, omega_list, i_list, theta0_list):
    """
    Creat the corresponding table
    :param a_list: List from the split fonction of a
    :param e_list: List from the split fonction of e
    :param t0_list: List from the split fonction of t0
    :param omega_list: List from the split fonction of omega
    :param i_list: List from the split fonction of i
    :param theta0_list: List from the split fonction of theta0
    :return: the corresponding table that is a matrix 2D of silce object, where each line in a range for the brut force
    """
    Sa = np.shape(a_list)[0]
    Se = np.shape(e_list)[0]
    St0 = np.shape(t0_list)[0]
    Somega = np.shape(omega_list)[0]
    Si = np.shape(i_list)[0]
    Stheta0 = np.shape(theta0_list)[0]
    table = np.zeros((Sa * Se * Stheta0 * St0 * Somega * Si, 6), dtype=object)
    K = 0
    for ka in range(Sa):
        for ke in range(Se):
            for kt0 in range(St0):
                for komega in range(Somega):
                    for ki in range(Si):
                        for ktheta0 in range(Stheta0):
                            table[K, :] = np.array(
                                [
                                    a_list[ka],
                                    e_list[ke],
                                    t0_list[kt0],
                                    omega_list[komega],
                                    i_list[ki],
                                    theta0_list[ktheta0],
                                ],
                                dtype=object,
                            )
                            K = K + 1
    return table


def brute_force(params):
    # name of the directory where one loads and saves the images and values
    images_dir = get_path(params, "images_dir")
    profile_dir = get_path(params, "profile_dir")
    grid_dir = get_path(params, "grid_dir")
    # values_dir = get_path(params, "values_dir")
    id_number = int(params["id_number"])

    m0 = float(params["m0"])

    # parameters of the brute force search grid.
    # Format is [min value, max value, number of points].
    a_min, a_max, Na, Sa = [
        float(params["a_min"]),
        float(params["a_max"]),
        float(params["Na"]),
        float(params["Sa"]),
    ]  # (A.U)
    e_min, e_max, Ne, Se = [
        float(params["e_min"]),
        float(params["e_max"]),
        float(params["Ne"]),
        float(params["Se"]),
    ]
    t0_min, t0_max, Nt0, St0 = [
        float(params["t0_min"]),
        float(params["t0_max"]),
        float(params["Nt0"]),
        1,
    ]  # (years)
    omega_min, omega_max, Nomega, Somega = [
        float(params["omega_min"]),
        float(params["omega_max"]),
        float(params["Nomega"]),
        float(params["Somega"]),
    ]
    i_min, i_max, Ni, Si = [
        float(params["i_min"]),
        float(params["i_max"]),
        float(params["Ni"]),
        float(params["Si"]),
    ]
    theta_0_min, theta_0_max, Ntheta_0, Stheta0 = [
        float(params["theta_0_min"]),
        float(params["theta_0_max"]),
        float(params["Ntheta_0"]),
        float(params["Stheta0"]),
    ]

    print([a_min, a_max, Na])

    # total time of the observation (years)
    total_time = float(params["total_time"])
    p = params["p"]  # number of timesteps
    p_prev = params["p_prev"]

    if total_time == 0:
        time = [float(x) for x in params["time"].split("+")]
        ts = time[p_prev:]
    else:
        ts = np.linspace(0, total_time, p + p_prev)
    print(time)

    # instrument parameters
    dist = float(params["dist"])  # distance to the star (parsec)
    wav = float(params["wav"])  # wavelength of observation (meter)
    d = float(params["d"])  # diameter of the primary miror (meter)
    resol = float(params["resol"])  # resolution in marsec/pixel
    fwhm = (
        (1.028 * wav / d) * (180.0 / np.pi) * 3600 / (resol / 1000.0)
    )  # apodized fwhm of the PSF (in pixels)

    # image parameters
    r_mask = float(params["r_mask"])  # radius of the inner mask in pixels
    n = int(params["n"])  # number of pixels in one direction
    # scale factor used to convert pixel to astronomical unit (in pixel/a.u.)
    scale = 1.0 / (dist * (resol / 1000.0))

    adding = params["adding"]  # addition of images to a previous run
    temporary_files = params["Temporary"]
    restart = params["restart"]

    # initialization
    x_profile = np.linspace(0, n / 2 - 1, n / 2)

    # Path definition

    # if os.path.exists(grid_dir):
    #    print 'brute_grid directory exist !'
    # else:
    #    os.mkdir(grid_dir)

    # load the images .fits or .txt and the noise profiles
    images, bkg_profiles, noise_profiles = [], [], []
    for k in range(p):
        i = k + p_prev
        images.append(fits.getdata(f"{images_dir}/image_{i}_preprocessed.fits"))
        bkg_profiles.append(np.load(f"{profile_dir}/background_prof{i}.npy"))
        noise_profiles.append(np.load(f"{profile_dir}/noise_prof{i}.npy"))

    # grid on which the brute force algorithm will be computed on one node/core
    ranges = (
        slice(a_min, a_max, (a_max - a_min) / Na),
        slice(e_min, e_max, (e_max - e_min) / Ne),
        slice(t0_min, t0_max, (t0_max - t0_min) / Nt0),
        slice(omega_min, omega_max, (omega_max - omega_min) / Nomega),
        slice(i_min, i_max, (i_max - i_min) / Ni),
        slice(theta_0_min, theta_0_max, (theta_0_max - theta_0_min) / Ntheta_0),
    )

    print("[a, e, t0, omega, i, theta_0]")

    # brute force
    args_signal = (ts, m0, n, scale, images, fwhm, x_profile, bkg_profiles, r_mask)
    args_noise = (ts, m0, scale, x_profile, noise_profiles)

    if adding == "no":
        if temporary_files == "no":
            opt_result = scipy.optimize.brute(
                compute_signal,
                ranges=ranges,
                args=args_signal,
                full_output=True,
                finish=None,
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
            )
            n_values = opt_result[3]  # get the values of noise on the grid

            np.save(f"{grid_dir}/s_values{id_number}.npy", s_values)
            np.save(f"{grid_dir}/grid{id_number}.npy", grid)
            np.save(f"{grid_dir}/n_values{id_number}.npy", n_values)
        elif temporary_files == "yes":
            path = grid_dir + "/core_" + str(id_number)
            if restart == "no":  # creation of the table
                a_list = split_ranges(a_min, a_max, Na, Sa)
                e_list = split_ranges(e_min, e_max, Ne, Se)
                t0_list = split_ranges(t0_min, t0_max, Nt0, St0)
                omega_list = split_ranges(omega_min, omega_max, Nomega, Somega)
                i_list = split_ranges(i_min, i_max, Ni, Si)
                theta0_list = split_ranges(theta_0_min, theta_0_max, Ntheta_0, Stheta0)
                table = Table(a_list, e_list, t0_list, omega_list, i_list, theta0_list)
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
                )
                grid = opt_result[2]
                s_values = opt_result[3]
                opt_result = scipy.optimize.brute(
                    compute_noise,
                    ranges=ranges,
                    args=args_noise,
                    full_output=True,
                    finish=None,
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
