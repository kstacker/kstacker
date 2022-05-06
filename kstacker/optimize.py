"""
Core script used to compute the values of the signal and the noise on a given
part of the total grid (will be run on several nodes). A brute force algorithm
is used.
"""

import h5py
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack

from .imagerie import photometry, photometry_preprocessed
from .orbit import orbit
from .utils import create_output_dir, get_image_suffix

__author__ = "Mathias Nowak, Dimitri Estevez"
__email__ = "mathias.nowak@ens-cachan.fr, destevez@lam.fr"
__status__ = "Development"


def reject_invalid_orbits(orbital_grid, projection_grid, m0):
    a, e, t0 = orbital_grid.T
    omega, i, theta_0 = projection_grid.T

    print("Rejecting invalid orbits:")
    rej = t0 <= -np.sqrt((a**3.0) / m0)
    nrej = np.count_nonzero(rej)
    if nrej:
        print(f"- {nrej:,} rejected because t0 <= -np.sqrt(a**3 / starMass)")
        orbital_grid = orbital_grid[~rej]

    i_0_pi = np.isclose(i, 0) | np.isclose(i, 3.14)
    theta_non_null = ~np.isclose(theta_0, 0)
    rej = i_0_pi & theta_non_null
    nrej = np.count_nonzero(rej)
    if nrej:
        print(f"- {nrej:,} rejected because (i = 0 or i = 3.14) and theta0 != 0")
        projection_grid = projection_grid[~rej]

    # if e == 0. and (theta0 != 0. or omega !=0.):
    #    # JE NE COMPREND PAS CETTE SOLUTION DE LOUIS-XAVIER !!
    #    print('One orbit rejected because e == 0. and (theta0 != 0. or omega !=0.)')
    #    return True

    # e_null = np.isclose(e, 0)
    # rej2 = e_null & theta_non_null
    # nrej2 = np.count_nonzero(rej2)
    # rej |= rej2
    # if nrej2:
    #     print(f"- {nrej2:,} rejected because e = 0 and (theta0 != 0 or omega !=0)")

    # omega_non_null = ~np.isclose(omega, 0)
    # rej2 = (e_null & i_0_pi) & (theta_non_null | omega_non_null)
    # nrej2 = np.count_nonzero(rej2)
    # rej |= rej2
    # if nrej2:
    #     print(
    #         f"- {nrej2:,} rejected because "
    #         "(e = 0 and (i = 0 or i = 3.14)) and (theta0 != 0 or omega != 0)"
    #     )

    # print(f"Rejecting {np.count_nonzero(rej):,} orbits out of {grid.shape[0]:,}")
    # return rej


def evaluate(
    grid,
    ts,
    m0,
    size,
    scale,
    fwhm,
    images,
    x_profile,
    bkg_profiles,
    noise_profiles,
    upsampling_factor,
    r_mask,
    method,
    outfile,
    nchunks=1,
    dtype_index=np.int32,
):
    """Evaluate a function on the grid.

    Adapted from `scipy.optimize.brute`.

    Parameters
    ----------
    args : tuple, optional
        Any additional fixed parameters needed to completely specify
        the function.

    Returns
    -------
    grid : tuple
        Representation of the evaluation grid. It has the same
        length as `x0`.
    Jout : ndarray
        Function values at each point of the evaluation
        grid, i.e., ``Jout = func(*grid)``.

    """

    orbital_grid = grid.make_2d_grid(("a", "e", "t0"))
    projection_grid = grid.make_2d_grid(("omega", "i", "theta_0"))
    print(f"Orbital grid: {orbital_grid.shape[0]:,} x {orbital_grid.shape[1]}")
    print(f"Projection grid: {projection_grid.shape[0]:,} x {projection_grid.shape[1]}")

    # skip invalid/redundant orbits
    reject_invalid_orbits(orbital_grid, projection_grid, m0)

    with h5py.File(outfile, "w") as f:
        f["Orbital grid"] = orbital_grid
        f["Projection grid"] = projection_grid

    # solve kepler equation on the a/e/t0 grid
    positions = orbit.positions_at_multiple_times(ts, orbital_grid, m0)

    # (2, Nimages, Norbits) -> (Norbits, Nimages, 2)
    positions = np.transpose(positions)

    orbital_grid_index = np.arange(orbital_grid.shape[0], dtype=dtype_index)
    orbital_grid = None
    omega, i, theta_0 = projection_grid.T
    projection_grid_index = np.arange(projection_grid.shape[0], dtype=dtype_index)
    projection_grid = None

    res = []
    res_names = ("orbit index", "projection index", "signal", "noise")

    for j in orbital_grid_index:
        signal, noise = [], []
        for k in range(len(images)):
            position = orbit.project_position(positions[j, k], omega, i, theta_0).T
            xx, yy = position

            # convert position into pixel in the image
            position = scale * position + size // 2
            temp_d = np.sqrt(xx**2 + yy**2) * scale  # distance to the center

            # compute the signal by integrating flux on a PSF, and correct it for
            # background (using pre-computed background profile)
            if method == "convolve":
                sig = photometry_preprocessed(images[k], position, upsampling_factor)
            elif method == "aperture":
                sig = photometry(images[k], position, 2 * fwhm)
            else:
                raise ValueError(f"invalid method {method}")

            sig -= np.interp(temp_d, x_profile, bkg_profiles[k])

            if r_mask is not None:
                sig[temp_d <= r_mask] = 0.0

            signal.append(sig)

            # get noise at position using pre-computed radial noise profil
            noise.append(np.interp(temp_d, x_profile, noise_profiles[k]))

        signal = np.nansum(signal, axis=0)
        noise = np.sqrt(np.nansum(np.array(noise) ** 2, axis=0))
        # if the value of total noise is 0 (i.e. all values of noise are 0,
        # i.e. the orbit is completely out of the image) then snr=0
        noise[np.isnan(noise) | (noise == 0)] = 1

        columns = [
            np.full(signal.shape[0], j, dtype=dtype_index),
            projection_grid_index,
            signal,
            noise,
        ]
        res.append(Table(columns, names=res_names))

    res = vstack(res)
    res["snr"] = -res["signal"] / res["noise"]
    return res

    # Jout = []
    # for i, chunk in enumerate(np.array_split(grid, nchunks), start=1):
    #     print(f"- chunk {i}/{nchunks}")
    #     Jout.append(compute_signal_and_noise_grid())

    # Jout = np.concatenate(Jout, axis=1)
    # snr = -Jout[0] / Jout[1]
    # return np.concatenate([grid, Jout.T, snr[:, None]], axis=1)


def brute_force(params):
    # name of the directory where one loads and saves the images and values
    images_dir = params.get_path("images_dir")
    profile_dir = params.get_path("profile_dir")
    grid_dir = params.get_path("grid_dir")
    values_dir = params.get_path("values_dir")
    create_output_dir(grid_dir)
    create_output_dir(values_dir)

    # total time of the observation (years)
    total_time = float(params["total_time"])
    nimg = params["p"]  # number of timesteps
    p_prev = params["p_prev"]

    if total_time == 0:
        ts = [float(x) for x in params["time"].split("+")]
        ts = np.array(ts[p_prev:])
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

    if params.adding == "no":
        outfile = f"{grid_dir}/res.h5"
    elif params.adding == "yes":
        outfile = f"{grid_dir}/res_add.h5"

    # brute force
    res = evaluate(
        params.grid,
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
        outfile,
        nchunks=params.nchunks,
    )

    res.write(outfile, path="Full SNR", append=True)

    if params.adding == "yes":
        prev = Table.read(f"{grid_dir}/res.npy", path="Full SNR")
        res["signal"] += prev["signal"]  # signal
        res["noise"] = np.sqrt(res["noise"] ** 2 + prev["noise"] ** 2)  # noise
        res["snr"] = -res["signal"] / res["noise"]  # recompute SNR
        res.write(f"{grid_dir}/res_new.npy", path="Full SNR", append=True)

    # Sort on the SNR column and store the q best results
    ind = np.argsort(res["snr"])
    res = res[ind[: params.q]]
    # TODO: add a,e,t0 etc.
    # remove signal & noise
    res.remove_columns(("signal", "noise"))
    res.write(f"{values_dir}/res_grid.h5", path="Best solutions")
