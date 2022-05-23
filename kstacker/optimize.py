"""
Core script used to compute the values of the signal and the noise on a given
part of the total grid (will be run on several nodes). A brute force algorithm
is used.
"""

import time

import h5py
import numpy as np
from astropy.io import fits

from ._utils import compute_snr
from .orbit import orbit
from .utils import create_output_dir, get_image_suffix

__author__ = "Mathias Nowak, Dimitri Estevez"
__email__ = "mathias.nowak@ens-cachan.fr, destevez@lam.fr"
__status__ = "Development"


def reject_invalid_orbits(orbital_grid, projection_grid, m0):
    a, e, t0 = orbital_grid.T
    omega, i, theta_0 = projection_grid.T
    proj_shape = projection_grid.shape[0]
    norbits = orbital_grid.shape[0] * proj_shape

    def _fmt(n):
        """Format number with thousands separator."""
        return f"{n:,}".replace(",", "'")

    print("Rejecting invalid orbits:")
    print(f"- {_fmt(norbits)} orbits before rejection")
    rej = t0 <= -np.sqrt((a**3.0) / m0)
    nrej = np.count_nonzero(rej) * proj_shape
    if nrej:
        print(f"- {_fmt(nrej)} rejected because t0 <= -np.sqrt(a**3 / starMass)")
        orbital_grid = orbital_grid[~rej]

    i_0_pi = np.isclose(i, 0) | np.isclose(i, 3.14)
    theta_non_null = ~np.isclose(theta_0, 0)
    rej = i_0_pi & theta_non_null
    nrej = np.count_nonzero(rej) * proj_shape
    if nrej:
        print(f"- {_fmt(nrej)} rejected because (i = 0 or i = 3.14) and theta0 != 0")
        projection_grid = projection_grid[~rej]

    # if e == 0. and (theta0 != 0. or omega !=0.):
    #    # JE NE COMPREND PAS CETTE SOLUTION DE LOUIS-XAVIER !!
    #    print('One orbit rejected because e == 0. and (theta0 != 0. or omega !=0.)')
    #    return True

    # The two next conditions apply to e=0, but since the two orbital and
    # projection grids are separate we keep the boolean condition array to
    # apply it later.

    omega, i, theta_0 = projection_grid.T
    i_0_pi = np.isclose(i, 0) | np.isclose(i, 3.14)
    theta_non_null = ~np.isclose(theta_0, 0)
    omega_non_null = ~np.isclose(omega, 0)

    count_e_null = np.count_nonzero(np.isclose(e, 0))
    # e_null = np.isclose(e, 0)
    # rej2 = e_null & theta_non_null
    rej_proj = theta_non_null
    nrej = np.count_nonzero(rej_proj) * count_e_null
    if nrej:
        print(f"- {_fmt(nrej)} rejected because e = 0 and (theta0 != 0 or omega !=0)")

    # rej2 = (e_null & i_0_pi) & (theta_non_null | omega_non_null)
    rej2 = i_0_pi & (theta_non_null | omega_non_null)
    nrej = np.count_nonzero(rej2) * count_e_null
    rej_proj |= rej2
    if nrej:
        print(
            f"- {_fmt(nrej)} rejected because "
            "(e = 0 and (i = 0 or i = 3.14)) and (theta0 != 0 or omega != 0)"
        )

    nrej = np.count_nonzero(rej_proj)
    norbits = orbital_grid.shape[0] * projection_grid.shape[0] - nrej
    print(f"- {_fmt(norbits)} orbits after rejection")
    return orbital_grid, projection_grid, ~rej_proj


def evaluate(
    params,
    ts,
    size,
    images,
    x_profile,
    bkg_profiles,
    noise_profiles,
    outfile,
    nbest,
    nchunks=1,
    dtype_index=np.int32,
    dry_run=False,
    num_threads=0,
    show_progress=False,
):
    """Compute SNR for all the orbits."""

    orbital_grid = params.grid.make_2d_grid(("a", "e", "t0"))
    projection_grid = params.grid.make_2d_grid(("omega", "i", "theta_0"))
    print(f"Orbital grid: {orbital_grid.shape[0]:,} x {orbital_grid.shape[1]}")
    print(f"Projection grid: {projection_grid.shape[0]:,} x {projection_grid.shape[1]}")

    # skip invalid/redundant orbits
    orbital_grid, projection_grid, valid_proj = reject_invalid_orbits(
        orbital_grid, projection_grid, params.m0
    )

    if dry_run:
        return

    # solve kepler equation on the a/e/t0 grid for all images
    positions = orbit.positions_at_multiple_times(ts, orbital_grid, params.m0)
    # (2, Nimages, Norbits) -> (Norbits, Nimages, 2)
    positions = np.ascontiguousarray(np.transpose(positions))

    norbits = orbital_grid.shape[0]
    e_null = np.isclose(orbital_grid[:, 1], 0)

    # pre-compute the projection matrices
    omega, i, theta_0 = projection_grid.T
    proj_matrices = orbit.compute_projection_matrices(omega, i, theta_0)
    proj_matrices = np.ascontiguousarray(proj_matrices)
    omega = i = theta_0 = None

    # Save results every nsave iterations
    nsave = min(norbits, 10_000)
    isave = 0
    idata = 0
    out_full = np.empty((nbest * nsave, 9), dtype=np.float32)
    t0 = time.time()

    with h5py.File(outfile, "w") as f:
        f["Orbital grid"] = orbital_grid
        f["Projection grid"] = projection_grid
        data = f.create_dataset(
            "DATA", dtype=np.float32, shape=(norbits * nbest, 9), chunks=(nbest, 9)
        )

        # For each iteration:
        # - compute the SNR for 'nvalid' projections
        # - keep the 'nbest' best results
        # Then:
        # - concatenate the results (a, e, t0, omega, i, theta0, signal, noise,
        #   snr) in memory for 'nsave' iterations (in 'out_full') to avoid to
        #   many small writes
        # - every 'nsave' iterations, save 'out_full' to the HDF5 file

        for j in range(norbits):
            # reject more invalid orbits for the e=0 case
            valid = valid_proj if e_null[j] else slice(None)
            proj_mat = proj_matrices[valid]

            nvalid = proj_mat.shape[0]
            if nvalid == 0:
                continue
            out = np.zeros((nvalid, 3))
            compute_snr(
                images,
                positions[j],
                bkg_profiles,
                noise_profiles,
                proj_mat,
                params.r_mask,
                params.scale,
                size,
                params.upsampling_factor,
                out,
                num_threads=num_threads,
            )

            # Sort by SNR
            ind = np.argpartition(out[:, 2], nbest)[:nbest]

            # Save best results in out_full
            sl = slice(isave * nbest, (isave + 1) * nbest)
            out_full[sl, :3] = orbital_grid[j]  # a, e, t0
            out_full[sl, 3:6] = projection_grid[valid][ind]  # omega, i, theta0
            out_full[sl, 6:] = out[ind]  # signal, noise, snr
            isave += 1

            if isave == nsave:
                # write to disk
                data[idata * nbest * nsave : (idata + 1) * nbest * nsave] = out_full
                idata += 1
                isave = 0
                if show_progress:
                    tt = time.time() - t0
                    remaining = tt * (norbits / (idata * nsave) - 1)
                    print(
                        f"- {idata*nsave}/{norbits}, {tt:.2f} sec., remains"
                        f" {remaining:.2f} sec.",
                        flush=True,
                    )

        if isave > 0:
            # write the remaining orbits (isave < nsave) to disk
            istart = idata * nbest * nsave
            size = isave * nbest
            data[istart : istart + size] = out_full[:size]


def brute_force(params, dry_run=False, num_threads=0, show_progress=False):
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
    print("time_vector:", ts)

    # grid on which the brute force algorithm will be computed on one node/core
    print(repr(params.grid))

    size = int(params["n"])  # number of pixels in one direction
    x_profile = np.linspace(0, size // 2 - 1, size // 2)

    # load the images .fits or .txt and the noise profiles
    images, bkg_profiles, noise_profiles = [], [], []
    img_suffix = get_image_suffix(params.method)
    for k in range(nimg):
        i = k + p_prev
        im = fits.getdata(f"{images_dir}/image_{i}{img_suffix}.fits")
        images.append(im.astype("float", order="C", copy=False))
        bkg_profiles.append(np.load(f"{profile_dir}/background_prof{i}.npy"))
        noise_profiles.append(np.load(f"{profile_dir}/noise_prof{i}.npy"))

    images = np.array(images)
    bkg_profiles = np.array(bkg_profiles)
    noise_profiles = np.array(noise_profiles)

    if params.adding == "no":
        outfile = f"{grid_dir}/res.h5"
    elif params.adding == "yes":
        outfile = f"{grid_dir}/res_add.h5"

    # brute force
    evaluate(
        params,
        ts,
        size,
        images,
        x_profile,
        bkg_profiles,
        noise_profiles,
        outfile,
        params.q,
        nchunks=params.nchunks,
        dry_run=dry_run,
        num_threads=num_threads,
        show_progress=show_progress,
    )

    if dry_run:
        return

    # if params.adding == "yes":
    #     prev = Table.read(f"{grid_dir}/res.npy", path="Full SNR")
    #     res["signal"] += prev["signal"]  # signal
    #     res["noise"] = np.sqrt(res["noise"] ** 2 + prev["noise"] ** 2)  # noise
    #     res["snr"] = -res["signal"] / res["noise"]  # recompute SNR
    #     res.write(f"{grid_dir}/res_new.npy", path="Full SNR", append=True)

    # Sort on the SNR column and store the q best results
    with h5py.File(outfile, "r") as f:
        res = f["DATA"][:]

    ind = np.argsort(res[:, 8])
    res = res[ind]

    with h5py.File(f"{values_dir}/res_grid.h5", "w") as f:
        f["Best solutions"] = res[: params.q]
