"""
Core script used to compute the values of the signal and the noise on a given
part of the total grid (will be run on several nodes). A brute force algorithm
is used.
"""

import time

import h5py
import numpy as np

from ._utils import cy_compute_snr
from .orbit import orbit


def reject_invalid_orbits(orbital_grid, projection_grid):
    a, e, t0, m0 = orbital_grid.T
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
    outfile,
    dry_run=False,
    num_threads=0,
    show_progress=False,
):
    """Compute SNR for all the orbits."""

    orbital_grid = params.grid.make_2d_grid(("a", "e", "t0", "m0"))
    projection_grid = params.grid.make_2d_grid(("omega", "i", "theta_0"))
    print(f"Orbital grid: {orbital_grid.shape[0]:,} x {orbital_grid.shape[1]}")
    print(f"Projection grid: {projection_grid.shape[0]:,} x {projection_grid.shape[1]}")

    # skip invalid/redundant orbits
    orbital_grid, projection_grid, valid_proj = reject_invalid_orbits(
        orbital_grid, projection_grid
    )

    if dry_run:
        return

    if params.invvar_weight:
        print("Using inverse variance weights")

    # load the images and the noise/background profiles
    input_data = params.load_data()

    # total time of the observation (years)
    ts = params.get_ts(use_p_prev=True)

    # number of solutions to keep
    nbest = params.q

    # solve kepler equation on the a, e, t0, m0 grid for all images
    positions = orbit.positions_at_multiple_times(ts, orbital_grid)
    # (2, Nimages, Norbits) -> (Norbits, Nimages, 2)
    positions = np.ascontiguousarray(np.transpose(positions))

    e_null = np.isclose(orbital_grid[:, 1], 0)

    # pre-compute the projection matrices
    proj_matrices = orbit.compute_projection_matrices(*projection_grid.T)
    proj_matrices = np.ascontiguousarray(proj_matrices)

    norbits = orbital_grid.shape[0]
    nproj = projection_grid.shape[0]

    # Results are saved in chuncks of nsave to avoid keeping all results in memory
    nsave = int(1e8)  # 10 * 1e8 * 4 / 1e6 = 4Gb
    isave = 0
    idata = 0
    ncols = 10
    out_full = np.empty((nsave, ncols), dtype=np.float32)
    t0 = time.time()

    with h5py.File(outfile, "w") as f:
        f["Orbital grid"] = orbital_grid
        f["Projection grid"] = projection_grid
        data = f.create_dataset(
            "DATA",
            dtype=np.float32,
            shape=(norbits * nbest, ncols),
            chunks=(nbest, ncols),
            maxshape=(norbits * nproj, 10),
        )

        # For each iteration:
        # - compute the SNR for 'nvalid' projections
        # - keep the 'nbest' best results
        # Then:
        # - concatenate the results (a, e, t0, m0, omega, i, theta0, signal,
        #   noise, snr) in memory for 'nsave' iterations (in 'out_full') to
        #   avoid to many small writes
        # - every 'nsave' iterations, save 'out_full' to the HDF5 file

        for j in range(norbits):
            # reject more invalid orbits for the e=0 case
            valid = valid_proj if e_null[j] else slice(None)
            proj_mat = proj_matrices[valid]

            nvalid = proj_mat.shape[0]
            if nvalid == 0:
                continue
            out = np.zeros((nvalid, 3))
            cy_compute_snr(
                input_data["images"],
                positions[j],
                input_data["bkg"],
                input_data["noise"],
                proj_mat,
                params.r_mask,
                params.r_mask_ext,
                params.scale,
                params.n,
                params.upsampling_factor,
                out,
                num_threads,
                int(params.invvar_weight),
            )

            # Keep the nbest results, based on their SNR
            sel = out[:, 2] >= params.min_snr
            nkeep = np.count_nonzero(sel)

            if isave + nkeep > nsave:
                if idata + isave > data.shape[0]:
                    data.resize((1.2 * (idata + isave), ncols))
                # write to disk the results that have been computed so far
                data[idata : idata + isave] = out_full[:isave]
                idata += isave
                isave = 0
                if show_progress:
                    tt = time.time() - t0
                    remaining = tt * (norbits / j - 1)
                    print(
                        f"- {j}/{norbits}, {tt:.2f} sec., remains {remaining:.2f} sec.",
                        flush=True,
                    )

            # Save best results in out_full
            if nkeep > 0:
                sl = slice(isave, isave + nkeep)
                out_full[sl, :4] = orbital_grid[j]  # a, e, t0, m0
                out_full[sl, 4:7] = projection_grid[valid][sel]  # omega, i, theta0
                out_full[sl, 7:] = out[sel]  # signal, noise, snr
                isave += nkeep

        if isave > 0:
            if isave > data.shape[0]:
                data.resize((isave, ncols))

            # write the remaining orbits (isave < nsave) to disk
            data[idata : idata + isave] = out_full[:isave]
            idata += isave

        # resize the dataset to the actual number of results that have been computed
        data.resize((idata, ncols))

    print(f"Optimization is done, results saved in {outfile}", flush=True)


def extract_best_solutions(params, nbest=None):
    """Sort on the SNR column and store the q best results."""

    if nbest is None:
        nbest = params.q

    print("Reading SNR column")
    grid_dir = params.get_path("grid_dir")
    # To keep memory usage low we load only the SNR column
    with h5py.File(f"{grid_dir}/res.h5", "r") as f:
        snr = f["DATA"][:, 9]

    print("Sorting")
    ind = np.argsort(snr)  # sort by SNR
    ind = ind[-nbest:]  # keep nbest solutions
    ind = ind[::-1]  # reverse (decreasing SNR)
    snr = None

    print("Loading best solutions")
    # Now load all columns for the best solutions
    # H5Py needs a sorted index
    with h5py.File(f"{grid_dir}/res.h5", "r") as f:
        res = f["DATA"][np.sort(ind)]

    # Need to sort again by decreasing SNR
    ind = np.argsort(res[:, 9])[::-1]
    res = res[ind]

    values_dir = params.get_path("values_dir", remove_if_exist=True)
    outfile = f"{values_dir}/res_grid.h5"
    print(f"Saving {nbest} best solutions in {outfile}")
    with h5py.File(outfile, "w") as f:
        f["Best solutions"] = res


def brute_force(params, dry_run=False, num_threads=0, show_progress=False):
    # name of the directory where one loads and saves the images and values
    grid_dir = params.get_path("grid_dir", remove_if_exist=True)

    # grid on which the brute force algorithm will be computed on one node/core
    print(repr(params.grid))

    outfile = f"{grid_dir}/res.h5"
    # if params.adding == "yes":
    #     outfile = f"{grid_dir}/res_add.h5"

    # brute force
    evaluate(
        params,
        outfile,
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

    extract_best_solutions(params)
