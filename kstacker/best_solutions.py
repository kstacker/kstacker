"""
Script used to compute the SNR values on each sub-grid of the total grid. It
selects the q best values of SNR that will be re-optimized.
"""

__author__ = "Mathias Nowak, Dimitri Estevez, Le Coroller"
__email__ = "mathias.nowak@ens-cachan.fr, destevez@lam.fr"
__status__ = "Development"


import numpy as np

from .utils import create_output_dir


def find_best_solutions(params):
    grid_dir = params.get_path("grid_dir")
    values_dir = params.get_path("values_dir")
    create_output_dir(values_dir)

    q = params.q  # number of maxima that will be re-optimized

    if params.adding == "yes":
        signal = np.load(f"{grid_dir}/s_values_new.npy")
        noise = np.load(f"{grid_dir}/n_values_new.npy")
    elif params.adding == "no":
        signal = np.load(f"{grid_dir}/s_values.npy")
        noise = np.load(f"{grid_dir}/n_values.npy")

    sub_fun_values = -(signal / noise)
    np.save(f"{values_dir}/fun_values.npy", sub_fun_values)

    sub_grid = np.load(f"{grid_dir}/grid.npy")
    np.save(f"{values_dir}/grid.npy", sub_grid)

    flat_values = np.ndarray.flatten(sub_fun_values)
    # Sort the SNR
    sorted_values = np.sort(flat_values)

    res_grid = []
    l = 0
    while l < q:  # we save the q best orbits
        # find the position k-th best value among the snr results
        indices = np.array(np.where(sub_fun_values == sorted_values[l])).T
        # print("indices", indices.ravel())
        snr_val = sorted_values[l]  # get the corresponding snr value
        for ind in indices:
            # some values might be equals (i.e. k-th best snr value found on
            # different orbits) in this case we want all these orbits

            # get the corresponding orbit: a, e, t0, omega, i, theta_0
            x_val = sub_grid[:, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5]]
            # Write in one file the q best SNR with their orbital parameters
            # (on each line; nfiles sorted for n nodes)
            res_grid.append([snr_val, *x_val])
            l = l + 1

    np.save(f"{values_dir}/res_grid.npy", res_grid)
