"""
Script used to put the sub-grid together and re-optimize the q best values of
SNR with a gradient descent method (L-BFGS-B).
"""

import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from astropy.io import ascii, fits
from astropy.visualization import ZScaleInterval
from joblib import Parallel, delayed

from .imagerie import recombine_images
from .orbit import orbit, plot_ontop, plot_orbites
from .snr import compute_snr, compute_snr_detailed
from .utils import read_results


def plot_coadd(idx, coadded, x, params, outdir):
    a, e, t0, m0, omega, i, theta_0 = x
    # plot the corresponding image and save it as a png (for quick view)
    plt.figure()
    vmin, vmax = ZScaleInterval().get_limits(coadded)
    plt.imshow(
        coadded, origin="lower", interpolation="none", cmap="gray", vmin=vmin, vmax=vmax
    )
    plt.colorbar()
    xa, ya = orbit.project_position_full(t0, a, e, t0, m0, omega, i, theta_0)
    xpix = params.n // 2 + params.scale * xa
    ypix = params.n // 2 + params.scale * ya
    # comment this line if you don't want to see where the planet is recombined:
    # decalage 2 fwhm Antoine Schneeberger
    plt.scatter(ypix - 2 * params.fwhm, xpix, color="b", marker=">")
    plt.savefig(f"{outdir}/fin_png/fin_{idx}.png")
    plt.close()

    fits.writeto(f"{outdir}/fin_fits/fin_{idx}.fits", coadded, overwrite=True)


def make_plots(x_best, k, params, images, ts, values_dir, args):
    print(f"Make plots for solution {k+1}")
    # create combined images (for the q eme best SNR)
    coadded = recombine_images(images, ts, params.scale, *x_best)
    plot_coadd(k, coadded, x_best, params, values_dir)

    # plot the orbits
    ax = [params.xmin, params.xmax, params.ymin, params.ymax]
    plot_orbites(ts, x_best, ax, f"{values_dir}/orbites/orbites{k}")

    # If single_plot=='yes' a cross is ploted on each image where the
    # planet is found (by default no);
    if params.single_plot == "yes":
        for l in range(len(ts)):
            plot_ontop(
                x_best,
                params.dist,
                [ts[l]],
                params.resol,
                images[l],
                f"{values_dir}/single/single_{k}fin_{l}",
            )


def compute_snr_objfun(x, ts, size, scale, fwhm, data, invvar_weighted, r_mask):
    """Function to minimize, returns -SNR."""
    return -compute_snr(
        x,
        ts,
        size,
        scale,
        fwhm,
        data,
        invvar_weighted=invvar_weighted,
        exclude_source=True,
        exclude_lobes=True,
        use_interp_bgnoise=False,
        r_mask=r_mask,
    )


def optimize_orbit(result, k, args, bounds):
    # get orbit and snr value before reoptimization for the k-th best value
    *x, signal, noise, snr_i = result
    snr_init = -compute_snr_objfun(x, *args)

    with np.printoptions(precision=3, suppress=True):
        print(f"init  {k}: {np.array(x)} => {snr_init:.2f} (aper) {snr_i:.2f} (conv)")

    # Gradient re-optimization:
    opt_result = scipy.optimize.minimize(
        compute_snr_objfun,
        x,
        args=args,
        method="L-BFGS-B",
        bounds=bounds,
        options={"gtol": 1e-5},
    )
    x_best = opt_result.x
    snr_best = -opt_result.fun

    with np.printoptions(precision=3, suppress=True):
        print(f"reopt {k}: {x_best} => {snr_best:.2f}", flush=True)

    return snr_i, snr_best, *x_best


def reoptimize_gradient(params, n_jobs=1, n_orbits=None):
    # We sort the results in several directories
    values_dir = params.get_path("values_dir")
    os.makedirs(f"{values_dir}/fin_fits", exist_ok=True)
    os.makedirs(f"{values_dir}/fin_png", exist_ok=True)
    os.makedirs(f"{values_dir}/orbites", exist_ok=True)
    os.makedirs(f"{values_dir}/single", exist_ok=True)

    ts = params.get_ts()  # time of observations (years)
    size = params.n  # number of pixels
    data = params.load_data(method="aperture")

    with h5py.File(f"{values_dir}/res_grid.h5") as f:
        # note: results are already sorted by decreasing SNR
        results = f["Best solutions"][:]

    if n_orbits is not None:
        results = results[:n_orbits]
    else:
        n_orbits = params.q

    # define bounds
    bounds = params.grid.bounds()

    # Computation on the q best SNR
    args = (
        ts,
        size,
        params.scale,
        params.fwhm,
        data,
        params.invvar_weight,
        params.r_mask,
    )
    reopt = Parallel(n_jobs=n_jobs)(
        delayed(optimize_orbit)(results[k], k, args, bounds) for k in range(n_orbits)
    )
    # Sort values with the recomputed SNR
    reopt = np.array(reopt)
    reopt = reopt[np.argsort(reopt[:, 1])[::-1]]
    # Add index column
    reopt = np.concatenate([np.arange(reopt.shape[0])[:, None], reopt], axis=1)

    names = ("image_number", "snr_brut_force", "snr_gradient") + params.grid.grid_params
    ascii.write(
        reopt,
        f"{values_dir}/results.txt",
        names=names,
        format="fixed_width_two_line",
        formats={"image_number": "%d"},
        overwrite=True,
    )

    Parallel(n_jobs=n_jobs)(
        delayed(make_plots)(
            reopt[k, 3:], k, params, data["images"], ts, values_dir, args
        )
        for k in range(n_orbits)
    )

    print("Done!")


def compute_detailed_positions(
    params,
    method=None,
    invvar_weighted=False,
    exclude_source=True,
    exclude_lobes=True,
    use_interp_bgnoise=False,
    verbose=False,
):
    """Recompute the positions,signal,noise in each image."""
    values_dir = params.get_path("values_dir")
    tbl = read_results(f"{values_dir}/results.txt", params)
    x = tbl[["a", "e", "t0", "m0", "omega", "i", "theta_0"]].as_array()
    x = x.view((float, len(x.dtype.names)))
    res = compute_snr_detailed(
        params,
        x,
        method=method,
        invvar_weighted=invvar_weighted,
        exclude_source=exclude_source,
        exclude_lobes=exclude_lobes,
        use_interp_bgnoise=use_interp_bgnoise,
        verbose=verbose,
    )
    res.write(
        f"{values_dir}/results_detailed.txt",
        format="ascii.fixed_width_two_line",
        overwrite=True,
    )
