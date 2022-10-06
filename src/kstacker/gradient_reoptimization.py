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
from joblib import Parallel, delayed

from .imagerie import photometry, recombine_images
from .orbit import orbit as orb
from .orbit import plot_ontop, plot_orbites2


def get_res(x, ts, size, scale, fwhm, data, r_mask):
    """define snr function as a function of the orbit (used for the gradient;
    we maximise this function)
    """
    nimg = len(data["images"])
    a, e, t0, m0, omega, i, theta_0 = x
    # res will contain signal and noise for each image (hence the size 2*nimg)
    res = np.zeros([2, nimg])

    # compute position
    positions = orb.project_position(orb.position(ts, a, e, t0, m0), omega, i, theta_0)
    xx, yy = positions.T
    temp_d = np.sqrt(xx**2 + yy**2) * scale  # get distance to center
    # convert to pixel in the image
    positions = positions * scale + size // 2

    for k in range(nimg):
        if temp_d[k] > r_mask:
            # compute signal by integrating flux on a PSF, and correct it for
            # background (using pre-computed background profile)
            bkg = np.interp(temp_d[k], data["x"], data["bkg"][k])
            res[0, k] = photometry(data["images"][k], positions[k], 2 * fwhm) - bkg
            # get noise at position using pre-computed radial noise profil
            res[1, k] = np.interp(temp_d[k], data["x"], data["noise"][k])

    # if the value of signal is nan (outside of the image, consider it to be 0
    res[:, np.isnan(res[0])] = 0.0
    return res


def compute_snr(x, *args):
    signal, noise = get_res(x, *args)
    noise = np.sqrt(np.sum(noise**2))
    if noise == 0:
        # if the value of total noise is 0 (i.e. all values of noise are 0,
        # i.e. the orbi is completely out of the image) then snr=0
        snr = 0.0
    else:
        # compute theoretical snr in combined image
        snr = np.sum(signal) / noise
    return -snr


def plot_coadd(idx, coadded, x, params, outdir):
    a, e, t0, m0, omega, i, theta_0 = x
    # plot the corresponding image and save it as a png (for quick view)
    plt.figure()
    plt.imshow(coadded.T, origin="lower", interpolation="none", cmap="gray")
    plt.colorbar()
    xa, ya = orb.project_position(
        orb.position(t0, a, e, t0, m0),
        omega,
        i,
        theta_0,
    )
    xpix = params.n // 2 + params.scale * xa
    ypix = params.n // 2 + params.scale * ya
    # comment this line if you don't want to see where the planet is recombined:
    # decalage 2 fwhm Antoine Schneeberger
    plt.scatter(xpix - 2 * params.fwhm, ypix, color="b", marker=">")
    # '.png' old format  #New save format: tiff who have deeeper dynamics
    # to manipulate with imageJ Antoine Schneeberger
    plt.savefig(f"{outdir}/fin_tiff/fin_{idx}.tiff")
    plt.close()

    # fits.writeto(f"{outdir}/pla/pla_extracted_{idx}.fits", coadded, overwrite=True)


def make_plots(x_best, k, params, images, ts, values_dir, args):
    print(f"Make plots for solution {k+1}")
    # create combined images (for the q eme best SNR)
    coadded = recombine_images(images, ts, params.scale, *x_best)

    plot_coadd(k, coadded, x_best, params, values_dir)

    # also save it as a fits file
    # FIXME: also saved in plot_coadd ? (but without transpose...)
    fits.writeto(f"{values_dir}/fin_fits/fin_{k}.fits", coadded.T, overwrite=True)

    # save full signal and noise values
    res = get_res(x_best, *args)
    np.savetxt(f"{values_dir}/summed_snr_{k}.txt", res)

    # plot the orbits
    ax = [params.xmin, params.xmax, params.ymin, params.ymax]
    # orbit.plot.plot_orbites(x_best, x0, sim_name + "/orbites{k}")
    # orbit.plot.plot_orbites2(ts, x_best, ax, f"{values_dir}/orbites{k}")
    plot_orbites2(ts, x_best, ax, f"{values_dir}/orbites/orbites{k}")

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


def optimize_orbit(result, k, args, bounds):
    # get orbit and snr value before reoptimization for the k-th best value
    *x, signal, noise, snr_i = result

    # Gradient re-optimization:
    opt_result = scipy.optimize.minimize(
        compute_snr,
        x,
        args=args,
        method="L-BFGS-B",
        bounds=bounds,
        options={"gtol": 1e-5},
    )
    x_best = opt_result.x
    snr_best = opt_result.fun
    with np.printoptions(precision=3, suppress=True):
        print(f"init  {k}: {np.array(x)} => {snr_i:.2f}")
        print(f"reopt {k}: {x_best} => {snr_best:.2f}", flush=True)

    return snr_i, snr_best, *x_best


def reoptimize_gradient(params, n_jobs=1, n_orbits=None):
    # We sort the results in several directories
    values_dir = params.get_path("values_dir")
    os.makedirs(f"{values_dir}/fin_fits", exist_ok=True)
    os.makedirs(f"{values_dir}/fin_tiff", exist_ok=True)
    os.makedirs(f"{values_dir}/orbites", exist_ok=True)
    os.makedirs(f"{values_dir}/single", exist_ok=True)
    # os.makedirs(f"{values_dir}/pla", exist_ok=True)

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
    args = (ts, size, params.scale, params.fwhm, data, params.r_mask)
    reopt = Parallel(n_jobs=n_jobs)(
        delayed(optimize_orbit)(results[k], k, args, bounds) for k in range(n_orbits)
    )
    # Sort values with the recomputed SNR
    reopt = np.array(reopt)
    reopt = reopt[np.argsort(reopt[:, 1])]
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
