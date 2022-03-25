"""
Script used to put the sub-grid together and re-optimize the q best values of
SNR with a gradient descent method (L-BFGS-B).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from astropy.io import fits

from .imagerie import photometry, recombine_images
from .orbit import orbit as orb
from .orbit import plot as orbplot

__author__ = "Mathias Nowak, Dimitri Estevez"
__email__ = "mathias.nowak@ens-cachan.fr, destevez@lam.fr"
__status__ = "Development"


# define snr function as a function of the orbit (used for the gradient; we maximise this function)
def get_res(
    x,
    ts,
    m0,
    size,
    scale,
    images,
    fwhm,
    x_profile,
    bkg_profiles,
    noise_profiles,
    r_mask,
):
    nimg = len(images)
    a, e, t0, omega, i, theta_0 = x
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
            bkg = np.interp(temp_d[k], x_profile, bkg_profiles[k])
            res[0, k] = photometry(images[k], positions[k], 2 * fwhm) - bkg
            # get noise at position using pre-computed radial noise profil
            res[1, k] = np.interp(temp_d[k], x_profile, noise_profiles[k])

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


def sort_results(path, filename, col=1):
    """
    Sort the result file following a reference colone. by default the reference
    colone is the snr_gradient (col=1)
    :param path: folder of the result file
    :param filename: name of the result file
    :param col: default 1 reference colone fort the sort.
    :return: save the sort result file in the same folder as the result's folder
    """
    results = np.loadtxt(path + "/" + filename)
    arg_sort = results[:, col].argsort()
    Size = np.shape(results)
    results_sort = np.zeros((Size[0], Size[1] + 1))
    print(arg_sort)
    l = 0
    for arg in arg_sort:
        results_sort[l, :] = np.append(arg, results[arg, :])
        l = l + 1
    np.savetxt(
        f"{path}/results_sort.txt",
        results_sort,
        fmt="%5.0f %5.16f %5.16f %5.16f %5.16f %5.16f %5.16f %5.16f %5.16f",
        header=(
            "image_number , snr_brut_force , snr_gradient ,         a            e     "
            "          t0                  omega               i               theta_0 "
        ),
    )


def reoptimize_gradient(params):

    images_dir = params.get_path("images_dir")
    profile_dir = params.get_path("profile_dir")
    values_dir = params.get_path("values_dir")
    os.makedirs(values_dir, exist_ok=True)

    # We sort the results in several directories
    dir_path = values_dir
    os.makedirs(f"{dir_path}/fin_fits", exist_ok=True)
    os.makedirs(f"{dir_path}/fin_tiff", exist_ok=True)
    os.makedirs(f"{dir_path}/orbites", exist_ok=True)
    os.makedirs(f"{dir_path}/single", exist_ok=True)
    os.makedirs(f"{dir_path}/pla", exist_ok=True)

    m0 = params.m0
    q = params.q
    size = params.n  # number of pixels
    nimg = params.p + params.p_prev  # number of timesteps

    # total time of the observation (years)
    total_time = float(params["total_time"])
    if total_time == 0:
        ts = [float(x) for x in params["time"].split("+")]
    else:
        ts = np.linspace(0, total_time, nimg)

    ax = [params.xmin, params.xmax, params.ymin, params.ymax]
    x_profile = np.linspace(0, size // 2 - 1, size // 2)
    images, images_nonan, bkg_profiles, noise_profiles = [], [], [], []

    for k in range(nimg):
        im = fits.getdata(f"{images_dir}/image_{k}_preprocessed.fits")
        images.append(im)
        im[np.isnan(im)] = 0
        images_nonan.append(im)
        bkg_profiles.append(np.load(f"{profile_dir}/background_prof{k}.npy"))
        noise_profiles.append(np.load(f"{profile_dir}/noise_prof{k}.npy"))

    # gradient optimization
    # Load all the SNR+orbital param  files, gather and re-sort
    # ncores = params.ncores
    # results = np.zeros([ncores * q, 7])
    # for k in range(ncores):
    #     results[k * q : (k + 1) * q, :] = np.load(
    #         f"{values_dir}/res_grid{k}.npy", allow_pickle=True
    #     )

    results = np.load(f"{values_dir}/res_grid.npy")
    sorted_arg = np.argsort(results[:, 0])

    # define bounds
    bounds = params.grid.bounds()

    output_file = open(f"{values_dir}/results.txt", "w")
    # Computation on the q best SNR on one node:
    for k in range(q):
        print(f"Reoptimizing minimum {k+1} of {q}")

        args = (
            ts,
            m0,
            size,
            params.scale,
            images,
            params.fwhm,
            x_profile,
            bkg_profiles,
            noise_profiles,
            params.r_mask,
        )

        # get orbit and snr value before reoptimization for the k-th best value
        snr_i, *x = results[sorted_arg[k]]
        print(f"init: {x} => {snr_i:.2f}")

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
        a_best, e_best, t0_best, omega_best, i_best, theta_0_best = x_best
        print(f"reopt: {x_best} => {snr_best:.2f}")

        # create combined images (for the q eme best SNR)
        coadded = recombine_images(
            images_nonan,
            ts,
            params.scale,
            a_best,
            e_best,
            t0_best,
            m0,
            omega_best,
            i_best,
            theta_0_best,
        )

        output_file.write(
            str(snr_i)
            + " "
            + str(snr_best)
            + " "
            + str(a_best)
            + " "
            + str(e_best)
            + " "
            + str(t0_best)
            + " "
            + str(omega_best)
            + " "
            + str(i_best)
            + " "
            + str(theta_0_best)
        )  # save
        output_file.write("\n")

        # plot the corresponding image and save it as a png (for quick view)
        plt.figure(k + 1)
        plt.imshow(coadded.T, origin="lower", interpolation="none", cmap="gray")
        plt.colorbar()
        xa, ya = orb.project_position(
            orb.position(t0_best, a_best, e_best, t0_best, m0),
            omega_best,
            i_best,
            theta_0_best,
        )
        xpix = size // 2 + params.scale * xa
        ypix = size // 2 + params.scale * ya
        # comment this line if you don't want to see where the planet is recombined:
        # decalage 2 fwhm Antoine Schneeberger
        plt.scatter(xpix - 2 * params.fwhm, ypix, color="b", marker=">")
        # '.png' old format  #New save format: tiff who have deeeper dynamics
        # to manipulate with imageJ Antoine Schneeberger
        plt.savefig(f"{values_dir}/fin_fits/fin_{k}.tiff")
        plt.close()

        # extract small part for future ML algorithm
        # xmin = np.min([xpix - 5, 0])
        # xmax = np.max([xpix + 5, size])
        # ymin = np.min([ypix - 5, 0])
        # ymax = np.max([ypix + 5, size])
        fits.writeto(
            f"{values_dir}/pla/pla_extracted_{k}.fits", coadded, overwrite=True
        )
        # [xmin:xmax, ymin:ymax]) The crop of the image as put as here don't
        # work, so it's commented until a solution is found Antoine Schneeberger

        # also save it as a fits file
        hdu = fits.PrimaryHDU(coadded.T)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(f"{values_dir}/fin_fits/fin_{k}.fits", overwrite=True)

        # save full signal and noise values
        res = get_res(x, *args)
        np.savetxt(f"{values_dir}/summed_snr_{k}.txt", res)

        # plot the orbits
        # orbit.plot.plot_orbites(x_best, x0, m0, sim_name + "/orbites{k}")
        # orbit.plot.plot_orbites2(ts, x_best, m0, ax, f"{values_dir}/orbites{k}")
        orbplot.plot_orbites2(ts, x_best, m0, ax, f"{values_dir}/orbites/orbites{k}")

        # If single_plot=='yes' a cross is ploted on each image where the
        # planet is found (by default no);
        if params.single_plot == "yes":
            for l in range(len(ts)):
                orbplot.plot_ontop(
                    x_best,
                    m0,
                    params.dist,
                    [ts[l]],
                    params.resol,
                    images[l],
                    f"{values_dir}/single/single_{k}fin_{l}",
                )

    output_file.close()

    # for k in range(ncores):
    #     if os.path.exists(f"{values_dir}/fun_values{k}.npy"):
    #         os.remove(f"{values_dir}/fun_values{k}.npy")
    #     if os.path.exists(f"{values_dir}/grid{k}.npy"):
    #         os.remove(f"{values_dir}/grid{k}.npy")

    # Sorting of the final results by SNR (after gradient)
    sort_results(dir_path, "results.txt")

    print("Done!")
