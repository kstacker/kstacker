"""
Functions used to represent the orbit of the planet
"""

import math
import os
import h5py

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from photutils import CircularAperture, aperture_photometry

import seaborn as sns  # used for the scatterplot colormap
from astropy.visualization import ZScaleInterval
from matplotlib.colors import ListedColormap  # used for the scatterplot

from . import orbit


def plot_orbites(ts, x, ax, filename):
    """
    Plot the true orbit (in red) and the best orbit (in blue)
    found in the projected plane of sky (au-au).

    Parameters
    ----------
    x : np.ndarray
        2D array of orbits, N x (a, e, t0, m, omega, i, theta0) in
        (au, nounit, year, rad, rad, rad)
    filename : str
        name of the file where to save the plot (extension will be .png)
    ax : list of float
        scale of axes, xmin, xmax, ymin, ymax in astronomical units

    """
    a, e, t0, m0, omega, i, theta_0 = x
    p = a * (1 - e**2)
    thetas = np.linspace(0, 2 * math.pi, 1000)
    r = p / (1 + e * np.cos(thetas))
    positions = r * np.array([np.cos(thetas), np.sin(thetas)])
    x_proj, y_proj = orbit.project_position(positions.T, omega, i, theta_0).T

    plt.figure(0, figsize=(6, 6))
    plt.plot(y_proj, x_proj, color="blue")
    plt.plot([0], [0], "+", color="red")
    plt.axis(ax)

    xp, yp = orbit.project_position_full(ts, a, e, t0, m0, omega, i, theta_0).T
    plt.scatter(yp, xp, marker="+")
    plt.xlabel("Astronomical Units")
    plt.ylabel("Astronomical Units")
    plt.savefig(filename + ".png")
    plt.close()


def plot_ontop(x, d, ts, res, back_image, filename):
    """
    Function used to plot an orbit on top of a background corono image.
    @param float[6] x: parameters of the best orbit found (a, e, t0, omega, i, theta0) in (au, nounit, year, rad, rad, rad)
    @param float d: distance of the star (in pc)
    @param float[q] ts: time steps (in years) at which the planet shall be plotted
    @param float res: res of the image (in mas/pixel)
    @param float[n, n]: background image
    @param string filename: name of the file where the image shall be saved (extension .png will be added)
    """
    npix = np.size(back_image[0])

    scale = 1.0 / (d * (res / 1000.0))
    a, e, t0, m0, omega, i, theta_0 = x
    p = a * (1 - e**2)
    thetas = np.linspace(-2 * math.pi, 0, 1000)
    r = p / (1 + e * np.cos(thetas))

    positions = r * np.array([np.cos(thetas), np.sin(thetas)])
    x_proj, y_proj = orbit.project_position(positions.T, omega, i, theta_0).T

    plt.figure(1)
    plt.axis("off")
    vmin, vmax = ZScaleInterval().get_limits(back_image)
    plt.imshow(
        back_image,
        origin="lower",
        interpolation="none",
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )
    plt.scatter(
        npix // 2 + scale * y_proj, npix // 2 + scale * x_proj, color="b", s=0.1
    )

    xp, yp = orbit.project_position_full(ts, a, e, t0, m0, omega, i, theta_0).T
    xpix = npix // 2 + scale * xp
    ypix = npix // 2 + scale * yp
    plt.plot(ypix, xpix, "+", color="r")

    length = 1000.0 / res
    plt.plot([npix - length, npix], [-10, -10], "y")
    plt.text(npix - 2 * length / 3, -20, "1 arcsec", color="y")

    plt.savefig(filename + ".png")
    plt.close()


def plot_orbits(x, snr, img, scale, ax=None, norbits=None):
    if ax is None:
        _, ax = plt.subplots()

    vmin, vmax = ZScaleInterval().get_limits(img)
    ax.imshow(
        img,
        origin="lower",
        interpolation="none",
        cmap="gray",
        alpha=0.5,
        vmin=vmin,
        vmax=vmax,
    )

    norbits = min(norbits or x.shape[0], x.shape[0])
    cmap = plt.get_cmap("Blues")
    norm = mpl.colors.Normalize(vmin=snr.min() - 0.1, vmax=snr.max())

    npix = img.shape[0]
    thetas = np.linspace(-2 * np.pi, 0, 100)

    for j in reversed(range(norbits)):
        a, e, t0, m0, omega, i, theta_0 = x[j]
        p = a * (1 - e**2)
        r = p / (1 + e * np.cos(thetas))
        positions = r * np.array([np.cos(thetas), np.sin(thetas)])
        x_proj, y_proj = (
            npix // 2 + scale * orbit.project_position(positions.T, omega, i, theta_0).T
        )
        ax.plot(y_proj, x_proj, lw=1, color=cmap(norm(snr[j])), alpha=0.2)


def plot_snr_hist(snr_gradient, snr_brut_force, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    min_ = int(min(snr_gradient.min(), snr_brut_force.min()) * 10) / 10
    max_ = int(max(snr_gradient.max(), snr_brut_force.max()) * 10 + 1) / 10
    bins = np.linspace(min_, max_, int((max_ - min_) / 0.01) + 1)

    ax.hist(snr_gradient, bins=bins, histtype="step", label="snr_gradient")
    ax.hist(snr_brut_force, bins=bins, histtype="step", label="snr_brut_force")
    ax.legend()
    ax.set(title="SNR Histogram")


def plot_snr_curve(snr_gradient, snr_brut_force, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(snr_gradient, label="snr_gradient", drawstyle="steps-mid")
    snr_brut = np.sort(snr_brut_force)[::-1]
    ax.plot(snr_brut, label="snr_brut_force", drawstyle="steps-mid")
    ax.legend()
    ax.set(title="SNR Curves")

def corner_plots_mcmc(
    params, nbins, norbits=None, omegatheta=None, savefig=None, figsize=(25, 25)
):
    from ..utils import Params, read_results

    fig, axes = mpl.pyplot.subplots(ncols=7, nrows=7, figsize=figsize)

    # fmt: off
    flatui = ["#001EF5", "#002DF5", "#003CF5", "#004CF5", "#005BF5", "#016BF5", "#017AF5", "#0189F5", "#0199F5",
              "#02A8F5", "#02B7F5", "#02C6F5", "#02D6F5", "#03E5F6", "#03F4F6", "#03F6E8", "#03F6D9", "#03F6CA",
              "#04F6BB", "#04F6AC", "#04F69D", "#04F68E", "#05F680", "#05F671", "#05F662", "#05F653", "#06F744",
              "#06F735", "#06F727", "#06F718", "#07F709", "#13F707", "#22F707", "#31F707", "#41F708", "#50F708",
              "#5FF708", "#6EF708", "#7DF808", "#8CF809", "#9BF809", "#AAF809", "#B9F809", "#C8F80A", "#D7F80A",
              "#E6F80A", "#F5F80A", "#F8ED0B", "#F8DE0B", "#F8D00B", "#F8C10B", "#F9B20C", "#F9A40C", "#F9950C",
              "#F9870C", "#F9780D", "#F96A0D", "#F95B0D", "#F94D0D", "#F93E0E", "#F9300E", "#F9210E", "#F9130E",
              "#F90F18"]  # creation of a color gradient
    # fmt: on
    # creating the colormap
    color_scatter = ListedColormap(sns.color_palette(flatui).as_hex())
    
    if isinstance(params, str):
        path = os.path.dirname(params)
        params = Params.read(params)
        params.work_dir = path

    res = read_results(os.path.join(params.work_dir, "values", "results_mcmc.txt"), params)

    grid = res.as_array(names=("a", "e", "t0", "m0", "omega", "i", "theta_0"))
    grid = grid.view("f8").reshape(grid.shape[0], 7)

    if norbits is not None:
        grid = grid[:norbits]
    else:
        norbits = grid.shape[0]

    a, e, t0, m0, omega, i, theta_0 = grid.T

    a_mean = np.mean(a)
    a_std = np.std(a)
    m0_mean = np.mean(m0)
    m0_std = np.std(m0)
    e_mean = np.mean(e)
    e_std = np.std(e)

    period_mean = np.sqrt(a_mean**3 / m0_mean)
    t0 = t0 % period_mean
    t0_mean = np.mean(t0)
    t0_std = np.std(t0)

    # Transformation in the classical reference frame (ex: ORBITIZE):
    i = -i + np.pi
    omega = -omega + np.pi
    theta_0 = theta_0 + np.pi

    # omega = omega % np.pi  # Check that this transformation % pi for omega and theta_0 is ok
    # theta_0 = theta_0 % np.pi

    i_mean = np.mean(i)
    i_std = np.std(i)

    omega_plus_theta_0 = omega + theta_0
    omega_min_theta_0 = omega - theta_0

    omega_plus_theta_0 = omega_plus_theta_0 % np.pi * 2.0
    omega_min_theta_0 = omega_min_theta_0 % np.pi * 2.0

    omega_plus_theta_0_mean = np.mean(omega_plus_theta_0)
    omega_plus_theta_0_std = np.std(omega_plus_theta_0)
    omega_min_theta_0_mean = np.mean(omega_min_theta_0)
    omega_min_theta_0_std = np.std(omega_min_theta_0)

    omega_mean = (omega_plus_theta_0_mean + omega_min_theta_0_mean) / 2
    theta_0_mean = (omega_plus_theta_0_mean - omega_min_theta_0_mean) / 2

    omega_mean_err = (
        math.sqrt(omega_plus_theta_0_std**2 + omega_min_theta_0_std**2) / 2
    )
    theta_0_mean_err = (
        math.sqrt(omega_plus_theta_0_std**2 + omega_min_theta_0_std**2) / 2
    )

    with open("orbite_moyenne.txt", "w") as f:
        # écriture des paramètres dans le fichier
        f.write("Mean Orbital parameters in a reference frame similar to orbitize\n")
        f.write(f"a = {a_mean:.3f} +- {a_std:.3f}\n")
        f.write(f"e = {e_mean:.3f} +- {e_std:.3f}\n")
        f.write(f"t_0 = {t0_mean:.3f} +- {t0_std:.3f}\n")
        f.write(f"i = {math.degrees(i_mean):.3f} +- {math.degrees(i_std):.3f}\n")
        f.write(f"m_star = {m0_mean:.3f} +- {m0_std:.3f}\n")
        f.write(
            f"omega = {math.degrees(omega_mean):.3f} +-"
            f" {math.degrees(omega_mean_err):.3f}\n"
        )
        f.write(
            f"theta_0 = {math.degrees(theta_0_mean):.3f} +-"
            f" {math.degrees(theta_0_mean_err):.3f}\n"
        )

    f.close()

    log_prob = res["log_prob"]
    log_prob = log_prob[:norbits] # verifier cette ligne

    # 1st row : a as a function of others parameters
    axes[0, 0].hist(a, bins=nbins, color="darkcyan")
    # 2nd row : e as a function of others parameters
    im = axes[1, 0].scatter(a, e, c=log_prob, cmap=color_scatter)
    axes[1, 0].errorbar(
        a_mean, e_mean, yerr=e_std, xerr=a_std, ecolor="black", elinewidth=2.5
    )
    axes[1, 1].hist(e, bins=nbins, color="darkcyan")
    # 3rd row : t0 as a function of others parameters
    axes[2, 0].scatter(a, t0, c=log_prob, cmap=color_scatter)
    axes[2, 0].errorbar(
        a_mean, t0_mean, yerr=t0_std, xerr=a_std, ecolor="black", elinewidth=2.5
    )
    axes[2, 1].scatter(e, t0, c=log_prob, cmap=color_scatter)
    axes[2, 1].errorbar(
        e_mean, t0_mean, yerr=t0_std, xerr=e_std, ecolor="black", elinewidth=2.5
    )
    axes[2, 2].hist(t0, bins=nbins, color="darkcyan")
    # 4rd row : m0 as a function of others parameters
    axes[3, 0].scatter(a, m0, c=log_prob, cmap=color_scatter)
    axes[3, 0].errorbar(
        a_mean, m0_mean, yerr=m0_std, xerr=a_std, ecolor="black", elinewidth=2.5
    )
    axes[3, 1].scatter(e, m0, c=log_prob, cmap=color_scatter)
    axes[3, 1].errorbar(
        e_mean, m0_mean, yerr=m0_std, xerr=e_std, ecolor="black", elinewidth=2.5
    )
    axes[3, 2].scatter(t0, m0, c=log_prob, cmap=color_scatter)
    axes[3, 2].errorbar(
        t0_mean, m0_mean, yerr=m0_std, xerr=t0_std, ecolor="black", elinewidth=2.5
    )
    axes[3, 3].hist(m0, bins=nbins, color="darkcyan")
    # 5th row : omega_plus_theta_0 as a function of others parameters
    axes[4, 0].scatter(a, omega_plus_theta_0, c=log_prob, cmap=color_scatter)
    axes[4, 0].errorbar(
        a_mean,
        omega_plus_theta_0_mean,
        yerr=omega_plus_theta_0_std,
        xerr=a_std,
        ecolor="black",
        elinewidth=2.5,
    )
    axes[4, 1].scatter(e, omega_plus_theta_0, c=log_prob, cmap=color_scatter)
    axes[4, 1].errorbar(
        e_mean,
        omega_plus_theta_0_mean,
        yerr=omega_plus_theta_0_std,
        xerr=e_std,
        ecolor="black",
        elinewidth=2.5,
    )
    axes[4, 2].scatter(t0, omega_plus_theta_0, c=log_prob, cmap=color_scatter)
    axes[4, 2].errorbar(
        t0_mean,
        omega_plus_theta_0_mean,
        yerr=omega_plus_theta_0_std,
        xerr=t0_std,
        ecolor="black",
        elinewidth=2.5,
    )
    axes[4, 3].scatter(m0, omega_plus_theta_0, c=log_prob, cmap=color_scatter)
    axes[4, 3].errorbar(
        m0_mean,
        omega_plus_theta_0_mean,
        yerr=omega_plus_theta_0_std,
        xerr=m0_std,
        ecolor="black",
        elinewidth=2.5,
    )
    axes[4, 4].hist(omega_plus_theta_0, bins=nbins, color="darkcyan")
    # 6th row : i as a function of others parameters
    axes[5, 0].scatter(a, i, c=log_prob, cmap=color_scatter)
    axes[5, 0].errorbar(
        a_mean, i_mean, yerr=i_std, xerr=a_std, ecolor="black", elinewidth=2.5
    )
    axes[5, 1].scatter(e, i, c=log_prob, cmap=color_scatter)
    axes[5, 1].errorbar(
        e_mean, i_mean, yerr=i_std, xerr=e_std, ecolor="black", elinewidth=2.5
    )
    axes[5, 2].scatter(t0, i, c=log_prob, cmap=color_scatter)
    axes[5, 2].errorbar(
        t0_mean, i_mean, yerr=i_std, xerr=t0_std, ecolor="black", elinewidth=2.5
    )
    axes[5, 3].scatter(m0, i, c=log_prob, cmap=color_scatter)
    axes[5, 3].errorbar(
        m0_mean, i_mean, yerr=i_std, xerr=m0_std, ecolor="black", elinewidth=2.5
    )
    axes[5, 4].scatter(omega_plus_theta_0, i, c=log_prob, cmap=color_scatter)
    axes[5, 4].errorbar(
        omega_plus_theta_0_mean,
        i_mean,
        yerr=i_std,
        xerr=omega_plus_theta_0_std,
        ecolor="black",
        elinewidth=2.5,
    )
    axes[5, 5].hist(i, bins=nbins, color="darkcyan")
    # 7th row : theta0 as a function of others parameters
    axes[6, 0].scatter(a, omega_min_theta_0, c=log_prob, cmap=color_scatter)
    axes[6, 0].errorbar(
        a_mean,
        omega_min_theta_0_mean,
        yerr=omega_min_theta_0_std,
        xerr=a_std,
        ecolor="black",
        elinewidth=2.5,
    )
    axes[6, 1].scatter(e, omega_min_theta_0, c=log_prob, cmap=color_scatter)
    axes[6, 1].errorbar(
        e_mean,
        omega_min_theta_0_mean,
        yerr=omega_min_theta_0_std,
        xerr=e_std,
        ecolor="black",
        elinewidth=2.5,
    )
    axes[6, 2].scatter(t0, omega_min_theta_0, c=log_prob, cmap=color_scatter)
    axes[6, 2].errorbar(
        t0_mean,
        omega_min_theta_0_mean,
        yerr=omega_min_theta_0_std,
        xerr=t0_std,
        ecolor="black",
        elinewidth=2.5,
    )
    axes[6, 3].scatter(m0, omega_min_theta_0, c=log_prob, cmap=color_scatter)
    axes[6, 3].errorbar(
        m0_mean,
        omega_min_theta_0_mean,
        yerr=omega_min_theta_0_std,
        xerr=m0_std,
        ecolor="black",
        elinewidth=2.5,
    )
    axes[6, 4].scatter(
        omega_plus_theta_0, omega_min_theta_0, c=log_prob, cmap=color_scatter
    )
    axes[6, 4].errorbar(
        omega_plus_theta_0_mean,
        omega_min_theta_0_mean,
        yerr=omega_min_theta_0_std,
        xerr=omega_plus_theta_0_std,
        ecolor="black",
        elinewidth=2.5,
    )
    axes[6, 5].scatter(i, omega_min_theta_0, c=log_prob, cmap=color_scatter)
    axes[6, 5].errorbar(
        i_mean,
        omega_min_theta_0_mean,
        yerr=omega_min_theta_0_std,
        xerr=i_std,
        ecolor="black",
        elinewidth=2.5,
    )
    axes[6, 6].hist(omega_min_theta_0, bins=nbins, color="darkcyan")

    # Figure Title
    fig.suptitle(
        f"Corner-plot of the {norbits} K-Stacker orbits at higher SNR", fontsize=16
    )

    # Axes Labels
    axes[6, 0].set_xlabel("a (a.u.)")
    axes[6, 1].set_xlabel("e")
    axes[6, 2].set_xlabel("$t_0$ (yrs)")
    axes[6, 3].set_xlabel("$m0$ (solar_mass)")
    axes[6, 4].set_xlabel(r"$\Omega$ + $\omega$ (rad)")
    axes[6, 5].set_xlabel("i (rad)")
    axes[6, 6].set_xlabel(r"$\Omega$ - $\omega$ (rad)")
    axes[1, 0].set_ylabel("e")
    axes[2, 0].set_ylabel("$t_0$ (yrs)")
    axes[3, 0].set_ylabel("$m0$ (solar_mass)")
    axes[4, 0].set_ylabel(r"$\Omega$ + $\omega$ (rad)")
    axes[5, 0].set_ylabel("i (rad)")
    axes[6, 0].set_ylabel(r"$\Omega$ - $\omega$ (rad)")
    # Remove labels at the middle of the subplots
    for k in range(1, 6, 1):
        mpl.pyplot.setp([plot.get_xticklabels() for plot in axes[k, :]], visible=False)
        mpl.pyplot.setp([plot.get_yticklabels() for plot in axes[:, k]], visible=False)
    # Graduation on the right for the histograms
    for k in range(7):
        axes[k, k].yaxis.tick_right()
    # reducing spaces between subplots
    fig.subplots_adjust(wspace=0.15, hspace=0.2)
    # Remove sub-plots that we don't want to see
    for i in range(6):
        for k in range(i + 1, 7):
            axes[i, k].remove()

    # SNR Color bar
    cax = mpl.pyplot.axes([0.07, 0.11, 0.01, 0.77])
    cbar = mpl.pyplot.colorbar(im, cax, ticklocation="left")
    cbar.ax.set_title("Log Likelihood")

    if savefig:
        fig.savefig(savefig)


def corner_plots(
    params, nbins, norbits=None, omegatheta=None, savefig=None, figsize=(25, 25)
):
    from ..utils import Params, read_results

    fig, axes = plt.subplots(ncols=7, nrows=7, figsize=figsize)

    # fmt: off
    flatui = ["#001EF5", "#002DF5", "#003CF5", "#004CF5", "#005BF5", "#016BF5", "#017AF5", "#0189F5", "#0199F5",
              "#02A8F5", "#02B7F5", "#02C6F5", "#02D6F5", "#03E5F6", "#03F4F6", "#03F6E8", "#03F6D9", "#03F6CA",
              "#04F6BB", "#04F6AC", "#04F69D", "#04F68E", "#05F680", "#05F671", "#05F662", "#05F653", "#06F744",
              "#06F735", "#06F727", "#06F718", "#07F709", "#13F707", "#22F707", "#31F707", "#41F708", "#50F708",
              "#5FF708", "#6EF708", "#7DF808", "#8CF809", "#9BF809", "#AAF809", "#B9F809", "#C8F80A", "#D7F80A",
              "#E6F80A", "#F5F80A", "#F8ED0B", "#F8DE0B", "#F8D00B", "#F8C10B", "#F9B20C", "#F9A40C", "#F9950C",
              "#F9870C", "#F9780D", "#F96A0D", "#F95B0D", "#F94D0D", "#F93E0E", "#F9300E", "#F9210E", "#F9130E",
              "#F90F18"]  # creation of a color gradient
    # fmt: on
    # creating the colormap
    color_scatter = ListedColormap(sns.color_palette(flatui).as_hex())

    if isinstance(params, str):
        path = os.path.dirname(params)
        params = Params.read(params)
        params.work_dir = path

    res = read_results(os.path.join(params.work_dir, "values", "results.txt"), params)
    if res["snr_gradient"][0] < 0:
        res["snr_gradient"] *= -1
        res["snr_brut_force"] *= -1

    grid = res.as_array(names=("a", "e", "t0", "m0", "omega", "i", "theta_0"))
    grid = grid.view("f8").reshape(grid.shape[0], 7)

    if norbits is not None:
        grid = grid[:norbits]
    else:
        norbits = grid.shape[0]

    a, e, t0, m0, omega, i, theta_0 = grid.T

    a_mean = np.mean(a)
    a_std = np.std(a)
    m0_mean = np.mean(m0)
    m0_std = np.std(m0)
    e_mean = np.mean(e)
    e_std = np.std(e)

    period_mean = np.sqrt(a_mean**3 / m0_mean)
    t0 = t0 % period_mean
    t0_mean = np.mean(t0)
    t0_std = np.std(t0)

    # Transformation in the classical reference frame (ex: ORBITIZE):
    i = -i + np.pi
    omega = -omega + np.pi
    theta_0 = theta_0 + np.pi

    # omega = omega % np.pi  # Check that this transformation % pi for omega and theta_0 is ok
    # theta_0 = theta_0 % np.pi

    i_mean = np.mean(i)
    i_std = np.std(i)

    omega_plus_theta_0 = omega + theta_0
    omega_min_theta_0 = omega - theta_0

    omega_plus_theta_0 = omega_plus_theta_0 % np.pi * 2.0
    omega_min_theta_0 = omega_min_theta_0 % np.pi * 2.0

    omega_plus_theta_0_mean = np.mean(omega_plus_theta_0)
    omega_plus_theta_0_std = np.std(omega_plus_theta_0)
    omega_min_theta_0_mean = np.mean(omega_min_theta_0)
    omega_min_theta_0_std = np.std(omega_min_theta_0)

    omega_mean = (omega_plus_theta_0_mean + omega_min_theta_0_mean) / 2
    theta_0_mean = (omega_plus_theta_0_mean - omega_min_theta_0_mean) / 2

    omega_mean_err = (
        math.sqrt(omega_plus_theta_0_std**2 + omega_min_theta_0_std**2) / 2
    )
    theta_0_mean_err = (
        math.sqrt(omega_plus_theta_0_std**2 + omega_min_theta_0_std**2) / 2
    )

    # omega_mean = np.mean(omega)
    # omega_std = np.std(omega)
    # theta_0_mean = np.mean(theta_0)
    # theta_0_std = np.std(theta_0)
    # if omegatheta:
    #     omega_theta = omega - theta_0
    #     omega_p_theta = omega + theta_0

    with open("orbite_moyenne.txt", "w") as f:
        # écriture des paramètres dans le fichier
        f.write("Mean Orbital parameters in a reference frame similar to orbitize\n")
        f.write(f"a = {a_mean:.3f} +- {a_std:.3f}\n")
        f.write(f"e = {e_mean:.3f} +- {e_std:.3f}\n")
        f.write(f"t_0 = {t0_mean:.3f} +- {t0_std:.3f}\n")
        f.write(f"i = {math.degrees(i_mean):.3f} +- {math.degrees(i_std):.3f}\n")
        f.write(f"m_star = {m0_mean:.3f} +- {m0_std:.3f}\n")
        f.write(
            f"omega = {math.degrees(omega_mean):.3f} +-"
            f" {math.degrees(omega_mean_err):.3f}\n"
        )
        f.write(
            f"theta_0 = {math.degrees(theta_0_mean):.3f} +-"
            f" {math.degrees(theta_0_mean_err):.3f}\n"
        )

    f.close()

    snr_grad = res["snr_gradient"]
    snr_grad = snr_grad[:norbits] # verifier cette ligne

    # 1st row : a as a function of others parameters
    axes[0, 0].hist(a, bins=nbins, color="darkcyan")
    # 2nd row : e as a function of others parameters
    im = axes[1, 0].scatter(a, e, c=snr_grad, cmap=color_scatter)
    axes[1, 0].errorbar(
        a_mean, e_mean, yerr=e_std, xerr=a_std, ecolor="black", elinewidth=2.5
    )
    axes[1, 1].hist(e, bins=nbins, color="darkcyan")
    # 3rd row : t0 as a function of others parameters
    axes[2, 0].scatter(a, t0, c=snr_grad, cmap=color_scatter)
    axes[2, 0].errorbar(
        a_mean, t0_mean, yerr=t0_std, xerr=a_std, ecolor="black", elinewidth=2.5
    )
    axes[2, 1].scatter(e, t0, c=snr_grad, cmap=color_scatter)
    axes[2, 1].errorbar(
        e_mean, t0_mean, yerr=t0_std, xerr=e_std, ecolor="black", elinewidth=2.5
    )
    axes[2, 2].hist(t0, bins=nbins, color="darkcyan")
    # 4rd row : m0 as a function of others parameters
    axes[3, 0].scatter(a, m0, c=snr_grad, cmap=color_scatter)
    axes[3, 0].errorbar(
        a_mean, m0_mean, yerr=m0_std, xerr=a_std, ecolor="black", elinewidth=2.5
    )
    axes[3, 1].scatter(e, m0, c=snr_grad, cmap=color_scatter)
    axes[3, 1].errorbar(
        e_mean, m0_mean, yerr=m0_std, xerr=e_std, ecolor="black", elinewidth=2.5
    )
    axes[3, 2].scatter(t0, m0, c=snr_grad, cmap=color_scatter)
    axes[3, 2].errorbar(
        t0_mean, m0_mean, yerr=m0_std, xerr=t0_std, ecolor="black", elinewidth=2.5
    )
    axes[3, 3].hist(m0, bins=nbins, color="darkcyan")
    # 5th row : omega_plus_theta_0 as a function of others parameters
    axes[4, 0].scatter(a, omega_plus_theta_0, c=snr_grad, cmap=color_scatter)
    axes[4, 0].errorbar(
        a_mean,
        omega_plus_theta_0_mean,
        yerr=omega_plus_theta_0_std,
        xerr=a_std,
        ecolor="black",
        elinewidth=2.5,
    )
    axes[4, 1].scatter(e, omega_plus_theta_0, c=snr_grad, cmap=color_scatter)
    axes[4, 1].errorbar(
        e_mean,
        omega_plus_theta_0_mean,
        yerr=omega_plus_theta_0_std,
        xerr=e_std,
        ecolor="black",
        elinewidth=2.5,
    )
    axes[4, 2].scatter(t0, omega_plus_theta_0, c=snr_grad, cmap=color_scatter)
    axes[4, 2].errorbar(
        t0_mean,
        omega_plus_theta_0_mean,
        yerr=omega_plus_theta_0_std,
        xerr=t0_std,
        ecolor="black",
        elinewidth=2.5,
    )
    axes[4, 3].scatter(m0, omega_plus_theta_0, c=snr_grad, cmap=color_scatter)
    axes[4, 3].errorbar(
        m0_mean,
        omega_plus_theta_0_mean,
        yerr=omega_plus_theta_0_std,
        xerr=m0_std,
        ecolor="black",
        elinewidth=2.5,
    )
    axes[4, 4].hist(omega_plus_theta_0, bins=nbins, color="darkcyan")
    # 6th row : i as a function of others parameters
    axes[5, 0].scatter(a, i, c=snr_grad, cmap=color_scatter)
    axes[5, 0].errorbar(
        a_mean, i_mean, yerr=i_std, xerr=a_std, ecolor="black", elinewidth=2.5
    )
    axes[5, 1].scatter(e, i, c=snr_grad, cmap=color_scatter)
    axes[5, 1].errorbar(
        e_mean, i_mean, yerr=i_std, xerr=e_std, ecolor="black", elinewidth=2.5
    )
    axes[5, 2].scatter(t0, i, c=snr_grad, cmap=color_scatter)
    axes[5, 2].errorbar(
        t0_mean, i_mean, yerr=i_std, xerr=t0_std, ecolor="black", elinewidth=2.5
    )
    axes[5, 3].scatter(m0, i, c=snr_grad, cmap=color_scatter)
    axes[5, 3].errorbar(
        m0_mean, i_mean, yerr=i_std, xerr=m0_std, ecolor="black", elinewidth=2.5
    )
    axes[5, 4].scatter(omega_plus_theta_0, i, c=snr_grad, cmap=color_scatter)
    axes[5, 4].errorbar(
        omega_plus_theta_0_mean,
        i_mean,
        yerr=i_std,
        xerr=omega_plus_theta_0_std,
        ecolor="black",
        elinewidth=2.5,
    )
    axes[5, 5].hist(i, bins=nbins, color="darkcyan")
    # 7th row : theta0 as a function of others parameters
    axes[6, 0].scatter(a, omega_min_theta_0, c=snr_grad, cmap=color_scatter)
    axes[6, 0].errorbar(
        a_mean,
        omega_min_theta_0_mean,
        yerr=omega_min_theta_0_std,
        xerr=a_std,
        ecolor="black",
        elinewidth=2.5,
    )
    axes[6, 1].scatter(e, omega_min_theta_0, c=snr_grad, cmap=color_scatter)
    axes[6, 1].errorbar(
        e_mean,
        omega_min_theta_0_mean,
        yerr=omega_min_theta_0_std,
        xerr=e_std,
        ecolor="black",
        elinewidth=2.5,
    )
    axes[6, 2].scatter(t0, omega_min_theta_0, c=snr_grad, cmap=color_scatter)
    axes[6, 2].errorbar(
        t0_mean,
        omega_min_theta_0_mean,
        yerr=omega_min_theta_0_std,
        xerr=t0_std,
        ecolor="black",
        elinewidth=2.5,
    )
    axes[6, 3].scatter(m0, omega_min_theta_0, c=snr_grad, cmap=color_scatter)
    axes[6, 3].errorbar(
        m0_mean,
        omega_min_theta_0_mean,
        yerr=omega_min_theta_0_std,
        xerr=m0_std,
        ecolor="black",
        elinewidth=2.5,
    )
    axes[6, 4].scatter(
        omega_plus_theta_0, omega_min_theta_0, c=snr_grad, cmap=color_scatter
    )
    axes[6, 4].errorbar(
        omega_plus_theta_0_mean,
        omega_min_theta_0_mean,
        yerr=omega_min_theta_0_std,
        xerr=omega_plus_theta_0_std,
        ecolor="black",
        elinewidth=2.5,
    )
    axes[6, 5].scatter(i, omega_min_theta_0, c=snr_grad, cmap=color_scatter)
    axes[6, 5].errorbar(
        i_mean,
        omega_min_theta_0_mean,
        yerr=omega_min_theta_0_std,
        xerr=i_std,
        ecolor="black",
        elinewidth=2.5,
    )
    axes[6, 6].hist(omega_min_theta_0, bins=nbins, color="darkcyan")

    # Figure Title
    fig.suptitle(
        f"Corner-plot of the {norbits} K-Stacker orbits at higher SNR", fontsize=16
    )

    # Axes Labels
    axes[6, 0].set_xlabel("a (a.u.)")
    axes[6, 1].set_xlabel("e")
    axes[6, 2].set_xlabel("$t_0$ (yrs)")
    axes[6, 3].set_xlabel("$m0$ (solar_mass)")
    axes[6, 4].set_xlabel(r"$\Omega$ + $\omega$ (rad)")
    axes[6, 5].set_xlabel("i (rad)")
    axes[6, 6].set_xlabel(r"$\Omega$ - $\omega$ (rad)")
    axes[1, 0].set_ylabel("e")
    axes[2, 0].set_ylabel("$t_0$ (yrs)")
    axes[3, 0].set_ylabel("$m0$ (solar_mass)")
    axes[4, 0].set_ylabel(r"$\Omega$ + $\omega$ (rad)")
    axes[5, 0].set_ylabel("i (rad)")
    axes[6, 0].set_ylabel(r"$\Omega$ - $\omega$ (rad)")
    # Remove labels at the middle of the subplots
    for k in range(1, 6, 1):
        plt.setp([plot.get_xticklabels() for plot in axes[k, :]], visible=False)
        plt.setp([plot.get_yticklabels() for plot in axes[:, k]], visible=False)
    # Graduation on the right for the histograms
    for k in range(7):
        axes[k, k].yaxis.tick_right()
    # reducing spaces between subplots
    fig.subplots_adjust(wspace=0.15, hspace=0.2)
    # Remove sub-plots that we don't want to see
    for i in range(6):
        for k in range(i + 1, 7):
            axes[i, k].remove()

    # SNR Color bar
    cax = plt.axes([0.07, 0.11, 0.01, 0.77])
    cbar = plt.colorbar(im, cax, ticklocation="left")
    cbar.ax.set_title("SNR_KS")

    if savefig:
        fig.savefig(savefig)
        
        
def plot_results_mcmc(params, nimg=None, savefig=None):
    """

    Parameters
    ----------
    params : Params
        the yamel doc for this simulation
    nimg : int, optional
        number of time step ploted
    savefig : string, optional
        path location to save the plot

    Returns
    -------
    None.

    """
    from ..utils import Params, read_results

    if isinstance(params, str):
        path = os.path.dirname(params)
        params = Params.read(params)
        params.work_dir = path

    res = read_results(os.path.join(params.work_dir, "values", "results_mcmc.txt"), params)

    data = params.load_data(method="aperture")
    grid = res.as_array(names=("a", "e", "t0", "m0", "omega", "i", "theta_0"))
    grid = grid.view("f8").reshape(grid.shape[0], 7)

    nimg = nimg or len(data["images"])
    ncols = max(4, nimg)
    fig, axes = mpl.pyplot.subplots(1, ncols, figsize=(ncols * 3, 6), layout="constrained")

    for i in range(ncols):
        ax = axes[i]
        if i < nimg:
            plot_orbits(
                grid, res["log_prob"], data["images"][i], params.scale, ax=ax
            )
            ax.set(title=f"Image {i}")
        else:
            ax.axis("off")

    if savefig:
        fig.savefig(savefig)


def plot_results(params, nimg=None, savefig=None, snr_grad_limits=None):
    from ..utils import Params, read_results

    if isinstance(params, str):
        path = os.path.dirname(params)
        params = Params.read(params)
        params.work_dir = path

    res = read_results(os.path.join(params.work_dir, "values", "results.txt"), params)
    if res["snr_gradient"][0] < 0:
        res["snr_gradient"] *= -1
        res["snr_brut_force"] *= -1

    if snr_grad_limits is not None:
        sel = (res["snr_gradient"] > snr_grad_limits[0]) & (
            res["snr_gradient"] < snr_grad_limits[1]
        )
        res = res[sel]

    data = params.load_data(method="aperture")
    grid = res.as_array(names=("a", "e", "t0", "m0", "omega", "i", "theta_0"))
    grid = grid.view("f8").reshape(grid.shape[0], 7)

    nimg = nimg or len(data["images"])
    ncols = max(4, nimg)
    fig, axes = plt.subplots(2, ncols, figsize=(ncols * 3, 6), layout="constrained")

    for i in range(ncols):
        ax = axes[0, i]
        if i < nimg:
            plot_orbits(
                grid, res["snr_gradient"], data["images"][i], params.scale, ax=ax
            )
            ax.set(title=f"Image {i}")
        else:
            ax.axis("off")

    plot_snr_hist(res["snr_gradient"], res["snr_brut_force"], ax=axes[1, 0])
    plot_snr_curve(res["snr_gradient"], res["snr_brut_force"], ax=axes[1, 1])

    ax = axes[1, 2]
    for i, arr in enumerate(data["noise"]):
        ax.plot(arr, lw=1, alpha=0.8, label=str(i) if i < 10 else None)
    ax.legend(fontsize="x-small")
    ax.set(title="Noise", yscale="log")

    ax = axes[1, 3]
    for i, arr in enumerate(data["bkg"]):
        ax.plot(arr, lw=1, alpha=0.8, label=str(i) if i < 10 else None)
    ax.legend(fontsize="x-small")

    arr = data["bkg"][:, int(params.r_mask - 1) :]
    ymin = np.nanmin(arr)
    ymax = np.nanmax(arr)
    ymin = ymin / 2 if ymin > 0 else ymin * 2
    ymax = ymax * 2 if ymax > 0 else ymax / 2
    ax.set(title="Background", ylim=(ymin, ymax))

    if savefig:
        fig.savefig(savefig)


def create_masked_image(N, r_mask, r_mask_ext):
    """
    Create an NxN image with zeros for r <= r_mask and r >= r_mask_ext, and ones elsewhere.
    """
    image = np.zeros((N, N))
    center = N // 2
    for x in range(N):
        for y in range(N):
            r = np.sqrt((x - center)**2 + (y - center)**2)
            if r_mask < r < r_mask_ext:
                image[x, y] = 1
    return image

def calculate_unmasked_light_percentage(fwhm, image, center, dist):
    """
    Calculate the percentage of unmasked light for each position in `dist` using an image.
    """
    N = image.shape[0]
    radius = fwhm
    percentages = []

    for d in dist:
        mask = np.zeros_like(image)
        x0, y0 = center[0] + d, center[1]

        # Create a mask for the integration circle
        for x in range(N):
            for y in range(N):
                if np.sqrt((x - x0)**2 + (y - y0)**2) <= radius:
                    mask[x, y] = 1

        # Calculate the unmasked area
        unmasked_area = np.sum(image * mask)
        total_area = np.pi * radius**2

        # Calculate percentage of unmasked area
        percentage = unmasked_area / total_area
        percentages.append(percentage)

    return percentages

def calculate_unmasked_light_percentage_photutils(r_mask, r_mask_ext, fwhm, image, center, dist):
    """
    Calculate the percentage of unmasked light using photutils for sub-pixel integration.
    """

    radius = fwhm
    percentages = []

    for d in dist:

        if d + radius <= r_mask or d - radius >= r_mask_ext:
            # Entire circle is masked
            percentages.append(0)
        elif d - radius >= r_mask and d + radius <= r_mask_ext:
            # Entire circle is unmasked
            percentages.append(1)
        else:
            # Define the position of the aperture
            x0, y0 = center[0] + d, center[1]
            aperture = CircularAperture((x0, y0), r=radius)

            # Perform aperture photometry
            phot_table = aperture_photometry(image, aperture)

            # Calculate the unmasked area
            unmasked_area = phot_table['aperture_sum'][0]
            total_area = np.pi * radius**2

            # Calculate percentage of unmasked area
            percentage = unmasked_area / total_area
            percentages.append(percentage)

    return percentages


def percent_light_unmasked(orbital_parameters, ts, size_image, scale, fwhm, r_mask, r_mask_ext):

    # Compute image with 0 for r<r_mask and r>r_mask_ext and one elsewhere
    image_masked = create_masked_image(size_image, r_mask, r_mask_ext)
    center = (size_image // 2, size_image // 2)

    percentages_lum_unmasked = []

    for j in range(orbital_parameters.shape[0]):
        a, e, t0, m0, omega, i, theta_0 = orbital_parameters[j]
        xp, yp = orbit.project_position_full(ts, a, e, t0, m0, omega, i, theta_0).T # Check .T is required !!!
        xpix = scale * xp
        ypix = scale * yp
        dist = [math.sqrt(x ** 2 + y ** 2) for x, y in zip(xpix, ypix)]

        # Calculate the percentage of unmasked light
        # Calculate at each epoch (i.e. for n dist)
        percentages = calculate_unmasked_light_percentage_photutils(r_mask, r_mask_ext, fwhm, image_masked, center, dist)
        percentages_lum_unmasked.append(sum(percentages) / len(dist))

    return percentages_lum_unmasked


def proba_detection_file(path, params, snr_ks_seuil):
    """
    Process each row in the 'DATA' dataset of an H5 file, compute DeltaS for each row,
    and save the results in a .npy file.

    Parameters:
    h5_file_path (str): Path to the H5 file.
    snr_ks_seuil (float): The SNR KS threshold value.

    Output:
    A .npy file named 'proba_detection_file.npy' containing the processed data.
    """
    from ..utils import Params

    if isinstance(path + params, str):
        params = Params.read(path + params)


    time = params.time
    time_parts = time.split("+")
    epochs = [float(part) for part in time_parts]

    # Open the H5 file for reading
    with h5py.File(path + '/brute_grid/res.h5', 'r') as h5_file:
        # Extract the entire 'DATA' dataset
        data = h5_file['DATA'][:]

        # Computation of the percentage behind masks for each orbit (with a mean on the epochs)

        orbital_parameters = data[:, 0:7]
        percentages_lum_unmasked = percent_light_unmasked(orbital_parameters, epochs, params.n, params.scale, params.fwhm, params.r_mask, params.r_mask_ext)

        # Initialize an empty array for the output data
        output_data = np.zeros((data.shape[0], 10), dtype=np.float32)

        # Columns: DeltaS, contrast, mass_planet, a, e, t0, m0, omega, i, theta0
        # a, e, t0, m0, omega, i, theta0 are columns 0 to 6 in the 'DATA' dataset
        output_data[:, 3:10] = orbital_parameters

        # Calculate DeltaS for each row
        output_data[:, 0] = snr_ks_seuil * data[:, 8] - data[:, 7]  # DeltaS = SNR_KS_seuil * noise - signal

        # contrast (column 1) and mass_planet (column 2) are set to zero for now
        output_data[:, 1:3] = 0

    # Save the output data to a .npy file
    np.save(path + '/proba_detection_file.npy', output_data)


#def create_proba_detection_file(path, starAge, starAppMag, filter, model_mag_mass, size_images, N_file, numCore, maxSnrKS, kContrast, starMass, starDist, nbIm, epochs):
    """
    Create a numpy array file for a given core containing, for each (non redundant) orbit, the signal difference to
    reach maxSnrKS (1st column), the contrast (2nd column), the mass (3rd column) and the orbital parameters

    :param path: Path to the directory where signal, noise and grid files are for one core (same as numpy array saving directory)
    :param images_cube: cube of images. Used only in the simplified version (I=0 => in mask; I !=0 => out of mask)
    :param size_images: integer of the size of the square image
    :param N_file: Number of files per core
    :param numCore: Core number
    :param maxSnrKS: Snr to reach
    :param kContrast: Correcting coefficient due to images normalization

    res=np.array([[]]*9).transpose()

    for k in range(N_file):
        sub_s_values = np.load(path+'/s_values'+str(numCore)+'_'+str(k)+'.npy', allow_pickle=True)
        sub_n_values = np.load(path+'/n_values' + str(numCore) + '_' + str(k) + '.npy', allow_pickle=True)
        sub_grid = np.load(path+'/grid'+str(numCore)+'_'+str(k)+'.npy', allow_pickle=True)

        #read each orbit in grid files
        for xInd in [np.array([aInd,eInd,t0Ind,omegaInd,iInd,theta0Ind]) for aInd in range(int(sub_grid.shape[1])) for eInd in range(int(sub_grid.shape[2])) for t0Ind in range(int(sub_grid.shape[3])) for omegaInd in range(int(sub_grid.shape[4])) for iInd in range(int(sub_grid.shape[5])) for theta0Ind in range(int(sub_grid.shape[6]))]:
            x = sub_grid[:,xInd[0],xInd[1],xInd[2],xInd[3],xInd[4],xInd[5]] #orbital parameters
            s = sub_s_values[xInd[0],xInd[1],xInd[2],xInd[3],xInd[4],xInd[5]] #signal
            n = sub_n_values[xInd[0],xInd[1],xInd[2],xInd[3],xInd[4],xInd[5]] #noise



            DeltaS = maxSnrKS * n - s #compute signal to reach maxSnrKS


            # nb_epochs_out_mask = numb_epochs_out_mask_proportion(x, size_images, starMass, nbIm, epochs)
            #print ('nb_epochs_out_mask = ', nb_epochs_out_mask)

            if DeltaS < 0:
                print ('Warning: One DeltaS value < 0 and not taken into account in the statistic = ', str(DeltaS))

            #if nb_epochs_out_mask > 0.01:

                #if DeltaS > 0:
                #    contrast = DeltaS * kContrast / nb_epochs_out_mask  # compute contrast from DeltaS
                #    mass = mag_to_mass(starAge, starDist, starAppMag, -2.5 * np.log10(contrast), filter, model=model_mag_mass)[0]  # convert contrast in mass, the vigan function takes as argument the apparent magnitude relative to the star
                #    res = np.insert(res, 0, np.array([DeltaS, contrast, mass, x[0], x[1], x[2], x[3], x[4], x[5]]),
                #                    axis=0)

                #elif nb_epochs_out_mask <= 0.01 :
                #    res = np.insert(res, 0, np.array([DeltaS, 999., 999., x[0], x[1], x[2], x[3], x[4], x[5]]),
                #                    axis=0)

    #np.save(path + '/mess_'+ str(numCore)+ '.npy', res)
    """


def plot_converge_points_map(images, ts, scale, N, M, log_probabilities, samples, values_dir):
    """
    Displays the projected positions of orbital samples on images,
    colored by their posterior log-probabilities.
    
    Parameters:
    -----------
    log_probabilities : array-like
        Log-posterior probabilities associated with each sample.
    samples : array-like
        Orbital parameter samples (N x 7).
    """

    # Spatial grid of the image
    image_size = M
    X, Y = np.meshgrid(np.arange(image_size), np.arange(image_size))
    distances = np.hypot(X - image_size // 2 + 0.5, Y - image_size // 2 + 0.5)
    
    # Spatial mask (excludes the center and edges)
    
    mask = (distances > 20) & (distances < image_size // 2)

    # Storage of coordinates and log-probabilities for each image
    x_coords, y_coords, color_values = [[], [], [], []], [[], [], [], []], [[], [], [], []]

    for i, sample in enumerate(samples):
        # Orbital projection of positions
        projected_positions = orbit.project_position_full(
            ts, *sample[:7]
        )
        projected_positions += scale
        projected_positions += image_size // 2  # Centering on the image

        for j in range(N):
            x_coords[j].append(projected_positions[j][0])
            y_coords[j].append(projected_positions[j][1])
            color_values[j].append(log_probabilities[i])
    
    best_shape = None
    min_max_dim = float('inf')
    
    # Determine the best subplot grid layout (rows, cols)
    for rows in range(1, N + 1):
        cols = -(-N // rows)  # ceil division
        max_dim = max(rows, cols)
        if max_dim < min_max_dim:
            min_max_dim = max_dim
            best_shape = (rows, cols)
        elif max_dim == min_max_dim:
            if abs(rows - cols) < abs(best_shape[0] - best_shape[1]):
                best_shape = (rows, cols)

    
    rows, cols = best_shape
    fig, axs = mpl.pyplot.subplots(rows, cols, figsize=(10 * cols, 10 * rows))

    # Ensure axs is 2D for consistent access
    if rows == 1 and cols == 1:
        axs = np.array([[axs]])
    elif rows == 1:
        axs = axs[np.newaxis, :]
    elif cols == 1:
        axs = axs[:, np.newaxis]


    # Display the N subplots, hide the unused ones
    for k in range(rows * cols):
        r, c = divmod(k, cols)
        ax = axs[r][c]
        if k < N:
            # Apply the mask to the image
            masked_image = np.where(mask, images[k], np.nan)

            # Display the image in grayscale
            ax.imshow(masked_image, origin='lower', cmap='gray')

            # Display the projected positions
            scatter = ax.scatter(
                y_coords[k], x_coords[k], c=color_values[k], cmap='coolwarm',
                s=20, alpha=0.8
            )

            # Add a colorbar
            mpl.pyplot.colorbar(scatter, ax=ax, label="Log Posterior Probability")
            ax.set_title(f"Time step number {k+1}", fontsize=25)
        else:
            ax.axis('off')  # Hide empty subplot

    mpl.pyplot.tight_layout()
    mpl.pyplot.show()
    mpl.pyplot.close()
