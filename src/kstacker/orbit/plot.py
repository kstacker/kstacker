"""
Functions used to represent the orbit of the planet
"""


import math
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
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

    # if omegatheta:
    #     omega_theta = omega - theta_0
    #     omega_p_theta = omega + theta_0

    snr_grad = res["snr_gradient"]

    # 1st row : a as a function of others parameters
    axes[0, 0].hist(a, bins=nbins, color="darkcyan")
    # 2nd row : e as a function of others parameters
    im = axes[1, 0].scatter(a, e, c=snr_grad, cmap=color_scatter)
    axes[1, 1].hist(e, bins=nbins, color="darkcyan")
    # 3rd row : t0 as a function of others parameters
    axes[2, 0].scatter(a, t0, c=snr_grad, cmap=color_scatter)
    axes[2, 1].scatter(e, t0, c=snr_grad, cmap=color_scatter)
    axes[2, 2].hist(t0, bins=nbins, color="darkcyan")
    # 4rd row : m0 as a function of others parameters
    axes[3, 0].scatter(a, m0, c=snr_grad, cmap=color_scatter)
    axes[3, 1].scatter(e, m0, c=snr_grad, cmap=color_scatter)
    axes[3, 2].scatter(t0, m0, c=snr_grad, cmap=color_scatter)
    axes[3, 3].hist(m0, bins=nbins, color="darkcyan")
    # 5th row : omega as a function of others parameters
    axes[4, 0].scatter(a, omega, c=snr_grad, cmap=color_scatter)
    axes[4, 1].scatter(e, omega, c=snr_grad, cmap=color_scatter)
    axes[4, 2].scatter(t0, omega, c=snr_grad, cmap=color_scatter)
    axes[4, 3].scatter(m0, omega, c=snr_grad, cmap=color_scatter)
    axes[4, 4].hist(omega, bins=nbins, color="darkcyan")
    # 6th row : i as a function of others parameters
    axes[5, 0].scatter(a, i, c=snr_grad, cmap=color_scatter)
    axes[5, 1].scatter(e, i, c=snr_grad, cmap=color_scatter)
    axes[5, 2].scatter(t0, i, c=snr_grad, cmap=color_scatter)
    axes[5, 3].scatter(m0, i, c=snr_grad, cmap=color_scatter)
    axes[5, 4].scatter(omega, i, c=snr_grad, cmap=color_scatter)
    axes[5, 5].hist(i, bins=nbins, color="darkcyan")
    # 7th row : theta0 as a function of others parameters
    axes[6, 0].scatter(a, theta_0, c=snr_grad, cmap=color_scatter)
    axes[6, 1].scatter(e, theta_0, c=snr_grad, cmap=color_scatter)
    axes[6, 2].scatter(t0, theta_0, c=snr_grad, cmap=color_scatter)
    axes[6, 3].scatter(m0, theta_0, c=snr_grad, cmap=color_scatter)
    axes[6, 4].scatter(omega, theta_0, c=snr_grad, cmap=color_scatter)
    axes[6, 5].scatter(i, theta_0, c=snr_grad, cmap=color_scatter)
    axes[6, 6].hist(theta_0, bins=nbins, color="darkcyan")

    # Figure Title
    fig.suptitle(
        f"Corner-plot of the {norbits} K-Stacker orbits at higher SNR", fontsize=16
    )

    # Axes Labels
    axes[6, 0].set_xlabel("a (a.u.)")
    axes[6, 1].set_xlabel("e")
    axes[6, 2].set_xlabel("$t_0$ (yrs)")
    axes[6, 3].set_xlabel("$m0$ (solar_mass)")
    axes[6, 4].set_xlabel(r"$\Omega$ (rad)")
    axes[6, 5].set_xlabel("i (rad)")
    axes[6, 6].set_xlabel(r"$\omega$ (rad)")
    axes[1, 0].set_ylabel("e")
    axes[2, 0].set_ylabel("$t_0$ (yrs)")
    axes[3, 0].set_ylabel("$m0$ (solar_mass)")
    axes[4, 0].set_ylabel(r"$\Omega$ (rad)")
    axes[5, 0].set_ylabel("i (rad)")
    axes[6, 0].set_ylabel(r"$\omega$ (rad)")
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
