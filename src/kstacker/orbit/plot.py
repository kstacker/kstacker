"""
Functions used to represent the orbit of the planet
"""


import math
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import ZScaleInterval

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
    plt.imshow(back_image, origin="lower", interpolation="none", cmap="gray")
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
    ax.legend(loc="upper left")
    ax.set(title="SNR Histogram")


def plot_snr_curve(snr_gradient, snr_brut_force, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(snr_gradient, label="snr_gradient", drawstyle="steps-mid")
    snr_brut = np.sort(snr_brut_force)[::-1]
    ax.plot(snr_brut, label="snr_brut_force", drawstyle="steps-mid")
    ax.legend()
    ax.set(title="SNR Curves")


def plot_results(params, nimg=None, savefig=None):
    from ..utils import Params, read_results

    if isinstance(params, str):
        path = os.path.dirname(params)
        params = Params.read(params)
        params.work_dir = path

    res = read_results(os.path.join(params.work_dir, "values", "results.txt"), params)
    res["snr_gradient"] *= -1
    res["snr_brut_force"] *= -1

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
    ax.legend(fontsize="x-small", loc="upper left")
    ax.set(title="Noise", yscale="log")

    ax = axes[1, 3]
    for i, arr in enumerate(data["bkg"]):
        ax.plot(arr, lw=1, alpha=0.8, label=str(i) if i < 10 else None)
    ax.legend(fontsize="x-small", loc="upper left")

    arr = data["bkg"][:, params.r_mask - 1 :]
    ymin = np.nanmin(arr)
    ymax = np.nanmax(arr)
    ymin = ymin / 2 if ymin > 0 else ymin * 2
    ymax = ymax * 2 if ymax > 0 else ymax / 2
    ax.set(title="Background", ylim=(ymin, ymax))

    if savefig:
        fig.savefig(savefig)
