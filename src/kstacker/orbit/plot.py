"""
Functions used to represent the orbit of the planet
"""


import math

import matplotlib.pyplot as plt
import numpy as np

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

    xp, yp = orbit.project_position(
        orbit.position(ts, a, e, t0, m0), omega, i, theta_0
    ).T
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

    xp, yp = orbit.project_position(
        orbit.position(ts, a, e, t0, m0), omega, i, theta_0
    ).T
    xpix = npix // 2 + scale * xp
    ypix = npix // 2 + scale * yp
    plt.plot(ypix, xpix, "+", color="r")

    length = 1000.0 / res
    plt.plot([npix - length, npix], [-10, -10], "y")
    plt.text(npix - 2 * length / 3, -20, "1 arcsec", color="y")

    plt.savefig(filename + ".png")
    plt.close()
