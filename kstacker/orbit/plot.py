"""
Functions used to represent the orbit of the planet
"""

__author__ = "Mathias Nowak"
__email__ = "mathias.nowak@ens-cachan.fr"
__status__ = "Testing"

import math

import matplotlib.pyplot as plt
import numpy as np

from . import orbit

# import pyfits


def plot_orbites3(ts, x, m0, filename):
    """
    Function used to plot the true orbit (in red) and the best orbit (in blue) found in the projected plane of sky (au-au).
    @param float[6] x: parameters of the best orbit found (a, e, t0, omega, i, theta0) in (au, nounit, year, rad, rad, rad)
    @param float m0: mass of the central star (in solar mass)
    @param string filename: name of the file where to save the plot (extension will be .png)
    """

    [a, e, t0, omega, i, theta_0] = x
    p = a * (1 - e**2)
    thetas = np.linspace(0, 2 * math.pi, 1000)
    r = p / (1 + e * np.cos(thetas))
    xvalues = r * np.cos(thetas)
    yvalues = r * np.sin(thetas)

    x_proj = np.zeros(1000)
    y_proj = np.zeros(1000)

    for k in range(1000):
        [xp, yp] = orbit.project_position([xvalues[k], yvalues[k]], omega, i, theta_0)
        x_proj[k] = xp
        y_proj[k] = yp

    plt.figure(0, figsize=(6, 6))
    plt.plot(y_proj, x_proj, color="blue")
    plt.plot([0], [0], "+", color="red")
    plt.axis([-35.0, 35.0, -35.0, 35.0])
    for t in ts:
        [xp, yp] = orbit.project_position(
            orbit.position(t, a, e, t0, m0), omega, i, theta_0
        )
        plt.scatter(yp, xp, marker="+")
    #    for t in ts:
    #        [xp, yp]=orbit.project_position(orbit.position(t, 5., 0., -1.0, 1.),0.,0.,0.)
    #        plt.scatter(xp,yp,marker='+', color='red')

    plt.xlabel("Astronomical Units")
    plt.ylabel("Astronomical Units")
    #    plt.legend(["Orbite reelle", "Orbite trouvee"])

    plt.savefig(filename + ".png")
    plt.close()
    return None


def plot_orbites2(ts, x, m0, ax, filename):
    """
    Function used to plot the true orbit (in red) and the best orbit (in blue) found in the projected plane of sky (au-au).
    @param float[6] x: parameters of the best orbit found (a, e, t0, omega, i, theta0) in (au, nounit, year, rad, rad, rad)
    @param float m0: mass of the central star (in solar mass)
    @param string filename: name of the file where to save the plot (extension will be .png)
    @param float[4] ax: scale of axes, xmin, xmax, ymin, ymax in astronomical units
    """

    [a, e, t0, omega, i, theta_0] = x
    p = a * (1 - e**2)
    thetas = np.linspace(0, 2 * math.pi, 1000)
    r = p / (1 + e * np.cos(thetas))
    xvalues = r * np.cos(thetas)
    yvalues = r * np.sin(thetas)

    x_proj = np.zeros(1000)
    y_proj = np.zeros(1000)

    for k in range(1000):
        [xp, yp] = orbit.project_position([xvalues[k], yvalues[k]], omega, i, theta_0)
        x_proj[k] = xp
        y_proj[k] = yp

    plt.figure(0, figsize=(6, 6))
    plt.plot(y_proj, x_proj, color="blue")
    plt.plot([0], [0], "+", color="red")
    plt.axis(ax)
    for t in ts:
        [xp, yp] = orbit.project_position(
            orbit.position(t, a, e, t0, m0), omega, i, theta_0
        )
        plt.scatter(yp, xp, marker="+")
    #    for t in ts:
    #        [xp, yp]=orbit.project_position(orbit.position(t, 5., 0., -1.0, 1.),0.,0.,0.)
    #        plt.scatter(xp,yp,marker='+', color='red')

    plt.xlabel("Astronomical Units")
    plt.ylabel("Astronomical Units")
    #    plt.legend(["Orbite reelle", "Orbite trouvee"])

    plt.savefig(filename + ".png")
    plt.close()
    return None


def plot_orbites(x, m, x0, m0, ts, filename):
    """
    Function used to plot the true orbit (in red) and the best orbit (in blue) found in the projected plane of sky (au-au).
    @param float[6] x: parameters of the best orbit found (a, e, t0, omega, i, theta0) in (au, nounit, year, rad, rad, rad)
    @param float[6] x0: parameters of the true orbit (a, e, t0, omega, i, theta0) in (au, nounit, year, rad, rad, rad)
    @param float m0: mass of the central star (in solar mass)
    @param string filename: name of the file where to save the plot (extension will be .png)
    """

    [a, e, t0, omega, i, theta_0] = x
    [a0, e0, t00, omega0, i0, theta_00] = x0
    p = a * (1 - e**2)
    p0 = a0 * (1 - e0**2)
    thetas = np.linspace(0, 2 * math.pi, 1000)
    r = p / (1 + e * np.cos(thetas))
    r0 = p0 / (1 + e0 * np.cos(thetas))
    xvalues = r * np.cos(thetas)
    yvalues = r * np.sin(thetas)
    x0values = r0 * np.cos(thetas)
    y0values = r0 * np.sin(thetas)

    # [x, y]=position
    # going to polar coordinates and adding the defect angle to the initial angle
    # r=np.sqrt(x**2+y**2)
    # theta=np.angle(x+1j*y)+((def_ang/2.0)*np.pi)/180.0*random.randint(-1,1)
    # getting new coordinates and return position
    # x_new=r*math.cos(theta)
    # y_new=r*math.sin(theta)

    x_proj = np.zeros(1000)
    y_proj = np.zeros(1000)
    x0_proj = np.zeros(1000)
    y0_proj = np.zeros(1000)

    for k in range(1000):
        [xp, yp] = orbit.project_position([xvalues[k], yvalues[k]], omega, i, theta_0)
        [x0p, y0p] = orbit.project_position(
            [x0values[k], y0values[k]], omega0, i0, theta_00
        )
        x_proj[k] = xp
        y_proj[k] = yp
        x0_proj[k] = x0p
        y0_proj[k] = y0p

    plt.figure(0, figsize=(6, 6))
    plt.plot(y0_proj, x0_proj, color="red")
    plt.plot(y_proj, x_proj, color="blue")
    plt.plot([0], [0], "+", color="red")
    plt.axis([-10.0, 10.0, -10.0, 10.0])

    for t in ts:
        [xp, yp] = orbit.project_position(
            orbit.position(t, a, e, t0, m), omega, i, theta_0
        )
        plt.plot(yp, xp, marker="o", color="blue", markersize=13)
    for t in ts:
        [xp, yp] = orbit.project_position(
            orbit.position(t, a0, e0, t00, m0), omega0, i0, theta_00
        )
        plt.plot(yp, xp, marker="o", color="red", markersize=13)

    plt.xlabel("Astronomical Units")
    plt.ylabel("Astronomical Units")
    #    plt.legend(["Orbite reelle", "Orbite trouvee"])

    plt.savefig(filename + ".png")
    plt.close()
    return None


def plot_orbites_TN(x, m, angles, x0, m0, ts, num_im, filename):
    """
    Function used to plot the true orbit (in red) and the best orbit (in blue) found in the projected plane of sky (au-au).
    @param float[6] x: parameters of the best orbit found (a, e, t0, omega, i, theta0) in (au, nounit, year, rad, rad, rad)
    @param float[6] x0: parameters of the true orbit (a, e, t0, omega, i, theta0) in (au, nounit, year, rad, rad, rad)
    @param float m0: mass of the central star (in solar mass)
    @param float[p] angles: defect angles on the p images
    @param int num_im: number of images
    @param string filename: name of the file where to save the plot (extension will be .png)
    """

    [a, e, t0, omega, i, theta_0] = x
    [a0, e0, t00, omega0, i0, theta_00] = x0
    p = a * (1 - e**2)
    p0 = a0 * (1 - e0**2)
    thetas = np.linspace(0, 2 * math.pi, 1000)
    r = p / (1 + e * np.cos(thetas))
    r0 = p0 / (1 + e0 * np.cos(thetas))
    xvalues = r * np.cos(thetas)
    yvalues = r * np.sin(thetas)
    x0values = r0 * np.cos(thetas)
    y0values = r0 * np.sin(thetas)
    ang = angles

    # [x, y]=position
    # going to polar coordinates and adding the defect angle to the initial angle

    # r=np.sqrt(x**2+y**2)
    # theta=np.angle(x+1j*y)+((def_ang/2.0)*np.pi)/180.0*random.randint(-1,1)                                                                                                                                          #getting new coordinates and return position

    # x_new=r*math.cos(theta)
    # y_new=r*math.sin(theta)

    x_proj = np.zeros(1000)
    y_proj = np.zeros(1000)
    x0_proj = np.zeros(1000)
    y0_proj = np.zeros(1000)

    for k in range(1000):
        [xp, yp] = orbit.project_position([xvalues[k], yvalues[k]], omega, i, theta_0)
        [x0p, y0p] = orbit.project_position(
            [x0values[k], y0values[k]], omega0, i0, theta_00
        )
        x_proj[k] = xp
        y_proj[k] = yp
        x0_proj[k] = x0p
        y0_proj[k] = y0p

    plt.figure(0, figsize=(6, 6))
    plt.plot(y0_proj, x0_proj, color="red")
    plt.plot(y_proj, x_proj, color="blue")
    plt.plot([0], [0], "+", color="red")
    plt.axis([-10.0, 10.0, -10.0, 10.0])

    for l in range(num_im):
        [xp, yp] = orbit.project_position(
            orbit.position(ts[l], a, e, t0, m), omega, i, theta_0
        )
        rp = np.sqrt(xp**2 + yp**2)
        thetap = np.angle(xp + 1j * yp) + (ang[l] * np.pi) / 180.0
        xp = rp * math.cos(thetap)
        yp = rp * math.sin(thetap)
        plt.plot(yp, xp, marker="o", color="blue", markersize=13)
    for t in ts:
        [x0p, y0p] = orbit.project_position(
            orbit.position(t, a0, e0, t00, m0), omega0, i0, theta_00
        )
        plt.plot(y0p, x0p, marker="o", color="red", markersize=13)

    plt.xlabel("Astronomical Units")
    plt.ylabel("Astronomical Units")
    #    plt.legend(["Orbite reelle", "Orbite trouvee"])

    plt.savefig(filename + ".png")
    plt.close()
    return None


def plot_ontop(x, m0, d, ts, res, back_image, filename):
    """
    Function used to plot an orbit on top of a background corono image.
    @param float[6] x: parameters of the best orbit found (a, e, t0, omega, i, theta0) in (au, nounit, year, rad, rad, rad)
    @param float m0: mass of the central star (in solar mass)
    @param float d: distance of the star (in pc)
    @param float[q] ts: time steps (in years) at which the planet shall be plotted
    @param float res: res of the image (in mas/pixel)
    @param float[n, n]: background image
    @param string filename: name of the file where the image shall be saved (extension .png will be added)
    """
    n = np.size(back_image[0])
    q = np.size(ts)

    scale = 1.0 / (d * (res / 1000.0))

    [a, e, t0, omega, i, theta_0] = x

    p = a * (1 - e**2)
    thetas = np.linspace(-2 * math.pi, 0, 1000)
    r = p / (1 + e * np.cos(thetas))
    xvalues = r * np.cos(thetas)
    yvalues = r * np.sin(thetas)

    x_proj = np.zeros(1000)
    y_proj = np.zeros(1000)

    for k in range(1000):
        [xp, yp] = orbit.project_position([xvalues[k], yvalues[k]], omega, i, theta_0)
        x_proj[k] = xp
        y_proj[k] = yp

    plt.figure(1)
    plt.axis("off")
    plt.imshow(
        back_image, origin="lower", interpolation="none", cmap=plt.get_cmap("gray")
    )
    plt.scatter(n / 2 + scale * y_proj, n / 2 + scale * x_proj, color="b", s=0.1)

    for k in range(q):
        [x, y] = orbit.project_position(
            orbit.position(ts[k], a, e, t0, m0), omega, i, theta_0
        )
        xpix = n / 2 + scale * x
        ypix = n / 2 + scale * y
        plt.plot(ypix, xpix, "+", color="r")
    #       plt.plot(xpix, ypix,'o', color='r', markersize=10)
    #    [x,y]=orbit.project_position(orbit.position(t0, a, e, t0, m0), omega, i, theta_0)
    #    xpix=n/2+scale*x
    #    ypix=n/2+scale*y
    #    plt.scatter(xpix-2, ypix, color='b', marker='>')

    length = 1000.0 / res
    plt.plot([n - length, n], [-10, -10], "y")
    plt.text(n - 2 * length / 3, -20, "1 arcsec", color="y")

    plt.savefig(filename + ".png")
    plt.close()
    # hdu = pyfits.PrimaryHDU(back_image.T)
    # hdulist = pyfits.HDUList([hdu])
    # hdulist.writeto("/home/destevez/Kstacker/k-stacker_code/branches/kstacker_real_arg/"+filename+".fits")
    return None
