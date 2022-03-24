# Function to get the position vector (cartesian coordinates) of a planet on a
# given orbit, at a given time

__author__ = "Mathias Nowak"
__email__ = "mathias.nowak@ens-cachan.fr"
__status__ = "Working"

import kepler
import numpy as np
from scipy.spatial.transform import Rotation


def position(t, a, e, t0, m):
    """
    Compute the position of a planet at given time t from orbital parameters .
    @param float t: time at which position shall be computed (in year)
    @param float a: semi-major axis (astronomical unit)
    @param float e: excentricity
    @param float t0: time at perihelion (year)
    @param float m: mass of the central body (in unit of solar masses)
    @return float[2]: position [x, y] of the planet at time t (astronomical unit) in the standard orbital frame
    (perihelion along [1, 0] vector, star in [0, 0])
    """
    # compute other useful orbital parameters
    # p = a * (1 - e**2)  # ellipse parameter
    n = 2 * np.pi * np.sqrt(m / a**3)  # orbital pulsation (in rad/year)

    M = n * (t - t0)
    E_t = kepler.solve(M, e)

    # convert anomaly to position in reference frame and return vector
    x = a * (np.cos(E_t) - e)
    y = a * np.sqrt(1 - e**2) * np.sin(E_t)

    if np.isscalar(E_t):
        return x, y
    else:
        return np.stack([x, y], axis=1)


def rotation_3d(vector, axis, theta):
    """
    A rather simple function to rotate a vector around one of its coordinates.
    @param float[3] vector: a 3d vector
    @param int axis: axis around which to rotate (can be 1, 2, or 3)
    @param float theta: rotation angle (rad)
    @return float[3]: rotated vector
    """
    if axis == 1:
        rot = Rotation.from_euler("x", -theta)
    if axis == 2:
        rot = Rotation.from_euler("y", -theta)
    if axis == 3:
        rot = Rotation.from_euler("z", -theta)

    return rot.apply(vector)


def euler_rotation(vector, alpha, beta, gamma):
    """
    A more sophisticated function to change coordinates of a 3d vector form orbital
    frame to observer frame, where the orbital frame is given with its eulerian angles.

    @param float[3] vector: initial vector, given in solid frame
    @param float alpha: precession angle around Z axis (rad)
    @param float beta: nutation angle around node line (rad)
    @param float gamma: proper rotation (rad)
    return float[3] vector given in observer frame
    """
    # We basically have to undo the eulerian transformation.
    # Last rotation of eulerian transformation is proper rotation. We undo that first.
    vector = rotation_3d(vector, 3, -gamma)
    # then we undo the rotatation around the node line, which is now axis X
    vector = rotation_3d(vector, 1, -beta)
    # Finally, we undo the precession rotation around Z axis
    vector = rotation_3d(vector, 3, -alpha)
    return vector


def project_position(position, omega, i, theta_0):
    """
    A function that projects the position of the planet initially given as a 2d
    vector in the orbital reference frame in the CCD frame

    @param float[2]: XY position in the orbital reference frame
    @param float omega: longitude of the ascending node, counted from North axis, in (North, West, Earth) coordinate system (in rad)
    @param float i: inclination of the orbital plane, counted from sky plane (in rad)
    @param float theta_0: argument of the periapsis (counted from line of nodes, in rad)
    @return float[2]: XY position in the observer frame projected along Z vector
    """
    # going to real 3d orbital reference fram
    position = np.atleast_2d(position)
    vector = np.pad(position, [(0, 0), (0, 1)], constant_values=0)
    # transforming to observer frame
    vector = euler_rotation(vector, omega, i, theta_0)
    # remove the z column
    vector = vector[:, :2]

    if vector.shape[0] == 1:
        # return (x, y) when the input contains only one position
        return vector[0]
    else:
        return vector
