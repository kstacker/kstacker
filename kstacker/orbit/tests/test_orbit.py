from kstacker.orbit import position, project_position
import numpy as np

A = 0.9
E = 0.55
T0 = -0.55
M0 = 1.133
OMEGA = 0.0
I = 1.13
THETA_0 = 0.0


def test_position_scalar():
    pos = position(0.0, A, E, T0, M0)
    expected = [-1.13492103, -0.52853524]
    np.testing.assert_array_almost_equal(pos, expected, decimal=8)


def test_position_array():
    pos = position(np.array([0.0, 0.3, 0.6]), A, E, T0, M0)
    expected = [
        [-1.13492103, -0.52853524],
        [0.16371403, 0.51217852],
        [-1.36238363, 0.2005189],
    ]
    np.testing.assert_array_almost_equal(pos, expected, decimal=8)


def test_project_position_scalar():
    pos = position(0.0, A, E, T0, M0)
    pos = project_position(pos, OMEGA, I, THETA_0)
    expected = [-1.13492103, -0.22550475]
    np.testing.assert_array_almost_equal(pos, expected, decimal=8)


def test_project_position_array():
    pos = position(np.array([0.0, 0.3, 0.6]), A, E, T0, M0)
    pos = project_position(pos, OMEGA, I, THETA_0)
    expected = [
        [-1.13492103, -0.22550475],
        [0.16371403, 0.21852599],
        [-1.36238363, 0.08555336],
    ]
    np.testing.assert_array_almost_equal(pos, expected, decimal=8)
