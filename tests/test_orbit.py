import numpy as np

from kstacker.orbit import (
    position,
    positions_at_multiple_times,
    project_position,
    project_position_full,
)

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


def test_position_multiple_times():
    pos = positions_at_multiple_times(
        np.array([0, 1]), np.array([[A, E, T0, M0], [A, E, T0, M0]])
    )
    expected = (
        [[-1.13492103, -1.13492103], [0.11133086, 0.11133086]],
        [[-0.52853524, -0.52853524], [-0.55547108, -0.55547108]],
    )
    np.testing.assert_array_almost_equal(pos, expected, decimal=8)


def test_project_position_scalar():
    pos = position(0.0, A, E, T0, M0)
    pos = project_position(pos, OMEGA, I, THETA_0)
    expected = [-1.13492103, -0.22550475]
    np.testing.assert_array_almost_equal(pos, expected, decimal=8)


def test_project_position_array():
    ts = np.array([0.0, 0.3, 0.6])
    pos = position(ts, A, E, T0, M0)
    pos = project_position(pos, OMEGA, I, THETA_0)
    expected = [
        [-1.13492103, -0.22550475],
        [0.16371403, 0.21852599],
        [-1.36238363, 0.08555336],
    ]
    np.testing.assert_array_almost_equal(pos, expected, decimal=8)

    pos = project_position_full(ts, A, E, T0, M0, OMEGA, I, THETA_0)
    np.testing.assert_array_almost_equal(pos, expected, decimal=8)
