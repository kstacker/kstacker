import pathlib

import h5py
import numpy as np
from astropy.io import ascii
from numpy.testing import assert_almost_equal

from kstacker.gradient_reoptimization import reoptimize_gradient
from kstacker.noise_profile import compute_noise_profiles
from kstacker.optimize import brute_force


def test_full_run(params_tmp):
    """Test a complete run, all in one to avoid recomputing several times the
    intermediate steps.
    """
    np.random.seed(42)
    path = pathlib.Path(params_tmp.work_dir)

    # Noise and background profiles ------------------------------------------
    compute_noise_profiles(params_tmp)

    path = pathlib.Path(params_tmp.work_dir)
    assert (path / "profiles" / "background_prof0.npy").is_file()
    assert (path / "profiles" / "noise_prof0.npy").is_file()

    # Brute force ------------------------------------------------------------
    brute_force(params_tmp, dry_run=False, num_threads=1, show_progress=True)

    with h5py.File(path / "brute_grid" / "res.h5") as f:
        assert f["Orbital grid"].shape == (3130, 3)
        assert f["Projection grid"].shape == (8862, 3)
        assert f["DATA"].shape == (313000, 9)

        assert np.isclose(f["DATA"][:, 8].min(), -16.74728)
        assert np.isclose(f["DATA"][:, 8].max(), 0.08774643)

    with h5py.File(path / "values" / "res_grid.h5") as f:
        best = f["Best solutions"][:]
        assert best.shape == (100, 9)
        # a, e, t0, omega, i, theta_0
        expected = [53.75, 0.08, -98.257, -1.944, 0.523, -2.841]
        assert_almost_equal(best[0, :6], expected, decimal=3)
        # signal, noise, snr
        expected = [4.4286370e-04, 2.6443917e-05, -1.6747280e01]
        assert_almost_equal(best[0, 6:], expected, decimal=6)
        # 10 best SNRs
        expected = [
            -16.74728,
            -16.726263,
            -16.724823,
            -16.72005,
            -16.718771,
            -16.717426,
            -16.707235,
            -16.701384,
            -16.694956,
            -16.69361,
        ]
        assert_almost_equal(best[:10, 8], expected, decimal=5)

    # Reoptimize -------------------------------------------------------------
    reoptimize_gradient(params_tmp, n_orbits=5)
    names = ["idx", "snr_brut", "snr_grad", "a", "e", "t0", "omega", "i", "theta_0"]
    res = ascii.read(path / "values" / "results.txt", names=names)
    expected = [-17.0618, -17.0618 , -17.0617, -17.0588, -17.0514]
    assert_almost_equal(res["snr_grad"], expected, decimal=3)
