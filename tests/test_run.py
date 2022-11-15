import pathlib

import h5py
import numpy as np
from astropy.io import ascii
from numpy.testing import assert_almost_equal

from kstacker.gradient_reoptimization import reoptimize_gradient
from kstacker.noise_profile import compute_noise_profiles, compute_snr_plots
from kstacker.optimize import brute_force
from kstacker.orbit import plot_results


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
    assert (path / "profiles" / "snr_plot_steps" / "Plot_noise0.pdf").is_file()

    compute_snr_plots(params_tmp)
    assert (
        path / "profiles" / "snr_plot_steps" / "snr_graph" / "steps_a_40.0-62.0.pdf"
    ).is_file()

    # Brute force ------------------------------------------------------------
    brute_force(params_tmp, dry_run=False, num_threads=1, show_progress=True)

    with h5py.File(path / "brute_grid" / "res.h5") as f:
        assert f["Orbital grid"].shape == (3130, 4)
        assert f["Projection grid"].shape == (8862, 3)
        assert f["DATA"].shape == (313000, 10)

        assert np.isclose(f["DATA"][:, 9].min(), -17.501013)
        assert np.isclose(f["DATA"][:, 9].max(), 0.12237881)

    with h5py.File(path / "values" / "res_grid.h5") as f:
        best = f["Best solutions"][:]
        assert best.shape == (100, 10)
        # a, e, t0, omega, i, theta_0
        expected = [59.25, 0.28, -49.133, 1.59, 1.046, 0.523, 1.196]
        assert_almost_equal(best[0, :7], expected, decimal=3)
        # signal, noise, snr
        expected = [3.585e-04, 2.048e-05, -17.501013]
        assert_almost_equal(best[0, 7:], expected, decimal=6)
        # 10 best SNRs
        expected = [
            -17.50101,
            -17.49456,
            -17.46663,
            -17.41481,
            -17.40319,
            -17.39916,
            -17.39638,
            -17.39090,
            -17.37733,
            -17.37152,
        ]
        assert_almost_equal(best[:10, 9], expected, decimal=5)

    # Reoptimize -------------------------------------------------------------
    reoptimize_gradient(params_tmp, n_orbits=5)
    res = ascii.read(path / "values" / "results.txt")
    expected = [-17.236, -17.216, -17.181, -17.151, -17.137]
    assert_almost_equal(res["snr_gradient"], expected, decimal=3)

    # Plot results -----------------------------------------------------------
    plot_results(params_tmp, nimg=3)
