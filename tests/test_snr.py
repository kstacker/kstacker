import numpy as np
import pytest
from kstacker._utils import cy_compute_snr
from kstacker.noise_profile import compute_noise_profiles
from kstacker.orbit import orbit
from kstacker.snr import (
    compute_signal_and_noise_grid,
    compute_snr,
    compute_snr_detailed,
)
from numpy.testing import assert_allclose

# orbit parameters: a, e, t0, m, omega, i theta_0
x = [53.75, 0.08, -98.2575, 1.59, -1.9438095, 0.5233333, -2.8409524]
# expected values with plain summation
expected = {
    # method  :  signal,       noise,          snr
    "convolve": [0.000443, 2.811e-05, 15.739],
    "aperture": [0.000448, 2.863e-05, 15.636],
    "aperture_exc": [0.000448, 2.837e-05, 15.514],
    "aperture_interp": [0.000448, 2.811e-05, 15.964],
}
# expected values with inverse variance weighting
expected_invvar = {
    "convolve": [0.000110, 6.91e-06, 15.863],
    "aperture": [0.000112, 7.08e-06, 15.771],
    "aperture_exc": [0.000112, 7.02e-06, 15.641],
}


@pytest.fixture(scope="module")
def params_with_images(params_tmp):
    # Pre-compute convolved images
    np.random.seed(42)
    params_tmp.snr_plot = "no"
    compute_noise_profiles(params_tmp)
    return params_tmp


@pytest.mark.parametrize("method", ["convolve", "aperture"])
def test_compute_snr_detailed(params_with_images, method):
    params = params_with_images
    res = compute_snr_detailed(params, x, method=method, verbose=True)
    assert_allclose(res.meta["signal_sum"], expected[method][0], atol=1e-6, rtol=0)
    assert_allclose(res.meta["noise_sum"], expected[method][1], atol=1e-6, rtol=0)
    assert_allclose(res.meta["snr_sum"], expected[method][2], atol=1e-3, rtol=0)


@pytest.mark.parametrize("method", ["convolve", "aperture"])
def test_compute_signal_and_noise_grid(params_with_images, method):
    params = params_with_images
    ts = params.get_ts()
    xtest = np.array([x, x])

    data = params.load_data(method=method)
    signal, noise = compute_signal_and_noise_grid(
        xtest,
        ts,
        params.n,
        params.scale,
        params.fwhm,
        data,
        params.upsampling_factor,
        r_mask=None,
        method=method,
    )
    assert_allclose(signal[0], expected[method][0], atol=1e-6, rtol=0)
    assert_allclose(noise[0], expected[method][1], atol=1e-6, rtol=0)


def test_compute_snr(params_with_images):
    params = params_with_images
    ts = params.get_ts()
    data = params.load_data(method="aperture")
    args = (x, ts, params.n, params.scale, params.fwhm, data)

    snr = compute_snr(*args, exclude_source=False, exclude_lobes=False)
    assert_allclose(snr, expected["aperture"][2], atol=1e-3, rtol=0)

    snr = compute_snr(*args, exclude_source=True, exclude_lobes=True)
    assert_allclose(snr, expected["aperture_exc"][2], atol=1e-3, rtol=0)

    snr = compute_snr(
        *args, exclude_source=False, exclude_lobes=False, use_interp_bgnoise=True
    )
    assert_allclose(snr, expected["aperture_interp"][2], atol=1e-3, rtol=0)

    snr = compute_snr(
        *args, invvar_weighted=True, exclude_source=False, exclude_lobes=False
    )
    assert_allclose(snr, expected_invvar["aperture"][2], atol=1e-3, rtol=0)

    snr = compute_snr(
        *args, invvar_weighted=True, exclude_source=True, exclude_lobes=True
    )
    assert_allclose(snr, expected_invvar["aperture_exc"][2], atol=1e-3, rtol=0)


def test_compute_snr_cython(params_with_images):
    params = params_with_images
    ts = params.get_ts()
    orbital_grid = np.atleast_2d(x[:4])
    positions = orbit.positions_at_multiple_times(ts, orbital_grid)
    # (2, Nimages, Norbits) -> (Norbits, Nimages, 2)
    positions = np.ascontiguousarray(np.transpose(positions))
    projection_grid = np.atleast_2d(x[4:])
    proj_matrices = orbit.compute_projection_matrices(*projection_grid.T)

    data = params.load_data(method="convolve")
    out = np.zeros((1, 3))
    cy_compute_snr(
        data["images"],
        positions[0],
        data["bkg"],
        data["noise"],
        proj_matrices,
        params.r_mask,
        params.r_mask_ext,
        params.scale,
        params.n,
        params.upsampling_factor,
        out,
        1,
        0,
    )
    assert_allclose(out[0][:2], expected["convolve"][:2], atol=1e-6, rtol=0)
    assert_allclose(out[0][2], expected["convolve"][2], atol=1e-3, rtol=0)

    # With inverse variance weight
    out = np.zeros((1, 3))
    cy_compute_snr(
        data["images"],
        positions[0],
        data["bkg"],
        data["noise"],
        proj_matrices,
        params.r_mask,
        params.r_mask_ext,
        params.scale,
        params.n,
        params.upsampling_factor,
        out,
        1,
        1,
    )
    assert_allclose(out[0][:2], expected_invvar["convolve"][:2], atol=1e-6, rtol=0)
    assert_allclose(out[0][2], expected_invvar["convolve"][2], atol=1e-3, rtol=0)
    # res = compute_snr_detailed(params, x, method="convolve", verbose=True)
    # assert_allclose(out[0, 0], res.meta["signal_invvar"], atol=1e-6, rtol=0)
    # assert_allclose(out[0, 1], res.meta["noise_invvar"], atol=1e-6, rtol=0)
