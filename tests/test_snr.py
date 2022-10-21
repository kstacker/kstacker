import numpy as np
import pytest
from numpy.testing import assert_allclose

from kstacker._utils import cy_compute_snr
from kstacker.gradient_reoptimization import compute_snr
from kstacker.noise_profile import compute_noise_profiles, compute_signal_and_noise_grid
from kstacker.orbit import orbit
from kstacker.utils import compute_snr_detailed

# orbit parameters: a, e, t0, m, omega, i theta_0
x = [53.75, 0.08, -98.2575, 1.59, -1.9438095, 0.5233333, -2.8409524]
# expected values with plain summation
expected = {
    # method  :  signal,       noise,          snr
    "convolve": [0.0004428637, 2.64439168e-05, 16.74728012],
    "aperture": [0.00044921199, 2.64439177e-05, 16.987346605],
}
# expected values with inverse variance weighting
expected_invvar = {
    "convolve": [0.000110405, 6.542620672e-06, 16.8747297135],
    "aperture": [0.000112108, 6.542620672e-06, 17.1349995529],
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
    assert_allclose(res.meta["signal_sum"], expected[method][0], rtol=1e-6)
    assert_allclose(res.meta["noise_sum"], expected[method][1], rtol=1e-6)
    assert_allclose(res.meta["snr_sum"], expected[method][2], rtol=1e-6)


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
    assert_allclose(signal[0], expected[method][0], rtol=1e-6)
    assert_allclose(noise[0], expected[method][1], rtol=1e-6)


def test_compute_snr_grad(params_with_images):
    params = params_with_images
    ts = params.get_ts()

    data = params.load_data(method="aperture")
    snr = compute_snr(
        x,
        ts,
        params.n,
        params.scale,
        params.fwhm,
        data,
        r_mask=0,
        invvar_weighted=False,
    )
    assert_allclose(-snr, expected["aperture"][2], rtol=1e-6)

    snr = compute_snr(
        x,
        ts,
        params.n,
        params.scale,
        params.fwhm,
        data,
        r_mask=0,
        invvar_weighted=True,
    )
    assert_allclose(-snr, expected_invvar["aperture"][2], rtol=1e-6)


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
    out[:, 2] *= -1  # positive SNR
    assert_allclose(out[0], expected["convolve"], rtol=1e-6)

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
    out[:, 2] *= -1  # positive SNR
    assert_allclose(out[0], expected_invvar["convolve"], rtol=1e-6)
    # res = compute_snr_detailed(params, x, method="convolve", verbose=True)
    # assert_allclose(out[0, 0], res.meta["signal_invvar"], rtol=1e-6)
    # assert_allclose(out[0, 1], res.meta["noise_invvar"], rtol=1e-6)
