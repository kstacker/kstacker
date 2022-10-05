import pytest
from numpy.testing import assert_allclose


def test_params(params_tmp):
    params = params_tmp
    assert params.m0 == 1.59
    assert params["m0"] == 1.59

    assert_allclose(params.wav, 0.8e-6)
    assert_allclose(params.fwhm, 1.688723, rtol=1e-6)
    assert_allclose(params.scale, 0.947015, rtol=1e-6)
    assert_allclose(params.get_ts(), [1.32238193, 2.2642026, 2.92402464, 4.1889117])

    with pytest.raises(KeyError):
        params["foo"]
    with pytest.raises(AttributeError):
        params.foo


def test_paths(params_tmp):
    assert params_tmp.get_path("images_dir").endswith("/images")
    assert params_tmp.get_image_suffix("convolve") == "_resampled"
    assert params_tmp.get_image_suffix("aperture") == "_preprocessed"
    with pytest.raises(ValueError):
        params_tmp.get_image_suffix("other")


def test_grid(params_small):
    grid = params_small.grid

    assert (
        repr(grid)
        == """\
Grid(
    a: 40.0 → 62.0, 2 steps
    e: 0.0 → 0.4, 2 steps
    t0: -393.0 → -0.01, 2 steps
    omega: -3.14 → 3.14, 2 steps
    i: 0.0 → 3.14, 2 steps
    theta_0: -3.14 → 3.14, 2 steps
)
64 orbits"""
    )

    assert grid.limits("a") == (40.0, 62.0, 2)
    assert grid.range("a") == slice(40.0, 62.0, 11.0)
    assert grid.bounds() == [
        (40.0, 62.0),
        (0.0, 0.4),
        (-393.0, -0.01),
        (-3.14, 3.14),
        (0.0, 3.14),
        (-3.14, 3.14),
    ]

    orbital_grid = grid.make_2d_grid(("a", "e", "t0"))
    assert orbital_grid.shape == (8, 3)
    # projection_grid = grid.make_2d_grid(("omega", "i", "theta_0"))


def test_grid_fixed_param(params_small):
    grid = params_small.grid
    orbital_grid = grid.make_2d_grid(("a", "e", "t0"))
    assert orbital_grid.shape[0] == 8  # 2x2x2

    params_small._params['Ne'] = 1
    grid = params_small.grid
    orbital_grid = grid.make_2d_grid(("a", "e", "t0"))
    assert orbital_grid.shape[0] == 4  # 1x2x2

    params_small._params['Na'] = 1
    params_small._params['Ne'] = 1
    params_small._params['Nt0'] = 1
    grid = params_small.grid
    orbital_grid = grid.make_2d_grid(("a", "e", "t0"))
    assert orbital_grid.shape == (1, 3)  # 1x1x1
