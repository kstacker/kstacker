import pathlib
from kstacker.utils import Params
from numpy.testing import assert_allclose

CURRENT_PATH = pathlib.Path(__file__).absolute().parent
EXAMPLE_PARAMS = CURRENT_PATH / ".." / "example" / "parameters_HD95086.yml"


def test_params():
    params = Params.read(EXAMPLE_PARAMS)

    assert params.m0 == 1.59
    assert params["m0"] == 1.59

    assert_allclose(params.wav, 0.8e-6)
    assert_allclose(params.fwhm, 1.688723, rtol=1e-6)
    assert_allclose(params.scale, 0.947015, rtol=1e-6)

    assert params.get_path("images_dir") == "./images"
    assert_allclose(params.get_ts(), [1.32238193, 2.2642026, 2.92402464, 4.1889117])
