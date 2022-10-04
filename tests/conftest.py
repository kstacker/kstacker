import pathlib
import shutil

import pytest

from kstacker.utils import Params

CURRENT_PATH = pathlib.Path(__file__).absolute().parent
EXAMPLE_PATH = CURRENT_PATH / ".." / "example"
EXAMPLE_PARAMS = EXAMPLE_PATH / "parameters_HD95086.yml"


@pytest.fixture()
def params():
    return Params.read(EXAMPLE_PARAMS)


@pytest.fixture()
def params_small(params):
    params._params["Na"] = 2
    params._params["Ne"] = 2
    params._params["Nt0"] = 2
    params._params["Nomega"] = 2
    params._params["Ni"] = 2
    params._params["Ntheta_0"] = 2
    return params


@pytest.fixture(scope="module")
def params_tmp(tmp_path_factory):
    p = Params.read(EXAMPLE_PARAMS)
    work_path = tmp_path_factory.mktemp("data")
    p.work_dir = str(work_path)
    shutil.copytree(EXAMPLE_PATH / "images", work_path / "images")
    return p
