import matplotlib.pyplot as plt  # noqa
import numpy as np
import pytest
from astropy.io import fits

from kstacker.noise_profile import pre_process_image


def test_pre_process_image(tmp_path):
    with pytest.raises(ValueError, match="Image must be a .fits file!"):
        pre_process_image("foo.h5", 1)

    testfile = tmp_path / "img.fits"
    img = np.ones((10, 11))
    img[0, 0] = 1
    fits.writeto(testfile, img)

    with pytest.raises(ValueError, match="Internal mask diameter"):
        pre_process_image(testfile, 1, r_mask=-1, r_mask_ext=10)
    with pytest.raises(ValueError, match="External mask diameter"):
        pre_process_image(testfile, 1, r_mask=0, r_mask_ext=20)

    pre_process_image(
        testfile, 1, r_mask=1, r_mask_ext=5, upsampling_factor=3, plot=True
    )

    data = fits.getdata(tmp_path / "img_preprocessed.fits")
    assert data.shape == (10, 10)
    assert data[0, 0] == 0
    assert data[5, 5] == 0

    data = fits.getdata(tmp_path / "img_resampled.fits")
    assert data.shape == (30, 30)

    assert (tmp_path / "img_preprocessed.png").is_file()
    assert (tmp_path / "img_resampled.png").is_file()
