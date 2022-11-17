import matplotlib.pyplot as plt  # noqa
import numpy as np
import pytest
from astropy.io import fits

from kstacker.imagerie import (
    compute_noise_profile_apertures,
    monte_carlo_profiles,
    monte_carlo_profiles_remove_planet,
)
from kstacker.noise_profile import pre_process_image


@pytest.fixture(scope="module")
def params_with_images(params_tmp):
    # Pre-compute convolved images
    np.random.seed(42)
    images_dir = params_tmp.get_path("images_dir")
    pre_process_image(
        f"{images_dir}/image_0.fits",
        params_tmp.fwhm,
        size=params_tmp.n,
        upsampling_factor=params_tmp.upsampling_factor,
    )
    return params_tmp


def test_pre_process_image(tmp_path):
    with pytest.raises(ValueError, match="Image must be a .fits file!"):
        pre_process_image("foo.h5", 1)

    testfile = tmp_path / "img.fits"
    img = np.ones((10, 11))
    img[0, 0] = 1
    fits.writeto(testfile, img)

    pre_process_image(testfile, 1, upsampling_factor=3, plot=True)

    data = fits.getdata(tmp_path / "img_preprocessed.fits")
    assert data.shape == (10, 10)

    data = fits.getdata(tmp_path / "img_resampled.fits")
    assert data.shape == (30, 30)

    assert (tmp_path / "img_preprocessed.png").is_file()
    assert (tmp_path / "img_resampled.png").is_file()


def test_profiles_montecarlo(params_with_images):
    params = params_with_images
    images_dir = params.get_path("images_dir")

    img = fits.getdata(f"{images_dir}/image_0_preprocessed.fits")
    bg_aper, noise_aper = monte_carlo_profiles(
        img,
        params.n,
        params.fwhm,
        params.upsampling_factor,
        method="aperture",
    )

    img = fits.getdata(f"{images_dir}/image_0_resampled.fits").astype(float)
    bg_conv, noise_conv = monte_carlo_profiles(
        img,
        params.n,
        params.fwhm,
        params.upsampling_factor,
        method="convolve",
    )

    np.isclose(bg_aper[40:].mean(), -2e-6, atol=1e-7, rtol=0)
    np.isclose(bg_conv[40:].mean(), -2e-6, atol=1e-7, rtol=0)
    np.isclose(noise_aper[30], 2.65e-5, atol=1e-6, rtol=0)
    np.isclose(noise_conv[30], 2.65e-5, atol=1e-6, rtol=0)


# def test_profiles_remove_planet(params_with_images):
#     params = params_with_images
#     images_dir = params.get_path("images_dir")
#     img = fits.getdata(f"{images_dir}/image_0_preprocessed.fits")
#     bg, noise = monte_carlo_profiles_remove_planet(
#         img,
#         params.n,
#         planet_coord=[20, 35],
#         remove_box=[10, 10, 10, 10],
#         fwhm=params.fwhm,
#         upsampling_factor=params.upsampling_factor,
#         method="aperture",
#     )

#     np.isclose(bg[40:].mean(), -2e-6, atol=3e-7, rtol=0)
#     np.isclose(noise[30], 1e-5, atol=1e-6, rtol=0)


def test_profiles_apertures(params_with_images):
    params = params_with_images
    images_dir = params.get_path("images_dir")
    img = fits.getdata(f"{images_dir}/image_0.fits")
    bg, noise, naper = compute_noise_profile_apertures(img, params.fwhm)
    np.isclose(bg[40:].mean(), -2e-6, atol=1e-7, rtol=0)
    np.isclose(noise[30], 1.95e-5, atol=1e-6, rtol=0)


def test_profiles_apertures_mask(params_with_images):
    params = params_with_images
    images_dir = params.get_path("images_dir")
    img = fits.getdata(f"{images_dir}/image_0.fits")
    bg, noise, naper = compute_noise_profile_apertures(
        img, params.fwhm, mask_apertures=[(20, 35, 10)]
    )
    np.isclose(bg[40:].mean(), -2e-6, atol=3e-7, rtol=0)
    np.isclose(noise[30], 8e-6, atol=1e-6, rtol=0)
