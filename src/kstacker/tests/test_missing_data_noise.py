"""No-data pixels and the strict sigma = 0 rule (iteration 02, 2026-11).

Run with: python -m unittest discover -s src/kstacker/tests -t src
"""
from __future__ import annotations

import contextlib
import io
import unittest

import numpy as np
from scipy.signal import convolve2d

from ..direct_imaging import (
    _bilinear_support_valid,
    _compute_local_ring_maps_for_epoch,
    _local_aperture_ring_cache_stem,
    _local_aperture_ring_stats,
    _measurement_missing_grid,
    _resolve_background_noise,
    snr_from_hkpq,
)

CFG = dict(aperture_spacing_fwhm=2.0, exclusion_radius_fwhm=4.0, radial_samples=3,
           radial_step_fwhm=2.0, min_reference_apertures=1, sigma_statistic="mad",
           small_sample_correction="none", missing_data="zero")


def _upsample(image, factor, radius):
    """Same construction as _build_resampled_image (without FITS I/O)."""
    replicated = np.kron(image, np.ones((factor, factor))) / factor ** 2
    mask_size = (int(radius + 0.5) * 2 + 1) * factor
    xx, yy = np.mgrid[:mask_size, :mask_size] - mask_size // 2
    kernel = (np.hypot(xx, yy) < radius * factor).astype(float)
    return convolve2d(replicated, kernel, mode="same")


class MissingGridTests(unittest.TestCase):

    def test_missing_grid_matches_convolution_support(self):
        rng = np.random.default_rng(1)
        native = rng.normal(1.0, 0.1, (21, 21))
        native[:, 15:] = 0.0          # no-data band
        native[5, 5] = 0.0            # isolated bad pixel
        grid = _measurement_missing_grid(photometry_method="convolve", image_native=native,
                                         upsampling_factor=3, fwhm=1.6)
        indicator = _upsample((native == 0).astype(float), 3, 1.6)
        np.testing.assert_array_equal(grid, indicator > 1e-9)
        # Far from the band and from the bad pixel the measurement is clean.
        self.assertFalse(grid[10 * 3 + 1, 10 * 3 + 1])
        self.assertTrue(grid[10 * 3 + 1, 16 * 3 + 1])


class RingStatsTests(unittest.TestCase):

    def test_zero_padded_references_are_excluded(self):
        """With half of the ring in a no-data zone, sigma must stay close to the truth."""
        rng = np.random.default_rng(3)
        size, factor, fwhm, sigma_true = 81, 1, 2.0, 1.0
        native = rng.normal(0.0, sigma_true, (size, size))
        native[:, 50:] = 0.0
        missing = _measurement_missing_grid(photometry_method="convolve", image_native=native,
                                            upsampling_factor=factor, fwhm=fwhm)
        kwargs = dict(photometry_method="convolve", x_target=20.0, y_target=40.0,
                      r_target=20.0, size=size, upsampling_factor=factor, fwhm=fwhm,
                      image_up=native, image_native=None, cfg=CFG)
        _, s_old, n_old, ok_old = _local_aperture_ring_stats(**kwargs)
        _, s_new, n_new, ok_new = _local_aperture_ring_stats(**kwargs, missing_grid=missing)
        self.assertTrue(ok_new)
        self.assertLess(n_new, n_old)
        self.assertGreater(s_new, 0.6 * sigma_true)
        self.assertLess(s_new, 1.6 * sigma_true)
        self.assertLess(s_old, s_new)       # the zeros shrink the old MAD

    def test_failed_or_no_data_pixels_are_zero(self):
        rng = np.random.default_rng(4)
        size = 41
        native = rng.normal(0.0, 1.0, (size, size))
        native[:, 30:] = 0.0
        missing = _measurement_missing_grid(photometry_method="convolve", image_native=native,
                                            upsampling_factor=1, fwhm=2.0)
        with contextlib.redirect_stderr(io.StringIO()):
            bg, sig = _compute_local_ring_maps_for_epoch(
                epoch_index=0, photometry_method="convolve", size=size, upsampling_factor=1,
                fwhm=2.0, image_up=native, image_native=native, cfg=CFG, noise_floor=0.5,
                missing_grid=missing)
        # No-data columns and their kernel neighbourhood are invalid (sigma = 0,
        # never the build floor 0.5), real data far from them is valid.
        self.assertTrue(np.all(sig[:, 29:] == 0.0))
        self.assertTrue(np.all(bg[:, 29:] == 0.0))
        self.assertGreater(np.median(sig[5:35, 5:20]), 0.5)

    def test_cache_stem_changes_with_rule(self):
        stem = _local_aperture_ring_cache_stem(photometry_method="convolve", size=100,
                                               upsampling_factor=5, fwhm=2.5, cfg=CFG)
        self.assertIn("nd2=zero", stem)
        self.assertLess(len(stem) + len("__epoch=000__background.npy"), 250)
        other = _local_aperture_ring_cache_stem(photometry_method="convolve", size=100,
                                                upsampling_factor=5, fwhm=2.5,
                                                cfg=dict(CFG, missing_data="none"))
        self.assertNotEqual(stem, other)

    def test_config_key(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(_resolve_background_noise({})["missing_data"], "zero")
            self.assertEqual(_resolve_background_noise(
                {"background_noise": {"missing_data": "none"}})["missing_data"], "none")
            with self.assertRaises(ValueError):
                _resolve_background_noise({"background_noise": {"missing_data": "nan"}})


class StrictBilinearTests(unittest.TestCase):

    def setUp(self):
        self.maps = np.ones((1, 10, 10), dtype=float)
        self.maps[0, 5, 5] = 0.0   # one invalid pixel (y=5, x=5)

    def _valid(self, x, y):
        x = np.array([[x]], dtype=float)
        y = np.array([[y]], dtype=float)
        return bool(_bilinear_support_valid(self.maps, x, y, np.zeros((1, 1), dtype=int))[0, 0])

    def test_neighbour_of_invalid_pixel_is_rejected(self):
        self.assertFalse(self._valid(4.99, 4.99))   # weight 0.98 on (5,5)
        self.assertFalse(self._valid(4.2, 4.3))     # small but non-zero weight
        self.assertFalse(self._valid(5.0, 5.0))     # exactly on it

    def test_zero_weight_neighbour_does_not_reject(self):
        self.assertTrue(self._valid(4.0, 4.0))      # (5,5) has weight 0
        self.assertTrue(self._valid(2.5, 2.5))
        self.assertFalse(self._valid(-0.5, 2.0))    # outside the map


class LikelihoodExclusionTests(unittest.TestCase):
    """An epoch whose sample touches sigma = 0 is excluded, not interpolated."""

    def test_epoch_excluded_and_snr_normalised_on_valid_epochs(self):
        size, K = 41, 3
        images = np.zeros((K, size, size))
        noise = np.ones((K, size, size))
        bkg = np.zeros((K, size, size))
        # A face-on circular orbit observed at the same phase: position fixed.
        theta = np.array([[10.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
        from ..orbital_geometry import _equinoctial_sky_tracks
        north, east = _equinoctial_sky_tracks(*[np.array([v]) for v in theta[0]],
                                              np.zeros(K), t_ref=0.0)
        scale = 1.0
        x = -east[0, 0] * scale + 20.0
        y = north[0, 0] * scale + 20.0
        images[:] = 0.0
        images[:, int(np.floor(y)), int(np.floor(x))] = 2.0
        noise[:] = 1.0
        noise[2, int(np.floor(y)), int(np.floor(x))] = 0.0
        out = snr_from_hkpq(theta, ts=np.zeros(K), images=images, data={}, size=size,
                            scale=scale, upsampling_factor=1.0, noise_floor=1e-3,
                            weighting="invvar", photometry_method="convolve",
                            background_noise_cfg={"local_map_mode": "precompute_cache"},
                            local_bkg_maps=bkg, local_noise_maps=noise,
                            return_epoch_valid=True)
        signal, sig_comb, snr, epoch_snr, epoch_ok = out
        self.assertEqual(epoch_ok[0].tolist(), [True, True, False])
        # Combined S/N over the 2 valid epochs: sum(z)/sqrt(2) with z = flux/sigma.
        z = epoch_snr[0, :2]
        self.assertAlmostEqual(float(snr[0]), float(np.sum(z) / np.sqrt(2.0)), places=5)
        self.assertEqual(float(epoch_snr[0, 2]), 0.0)


if __name__ == "__main__":
    unittest.main()
