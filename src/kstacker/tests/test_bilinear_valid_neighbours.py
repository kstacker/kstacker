"""Rule (a) of iteration 07: measurement validity + interpolation on valid neighbours."""
import unittest
from types import SimpleNamespace

import numpy as np

from kstacker.direct_imaging import (
    _bilinear_interp_valid_neighbours,
    _bilinear_support_valid,
    _build_measurement_missing_maps,
    _interp_map_bilinear,
    _measurement_missing_at,
    _measurement_missing_grid,
    _measurement_touches_missing,
    snr_from_hkpq,
)
from kstacker.orbital_geometry import _equinoctial_sky_tracks


def _one(fn, maps_b, maps_s, x, y):
    xa = np.array([[x]], dtype=float)
    ya = np.array([[y]], dtype=float)
    k = np.zeros((1, 1), dtype=int)
    ok, b, s = fn(maps_b, maps_s, xa, ya, k)
    return bool(ok[0, 0]), float(b[0, 0]), float(s[0, 0])


class ValidNeighbourInterpolation(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.sig = rng.uniform(1.0, 2.0, (1, 10, 10)).astype(np.float32)
        self.bkg = rng.normal(0.0, 1.0, (1, 10, 10)).astype(np.float32)

    def _interp(self, x, y):
        return _one(_bilinear_interp_valid_neighbours, self.bkg, self.sig, x, y)

    def test_four_valid_neighbours_bit_identical(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(0, 9, (50, 3))
        y = rng.uniform(0, 9, (50, 3))
        k = np.zeros_like(x, dtype=int)
        ok, b, s = _bilinear_interp_valid_neighbours(self.bkg, self.sig, x, y, k)
        self.assertTrue(np.all(ok))
        np.testing.assert_array_equal(s, _interp_map_bilinear(self.sig, x, y, k))
        np.testing.assert_array_equal(b, _interp_map_bilinear(self.bkg, x, y, k))

    def test_one_invalid_neighbour_is_dropped_and_weights_renormalised(self):
        self.sig[0, 5, 5] = 0.0
        self.bkg[0, 5, 5] = 0.0
        x, y = 4.3, 4.6           # (5,5) has weight 0.3 * 0.6 = 0.18
        ok, b, s = self._interp(x, y)
        self.assertTrue(ok)
        self.assertFalse(bool(_bilinear_support_valid(
            self.sig, np.array([[x]]), np.array([[y]]), np.zeros((1, 1), int))[0, 0]))
        w = {(4, 4): 0.7 * 0.4, (4, 5): 0.3 * 0.4, (5, 4): 0.7 * 0.6}
        tot = sum(w.values())
        exp_s = sum(wt * float(self.sig[0, yy, xx]) for (yy, xx), wt in w.items()) / tot
        exp_b = sum(wt * float(self.bkg[0, yy, xx]) for (yy, xx), wt in w.items()) / tot
        self.assertAlmostEqual(s, exp_s, places=5)
        self.assertAlmostEqual(b, exp_b, places=5)

    def test_three_invalid_neighbours_tiny_weight_left(self):
        for yy, xx in ((4, 4), (4, 5), (5, 4)):
            self.sig[0, yy, xx] = 0.0
        ok, b, s = self._interp(4.0001, 4.0001)   # (5,5) weight 1e-8
        self.assertTrue(ok)
        self.assertAlmostEqual(s, float(self.sig[0, 5, 5]), places=5)
        self.assertAlmostEqual(b, float(self.bkg[0, 5, 5]), places=5)

    def test_all_weighted_neighbours_invalid(self):
        for yy, xx in ((4, 4), (4, 5), (5, 4), (5, 5)):
            self.sig[0, yy, xx] = 0.0
        ok, b, s = self._interp(4.5, 4.5)
        self.assertEqual((ok, b, s), (False, 0.0, 0.0))
        # On a pixel: only that pixel has a weight.
        ok, _, _ = self._interp(4.0, 4.0)
        self.assertFalse(ok)

    def test_zero_weight_invalid_neighbour_and_outside(self):
        self.sig[0, 5, 5] = 0.0
        ok, b, s = self._interp(4.0, 4.0)
        self.assertTrue(ok)
        self.assertEqual(s, float(self.sig[0, 4, 4]))
        ok, _, _ = self._interp(-0.5, 2.0)
        self.assertFalse(ok)
        ok, _, _ = self._interp(9.2, 2.0)
        self.assertFalse(ok)

    def test_non_finite_sigma_is_invalid(self):
        self.sig[0, 5, 5] = np.nan
        ok, _, s = self._interp(4.5, 4.5)
        self.assertTrue(ok)
        self.assertTrue(np.isfinite(s))


class MeasurementMissing(unittest.TestCase):
    def test_grid_lookup_matches_scalar_helper(self):
        img = np.ones((20, 20))
        img[:, 14:] = 0.0                         # no data for x >= 14
        grid = _measurement_missing_grid(photometry_method="convolve", image_native=img,
                                         upsampling_factor=5, fwhm=2.0)
        rng = np.random.default_rng(2)
        x = rng.uniform(0, 19, (200, 1))
        y = rng.uniform(0, 19, (200, 1))
        k = np.zeros_like(x, dtype=int)
        got = _measurement_missing_at(grid[None], photometry_method="convolve", x_pix=x, y_pix=y,
                                      k_idx=k, upsampling_factor=5, fwhm=2.0,
                                      candidates=np.ones_like(x, dtype=bool))
        exp = [_measurement_touches_missing(grid, photometry_method="convolve", x_native=float(a),
                                            y_native=float(b), upsampling_factor=5, r_ap=2.0)
               for a, b in zip(x[:, 0], y[:, 0])]
        self.assertEqual(got[:, 0].tolist(), exp)
        self.assertTrue(any(exp) and not all(exp))
        # Non-candidates are never flagged.
        none = _measurement_missing_at(grid[None], photometry_method="convolve", x_pix=x, y_pix=y,
                                       k_idx=k, upsampling_factor=5, fwhm=2.0,
                                       candidates=np.zeros_like(x, dtype=bool))
        self.assertFalse(np.any(none))

    def test_aperture_mode(self):
        img = np.ones((20, 20))
        img[:, 14:] = 0.0
        miss = _measurement_missing_grid(photometry_method="aperture", image_native=img,
                                         upsampling_factor=1, fwhm=2.0)
        x = np.array([[5.0], [12.5]])
        y = np.array([[10.0], [10.0]])
        got = _measurement_missing_at(miss[None], photometry_method="aperture", x_pix=x, y_pix=y,
                                      k_idx=np.zeros((2, 1), int), upsampling_factor=1, fwhm=2.0,
                                      candidates=np.ones((2, 1), bool))
        self.assertEqual(got[:, 0].tolist(), [False, True])

    def test_build_maps_modes(self):
        img = np.ones((2, 12, 12))
        img[:, :, 9:] = 0.0
        inst = SimpleNamespace(photometry_method="convolve", images_native=img,
                               images_up=np.zeros((2, 60, 60)), local_noise_maps=np.ones((2, 12, 12)),
                               bgnoise_cfg={"missing_data": "zero"}, fwhm=2.0, upsampling_factor=5,
                               name="T")
        maps = _build_measurement_missing_maps(inst)
        self.assertEqual(maps.shape, (2, 60, 60))
        self.assertTrue(maps.dtype == bool and maps.any() and not maps.all())
        inst.bgnoise_cfg = {"missing_data": "none"}
        self.assertFalse(_build_measurement_missing_maps(inst).any())
        inst.photometry_method = "paco"
        self.assertIsNone(_build_measurement_missing_maps(inst))


class LikelihoodRuleA(unittest.TestCase):
    """End-to-end on snr_from_hkpq: fixed position, 3 epochs."""

    def _position(self):
        theta = np.array([[10.0, 0.7, 1.0, 0.0, 0.0, 0.0, 0.0]])
        north, east = _equinoctial_sky_tracks(*[np.array([v]) for v in theta[0]],
                                              np.zeros(3), t_ref=0.0)
        return theta, -east[0, 0] + 20.0, north[0, 0] + 20.0

    def _run(self, noise, missing):
        size, K = 41, 3
        theta, x, y = self._position()
        images = np.zeros((K, size, size))
        images[:, int(np.floor(y)), int(np.floor(x))] = 2.0
        out = snr_from_hkpq(theta, ts=np.zeros(K), images=images, data={}, size=size, scale=1.0,
                            upsampling_factor=1.0, noise_floor=1e-3, weighting="invvar",
                            photometry_method="convolve",
                            background_noise_cfg={"local_map_mode": "precompute_cache"},
                            local_bkg_maps=np.zeros((K, size, size)), local_noise_maps=noise,
                            return_epoch_valid=True, measurement_missing_maps=missing)
        return out, x, y

    def test_invalid_neighbour_no_longer_rejects_epoch(self):
        size, K = 41, 3
        _, x, y = self._position()
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        self.assertTrue(x - x0 > 0 and y - y0 > 0)   # all 4 neighbours weighted
        noise = np.ones((K, size, size))
        noise[2, y0 + 1, x0 + 1] = 0.0                # weighted neighbour invalid at epoch 3
        missing = np.zeros((K, size, size), dtype=bool)
        (_, _, snr, es, ok), _, _ = self._run(noise, missing)
        self.assertEqual(ok[0].tolist(), [True, True, True])
        # Legacy behaviour without the maps: epoch 3 rejected.
        (_, _, _, _, ok_legacy), _, _ = self._run(noise, None)
        self.assertEqual(ok_legacy[0].tolist(), [True, True, False])
        # sigma = 1 on the valid neighbours: same S/N on every epoch.
        self.assertAlmostEqual(float(es[0, 2]), float(es[0, 0]), places=5)

    def test_measurement_on_missing_data_rejects_epoch(self):
        size, K = 41, 3
        _, x, y = self._position()
        noise = np.ones((K, size, size))
        missing = np.zeros((K, size, size), dtype=bool)
        missing[1, int(np.floor(y + 0.5)), int(np.floor(x + 0.5))] = True
        # upsampling 1: the read index is floor(x); flag that one too.
        missing[1, int(np.floor(y)), int(np.floor(x))] = True
        (_, _, snr, es, ok), _, _ = self._run(noise, missing)
        self.assertEqual(ok[0].tolist(), [True, False, True])
        self.assertEqual(float(es[0, 1]), 0.0)
        z = es[0, [0, 2]]
        self.assertAlmostEqual(float(snr[0]), float(np.sum(z) / np.sqrt(2.0)), places=5)


if __name__ == "__main__":
    unittest.main()
