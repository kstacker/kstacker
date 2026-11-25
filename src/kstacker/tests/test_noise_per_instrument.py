"""Per-instrument radial noise recalibration: settings, cache separation, checks.

Run with: python -m unittest discover -s src/kstacker/tests -t src
"""
from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from ..direct_imaging import (
    _resolve_radial_noise_recalibration,
    _recalibrate_classical_noise,
    _radial_recalibration_cache_dir,
    _local_aperture_ring_cache_stem,
    _cache_epoch_paths,
    _instrument_profile_dir,
)
from ..noise_plots import create_noise_figures


GLOBAL = {"radial_noise_recalibration": {
    "mode": "monotonic_inward", "bin_width_px": 2.0, "max_distance_px": "auto",
    "min_samples_per_bin": 30, "smoothing_bins": 3, "min_factor": 1.0,
    "monotonic_extent_px": 4.0, "cache": True, "overwrite_cache": False,
}}


def _synthetic(size: int, n_epochs: int, seed: int):
    """Images, local background and noise maps for one fake instrument."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    radius = np.hypot(xx - (size - 1) / 2.0, yy - (size - 1) / 2.0)
    sigma = (1.0 + 0.02 * radius)[None].repeat(n_epochs, axis=0)
    images = (rng.standard_normal((n_epochs, size, size)) * sigma * 1.7).astype(np.float32)
    return images, np.zeros_like(images), sigma.astype(np.float32)


def _run(profile_dir, name, size, n_epochs, r_mask, r_mask_ext, seed, fwhm=2.0):
    cfg = _resolve_radial_noise_recalibration(
        GLOBAL, {"name": name, "r_mask": r_mask, "r_mask_ext": r_mask_ext, "n": size})
    images, bkg, noise = _synthetic(size, n_epochs, seed)
    with contextlib.redirect_stdout(io.StringIO()) as out:
        result = _recalibrate_classical_noise(
            photometry_method="convolve", images_up=images, images_native=images,
            size=size, upsampling_factor=1.0, fwhm=fwhm, r_mask=r_mask,
            xgrid=np.zeros(2), bkg=np.zeros((n_epochs, 2)), noise=np.ones((n_epochs, 2)),
            local_bkg_maps=bkg, local_noise_maps=noise, cfg=cfg,
            profile_dir=profile_dir, instrument_name=name)
    cache_dir = _radial_recalibration_cache_dir(profile_dir, name, cfg, "convolve")
    return np.asarray(result[3]), cache_dir, out.getvalue()


class MaxDistanceTests(unittest.TestCase):

    def test_auto_value_per_instrument(self):
        visir = _resolve_radial_noise_recalibration(
            GLOBAL, {"name": "VISIR_2019", "r_mask": 25., "r_mask_ext": 45., "n": 100})
        miri = _resolve_radial_noise_recalibration(
            GLOBAL, {"name": "MIRI_2024-2025", "r_mask": 5., "r_mask_ext": 29., "n": 73})
        self.assertEqual(visir["max_distance_px"], 20.0)
        self.assertEqual(miri["max_distance_px"], 24.0)
        self.assertTrue(visir["max_distance_origin"].startswith("computed"))

    def test_instrument_override_and_global_value(self):
        inst = {"name": "A", "r_mask": 5., "r_mask_ext": 29., "n": 73,
                "radial_noise_recalibration": {"max_distance_px": 12}}
        cfg = _resolve_radial_noise_recalibration(GLOBAL, inst)
        self.assertEqual(cfg["max_distance_px"], 12.0)
        self.assertEqual(cfg["max_distance_origin"], "fixed: instrument section")
        root = {"radial_noise_recalibration": {**GLOBAL["radial_noise_recalibration"],
                                               "max_distance_px": 30}}
        cfg = _resolve_radial_noise_recalibration(root, {"r_mask": 5., "r_mask_ext": 29.})
        self.assertEqual(cfg["max_distance_px"], 30.0)
        self.assertTrue(cfg["max_distance_origin"].startswith("fixed: global"))


class CacheSeparationTests(unittest.TestCase):

    def test_instruments_have_distinct_caches_identical_to_single_runs(self):
        with tempfile.TemporaryDirectory() as shared, \
                tempfile.TemporaryDirectory() as alone_a, \
                tempfile.TemporaryDirectory() as alone_b:
            a_joint, dir_a, _ = _run(shared, "VISIR_2019", 60, 3, 10., 28., seed=1)
            b_joint, dir_b, log_b = _run(shared, "MIRI_2024-2025", 41, 2, 4., 19., seed=2)
            self.assertNotEqual(dir_a, dir_b)
            self.assertIn("VISIR_2019", dir_a)
            self.assertIn("MIRI_2024-2025", dir_b)
            self.assertIn("MISS", log_b)
            self.assertEqual(a_joint.shape, (3, 60, 60))
            self.assertEqual(b_joint.shape, (2, 41, 41))
            with np.load(os.path.join(dir_a, "g_profile.npz")) as ga, \
                    np.load(os.path.join(dir_b, "g_profile.npz")) as gb:
                self.assertEqual(float(ga["inner_radius_px"]), 10.0)
                self.assertEqual(float(gb["inner_radius_px"]), 4.0)
                self.assertEqual(float(ga["max_distance_px"]), 18.0)
                self.assertEqual(float(gb["max_distance_px"]), 15.0)
            a_alone, _, _ = _run(alone_a, "VISIR_2019", 60, 3, 10., 28., seed=1)
            b_alone, _, _ = _run(alone_b, "MIRI_2024-2025", 41, 2, 4., 19., seed=2)
            np.testing.assert_array_equal(a_joint, a_alone)
            np.testing.assert_array_equal(b_joint, b_alone)
            # Second call: cache hit, same values.
            b_again, _, log = _run(shared, "MIRI_2024-2025", 41, 2, 4., 19., seed=2)
            self.assertIn("HIT", log)
            np.testing.assert_array_equal(b_again, b_joint)

    def test_inconsistent_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as shared, tempfile.TemporaryDirectory() as fresh:
            # Same name, same settings, but another image size and other data:
            # reproduces the original bug (cache of another instrument).
            _run(shared, "X", 60, 3, 4., 19., seed=1)
            values, _, log = _run(shared, "X", 41, 3, 4., 19., seed=2)
            self.assertIn("rejected", log)
            self.assertEqual(values.shape, (3, 41, 41))
            expected, _, _ = _run(fresh, "X", 41, 3, 4., 19., seed=2)
            np.testing.assert_array_equal(values, expected)

    def test_tampered_map_shape_is_rejected(self):
        with tempfile.TemporaryDirectory() as shared:
            first, cache_dir, _ = _run(shared, "Y", 41, 2, 4., 19., seed=3)
            np.save(os.path.join(cache_dir, "noise_recalibrated_epoch_1.npy"), np.ones((7, 7)))
            again, _, log = _run(shared, "Y", 41, 2, 4., 19., seed=3)
            self.assertIn("rejected", log)
            np.testing.assert_array_equal(again, first)


class NoiseFigureTests(unittest.TestCase):

    def test_pdfs_and_g_profile_parts(self):
        with tempfile.TemporaryDirectory() as profile_dir, tempfile.TemporaryDirectory() as fig_dir:
            n_epochs, size = 10, 41
            _, recal_dir, _ = _run(profile_dir, "INST", size, n_epochs, 4., 19., seed=4)
            ring_dir = _instrument_profile_dir(profile_dir, "INST")
            stem = _local_aperture_ring_cache_stem(
                photometry_method="convolve", size=size, upsampling_factor=1.0,
                fwhm=2.0, cfg={})
            _, bkg, noise = _synthetic(size, n_epochs, 4)
            noise[:, 0, 0] = 0.0   # one invalid pixel
            for k in range(n_epochs):
                bg_path, noise_path = _cache_epoch_paths(ring_dir, stem, k)
                np.save(bg_path, bkg[k])
                np.save(noise_path, noise[k])
            inst = SimpleNamespace(
                name="INST", ts=np.arange(n_epochs, dtype=float), photometry_method="convolve",
                r_mask=4., r_mask_ext=19.,
                noise_products=dict(local_ring_dir=ring_dir, local_ring_stem=stem,
                                    recalibration_dir=recal_dir))
            with contextlib.redirect_stdout(io.StringIO()):
                written = create_noise_figures([inst], fig_dir)
            names = sorted(os.path.basename(p) for p in written)
            self.assertEqual(names, sorted([
                "INST_local_noise.pdf", "INST_local_background.pdf",
                "INST_recalibrated_noise.pdf", "INST_g_profile.pdf",
                "INST_g_profile_part2.pdf"]))
            for path in written:
                self.assertTrue(os.path.getsize(path) > 0)


if __name__ == "__main__":
    unittest.main()
