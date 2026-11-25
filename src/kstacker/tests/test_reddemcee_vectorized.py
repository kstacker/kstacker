"""reddemcee: batched prior/likelihood (iteration 07) equals the scalar path."""
import contextlib
import io
import unittest
from unittest import mock

import numpy as np

from kstacker import mcmc as M

CFG = dict(ntemps=4, beta_min=0.05, betas=None, inner_steps=2, adapt_mode="SAR",
           adapt_tau=50.0, adapt_nu=0.3, stretch_a=1.5, smd_history=True, tsw_history=True)


def _logl_scalar(x):
    return float(0.5 * np.sum((x - 0.3) ** 2 * np.arange(1, 8)) ** 0.5)


def _logl_batch(samples, **_):
    return 0.5 * np.sum((samples - 0.3) ** 2 * np.arange(1, 8), axis=1) ** 0.5


def _logp_scalar(x):
    return 0.0 if np.all(np.abs(x) < 1.0) else -np.inf


def _logp_batch(samples, _):
    return np.where(np.all(np.abs(samples) < 1.0, axis=1), 0.0, -np.inf)


def _run(vectorize, seed=42, nwalkers=16, burnin=4, production=6):
    np.random.seed(seed)
    p0 = np.random.uniform(-0.5, 0.5, (nwalkers, 7))
    cfg = dict(CFG, vectorize=vectorize)
    kwargs = dict(instruments=[], noise_floor=1.0, weighting="invvar",
                  likelihood_mode="signed_snr", fp_bounds=(1e-4, 1e2))
    with mock.patch.object(M, "_prior_only_scalar", side_effect=lambda t, _: _logp_scalar(t)), \
         mock.patch.object(M, "_likelihood_only_scalar", side_effect=lambda t, _: _logl_scalar(t)), \
         mock.patch.object(M, "_prior_only_batch", side_effect=_logp_batch), \
         mock.patch.object(M, "total_direct_imaging_log_likelihood", side_effect=_logl_batch), \
         mock.patch.object(M, "log_probability",
                           side_effect=lambda flat, **_: _logp_batch(flat, None) + _logl_batch(flat)), \
         contextlib.redirect_stdout(io.StringIO()):
        return M.run_reddemcee_adaptive(p0, kwargs, nsteps=production, burnin=burnin,
                                        cfg=cfg, progress=False)


class VectorizedLogProb(unittest.TestCase):
    def test_pairs_match_ptwrapper_convention(self):
        coords = np.array([[0.1] * 7, [2.0] * 7, [-0.2] * 7])
        betas = np.array([1.0, 0.5, 0.25])
        with mock.patch.object(M, "_prior_only_batch", side_effect=_logp_batch), \
             mock.patch.object(M, "total_direct_imaging_log_likelihood", side_effect=_logl_batch):
            out = M._VectorizedPTLogProb(dict(instruments=[], noise_floor=1.0, weighting="invvar",
                                              likelihood_mode="signed_snr",
                                              fp_bounds=(1e-4, 1e2)))(zip(coords, betas))
        self.assertEqual(len(out), 3)
        for (lp, ll), x, b in zip(out, coords, betas):
            if _logp_scalar(x) == -np.inf:
                self.assertEqual((lp, ll), (-np.inf, -np.inf))
            else:
                self.assertEqual(ll, _logl_scalar(x))
                self.assertEqual(lp, _logp_scalar(x) + _logl_scalar(x) * b)

    def test_real_prior_batch_equals_scalar(self):
        rng = np.random.default_rng(3)
        kw = dict(instruments=[], a_bounds=(2.0, 3.5), m0_bounds=(0.8, 1.3), e_max=0.95,
                  orbit_direction="retrograde")
        theta = np.column_stack([rng.uniform(1.9, 3.6, 500), rng.uniform(-0.1, 6.4, 500),
                                 rng.uniform(0.7, 1.4, 500), rng.uniform(-1, 1, (500, 4))])
        batch = M._prior_only_batch(theta, kw)
        scalar = np.array([M._prior_only_scalar(t, kw) for t in theta])
        np.testing.assert_array_equal(batch, scalar)
        self.assertTrue(np.any(np.isfinite(batch)) and np.any(~np.isfinite(batch)))

    def test_vectorized_run_identical_to_scalar_run(self):
        try:
            import reddemcee  # noqa: F401
        except ImportError:
            self.skipTest("reddemcee not installed")
        a = _run(False)
        b = _run(True)
        np.testing.assert_array_equal(a.pt_sampler.get_chain(flat=False),
                                      b.pt_sampler.get_chain(flat=False))
        np.testing.assert_array_equal(a.pt_sampler.get_log_like(flat=False),
                                      b.pt_sampler.get_log_like(flat=False))
        np.testing.assert_array_equal(np.asarray(a.pt_sampler.betas), np.asarray(b.pt_sampler.betas))
        np.testing.assert_array_equal(a.acceptance_fraction, b.acceptance_fraction)


if __name__ == "__main__":
    unittest.main()
