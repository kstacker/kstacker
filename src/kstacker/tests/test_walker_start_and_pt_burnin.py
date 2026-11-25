"""Walker starting points stay inside the prior; reddemcee burn-in removal."""
import unittest
from unittest import mock

import numpy as np

from kstacker import mcmc as M

BOUNDS = dict(a_bounds=(1.0, 5.0), m0_bounds=(0.5, 1.5), la0_bounds=(0.0, 2.0 * np.pi))
SPREAD = dict(a=0.02, la0=0.2, m0=0.02, hk=0.02, pq=0.02)


def _draw(theta, nwalkers=4000, e_max=0.95, direction="any", seed=1):
    np.random.seed(seed)
    return M.draw_walkers_around_theta_init(
        nwalkers, np.asarray(theta, dtype=float), e_max=e_max, spread=SPREAD,
        orbit_direction=direction, max_tries=100 * nwalkers, **BOUNDS,
    )


def _log_prior(p0, e_max=0.95, direction="any"):
    return M.log_prior_hkpq(p0, e_max=e_max, orbit_direction=direction, **BOUNDS)


class WalkerStartInsidePrior(unittest.TestCase):
    def test_inclination_limit(self):
        # Centre 2 of iteration 04 (r = 0.985), then a centre exactly on r = 1.
        for p, q in ((-0.951251, -0.254887), (1.0, 0.0), (0.6, 0.8)):
            p0 = _draw([3.0, 1.0, 1.0, 0.1, 0.1, p, q])
            r2 = p0[:, 5] * p0[:, 5] + p0[:, 6] * p0[:, 6]
            self.assertTrue(np.all(r2 <= 1.0))
            self.assertTrue(np.all(np.isfinite(_log_prior(p0))))
            # No projection: nobody sits on the limit, and p, q differ across walkers.
            self.assertEqual(int(np.sum(np.abs(r2 - 1.0) < 1.0e-12)), 0)

    def test_eccentricity_limit(self):
        for e_max in (0.95, 0.3):
            h = e_max * 0.6
            k = e_max * 0.8  # e = e_max exactly, up to rounding.
            p0 = _draw([3.0, 1.0, 1.0, h, k, 0.2, 0.2], e_max=e_max)
            e = np.hypot(p0[:, 3], p0[:, 4])
            self.assertTrue(np.all(e <= e_max))
            self.assertTrue(np.all(np.isfinite(_log_prior(p0, e_max=e_max))))
            self.assertEqual(int(np.sum(np.abs(e - e_max) < 1.0e-12)), 0)

    def test_both_limits_and_direction(self):
        p0 = _draw([3.0, 1.0, 1.0, 0.57, 0.76, 0.8, 0.6], direction="retrograde")
        self.assertTrue(np.all(np.isfinite(_log_prior(p0, direction="retrograde"))))
        self.assertTrue(np.all(p0[:, 5] ** 2 + p0[:, 6] ** 2 >= 0.5))

    def test_all_temperatures(self):
        p0 = _draw([3.0, 1.0, 1.0, 0.1, 0.1, -0.951251, -0.254887], nwalkers=128)
        for shift in range(12):  # p0_pt of run_reddemcee_adaptive
            self.assertTrue(np.all(np.isfinite(_log_prior(np.roll(p0, shift, axis=0)))))

    def test_distribution_is_truncated_gaussian(self):
        # Rejection keeps the Gaussian: compare with an explicit truncation.
        centre = np.array([3.0, 1.0, 1.0, 0.1, 0.1, 0.99, 0.0])
        p0 = _draw(centre, nwalkers=20000)
        np.random.seed(7)
        reference = centre[5:] + 0.02 * np.random.randn(400000, 2)
        reference = reference[np.sum(reference ** 2, axis=1) <= 1.0]
        self.assertLess(abs(p0[:, 5].mean() - reference[:, 0].mean()), 5.0e-4)
        self.assertLess(abs(p0[:, 5].std() - reference[:, 0].std()), 5.0e-4)

    def test_interior_centre_draws_unchanged(self):
        # Far from the limits nothing is rejected: same draws as the Gaussian jitter.
        theta = np.array([3.0, 1.0, 1.0, 0.1, 0.1, 0.3, 0.3])
        p0 = _draw(theta, nwalkers=50, seed=11)
        np.random.seed(11)
        z = np.random.randn(50, 7)
        s = np.array([0.02, 0.2, 0.02, 0.02, 0.02, 0.02, 0.02])
        expected = theta + s * z
        expected[:, [0, 2]] = theta[[0, 2]] * (1.0 + s[[0, 2]] * z[:, [0, 2]])
        np.testing.assert_allclose(p0, expected)

    def test_clip_centre_inside_prior(self):
        rng = np.random.default_rng(3)
        for angle in rng.uniform(0.0, 2.0 * np.pi, 2000):
            theta = np.array([9.0, 1.0, 9.0, 2.0 * np.cos(angle), 2.0 * np.sin(angle),
                              3.0 * np.sin(angle), 3.0 * np.cos(angle)])
            clipped = M._clip_theta_init_to_bounds(theta, e_max=0.95, **BOUNDS)
            self.assertLessEqual(np.hypot(clipped[3], clipped[4]), 0.95)
            self.assertLessEqual(clipped[5] * clipped[5] + clipped[6] * clipped[6], 1.0)
            self.assertTrue(np.isfinite(_log_prior(clipped)))


class _FakePT:
    """Chain whose value is the stored-line index, as (ntemps, lines, walkers, dim)."""

    def __init__(self, lines, ntemps=3, nwalkers=4, ndim=7):
        self.chain = np.ones((ntemps, lines, nwalkers, ndim)) * np.arange(lines)[None, :, None, None]

    def get_chain(self, flat=False):
        return self.chain


class ReddemceeBurnIn(unittest.TestCase):
    def _adapter(self, fake, burnin=5, production=7, inner=3, accepted=None):
        accepted = np.full((3, 4), 6.0) if accepted is None else accepted
        logp = lambda flat, **_: np.sum(flat, axis=1)
        with mock.patch.object(M, "log_probability", side_effect=logp):
            return M._ReddenColdSamplerAdapter(
                fake, burnin_sweeps=burnin, production_sweeps=production,
                inner_steps=inner, production_accepted=accepted, lp_kwargs={},
            )

    def test_removes_inner_steps_lines_per_sweep(self):
        sampler = self._adapter(_FakePT(36))
        chain = sampler.get_chain()
        self.assertEqual(chain.shape, (21, 4, 7))
        self.assertEqual(chain[0, 0, 0], 15.0)  # 5 sweeps x 3 lines removed
        self.assertEqual(sampler.get_log_prob().shape, (21, 4))
        self.assertEqual(sampler.get_chain(thin=10, flat=True).shape, (3 * 4, 7))
        np.testing.assert_allclose(sampler.acceptance_fraction, 6.0 / 21.0)

    def test_unexpected_storage_uses_measured_lines(self):
        with mock.patch("builtins.print") as printed:
            sampler = self._adapter(_FakePT(12))  # one line per sweep
        self.assertEqual(sampler.lines_per_sweep, 1)
        self.assertEqual(sampler.get_chain().shape[0], 7)
        self.assertTrue(printed.called)

    def test_tau_in_stored_lines(self):
        sampler = self._adapter(_FakePT(36))
        chain = np.random.default_rng(0).normal(size=(2000, 4, 7))
        sampler._cold_chain = chain
        tau_1 = sampler.get_autocorr_time(thin=1)
        with mock.patch.object(M.emcee.autocorr, "integrated_time", return_value=np.ones(7)):
            tau_10 = sampler.get_autocorr_time(thin=10)
        self.assertEqual(tau_1.shape, (7,))
        np.testing.assert_allclose(tau_10, 10.0)

    def test_real_reddemcee_run(self):
        try:
            import reddemcee  # noqa: F401
        except ImportError:
            self.skipTest("reddemcee not installed")
        np.random.seed(5)
        burnin, production, inner, nwalkers = 5, 7, 3, 16
        p0 = np.random.normal(0.0, 0.5, (nwalkers, 7))

        def logl(x):
            return -0.5 * float(np.sum(x * x))

        def logp(x):
            return 0.0 if np.all(np.abs(x) < 50.0) else -np.inf

        def log_probability(flat, **_):
            return -0.5 * np.sum(flat * flat, axis=1)

        cfg = dict(ntemps=3, beta_min=0.1, betas=None, inner_steps=inner, adapt_mode="SAR",
                   adapt_tau=50.0, adapt_nu=0.3, stretch_a=1.5, smd_history=True, tsw_history=True,
                   vectorize=False)  # scalar path: the mocks below are the scalar functions
        with mock.patch.object(M, "_prior_only_scalar", side_effect=lambda t, _: logp(t)), \
             mock.patch.object(M, "_likelihood_only_scalar", side_effect=lambda t, _: logl(t)), \
             mock.patch.object(M, "log_probability", side_effect=log_probability):
            sampler = M.run_reddemcee_adaptive(p0, {}, nsteps=production, burnin=burnin,
                                               cfg=cfg, progress=False)
        raw = np.asarray(sampler.pt_sampler.get_chain(flat=False))
        self.assertEqual(raw.shape, (3, (burnin + production) * inner, nwalkers, 7))
        chain = sampler.get_chain()
        self.assertEqual(chain.shape, (production * inner, nwalkers, 7))
        np.testing.assert_array_equal(chain, raw[0, burnin * inner:])
        # Stored cold log-probabilities (beta = 1) match the recomputed ones.
        stored = np.asarray(sampler.pt_sampler.get_log_prob(flat=False))[0, burnin * inner:]
        np.testing.assert_allclose(sampler.get_log_prob(), stored)
        # Production acceptance counted on production lines only.
        self.assertEqual(sampler.acceptance_fraction.shape, (nwalkers,))
        self.assertTrue(np.all((sampler.acceptance_fraction >= 0.0) & (sampler.acceptance_fraction <= 1.0)))
        total = np.asarray(sampler.pt_sampler.backend.accepted)[0]
        self.assertTrue(np.all(sampler.acceptance_fraction * chain.shape[0] <= total + 1.0e-9))
        self.assertEqual(np.asarray(sampler.pt_sampler.get_tsw(discard=burnin)).shape, (production, 2))


if __name__ == "__main__":
    unittest.main()
