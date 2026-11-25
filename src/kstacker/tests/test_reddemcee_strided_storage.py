"""reddemcee keeps one line in `thin` in memory (iteration 08): sampling unchanged."""
import contextlib
import io
import unittest
from unittest import mock

import numpy as np

from kstacker import mcmc as M

CFG = dict(ntemps=4, beta_min=0.05, betas=None, inner_steps=2, adapt_mode="SAR",
           adapt_tau=50.0, adapt_nu=0.3, stretch_a=1.5, smd_history=True, tsw_history=True,
           vectorize=True)


def _logl_batch(samples, **_):
    return -0.5 * np.sum((samples - 0.3) ** 2 * np.arange(1, 8), axis=1)


def _logp_batch(samples, _):
    return np.where(np.all(np.abs(samples) < 1.0, axis=1), 0.0, -np.inf)


def _run(thin, keep_all, seed=7, nwalkers=16, burnin=4, production=9, **cfg):
    np.random.seed(seed)
    p0 = np.random.uniform(-0.5, 0.5, (nwalkers, 7))
    kwargs = dict(instruments=[], noise_floor=1.0, weighting="invvar",
                  likelihood_mode="signed_snr", fp_bounds=(1e-4, 1e2))
    with mock.patch.object(M, "_prior_only_batch", side_effect=_logp_batch), \
         mock.patch.object(M, "total_direct_imaging_log_likelihood", side_effect=_logl_batch), \
         mock.patch.object(M, "log_probability",
                           side_effect=lambda flat, **_: _logp_batch(flat, None) + _logl_batch(flat)), \
         contextlib.redirect_stdout(io.StringIO()):
        sampler = M.run_reddemcee_adaptive(
            p0, kwargs, nsteps=production, burnin=burnin,
            cfg=dict(CFG, keep_all_lines=keep_all, **cfg), progress=False, thin=thin)
        map_state = sampler.map_state()
    return sampler, map_state


def _has_reddemcee():
    try:
        import reddemcee  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(_has_reddemcee(), "reddemcee not installed")
class StridedStorage(unittest.TestCase):
    def test_strided_run_equals_full_run(self):
        for thin in (2, 3, 5):
            full, map_full = _run(thin, keep_all=True)
            part, map_part = _run(thin, keep_all=False)
            self.assertEqual(full.chain_stride, 1)
            self.assertEqual(part.chain_stride, thin)
            first = (4 * 2) % thin  # burn-in lines = 4 sweeps x 2 inner steps
            for name in ("chain", "log_like", "log_prob"):
                a = np.asarray(full.pt_sampler.backend.get_value(name))
                b = np.asarray(part.pt_sampler.backend.get_value(name))
                np.testing.assert_array_equal(a[:, first::thin], b)
            for getter in ("get_tsw", "get_smd", "get_betas"):
                np.testing.assert_array_equal(getattr(full.pt_sampler, getter)(),
                                              getattr(part.pt_sampler, getter)())
            np.testing.assert_array_equal(full.pt_sampler.backend.accepted,
                                          part.pt_sampler.backend.accepted)
            np.testing.assert_array_equal(full.get_chain(thin=thin), part.get_chain(thin=thin))
            np.testing.assert_array_equal(full.get_chain(thin=thin), part.get_chain())
            np.testing.assert_array_equal(full.get_log_prob(thin=thin, flat=True),
                                          part.get_log_prob(thin=thin, flat=True))
            np.testing.assert_array_equal(full.acceptance_fraction, part.acceptance_fraction)
            self.assertEqual(full.production_lines, part.production_lines)
            np.testing.assert_array_equal(map_full[0], map_part[0])
            self.assertEqual(map_full[1], map_part[1])
            self.assertEqual(part.pt_chain_burnin_rows, (4 * 2) // thin)

    def test_thin_must_be_a_multiple_of_the_stride(self):
        part, _ = _run(3, keep_all=False)
        with self.assertRaises(ValueError):
            part.get_chain(thin=2)
        self.assertEqual(part.get_chain(thin=6).shape[0], part.get_chain().shape[0] // 2
                         + part.get_chain().shape[0] % 2)

    def test_history_dependent_adaptation_keeps_every_line(self):
        sampler, _ = _run(3, keep_all=False, adapt_mode="SGG")
        self.assertEqual(sampler.chain_stride, 1)


@unittest.skipUnless(_has_reddemcee(), "reddemcee not installed")
class StridedBackendUnit(unittest.TestCase):
    """Final line overwritten after the last swap: kept and tracked as in reddemcee."""

    def _state(self, value, lp):
        from reddemcee.state import State
        return State(np.full((2, 7), float(value)), log_prob=np.array(lp, float),
                     log_like=np.array(lp, float), beta=1.0)

    def test_final_line_and_map_tracking(self):
        from reddemcee.state import PTState
        backend = M._strided_pt_backend_classes()(stride=2, anchor=2, track_map=True)
        backend.reset(1, 2, 7, smd_hist=False, tsw_hist=True)
        backend.grow(3)                 # 3 sweeps
        backend[0].grow(6, None)        # 2 lines per sweep
        lps = [[50.0, 0.0], [0.0, 0.0], [1.0, 2.0], [3.0, 0.0], [0.0, 0.0], [9.0, 0.0]]
        for sweep in range(3):
            for k in range(2):
                line = 2 * sweep + k
                backend[0].save_step(self._state(line, lps[line]), np.ones(2, bool))
            post = PTState(np.full((1, 2, 7), 100.0 + sweep))
            post[0].log_prob = np.array([5.0, 4.0])
            post[0].log_like = np.array([5.0, 4.0])
            backend.save_step(post, np.zeros(0))
        cold = backend[0]
        self.assertEqual(cold.stored, 3)              # lines 0 (burn-in), 2 and 4
        self.assertEqual(cold.rows_before(2), 1)      # one burn-in row
        for row, line in enumerate((0, 2, 4)):
            np.testing.assert_array_equal(cold.chain[row], np.full((2, 7), float(line)))
        # Line 0 (burn-in, logP 50) is ignored; line 5's inner-step state
        # (logP 9) is overwritten by the post-swap state (logP 5): MAP = line 5 post-swap.
        self.assertEqual(len(cold.map_candidates), 1)
        line, walker, state, lp = cold.map_candidates[0]
        self.assertEqual((line, walker, lp), (5, 0, 5.0))
        np.testing.assert_array_equal(state, np.full(7, 102.0))

    def test_final_line_kept_is_overwritten(self):
        from reddemcee.state import PTState
        backend = M._strided_pt_backend_classes()(stride=3, anchor=0, track_map=False)
        backend.reset(1, 2, 7, smd_hist=False, tsw_hist=True)
        backend.grow(2)
        backend[0].grow(4, None)        # lines 0..3, kept 0 and 3
        for sweep in range(2):
            for k in range(2):
                backend[0].save_step(self._state(2 * sweep + k, [0.0, 0.0]), np.ones(2, bool))
            post = PTState(np.full((1, 2, 7), 100.0 + sweep))
            post[0].log_prob = np.zeros(2)
            post[0].log_like = np.zeros(2)
            backend.save_step(post, np.zeros(0))
        np.testing.assert_array_equal(backend[0].chain[-1], np.full((2, 7), 101.0))
        np.testing.assert_array_equal(backend[0].chain[0], np.full((2, 7), 0.0))


if __name__ == "__main__":
    unittest.main()
