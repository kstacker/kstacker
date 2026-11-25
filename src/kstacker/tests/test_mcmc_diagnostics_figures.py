"""Convergence figures (iteration 09): strided chains, cold-chain fallback, short chains."""
import contextlib
import io
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from kstacker import mcmc_diagnostics as D
from kstacker.tests.test_reddemcee_strided_storage import _run

FILES = ("parameter_traces", "sampling_diagnostics", "autocorrelation_functions",
         "temperature_diagnostics")


class _StoredPT:
    """reddemcee diagnostics as read back by `kstacker plot` without `pt_chain`."""

    def __init__(self, nsweeps, ntemps):
        rng = np.random.default_rng(1)
        self.betas = np.geomspace(1.0, 1e-3, ntemps)
        self._tsw = rng.uniform(0.5, 0.8, (nsweeps, ntemps - 1))

    def get_chain(self, flat=False):
        raise AttributeError("The full temperature chain was not stored")

    def get_tsw(self):
        return self._tsw

    def get_smd(self):
        return self._tsw * 2.0


class _StoredSampler:
    def __init__(self, rows=3, nwalkers=8, stride=10):
        rng = np.random.default_rng(2)
        self.chain = rng.normal(3.0, 0.1, (rows, nwalkers, 7))
        self.logp = rng.normal(80.0, 1.0, (rows, nwalkers))
        self.acceptance_fraction = np.full(nwalkers, 0.07)
        self.pt_sampler = _StoredPT(nsweeps=12, ntemps=4)
        self.burnin_sweeps = 4
        self.lines_per_sweep = 4
        self.chain_stride = stride
        self.pt_chain_burnin_rows = 2

    def get_chain(self, discard=0, thin=1, flat=False):
        return self.chain

    def get_log_prob(self, discard=0, thin=1, flat=False):
        return self.logp


class DiagnosticsFigures(unittest.TestCase):
    def _check(self, sampler):
        with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with contextlib.redirect_stdout(io.StringIO()):
                D.create_mcmc_diagnostics(sampler, output_dir=tmp, dpi=40)
            for name in FILES:
                self.assertTrue((Path(tmp) / f"{name}.pdf").exists(), name)

    def test_live_strided_reddemcee_short_chain(self):
        # 9 production sweeps x 2 lines, 1 line in 3 kept: 6 stored rows.
        sampler, _ = _run(thin=3, keep_all=False)
        self.assertEqual(sampler.chain_stride, 3)
        self._check(sampler)

    def test_stored_without_tempered_chains_uses_cold_chain(self):
        self._check(_StoredSampler())

    def test_nanpercentile_columns_all_nan_column(self):
        values = np.array([[1.0, np.nan], [3.0, np.nan]])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            q = D._nanpercentile_columns(values, [50])
        self.assertEqual(q[0, 0], 2.0)
        self.assertTrue(np.isnan(q[0, 1]))

    def test_step_axis_label(self):
        self.assertIn("Production step", D._step_axis_label(1, None))
        label = D._step_axis_label(10, 4)
        self.assertIn("1 sweep = 4 lines", label)
        self.assertIn("1 stored row every 10", label)


class AutocorrelationMaxLag(unittest.TestCase):
    """Iteration 10: ACF lag range ~ 3 tau_max, capped at the chain length."""

    @staticmethod
    def _ar1(rows, nwalkers, ndim, rho, seed=3):
        rng = np.random.default_rng(seed)
        x = np.zeros((rows, nwalkers, ndim))
        noise = rng.normal(size=(rows, nwalkers, ndim))
        for i in range(1, rows):
            x[i] = rho * x[i - 1] + noise[i]
        return x

    def test_three_tau_max(self):
        # AR(1): tau = (1 + rho) / (1 - rho) = 19 for rho = 0.9.
        chain = self._ar1(4000, 16, 2, 0.9)
        chain[:, :, 1] = chain[:, :, 1] * 0.5 + np.random.default_rng(4).normal(size=(4000, 16))
        import emcee
        tau = emcee.autocorr.integrated_time(chain, tol=0, quiet=True)
        max_lag, tau_max = D._default_acf_max_lag(chain)
        self.assertAlmostEqual(tau_max, float(np.max(tau)))
        self.assertEqual(max_lag, int(np.ceil(3 * tau_max)))
        self.assertTrue(10 < tau_max < 30)

    def test_capped_at_chain_length(self):
        # emcee's windowed estimator keeps tau below ~ rows / 5, so the cap is
        # exercised through the 20-row minimum on a 10-row chain.
        chain = self._ar1(10, 8, 1, 0.9)
        max_lag, _ = D._default_acf_max_lag(chain)
        self.assertEqual(max_lag, 9)

    def test_constant_chain_falls_back(self):
        chain = np.ones((50, 4, 2))
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            max_lag, tau_max = D._default_acf_max_lag(chain)
        self.assertTrue(np.isnan(tau_max))
        self.assertEqual(max_lag, 20)


class ExplicitPlotCommand(unittest.TestCase):
    """Iteration 10: `kstacker plot` ignores plots.enabled: false, with a message."""

    def test_disabled_is_overridden_with_message(self):
        from kstacker import scientific_plots as S
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertTrue(S._explicit_plot_overrides_disabled({"enabled": False}))
        self.assertIn("plots.enabled is false", out.getvalue())
        self.assertIn("ignored", out.getvalue())

    def test_enabled_is_silent(self):
        from kstacker import scientific_plots as S
        for settings in ({"enabled": True}, {}, None):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertTrue(S._explicit_plot_overrides_disabled(settings))
            self.assertEqual(out.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
