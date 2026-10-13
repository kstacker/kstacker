"""Synthetic PACO-map tests of multi-instrument posterior aggregation.

Run with: python -m unittest keplerian_inference.tests.test_likelihood_assembly
"""
from __future__ import annotations
import unittest
import numpy as np

from ..direct_imaging import (
    Instrument, instrument_log_likelihood, total_direct_imaging_log_likelihood,
)
from ..mcmc import log_probability, log_prior_hkpq, _likelihood_only_scalar
from ..mcmc_init_search import _keep_top_rows


def instrument(name: str, mode: str) -> Instrument:
    """Return two constant PACO epochs with known per-epoch S/N values."""
    side = 31
    signal = np.stack([np.full((side, side), 2.), np.full((side, side), -1.)])
    return Instrument(
        name=name, size=side, scale=1., upsampling_factor=1., fwhm=2.,
        r_mask=None, r_mask_ext=None, t_ref=0., ts=np.array([0., 1.]),
        photometry_method="paco", images_up=None, images_native=None,
        paco_alpha_maps=signal, paco_var_alpha_maps=np.ones_like(signal),
        xgrid=np.zeros(1), bkg=np.zeros((2, 1)), noise=np.ones((2, 1)),
        likelihood_mode=mode,
    )


class IndependentLikelihoodTests(unittest.TestCase):
    """Check fixed numerical criteria and posterior/prior decomposition."""

    def setUp(self):
        self.orbit = np.array([9., 0.7, 1.59, 0.08, 0.15, 0.10, 0.11])
        self.positive = instrument("PACO", "positive_snr_profile")
        self.signed = instrument("IMAGER", "signed_snr")

    def test_individual_criteria(self):
        positive = instrument_log_likelihood(self.orbit, self.positive,
                                              noise_floor=0.01, weighting="invvar")
        signed = instrument_log_likelihood(self.orbit, self.signed,
                                            noise_floor=0.01, weighting="invvar")
        # Positive profile: 0.5 * (max(2,0)^2 + max(-1,0)^2).
        self.assertAlmostEqual(positive, 2.)
        # Signed common-flux profile: 0.5 * ((2-1)/sqrt(2))^2.
        self.assertAlmostEqual(signed, 0.25)

    def test_flux_criterion(self):
        flux_inst = instrument("FLUX", "flux")
        value = instrument_log_likelihood(np.r_[self.orbit, 1.], flux_inst,
                                           noise_floor=0.01, weighting="invvar")
        # S1=2, S2=1, fp=1 gives -0.5*(2-2)=0.
        self.assertAlmostEqual(value, 0.)

    def test_total_and_posterior(self):
        insts = [self.positive, self.signed]
        total = total_direct_imaging_log_likelihood(
            self.orbit, instruments=insts, noise_floor=0.01, weighting="invvar")
        self.assertAlmostEqual(total, 2.25)
        prior = log_prior_hkpq(self.orbit, a_bounds=(8.,30.),
                               m0_bounds=(1.,2.), e_max=.95)
        posterior = log_probability(self.orbit, instruments=insts,
                                    a_bounds=(8.,30.), m0_bounds=(1.,2.),
                                    e_max=.95, noise_floor=.01)
        self.assertAlmostEqual(posterior, prior+total)
        args = dict(instruments=insts, a_bounds=(8.,30.), m0_bounds=(1.,2.),
                    noise_floor=.01, e_max=.95, orbit_direction="any",
                    weighting="invvar", likelihood_mode="signed_snr",
                    fp_bounds=(1e-4,1e2), fp_prior="uniform")
        self.assertAlmostEqual(_likelihood_only_scalar(self.orbit,args), total)

    def test_grid_rank_uses_signed_log_likelihood(self):
        data = np.array([[1, -10.], [2, -1.], [3, -3.]])
        best = _keep_top_rows(None, data, top_n=2)
        self.assertEqual(best[:,0].tolist(), [2., 3.])


if __name__ == "__main__":
    unittest.main()
