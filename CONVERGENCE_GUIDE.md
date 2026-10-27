# Convergence guide

This guide describes a practical workflow for assessing posterior convergence in K-Stacker. The figures created by `kstacker plot` are intended to support that assessment, but no single plot should be treated as a proof of convergence.

## 1. Start with multiple independent starts

A single well-behaved chain is not enough when the posterior may be multi-modal. In practice, it is safer to:

- keep `multistart.enabled: true`,
- launch several independent runs from different initial seeds or top-ranked grid-search states,
- compare the resulting marginal posteriors and MAP solutions.

Agreement between independent starts is often more informative than any one numerical summary.

## 2. Inspect the trace plots

The file `figures/convergence/parameter_traces.*` shows every saved production step and every walker. A healthy trace plot usually shows:

- no long-lived walker groups stuck in clearly separated bands,
- a stable central tendency over time,
- repeated overlap between walkers rather than isolated trajectories.

Slow drift, separated plateaus, or walkers that never mix with the rest of the ensemble are warning signs.

## 3. Inspect the log-posterior history

The file `figures/convergence/sampling_diagnostics.*` shows all walker histories in log posterior. This plot is useful for identifying:

- walkers trapped in poor regions,
- sudden transitions between modes,
- unequal exploration among walkers,
- suspiciously small within-walker variation.

A narrow cloud is not automatically good: it can reflect genuine convergence, but it can also mean that the ensemble failed to explore beyond one local mode.

## 4. Inspect the autocorrelation functions

The file `figures/convergence/autocorrelation_functions.*` summarizes the walker-wise ACF for each parameter. For each lag, the central line is the median across walkers and the shaded band spans the 16th to 84th percentiles.

This plot is conceptually similar to the ACF summaries often shown in radial-velocity MCMC analyses. Use it to check whether correlation decays on a reasonable lag scale. Very slow decay indicates highly correlated samples and therefore a lower effective sample size.

In practice, you want:

- a visible decay toward zero,
- broadly similar behaviour across walkers,
- no parameter whose ACF remains extremely high over most of the saved window.

## 5. Check acceptance and temperature behaviour

For `emcee`, the acceptance histogram gives a quick sanity check on proposal efficiency. For adaptive parallel tempering, `temperature_diagnostics.*` helps assess whether hot chains move and whether temperature swaps behave sensibly.

The exact acceptable range depends on the problem, so these should be treated as diagnostics, not hard pass/fail thresholds.

## 6. Compare science outputs, not only diagnostics

Convergence should also be reflected in the scientific figures. In particular:

- the posterior sky-density map should not depend strongly on a small subset of samples,
- the corner plots should not change qualitatively when the second half of the production chain is used instead of the full production chain,
- independent runs should give compatible orbital families.

## 7. Practical recommendations

A useful workflow is:

1. run the deterministic initial search,
2. enable multistart,
3. run a shorter exploratory MCMC,
4. inspect traces, log posterior, and ACFs,
5. lengthen the production run only after the exploratory run looks healthy,
6. compare at least two independent full runs.

## 8. Limits of the diagnostics

These diagnostics are designed to be informative and easy to read, but they are not a replacement for scientific judgement. Multi-modal posteriors, strong degeneracies, and optimistic noise models can all produce misleadingly clean-looking plots.

For that reason, convergence assessment should always combine:

- sampler diagnostics,
- multiple independent runs,
- sensitivity checks to priors and noise settings,
- direct inspection of the resulting science figures.
