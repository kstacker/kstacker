# K-Stacker plot guide

## Observation maps

Each epoch uses its own symmetric colour scale `[-A, A]`, with `A = max(abs(min), abs(max))` computed from the finite displayed pixels. When `plots.mask_to_working_region` is enabled, the display is limited to the working annulus used by the science setup. The MAP position is shown with a large open circle so the local image structure remains visible.

## Aligned observation stacks

The aligned-stack figure is built independently for each instrument. Every epoch is shifted so that the MAP-predicted planet location matches the planet location at the first epoch of that same instrument. Only the final aligned mean is shown. The stack is meant for visual inspection; it is not an additional statistic used by the likelihood.

## Projected orbits

The projected-orbit panel shows a single highlighted MAP orbit and a larger set of pale posterior orbit draws. Open circles indicate the MAP positions at the observation times. These circles are markers of the sampled MAP trajectory, not uncertainty regions.

## Corner plots

The diagonal panels show one-dimensional marginals with the median and the 16th/84th percentiles. The off-diagonal panels show binned sample occupancy with logarithmic colour scaling. The number of plotted samples is limited by `plots.max_posterior_samples` for readability.

## Sampler diagnostics

The diagnostic plots show every saved production step and every walker. They are intended to make poor mixing, stuck walkers and multimodality easier to spot. They are useful, but they are not a convergence proof on their own.
