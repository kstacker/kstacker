|kstacker| Documentation
========================

.. important:: Before using this software, please read the
   :ref:`acknowledgements` and :ref:`citation` at the end of this documentation.

How to install
--------------

|kstacker| requires Python>=3.7 and a C compiler (you can use e.g. Anaconda to
get a recent Python if needed).

To install |kstacker| in your Python environment, from the git repository::

    $ git clone https://github.com/kstacker/kstacker.git
    $ cd kstacker
    $ pip install -e .

Quickstart
----------

This documentation now covers two related workflows:

1. the historical |kstacker| workflow based on noise profiles + brute-force grid
   search + optional gradient re-optimization,

2. the new MCMC workflow implemented in ``mcmc.py``, which supports:
   - a new non-singular orbital parameterization,
   - different photometry backends,
   - optional flux sampling,
   - multi-instrument inference from a single YAML file.

The important practical point is the following:

- use the "classical" YAML files for::

      kstacker noise_profiles ...
      kstacker optimize ...
      kstacker reopt ...

- use the new ``parameters_harmoni_mcmc.yml`` file for::

      kstacker mcmc ...

This separation is currently intentional. It is not yet fully convenient, but it
is temporary: the long-term goal is that noise profile computation, brute-force
search, re-optimization, and MCMC will all be driven by the same multi-instrument
YAML file.

Before running |kstacker|
^^^^^^^^^^^^^^^^^^^^^^^^^

Prepare a working directory for your dataset.

Single-instrument example
"""""""""""""""""""""""""

The directory ``mcmc_example/`` illustrates the simplest case, with a single
instrument (here HARMONI) and one MCMC configuration file.

A typical layout is::

    mcmc_example/
    ├── images/
    ├── parameters_harmoni.yml
    └── parameters_harmoni_mcmc.yml

where:

- ``images/`` contains the reduced FITS images,
- ``parameters_harmoni.yml`` is the classical parameter file used for
  ``noise_profiles``, ``optimize``, and optionally ``reopt``,
- ``parameters_harmoni_mcmc.yml`` is the new parameter file used only by the
  MCMC pipeline.

Image naming convention
"""""""""""""""""""""""

In the image directory, the FITS files must be named::

    image_0.fits, image_1.fits, ..., image_n.fits

The images must be square and ordered chronologically by epoch.

If you use the new MCMC pipeline with preprocessed native images or SNR maps,
additional files may also be present, for example::

    image_0_preprocessed.fits
    image_1_preprocessed.fits
    ...
    image_0_snr_map.fits
    image_1_snr_map.fits
    ...

depending on the photometry backend selected in the MCMC YAML file.

Classical workflow: noise profiles, brute-force search, gradient re-optimization
---------------------------------------------------------------------------------

The classical workflow is still required before MCMC in many practical cases,
especially if you want to initialize the MCMC from the best brute-force solution.

Use the *classical* YAML file here, for example::

    parameters_harmoni.yml

Single-instrument case
^^^^^^^^^^^^^^^^^^^^^^

For the single-instrument example in ``mcmc_example/``, the recommended order is:

1. Compute noise, background, and SNR profiles::

      kstacker noise_profiles parameters_harmoni.yml

2. Adjust the brute-force sampling grid if needed by inspecting the SNR plots
   produced in the profiles directory.

3. Run the brute-force orbital search::

      kstacker optimize parameters_harmoni.yml

4. Optionally refine the best brute-force solutions with the gradient
   re-optimization step::

      kstacker reopt parameters_harmoni.yml

This produces the usual outputs in the classical |kstacker| directories,
including the brute-force result file used later by the MCMC initialization
when ``init_mode: "bruteforce"`` is selected.

New MCMC workflow
-----------------

The new MCMC workflow is driven by the file::

    parameters_harmoni_mcmc.yml

It is implemented in ``mcmc.py`` and run with::

    kstacker mcmc parameters_harmoni_mcmc.yml

This MCMC driver is designed for direct-imaging orbit inference and supports:

- a non-singular orbital parameterization,
- several photometric extraction backends,
- two likelihood modes,
- soft mask handling,
- optional initialization from brute-force results,
- multi-instrument inference from a single YAML file,
- posterior plots, coadds, orbit overlays, log-probability maps, and GLRT
  off-track significance estimation.

New orbital variables: why the change of variables?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The MCMC does **not** sample the classical orbital variables directly.

Instead, it uses the non-singular parameter vector::

    θ = (a, λ0, m0, h, k, p, q)

with:

- ``a``: semi-major axis [AU]
- ``λ0``: mean longitude at reference epoch [rad]
- ``m0``: stellar mass [solar masses]
- ``h = e * sin(ω + θ0)``
- ``k = e * cos(ω + θ0)``
- ``p = sin(i/2) * cos(ω - θ0)``
- ``q = sin(i/2) * sin(ω - θ0)``

This change of variables is important because it avoids the usual singularities:

- when eccentricity goes to zero, the classical pair ``(M0, ω)`` becomes poorly
  defined,
- when inclination goes to zero, the classical angular orientation becomes
  degenerate.

With the new variables, the MCMC explores the parameter space more robustly and
more smoothly.

The classical parameters can still be recovered afterwards:

- ``e = sqrt(h² + k²)``
- ``ω + θ0 = atan2(h, k)``
- ``i = 2 * arcsin(sqrt(p² + q²))``
- ``ω - θ0 = atan2(q, p)``

The plotting tools in ``mcmc.py`` automatically reconstruct the derived physical
quantities such as ``e``, ``i``, ``M0``, ``t0``, ``omega``, and ``theta0``.

Single-instrument MCMC example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the single-instrument example in ``mcmc_example/``, the workflow is:

1. Run the classical preprocessing with ``parameters_harmoni.yml``::

      kstacker noise_profiles parameters_harmoni.yml
      kstacker optimize parameters_harmoni.yml

2. Optionally run::

      kstacker reopt parameters_harmoni.yml

3. Run the MCMC with the dedicated MCMC YAML::

      kstacker mcmc parameters_harmoni_mcmc.yml

Important: ``parameters_harmoni_mcmc.yml`` is **not** a replacement for the
classical YAML used by ``noise_profiles`` and ``optimize``. It is a dedicated
configuration file for the MCMC stage only.

Understanding ``parameters_harmoni_mcmc.yml``
---------------------------------------------

This section explains the main choices available in the new MCMC YAML file and
the points that are most likely to cause confusion.

Compatibility between photometry backend and likelihood
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are three photometry backends and two likelihood modes.

Photometry backends
"""""""""""""""""""

``convolve``
~~~~~~~~~~~~

This is the default backend and the closest to the historical |kstacker|
behavior.

It uses upsampled images and reads the value at the predicted position in the
upsampled frame. Background and noise are interpolated from radial profiles.

Use this when:

- you want behavior close to the legacy K-Stacker pipeline,
- you have upsampled preprocessed images,
- you want to use either the ``snr`` or ``flux`` likelihood.

Relevant YAML keys are typically::

    method: "convolve"
    upsampling_factor: 5

``aperture``
~~~~~~~~~~~~

This backend performs circular aperture photometry on native-resolution images,
using an aperture radius equal to ``fwhm``.

Use this when:

- you prefer aperture photometry instead of reading a single pixel value,
- you have native preprocessed images,
- you still want to use either the ``snr`` or ``flux`` likelihood.

Relevant YAML keys are typically::

    method: "aperture"
    fwhm: 2.0

``snr_map``
~~~~~~~~~~~

This backend uses precomputed SNR maps. At each epoch, the MCMC reads directly
the SNR value at the predicted pixel position.

This means:

- no background subtraction is done inside the MCMC,
- no per-pixel noise division is done inside the MCMC,
- the image already encodes the local SNR.

Use this when:

- your preprocessing pipeline already produced per-epoch SNR maps,
- you want the MCMC to work directly from those maps.

To activate it, set in an instrument block::

    snr_maps_suffix: "_snr_map"

The pipeline will then automatically load files such as::

    image_0_snr_map.fits
    image_1_snr_map.fits
    ...

and switch the backend to ``snr_map``.

Likelihood modes
""""""""""""""""

``likelihood_mode: "snr"``
~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the recommended default.

The likelihood is a surrogate based on the combined SNR along the orbit::

    log L ≈ snr_scale × SNR_total(θ)

This mode is compatible with:

- ``convolve``
- ``aperture``
- ``snr_map``

If at least one instrument uses ``snr_map``, the code automatically forces::

    likelihood_mode: "snr"

because flux-based likelihood is not meaningful once only SNR maps are available.

``likelihood_mode: "flux"``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This mode uses a Gaussian flux model with sufficient statistics.

It is only valid when the MCMC still has access to photometric fluxes and noise
profiles. Therefore it is compatible with:

- ``convolve``
- ``aperture``

and **incompatible** with:

- ``snr_map``

If ``sample_fp: true``, the planet flux ``fp`` is sampled as an additional 8th
parameter.

If ``sample_fp: false``, ``fp`` is fixed to the lower bound ``fp_bounds[0]``.

In short:

- use ``snr`` when in doubt,
- use ``flux`` only if you explicitly want a flux parameter model and are not
  using SNR maps.

Single-instrument setup in the YAML
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Even for a single instrument, the new MCMC YAML expects an ``instruments`` list.

A minimal single-instrument configuration looks like this::

    instruments:
      - name: "HARMONI"
        images_dir: "images"
        profile_dir: "profiles"
        p: 4
        p_prev: 0
        time: "0.0+4.0+8.0+12.0"
        n: 150
        method: "convolve"
        upsampling_factor: 5
        fwhm: 2.0
        resol: 4.0
        r_mask: 30.0
        r_mask_ext: 74

Main keys to understand
"""""""""""""""""""""""

``name``
~~~~~~~~

Human-readable instrument label. It is used in console messages and output
filenames.

``images_dir``
~~~~~~~~~~~~~~

Directory containing the FITS images for that instrument.

``profile_dir``
~~~~~~~~~~~~~~~

Directory containing the radial background/noise profiles for that instrument.

``p`` and ``p_prev``
~~~~~~~~~~~~~~~~~~~~

These define the number of epochs.

- ``p``: number of new images in the current run,
- ``p_prev``: number of images inherited from a previous run.

In many simple cases, set::

    p_prev: 0

and ``p`` equal to the number of images.

``time``
~~~~~~~~

Observation times in years, written as a ``+``-separated string in the YAML.

For example::

    time: "0.0+4.0+8.0+12.0"

The number of time values must match ``p + p_prev``.

In your updated code, time handling is more flexible internally, and the MCMC
supports multi-instrument time series cleanly, but from the user point of view
the YAML must still provide one valid time sequence per instrument.

``n``
~~~~~

Native image size in pixels. Images are assumed to be square ``n x n``.

``resol`` and ``dist``
~~~~~~~~~~~~~~~~~~~~~~

These define the AU-to-pixel conversion used for the projected orbit.

``dist`` is global and corresponds to the stellar distance in parsec.

``resol`` is instrument-specific and corresponds to the plate scale in mas/pixel.

``fwhm``
~~~~~~~~

PSF full width at half maximum in native pixels.

It is required for:

- ``aperture`` photometry,
- GLRT off-track offsets.

``r_mask`` and ``r_mask_ext``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These define the inner and outer working angles.

Important: in the new MCMC they are handled as **soft masks**.

This means:

- epochs falling inside ``r_mask`` or outside ``r_mask_ext`` are not rejected
  at the orbit level,
- instead, their photometric contribution is set to zero,
- the orbit can still be explored if other epochs remain informative.

This is more flexible than a hard rejection and is useful for planets that are
inside the mask only at some epochs.

Initialization choices
^^^^^^^^^^^^^^^^^^^^^^

The MCMC can be initialized in two ways.

``init_mode: "manual"``
"""""""""""""""""""""""

In this case, the starting point is read from the ``init`` section::

    init:
      a_init:   20.0
      la0_init: 0.0
      m0_init:  1.59
      h_init:   0.0
      k_init:   0.0
      p_init:   0.0
      q_init:   0.0

Walkers are then drawn around that point using the Gaussian spreads from
``init_spread``.

Use this when:

- you have a physically meaningful initial guess,
- you want to start without running the brute-force grid,
- you want a fully manual setup.

``init_mode: "bruteforce"``
"""""""""""""""""""""""""""

In this case, the MCMC reads the best solution from the brute-force result file
stored in::

    values_dir/res_grid.h5

and converts the classical orbital elements into the new non-singular variables
``(a, λ0, m0, h, k, p, q)``.

Use this when:

- you have already run ``noise_profiles`` and ``optimize``,
- you want the MCMC to refine the best brute-force orbit.

This is the recommended mode when available.

``bruteforce_swap_xy``
""""""""""""""""""""""

If the brute-force solution was produced with a different x/y convention than
the one used by the MCMC, set::

    bruteforce_swap_xy: true

This applies the corrective transformation before converting to the MCMC
parameterization.

Walker spreads
^^^^^^^^^^^^^^

The section::

    init_spread:
      a:   0.02
      la0: 0.2
      m0:  0.02
      hk:  0.02
      pq:  0.02
      fp:  0.2

controls the random cloud of walkers around the initial point.

Practical advice:

- when ``init_mode: "bruteforce"``, use small spreads,
- when ``init_mode: "manual"``, larger spreads may help exploration.

If the code fails to initialize enough walkers, the spread may be too narrow or
the initial point may be too close to prior boundaries.

Priors
^^^^^^

The new MCMC YAML separates priors clearly in a ``priors`` section.

``a_bounds`` and ``m0_bounds``
""""""""""""""""""""""""""""""

These are simple box priors on semi-major axis and stellar mass.

``la0_bounds``
""""""""""""""

Usually this is::

    [0, 2π]

and defines the allowed range of mean longitude at the reference epoch.

Eccentricity prior
""""""""""""""""""

Available options are:

``ecc_prior: "kipping"``
~~~~~~~~~~~~~~~~~~~~~~~~

This is a physically motivated Beta prior on eccentricity, recommended when you
want a prior favoring moderate and low eccentricities.

``ecc_prior: "uniform_e"``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Uniform prior in eccentricity from 0 to ``e_max``.

``ecc_prior: "uniform_disk"``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Uniform prior in the ``(h, k)`` disk.

``e_max``
~~~~~~~~~

Hard upper limit on eccentricity.

Orientation prior
"""""""""""""""""

The key::

    orientation_isotropic: true

uses the physically motivated isotropic prior in orientation space.

This is usually the right choice.

Reference epoch ``t_ref`` and ``λ0``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The parameter ``λ0`` is the mean longitude at a reference epoch ``t_ref``.

Two options exist:

``t_ref_mode: "min_ts"``
""""""""""""""""""""""""

This is the recommended setting.

The code uses the earliest observation time across all instruments as reference
epoch.

This is convenient because ``λ0`` then naturally refers to the first available
observation.

``t_ref_mode: "fixed"``
"""""""""""""""""""""""

In this case, the YAML value::

    t_ref: ...

is used explicitly.

Use this only if you need a specific external epoch convention.

Weighting
^^^^^^^^^

The MCMC combines epochs using either:

``weighting: "invvar"``
"""""""""""""""""""""""

Inverse-variance weighting. This is recommended in most cases because noisier
epochs contribute less.

``weighting: "simple"``
"""""""""""""""""""""""

Uniform weighting across epochs.

The old key::

    invvar_weight: true

is still understood for compatibility, but ``weighting`` is clearer and should
be preferred.

Flux parameter ``fp``
^^^^^^^^^^^^^^^^^^^^^

These settings matter only in ``flux`` likelihood mode.

``sample_fp: false``
""""""""""""""""""""

The flux is fixed at the lower bound of ``fp_bounds``.

``sample_fp: true``
"""""""""""""""""""

The planet flux is sampled as an extra 8th MCMC parameter.

The prior can be:

- ``fp_prior: "uniform"``
- ``fp_prior: "loguniform"``

``loguniform`` is generally more appropriate when the flux scale is poorly known.

Parallelization
^^^^^^^^^^^^^^^

The MCMC uses chunked multiprocessing for vectorized log-probability evaluation.

Relevant keys are::

    parallel:
      max_workers: null
      chunk_size: 256

- ``max_workers: null`` means "use all available CPUs",
- ``chunk_size`` controls how many walkers are grouped in one worker job.

Larger chunk sizes reduce overhead but increase per-worker memory usage.

Plots and diagnostics
^^^^^^^^^^^^^^^^^^^^^

The new MCMC can automatically generate a full set of diagnostic plots.

These include:

- a corner plot,
- one-dimensional posterior histograms,
- posterior-aligned native coadds,
- orbit overlays on each epoch,
- per-epoch log-probability maps,
- summed-image orbit plots,
- GLRT off-track significance estimates.

All of this is controlled from the ``plots`` section of the YAML.

Single-instrument example summary
---------------------------------

For the directory ``mcmc_example/``:

1. Use the classical file for preprocessing and brute-force search::

      kstacker noise_profiles parameters_harmoni.yml
      kstacker optimize parameters_harmoni.yml
      kstacker reopt parameters_harmoni.yml

2. Then use the MCMC file for posterior inference::

      kstacker mcmc parameters_harmoni_mcmc.yml

The MCMC file contains the new orbit parameterization, the likelihood settings,
the plotting configuration, and the optional GLRT significance analysis.

Multi-instrument example
------------------------

The directory ``mcmc_multiinstrument_example/`` illustrates a very simple
multi-instrument use case.

In this example, the original 4 HARMONI epochs have simply been split into two
groups in order to simulate a two-instrument configuration:

- ``images_HARMONI_1st_half`` contains the first half of the epochs,
- ``images_HARMONI_2nd_half`` contains the second half of the epochs.

This is only a pedagogical example, but it demonstrates how the new MCMC code
handles multiple instruments with independent image directories and time
sampling, while fitting one common orbital solution.

A typical layout is::

    mcmc_multiinstrument_example/
    ├── images_HARMONI_1st_half/
    ├── images_HARMONI_2nd_half/
    ├── parameters_harmoni_1st_half.yml
    ├── parameters_harmoni_2nd_half.yml
    └── parameters_harmoni_mcmc.yml

Temporary practical limitation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At the moment, the classical commands:

- ``noise_profiles``
- ``optimize``
- ``reopt``

are still run **instrument by instrument**.

This means that for the multi-instrument example you must launch them
separately for each instrument using separate classical YAML files.

For example:

First half
""""""""""

::

    kstacker noise_profiles parameters_harmoni_1st_half.yml
    kstacker optimize parameters_harmoni_1st_half.yml
    kstacker reopt parameters_harmoni_1st_half.yml

Second half
"""""""""""

::

    kstacker noise_profiles parameters_harmoni_2nd_half.yml
    kstacker optimize parameters_harmoni_2nd_half.yml
    kstacker reopt parameters_harmoni_2nd_half.yml

Then, once both halves have their own profiles and brute-force results, you can
run the multi-instrument MCMC **once**, using only::

    kstacker mcmc parameters_harmoni_mcmc.yml

This MCMC YAML contains an ``instruments`` list with one block for each dataset.
The code then:

- loads each instrument independently,
- reads each instrument's own images, profiles, and times,
- combines all instruments into a single log-posterior,
- infers one common orbit shared by all instruments.

Why is the multi-instrument workflow split like this?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because the MCMC has already been generalized to multi-instrument inference,
but the older preprocessing stages have not yet been fully merged into the same
multi-instrument interface.

So yes, at the moment this is not very practical.

This is temporary.

The intended future workflow is that:

- noise profiles,
- brute-force search,
- re-optimization,
- and MCMC

will all become multi-instrument aware and will all be launched from one single
YAML file.

For now, the correct usage is:

- one classical YAML per instrument for ``noise_profiles``, ``optimize``,
  and optional ``reopt``,
- one global MCMC YAML for the final multi-instrument posterior inference.

How multi-instrument inference works in the new MCMC
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each instrument block in ``parameters_harmoni_mcmc.yml`` defines its own:

- image directory,
- radial profiles,
- time sequence,
- image geometry,
- photometry backend,
- masks.

All instruments share the **same orbital parameters**.

The MCMC therefore estimates one posterior distribution for a single orbit,
given all instruments jointly.

Mathematically, the posterior is the sum of the prior and of all
per-instrument likelihood contributions.

This means that with:

- one instrument, the code behaves like a single-dataset MCMC,
- several instruments, the code combines all constraints consistently.

Notes on GLRT output in the multi-instrument case
-------------------------------------------------

The GLRT output prints per-instrument values and a global combined value.

The global ``Z_obs`` is **not** generally equal to the sum of the per-instrument
``Z_obs`` values, and the global ``Z²`` is **not** generally equal to the sum of
the per-instrument ``Z²`` values either.

This is normal.

The reason is that the global statistic is recomputed from the jointly combined
signal/noise model, using the chosen weighting rule, and not by simply adding
already-normalized per-instrument test statistics.

In particular:

- with inverse-variance weighting, the global combination is a weighted
  combination of signals and variances,
- with SNR-map instruments, the combined statistic may behave like a quadratic
  combination across instruments,
- therefore only the raw combined model defines the global ``Z_obs``.

So the global GLRT block should be interpreted as the detection significance of
the **joint dataset**, not as a trivial arithmetic sum of the individual
instrument significances.

Running on a Slurm cluster
--------------------------

The ``example/`` directory contains some examples of Slurm scripts to run
|kstacker|::

    slurm_launch_noise_prof.sh
    slurm_launch_brute_force.sh
    slurm_launch_reopt.sh

You can adapt this logic to the MCMC workflow as well, by launching the
classical steps separately for each instrument and then launching one final
multi-instrument MCMC job.

Results
-------

The results of your |kstacker| run will be stored in the values directory.

For the MCMC workflow, this typically includes:

- the sampled chain,
- posterior plots,
- coadds and orbit overlays,
- log-probability maps,
- summed-image orbit figures,
- the GLRT off-track JSON file.

In case of difficulty, you can :ref:`contact us <contact>`.

.. _acknowledgements:

Training
--------

You can train yourself by launching |kstacker| on the data provided in the
example directory.

The file ``Parameters_test_HD95086.yml`` has been pre-configured to search for
HD95086b in the field of view of SPHERE-IFS images.

The IFS (K-band) images of HD95086 provided in this example come from the
SPHERE / SHINE survey ([Chauvin2018]_, [Desgrange2022]_).

Chauvin, Gratton, Bonnefoy, et al. 2018, A&A, 617, A76; Desgrange et al. 2022,
accepted. They have been reduced by SPHERE-DC with an ASDI-TLOCI algorithm.
These data are available in the HC-DC.DIVA database
(https://cesam.lam.fr/diva/).

We also provide example Slurm files that can be used on a cluster.

Acknowledgements
----------------

The idea to search for hidden planets in series of observations was proposed
during the Observatoire de Haute-Provence 2015 meeting (Le Coroller et al. 2015,
'Twenty years of giant exoplanets' Edited by I. Boisse, O. Demangeon, F. Bouchy
& L. Arnold, p. 59-65). [Nowak2018]_ has written the first version of the
|kstacker| algorithm and tested its capability for detecting hidden planets
(snr_ks < 2 at each epoch) in simulated coronagraphic images. In
[LeCoroller2020]_, |kstacker| was validated through a dry run where fake planets
were injected and recovered in real SPHERE SHINE data. In this paper, we also
discussed the capability for |kstacker| to recover the orbital parameter space.
Recently, |kstacker| has been fully rewritten by Simon Conseil, a computer
engineer working at CeSAM / Laboratoire d'Astrophysique de Marseille
([LeCoroller2022]_, a scientific paper on Alphacen A NEAR-VISIR survey, where
this git repository link is given for the first time).

Students of L3-M2 also contributed to the initial project:
Antoine Schneeberger; Marie Devinat; Justin Bec-Canet; Dimitri Estevez

This research has been financed by PNP-INSU-CNRS.

Citation
--------

If you use this |kstacker| software for your research, please add this sentence
in the acknowledgements of your paper:

    "This work made use of the |kstacker| algorithm maintained by CeSAM at
    Laboratoire d'Astrophysique de Marseille"

You also have to cite the three original papers:

.. [Nowak2018] Nowak, M., Le Coroller, H., Arnold, L., et al. 2018, A&A, 615,
   A144, https://ui.adsabs.harvard.edu/abs/2018A%26A...615A.144N

.. [LeCoroller2020] Le Coroller, H., Nowak, M., Delorme, P., et al. 2020,
   A&A, 639, A113, https://ui.adsabs.harvard.edu/abs/2020A%26A...639A.113L

.. [LeCoroller2022] Le Coroller, H., Nowak, M., Wagner, K. et al. 2022, A&A, submitted

Papers describing the data used in the example directory:

.. [Chauvin2018] Chauvin, C., Gratton, R., Bonnefoy, M. et al. 2018, A&A, 617, A76

.. [Desgrange2022] Desgrange, C., Chauvin, G., Christiaens, et al. 2022, A&A, Accepted

.. _contact:

Contact
-------

If you need some help, you can contact us at:

herve.lecoroller@lam.fr, mcn35@cam.ac.uk, simon.conseil@lam.fr

Our team would be happy to collaborate on scientific projects using |kstacker|.

.. |kstacker| replace:: K-Stacker