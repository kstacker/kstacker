|kstacker| Documentation
========================

.. important::

   Before using this software, please read the acknowledgements and citation sections at the end.

Overview
--------

|kstacker| is a tool designed to detect faint companions (e.g. exoplanets)
by combining multiple high-contrast imaging epochs along Keplerian orbits.

There are now **two workflows**:

1. The classical pipeline:
   - noise profiles
   - brute-force grid search
   - optional gradient re-optimization

2. The new MCMC pipeline:
   - robust orbital inference
   - flexible likelihoods
   - multi-instrument support

These workflows currently **coexist**.

---

Installation
------------

Clone the repository and install in editable mode:

::

    git clone https://github.com/kstacker/kstacker.git
    cd kstacker
    pip install -e .

You need:
- Python ≥ 3.7
- a working C compiler

---

Data preparation
----------------

Images must be:

- square
- ordered in time
- named:

::

    image_0.fits
    image_1.fits
    ...

Each file corresponds to one epoch.

---

Classical pipeline
------------------

This part is unchanged and still very useful.

Run:

::

    kstacker noise_profiles parameters.yml
    kstacker optimize parameters.yml
    kstacker reopt parameters.yml

What happens:

- noise/background profiles are computed
- a brute-force orbital search is performed
- best solutions can be refined

These results are especially useful to initialize the MCMC.

---

MCMC pipeline
-------------

Run with:

::

    kstacker mcmc parameters_harmoni_mcmc.yml

This is the **new core inference engine**.

It estimates orbital parameters using a likelihood-based approach.

---

Orbital parametrization
-----------------------

Instead of classical orbital elements, the MCMC uses:

::

    (a, λ0, m0, h, k, p, q)

Why?

Because classical parameters become unstable:

- e → 0 → ω undefined
- i → 0 → degeneracy

The new variables avoid these singularities:

- h = e sin(ω + θ0)
- k = e cos(ω + θ0)
- p = sin(i/2) cos(ω - θ0)
- q = sin(i/2) sin(ω - θ0)

This makes the MCMC **much more stable**.

---

Likelihoods
-----------

This is the core of the method.

Single instrument
^^^^^^^^^^^^^^^^^

At each epoch k:

- signal: S_k
- noise: σ_k

SNR:

::

    SNR_k = S_k / σ_k

---

SNR mode (default)
"""""""""""""""""

We combine all epochs into a single quantity.

Inverse-variance weighting:

::

    SNR_total =
        ( Σ S_k / σ_k² ) / sqrt( Σ 1 / σ_k² )

Then:

::

    log L = snr_scale × SNR_total

This is a **surrogate likelihood**:
- simple
- robust
- very efficient

---

Flux mode (Gaussian likelihood)
"""""""""""""""""""""""""""""""

Here we model the actual flux.

At each epoch:

::

    d_k = f_p + noise

Likelihood:

::

    log L =
        -1/2 Σ (d_k - f_p)² / σ_k²

We can rewrite it using:

::

    S1 = Σ d_k / σ_k²
    S2 = Σ 1 / σ_k²

Then:

::

    log L =
        -1/2 ( S2 f_p² - 2 f_p S1 )

If we optimize f_p:

::

    f_p = S1 / S2

Final form:

::

    log L = S1² / (2 S2)

Important:

This is mathematically equivalent to a **squared SNR**.

---

Multi-instrument likelihood
^^^^^^^^^^^^^^^^^^^^^^^^^^

Now we combine several instruments.

Each instrument has its own:
- images
- noise
- time sampling

But the orbit is shared.

---

Correct combination
"""""""""""""""""""

We combine **signal and variance**, not SNR.

::

    S1 = Σ_i Σ_k d_ik / σ_ik²
    S2 = Σ_i Σ_k 1 / σ_ik²

Then:

::

    log L = S1² / (2 S2)

Key point:

::

    global SNR ≠ sum of SNR per instrument

This is very important.

---

SNR-map mode
^^^^^^^^^^^^

If you use precomputed SNR maps:

::

    image_k_snr_map.fits

Then:

- no noise model is used
- the value at each pixel is already SNR

So:

::

    log L ∝ Σ SNR values

This is simpler but less physical.

---

Multi-instrument workflow
-------------------------

Each instrument is defined independently in the YAML.

Example:

::

    instruments:
      - name: "HARMONI_1"
        images_dir: "images1"
        time: [0.0, 4.0]

      - name: "HARMONI_2"
        images_dir: "images2"
        time: [8.0, 12.0]

Each instrument contributes to the same likelihood.

---

Important limitation (current)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Preprocessing is still done per instrument.

You must run:

::

    kstacker noise_profiles params_inst1.yml
    kstacker optimize params_inst1.yml

    kstacker noise_profiles params_inst2.yml
    kstacker optimize params_inst2.yml

Then:

::

    kstacker mcmc parameters_harmoni_mcmc.yml

---

GLRT and FAP (offtracks)
------------------------

This part is critical to interpret detections.

What is computed?
^^^^^^^^^^^^^^^^^

The code evaluates a detection statistic:

::

    Z_obs

This measures how strong your signal is along the best orbit.

---

Offtracks (null hypothesis)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To understand if this detection is real, we simulate **fake orbits**:

- random trajectories
- not matching a real Keplerian orbit

For each fake orbit, we compute:

::

    Z_fake

This builds a **null distribution**.

---

False Alarm Probability (FAP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The FAP is:

::

    FAP = P(Z_fake ≥ Z_obs)

In words:

    "What is the probability that noise alone produces a signal
    at least as strong as the observed one?"

---

Interpretation
^^^^^^^^^^^^^^

- FAP ≈ 0.1 → not significant
- FAP ≈ 1e-2 → weak detection
- FAP ≈ 1e-3 → strong detection
- FAP ≈ 1e-5 → very strong detection

Important:

FAP depends on:

- number of offtracks (n_off)
- noise properties
- masking

---

Multi-instrument GLRT
^^^^^^^^^^^^^^^^^^^^^

Each instrument has its own contribution.

But the global statistic is computed from the **combined model**.

So:

::

    Z_total ≠ Z_1 + Z_2

This is normal.

---

Best practices
--------------

- Use **snr mode** unless you really need flux inference
- Use **bruteforce initialization** when possible
- Always check:
  - corner plot
  - orbit overlay
  - coadd image
- Always look at **FAP**, not only SNR

---

Outputs
-------

Results are saved in:

::

    values/

You will find:

- MCMC chains
- posterior plots
- coadded images
- orbit overlays
- GLRT results (JSON)

---

Conclusion
----------

The new MCMC pipeline provides:

- robust orbital inference
- multi-instrument combination
- statistical significance (FAP)

It is more powerful than the classical pipeline, but both are still useful together.

---

Contact
-------

herve.lecoroller@lam.fr  
mcn35@cam.ac.uk  
simon.conseil@lam.fr