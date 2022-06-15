|kstacker| Documentation
========================

.. important:: Before using this software, please read the
   :ref:`acknowledgements` at the end of this documentation.

How to install
--------------

|kstacker| requires Python>=3.7 and a C compiler (you can use e.g. Anaconda to
get a recent Python if needed).

To install |kstacker| in your Python environment, from the git repository::

    $ git clone https://gitlab.lam.fr/RHCI/kstacker.git
    $ cd kstacker
    $ pip install -e .

Quickstart
----------

Before running |kstacker|
^^^^^^^^^^^^^^^^^^^^^^^^^

- Create a directory where you will run |kstacker|.

- In this directory, create a sub-directory called ``images/`` (can be
  configured with the ``images_dir`` parameter).

- In the images directory put your reduced images. The files must be names
  ``image_0.fits``, ``image_1.fits``,... ``image_n.fits`` (squared images,
  sorted by epochs).

- Customize the ``parameters.yml`` configuration file for your data (you can
  start from the example file ``example/Parameters_test_HD95086.yml``):

    * Adjusts the numbers such as explained in the comments of
      ``parameters.yml``.

    * ``a_init`` must be a value near the maximum you want to search (but
      smaller than ``r_mask_ext`` in a.u.); you don't need to change the other
      ``_init`` values.

    * Adjust the range of the orbital parameters where you want to search for
      a planet : ``a_min``, ``a_max``; ``e_min``, ``e_max``, etc. ``t_0_min``
      must be equal to ``- sqrt(a_max^3 / m0)``.

    * You can put your ``parameters.yml`` file in the |kstacker| directory (at
      the same level than the images directory).

Running |kstacker|
^^^^^^^^^^^^^^^^^^

- The first step is to compute noise, background, and SNR profiles::

    kstacker noise_profiles parameters.yml

- When the noise profiles computation has finished, adjust the size of the grid
  by counting the number of maximums (peaks) of each ``snr_ks`` plot (see the
  PDF files in ``profiles/snr_plot_steps_remove_noise_no_999999/snr_graph/``)
  and report these numbers in the ``parameters.yml`` file (Na, Ne, Nt0, etc.)

- Then run the optimization step, which can be computationally intensive
  depending on the number of steps for the orbital parameters. To get an idea of
  the number of orbits that will be computed use the ``--dry-run`` argument. If
  |kstacker| has been compiled with the support of OpenMP, which should be the
  case by default on Linux, you can use the ``--nthreads`` argument to specify
  the number of parallel threads.

  ::

      kstacker optimize parameters.yml

- When the optimization computation is finished, run the gradiant by launching
  in a terminal::

    kstacker reopt parameters.yml

Running on a Slurm cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``example/`` directory contains some examples of Slurm scripts to run
|kstacker|::

    slurm_launch_noise_prof.sh
    slurm_launch_brute_force.sh
    slurm_launch_reopt.sh

Results
-------

The results of your |kstacker| run will be in the values directory.  In case of
difficulty, you can :ref:`contact us <contact>`.

.. _acknowledgements:

Acknowledgements
----------------

The idea to search for hidden planets in series of observations was proposed
during the Observatoire de Haute-Provence 2015 meeting (Le Coroller et al. 2015,
'Twenty years of giant exoplanets' Edited by I. Boisse, O. Demangeon, F. Bouchy
& L. Arnold, p. 59-65). Nowak, M. et al. 2018 has written the first version of
the |kstacker| algorithm and tested its capability for detecting hidden planets
(snr_ks < 2 at each epoch) in simulated coronagraphic images. In Le Coroller et
al. 2020, |kstacker| was validated through a dry run where fake planets where
injected and recovered in real SPHERE SHINE data. In this paper, we also
discussed the capability for |kstacker| to recover the orbital parameters space.
Recently, |kstacker| has been fully rewritten by Simon Conseil, a computer
engineer working at CeSAM / Laboratoire d'Astrophysique de Marseille (Le
Coroller et al. 2022, a scientific paper on Alphacen A NEAR-VISIR survey, where
this git repository link is given for the first time).

Students of L3-M2 had also contributed to the initial project:
Antoine Schneeberger; Marie Devinat; Justin Bec-Canet; Dimitri Estevez

---------

If you use this |kstacker| software for your research, please add this sentence
in the acknowledgements of your paper:

    "This work has make used of the |kstacker| algorithm maintained by CeSAM at
    Laboratoire d'Astrophysique de Marseille"

You also have to cite the three original papers:

    Nowak, M., Le Coroller, H., Arnold, L., et al. 2018, A&A, 615, A144

    Le Coroller, H., Nowak, M., Delorme, P., et al. 2020, A&A, 639, A113

    Le Coroller, H., Nowak, M., Wagner, K. et al. 2022, A&A, submitted

.. _contact:

Contact
-------

If you need some help, you can contact us at this email address :

herve.lecoroller@lam.fr, mcn35@cam.ac.uk, simon.conseil@lam.fr

Our team would be happy to collaborate on scientific projects using |kstacker|.


.. |kstacker| replace:: K-Stacker
