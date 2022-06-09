K-Stacker Documentation
======================

Important
---------

Before you use this software, please read the acknowledgements at the end of this documentation

How to install
--------------

1) The most simple is to install kstacker in a conda environment:

        https://www.anaconda.com/products/distribution

2) In a terminal type:

    - conda activate "path of your k-stacker conda environement"

    - git clone https://gitlab.lam.fr/RHCI/kstacker.git

    - cd "path of the kstacker directory"

    - pip install -e .

Quick launch
------------

1) Before launching K-Stacker

    - Create a directory where you will run K-Stacker
    - In this directory, create a sub-directory called /images
    - In /images directory put your reduced images re-called image_0, image_1,... image_n (squared images, sorted by epochs)
    - Fill the parameters.yml configuration file for your data (you can change with your values in " this exemple "):

        * Adjusts the numbers such as explained in the comments of parameters.yml
        * a_init must be a value near the maximum you want to search (but smaller than r_mask_ext in a.u.); you don't need to change the other _init values
        * Adjust the range of the orbital parameters where you want to search for a planet : a_min, a_max; e_min, e_max, etc. t_0_min must be equal to: - sqrt(a_max^3 / m0)
        * You can put your parameters.yml file in the k-stacker directory (at the same level than the images directory)

2) Run K-Stacker on one core:

- Activate your conda environement by typing in a terminal :

            conda activate "path of your k-stacker conda environement"

- Launch the noise profiles software by typing in a terminal:

            kstacker noise_profiles parameters.yml

- When the noise profile computation has finished, adjust the size of the grid by counting the number of maximums (peaks) of each snr_ks plot (pdf in profiles/snr_plot_steps_remove_noise_no_999999/snr_graph/) and report these numbers in the parameters.yml file (Na, Ne, Nt0, etc.)

- Run the optimization software by typing in a terminal:

            kstacker optimize Parameters.yml

- When the optimization computation is finished, run the gradiant by launching in a terminal :

            kstacker reopt Parameters.yml

3) Alternativelly, you can run k-Stacker on a cluster of computation by using slurm:

- Follow the same steps than in 2. but by launching, in a terminal:

    sh slurm_launch_noise_prof.sh

    sh slurm_launch_brute_force.sh

    sh slurm_launch_reopt.sh

k-stacker use opennp to run on several cores. You can find an exemple of the slurm_*.sh files in the example directory.

Results
-------

The results of your k-stacker run will be in the values directory

In case of difficulty, you can contact us at herve.lecoroller@lam.fr, mcn35@cam.ac.uk

acknowledgements
----------------

The idea to search for hidden planets in series of observations was proposed during the Observatoire de Haute-Provence 2015
meeting (Le Coroller et al. 2015, 'Twenty years of giant exoplanets' Edited by I. Boisse, O. Demangeon, F. Bouchy & L. Arnold, p. 59-65). Nowak, M. et al. 2018 has written the first version of the K-Stacker algorithm and tested its capability for detecting hidden planets (snr_ks < 2 at each epoch) in simulated coronagraphic images. In Le Coroller et al. 2020, k-Stacker was validated through a dry run where fake planets where injected and recovered in real SPHERE SHINE data. In this paper, we also discussed the capability for K-Stacker to recover the orbital parameters space. Recently, K-Stacker has been fully rewritten by Simon Conseil, a computer engineer working at CeSAM / Laboratoire d'astrophysique de Marseille (Le Coroller et al. 2022, a scientific paper on Alphacen A NEAR-VISIR survey, where this git repository link is given for the first time).

Students of L3-M2 had also contributed to the initial project:
Antoine Schneeberger; Marie Devinat; Justin Bec-Canet; Dimitri Estevez

---------

If you use this k-stacker software for your research, please add this sentence in the acknowledgements of your paper:

      "This work has make used of the K-Stacker algorithm maintened by CeSAM at Laboratoire d'Astrophysique de Marseille"

You also have to cite the three original papers:

         Nowak, M., Le Coroller, H., Arnold, L., et al. 2018, A&A, 615, A144

         Le Coroller, H., Nowak, M., Delorme, P., et al. 2020, A&A, 639, A113

         Le Coroller, H., Nowak, M., Wagner, K. et al. 2022, A&A, submitted

---------

If you need some help, you can contact us at this email address :

herve.lecoroller@lam.fr, mcn35@cam.ac.uk, simon.conseil@lam.fr

Our K-Stacker team would be happy to collaborate on scientific projects using k-Stacker.