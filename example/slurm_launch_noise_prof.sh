#!/bin/bash

sbatch \
    --mail-user=herve.lecoroller@lam.fr \
    --mail-type=BEGIN,END \
    -t 01:00:00 \
    -c 1 \
    --mem=12GB \
    -J noise_testHD95 \
    -p batch \
    kstacker noise_profiles Parameters_test_HD95086.yml

echo "Computation of the profiles and SNRs"
