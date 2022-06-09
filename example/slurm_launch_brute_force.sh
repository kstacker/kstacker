#!/bin/bash

sbatch \
    --mail-user=herve.lecoroller@lam.fr \
    --mail-type=BEGIN,END \
    -t 05:00:00 \
    -c 22 \
    --mem=12GB \
    -J brute_testH95 \
    -p batch \
    kstacker optimize Parameters_test_HD95086.yml --progress --nthreads 22

echo "Computation of the profiles and SNRs"
