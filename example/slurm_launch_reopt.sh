#!/bin/bash

sbatch \
    --mail-user=herve.lecoroller@lam.fr \
    --mail-type=BEGIN,END \
    -t 01:00:00 \
    -c 22 \
    --mem=24GB \
    -J reopt_testHD95 \
    -p batch \
    kstacker reopt Parameters_test_HD95086.yml --njobs 22
