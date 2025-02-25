#!/bin/bash


for num_labels in 3 10 20 100
do
    sbatch --partition=h100 /weka/scratch/bvandur1/zjiang31/decoding-based-regression/runs/run_pipe.sh \
        --num-labels $num_labels \
        --reg-method "fd" \
        --scale 1.0
done