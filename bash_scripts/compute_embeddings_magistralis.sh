#!/bin/bash

# write a for cycle to submit multiple jobs

n=100000

for (( i = 0; i < 140*n; i+=n )); do
    sbatch compute_embeddings.sh $((i)) $((n))
done

