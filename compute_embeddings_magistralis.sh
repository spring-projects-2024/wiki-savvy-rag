#!/bin/bash

# write a for cycle to submit multiple jobs

n=100

for (( i = 0; i < 14*n; i+=n )); do
    sbatch compute_embeddings.sh $((i)) $((n))
done

