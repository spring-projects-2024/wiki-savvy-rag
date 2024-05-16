#!/bin/bash

sbatch compute_embeddings.sh 0 1000
sbatch compute_embeddings.sh 1000 1000
sbatch compute_embeddings.sh 2000 1000
sbatch compute_embeddings.sh 3000 1000
sbatch compute_embeddings.sh 4000 1000
sbatch compute_embeddings.sh 5000 1000