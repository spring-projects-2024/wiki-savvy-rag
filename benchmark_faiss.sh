#!/bin/bash

#SBATCH --job-name="compute_embeddings"

#SBATCH --account=3144366

#SBATCH --partition=stud

#SBATCH --gpus=1

#SBATCH --output=out/%x_%j.out # %x gives job name and %j gives job id

#SBATCH --error=err/%x_%j.er

#SBATCH --nodelist=sgnode01

cd /home/3144366/textbook-savvy-rag

module load modules/miniconda3

source activate base

conda info --envs

python3 scripts/embeddings/bench_quantizer.py --centroids 131072 --knn_neighbors 100 --nprobe 32 --training_size 0.1

conda deactivate
