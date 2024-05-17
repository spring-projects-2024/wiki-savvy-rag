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

python3 scripts/embeddings/cluster_script_embedding.py --device "cuda" --max_accumulation 250 --offset $1 --chunks $2

conda deactivate
