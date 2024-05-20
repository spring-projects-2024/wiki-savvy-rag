#!/bin/bash

#SBATCH --job-name="populate_dataset"

#SBATCH --account=3144366

#SBATCH --partition=stud

#SBATCH --gpus=1

#SBATCH --output=out/%x_%j.out # %x gives job name and %j gives job id

#SBATCH --error=err/%x_%j.er

cd /home/3144366/textbook-savvy-rag

module load modules/miniconda3

source activate base

conda activate py3-12

conda info --envs

python3 scripts/dataset/populate_dataset.py 

conda deactivate
