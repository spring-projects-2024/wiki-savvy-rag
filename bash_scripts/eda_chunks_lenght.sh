#!/bin/bash

#SBATCH --job-name="eda_chunks_length"

#SBATCH --account=3144366

#SBATCH --partition=stud

#SBATCH --gpus=1

#SBATCH --output=out/%x_%j.out # %x gives job name and %j gives job id

#SBATCH --error=err/%x_%j.er

#SBATCH --nodelist=sgnode01

cd /home/3144366/textbook-savvy-rag

module load modules/miniconda3

source activate base

conda activate py3-12

conda info --envs

python3 scripts/embeddings/eda_chunk_size_tokenized.py 

conda deactivate
