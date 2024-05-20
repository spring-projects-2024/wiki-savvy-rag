#!/bin/bash

#SBATCH --job-name="write_faiss"
#SBATCH --account=3144366
#SBATCH --partition=stud
#SBATCH --gpus=1
#SBATCH --output=out/%x_%j.out # %x gives job name and %j gives job id
#SBATCH --error=err/%x_%j.er
#SBATCH --nodelist=sgnode01
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G


cd /home/3144366/textbook-savvy-rag

module load modules/miniconda3

source activate base

conda info --envs

python3 scripts/vector_database/train_vector_database.py --index "PQ128" --training_size .01 --nprobe 32 --output "scripts/vector_database/data/PQ128.index"

conda deactivate
