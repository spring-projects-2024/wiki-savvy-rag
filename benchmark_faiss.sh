#!/bin/bash

#SBATCH --job-name="benchmark_faiss"
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

python3 scripts/vector_database/bench_quantizer.py --centroids 5000 --knn_neighbors 100 --nprobe 32 --training_size 0.01 --mmlu_sample_size 3000

conda deactivate
