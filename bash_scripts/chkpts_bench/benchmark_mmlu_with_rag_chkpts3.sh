#!/bin/bash

#SBATCH --job-name="mmlu_benchmark"

#SBATCH --account=3144366

#SBATCH --partition=stud

#SBATCH --gpus=1

#SBATCH --output=out/%x_%j.out # %x gives job name and %j gives job id

#SBATCH --error=err/%x_%j.er

#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

cd /home/3144366/textbook-savvy-rag

module load modules/miniconda3

source activate base

conda info --envs

for file in configs/checkpoints/step*.yaml; do
    python3 scripts/benchmark/mmlu.py --config_path $file --log_answers True --k_shot 0 --use_rag 1 --inference_type "replug" --n_docs_retrieved 5 --n_samples 500
done

conda deactivate
