#!/bin/bash

#SBATCH --job-name="train"
#SBATCH --account=3144860
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --output=/home/3144860/wiki/wiki-savvy-rag/out/%x_%j.out # %x gives job name and %j gives job id
#SBATCH --error=/home/3144860/wiki/wiki-savvy-rag/err/%x_%j.er
#SBATCH --nodelist=gnode02
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G


cd /home/3144860/wiki/wiki-savvy-rag

source /home/3144860/miniconda3/bin/activate nlp

conda info --envs

python scripts/training/train.py --config_path configs/training/final.yaml

conda deactivate
