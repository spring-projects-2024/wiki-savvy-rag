#!/bin/bash

#SBATCH --job-name="run_file"
#SBATCH --account=3144860
#SBATCH --partition=debug
#SBATCH --gpus=1
#SBATCH --output=/home/3144860/wiki/wiki-savvy-rag/out/%x_%j.out # %x gives job name and %j gives job id
#SBATCH --error=/home/3144860/wiki/wiki-savvy-rag/err/%x_%j.er
#SBATCH --nodelist=gnode04
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --qos=debug


cd /home/3144860/wiki/wiki-savvy-rag

source /home/3144860/miniconda3/bin/activate nlp

conda info --envs

python backend/vector_database/faiss_wrapper.py

conda deactivate
