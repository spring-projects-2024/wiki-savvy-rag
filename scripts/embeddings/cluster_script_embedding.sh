#!/bin/bash

python scripts/embeddings/cluster_script_embedding.py --device "cuda:1" --db_dir "scripts/dataset/data" --db_name="dataset" --output_dir "scripts/embeddings/data/" --max_accumulation 300000 