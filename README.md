# Wiki-Savvy

Wiki Savvy is a Retrieval Augmented LLM that can discuss any STEM-related topic through a chat interface, retrieving facts from the English Wikipedia and citing its sources. It was developed by Mattia Scardecchia, Dario Filatrella, Federico Zarantonello and Michele Palma as a project for a Natural Language Processing class at Bocconi University, in Spring 2024.

We downloaded, filtered and cleaned the English Wikipedia (~100GB) and built a vector database of semantic embeddings based on all STEM articles, using the FAISS library for efficient retrieval of embeddings and SQLite for accessing text chunks from disk. We used QLoRA for efficient supervised finetuning of [this](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat) open-source LLM on a subset of the Yahoo Answers question answering dataset, retrieving documents with [this](https://huggingface.co/BAAI/bge-small-en-v1.5) frozen pre-trained embedder, and tracking progress on the MMLU benchmark.

Below you can see a simple demonstration through a Streamlit demo. The LLM provides an answer to the user's question, and lists the titles and relevance scores of the wikipedia paragraphs used to inform its generation.

![](media/demo.gif)

## Installation

To install Python dependencies, run from the project root:

```
bash bash_scripts/installation.sh
```

If you want to finetune a model, additionally pass the argument `--finetuning-dependencies` to the installation script.
This will install the project at `https://github.com/MattiaSC01/JEPA` as a python package, in addition to the other dependencies.

To run inference with RAG, you will need the vector database. You can download all the necessary files
from [this link](https://bocconi-my.sharepoint.com/:u:/g/personal/federico_zarantonello_studbocconi_it/ESKBb1Hh7hFJhPzT-LvENOQBH0MqysZM9AtW_jdhYzqn1A?e=prSAOp); extract them and place them in the root directory of the project.

### Reproducing Vector Database Creation

If you prefer to reproduce our steps and create the vector database from the Wikipedia dump for yourself, you can follow these steps, after having installed python dependencies.

#### Download the Wikipedia Dump

The Wikipedia Dump can be downloaded from [WikiMedia](https://meta.wikimedia.org/wiki/Data_dump_torrents#English_Wikipedia).
The space required is about 100GB.
We used the version dated 2023-12-20.

#### Process the Wikipedia Dump

To filter and clean the Wikipedia dump, from the root directory, run:

```bash
bash wikidump_processing/clean_dump.sh /path/to/dump.xml True
```

This will create a file `wikidump_processing/data/subsample_chunkeder.xml` containing the cleaned and filtered wikipedia.  
The space required is about 26GB.

#### Create SQLite dataset

Run the following script to create and populate a SQLite dataset containing the Wikipedia chunks.
The space required is about 9GB.

```bash
python scripts/dataset/populate_dataset.py --input "wikidump_processing/data/subsample_chunkeder.xml" --db_dir "scripts/dataset/data" --db_name="dataset"
```

#### Compute embeddings

The following script calculates the embeddings and dumps them on multiple files in the specified directory (be sure that the directory exists)
The required space on disk for all the embeddings is about 20GB.

```bash
python scripts/embeddings/compute_embeddings.py --device "cuda" --db_dir "scripts/dataset/data" --db_name="dataset" --output_dir "scripts/embeddings/data/" --max_accumulation 250
```

Since the computation of the embeddings is heavy and might cause trouble (like out of memory errors), we created a bash script that runs multiple instances of the script:

```bash
bash bash_scripts/compute_embeddings_magistralis.sh
```

#### Build vector database

The following script builds the vector database with the given index factory string (refer to [faiss documentation](https://github.com/facebookresearch/faiss)).  
The required space on disk depends on the chosen index factory.

```bash
python scripts/vector_database/train_vector_database.py --index "PQ128" --training_size 0.01 --input_dir "scripts/embeddings/data" --output "scripts/vector_database/data/PQ128.index"
```

The exact configuration we used is in the following bash script:

```bash
bash bash_scripts/train_vector_database.sh
```

## Running the Demo

To run the Streamlit demo, from the root:

```bash
streamlit run frontend/app.py
```

To be able to use the Retrieval Augmented Generation with chunks from Wikipedia, the ChatBot requires to have in your system the following files:

- **SQLite database file**: this is produced by following [these instructions](#create-sqlite-dataset).
- **Vector database**: produced by following [these instructions](#build-vector-database).

To be able to run using a finetuned model, you need also the adapter files (see the [Finetuning the LLM](#finetuning-the-llm) section).

There are many options, that can be configured directly from the demo's GUI.

### ArXiv Papers on the fly

When the 'Use ArXiv' flag is enabled, you can provide a link to an arxiv paper, that will be processed on the fly and included in the vector database for retrieval. Papers will be downloaded locally and automatically deleted after usage.

![](media/arxiv.gif)

## Finetuning the LLM

Run the following script to finetune the LLM with QLoRA on a subset of Yahoo Answers, using a frozen embedder to retrieve relevant passages:

```bash
python scripts/training/train.py --config_path configs/training/final.yaml
```

## Benchmarking on MMLU

To benchmark your model on the subset of MMLU containing STEM-related questions, you can run the following script:

```bash
python scripts/benchmark/mmlu.py --split "test" --subset "stem" --output "/path/to/output.json" --k_shot 1 --batch_size 1 --config_path "/path/to/config.yaml" --use_rag True --n_docs_retrieved 3 --log_answers True --inference_type "replug"
```

Refer to the bash scripts in the folders `bash_scripts/benchmark_original_model` for benchmarks on different variation of the original model and to the `bash_scripts/chkpts_bench` for benchmarks on the training checkpoints.

## Benchmarking the FAISS indices

To systematically compare the retrieval performance (accuracy and efficiency) of different indexing algorithms in the FAISS vector database, you can use the following benchmarking script:

```bash
python scripts/vector_database/benchmark_faiss.py --knn_neighbors 100 --nprobe 32 --training_size 0.01 --mmlu_sample_size 300
```
