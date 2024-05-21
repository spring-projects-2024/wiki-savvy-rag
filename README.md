# Wikipedia-Savvy-RAG

For our project, we implemented, fine-tuned, and evaluated a Retrieval Augmented Generation system that could discuss a range of topics in the STEM domain. To showcase our system, we developed a simple application with a chat interface that replies to the user prompts with the help of retrieved Wikipedia passages.

We downloaded, cleaned, and filtered the English Wikipedia (~100GB) and built a vector database of semantic embeddings based on STEM articles. We then built a RAG system using one of open-source LLMs, `Minami-su/Qwen1.5-0.5B-Chat_llamafy` and `microsoft/phi-3-mini-128k-instruct`, and the open-source embedder `BAAI/bge-small-en-v1.5`. To evaluate our system, we considered its accuracy on question answering benchmarks focusing on the STEM domain, e.g., using an appropriate subset of MMLU.

To improve the performance of our system, we fine-tuned the LLM on a STEM question answering task. We assessed the performance with and without RAG, as well as before and after fine-tuning.

The chat was developed using Streamlit and allows users to configure key aspects of the chatbot, such as selecting the model, determining the number of retrieved documents, choosing the type of RAG inference, and setting the decoding strategy.

## Project Structure

Our project is organized into the following folders:

- `backend`: This folder contains the core code for the RAG handler, LLM handler, embedder wrapper, vector database wrapper, and model trainer. It also includes various utility functions used throughout the project. For convenience, it is installed as a package (see the Installation section).
- `bash_scripts`: This folder contains our bash script definitions.
- `configs`: This folder holds YAML configuration files for initializing the RAG handler.
- `frontend`: This folder includes the code for the ChatBot demo interface.
- `notebooks`: This folder contains Python notebooks primarily used for exploratory data analysis.
- `scripts`: This folder has scripts for building the dataset, computing embeddings, training models, constructing the FAISS vector database, fine-tuning the model, and benchmarking with MMLU.
- `wikidump_processing`: This folder includes all the code for retrieving, processing, and cleaning the Wikipedia dump.


## Preliminary steps 

This steps are required to both run the ChatBot and to replicate our results on the MMLU evaluation dataset.

### Installation

From the root directory:

```
pip install -r requirements.txt
pip install -e .
git clone https://github.com/MattiaSC01/JEPA.git
cd JEPA
pip install -r requirements.txt
pip install -e .
```

### Download the Wikipedia Dump

*TODO: Fill with info about where to download the dump file (I don't remember lol)*  
The required space on disk is about 100GB.

### Process the Wikipedia Dump

Run the following script from the root folder to run the processing of the Wikipedia dump.  
At the end of the script you should have the cleaned and processed version in the `wikidump_processing/data/subsample_chunkeder.xml` file.  
The required space on disk is about 26GB *(to confirm)*.

```bash
chmod +x wikidump_processing/clean_dump.sh
wikidump_processing/clean_dump.sh /path/to/dump.xml True
```

### Create SQLite dataset

Run the following script to create and populate a SQLite dataset containing the Wikipedia chunks.
The required space on disk is about 9GB.

```bash
python scripts/dataset/populate_dataset.py
    --input "wikidump_processing/data/subsample_chunkeder.xml"
    --db_dir "scripts/dataset/data"
    --db_name="dataset"
```

### Compute embeddings

The following script calculates the embeddings and dumps them on multiple files in the specified folder.
The required space on disk for all the embeddings is about 20GB (*to confirm*).

```bash
python scripts/embeddings/compute_embeddings.py
    --device "cuda:0"
    --db_dir "scripts/dataset/data"
    --db_name="dataset"
    --output_dir "scripts/embeddings/data/"
    --max_accumulation 250
```

Since the computation of the embeddings is heavy and might cause troubles (like out of memory errors), we created a bash script that runs multiple instances of the script:

```bash
chmod +x bash_scripts/compute_embeddings_magistralis.sh
bash_scripts/compute_embeddings_magistralis.sh
```

### Build vector database

The following script builds the vector database with the given index factory string (refer to [faiss documentation](https://github.com/facebookresearch/faiss)).  
The required space on disk depends on the chosen index factory.

```bash
python scripts/vector_database/train_vector_database.py
    --index "PQ128"
    --training_size 0.1
    --input_dir "scripts/embeddings/data"
    --output "scripts/vector_database/data/default.index"
```

### Troubleshooting libmagic dependency

If `refextract` import throws error and can't find libmagic, based on your virtual env, you may also need to run:

For Conda
```bash
conda install -c conda-forge libmagic
```

On MacOS (Brew), Python Virtual Environment
```bash
brew install libmagic
```

On Windows, Python Virtual Environment 
```bash
pip install python-magic-bin
```

## Reproduce our results 

### Benchmark on FAISS indices

### Benchmark on MMLU  

### Train LLM

## ChatBot

*Include animated gif*
