# Wikipedia-Savvy-RAG

*Include animated gif of the ChatBot*

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

## 


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

These commands are neatly packaged in a shell script. You can simply run, from the root:

```
bash bash_scripts/installation.sh
```

### Download the Wikipedia Dump

The Wikipedia Dump can be downloaded from [WikiMedia](https://meta.wikimedia.org/wiki/Data_dump_torrents#English_Wikipedia).
The required space on disk is about 100GB.
We used the version dated 2023-12-20.

### Process the Wikipedia Dump

Run the following script from the root folder to run the processing of the Wikipedia dump.  
At the end of the script you should have the cleaned and processed version in the `wikidump_processing/data/subsample_chunkeder.xml` file.  
The required space on disk is about 26GB *(to confirm)*.

```bash
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
The required space on disk for all the embeddings is about 20GB.

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

The exact configuration we used is in the following bash script:

```bash
bash_scripts/train_vector_database.sh
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

## ChatBot

To run the ChatBot demo, run the following script:

```bash
    streamlit run frontend/app.py
```

### Options

The demo allows users to customize configuration options, as shown in the following screenshot:

*insert screenshot*

All options can be configured directly through the chatbot's UI:

* **Device**: Users can select from all compatible devices available on their machine. If a CUDA-enabled graphics card is present, it is recommended to select it for improved performance.
* **Model**: Choose between `Minami-su/Qwen1.5-0.5B-Chat_llamafy` and `microsoft/phi-3-mini-128k-instruct`. Despite all our analysis were done on Qwen 1.5, we inserted also 
Microsoft's Phi-3 for comparison.
These models require approximately 2GB and 16GB of memory on the selected device, respectively.
* **Decoding Strategy**: Supported options include:
  * Greedy decoding
  * Top-k decoding (considering 50 tokens)
  * Top-p decoding (with a cumulative probability of 0.9)
* **Use RAG**: Option to retrieve and use documents to enhance the assistant's replies.
* **Database Path**: Path to the SQL database generated during the preliminary steps (see the Preliminary Steps section).
* **Vector Index Path**: Path to the vector database.
* **Inference Type**: Defines how to use retrieved documents during inference:
  * **Naive**: Append all documents before the query and perform inference based on that.
  * **REPLUG**: Append each document to the query separately, determine token probabilities, and calculate weighted averages of these tokens based on document similarity. For more information, see the [REPLUG](https://arxiv.org/abs/2301.12652) paper.
  * **Mock**: Mock response of the chatbot (for testing purposes).
* **Number of Documents to Retrieve**: Specify the number of documents to retrieve.


### Notes for future improvements

Currently, with the RAG option enabled, the chatbot only considers the user's prompt and does not take into account the history of previous messages. Integrating memory handling with RAG is outside the scope of this project. However, future enhancements may include this capability.

## Our results (and how to reproduce them)

This section outlines the benchmarks we performed and the results we obtained.
To reproduce our results, ensure to first run the preliminary scripts using our configurations, which are defined in the provided bash scripts in the folder `bash_scripts`.

### Benchmark on FAISS indices

To choose the best index among those offered by FAISS, we conducted benchmarks evaluating several metrics: the intersection measure of the results compared to the true ordering, the time taken to retrieve documents for all queries, and the index size on disk (which provides a lower bound for its true memory size).

To run your own benchmarks, use the following script:

```bash
python3 scripts/vector_database/bench_quantizer.py
    --knn_neighbors 100 
    --nprobe 32 
    --training_size 0.01 
    --mmlu_sample_size 300
```

In our evaluation, we trained our indexes on 1% of Wikipedia data and assessed the intersection measure for 1, 10, 50, and 100 results (with `knn_neighbors` specifying the maximum number of neighbors to retrieve). We evaluated 300 questions from the MMLU dataset and set the number of centroids for the IVF indexes to 32 (`nprobe`, for more details, refer to the [FAISS documentation](https://github.com/facebookresearch/faiss)).

Here is a table summarizing our results:

| Index                                 | Rank 1 | Rank 10 | Rank 50 | Rank 100 | Elapsed Time | Size on Disk (bytes) |
| ------------------------------------- | ------ | ------- | ------- | -------- | ------------ | -------------------- |
| SQ4                                   | 0.72   | 0.832   | 0.8477  | 0.8558   | 03:53.59     | 2,609,777,937        |
| SQ8                                   | 0.99   | 0.9873  | 0.9885  | 0.9892   | 03:29.50     | 5,219,552,721        |
| PQ64                                  | 0.4167 | 0.535   | 0.5798  | 0.5919   | 00:52.62     | 870,318,230          |
| PQ128                                 | 0.63   | 0.781   | 0.8038  | 0.8144   | 01:55.76     | 1,740,243,158        |
| OPQ64_256,IVF1000_HNSW32,PQ64x4fsr    | 0.18   | 0.2613  | 0.2797  | 0.2834   | 00:00.11     | 545,916,084          |
| OPQ64_256,IVF2000_HNSW32,PQ64x4fsr    | 0.14   | 0.2493  | 0.2694  | 0.2755   | 00:00.01     | 547,695,892          |
| OPQ64_256,IVF5000_HNSW32,PQ64x4fsr    | 0.1433 | 0.247   | 0.2717  | 0.2714   | 00:00.01     | 553,176,884          |
| OPQ128_512,IVF1000_HNSW32,PQ128x4fsr  | 0.2867 | 0.368   | 0.3849  | 0.3829   | 00:00.01     | 982,815,924          |
| OPQ128_512,IVF2000_HNSW32,PQ128x4fsr  | 0.2967 | 0.3433  | 0.3653  | 0.3656   | 00:00.01     | 986,153,236          |
| OPQ128_512,IVF5000_HNSW32,PQ128x4fsr  | 0.25   | 0.3233  | 0.3409  | 0.3340   | 00:00.01     | 996,253,492          |
| OPQ256_1024,IVF1000_HNSW32,PQ256x4fsr | 0.4067 | 0.4643  | 0.4737  | 0.4730   | 00:00.03     | 1,856,656,564        |
| OPQ256_1024,IVF2000_HNSW32,PQ256x4fsr | 0.3833 | 0.438   | 0.4435  | 0.4390   | 00:00.02     | 1,862,908,180        |
| OPQ256_1024,IVF5000_HNSW32,PQ256x4fsr | 0.3467 | 0.388   | 0.3802  | 0.3666   | 00:00.02     | 1,882,214,196        |
| HNSW16_SQ4                            | 0.5933 | 0.661   | 0.5829  | 0.4872   | 00:00.11     | 4,570,780,394        |
| HNSW16_PQ12                           | 0.0    | 0.0037  | 0.0055  | 0.0050   | 00:00.01     | 2,124,506,683        |
| HNSW16_PQ24                           | 0.0    | 0.002   | 0.0021  | 0.0021   | 00:00.01     | 2,287,617,607        |
| HNSW32_PQ12                           | 0.0333 | 0.0733  | 0.1001  | 0.1097   | 00:00.02     | 3,862,425,891        |
| HNSW32_PQ24                           | 0.24   | 0.3193  | 0.353   | 0.3486   | 00:00.04     | 4,025,536,815        |

The index we chose is PQ128 because it was a good compromise between accuracy, speed and size on memory.

### Benchmark on MMLU  

### Train LLM

