import json
import os
import torch
from time import time

import faiss
import numpy as np
from faiss.contrib.evaluation import knn_intersection_measure

from backend.benchmark.utils import load_mmlu
from backend.vector_database.embedder_wrapper import EmbedderWrapper
from scripts.vector_database.utils import embeddings_iterator, train_vector_db

INPUT_DIR_DEFAULT = "scripts/embeddings/data/"
OUTPUT_DIR_DEFAULT = "scripts/vector_database/data/"
CENTROIDS = 5  # 131072
DEVICE = "cpu"
KNN_NEIGHBORS = 100
MMLU_SAMPLE_SIZE = 10
TRAIN_ON_GPU = False
TRAINING_SIZE = 0.2

results = {}


def build_mmlu_embds():
    """Builds the embeddings for the MMLU dataset."""
    dataset = load_mmlu(split="test", subset="stem")
    embedder = EmbedderWrapper(DEVICE)

    questions = [x["question"] for x in dataset]

    embd_list = []
    for i in range(min(MMLU_SAMPLE_SIZE, len(questions))):
        embd_list.append(embedder.get_embedding(questions[i]))

    return torch.cat(embd_list).numpy()


def build_baselines(mmlu_embds: np.ndarray):
    """Builds the KNN baselines for the MMLU dataset."""
    Ds = []
    Is = []
    for embeddings in embeddings_iterator(INPUT_DIR_DEFAULT):
        D, I = faiss.knn(
            mmlu_embds,
            embeddings.numpy(),
            min(mmlu_embds.shape[0], KNN_NEIGHBORS),
            metric=faiss.METRIC_INNER_PRODUCT,
        )

        Ds.append(D)
        Is.append(I)

    return faiss.merge_knn_results(np.stack(Ds), np.stack(Is))


def benchmark(
    index_str: str,
    mmlu_embds: np.ndarray,
    I_base: np.ndarray,
):
    vector_db = train_vector_db(
        index_str=index_str,
        input_dir="scripts/embeddings/data/",
        training_size=TRAINING_SIZE,
        train_on_gpu=TRAIN_ON_GPU,
    )

    # measure the size of the index on disk
    dump_path = os.path.join(OUTPUT_DIR_DEFAULT, index_str.replace(",", "_") + ".index")
    vector_db.save_to_disk(dump_path)
    size_on_disk = os.path.getsize(dump_path)
    os.remove(dump_path)

    start = time.time()
    _, I = vector_db.search_vectors(mmlu_embds)
    end = time.time()

    # measure the elapsed time
    elapsed_time = end - start
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    results[index_str][
        elapsed_time
    ] = f"{int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}"

    # measure the knn intersection measure
    results[index_str] = {
        f"rank_{rank}": knn_intersection_measure(I[:, :rank], I_base[:, :rank])
        for rank in [1, 10, 50, 100]
    }

    # save the size of the index on disk
    results[index_str]["size_on_disk"] = size_on_disk

    # save checkpoint of results
    with open(
        os.path.join(OUTPUT_DIR_DEFAULT, "bench_quantizer.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(results, f)

    del vector_db


def main():
    mmlu_embds = build_mmlu_embds()
    _, I_base = build_baselines(mmlu_embds)

    # benchmark with scalar quantizers
    for sq_type in ["SQ8", "SQ4"]:
        index_str = f"IVF{CENTROIDS}_HNSW32,{sq_type}"
        benchmark(index_str, mmlu_embds, I_base)

    # benchmark with product quantizers
    for M in [32, 64, 128]:
        index_str = f"OPQ{M}_{M/4},IVF{CENTROIDS}_HNSW32,PQ{M}"
        benchmark(index_str, mmlu_embds, I_base)

    # benchmark with product quantizers (fast scan)
    for M in [64, 128, 256]:
        index_str = f"OPQ{M}_{M/4},IVF{CENTROIDS}_HNSW32,PQ{M}x4fsr"
        benchmark(index_str, mmlu_embds, I_base)

    return


if __name__ == "__main__":
    main()
