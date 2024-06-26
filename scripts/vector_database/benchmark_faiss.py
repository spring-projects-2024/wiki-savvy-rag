import json
import os

import torch
import time
import argparse

import faiss
import numpy as np
from faiss.contrib.evaluation import knn_intersection_measure

from backend.benchmark.utils import load_mmlu
from backend.vector_database.embedder_wrapper import EmbedderWrapper
from scripts.vector_database.utils import embeddings_iterator, train_vector_db


INPUT_DIR_DEFAULT = "scripts/embeddings/data/"
OUTPUT_DIR_DEFAULT = "scripts/vector_database/data/"
DEVICE = "cpu"
KNN_NEIGHBORS_DEFAULT = 100
MMLU_SAMPLE_SIZE_DEFAULT = 10
TRAINING_SIZE_DEFAULT = 0.2
NPROBE_DEFAULT = 10

# This script benchmarks the performance of different indexes.
# In particular, it measures the recall of the indexes when compared to the brute force search,
# the time taken to run all the search queries and the size of the index on disk (which is a proxy
# for the memory usage).
#
# The command line arguments are:
# --knn_neighbors: Max number of neighbors for computing recall
# --nprobe: Number of probes for the IVF quantizer
# --training_size: Fraction of the dataset to use for training
# --mmlu_sample_size: Number of samples to use from the MMLU dataset for the benchmarks
# --output_dir: Location of the directory where to store the output files


def build_mmlu_embds(mmlu_sample_size):
    """Builds the embeddings for the MMLU dataset."""
    dataset = load_mmlu(split="test", subset="stem")
    embedder = EmbedderWrapper(DEVICE)

    questions = [x["question"] for x in dataset]

    embd_list = []
    for i in range(min(mmlu_sample_size, len(questions))):
        embd_list.append(embedder.get_embedding(questions[i]))

    return torch.cat(embd_list).cpu().numpy()


def build_baselines(mmlu_embds: np.ndarray, knn_neighbors: int):
    """Builds the KNN baselines for the MMLU dataset."""
    Ds = []
    Is = []
    for offset, embeddings in enumerate(embeddings_iterator(INPUT_DIR_DEFAULT, DEVICE)):
        D, I = faiss.knn(
            mmlu_embds,
            embeddings.cpu().numpy(),
            min(embeddings.shape[0], knn_neighbors),
            metric=faiss.METRIC_INNER_PRODUCT,
        )

        Ds.append(D)
        Is.append(I + offset * len(embeddings))

    return faiss.merge_knn_results(np.stack(Ds), np.stack(Is), keep_max=True)


def benchmark(
    index_str: str,
    mmlu_embds: np.ndarray,
    I_base: np.ndarray,
    training_size: float,
    output_dir: str,
    nprobe: int,
    n_neighbors: int,
):
    """Runs the benchmark for a given index."""

    results = {}

    vector_db = train_vector_db(
        index_str=index_str,
        input_dir="scripts/embeddings/data/",
        training_size=training_size,
        nprobe=nprobe,
        device=DEVICE,
    )

    # measure the size of the index on disk
    dump_path = os.path.join(output_dir, index_str.replace(",", "_") + ".index")
    vector_db.save_to_disk(dump_path)
    size_on_disk = os.path.getsize(dump_path)
    os.remove(dump_path)

    start = time.time()
    _, I = vector_db.search_vectors(mmlu_embds, n_neighbors=n_neighbors)
    end = time.time()

    # measure the knn intersection measure
    results["index"] = index_str
    results["ranks"] = {
        f"rank_{rank}": knn_intersection_measure(I[:, :rank], I_base[:, :rank])
        for rank in [1, 10, 50, 100]
        if rank <= n_neighbors
    }

    # measure the elapsed time
    elapsed_time = end - start
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    results["elapsed_time"] = f"{int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}"

    # save the size of the index on disk
    results["size_on_disk"] = size_on_disk

    results["mmlu_sample_size"] = mmlu_embds.shape[0]
    results["nprobe"] = nprobe
    results["training_size"] = training_size
    results["mmlu_sample_size"] = mmlu_embds.shape[0]

    # save checkpoint of results
    with open(
        os.path.join(output_dir, "benchmark_faiss.json"), "a", encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=4)
        f.write(",\n")

    del vector_db


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--knn_neighbors",
        type=int,
        default=KNN_NEIGHBORS_DEFAULT,
        help="Max number of neighbors for computing recall",
    )
    parser.add_argument(
        "--nprobe",
        type=int,
        default=NPROBE_DEFAULT,
        help="Number of probes for the IVF quantizer",
    )
    parser.add_argument(
        "--training_size",
        type=float,
        default=TRAINING_SIZE_DEFAULT,
        help="Fraction of the dataset to use for training",
    )
    parser.add_argument(
        "--mmlu_sample_size",
        type=int,
        default=MMLU_SAMPLE_SIZE_DEFAULT,
        help="Number of samples to use from the MMLU dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR_DEFAULT,
        help="Location of the directory where to store the output files",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Building mmlu embeddings")
    emb_path = os.path.join(args.output_dir, f"mmlu_embds_{args.mmlu_sample_size}.npy")
    if not os.path.exists(emb_path):
        mmlu_embds = build_mmlu_embds(args.mmlu_sample_size)
        # dump the embeddings to disk
        np.save(emb_path, mmlu_embds)
    else:
        mmlu_embds = np.load(
            os.path.join(args.output_dir, f"mmlu_embds_{args.mmlu_sample_size}.npy")
        )

    print("Building mmlu baselines")

    d_path = os.path.join(
        args.output_dir, f"mmlu_baselines_{args.mmlu_sample_size}_D.npy"
    )
    i_path = os.path.join(
        args.output_dir, f"mmlu_baselines_{args.mmlu_sample_size}_I.npy"
    )

    if os.path.exists(d_path) and os.path.exists(i_path):
        D_base = np.load(d_path)
        I_base = np.load(i_path)
    else:
        D_base, I_base = build_baselines(mmlu_embds, args.knn_neighbors)
        np.save(d_path, D_base)
        np.save(i_path, I_base)

    # scalar quantizer
    for sq in [4, 8]:
        index_str = f"SQ{sq}"
        benchmark(
            index_str,
            mmlu_embds,
            I_base,
            args.training_size,
            args.output_dir,
            args.nprobe,
            args.knn_neighbors,
        )

    # product quantizer
    for pq in [64, 128]:
        index_str = f"PQ{pq}"
        benchmark(
            index_str,
            mmlu_embds,
            I_base,
            args.training_size,
            args.output_dir,
            args.nprobe,
            args.knn_neighbors,
        )

    # hierarchical indexes, product quantizer
    for nn in [16, 32]:
        for pq in [12, 24]:
            index_str = f"HNSW{nn}_PQ{pq}"
            benchmark(
                index_str,
                mmlu_embds,
                I_base,
                args.training_size,
                args.output_dir,
                args.nprobe,
                args.knn_neighbors,
            )

    # hierarchical indexes, scalar quantizer
    for sq in [4]:
        for sw_size in [16]:
            index_str = f"HNSW{sw_size}_SQ{sq}"
            benchmark(
                index_str,
                mmlu_embds,
                I_base,
                args.training_size,
                args.output_dir,
                args.nprobe,
                args.knn_neighbors,
            )

    # IVF indexes, product quantizer
    for sq in [64, 128, 256]:
        for centroids in [1000, 2000, 5000]:
            index_str = f"OPQ{sq}_{sq * 4},IVF{centroids}_HNSW32,PQ{sq}x4fsr"
            benchmark(
                index_str,
                mmlu_embds,
                I_base,
                args.training_size,
                args.output_dir,
                args.nprobe,
                args.knn_neighbors,
            )


if __name__ == "__main__":
    main()
