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
CENTROIDS_DEFAULT = 5  # 131072
DEVICE = "cpu"
KNN_NEIGHBORS_DEFAULT = 100
MMLU_SAMPLE_SIZE_DEFAULT = 10
TRAIN_ON_GPU_DEFAULT = False
TRAINING_SIZE_DEFAULT = 0.2
NPROBE_DEFAULT = 10


def build_mmlu_embds(mmlu_sample_size):
    """Builds the embeddings for the MMLU dataset."""
    dataset = load_mmlu(split="test", subset="stem")
    embedder = EmbedderWrapper(DEVICE)

    questions = [x["question"] for x in dataset]

    embd_list = []
    for i in range(min(mmlu_sample_size, len(questions))):
        embd_list.append(embedder.get_embedding(questions[i]))

    return torch.cat(embd_list).numpy()


def build_baselines(mmlu_embds: np.ndarray, knn_neighbors: int):
    """Builds the KNN baselines for the MMLU dataset."""
    Ds = []
    Is = []
    for offset, embeddings in enumerate(embeddings_iterator(INPUT_DIR_DEFAULT, DEVICE)):
        D, I = faiss.knn(
            mmlu_embds,
            embeddings.numpy(),
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
    train_on_gpu: bool,
    output_dir: str,
    nprobe: int,
    n_neighbors: int,
):
    results = {}

    vector_db = train_vector_db(
        index_str=index_str,
        input_dir="scripts/embeddings/data/",
        training_size=training_size,
        train_on_gpu=train_on_gpu,
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
    results["index_str"] = {
        f"rank_{rank}": knn_intersection_measure(I[:, :rank], I_base[:, :rank])
        for rank in [1, 10, 50, 100]
    }

    # measure the elapsed time
    elapsed_time = end - start
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    results["elapsed_time"] = f"{int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}"

    # save the size of the index on disk
    results["size_on_disk"] = size_on_disk

    results["nprobe"] = nprobe
    results["training_size"] = training_size
    results["train_on_gpu"] = train_on_gpu
    results["mmul_sample_size"] = mmlu_embds.shape[0]
    # save checkpoint of results
    with open(
        os.path.join(output_dir, "bench_quantizer.json"), "a", encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=4)
        f.write(",\n")

    del vector_db


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--centroids",
        type=int,
        default=CENTROIDS_DEFAULT,
        help="Number of centroids for the IVF quantizer",
    )
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
        "--train_on_gpu",
        type=bool,
        default=TRAIN_ON_GPU_DEFAULT,
        help="Whether to train on GPU",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR_DEFAULT,
        help="Location of the directory where to store the output files",
    )

    args = parser.parse_args()

    centroids = args.centroids

    os.makedirs(args.output_dir, exist_ok=True)

    print("Building mmlu embeddings")
    mmlu_embds = build_mmlu_embds(args.mmlu_sample_size)

    # random embeddings
    # mmlu_embds = np.random.rand(args.mmlu_sample_size, 384).astype(np.float32)

    print("Building mmlu baselines")
    D_base, I_base = build_baselines(mmlu_embds, args.knn_neighbors)

    # benchmark with scalar quantizers
    # for sq_type in ["SQ4"]:
    #     index_str = f"IVF{centroids}_HNSW32,{sq_type}"
    #     benchmark(
    #         index_str,
    #         mmlu_embds,
    #         I_base,
    #         args.training_size,
    #         args.train_on_gpu,
    #         args.output_dir,
    #         args.nprobe
    #     )

    for M in [16, 32, 64, 128]:
        index_str = f"IVF{centroids},PQ{M}x4fsr"
        benchmark(
            index_str,
            mmlu_embds,
            I_base,
            args.training_size,
            args.train_on_gpu,
            args.output_dir,
            args.nprobe,
            args.knn_neighbors,
        )

    exit()
    # benchmark with product quantizers
    for M in [16, 32, 64, 128]:
        index_str = f"OPQ{M}_{M * 4},IVF{centroids}_HNSW32,PQ{M}"
        benchmark(
            index_str,
            mmlu_embds,
            I_base,
            args.training_size,
            args.train_on_gpu,
            args.output_dir,
            args.nprobe,
            args.knn_neighbors,
        )

    # benchmark with product quantizers (fast scan)
    for M in [64, 128, 256]:
        index_str = f"OPQ{M}_{M * 4},IVF{centroids}_HNSW32,PQ{M}x4fsr"
        benchmark(
            index_str,
            mmlu_embds,
            I_base,
            args.training_size,
            args.train_on_gpu,
            args.output_dir,
            args.nprobe,
            args.knn_neighbors,
        )

    return


if __name__ == "__main__":
    main()
