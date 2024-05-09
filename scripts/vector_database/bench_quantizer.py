from scripts.vector_database.utils import train_vector_db

INPUT_DIR_DEFAULT = "scripts/embeddings/data/"
CENTROIDS = 131072

results = {}


def benchmark(index_str):
    vector_db = train_vector_db(
        device="cpu",
        index_str=index_str,
        input_dir="scripts/embeddings/data/",
        training_size=0.2,
        train_on_gpu=True,
    )

    results[index_str] = {}


def main():
    # benchmark with scalar quantizers
    for sq_type in ["SQ8", "SQ4"]:
        index_str = f"IVF{CENTROIDS}_HNSW32,{sq_type}"
        benchmark(index_str)

    # benchmark with product quantizers
    for M in [32, 64, 128]:
        index_str = f"OPQ{M}_{M/4},IVF{CENTROIDS}_HNSW32,PQ{M}"
        benchmark(index_str)

    # benchmark with product quantizers (fast scan)
    for M in [64, 128, 256]:
        index_str = f"OPQ{M}_{M/4},IVF{CENTROIDS}_HNSW32,PQ{M}x4fsr"
        benchmark(index_str)

    return


if __name__ == "__main__":
    main()
