import argparse
import os
import torch
import re

from backend.vector_database.faiss_wrapper import FaissWrapper
from backend.vector_database.embedder_wrapper import EmbedderWrapper

INPUT_DIR_DEFAULT = "scripts/embeddings/data/"
OUTPUT_FILE_DEFAULT = "scripts/vector_database/data/flat.index"
INDEX_DEFAULT = "HNSW,Flat"

INPUT_FILE_REGEX = "embeddings_[a-z]+.pt"


def embeddings_iterator():
    file_regex = re.compile(INPUT_FILE_REGEX)

    files = os.listdir(args.input_dir)
    files.sort()
    for file in files:
        if file_regex.match(file):
            embeddings = torch.load(os.path.join(args.input_dir, file))
            yield embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
        default=INPUT_DIR_DEFAULT,
        help="Location of the directory where the embedding files are located",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_FILE_DEFAULT,
        help="Location where to store the index file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device where to run the embedder",
    )
    parser.add_argument(
        "--index",
        type=str,
        default=INDEX_DEFAULT,
        help="String representing the index that needs to be built",
    )

    args = parser.parse_args()

    vector_db = FaissWrapper(
        device=args.device,
        dataset=None,
        embedder=EmbedderWrapper(args.device),
        index_str=args.index,
    )

    for embeddings in embeddings_iterator():
        vector_db.train_from_vectors(embeddings)

    for embeddings in embeddings_iterator():
        vector_db.add_vectors(embeddings)

    vector_db.save_to_disk(args.output)
