import os
import re
import time
from typing import Generator

import torch
import numpy as np
from tqdm import tqdm

from backend.vector_database.faiss_wrapper import FaissWrapper

INPUT_FILE_REGEX = "embeddings_[a-z]+.pt"


def embeddings_iterator(
    input_dir: str, device: str
) -> Generator[torch.Tensor, None, None]:
    """Iterates over the embeddings files in the input directory.
    :param input_dir: the directory containing the embeddings files
    :return: a generator of embeddings"""

    file_regex = re.compile(INPUT_FILE_REGEX)

    files = os.listdir(input_dir)
    files.sort()
    for file in files:
        if file_regex.match(file):
            # pass device to load
            embeddings = torch.load(os.path.join(input_dir, file), map_location=device)
            # embeddings = torch.load(os.path.join(input_dir, file))
            yield embeddings


def train_vector_db(
    index_str: str,
    input_dir: str,
    nprobe: int,
    device: str,
    training_size: float,
):
    """Trains a vector database with the given configuration.
    :param index_str: the index factory string
    :param input_dir: the directory containing the embeddings files
    :param nprobe: the number of probes for the IVF quantizer
    :param device: the device to use for the embedder
    :param training_size: the fraction of the data to use for training"""

    print(f"Initializing index {index_str}")

    vector_db = FaissWrapper(
        device=device,
        dataset=None,
        index_str=index_str,
        embedder=None,
        nprobe=nprobe,
    )

    print("Initiated training of vector database with the following configuration:")
    print(f"Index: {index_str}")
    print(f"Training size: {training_size}")
    print(f"Input directory: {input_dir}")
    print("\n")

    training_set = []
    for embeddings in embeddings_iterator(input_dir, device):
        indices = torch.randperm(embeddings.size(dim=0))[
            : int(embeddings.size(dim=0) * training_size)
        ]
        training_set.append(embeddings.cpu().numpy()[indices])

    training_set = np.concatenate(training_set)

    print(
        f"Collected {len(training_set)} samples for training.\nStarting the training of the index..."
    )

    # print start time
    start = time.time()
    print("Start: ", start)

    vector_db.train_from_vectors(training_set)

    end = time.time()
    print("Training done!")
    print("End: ", end)
    print("Elapsed time: ", end - start)

    print("\nAdding vectors to the index...")
    start = time.time()
    print("Start: ", start)

    for embeddings in tqdm(embeddings_iterator(input_dir, device), total=272):
        vector_db.add_vectors(embeddings)

    end = time.time()
    print("Adding done!")
    print("End: ", end)
    print("Elapsed time: ", end - start)

    return vector_db
