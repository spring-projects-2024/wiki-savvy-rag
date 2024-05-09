import os
import torch
import re
import time

from backend.vector_database.faiss_wrapper import FaissWrapper

INPUT_FILE_REGEX = "embeddings_[a-z]+.pt"


def embeddings_iterator(input_dir: str):
    file_regex = re.compile(INPUT_FILE_REGEX)

    files = os.listdir(input_dir)
    files.sort()
    for file in files:
        if file_regex.match(file):
            embeddings = torch.load(os.path.join(input_dir, file))
            yield embeddings


def train_vector_db(
    index_str: str,
    input_dir: str,
    training_size: float,
    train_on_gpu: bool = True,
) -> FaissWrapper:
    vector_db = FaissWrapper(
        device="cpu",
        dataset=None,
        index_str=index_str,
    )

    print(
        f"Initiated training of vector database with the following configuration: \n\n"
    )
    print(f"Index: {index_str}")
    print(f"Training size: {training_size}")
    print(f"Input directory: {input_dir}")
    print("\n\n")

    training_set = torch.Tensor()
    for embeddings in embeddings_iterator(input_dir):
        indices = torch.randperm(embeddings.size(dim=0))[
            : int(len(embeddings) * training_size)
        ]
        training_set = torch.cat((training_set, (embeddings[indices])), dim=0)

    print(
        f"Collected {len(training_set)} samples for training.\nStarting the training of the index..."
    )

    # print start time
    start = time.time()
    print("Start: ", start)

    vector_db.train_from_vectors(training_set, train_on_gpu=train_on_gpu)

    end = time.time()
    print("Training done!")
    print("End: ", end)
    print("Elapsed time: ", end - start)

    print("\nAdding vectors to the index...")
    start = time.time()
    print("Start: ", start)

    for embeddings in embeddings_iterator():
        vector_db.add_vectors(embeddings)

    end = time.time()
    print("Adding done!")
    print("End: ", end)
    print("Elapsed time: ", end - start)
