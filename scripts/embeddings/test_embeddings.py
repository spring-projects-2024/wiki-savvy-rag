import os
import re
import torch
from backend.data_cleaning import utils
from backend.vector_database.dataset import DatasetSQL
from backend.vector_database.embedder_wrapper import EmbedderWrapper


DB_DIR = "scripts/dataset/data"
DB_NAME = "dataset"
INPUT_DIR = "scripts/embeddings/data/"
INPUT_FILE_REGEX = "embeddings_[a-z]+.pt"

# This script loads the embeddings files and checks if the embeddings are correct.
# It is used to check that the output of compute_embeddings.py is correct.
# Should run this script only when the embeddings files are small

if __name__ == "__main__":
    file_regex = re.compile(INPUT_FILE_REGEX)

    embeddings = []
    files = os.listdir(INPUT_DIR)
    files.sort()
    for file in files:
        if file_regex.match(file):
            embedding = torch.load(os.path.join(INPUT_DIR, file))
            embeddings.append(embedding)

    embeddings = torch.cat(embeddings, dim=0)
    print(embeddings)

    embedder = EmbedderWrapper("cpu")
    dataset = DatasetSQL(db_path=os.path.join(DB_DIR, DB_NAME + ".db"))
    assert torch.equal(
        embeddings[2], embedder.get_embedding(dataset.search_chunk(2)["text"])[0]
    )
