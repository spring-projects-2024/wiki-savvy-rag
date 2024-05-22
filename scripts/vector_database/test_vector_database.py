import os
from backend.vector_database.dataset import DatasetSQL
from backend.vector_database.embedder_wrapper import EmbedderWrapper
from backend.vector_database.faiss_wrapper import FaissWrapper

DB_DIR = "scripts/dataset/data"
DB_NAME = "dataset"
INDEX_PATH = "scripts/vector_database/data/default.index"

# This script loads the embeddings files and checks if the embeddings are correct.
# It is used to check that the output of train_vector_database.py is correct.
# Should run this script only when the index file small

if __name__ == "__main__":
    dataset = DatasetSQL(db_path=os.path.join(DB_DIR, DB_NAME + ".db"))
    embedder = EmbedderWrapper("cpu")
    vector_db = FaissWrapper(
        dataset=dataset,
        embedder=embedder,
        device="cpu",
        index_path=INDEX_PATH,
    )

    # ok this is impressive
    print(vector_db.search_text("What is Anarchism?")[0])
    print(vector_db.search_text("What women find attractive in men?")[0])
    print(
        vector_db.search_text(
            "What party did the 16ht president of America belong to?"
        )[0]
    )
