import os

from tqdm import tqdm
from transformers import AutoTokenizer
import json

from backend.vector_database.dataset import Dataset
from backend.vector_database.embedder_wrapper import EmbedderWrapper

DB_DIR_DEFAULT = "scripts/dataset/data"
DB_NAME_DEFAULT = "dataset"
model_path = "BAAI/bge-small-en-v1.5"

COUNT = 1000
DUMP = 200_000
curr_d = DUMP

dataset = Dataset(db_path=os.path.join(DB_DIR_DEFAULT, DB_NAME_DEFAULT + ".db"))

tokenizer = AutoTokenizer.from_pretrained(model_path)

total_chunks = dataset.count_of_chunks()

count = 0
report = {
    "total": 0,
    "n_long_chunks": 0,
    "lengths": [],
}

total_processed = 0
with tqdm(total=total_chunks) as pbar:
    for chunks in dataset.paginate_chunks(COUNT):
        total_processed += len(chunks)

        report["total"] = total_processed

        input_texts = [chunk["text"] for chunk in chunks]

        batch_dict = tokenizer(
            input_texts,
            max_length=8192,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        length = batch_dict["input_ids"].shape[1]

        if length > 512:
            print(f"Chunk with length {length} found")
            report["n_long_chunks"] += 1
            report["lengths"].append(length)

        if total_processed >= curr_d:
            with open("scripts/embeddings/report_length_chunks.json", "w") as f:
                json.dump(report, f)

            curr_d += DUMP

        pbar.update(len(chunks))

with open("scripts/embeddings/report_length_chunks.json", "w") as f:
    json.dump(report, f)
