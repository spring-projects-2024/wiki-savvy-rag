import os

from backend.data_cleaning import utils
from backend.vector_database.dataset import DatasetSQL

DB_DIR = "scripts/dataset/data"
DB_NAME = "dataset"

DUMP_PATH = "wikidump_processing/data/subsample_chunkeder.xml"

# This script checks if the dataset is correctly populated and if the search functions work as expected.
# It is used to check that the output of the populate_dataset.py script is correct.

if __name__ == "__main__":
    dataset = DatasetSQL(db_path=os.path.join(DB_DIR, DB_NAME + ".db"))

    print(dataset.search_chunk(2))

    print(dataset.search_chunks([1, 3, 8]))

    chunks = []

    with open(DUMP_PATH, "r") as f:
        page_count = 0
        for raw_page in utils.scroll_pages(f):

            page = utils.get_extracted_page_chunks(raw_page)

            chunks += page

            page_count += 1

            if page_count >= 2:
                break

    assert chunks[2]["text"] == dataset.search_chunk(2)["text"]
