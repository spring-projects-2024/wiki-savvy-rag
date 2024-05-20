import argparse
import os

from tqdm import tqdm
from backend.data_cleaning import utils
from backend.vector_database.dataset import DatasetSQL

INPUT_FILE_DEFAULT = "wikidump_processing/data/subsample_chunkeder.xml"
DB_DIR_DEFAULT = "scripts/dataset/data"
DB_NAME_DEFAULT = "dataset"
PAGES_TO_PROCESS_DEFAULT = utils.N_PAGES

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default=INPUT_FILE_DEFAULT,
        help="Location of the subsample_chunkeder.xml file",
    )
    parser.add_argument(
        "--db_dir",
        type=str,
        default=DB_DIR_DEFAULT,
        help="Directory where the database is/will be located",
    )
    parser.add_argument(
        "--db_name",
        type=str,
        default=DB_NAME_DEFAULT,
        help="Name of the database",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=PAGES_TO_PROCESS_DEFAULT,
        help="Maximum amount of pages to insert in database",
    )

    args = parser.parse_args()

    if not os.path.exists(args.db_dir):
        print("Specified directory doesn't exist. Creating it...")
        os.mkdir(args.db_dir)

    dataset = DatasetSQL(db_path=os.path.join(args.db_dir, args.db_name + ".db"))

    print("Resetting the database...")
    dataset.drop_tables()
    dataset.create_tables()

    with open(args.input, "r") as f:
        page_count = 0
        chunk_count = 0
        for raw_page in tqdm(utils.scroll_pages(f), total=args.pages):

            page = utils.get_extracted_page_chunks(raw_page)

            dataset.insert_chunks(
                {
                    "id": chunk_count + index,
                    "text": chunk["text"],
                    "titles": str(chunk["titles"]),
                }
                for index, chunk in enumerate(page)
            )

            chunk_count += len(page)
            page_count += 1

            if page_count >= args.pages:
                break
