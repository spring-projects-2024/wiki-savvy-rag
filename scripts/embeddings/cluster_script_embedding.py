from typing import List

import os
import torch
from tqdm import tqdm

from backend.vector_database.embedder_wrapper import EmbedderWrapper
from backend.data_cleaning import utils
import argparse


INPUT_FILE_DEFAULT = "wikidump_processing/data/subsample_chunkeder.xml"
OUTPUT_DIR_DEFAULT = "scripts/embeddings/data/"
PAGES_TO_PROCESS_DEFAULT = utils.N_PAGES
MAX_ACCUMULATION_DEFAULT = 300_000


def dump_embeddings(acc_embeds_list, filename):
    print("Dump embeddings file: ", filename)
    stacked_embeddings = torch.cat(acc_embeds_list, dim=0)

    torch.save(stacked_embeddings, filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default=INPUT_FILE_DEFAULT,
        help="Location of the subsample_chunkeder.xml file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR_DEFAULT,
        help="Location of the directory where to store the output files",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=PAGES_TO_PROCESS_DEFAULT,
        help="Maximum amount of pages to process",
    )
    parser.add_argument(
        "--max_accumulation",
        type=int,
        default=MAX_ACCUMULATION_DEFAULT,
        help="Maximum number of chunks to keep on memory before dumping the embeddings on file",
    )

    args = parser.parse_args()

    embedder = EmbedderWrapper("cpu")
    file_counter = 0
    acc_embeds_list: List = []
    running_sum = 0

    if not os.path.exists(args.output_dir):
        print("Specified directory doesn't exist. Creating it...")
        os.mkdir(args.output_dir)

    with open(args.input, "r") as f:
        count = 0
        for raw_page in tqdm(utils.scroll_pages(f), total=args.pages):

            page = utils.get_extracted_page_chunks(raw_page)

            input_texts = [chunk["text"] for chunk in page]
            embeddings: torch.Tensor = embedder.get_embedding(input_texts)

            acc_embeds_list.append(embeddings)
            running_sum += embeddings.shape[0]

            if running_sum > args.max_accumulation:
                dump_embeddings(
                    acc_embeds_list,
                    filename=args.output_dir + f"embeddings{file_counter}.pt",
                )
                file_counter += 1
                running_sum = 0
                acc_embeds_list = []

            count += 1

            if count >= args.pages:
                break

    if len(acc_embeds_list) > 0:
        dump_embeddings(
            acc_embeds_list, filename=args.output_dir + f"{file_counter}.pt"
        )
