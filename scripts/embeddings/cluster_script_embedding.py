from typing import List

import torch
from tqdm import tqdm

from backend.vector_database.embedder_wrapper import EmbedderWrapper
from backend.data_cleaning import utils

INPUT_FILE = "wikidump_processing/data/subsample_chunkeder.xml"
OUTPUT_FILE = "scripts/embeddings/data/embeddings"
PAGES_TO_PROCESS = 2
MAX_ACCUMULATION = 1


def dump_embeddings(acc_embeds_list, filename):
    stacked_embeddings = torch.cat(acc_embeds_list, dim=0)

    torch.save(stacked_embeddings, filename)


if __name__ == "__main__":

    embedder = EmbedderWrapper("cpu")
    file_counter = 0
    acc_embeds_list: List = []
    running_sum = 0

    with open(INPUT_FILE, "r") as f:
        count = 0
        for raw_page in tqdm(utils.scroll_pages(f), total=utils.N_PAGES):

            page = utils.get_extracted_page_chunks(raw_page)

            input_texts = [chunk["text"] for chunk in page]
            embeddings: torch.Tensor = embedder.get_embedding(input_texts)

            acc_embeds_list.append(embeddings)
            running_sum += embeddings.shape[0]

            if running_sum > MAX_ACCUMULATION:
                dump_embeddings(
                    acc_embeds_list, filename=OUTPUT_FILE + f"{file_counter}.pt"
                )
                file_counter += 1
                running_sum = 0
                acc_embeds_list = []

            count += 1

            if count >= PAGES_TO_PROCESS:
                break

    if len(acc_embeds_list) > 0:
        dump_embeddings(acc_embeds_list, filename=OUTPUT_FILE + f"{file_counter}.pt")
