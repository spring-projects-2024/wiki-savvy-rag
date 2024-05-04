import json
from typing import List

import torch
from tqdm import tqdm

from embeddings import EmbedderWrapper
from backend.data_cleaning import utils

input_file = "../wikipedia_data/subsample_chunkeder.xml"
output_file = "../wikipedia_data/embeddings"

if __name__ == "__main__":

    embedder = EmbedderWrapper("cpu")

    MAX_ACCUMULATION = 300_000
    file_counter = 0
    acc_embeds_list: List = []
    running_sum = 0

    with open(input_file, "r") as f:
        for raw_page in tqdm(utils.scroll_pages(f), total=utils.N_PAGES):

            page = utils.get_extracted_page_chunks(raw_page)

            input_texts = [chunk["text"] for chunk in page]
            embeddings: torch.Tensor = embedder.get_embedding(input_texts)

            acc_embeds_list.append(embeddings)
            running_sum += embeddings.shape[0]

            if running_sum > MAX_ACCUMULATION:

                stacked_embeddings = torch.cat(acc_embeds_list, dim=0)

                torch.save(stacked_embeddings, output_file + f"{file_counter}.pt")
                file_counter += 1
                running_sum = 0
                acc_embeds_list = []
