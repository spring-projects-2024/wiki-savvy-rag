import json

from embeddings import EmbedderWrapper
from wikipedia_data import utils

import utils as utils_retriever

input_file = "../wikipedia_data/subsample_chunked.xml"
output_file = "output.json"

if __name__ == '__main__':

    embedder = EmbedderWrapper()

    with open(input_file, "r") as f:
        with open(output_file, "w") as g:
            for raw_page in utils.scroll_pages(f):
                page = utils.get_extracted_page_chunks(raw_page)

                for chunk in page:
                    # todo: check conversion to list
                    chunk["embedding"] = embedder.get_embedding(
                        utils_retriever.construct_text_from_chunk(**chunk)).flatten().tolist()

                g.write("<page>/n" + json.dumps(page) + "</page>\n")
                exit()
