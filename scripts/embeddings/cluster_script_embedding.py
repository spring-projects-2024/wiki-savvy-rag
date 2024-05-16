import itertools
import os
from typing import List
import gc

import torch
from tqdm import tqdm

from backend.vector_database.embedder_wrapper import EmbedderWrapper
from backend.vector_database.dataset import Dataset
import argparse

DB_DIR_DEFAULT = "scripts/dataset/data"
DB_NAME_DEFAULT = "dataset"
OUTPUT_DIR_DEFAULT = "scripts/embeddings/data/"
MAX_ACCUMULATION_DEFAULT = 300_000
MAX_CHUNKS_DEFAULT = 100

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--db_dir",
        type=str,
        default=DB_DIR_DEFAULT,
        help="Directory where the database is located",
    )
    parser.add_argument(
        "--db_name",
        type=str,
        default=DB_NAME_DEFAULT,
        help="Name of the database",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR_DEFAULT,
        help="Location of the directory where to store the output files",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        help="Maximum amount of chunks to process",
        required=False,
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset from the beginning of the dataset",
        required=False,
    )
    parser.add_argument(
        "--max_accumulation",
        type=int,
        default=MAX_ACCUMULATION_DEFAULT,
        help="Maximum number of chunks to keep on memory before dumping the embeddings on file",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device where to run the embedder"
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        print("Specified directory doesn't exist. Creating it...")
        os.mkdir(args.output_dir)

    dataset = Dataset(db_path=os.path.join(args.db_dir, args.db_name + ".db"))
    embedder = EmbedderWrapper(args.device)

    count_of_chunks = dataset.count_of_chunks()
    print("Found a total of ", count_of_chunks, " chunks")

    alphabet = "abcdefghijklmnopqrstuvwxyz"
    length = 10
    iterator = itertools.product(alphabet, repeat=length)

    n_file_skipped = args.offset // args.max_accumulation
    for _ in range(n_file_skipped):
        next(iterator)

    processed_count = 0

    print(f"MAX ACCUMULATION IS {args.max_accumulation}")

    with tqdm(
        total=(min(args.chunks, count_of_chunks) if args.chunks else count_of_chunks)
    ) as pbar:
        for chunks in dataset.paginate_chunks(
            args.max_accumulation, offset=args.offset
        ):

            print(f"N CHUNKS IS {len(chunks)}")
            input_texts = [chunk["text"] for chunk in chunks]
            if args.chunks:
                input_texts = input_texts[
                    : min(len(input_texts), args.chunks - processed_count)
                ]

            embs_list = []
            # for inp_tex in input_texts:
            #     embedding_single: torch.Tensor = embedder.get_embedding(inp_tex)
            #     embs_list.append(embedding_single)
            #
            # embeddings = torch.stack(embs_list)

            embeddings = embedder.get_embedding(input_texts)

            print(f"Embeddings shape {embeddings.shape}")

            filename = os.path.join(
                args.output_dir, f"embeddings_{''.join(next(iterator))}.pt"
            )
            print("Dump embeddings file: ", filename)
            torch.save(embeddings, filename)
            processed_count += len(input_texts)
            pbar.update(len(input_texts))

            del embeddings, input_texts, embs_list
            gc.collect()
            torch.cuda.empty_cache()

            if args.chunks and processed_count >= args.chunks:
                break
