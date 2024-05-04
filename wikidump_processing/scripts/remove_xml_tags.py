from backend.data_cleaning.utils import extract_xml_tags, scroll_pages
from tqdm import tqdm


input_file = "wikidump_processing/data/subsample_clean.xml"
output_file = "wikidump_processing/data/subsample_cleaner.xml"
N_PAGES = 2357969

# md5 of output: a869504f6252a9e59f23f183cebaca8e


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete_old_files", type=bool, default=False, help="Delete old files after creating new ones, to save storage")
    args = parser.parse_args()

    with open(output_file, "w") as out:
        with open(input_file, "r") as f:
            for page in tqdm(scroll_pages(f), total=N_PAGES):
                out.write(extract_xml_tags(page))

    if args.delete_old_files:
        import os
        os.remove(input_file)