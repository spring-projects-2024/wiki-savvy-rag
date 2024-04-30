from utils import extract_xml_tags, scroll_pages
from tqdm import tqdm


input_file = "subsample_clean.xml"
output_file = "subsample_cleaner.xml"
N_PAGES = 2357969

# md5 of output: a869504f6252a9e59f23f183cebaca8e


if __name__ == "__main__":
    with open(output_file, "w") as out:
        with open(input_file, "r") as f:
            for page in tqdm(scroll_pages(f), total=N_PAGES):
                out.write(extract_xml_tags(page))
