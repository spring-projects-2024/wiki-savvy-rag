import re
from backend.data_cleaning.utils import scroll_pages, extract_tag, prepare_for_disk
import json
from tqdm import tqdm


regex = r"^\s*(={2,10}).*\1\s*$"
regex = re.compile(regex, re.MULTILINE)
input_file = "wikidump_processing/data/subsample_cleaner.xml"
output_file = "wikidump_processing/data/subsample_chunked.xml"
N_PAGES = 2357969
SHORT_TEXT_LENGTH = 50

# md5 of output: 8b602ef6d2a6f356e70c4a15d9e49382


def get_list_of_titles(stack):
    return [title for title, _ in stack]


def extract_tree(page):
    title = extract_tag(page, "title", False).strip()

    # remove meta Wikipedia pages (e.g. https://en.wikipedia.org/wiki/Wikipedia:Help_desk/Archive_44)
    if title[:10] == "Wikipedia:" or title[:9] == "Template:":
        return None

    page = extract_tag(page, "text", False)
    stack = []
    chunks = []
    matches = regex.finditer(page)

    stack.append((title, 1))

    last_end = 0

    # if the text is a redirect, we can ignore the page
    if page.lower().find("#redirect") != -1:
        return None

    for match in matches:
        level = match.group().count("=") // 2
        title = match.group().strip()[level:-level].strip()
        text = page[last_end : match.start()].strip()

        if text.lower().find("#redirect") != -1:
            raise 1

        if len(text) > SHORT_TEXT_LENGTH:
            chunks.append({"titles": get_list_of_titles(stack), "text": text})

        last_end = match.end()

        while len(stack) != 0 and stack[-1][1] >= level:
            stack.pop()
        stack.append((title, level))

    text = page[last_end:].strip()

    if len(text) > SHORT_TEXT_LENGTH:
        chunks.append({"titles": get_list_of_titles(stack), "text": text})

    if text.lower().find("#redirect") != -1:
        raise 1

    if len(chunks) == 0:
        return None

    return chunks


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete_old_files", type=bool, default=False, help="Delete old files after creating new ones, to save storage")
    args = parser.parse_args()

    with open(input_file, "r") as f:
        with open(output_file, "w") as out:
            for page in tqdm(scroll_pages(f), total=N_PAGES):
                chunk = extract_tree(page)
                if chunk is not None:
                    p = prepare_for_disk(chunk)
                    out.write(p)

    if args.delete_old_files:
        import os
        os.remove(input_file)


# # example of how to read the data
# dump_path = "../subsample_chunked.xml"
# with open(dump_path, "r") as f:
#     for page in scroll_pages(f):
#         page = extract_tag(page, tag="page", add_tag=False)
#         page = json.loads(page)
#         for chunk in page:
#             print(chunk["text"])
#             print(chunk["titles"])
#             break
#         break
