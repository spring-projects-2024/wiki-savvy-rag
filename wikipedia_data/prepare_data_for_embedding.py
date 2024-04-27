import re
from utils import scroll_pages, extract_tag
import json
from tqdm import tqdm


regex = r"^\s*(={2,10}).*\1\s*$"
regex = re.compile(regex, re.MULTILINE)


def get_list_of_titles(stack):
    return [title for title, _ in stack]


def extract_tree(page):
    title = extract_tag(page, "title", False).strip()

    page = extract_tag(page, "text", False)
    stack = []
    chunks = []
    matches = regex.finditer(page)

    stack.append((title, 1))

    last_end = 0

    for match in matches:
        level = match.group().count("=") // 2
        title = match.group().strip()[level:-level].strip()
        text = page[last_end:match.start()].strip()

        if text != "":
            chunks.append({
                "titles": get_list_of_titles(stack),
                "text": text
            })

        last_end = match.end()

        while len(stack) != 0 and stack[-1][1] >= level:
            stack.pop()
        stack.append((title, level))

    text = page[last_end:].strip()

    if text != "":
        chunks.append({
            "titles": get_list_of_titles(stack),
            "text": text
        })

    return chunks


def prepare_for_disk(chunks):
    s = "<page>\n"
    s += json.dumps(chunks)
    s += "\n</page>\n"
    return s


N_PAGES = 2357969

input_file = "subsample_cleaner.xml"
output_file = "subsample_chunked.xml"

if __name__ == '__main__':

    with open(input_file, "r") as f:
        with open(output_file, "a") as out:
            for page in tqdm(scroll_pages(f), total=N_PAGES):
                chunk = extract_tree(page)
                p = prepare_for_disk(chunk)
                out.write(p)
