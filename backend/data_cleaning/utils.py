import json
from typing import Dict, List

N_PAGES = 2357969


def prepare_for_disk(chunks):
    s = "<page>\n"
    s += json.dumps(chunks)
    s += "\n</page>\n"
    return s


def construct_text_from_chunk(titles: List, text: str):
    s = ""
    for title in titles:
        s += title + "\n"
    s += "\n"
    s += text
    return s


def extract_next_page(f):
    """
    Extract the next page from an xml file.
    pages are separated by <page> and </page> tags.
    """
    page = ""
    for line in f:
        if line.strip() == "<page>":
            page = ""
        page += line
        if line.strip() == "</page>":
            return page
    return None


def get_title(page):
    """
    Extract the title of a page.
    The title is the text between the <title> and </title> tags.
    """
    title = page.split("<title>")[1].split("</title>")[0]
    return title


def get_categories(page):
    """
    Extract the categories of a page.
    The categories are the text between the [[Category: and ]] tags.
    """
    categories = []
    for category in page.split("[[Category:")[1:]:
        category = category.split("]]")[0]
        categories.append(category)
    return categories


def scroll_pages(file):
    while True:
        page = extract_next_page(file)
        if page is None:
            break
        yield page


def remove_square_brackets_around_links(s):
    is_latex = False
    i = 0
    last_start = 0
    first_open = None
    open_counter = 0
    LEN = len(s)
    new_s = ""

    while i < LEN - 1:
        nxt = i + 1
        if s[i] == "<":
            if s[nxt : i + 5] == "math":
                is_latex = True
                i += 5
                continue
            elif s[nxt : i + 6] == "/math":
                # assumes latex is not nested
                is_latex = False
                i += 6
                continue

        if is_latex:
            i += 1
            continue

        if s[i] == "[" and s[nxt] == "[":
            if open_counter == 0:
                first_open = i
                title = None
            open_counter += 1
            i += 2
            continue
        elif s[i] == "|" and open_counter > 0 and title is None:
            title = s[first_open + 2 : i]  # skip the [[
        elif s[i] == "]" and s[nxt] == "]":
            if open_counter <= 0:
                i += 2
                continue
            open_counter -= 1
            if title is None:
                title = s[first_open + 2 : i]
            new_s += s[last_start:first_open] + title
            last_start = i + 2  # skip the ]]
        i += 1

    new_s += s[last_start:]
    return new_s


def remove_template_tags(s):
    is_latex = False
    i = 0
    last_start = 0
    first_open = None
    open_counter = 0
    LEN = len(s)
    new_s = ""

    while i < LEN - 1:
        nxt = i + 1
        if s[i] == "<":
            if s[nxt : i + 5] == "math":
                is_latex = True
                i += 5
                continue
            elif s[nxt : i + 6] == "/math":
                is_latex = False
                i += 6
                continue

        if is_latex:
            i += 1
            continue

        if s[i] == s[nxt] == "{":
            if open_counter == 0:
                first_open = i
            open_counter += 1
            i += 2
            continue
        elif s[i] == s[nxt] == "}":
            if open_counter > 0:
                open_counter -= 1
                if open_counter == 0:
                    new_s += s[last_start:first_open]
                    last_start = i + 2
            i += 2
            continue
        i += 1

    new_s += s[last_start:]
    return new_s


def remove_wiki_tags(s):
    is_template = False
    open_counter = 0

    i = 0
    LEN = len(s)
    new_s = ""

    last_start = 0
    first_open = None

    while i < LEN - 1:
        nxt = i + 1
        if s[i] == s[nxt] == "[":
            if open_counter == 0:
                is_template = False
                first_open = i
                open_counter += 1
                i += 2

                while s[i].isspace():  # skip spaces
                    i += 1
                while ord("a") <= ord(s[i]) <= ord("z") or ord("A") <= ord(s[i]) <= ord(
                    "Z"
                ):
                    i += 1
                while s[i].isspace():  # skip spaces
                    i += 1

                if s[i] == ":":
                    is_template = True
                    i += 1
            else:
                open_counter += 1
                i += 2
        elif s[i] == s[nxt] == "]":
            if open_counter <= 0:
                i += 2
                continue

            open_counter -= 1
            if open_counter == 0 and is_template:
                is_template = False
                new_s += s[last_start:first_open]
                last_start = i + 2

            i += 2
        else:
            i += 1

    new_s += s[last_start:]
    return new_s


def remove_table_tags(s):
    is_latex = False
    i = 0
    last_start = 0
    first_open = None
    open_counter = 0
    LEN = len(s)
    new_s = ""

    while i < LEN - 1:
        nxt = i + 1
        if s[i] == "<":
            if s[nxt : i + 5] == "math":
                is_latex = True
                i += 5
                continue
            elif s[nxt : i + 6] == "/math":
                is_latex = False
                i += 6
                continue

        if is_latex:
            i += 1
            continue

        if s[i] == "{" and s[nxt] == "|":
            if open_counter == 0:
                first_open = i
            open_counter += 1
            i += 2
            continue
        elif s[i] == "|" and s[nxt] == "}":
            if open_counter > 0:
                open_counter -= 1
                if open_counter == 0:
                    new_s += s[last_start:first_open]
                    last_start = i + 2
            i += 2
            continue
        i += 1

    new_s += s[last_start:]
    return new_s


def get_paragraph(page: str):
    for l in page.splitlines():
        pass


TAGS_TO_KEEP = [
    "title",
    "text",
]


def get_extracted_page_chunks(page) -> List[Dict]:
    page = extract_tag(page, tag="page", add_tag=False)
    return json.loads(page)


def extract_tag(page, tag, add_tag=True):
    INITIAL_TAG = f"<{tag}"
    FINAL_TAG = f"</{tag}>"

    page = page[page.find(INITIAL_TAG) :]
    page = page[page.find(">") + 1 :]

    if tag == "text" or tag == "page":
        page = page[: page.rfind(FINAL_TAG)]
    elif tag == "title":
        page = page[: page.find(FINAL_TAG)]
    else:
        raise ValueError(f"Tag {tag} not recognized")

    if add_tag:
        return INITIAL_TAG + ">\n" + page + "\n" + FINAL_TAG + "\n"
    else:
        return page


def extract_xml_tags(page: str):
    title = extract_tag(page, "title")
    s = "<page>\n" + title + extract_tag(page, "text") + "</page>"
    return s + "\n"
