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


def scroll_pages(file):
    while True:
        page = extract_next_page(file)
        if page is None:
            break
        yield page
