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


def greedy_replace(s, opening_tag, opening_delimiter, closing_delimiter):
    """
    I could not find a regex that would work for this, so I wrote this function.
    """
    i = s.find(opening_tag)
    if i == -1:
        return s, False
    if s[i:i + len(opening_tag)] == opening_tag:
        opened = 0
        for j in range(i, len(s)):
            if s[j] == opening_delimiter:
                opened += 1
            if s[j] == closing_delimiter:
                opened -= 1
            if opened == 0:
                break

        s = s[:i] + s[j + 1:]
        return s, True

    raise ValueError("This should not happen")
