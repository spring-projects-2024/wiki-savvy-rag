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


def greedy_remove_template_tags(s):
    is_latex = False
    i = 0
    last_start = 0
    first_open = None
    open_counter = 0
    LEN = len(s)
    new_s = ""

    while i < LEN - 1:
        nxt= i + 1
        if s[i] == "<":
            if s[nxt:i + 5] == "math":
                is_latex = True
                i += 5
                continue
            elif s[nxt:i + 6] == "/math":
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
        elif open_counter > 0 and s[i] == s[nxt] == "}":
            open_counter -= 1
            if open_counter == 0:

                new_s += s[last_start:first_open]
                last_start = i + 2

            i += 2
            continue

        i += 1

    new_s += s[last_start:]


    return new_s


if __name__ == '__main__':
    prova = "{{1{{2}}}}aa{{<math></math>}} <math>{{a}}</math>"
    p = greedy_remove_template_tags(prova)
    print(p)
