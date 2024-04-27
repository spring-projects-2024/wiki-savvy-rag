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
    TODO: check that this works
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
        nxt = i + 1
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
            if open_counter <= 0:
                i += 2
                continue
            open_counter -= 1
            if open_counter == 0:
                new_s += s[last_start:first_open]
                last_start = i + 2

            i += 2
            continue

        i += 1

    new_s += s[last_start:]

    return new_s


def greedy_remove_wiki_tags(s):
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

                while ord("a") <= ord(s[i]) <= ord("z") or ord("A") <= ord(s[i]) <= ord("Z"):
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


def greedy_remove_template_tags_table(s):
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

        if s[i] == "{" and s[nxt] == "|":
            if open_counter == 0:
                first_open = i
            open_counter += 1
            i += 2
            continue
        elif open_counter > 0 and s[i] == "|" and s[nxt] == "}":
            if open_counter <= 0:
                i += 2
                continue

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


if __name__ == '__main__':
    import re

    # prova = " alphabet, similar to Turkish.<ref></ref><ref><bdi>[https://www.akorda.kz/ru/legal_acts/decrees/o-perevode-alfavita-kazahskogo-yazyka-s-kirillicy-na-latinskuyu-grafiku О переводе алфавита казахского языка с кириллицы на латинскую графику]</bdi> [On the change of the alphabet of the Kazakh language from the Cyrillic to the Latin script] (in Russian). [[President of the Republic of Kazakhstan]]. 26 October 2017. Archived from the original on 27 October 2017. Retrieved 26 October 2017.</ref> The Cyrillic script used to be official in Uzbekistan and Turkmenistan before they all switched to the Latin alphabet, including Uzbekistan that is having a reform of the alphabet to use diacritics on the letters that are marked by apostrophes and the letters that are digraphs.<ref></ref><ref></ref>"
    # remove_ref_html_tag = r"<(ref|sub).*?/(\1|)>"
    # # remove_ref_html_tag = r"</ref>"
    # remove_html_comment = re.compile(remove_ref_html_tag, re.DOTALL)
    #
    # print(re.sub(remove_html_comment, "", prova))
    a = "a{| ciao {| buongionro |} |}b  a"

    # remove_ref_html_tag = r"<(ref|span|gallery|timeline|imagemap|mapframe|div).*?/(\1|)>"
    #
    # remove_html_comment = re.compile(remove_ref_html_tag, re.DOTALL)

    # a = remove_html_comment.sub("", a)
    print(greedy_remove_template_tags_table(a))