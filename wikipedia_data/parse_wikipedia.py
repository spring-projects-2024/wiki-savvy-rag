import json

dump_path = "../../enwiki-20231220-pages-articles-multistream.xml"
json_path = "selected_categories.json"

with open(json_path, "r") as f:
    selected_categories = json.load(f)
selected_categories = set(selected_categories)


def extract_next_page(f):
    """
    Extract the next page from an xml file.
    pages are separated by <page> and </page> tags.
    """
    page = ""
    print_ = False
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


def scroll_pages(stop_condition=lambda x: True):
    """
    Iterate over all pages in the wikipedia dump file.
    If print_condition(page) is True, print the page and
    wait for the user to press Enter before continuing.
    """
    i = 0
    titles = []
    with open(dump_path, "r") as f:
        while True:
            i += 1
            if i % 10000 == 0:
                print(i)
            page = extract_next_page(f)
            if page is None:
                break
            title = get_title(page)
            categories = get_categories(page)
            if stop_condition(page):
                titles.append(title)
                # input('Press Enter to continue...')
    return titles


def stop_condition(page):
    categories = get_categories(page)
    for category in categories:
        category = category.replace(" ", "_")
        if category.lower() in selected_categories:
            return True
    return False


def subsample_pages(selected_categories, write_path="subsample.xml"):
    """
    Create a subsample of pages from the wikipedia dump file.
    The subsample contains all pages that have at least one category
    that is in the selected_categories set.
    """
    with open(dump_path, "r") as f:
        with open(write_path, "w") as g:
            i = 0
            while True:
                i += 1
                if i % 10000 == 0:
                    print(i)
                page = extract_next_page(f)
                if page is None:
                    break
                if stop_condition(page):
                    g.write(page)

subsample_pages(selected_categories)

dump_path = "subsample.xml"
titles = scroll_pages()
with open("titles.txt", "w") as f:
    f.write('\n'.join(sorted(titles)))


# with open(dump_path, "r") as f:
#     total = 0
#     nonempty_categories = 0
#     kept = 0
#     while True:
#         page = extract_next_page(f)
#         total += 1
#         title = get_title(page)
#         categories = get_categories(page)
#         if len(categories) > 0:
#             nonempty_categories += 1
#             if stop_condition(page):
#                 kept += 1
#         if total % 1000 == 0:
#             print(total)
#         if page is None or total > 10**5:
#             break


# print("Total pages:", total)
# print("Nonempty categories:", nonempty_categories)
# print("Kept pages:", kept)
