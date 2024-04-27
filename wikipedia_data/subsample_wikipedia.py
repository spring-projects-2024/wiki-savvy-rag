import json
from utils import extract_next_page, get_title, get_categories


DUMP_PATH = "../../enwiki-20231220-pages-articles-multistream.xml"
CATEGORY_PATH = "selected_categories.json"


def scroll_pages(condition=lambda x: True, stop_on_hit=False):
    """
    Iterate over all pages in the wikipedia dump file.
    If condition(page) is True, add the page title to a list.
    Additionally, if stop_on_hit is True, print the title and
    categories of the page and wait for the user to press Enter.
    """
    i = 0
    titles = []
    with open(DUMP_PATH, "r") as f:
        while True:
            i += 1
            if i % 10000 == 0:
                print(i)
            page = extract_next_page(f)
            if page is None:
                break
            title = get_title(page)
            categories = get_categories(page)
            if condition(page):
                titles.append(title)
                if stop_on_hit:
                    print(title)
                    print(categories)
                    input('Press Enter to continue...')
    return titles


def stop_condition(page):
    categories = get_categories(page)
    for category in categories:
        category = category.replace(" ", "_")
        if category.lower() in selected_categories:
            return True
    return False


def save_subsample_of_pages(selected_categories, write_path="subsample.xml"):
    """
    Create a subsample of pages from the wikipedia dump file.
    The subsample contains all pages that have at least one category
    that is in the selected_categories set.
    """
    titles = []
    with open(DUMP_PATH, "r") as f:
        with open(write_path, "w") as out:
            i = 0
            while True:
                i += 1
                if i % 10000 == 0:
                    print(i)
                page = extract_next_page(f)
                if page is None:
                    break
                if stop_condition(page):
                    titles.append(get_title(page))
                    out.write(page)
    return titles


if __name__ == "__main__":
    with open(CATEGORY_PATH, "r") as f:
        selected_categories = json.load(f)
        selected_categories = set(selected_categories)
    dump_path = "subsample.xml"
    titles_path = "subsampled_titles.txt"

    titles = save_subsample_of_pages(selected_categories, dump_path)
    with open(titles_path, "w") as f:
        f.write('\n'.join(sorted(titles)))
