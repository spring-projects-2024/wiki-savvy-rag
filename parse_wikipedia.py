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
    Extract the categories of a page.
    The categories are the text between the [[Category: and ]] tags.
    """
    categories = []
    for category in page.split("[[Category:")[1:]:
        category = category.split("]]")[0]
        categories.append(category)
    return categories


def scroll_pages(print_condition=lambda x: True):
    """
    Iterate over all pages in the wikipedia dump file.
    If print_condition(page) is True, print the page and
    wait for the user to press Enter before continuing.
    """
    i = 0
    with open("enwiki-20231220-pages-articles-multistream.xml", "r") as f:
        while True:
            i += 1
            if i % 10000 == 0:
                print(i)
            page = extract_next_page(f)
            if page is None:
                break
            title = get_title(page)
            categories = get_categories(page)
            if print_condition(page):
                print(page)
                input('Press Enter to continue...')


def print_condition(page):
    return "Category:" in get_title(page)


scroll_pages(print_condition=print_condition)






with open("enwiki-20231220-pages-articles-multistream.xml", "r") as f:
    total, redirect, disambiguation = 0, 0, 1
    l_normal, l_redirect, l_disambiguation = 0, 0, 1
    while True:
        page = extract_next_page(f)
        if '<redirect title="' in page:
            redirect += 1
            l_redirect += len(page)
        # elif "disambiguation" in page:
        #     disambiguation += 1
        #     l_disambiguation += len(page)
        else:
            l_normal += len(page)
        total += 1
        title = get_title(page)
        categories = get_categories(page)
        if total % 1000 == 0:
            print(total)
        if page is None or total > 10**3:
            break


print("Total pages:", total)
print("Redirect pages:", redirect)
print("Disambiguation pages:", disambiguation)

print("Average length of normal pages:", l_normal / (total - redirect - disambiguation))
print("Average length of redirect pages:", l_redirect / redirect)
print("Average length of disambiguation pages:", l_disambiguation / disambiguation)

print("Total length of normal pages:", l_normal)
print("Total length of redirect pages:", l_redirect)
print("Total length of disambiguation pages:", l_disambiguation)
