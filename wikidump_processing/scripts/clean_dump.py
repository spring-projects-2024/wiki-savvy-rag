from html import unescape
from tqdm import tqdm
import re
from backend.data_cleaning.utils import (
    scroll_pages,
    remove_table_tags,
    remove_template_tags,
    remove_wiki_tags,
    remove_square_brackets_around_links,
)


N_PAGES = 2357969
input_file = "wikidump_processing/data/subsample.xml"
output_file = "wikidump_processing/data/subsample_clean.xml"

asterisk_reg = r"^\*.*$"  # match lines that start with *
asterisk_regex = re.compile(asterisk_reg, re.MULTILINE)
remove_html_comment_reg = r"<!--.*?-->"  # match html comments
remove_html_comment = re.compile(remove_html_comment_reg, re.DOTALL)
ref_tag_reg = r"<(ref|span|gallery|timeline|imagemap|mapframe|div|references).*?/(\1|)>"  # match <ref ... /> tags and similar
remove_ref_tag = re.compile(ref_tag_reg, re.DOTALL)


def process_page(page):
    page = unescape(page)
    page = asterisk_regex.sub("", page)
    page = remove_html_comment.sub("", page)
    page = remove_ref_tag.sub("", page)

    page = remove_table_tags(page)
    page = remove_template_tags(page)
    page = remove_wiki_tags(page)
    page = remove_square_brackets_around_links(page)

    page = "\n".join([s for s in page.split("\n") if s])  # remove empty lines
    return page + "\n"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("delete_old_files", type=bool, default=False, help="Delete old files after creating new ones, to save storage")
    args = parser.parse_args()

    with open(output_file, "w") as out:
        with open(input_file, "r") as f:
            for page in tqdm(scroll_pages(f), total=N_PAGES):
                out.write(process_page(page))

    if args.delete_old_files:
        import os
        os.remove(input_file)
