from html import unescape
from tqdm import tqdm
from utils import scroll_pages
import re
from utils import remove_template_tags_table, remove_template_tags, greedy_remove_wiki_tags


N_PAGES = 2357969
input_file = "subsample.xml"
output_file = "subsample_clean.xml"

asterisk_reg = r"^\*.*$"  # match lines that start with *
asterisk_regex = re.compile(asterisk_reg, re.MULTILINE)
remove_html_comment_reg = r"<!--.*?-->"  # match html comments
remove_html_comment = re.compile(remove_html_comment_reg, re.DOTALL)
ref_tag_reg = r"<(ref|span|gallery|timeline|imagemap|mapframe|div).*?/(\1|)>"
remove_ref_tag = re.compile(ref_tag_reg, re.DOTALL)


if __name__ == "__main__":
    with open(output_file, "w") as out:
        with open(input_file, "r") as f:
            for page in tqdm(scroll_pages(f), total=N_PAGES):

                page = unescape(page)
                page = remove_template_tags_table(page)
                page = remove_template_tags(page)
                page = asterisk_regex.sub("", page)
                page = remove_html_comment.sub("", page)
                page = remove_ref_tag.sub("", page)
                page = greedy_remove_wiki_tags(page)

                page = "\n".join([s for s in page.split("\n") if s])  # remove empty lines
                out.write(page)
                out.write("\n")  # ensure newline at end of page
