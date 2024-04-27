from tqdm import tqdm
from utils import scroll_pages, greedy_remove_template_tags
import re

N_PAGES = 2357969
input_file = "subsample_out_1.xml"
output_file = "subsample_out_2.xml"

# reg = r"^[\*\s]*$"
reg = r"^\*.*$"
regex = re.compile(reg, re.MULTILINE)

remove_html_comment_reg = r"<!--.*?-->"
remove_html_comment = re.compile(remove_html_comment_reg, re.DOTALL)

with open(output_file, "a") as out:
    with open(input_file, "r") as f:
        for page in tqdm(scroll_pages(f), total=N_PAGES):
            page = greedy_remove_template_tags(page)
            page = regex.sub("", page)
            page = remove_html_comment.sub("", page)
            page = "\n".join([s for s in page.split("\n") if s])  # remove empty lines

            out.write(page)
            out.write("\n")
