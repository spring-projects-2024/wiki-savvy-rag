from tqdm import tqdm
from utils import scroll_pages, greedy_remove_wiki_tags
import re

N_PAGES = 2357969
input_file = "subsample_out_2.xml"
output_file = "subsample_out_3.xml"


TAGS = ["ref", "span"]
# reg = r"^[\*\s]*$"
reg = r"^\*.*$"
regex = re.compile(reg, re.MULTILINE)

remove_ref_html_tag = r"<(ref|span|gallery|timeline|imagemap|mapframe|div).*?/(\1|)>"
remove_html_comment = re.compile(remove_ref_html_tag, re.DOTALL)

with open(output_file, "a") as out:
    with open(input_file, "r") as f:
        for page in tqdm(scroll_pages(f), total=N_PAGES):
            # replace ref tags
            page = remove_html_comment.sub("", page)
            # remove wiki tags
            page = greedy_remove_wiki_tags(page)
            # remove empty lines
            page = "\n".join([s for s in page.split("\n") if s])

            out.write(page)
            out.write("\n")
