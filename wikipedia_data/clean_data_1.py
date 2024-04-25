from html import unescape

from tqdm import tqdm
from utils import scroll_pages
import re
from utils import greedy_remove_template_tags_table

N_PAGES = 2357969

input_file = "subsample.xml"

output_file = "subsample_out_1.xml"

# r = r"\{\|\s?\s?\s?(class|style|align|border)((.|\n)*?)\|\}"
# regex = re.compile(r, re.IGNORECASE)


with open(output_file, "a") as out:
    with open(input_file, "r") as f:
        for page in tqdm(scroll_pages(f), total=N_PAGES):
            # replace all regex matches with empty string
            page = unescape(page)
            # page = regex.sub("", page)
            page = greedy_remove_template_tags_table(page)

            out.write(page)

