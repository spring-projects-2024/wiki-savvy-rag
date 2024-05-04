# todo:
# - remove {{cite }} tags from references ?

from tqdm import tqdm
import re, os

try:
    from wikipedia_data.utils import scroll_pages, greedy_replace
except ImportError:
    from utils import scroll_pages, greedy_replace

COMPUTE_PAGES = False

if COMPUTE_PAGES:
    # takes around ~20 seconds
    N_PAGES = os.popen('grep -c "<page>" subsample.xml').read()
else:
    N_PAGES = 2357969

# probably we should make it negative
REMOVE_WIKI_TEMPLATE_TAGS = ["cite", "sfn", "see also", "lang", "ref", "main"]

tags_string = "|".join(REMOVE_WIKI_TEMPLATE_TAGS)
base = r"{{(" + tags_string + ")[^}\\\]*}}"
pattern = re.compile(base, flags=re.IGNORECASE)


# prova = r"isplay=&quot;block&quot;&gt; \hat{\sigma}_\text{OC} = \frac{{\sigma}_\text{OC}}{\pi r^2} &lt;/math&gt;"
#
# print(re.sub(pattern, 'AAAA', prova))
# exit()


def remove_cite(text):
    text_no_cite = re.sub(pattern, "", text)

    text_no_file, res = greedy_replace(text_no_cite, "[[File:", "[", "]")
    while res:
        text_no_file, res = greedy_replace(text_no_file, "[[File:", "[", "]")

    return text_no_file


# input_file = "subsample_out_0.xml"

# output_file = "subsample_out_1.xml"

# with open(output_file, "a") as out:
#     with open(input_file, "r") as f:
#         for page in tqdm(scroll_pages(f), total=N_PAGES):
#             page = remove_cite(page)
#             out.write(page)

# counts pages
# grep -c "<page>" enwiki-latest-pages-articles.xml
