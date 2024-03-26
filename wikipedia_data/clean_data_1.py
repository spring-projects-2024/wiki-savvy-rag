# todo:
# - remove {{cite }} tags from references ?

from tqdm import tqdm

from wikipedia_data.utils import scroll_pages


def remove_cite(text):
    import re
    # remove cite or sfn templates
    return re.sub(r'{{(cite|sfn)[^}]*}}', '', text, flags=re.IGNORECASE)
    # return re.sub(r'{{cite[^}]*}}', '', text, flags=re.IGNORECASE)


input_file = "subsample.xml"

output_file = "subsample_out_1.xml"

with open(output_file, "w") as out:
    with open(input_file, "r") as f:
        for page in tqdm(scroll_pages(f)):
            page = remove_cite(page)
            out.write(page)

# count lines containing page with grep in a large file
# grep -c "<page>" enwiki-latest-pages-articles.xml
