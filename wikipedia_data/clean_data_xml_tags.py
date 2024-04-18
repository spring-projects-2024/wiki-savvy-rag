# - remove "metadata"
# - remove "&quot" and other html entities, check integrity and so on

from tqdm import tqdm
from utils import scroll_pages

TAGS = [
    "id",
    "ns",
    "parentid",
    "timestamp",
    "contributor",
    "username",
    "comment",
    "model",
    "format",
    "sha1"
]

from lxml import etree

input_file = "subsample_out_1.xml"
output_file = "subsample_out_2.xml"

c_err = 0
with open("subsample_out_2_no_xml.xml", "a") as error_f:
    with open(output_file, "w") as out:
        with open(input_file, "r") as f:
            for p in tqdm(scroll_pages(f)):
                try:
                    tree = etree.fromstring(p)
                    for tag in TAGS:
                        for e in tree.iter(tag):
                            e.getparent().remove(e)

                    new_page = etree.tostring(tree, encoding="unicode")
                    new_page = new_page.replace("* \n", "")
                    out.write(new_page)
                except:
                    c_err += 1
                    new_page = p
                    error_f.write(new_page)
                    print(f"Error number:  {c_err}")
                    # find unclosed xml tags with regex
                    # import re
                    #
                    # # find unclosed tags
                    # opening_tags = re.findall(r"<[^/][^/>]*", new_page)
                    # # find closed tags
                    # closed_tags = re.findall(r"</[^>]*", new_page)
                    # print()
                    # print(opening_tags)
                    # print(closed_tags)
                    # print(len(opening_tags), len(closed_tags))
                    # input()
