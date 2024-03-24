# todo:
# remove comments
# execute macros

import os
import tarfile
from urllib.request import urlretrieve

import arxiv

c = arxiv.Client()


def get_id_from_title(title, client):
    search = arxiv.Search(
        query=title,
        max_results=1,
    )

    results = client.results(search)

    for result in results:
        print(f"{title} - {result.title}")
        return result.get_short_id()

    raise Exception(f"No papers with title {title} found")


def download_source_files(id):
    """
    Skips downloading and extracting if the file already exists
    :param id:
    :return:
    """
    url = f"https://arxiv.org/src/{id}"
    path = f"{id}.tar.gz"

    if not os.path.exists(path):

        path, headers = urlretrieve(url, path)

        try:  # todo: policy to re-try
            tar = tarfile.open(path)
            tar.extractall(f"{id}/")
            tar.close()

        except:
            print(f"Error extracting {path}")


def get_cited_papers(id):
    extensions = [".bib", ".bbl"]
    # iterate over files in directory
    # check if file has extension in extensions
    # open file and extract citations
    # return list of citations
    bib_text = ""
    bbl_text = ""
    with os.scandir(id) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(tuple(extensions)):
                with open(entry, "r") as file:
                    if entry.name.endswith(".bib"):
                        bib_text += file.read()
                    if entry.name.endswith(".bbl"):
                        bbl_text += file.read()

    return bib_text, bbl_text


titles = [
    "ra-dit",
    "dino",
    "flat minima optimizers",
    "yolo",
    "barker dynamics"
]

# for t in titles:
#     x = get_id_from_title(t, c)
#     download_source_files(x)

bib, bbl = get_cited_papers("2310.01352v3")
print(bib)
print(bbl)

d = 0
