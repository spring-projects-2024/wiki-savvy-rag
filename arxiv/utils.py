# todo:
# remove comments
# execute macros
import concurrent.futures
import os
import tarfile
import time
from typing import List
from urllib.request import urlretrieve
import arxiv
import tqdm
from refextract import extract_references_from_url

arx_client = arxiv.Client(delay_seconds=.0)


def get_info_from_title(title, client) -> arxiv.Result:
    search = arxiv.Search(
        query=title,
        max_results=1,
    )

    results = client.results(search)

    for result in results:
        return result

    raise Exception(f"No papers with title {title} found")


def get_title_from_id(id, client) -> str:
    search = arxiv.Search(
        query=id,
        max_results=1,
    )

    results = client.results(search)

    for result in results:
        return result.title

    raise Exception(f"No papers with id {id} found")


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


def get_references_raw(id) -> List:
    """
    Extract list of references from an ArXiv paper. Given the ArXiv ID of the paper,
     builds the url of pdf version (required for the retrieval)
    :param id: paper id
    """
    reference = extract_references_from_url("http://arxiv.org/pdf/" + str(id) + ".pdf")
    return reference


def get_text_from_extensions(id, extensions) -> str:
    """
    Returns the text of all files in the directory with the given extensions
    :param id: paper id
    :param extensions: list of extensions to search for
    :return: text of all files with the given extensions
    """
    text = ""
    with os.scandir(id) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(tuple(extensions)):
                with open(entry, "r") as file:
                    text += file.read()
    return text


def parse_single_ref(ref) -> str | None:
    """
    :param ref: reference dict as output by refextract
    :return: id of the reference if found, None otherwise
    """
    if "reportnumber" in ref:
        rpnum = ref["reportnumber"]
        rpnum = rpnum[0].replace("arXiv:", "")
        return rpnum

    if "raw_ref" in ref:
        raw = ref["raw_ref"]
        # remove [number] from the beginning
        raw = raw[0].split("]")[1]
        pieces = raw.split(".")
        pieces.sort(key=len, reverse=True)
        for p in pieces:

            # if only contains spaces and numbers continue
            if all(ch == "," or ch.isdigit() or ch.isspace() for ch in p):
                continue

            p = p.strip()
            try:
                info_paper = get_info_from_title(p, arx_client)
                att_id = info_paper.get_short_id()
                title = info_paper.title

                if title.lower() in raw.lower():
                    return att_id
            except:
                continue
    print(ref["raw_ref"])
    return None


def parse_references(references):
    """
    :param references: list of references as output by refextract
    :return: list of ids of the references, not ordered and without the ones that could not be found
    """

    MAX_THREADS = 100
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        results = executor.map(parse_single_ref, references)

    id_list = [result for result in results if result is not None]

    return id_list


if __name__ == '__main__':
    title = "Deep Residual Learning for Image Recognition"  # random paper, 2016

    id = get_info_from_title(title, arx_client).get_short_id()
    print(f"{id=}")
    print(f"ID found in {time.process_time()} seconds")

    references = get_references_raw(id)
    print(f"Number of references to search: {len(references)}")
    print(f"References found in {time.process_time()} seconds")

    ids = parse_references(references)
    print(f"Number of references found: {len(ids)}")
    print(f"References parsed in {time.process_time()} seconds")
