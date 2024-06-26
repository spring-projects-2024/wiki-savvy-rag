import concurrent.futures
import os
import tarfile
import time
from typing import List
from urllib.request import urlretrieve
import arxiv
import re
import fitz
from pylatexenc.latex2text import LatexNodes2Text

try:
    from refextract import extract_references_from_url
except:
    extract_references_from_url = None

arx_client = arxiv.Client(delay_seconds=0.0)


def get_plain_doc_from_id(id: str):
    download_source_files(id)
    doc = get_text_from_extensions(id, ".tex")
    parsed_doc = LatexNodes2Text().latex_to_text(doc)
    return parsed_doc


def get_ids_from_link_prompt(query: str) -> List[str]:
    """
    Takes the query for the LLM, from the user and checks if it contains a link to archive paper.
    If that is the case, returns paper ids from the link, otherwise returns None
    """
    pattern = r"https:\/\/arxiv\.org\/(?:abs|pdf)\/([0-9]{4}\.[0-9]{5})"
    ret_ids = re.findall(pattern, query)
    if len(ret_ids) != 0:
        return set(ret_ids)
    return None


def get_info_from_title(title: str, client: arxiv.Client = arx_client) -> arxiv.Result:
    """
    Get info from a paper given its title.
    :raises Exception if no paper is found
    """
    search = arxiv.Search(
        query=title,
        max_results=20,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    results = client.results(search)

    results = list(results)

    for result in results:
        if result.title.lower() == title.lower():
            return result

    raise Exception(f"No papers with title {title} found")


def get_title_from_id(id: str, client: arxiv.Client) -> str:
    """
    Get the title from the id.
    :raises exception if no paper is found
    """
    search = arxiv.Search(
        query=id,
        max_results=1,
    )

    results = client.results(search)

    for result in results:
        return result.title

    raise Exception(f"No papers with id {id} found")


def download_source_files(id: str):
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
        time.sleep(5)
        tar.close()
    except:
        Exception("Error extracting {path}")


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
            if (
                all(ch == "," or ch.isdigit() or ch.isspace() for ch in p)
                or len(p) < 10
            ):
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


def get_id_from_pdf(pdf_file):
    """Given a pdf file, it will return the paper id (as title)"""
    doc = fitz.open(stream=pdf_file, filetype="pdf")
    first_page = doc[0]
    blocks = first_page.get_text("dict")["blocks"]
    title = ""
    max_font_size = 0
    for block in blocks:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                # Check if the current span has the largest font size found so far
                if span["size"] > max_font_size:
                    max_font_size = span["size"]
                    title = span["text"]

    title = title.strip()
    # use a regex to get the plain arxiv id
    pattern = r"arXiv:([\d.]+)v1"
    re_match = re.search(pattern, title)
    if re_match:
        return re_match.group(1)
    else:
        return None


if __name__ == "__main__":
    title = "Supersymmetric Quantum Mechanics, multiphoton algebras and coherent states"
    x = get_ids_from_link_prompt("https://arxiv.org/abs/2005.11401")
    print(get_plain_doc_from_id("2005.11401"))
