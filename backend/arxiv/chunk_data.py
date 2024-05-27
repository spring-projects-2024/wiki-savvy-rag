from backend.arxiv.utils import get_plain_doc_from_id, get_title_from_id
from typing import Dict, List
import re
import arxiv
import os
import shutil

client = arxiv.Client(delay_seconds=0.0)

SHORT_CHUNK_LENGTH = 100
MAX_CHUNK_LENGTH = 1000


def clean_latex_code(paragraph):
    cl = re.sub(r"\S*\\\S*", "", paragraph)
    latex_pattern = r"\\(?:[A-Za-z]+|.)"
    cl = re.sub(latex_pattern, "", cl)
    cl = re.sub(r".*\\.*", "", cl)
    cl = re.sub("(?:={4,}|.*\n)", "", cl)
    cl = re.sub(r"<cit\.>", "", cl)
    cl = re.sub(r"[\w.-]+\.[\w.-]+\.[\w.-]+", "", cl)
    cl = re.sub(r"\bbbl\.editors\b|\bbbl\.editor if\b", "", cl)
    cl = re.sub(r"\S*\\\S*", "", cl)
    return cl


def extract_chunks(id):
    """Given the id of a paper, extracts all relevant chunks"""
    doc = get_plain_doc_from_id(id)
    first_step_chunks = []
    for paragraph in doc.split("\n\n"):
        cleaned_par = clean_latex_code(paragraph)
        nospace = re.sub(r"\s+", "", cleaned_par)
        if len(nospace) > SHORT_CHUNK_LENGTH:
            first_step_chunks.append(cleaned_par)
    try:
        shutil.rmtree(os.getcwd() + "/" + id, ignore_errors=True)
        os.remove(os.getcwd() + "/" + id + ".tar.gz")
    except OSError as e:
        print(os.getcwd())
        print(f"The Folder does not exist:{e.strerror}")
    if len(first_step_chunks) == 0:
        return None

    return first_step_chunks


def papers_to_chunks(ids: List[str]):
    chunks = []
    for id in ids:
        paper_ch = extract_chunks(id)
        paper_title = get_title_from_id(id, client)
        i = 1
        for chunk in paper_ch:
            title = paper_title
            title += f" Section {i}"
            c_dictionary = {"titles": '[ "' + title + '"]', "text": chunk}
            chunks.append(c_dictionary)
            i += 1
    return chunks
