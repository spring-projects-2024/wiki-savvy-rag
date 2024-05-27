from backend.arxiv.utils import get_plain_doc_from_id, get_title_from_id
from typing import Dict, List
import re
import arxiv

client = arxiv.Client(delay_seconds=0.0)

SHORT_CHUNK_LENGTH = 45
TOO_SHORT_CHUNK_LENGTH = 1000


def clean_latex_code(paragraph):
    latex_pattern = r"\\(?:[A-Za-z]+|.)"
    cl = re.sub(latex_pattern, "", paragraph)
    cl = re.sub(r"\\.*?\\n", "", cl)
    cl = re.sub("(?:={4,}|.*\n)", "", cl)
    cl = re.sub(r"<cit\.>", "", cl)
    cl = re.sub(r"[\w.-]+\.[\w.-]+\.[\w.-]+", "", cl)
    cl = re.sub(r"\bbbl\.editors\b|\bbbl\.editor if\b", "", cl)
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

    if len(first_step_chunks) == 0:
        return None
    else:
        return first_step_chunks


def papers_to_chunks(ids: List[str]):
    chunks = {}
    for id in ids:
        ch = extract_chunks(id)
        title = get_title_from_id(id, client)
        chunks[title] = ch
    return chunks


def split_text_into_paragraphs(text):
    paragraphs = text.split("\n")
    return [para.strip() for para in paragraphs if para.strip()]


def process_text_list(text_list):
    processed_list = []
    for text in text_list:
        if len(text) > 1000:
            paragraphs = split_text_into_paragraphs(text)
            processed_list.extend(paragraphs)
        else:
            processed_list.append(text)
    return processed_list
