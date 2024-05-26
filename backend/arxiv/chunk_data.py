from backend.arxiv.utils import get_plain_doc_from_id
from typing import Dict, List

SHORT_CHUNK_LENGTH = 50


def extract_chunks(id):
    doc = get_plain_doc_from_id(id)
    chunks = []
    for paragraph in doc.split("\n\n"):
        if len(paragraph) > SHORT_CHUNK_LENGTH:
            chunks.append({"text": paragraph})

    if len(chunks) == 0:
        return None
    return chunks


def papers_to_chunks(ids: List[str]):
    chunks = {}
    for id in ids:
        ch = extract_chunks(id)
        chunks[id] = ch
    return chunks
