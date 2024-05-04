from typing import List


def construct_text_from_chunk(titles: List, text: str):
    s = ""
    for title in titles:
        s += title + "\n"
    s += "\n"
    s += text
    return s
