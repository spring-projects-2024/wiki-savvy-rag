from typing import Tuple, List, Dict
from copy import deepcopy


def join_messages_query_no_rag(history: List[Dict[str, str]], query):
    new_history = deepcopy(history)
    new_history.append({"role": "user", "content": query})

    return new_history


def join_messages_query_rag(
    history: List[Dict[str, str]], query: str, retrieved_docs: List[Tuple[str, float]]
):
    pass
