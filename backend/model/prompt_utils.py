from typing import Tuple, List, Dict
from copy import deepcopy


def join_messages_query_no_rag(history: List[Dict[str, str]], query):
    new_history = deepcopy(history)
    new_history.append({"role": "user", "content": query})
    return new_history


def join_messages_query_rag(
    history: List[Dict[str, str]], query: str, retrieved_docs: List[Tuple[str, float]]
):
    new_history = deepcopy(history)
    message = ""
    message += "== Retrieved documents ==\n"
    for i, (doc, _) in enumerate(retrieved_docs):
        message += f"=== Document {i} ===\n"
        message += doc + "\n"
    message += f"== User Query ==\n{query}"
    new_history.append({"role": "user", "content": message})
    return new_history


if __name__ == "__main__":
    history = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm fine, thank you."},
    ]

    query = "Can you provide ways to eat combinations of bananas and dragonfruits?"

    retrieved_docs = [
        ("Banana and dragonfruit smoothie", 0.1),
        ("Banana and dragonfruit salad", 0.2),
        ("Banana and dragonfruit pie", 0.3),
    ]

    print(join_messages_query_no_rag(history, query))
    print(join_messages_query_rag(history, query, retrieved_docs))
