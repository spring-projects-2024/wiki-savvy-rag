import json
import time
from itertools import chain
from typing import Dict, List

from backend.model.rag_handler import RagHandler


# fake generator on llm answer (string)
def string_generator(strlist: List[str]):
    # akes a string to make a generator out of this, allowing to have a nice live token generation effect
    for s in strlist:
        yield s
        time.sleep(0.008)


def naive_inference(
    rag: RagHandler,
    history: List[Dict],
    query: str,
):
    response, retrieved_docs = rag.naive_inference_with_retrieved_docs(
        histories=history,
        queries=query,
    )

    assert isinstance(response, str)

    return string_generator(response), retrieved_docs


def autoregressive_inference(
    rag: RagHandler,
    history: List[Dict],
    query: str,
):
    return rag.autoregressive_generation_iterator_with_retrieved_docs(query=query)


def post_process_titles(text: str):
    """This function is necessary because the json in titles is not formatted properly"""
    return (
        text.replace("['", '["')
        .replace("',", '",')
        .replace(", '", ', "')
        .replace(",'", ',"')
        .replace("']", '"]')
    )


def mock_inference(
    rag: RagHandler,
    history: List[Dict],
    query: str,
):
    response = 'Sure, here\'s an example of how you could write well formatted Markdown text:\n```markdown\n## Chunk 1: *Banana > Banana and dragonfruit salad > Banana and dragonfruit pie*\nI am a mock response.\n```\n\nThis will create a list of three chunks that are related to the topic of "Banana". Each chunk is separated by a line break, which makes it easier for readers to read and understand the content of each chunk. The first chunk is a brief introduction to the topic, followed by three examples of different ways to make banana and dragonfruit salads. The second chunk provides a recipe for a banana split, and the third chunk includes information about the ingredients needed to make banana pudding.'
    retrieved_docs = [
        (
            {
                "titles": '["Banana", "Banana and dragonfruit salad", "Banana and dragonfruit pie"]'
            },
            1,
        ),
        (
            {
                "titles": '["Pina colada", "Pina colada recipe", "Pina colada ingredients"]'
            },
            1,
        ),
        (
            {
                "titles": '["Banana split", "Banana split recipe", "Banana split ingredients"]'
            },
            1,
        ),
        (
            {
                "titles": '["Banana bread", "Banana bread recipe", "Banana bread ingredients"]'
            },
            1,
        ),
        (
            {
                "titles": '["Banana pudding", "Banana pudding recipe", "Banana pudding ingredients"]'
            },
            1,
        ),
    ]
    return string_generator(response), retrieved_docs


def build_retrieved_docs_str(retrieved_docs: List[Dict]):
    retrieved_docs_str = ""
    for i, (doc, score) in enumerate(retrieved_docs):
        titles = " > ".join(json.loads(post_process_titles(doc["titles"])))
        retrieved_docs_str += f"""  
        **Chunk {i+1}**: *{titles}* (score: {score})"""
    retrieved_docs_str += f"\n\n"
    return retrieved_docs_str
