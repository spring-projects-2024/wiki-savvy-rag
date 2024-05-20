from itertools import chain
import json
import streamlit as st
from backend.model.rag_handler import RagHandler
from backend.vector_database.dataset import DatasetSQL
from backend.vector_database.embedder_wrapper import EmbedderWrapper
import time

MODEL_NAME = "Minami-su/Qwen1.5-0.5B-Chat_llamafy"
# "microsoft/phi-3-mini-128k-instruct",
# "Minami-su/Qwen1.5-0.5B-Chat_llamafy"
DB_PATH = "scripts/dataset/data/dataset.db"
INDEX_PATH = "scripts/vector_database/data/default.index"
DEVICE = "cpu"

st.title("Wikipedia Savvy")


@st.cache_resource
def load_dataset():
    return DatasetSQL(db_path=DB_PATH)


@st.cache_resource
def load_embedder():
    return EmbedderWrapper(DEVICE)


@st.cache_resource
def load_rag_handler():
    return RagHandler(
        model_name=MODEL_NAME,
        device=DEVICE,
        llm_kwargs={
            "torch_dtype": "auto",
        },
        faiss_kwargs={
            "index_path": INDEX_PATH,
            "dataset": dataset,
            "embedder": embedder,
        },
        use_rag=True,
    )


dataset = load_dataset()
embedder = load_embedder()
rag = load_rag_handler()


# fake generator on llm answer (string)
def string_generator(strlist):
    # akes a string to make a generator out of this, allowing to have a nice live token generation effect
    for s in strlist:
        yield s
        time.sleep(0.008)


def naive_inference(history, query):
    response, retrieved_docs = rag.naive_inference_with_retrieved_docs(
        histories=history,
        queries=query,
    )

    assert isinstance(response, str)

    return string_generator(response), retrieved_docs


def autoregressive_inference(history, query):
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


def mock_inference(history, query):
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


def response_generator(history, query):
    response, retrieved_docs = naive_inference(history, query)

    # todo: print markdown correctly
    # craft references for the retrieved documents in markdown
    retrieved_docs_str = ""
    for i, (doc, _) in enumerate(retrieved_docs):
        titles = " > ".join(json.loads(post_process_titles(doc["titles"])))
        retrieved_docs_str += f"""  
        **Chunk {i+1}**: *{titles}*"""
    retrieved_docs_str += f"\n\n"

    return chain(string_generator(retrieved_docs_str), response)


if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(
            message["content"]
        )  # allow to have latex and markdown code after generation

if prompt := st.chat_input("What do you want to know?"):
    st.chat_message("user").markdown(prompt)

    stream = response_generator(st.session_state.messages, prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response = st.write_stream(stream)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )  # updates the chat history once a new prompt is given
