import torch
from utils import naive_inference, build_retrieved_docs_str
import streamlit as st

from backend.model.rag_handler import RagHandler
from backend.vector_database.dataset import DatasetSQL
from backend.vector_database.embedder_wrapper import EmbedderWrapper
from sidebar import build_sidebar, MODELS
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# TODOS:
# - Use memory to craft better query
# - Decide how to distinguish RAG vs non-RAG prompts

build_sidebar()
configs = st.session_state["configs"]


@st.cache_resource
def load_rag_handler(
    model_idx: str,
    db_path: str,
    index_path: str,
    device: str,
    use_rag: bool,
):
    rag_files_found = os.path.exists(index_path) and os.path.exists(db_path)
    if use_rag and rag_files_found:
        dataset = DatasetSQL(db_path=db_path)
        embedder = EmbedderWrapper(device)
        faiss_kwargs = {
            "index_path": index_path,
            "dataset": dataset,
            "embedder": embedder,
        }
    else:
        faiss_kwargs = None

    model = MODELS[model_idx]

    return RagHandler(
        model_name=model["model"],
        device=device,
        llm_kwargs={
            "torch_dtype": "auto",
            "do_sample": True,
        },
        faiss_kwargs=faiss_kwargs,
        use_rag=use_rag and rag_files_found,
        use_qlora=model["use_qlora"],
    )


rag = load_rag_handler(**configs)

st.title("Wikipedia Savvy")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if (
            message["role"] == "assistant"
            and message.get("retrieved_docs")
            and len(message["retrieved_docs"]) > 0
        ):
            st.markdown(build_retrieved_docs_str(message["retrieved_docs"]))

        st.markdown(message["content"])

if prompt := st.chat_input("What do you want to know?"):
    st.chat_message("user").markdown(prompt)

    stream, retrieved_docs = naive_inference(rag, st.session_state.messages, prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        st.markdown(build_retrieved_docs_str(retrieved_docs))
        response = st.write_stream(stream)

    st.session_state.messages.append(
        {"role": "assistant", "content": response, "retrieved_docs": retrieved_docs}
    )
