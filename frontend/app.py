import streamlit as st

from backend.arxiv.utils import get_id_from_link_prompt
from chatbot_controller import load_controller
from sidebar import build_sidebar
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# TODOS:
# - Use memory to craft better query
# - Decide how to distinguish RAG vs non-RAG prompts

build_sidebar()
configs = st.session_state["configs"]

controller = load_controller()

if controller.should_update_configs(configs):
    with st.spinner("Loading Chatbot. It could take a while..."):
        controller.update_configs(configs)

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
            st.markdown(controller.build_retrieved_docs_str(message["retrieved_docs"]))
        st.markdown(message["content"])

if prompt := st.chat_input(
    "Please ask a question. (You can also augment my reply by indicating arxiv paper links) "
):
    y = get_id_from_link_prompt(prompt)
    st.write(y)

    st.chat_message("user").markdown(prompt)
    stream, retrieved_docs = controller.naive_inference(
        st.session_state.messages, prompt
    )

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        if len(retrieved_docs) > 0:
            st.markdown(controller.build_retrieved_docs_str(retrieved_docs))
        response = st.write_stream(stream)

    st.session_state.messages.append(
        {"role": "assistant", "content": response, "retrieved_docs": retrieved_docs}
    )
