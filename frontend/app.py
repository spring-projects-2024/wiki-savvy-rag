import streamlit as st
from chatbot_controller import load_controller
from sidebar import build_sidebar
from backend.arxiv.utils import get_id_from_link_prompt, get_id_from_pdf
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
st.set_page_config("Wikipedia Savvy", page_icon=":books:")


build_sidebar()
configs = st.session_state["configs"]
controller = load_controller()


with st.spinner("Loading/Updating the Chatbot. It could take a while..."):
    controller.update_configs(configs)

st.title(":books: Wikipedia Savvy")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "file_upload_key" not in st.session_state:
    st.session_state["file_upload_key"] = 0

if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = []

if len(st.session_state["messages"]) == 2:
    st.toast(
        "Did you know that you can use link to arXiv papers to augment the model's reply? \n You can also directly upload the pdf of an arXiv paper, using the Upload button below!",
        icon="😍",
    )

# Display chat messages
for message in st.session_state.messages:
    if (
        message["role"] == "assistant"
        and message.get("retrieved_docs")
        and len(message["retrieved_docs"]) > 0
    ):
        with st.chat_message("rag", avatar=":material/article:"):
            st.write(
                controller.build_retrieved_docs_html(message["retrieved_docs"]),
                unsafe_allow_html=True,
            )
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

print(st.session_state["uploaded_file"])

if prompt := st.chat_input("Please ask a question."):
    st.chat_message("user").markdown(prompt)
    link_id = get_id_from_link_prompt(prompt)
    if (link_id) != None:
        print(link_id)

    with st.spinner("Thinking..."):
        stream, retrieved_docs = controller.inference(st.session_state.messages, prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})

        if len(retrieved_docs) > 0:
            with st.chat_message("rag", avatar=":material/article:"):
                st.write(
                    controller.build_retrieved_docs_html(retrieved_docs),
                    unsafe_allow_html=True,
                )
    with st.chat_message("assistant"):
        response = st.write_stream(stream)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
                "retrieved_docs": retrieved_docs,
            }
        )

uploaded_file = st.file_uploader(
    "📁 Upload File", key=st.session_state["file_upload_key"]
)
if uploaded_file:
    bytes_data = uploaded_file.getvalue()
    id = get_id_from_pdf(bytes_data)
    st.session_state["uploaded_file"] = id
    st.session_state["file_upload_key"] += 1
else:
    st.session_state["uploaded_file"] = []
