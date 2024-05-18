import streamlit as st
from backend.model.rag_handler import RagHandler


@st.cache_resource
def build_rag_handler():
    return RagHandler(
        model_name="Minami-su/Qwen1.5-0.5B-Chat_llamafy",
        device="cpu",
        model_kwargs={
            "torch_dtype": "auto",
        },
        use_rag=False,
    )


rag = build_rag_handler()

st.title("Wikipedia Savvy")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What do you want to know?"):
    st.chat_message("user").markdown(prompt)

    response = rag.naive_inference(
        histories=st.session_state.messages,
        queries=prompt,
    )

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
