import streamlit as st
from backend.model.rag_handler import RagHandler
import time


st.title("Wikipedia Savvy")

@st.cache_resource
def build_rag_handler():
    return RagHandler(
        model_name="Minami-su/Qwen1.5-0.5B-Chat_llamafy",
        #"microsoft/phi-3-mini-128k-instruct",
        #"Minami-su/Qwen1.5-0.5B-Chat_llamafy"
        device="cpu",
        model_kwargs={
            "torch_dtype": "auto",
        },
        use_rag=False,
    )


rag = build_rag_handler()


if "messages" not in st.session_state:
    st.session_state['messages'] = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What do you want to know?"):
    st.chat_message("user").markdown(prompt)

    response = rag.naive_inference(
        histories=st.session_state.messages,
        queries=prompt,
    )

    #fake generator on llm answer (string)
    def string_generator(strlist):
        """Takes a string to make a generator out of this, allowing to have a nice live token generation effect"""
        for s in strlist:
            yield s
            time.sleep(0.008)

    stream = string_generator(response)
    

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        st.write_stream(stream)
        #st.markdown(response)   #writes in the chat the llm answer
    st.session_state.messages.append({"role": "assistant", "content": response}) #updates the chat history once a new prompt is given

