import os
import torch
import streamlit as st
from chatbot_controller import (
    DB_PATH_DEFAULT,
    DEVICE_DEFAULT,
    INDEX_PATH_DEFAULT,
    MODEL_DEFAULT,
    MODELS,
)


@st.cache_data
def load_available_devices():
    devices = {
        "cpu": "CPU",
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices[f"cuda:{i}"] = torch.cuda.get_device_name(i)

    return devices


def build_sidebar():
    available_devices = load_available_devices()

    if "configs" not in st.session_state:
        st.session_state["configs"] = {
            "model_idx": MODEL_DEFAULT,
            "db_path": DB_PATH_DEFAULT,
            "index_path": INDEX_PATH_DEFAULT,
            "device": DEVICE_DEFAULT if DEVICE_DEFAULT in available_devices else "cpu",
            "use_rag": True,
        }

    configs = st.session_state["configs"].copy()

    with st.sidebar:

        rag_files_found = os.path.exists(configs["index_path"]) and os.path.exists(
            configs["db_path"]
        )

        if configs["use_rag"]:
            if rag_files_found:
                rag_str = "Use RAG ✅"
            else:
                rag_str = "Should use RAG, but files can't be found ⚠️"
        else:
            rag_str = "Don't use RAG ❌"

        st.markdown(
            f"""
        ## Configurations  
        Device: *{available_devices[configs["device"]]}*  
        Model: *{MODELS[configs["model_idx"]]["name"]}* 
        #### RAG   
        {rag_str}
        """
        )

        if configs["use_rag"]:
            st.markdown(
                f"""
            Database Path: *{configs["db_path"]}*  
            Index Path: *{configs["index_path"]}*  
            """
            )

        with st.form("configs_form", border=False) as form:
            st.markdown("## Edit Configurations")
            new_device = st.selectbox(
                "Device",
                list(available_devices.keys()),
                index=list(available_devices.keys()).index(configs["device"]),
                format_func=lambda x: available_devices[x],
            )
            new_model_idx = st.selectbox(
                "Model",
                [i for i in range(len(MODELS))],
                index=configs["model_idx"],
                format_func=lambda x: MODELS[x]["name"],
            )
            st.markdown("#### RAG")
            new_use_rag = st.checkbox("Use RAG", value=configs["use_rag"])
            new_db_path = st.text_input("Database Path", DB_PATH_DEFAULT)
            new_index_path = st.text_input("Vector Index Path", INDEX_PATH_DEFAULT)

            submitted = st.form_submit_button("Apply")
            if submitted:
                configs["model_idx"] = new_model_idx
                configs["db_path"] = new_db_path
                configs["index_path"] = new_index_path
                configs["device"] = new_device
                configs["use_rag"] = new_use_rag

                st.session_state["configs"] = configs
                st.rerun()
