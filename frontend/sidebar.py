import os
import torch
import streamlit as st
from chatbot_controller import (
    DB_PATH_DEFAULT,
    DEVICE_DEFAULT,
    INDEX_PATH_DEFAULT,
    INFERENCE_TYPES,
    MODEL_DEFAULT,
    MODELS,
)

INFERENCE_TYPE_MAP = {
    "naive": "Naive",
    "autoregressive": "Autoregressive (REPLUG)",
    "mock": "Mock (testing purposes)",
}


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
            "rag_initialization": {
                "model_idx": MODEL_DEFAULT,
                "db_path": DB_PATH_DEFAULT,
                "index_path": INDEX_PATH_DEFAULT,
                "device": (
                    DEVICE_DEFAULT if DEVICE_DEFAULT in available_devices else "cpu"
                ),
                "use_rag": True,
            },
            "inference_type": "naive",
            "retrieved_docs": 5,
        }

    configs = st.session_state["configs"].copy()

    with st.sidebar:
        rag_initialization_cfgs = configs["rag_initialization"]
        rag_files_found = os.path.exists(
            rag_initialization_cfgs["index_path"]
        ) and os.path.exists(rag_initialization_cfgs["db_path"])

        if rag_initialization_cfgs["use_rag"]:
            if rag_files_found:
                rag_str = "Use RAG ✅"
            else:
                rag_str = "Should use RAG, but files can't be found ⚠️"
        else:
            rag_str = "Don't use RAG ❌"

        st.markdown(
            f"""
        ## Configurations  
        Device: *{available_devices[rag_initialization_cfgs["device"]]}*  
        Model: *{MODELS[rag_initialization_cfgs["model_idx"]]["name"]}* 
        #### RAG   
        {rag_str}
        """
        )

        if rag_initialization_cfgs["use_rag"]:
            st.markdown(
                f"""
            Database Path: *{rag_initialization_cfgs["db_path"]}*  
            Index Path: *{rag_initialization_cfgs["index_path"]}*    
            Inference Type: **{INFERENCE_TYPE_MAP[configs["inference_type"]]}**  
            Number of documents to retrieve: **{configs["retrieved_docs"]}**
            """
            )

        with st.form("configs_form", border=False) as form:
            st.markdown("## Edit Configurations")
            new_device = st.selectbox(
                "Device",
                list(available_devices.keys()),
                index=list(available_devices.keys()).index(
                    rag_initialization_cfgs["device"]
                ),
                format_func=lambda x: available_devices[x],
            )
            new_model_idx = st.selectbox(
                "Model",
                [i for i in range(len(MODELS))],
                index=rag_initialization_cfgs["model_idx"],
                format_func=lambda x: MODELS[x]["name"],
            )
            st.markdown("#### RAG")
            new_use_rag = st.checkbox(
                "Use RAG", value=rag_initialization_cfgs["use_rag"]
            )
            new_db_path = st.text_input("Database Path", DB_PATH_DEFAULT)
            new_index_path = st.text_input("Vector Index Path", INDEX_PATH_DEFAULT)
            new_inference_type = st.selectbox(
                "Inference Type",
                INFERENCE_TYPES,
                index=INFERENCE_TYPES.index("naive"),
                format_func=lambda x: INFERENCE_TYPE_MAP[x],
            )
            new_retrieved_docs = st.number_input(
                "Number of retrieved documents", 1, 10, configs["retrieved_docs"]
            )

            submitted = st.form_submit_button("Apply")
            if submitted:
                configs["rag_initialization"] = {
                    "model_idx": new_model_idx,
                    "db_path": new_db_path,
                    "index_path": new_index_path,
                    "device": new_device,
                    "use_rag": new_use_rag,
                }
                configs["inference_type"] = new_inference_type
                configs["retrieved_docs"] = new_retrieved_docs

                st.session_state["configs"] = configs
                st.rerun()
