import os
import torch
import streamlit as st
from backend.model.rag_handler import DECODING_STRATEGIES, TOP_K, TOP_P
from chatbot_controller import (
    CUSTOM_MODEL_PATH_DEFAULT,
    DB_PATH_DEFAULT,
    DECODING_STRATEGY_DEFAULT,
    DEVICE_DEFAULT,
    INDEX_PATH_DEFAULT,
    INFERENCE_TYPE_DEFAULT,
    INFERENCE_TYPES,
    MOCK_RESPONSES_DEFAULT,
    MODEL_DEFAULT,
    MODELS,
    RETRIEVED_DOCS_DEFAULT,
)

INFERENCE_TYPE_MAP = {
    "naive": "Naive",
    "replug": "REPLUG",
}

DECODING_STRATEGY_MAP = {
    "top_k": f"Top-K ({TOP_K} tokens)",
    "top_p": f"Top-P ({TOP_P*100}%)",
    "greedy": "Greedy",
}


@st.cache_data
def load_available_devices():
    """Loads the available devices (CPU and CUDA devices) and returns them as a dictionary."""
    devices = {
        "cpu": "CPU",
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices[f"cuda:{i}"] = torch.cuda.get_device_name(i)

    return devices


def build_sidebar():
    """Builds the sidebar with the configurations form."""
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
                "custom_model_path": CUSTOM_MODEL_PATH_DEFAULT,
            },
            "inference_type": INFERENCE_TYPE_DEFAULT,
            "retrieved_docs": RETRIEVED_DOCS_DEFAULT,
            "decoding_strategy": DECODING_STRATEGY_DEFAULT,
            "mock_responses": MOCK_RESPONSES_DEFAULT,
            "use_arxiv": False,
        }

    configs = st.session_state["configs"].copy()

    with st.sidebar:
        rag_initialization_cfgs = configs["rag_initialization"]
        rag_files_found = os.path.exists(
            rag_initialization_cfgs["index_path"]
        ) and os.path.exists(rag_initialization_cfgs["db_path"])

        model = MODELS[rag_initialization_cfgs["model_idx"]]

        if rag_initialization_cfgs["use_rag"]:
            if rag_files_found:
                rag_str = "Use RAG ✅"
            else:
                rag_str = "Should use RAG, but files can't be found ⚠️"
        else:
            rag_str = "Don't use RAG ❌"

        if configs["mock_responses"]:
            mock_str = "Currently using mock responses 🤖"
        else:
            mock_str = ""

        st.markdown(
            f"""
            ## Configurations  
            {mock_str}  
            Device: *{available_devices[rag_initialization_cfgs["device"]]}*  
            Model: *{model["name"]}*  
            Finetuned Model Path: *{rag_initialization_cfgs['custom_model_path']}*  
            Decoding Strategy: **{DECODING_STRATEGY_MAP[configs["decoding_strategy"]]}**
            
            #### RAG 
            {rag_str}  
            Database Path: *{rag_initialization_cfgs["db_path"]}*  
            Index Path: *{rag_initialization_cfgs["index_path"]}*    
            Inference Type: **{INFERENCE_TYPE_MAP[configs["inference_type"]]}**  
            Number of documents to retrieve: **{configs["retrieved_docs"]}**
            """
        )

        with st.form("configs_form", border=False):
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
            new_custom_model_path = st.text_input(
                "Finetuned Model Path", rag_initialization_cfgs["custom_model_path"]
            )
            new_decoding_strategy = st.selectbox(
                "Decoding Strategy",
                DECODING_STRATEGIES,
                index=DECODING_STRATEGIES.index(configs["decoding_strategy"]),
                format_func=lambda x: DECODING_STRATEGY_MAP[x],
            )
            new_mock_responses = st.checkbox(
                "Use Mock Responses", value=configs["mock_responses"]
            )
            st.markdown("#### RAG")
            new_use_rag = st.checkbox(
                "Use RAG", value=rag_initialization_cfgs["use_rag"]
            )
            new_db_path = st.text_input(
                "Database Path", rag_initialization_cfgs["db_path"]
            )
            new_index_path = st.text_input(
                "Vector Index Path", rag_initialization_cfgs["index_path"]
            )
            new_inference_type = st.selectbox(
                "Inference Type",
                INFERENCE_TYPES,
                index=INFERENCE_TYPES.index(configs["inference_type"]),
                format_func=lambda x: INFERENCE_TYPE_MAP[x],
            )
            new_retrieved_docs = st.number_input(
                "Number of retrieved documents", 1, 10, configs["retrieved_docs"]
            )

            new_use_arxiv = st.checkbox(
                "Use ArXiv (Experimental Feature ⚠️)",
                value=configs["use_arxiv"],
            )

            submitted = st.form_submit_button("Apply")
            if submitted:
                configs["rag_initialization"] = {
                    "model_idx": new_model_idx,
                    "db_path": new_db_path,
                    "index_path": new_index_path,
                    "device": new_device,
                    "use_rag": new_use_rag,
                    "custom_model_path": new_custom_model_path,
                }
                configs["inference_type"] = new_inference_type
                configs["retrieved_docs"] = new_retrieved_docs
                configs["decoding_strategy"] = new_decoding_strategy
                configs["mock_responses"] = new_mock_responses
                configs["use_arxiv"] = new_use_arxiv

                st.session_state["configs"] = configs
                st.rerun()
