import os
from typing import List
import json
import time
from typing import Dict, List

import streamlit as st
from backend.model.rag_handler import RagHandler
from backend.vector_database.dataset import DatasetSQL
from backend.vector_database.embedder_wrapper import EmbedderWrapper

MODELS = [
    {
        "name": "Minami-su Qwen1.5 0.5B Chat Llamafy",
        "model": "Minami-su/Qwen1.5-0.5B-Chat_llamafy",
        "use_qlora": False,
    },
    {
        "name": "Microsoft Phi-3 Mini 128k Instruct (QLoRA)",
        "model": "microsoft/phi-3-mini-128k-instruct",
        "use_qlora": True,
    },
]

MODEL_DEFAULT = 0
DB_PATH_DEFAULT = "./scripts/dataset/data/dataset.db"
INDEX_PATH_DEFAULT = "./scripts/vector_database/data/default.index"
DEVICE_DEFAULT = "cuda:0"
USE_RAG_DEFAULT = True


@st.cache_resource(show_spinner="Loading Chatbot. It could take a while...")
def load_controller():
    return ChatbotController()


class ChatbotController:
    def __init__(self):
        self.configs = None
        self.rag = None

    def _init_rag_handler(self):
        cfgs = self.configs
        rag_files_found = os.path.exists(cfgs["index_path"]) and os.path.exists(
            cfgs["db_path"]
        )
        if cfgs["use_rag"] and rag_files_found:
            dataset = DatasetSQL(db_path=cfgs["db_path"])
            embedder = EmbedderWrapper(cfgs["device"])
            faiss_kwargs = {
                "index_path": cfgs["index_path"],
                "dataset": dataset,
                "embedder": embedder,
            }
        else:
            faiss_kwargs = None

        model = MODELS[cfgs["model_idx"]]

        self.rag = RagHandler(
            model_name=model["model"],
            device=cfgs["device"],
            llm_kwargs={
                "torch_dtype": "auto",
                "do_sample": True,
            },
            faiss_kwargs=faiss_kwargs,
            use_rag=cfgs["use_rag"] and rag_files_found,
            use_qlora=model["use_qlora"],
        )

    def should_update_configs(self, configs: Dict[str, str]):
        return self.configs != configs

    def update_configs(self, configs: Dict[str, str]):
        if self.configs == configs:
            return

        self.configs = configs
        if self.rag is not None:
            # for semplicity, delete and reinitialize the rag handler
            del self.rag
        self._init_rag_handler()

    def _string_generator(self, strlist: List[str]):
        for s in strlist:
            yield s
            time.sleep(0.008)

    def naive_inference(
        self,
        history: List[Dict],
        query: str,
    ):
        response, retrieved_docs = self.rag.naive_inference_with_retrieved_docs(
            histories=history,
            queries=query,
        )

        assert isinstance(response, str)

        return self._string_generator(response), retrieved_docs

    def autoregressive_inference(
        self,
        history: List[Dict],
        query: str,
    ):
        return self.rag.autoregressive_generation_iterator_with_retrieved_docs(
            query=query
        )

    def mock_inference(
        self,
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
        return self.string_generator(response), retrieved_docs

    def _post_process_titles(self, text: str):
        """This function is necessary because the json in titles is not formatted properly"""
        return (
            text.replace("['", '["')
            .replace("',", '",')
            .replace(", '", ', "')
            .replace(",'", ',"')
            .replace("']", '"]')
        )

    def build_retrieved_docs_str(self, retrieved_docs: List[Dict]):
        retrieved_docs_str = ""
        for i, (doc, score) in enumerate(retrieved_docs):
            titles = " > ".join(json.loads(self._post_process_titles(doc["titles"])))
            retrieved_docs_str += f"""  
            **Chunk {i+1}**: *{titles}* (score: {score})"""
        retrieved_docs_str += f"\n\n"
        return retrieved_docs_str
