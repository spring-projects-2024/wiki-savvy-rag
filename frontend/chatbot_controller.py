import os
from typing import List
import json
import time
from typing import Dict, List

import streamlit as st
from backend.model.rag_handler import TOP_K, TOP_P, RagHandler
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

INFERENCE_TYPES = ["naive", "autoregressive", "mock"]

MODEL_DEFAULT = 0
DB_PATH_DEFAULT = "./scripts/dataset/data/dataset.db"
INDEX_PATH_DEFAULT = "./scripts/vector_database/data/default.index"
DEVICE_DEFAULT = "cuda:0"
USE_RAG_DEFAULT = True
RETRIEVED_DOCS_DEFAULT = 5
INFERENCE_TYPE_DEFAULT = "naive"
DECODING_STRATEGY_DEFAULT = "top_k"


@st.cache_resource(show_spinner="Loading Chatbot. It could take a while...")
def load_controller():
    return ChatbotController()


class ChatbotController:
    def __init__(self):
        self.configs = None
        self.rag = None

    def _init_rag_handler(self):
        cfgs = self.configs["rag_initialization"]
        use_rag = self.real_use_rag()
        if use_rag:
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
            use_rag=use_rag,
            use_qlora=model["use_qlora"],
        )

    def real_use_rag(self):
        cfgs = self.configs["rag_initialization"]
        rag_files_found = os.path.exists(cfgs["index_path"]) and os.path.exists(
            cfgs["db_path"]
        )
        return cfgs["use_rag"] and rag_files_found

    def update_configs(self, configs: Dict[str, str]):
        if (
            self.configs is None
            or self.configs["rag_initialization"] != configs["rag_initialization"]
        ):
            if self.rag is not None:
                # for semplicity, delete and reinitialize the rag handler
                del self.rag

            self.configs = configs
            self._init_rag_handler()
        else:
            self.configs = configs

    def _string_generator(self, strlist: List[str]):
        for s in strlist:
            yield s
            time.sleep(0.008)

    def inference(self, history: List[Dict], query: str):
        if self.configs["inference_type"] == "mock":
            return self._mock_inference(history, query)
        if not self.real_use_rag() or self.configs["inference_type"] == "naive":
            return self._naive_inference(history, query)
        return self._autoregressive_inference(history, query)

    def _naive_inference(
        self,
        history: List[Dict],
        query: str,
    ):
        kwargs = {}
        if self.configs["decoding_strategy"] == "greedy":
            kwargs["do_sample"] = False
        elif self.configs["decoding_strategy"] == "top_k":
            kwargs["do_sample"] = True
            kwargs["top_k"] = TOP_K
        elif self.configs["decoding_strategy"] == "top_p":
            kwargs["do_sample"] = True
            kwargs["top_p"] = TOP_P

        response, retrieved_docs = self.rag.naive_inference_with_retrieved_docs(
            histories=history,
            queries=query,
            n_docs=self.configs["retrieved_docs"],
            **kwargs,
        )

        assert isinstance(response, str)

        return self._string_generator(response), retrieved_docs

    def _autoregressive_inference(
        self,
        history: List[Dict],
        query: str,
    ):
        return self.rag.autoregressive_inference(
            query=query,
            n_docs=self.configs["retrieved_docs"],
            return_generator=True,
            decoding_strategy=self.configs["decoding_strategy"],
        )

    def _mock_inference(
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
        return self._string_generator(response), retrieved_docs

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
