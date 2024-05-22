import os
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
DEVICE_DEFAULT = "cpu"
USE_RAG_DEFAULT = True
RETRIEVED_DOCS_DEFAULT = 5
INFERENCE_TYPE_DEFAULT = "naive"
DECODING_STRATEGY_DEFAULT = "top_k"


@st.cache_resource(show_spinner="Loading Chatbot. It could take a while...")
def load_controller():
    return ChatbotController()


class ChatbotController:
    """This class is responsible for handling the chatbot logic. It initializes the RAG model with
    the user-defined configurations and delegates the inference process to the RAG model.

    Attributes:
    - configs: Dict[str, str] - a dictionary containing the user-defined configurations. The configuration keys are:
        - rag_initialization: Dict[str, str] - a dictionary containing the RAG model initialization configurations. The keys are:
            - model_idx: int - the index in the MODELS list of the model to use (default: 0)
            - db_path: str - the path to the SQLite database file (default: "./scripts/dataset/data/dataset.db")
            - index_path: str - the path to the Faiss index file (default: "./scripts/vector_database/data/default.index")
            - device: str - the device to run the model on (default: "cpu")
            - use_rag: bool - whether to enhance the query with retrieved documents (default: True)
        - inference_type: str - the type of inference to perform. It can be "naive", "autoregressive", or "mock" (default: "naive")
        - retrieved_docs: int - the number of retrieved documents to use in the inference process (default: 5)
        - decoding_strategy: str - the decoding strategy to use in the inference process. It can be "top_k", "top_p", or "greedy" (default: "top_k")
    - rag: RagHandler - an instance of the RagHandler class that handles the RAG model
    """

    def __init__(self):
        self.configs = None
        self.rag = None

    def _init_rag_handler(self):
        cfgs = self.configs["rag_initialization"]
        use_rag = self.real_use_rag()

        # Create the dataset and embedder only if RAG is used
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
        """Check if the RAG model can be used. It requires the index and db files to be present."""
        cfgs = self.configs["rag_initialization"]
        rag_files_found = os.path.exists(cfgs["index_path"]) and os.path.exists(
            cfgs["db_path"]
        )
        return cfgs["use_rag"] and rag_files_found

    def update_configs(self, configs: Dict[str, str]):
        """Update the configurations of the chatbot controller. If the RAG initialization configurations have changed,
        the RAG handler is reinitialized."""
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
        """Generator that yields the strings in the list with a delay of 0.008 seconds from token to token.
        Used to simulate a typing effect in the frontend."""
        for s in strlist:
            yield s
            time.sleep(0.008)

    def inference(self, history: List[Dict], query: str):
        """Main entry point for the inference process. It delegates the inference to the RAG model, based on the
        user-defined configurations. If the inference type is "mock", a mock response is returned instead.
        """
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
        """Perform a naive inference using the RAG model. See RagHandler.naive_inference for more details."""
        kwargs = {}
        if self.configs["decoding_strategy"] == "greedy":
            kwargs["do_sample"] = False
        elif self.configs["decoding_strategy"] == "top_k":
            kwargs["do_sample"] = True
            kwargs["top_k"] = TOP_K
        elif self.configs["decoding_strategy"] == "top_p":
            kwargs["do_sample"] = True
            kwargs["top_p"] = TOP_P

        response, retrieved_docs = self.rag.naive_inference(
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
        """Perform an autoregressive inference using the RAG model. See RagHandler.autoregressive_inference for more details."""
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
        """Return a mock response instead of performing an inference. Used for testing purposes."""

        response = self.mock_responses[len(history) % len(self.mock_responses)]
        retrieved_docs = (
            self.mock_docs[: self.configs["retrieved_docs"]]
            if self.real_use_rag()
            else []
        )

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

    def build_retrieved_docs_html(self, retrieved_docs: List[Dict]):
        """Builds the HTML string for the retrieved documents to be displayed in the frontend.
        It includes the titles of the documents and the score."""
        retrieved_docs_str = "<p style='font-size: 0.9em; color: rgb(75, 85, 99); background-color: #f9fafb; padding: 0.8rem; border-radius: 8px'>"
        for i, (doc, score) in enumerate(retrieved_docs):
            titles = " > ".join(json.loads(self._post_process_titles(doc["titles"])))
            retrieved_docs_str += f"""  
            <strong>Chunk {i+1}</strong>: {titles} (score: {score:.2f})</br>"""
        retrieved_docs_str += "</p>"
        return retrieved_docs_str

    mock_responses = [
        "Banana and dragonfruit salad is a delicious and healthy dish that you can make in minutes.",
        "Pina colada is a refreshing cocktail that is perfect for a hot summer day.",
        "Banana split is a classic dessert that is loved by people of all ages.",
        "Banana bread is a simple and delicious treat that you can enjoy for breakfast or as a snack.",
    ]

    mock_docs = [
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
        (
            {
                "titles": '["Banana cake", "Banana cake recipe", "Banana cake ingredients"]'
            },
            1,
        ),
        (
            {
                "titles": '["Banana muffin", "Banana muffin recipe", "Banana muffin ingredients"]'
            },
            1,
        ),
        (
            {
                "titles": '["Banana smoothie", "Banana smoothie recipe", "Banana smoothie ingredients"]'
            },
            1,
        ),
        (
            {
                "titles": '["Banana chips", "Banana chips recipe", "Banana chips ingredients"]'
            },
            1,
        ),
        (
            {
                "titles": '["Banana split", "Banana split recipe", "Banana split ingredients"]'
            },
            1,
        ),
    ]
