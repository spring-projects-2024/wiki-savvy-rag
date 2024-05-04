from copy import deepcopy
from typing import Optional, List, Dict
from backend.model.llm_handler import LLMHandler
from backend.vector_database.faiss_wrapper import FaissWrapper
from backend.model.prompt_utils import (
    join_messages_query_no_rag,
    join_messages_query_rag,
)


class RagHandler:
    def __init__(
        self,
        model_name: str,
        device: str,
        use_rag: bool = True,
        llm_config: Optional[dict] = None,
        model_kwargs: Optional[dict] = None,
        tokenizer_kwargs: Optional[dict] = None,
        faiss_kwargs: Optional[dict] = None,
    ):
        model_kwargs = model_kwargs if model_kwargs is not None else {}
        tokenizer_kwargs = tokenizer_kwargs if tokenizer_kwargs is not None else {}
        faiss_kwargs = faiss_kwargs if faiss_kwargs is not None else {}

        if llm_config is None:
            llm_config = self.get_default_llm_config()
        self.llm_config = llm_config
        self.faiss = FaissWrapper(device=device, **faiss_kwargs)
        self.llm = LLMHandler(
            device=device,
            model_name=model_name,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
        )
        self.use_rag = use_rag

    @staticmethod
    def get_default_llm_config():
        # TODO: choose default values
        return {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
            "return-full-text": False,
        }

    def inference(
        self,
        histories: List[List[Dict]] | List[Dict],
        queries: List[str] | str,
        **kwargs
    ) -> List[str] | str:

        # we are assuming that queries and histories are coherent in type
        # we support both batch inference and single queries, but we assume that if queries is a string then histories
        # is a list of dictionaries containing the chat history
        # if queries is a list of strings then histories is a list of lists of dictionaries containing the chat history
        if isinstance(queries, list):
            messages = []
            for history, query in zip(histories, queries):
                if self.use_rag is False:
                    messages.append(join_messages_query_no_rag(history, query))
                else:
                    retrieved = self.faiss.search_text(query)
                    # here we would do some preprocessing on the retrieved documents
                    messages.append(join_messages_query_rag(history, query, retrieved))

        elif isinstance(queries, str):
            if self.use_rag is False:
                messages = join_messages_query_no_rag(histories, queries)
            else:
                retrieved = self.faiss.search_text(queries)
                # here we would do some preprocessing on the retrieved documents
                messages = join_messages_query_rag(histories, queries, retrieved)
        else:
            raise TypeError(
                "histories and queries must be either both strings or both lists of strings"
            )

        rag_config = deepcopy(self.llm_config)
        if kwargs:
            rag_config.update(kwargs)
        response = self.llm.inference(messages, rag_config)
        return response

    def add_arxiv_paper(self, paper):
        raise NotImplementedError
