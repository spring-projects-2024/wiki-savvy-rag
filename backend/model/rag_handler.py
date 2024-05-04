from copy import deepcopy
from typing import Optional, List
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
        rag_config: Optional[dict] = None,
        model_kwargs: Optional[dict] = None,
        tokenizer_kwargs: Optional[dict] = None,
        faiss_kwargs: Optional[dict] = None,
    ):
        for kwargs in (model_kwargs, tokenizer_kwargs, faiss_kwargs):
            if kwargs is None:
                kwargs = {}
        if rag_config is None:
            rag_config = self.get_default_rag_config()
        self.rag_config = rag_config
        self.faiss = FaissWrapper(device=device, **faiss_kwargs)
        self.llm = LLMHandler(
            device=device,
            model_name=model_name,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
        )

    def get_default_rag_config(self):
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
        histories: List[str] | str,
        queries: List[str] | str,
        use_rag=True,
        **kwargs
    ) -> str | List[str]:

        if isinstance(histories, list) and isinstance(queries, list):
            messages = []
            for history, query in zip(histories, queries):
                if use_rag is False:
                    messages.append(join_messages_query_no_rag(history, query))
                else:
                    retrieved = self.faiss.search_text(query)
                    # here we would do some preprocessing on the retrieved documents
                    messages.append(join_messages_query_rag(history, query, retrieved))

        elif isinstance(histories, str) and isinstance(queries, str):
            if use_rag is False:
                messages = join_messages_query_no_rag(histories, queries)
            else:
                retrieved = self.faiss.search_text(queries)
                # here we would do some preprocessing on the retrieved documents
                messages = join_messages_query_rag(histories, queries, retrieved)
        else:
            raise TypeError(
                "histories and queries must be either both strings or both lists of strings"
            )

        rag_config = deepcopy(self.rag_config)
        if kwargs:
            rag_config.update(kwargs)
        response = self.llm.inference(messages, rag_config)
        return response

    def add_arxiv_paper(self, paper):
        raise NotImplementedError
