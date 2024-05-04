from typing import Optional
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
            model_kwargs: Optional[dict] = None, 
            tokenizer_kwargs: Optional[dict] = None, 
            faiss_kwargs: Optional[dict] = None,
        ):
        for kwargs in (model_kwargs, tokenizer_kwargs, faiss_kwargs):
            if kwargs is None:
                kwargs = {}
        self.faiss = FaissWrapper(device=device, **faiss_kwargs)
        self.llm = LLMHandler(
            device=device, 
            model_name=model_name, 
            model_kwargs=model_kwargs, 
            tokenizer_kwargs=tokenizer_kwargs
        )

    def inference(self, history, query, use_rag=True, **kwargs) -> str:

        if use_rag is False:
            messages = join_messages_query_no_rag(history, query)
        else:
            retrieved = self.faiss.search_text(query)
            # here we would do some preprocessing on the retrieved documents
            messages = join_messages_query_rag(history, query, retrieved)

        response = self.llm.inference(messages, kwargs)
        return response

    def add_arxiv_paper(self, paper):
        raise NotImplementedError
