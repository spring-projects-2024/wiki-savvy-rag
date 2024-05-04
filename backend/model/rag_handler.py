from backend.model.llm_handler import LLMHandler
from backend.vector_database.faiss_wrapper import FaissWrapper
from backend.model.prompt_utils import (
    join_messages_query_no_rag,
    join_messages_query_rag,
)


class RagHandler:
    def __init__(self, device):
        self.faiss = FaissWrapper(device=device)
        self.llm = LLMHandler(device)

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
