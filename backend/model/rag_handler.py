from backend.model.llm_handler import LLMHandler
from backend.vector_database.faiss_wrapper import FaissWrapper


class RagHandler:
    def __init__(self, device):
        self.faiss = FaissWrapper(device=device)
        self.llm = LLMHandler(device)

    def inference(self, query, history) -> str:
        raise NotImplementedError

    def add_arxiv_paper(self, paper):
        raise NotImplementedError
