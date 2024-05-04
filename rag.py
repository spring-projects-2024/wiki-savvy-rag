from llm import LLMHandler
from retriever.faiss_wrapper import FaissWrapper


class Rag:
    def __init__(self, device):
        self.faiss = FaissWrapper(device=device)
        self.llm = LLMHandler(device)

    def inference(self, query, history) -> str:
        raise NotImplementedError

    def add_arxiv_paper(self, paper):
        raise NotImplementedError
