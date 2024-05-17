from copy import deepcopy
from typing import Optional, List, Dict

import torch
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
        faiss_kwargs = (
            faiss_kwargs
            if faiss_kwargs is not None
            else {
                "dataset": None,
                "embedder": None,
            }
        )

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
            "do_sample": False,
            # "temperature": 0.1,
        }

    def craft_replug_query(self, query: str, doc: str) -> str:
        out = ""
        out += "Context:\n"
        out += doc
        out += "\n\n"
        out += "Query:\n"
        out += query
        return out

    def get_logits_replug(
        self,
        queries: List[str],
    ) -> List[torch.Tensor]:
        """
        Probably not useful. This function takes a batch of queries. For each of them, it retrieves
        documents with faiss, feeds the query with each document to the model, and computes the average
        of the logits weighted by the scores of the retrieved documents.
        """
        retrieved_for_every_query = self.faiss.search_multiple_texts(queries)
        avg_logits_for_every_query = []
        for query, retrieved in zip(queries, retrieved_for_every_query):
            queries_with_context = []
            scores = []
            for tup in retrieved:
                doc, score = tup
                scores.append(score)
                query_with_context = self.craft_replug_query(query, doc)
                queries_with_context.append(query_with_context)
            logits = self.llm.get_logits(
                queries_with_context
            )  # (num_docs, seq_len, vocab_size)
            scores = torch.tensor(scores)
            scores /= scores.sum()
            avg_logits = (logits * scores[:, None, None]).sum(
                dim=0
            )  # (seq_len, vocab_size)
            avg_logits_for_every_query.append(avg_logits)
        return avg_logits_for_every_query  # (num_queries, seq_len, vocab_size)

    def naive_inference(
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
            updated_histories = []
            for history, query in zip(histories, queries):
                if self.use_rag is False:
                    updated_histories.append(join_messages_query_no_rag(history, query))
                else:
                    retrieved = self.faiss.search_text(query)
                    # here we would do some preprocessing on the retrieved documents
                    updated_histories.append(
                        join_messages_query_rag(history, query, retrieved)
                    )

        elif isinstance(queries, str):
            if self.use_rag is False:
                updated_histories = join_messages_query_no_rag(histories, queries)
            else:
                retrieved = self.faiss.search_text(queries)
                # here we would do some preprocessing on the retrieved documents
                updated_histories = join_messages_query_rag(
                    histories, queries, retrieved
                )
        else:
            raise TypeError(
                "histories and queries must be either both strings or both lists of strings"
            )

        rag_config = deepcopy(self.llm_config)
        if kwargs:
            rag_config.update(kwargs)
        response = self.llm.inference(updated_histories, rag_config)
        return response

    def add_arxiv_paper(self, paper):
        raise NotImplementedError


if __name__ == "__main__":
    print("i'm alive")
    from backend.model.llm_handler import DEFAULT_MODEL

    print("DEFAULT_MODEL:", DEFAULT_MODEL)

    rag_handler = RagHandler(
        model_name=DEFAULT_MODEL,
        device="cpu",
        use_rag=True,
    )

    print("rag_handler:", rag_handler)

    histories = [
        [
            {
                "role": "user",
                "content": "Can you provide ways to eat combinations of bananas and dragonfruits?",
            },
            {
                "role": "assistant",
                "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey.",
            },
            {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
        ]
    ]

    print("histories:", histories)

    queries = [
        "What about solving an 2x + 3 = 7 equation?",
    ]

    print("queries:", queries)

    responses = rag_handler.naive_inference(histories, queries)
    print(responses)
