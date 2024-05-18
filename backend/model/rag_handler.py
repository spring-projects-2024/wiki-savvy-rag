import time
from copy import deepcopy
from typing import Optional, List, Dict, Tuple, Iterable

import numpy as np
import torch
from backend.model.llm_handler import LLMHandler
from backend.vector_database.faiss_wrapper import FaissWrapper
from backend.vector_database.dataset import Dataset
from backend.vector_database.embedder_wrapper import EmbedderWrapper
from backend.model.prompt_utils import (
    join_messages_query_no_rag,
    join_messages_query_rag,
)


# TODO: write inference averaging probabilities over the retrieved documents:
# need to do it autoregressively, so we need to get from llm both logits and key value caches
# for the next token generation. The forward pass of the llm already returns the key value caches in
# the output in addition to the logits. We need to pass those around.
# good if we decouple getting probabilities from decoding, so that we can choose between greedy, beam search,
# having a temperature, etc.


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

        self.llm_config = self.get_default_llm_config()

        if llm_config is not None:
            self.llm_config.update(llm_config)

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
            "do_sample": False,
            # "temperature": 0.1,
        }

    # def craft_replug_query(self, query: str, doc: str) -> str:
    #     return f"Context:\n{doc}\n\nQuery:\n{query}"

    def craft_autoregressive_query(self, query: str, doc: str) -> str:
        # return f"Context:\n{doc}\n\nQuery:\n{query}\n\nAnswer:"
        return f"Context:\n{doc}\n\nQuery:\n{query}\n"

    def craft_training_prompt(self, query: str, doc: str, answer: str) -> str:
        return f"Context:\n{doc}\n\nQuery:\n{query}\n\nAnswer:\n{answer}"

    def prepare_prompt(
        self, query: str, doc: str, answer: str
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Assemble a prompt with the context, query, and answer.
        Return a dictionary with keys "input_ids", "attention_mask", "query_plus_context_length".
        "input_ids" is a tensor of shape (1, seq_len) with the tokenized prompt.
        "attention_mask" is a tensor of shape (1, seq_len) with 1s in positions corresponding to tokens
        and 0s in positions corresponding to padding tokens.
        "query_plus_context_length" is the length of the query and context part of the prompt. To train,
        look only at the logits corresponding to the answer part of the prompt.
        """
        header = f"Context:\n{doc}\n\nQuery:\n{query}\n\nAnswer:\n"
        header_tokens = self.llm.tokenizer(header, return_tensors="pt", padding=False)
        query_plus_context_length = header_tokens["input_ids"].shape[1]
        answer_tokens = self.llm.tokenizer(answer, return_tensors="pt", padding=False)
        # if the tokenizer adds a <s> token at the beginning, we remove it
        if answer_tokens["input_ids"][0] == self.llm.tokenizer.encode("<s>"):
            answer_tokens["input_ids"] = answer_tokens["input_ids"][1:]
            answer_tokens["attention_mask"] = answer_tokens["attention_mask"][1:]
        input_ids = torch.cat(header_tokens["input_ids"], answer_tokens["input_ids"])
        attention_mask = torch.cat(
            header_tokens["attention_mask"], answer_tokens["attention_mask"]
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }, query_plus_context_length

    def compute_probabilities_for_training(
        self,
        forward_output: Dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        This method takes the output of replug_forward, computes the probabilities of the tokens
        by aggregating the probabilities of the retrieved documents, and returns the probabilities
        together with a mask indicating which tokens correspond to the answer part of the prompt.
        """
        logits = forward_output["logits"]  # (num_docs, seq_len, vocab_size)
        scores = forward_output["scores"]
        context_lengths = forward_output["context_lengths"]
        # we want to keep only the logits corresponding to the answer part of the prompt
        answer_lengths = logits.shape[1] - context_lengths
        mask = (
            torch.arange(logits.shape[1])[None, :] < answer_lengths[:, None]
        )  # (num_docs, seq_len)
        probas = torch.nn.functional.softmax(
            logits, dim=-1
        )  # (num_docs, seq_len, vocab_size)
        probas = (probas * scores[:, None, None]).sum(dim=0)  # (seq_len, vocab_size)
        return {
            "probas": probas,
            "answer_mask": mask,
        }

    def logits_to_weighted_probas(self, logits: torch.Tensor, scores: torch.Tensor):
        """

        :param logits:
        :param scores: Need to be normalized
        :return:
        """
        probas = torch.nn.functional.softmax(
            logits, dim=-1
        )  # (num_docs, seq_len, vocab_size)
        return (probas * scores[:, None, None]).sum(dim=0)  # (seq_len, vocab_size)

    def replug_forward(
        self,
        query: str,
        answer: str,
    ) -> Dict[str, torch.Tensor]:
        """
        logits: (num_docs, seq_len, vocab_size). The logits for each retrieved document, with
        the given query and answer.
        scores: (num_docs,). The scores of the retrieved documents, normalized.
        Used to average probabilities later.
        context_lengths: (num_docs,). The length of the query and context part of the prompt. These will
        be used to look only at the logits corresponding to the answer part of the prompt.
        """
        retrieved_docs = self.faiss.search_text(query)
        prompts = []
        scores = []
        context_lengths = []
        for doc, score in retrieved_docs:
            tokenized_prompt, query_plus_context_length = self.prepare_prompt(
                query, doc, answer
            )
            prompts.append(tokenized_prompt)
            scores.append(score)
            context_lengths.append(query_plus_context_length)
        # make a batch of prompts (get_logits expects a single dict with batched tensors as values)
        input_ids = torch.stack([prompt["input_ids"] for prompt in prompts])
        attention_mask = torch.stack([prompt["attention_mask"] for prompt in prompts])
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        logits = self.llm.get_logits(batch)  # (num_docs, seq_len, vocab_size)
        scores = torch.tensor(scores)  # (num_docs,)
        scores /= scores.sum()
        context_lengths = torch.tensor(context_lengths)  # (num_docs,)
        return {
            "logits": logits,
            "scores": scores,
            "context_lengths": context_lengths,
        }

    # def get_logits_replug(
    #     self,
    #     queries: List[str],
    # ) -> List[torch.Tensor]:
    #     """
    #     Probably not useful. This function takes a batch of queries. For each of them, it retrieves
    #     documents with faiss, feeds the query with each document to the model, and computes the average
    #     of the logits weighted by the scores of the retrieved documents.
    #     Problems:
    #     - for training, we probably need to have also the answer concatenated to the query
    #     - the averaging is done over the logits, but we probably want to average over the probabilities
    #     """
    #     retrieved_for_every_query = self.faiss.search_multiple_texts(queries)
    #     avg_logits_for_every_query = []
    #     for query, retrieved in zip(queries, retrieved_for_every_query):
    #         queries_with_context = []
    #         scores = []
    #         for tup in retrieved:
    #             doc, score = tup
    #             scores.append(score)
    #             query_with_context = self.craft_replug_query(query, doc)
    #             queries_with_context.append(query_with_context)
    #         logits = self.llm.get_logits(
    #             queries_with_context
    #         )  # (num_docs, seq_len, vocab_size)
    #         scores = torch.tensor(scores)
    #         scores /= scores.sum()
    #         avg_logits = (logits * scores[:, None, None]).sum(
    #             dim=0
    #         )  # (seq_len, vocab_size)
    #         avg_logits_for_every_query.append(avg_logits)
    #     return avg_logits_for_every_query  # (num_queries, seq_len, vocab_size)

    @torch.no_grad()
    def autoregressive_generation_iterator_with_retrieved_docs(
        self, query: str, max_length=50
    ) -> Tuple[Iterable[str], List[Tuple[str, float]]]:
        """
        method that does autoregressive generation. It accepts a query. First it retrieves passages.
        For each passage it maintains its past key values. For each token, it calls k times the previous
         method and the previous method to get probas for next token. Then does greedy generation.
        :param query:
        :return:
        """
        retrieved_docs = self.faiss.search_text(query)

        def generator():
            answer = ""
            autoregressive_state = [
                {
                    "past_key_values": None,
                    "doc": doc,
                    "similarity": similarity,
                    "query": self.craft_autoregressive_query(query, doc),
                    "tokenized_query": self.llm.tokenizer(
                        self.craft_autoregressive_query(query, doc),
                        return_tensors="pt",
                        padding=False,
                    )["input_ids"],
                }
                for doc, similarity in retrieved_docs
            ]

            similarities = torch.tensor(
                [state["similarity"] for state in autoregressive_state]
            )

            similarities /= similarities.sum()

            while len(answer) < max_length:
                all_logits = []

                for state in autoregressive_state:
                    result = self.llm.model(
                        state["tokenized_query"],
                        past_key_values=state["past_key_values"],
                        use_cache=True,
                    )

                    logits: torch.Tensor = result.logits
                    logits = logits[:, -1, :]

                    state["past_key_values"] = result.past_key_values

                    all_logits.append(logits)

                # compute the average of the logits weighted by the scores of the retrieved documents
                torch_logits = torch.stack(all_logits)

                probs = self.logits_to_weighted_probas(torch_logits, similarities)

                # get the token with the highest probability
                next_token = torch.argmax(probs)

                # get the token from the tokenizer
                next_token_str = self.llm.tokenizer.decode(next_token)
                answer += next_token_str

                for state in autoregressive_state:
                    state["query"] += next_token_str
                    state["tokenized_query"] = torch.tensor([[next_token]])  # type: ignore

                yield next_token_str

        return generator(), retrieved_docs

    @torch.no_grad()
    def autoregressive_generation_iterator(
        self, query: str, max_length=50
    ) -> Iterable[str]:
        generator, _ = self.autoregressive_generation_iterator_with_retrieved_docs(
            query, max_length=max_length
        )
        return generator

    def auto_regressive_generation(self, query: str, max_length=50) -> str:
        return "".join(self.autoregressive_generation_iterator(query, max_length))

    def naive_inference_with_retrieved_docs(
        self,
        histories: List[List[Dict]] | List[Dict],
        queries: List[str] | str,
        **kwargs,
    ) -> Tuple[
        List[str] | str, List[List[Tuple[str, float]]] | List[Tuple[str, float]]
    ]:
        # we are assuming that queries and histories are coherent in type
        # we support both batch inference and single queries, but we assume that if queries is a string then histories
        # is a list of dictionaries containing the chat history
        # if queries is a list of strings then histories is a list of lists of dictionaries containing the chat history

        if isinstance(queries, list):
            updated_histories = []
            retrieved = []
            for history, query in zip(histories, queries):
                if self.use_rag is False:
                    updated_histories.append(join_messages_query_no_rag(history, query))
                else:
                    retrieved_now = self.faiss.search_text(query)
                    # here we would do some preprocessing on the retrieved documents
                    updated_histories.append(
                        join_messages_query_rag(history, query, retrieved_now)
                    )
                    retrieved.append(retrieved_now)

        elif isinstance(queries, str):
            if self.use_rag is False:
                updated_histories = join_messages_query_no_rag(histories, queries)
                retrieved = []
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

        return response, retrieved

    def naive_inference(
        self,
        histories: List[List[Dict]] | List[Dict],
        queries: List[str] | str,
        **kwargs,
    ) -> List[str] | str:
        response, _ = self.naive_inference_with_retrieved_docs(
            histories, queries, **kwargs
        )

        return response

    def add_arxiv_paper(self, paper):
        raise NotImplementedError


INDEX_PATH = "scripts/vector_database/data/default.index"
DB_PATH = "scripts/dataset/data/dataset.db"

if __name__ == "__main__":
    print("i'm alive")
    embedder = EmbedderWrapper("cpu")
    dataset = Dataset(db_path=DB_PATH)
    rag_handler = RagHandler(
        model_name="Minami-su/Qwen1.5-0.5B-Chat_llamafy",
        device="cpu",
        use_rag=True,
        faiss_kwargs={
            "index_path": INDEX_PATH,
            "dataset": dataset,
            "embedder": embedder,
        },
    )

    query = "What is Anarchism?"

    # t = time.time()
    # a = rag_handler.autoregressive_generation(query, 20)
    # print(f"Time taken: {time.time() - t}")

    # t = time.time()
    a = rag_handler.auto_regressive_generation(query, 100)
    # print(f"Time taken: {time.time() - t}")

    print(a)
