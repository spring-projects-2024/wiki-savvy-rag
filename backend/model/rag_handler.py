from copy import deepcopy
from typing import Optional, List, Dict, Tuple, Iterable

from torch import nn
import torch
from transformers import BatchEncoding

from backend.model.llm_handler import LLMHandler
from backend.vector_database.faiss_wrapper import FaissWrapper
from backend.vector_database.dataset import DatasetSQL
from backend.vector_database.embedder_wrapper import EmbedderWrapper
from backend.model.prompt_utils import (
    join_messages_query_no_rag,
    join_messages_query_rag,
)

AUTOREGRESSIVE_GEN_MAX_LENGTH = 1000
DECODING_STRATEGIES = ["greedy", "top_k", "top_p"]
TOP_K = 50
TOP_P = 0.9


def compute_probabilities_for_training(
    forward_output: Dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    This method takes the output of replug_forward, computes the probabilities of the tokens
    by aggregating the probabilities of the retrieved documents, and returns the probabilities
    together with a mask indicating which tokens correspond to the answer part of the prompt.
    """
    logits = forward_output["logits"]  # (num_docs, seq_len, vocab_size)
    scores = forward_output["scores"]  # (num_docs,)
    answer_length = forward_output["answer_length"]  # (1,)
    answer_logits = logits[
        :, -answer_length:, :
    ]  # (num_docs, answer_length, vocab_size)
    answer_probas = torch.nn.functional.softmax(answer_logits, dim=-1)
    aggregated_probas = (answer_probas * scores[:, None, None]).sum(
        dim=0
    )  # (answer_length, vocab_size)
    return {
        "probas": aggregated_probas,
    }


def craft_training_prompt(query: str, doc: str, answer: str) -> str:
    return f"Context:\n{doc}\n\nQuery:\n{query}\n\nAnswer:\n{answer}"


class RagHandler(nn.Module):
    """
    Handler for a RAG model. It uses a FaissWrapper to retrieve documents based on user queries,
    and an LLMHandler to generate text.

    Attributes:
    - llm: an LLMHandler object.
    - faiss: a FaissWrapper object.
    - use_rag: whether to use the RAG model.
    - device: the device on which the model is loaded.
    - llm_generation_config: a dictionary with default configuration for the LLM model.
    """

    def __init__(
        self,
        model_name: str,
        device: str,
        use_rag: bool = True,
        llm_generation_config: Optional[dict] = None,
        llm_kwargs: Optional[dict] = None,
        tokenizer_kwargs: Optional[dict] = None,
        faiss_kwargs: Optional[dict] = None,
        use_qlora: bool = False,
    ):
        super().__init__()
        llm_kwargs = llm_kwargs if llm_kwargs is not None else {}
        tokenizer_kwargs = tokenizer_kwargs if tokenizer_kwargs is not None else {}
        faiss_kwargs = (
            faiss_kwargs
            if faiss_kwargs is not None
            else {
                "dataset": None,
                "embedder": None,
            }
        )
        self.faiss = FaissWrapper(device=device, **faiss_kwargs)
        self.llm = LLMHandler(
            device=device,
            model_name=model_name,
            llm_kwargs=llm_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            use_qlora=use_qlora,
        )
        self.use_rag = use_rag
        self.device = device
        self.llm_generation_config = self.get_default_llm_config()
        if llm_generation_config is not None:
            self.llm_generation_config.update(llm_generation_config)

        # to make wandb.watch track the model
        self.idea_zozza_ma_brillante_di_dario = self.llm.model

    def forward(self, batch: dict) -> dict:
        # for use with RagTrainer
        return self.forward_batch_query_single_doc(batch)

    def to(self, device: str):
        """
        Move the model to the given device.
        We override this to move also faiss and embedder to device.
        """
        self.llm.model.to(device)
        self.llm.device = device
        self.faiss.to(device)

    def set_use_rag(self, use_rag: bool):
        self.use_rag = use_rag

    def craft_autoregressive_query(self, query: str, doc: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. You are specialized in answering STEM questions. "
                "You will be provided with a context and a question to answer based on the context.",
            },
            {
                "role": "user",
                "content": f"Context:\n{doc}\nQuestion:\n{query}",
            },
        ]

        mess_prep: str = self.llm.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return mess_prep

    def forward_batch_query_single_doc(self, batch: Dict):
        """
        Does rag inference with a single document retrieved for each query.
        :param batch: a dictionary with keys "query" and "answer", both lists of strings
        :return: a dictionary with keys "logits" and "answer_lengths". "logits" is a tensor of shape
        (batch_size, max_len, vocab_size), and it is on the same device as the model.
        "answer_lengths" is a list of integers, the lengths of the answers.
        """
        queries: List[str] = batch["query"]
        answers: List[str] = batch["answer"]

        eos_token = self.llm.tokenizer.eos_token
        answers = [answer + eos_token for answer in answers]

        retrieved_docs = self.faiss.search_multiple_texts(queries, n_neighbors=1)
        headers: List[str] = []
        for query, doc in zip(queries, retrieved_docs):
            doc_content, doc_score = doc[0]
            # header = f"Context:\n{doc_content}\n\nQuery:\n{query}\n\nAnswer:\n"
            header = self.craft_autoregressive_query(query, doc_content)
            headers.append(header)

        tokenized_headers: BatchEncoding = self.llm.tokenizer(
            headers, padding=False, return_tensors="pt"
        ).to(self.llm.device)

        if "targets" in batch:
            tokenized_answers = batch["targets"]
        else:
            tokenized_answers = self.llm.tokenizer(
                answers, padding=False, return_tensors="pt"
            )

        answer_lengths = [
            len(tokenized_answer) for tokenized_answer in tokenized_answers["input_ids"]
        ]
        concatenated_input_ids = [
            # concatenate the tokenized header and the tokenized answer which are tensors
            torch.cat([tokenized_header, tokenized_answer], dim=0)
            # tokenized_header + tokenized_answer
            for tokenized_header, tokenized_answer in zip(
                tokenized_headers["input_ids"], tokenized_answers["input_ids"]
            )
        ]

        padded_input_ids = self.llm.tokenizer.pad(
            {"input_ids": concatenated_input_ids},
            padding=True,
            return_tensors="pt",
        )
        padded_input_ids: BatchEncoding = padded_input_ids.to(self.llm.device)

        # make dictionary with keys "input_ids" and "attention_mask"
        llm_inputs = {
            "input_ids": padded_input_ids["input_ids"],
            "attention_mask": padded_input_ids["attention_mask"],
        }

        logits = self.llm.get_logits(llm_inputs)  # (batch_size, max_len, vocab_size)
        return {
            "logits": logits,
            "answer_lengths": answer_lengths,
        }

    def _logits_to_weighted_probs(self, logits: torch.Tensor, scores: torch.Tensor):
        """
        Compute the probabilities of the tokens by aggregating the probabilities of the retrieved documents.
        :param logits:
        :param scores: Need to be normalized
        :return:
        """
        probas = torch.nn.functional.softmax(
            logits, dim=-1
        )  # (num_docs, seq_len, vocab_size)
        return (probas * scores[:, None, None]).sum(dim=0)  # (seq_len, vocab_size)

    def _next_token_strategy(self, probs, decoding_strategy):
        assert decoding_strategy in DECODING_STRATEGIES
        # probs has messed up indices (there is an extra dimension at the beginning),
        # this code takes this into account

        if decoding_strategy == "greedy":
            next_token = torch.argmax(probs)
        elif decoding_strategy == "top_k":
            top_k_probs, top_k_indices = torch.topk(probs, TOP_K)
            next_token = top_k_indices[:, torch.multinomial(top_k_probs, 1).item()]
        elif decoding_strategy == "top_p":
            sorted_probs, top_k_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > TOP_P
            # shift to the right to keep the first token that exceeds TOP_P
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            sorted_probs[sorted_indices_to_remove] = 0
            sorted_probs /= sorted_probs.sum()
            next_token = top_k_indices[:, torch.multinomial(sorted_probs, 1).item()]
        return next_token

    @torch.no_grad()
    def autoregressive_inference(
        self,
        query: str,
        n_docs: int = 10,
        decoding_strategy: str = "greedy",
        return_generator: bool = False,
    ) -> Tuple[Iterable[str], List[Tuple[str, float]]]:
        """
        This method performs autoregressive generation. It operates as follows:

        1. Accepts a query as input.
        2. It retrieves relevant passages based on the query.
        3. For each passage, calculates the probability of the next token based on the query and the passage.
        4. It then computes the average of the probabilities weighted by the scores of the retrieved documents.
        6. It then applies a decoding strategy to determine the next token based on these probabilities.
        7. This process continues until either the maximum sequence length is reached or an end-of-sequence token is encountered.

        This method reuses cached values of the past key values of the model to avoid recomputing them.

        :param query: the query for which to generate text
        :param n_docs: the number of documents to retrieve
        :param decoding_strategy: the decoding strategy to use. Can be "greedy", "top_k", or "top_p"
        :param return_generator: whether to return a generator or the generated text

        :return: a tuple with the token generator and the retrieved documents
        """

        if self.use_rag:
            retrieved_docs = self.faiss.search_text(query, n_neighbors=n_docs)
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
                [state["similarity"] for state in autoregressive_state],
                device=self.device,
            )

            similarities /= similarities.sum()
        else:
            retrieved_docs = []
            autoregressive_state = [
                {
                    "past_key_values": None,
                    "doc": None,
                    "similarity": 1.0,
                    "query": query,
                    "tokenized_query": self.llm.tokenizer(
                        query,
                        return_tensors="pt",
                        padding=False,
                    )["input_ids"],
                }
            ]
            similarities = torch.tensor([1.0], device=self.device)

        @torch.no_grad()
        def generator():
            answer = []
            while len(answer) < AUTOREGRESSIVE_GEN_MAX_LENGTH:
                all_logits = []

                for state in autoregressive_state:
                    tokenized_query = state["tokenized_query"].to(self.device)
                    result = self.llm.model(
                        tokenized_query,
                        past_key_values=state["past_key_values"],
                        use_cache=True,
                    )

                    logits: torch.Tensor = result.logits
                    logits = logits[:, -1, :]

                    state["past_key_values"] = result.past_key_values

                    all_logits.append(logits)

                # compute the average of the logits weighted by the scores of the retrieved documents
                torch_logits = torch.stack(all_logits)

                probs = self._logits_to_weighted_probs(torch_logits, similarities)

                next_token = self._next_token_strategy(probs, decoding_strategy)

                # get the token from the tokenizer
                next_token_str = self.llm.tokenizer.decode(next_token)

                for state in autoregressive_state:
                    state["query"] += next_token_str
                    state["tokenized_query"] = torch.tensor([[next_token]])  # type: ignore

                if len(answer) > 0 and next_token_str == self.llm.tokenizer.eos_token:
                    break

                if next_token_str in self.llm.tokenizer.all_special_tokens:
                    continue

                answer.append(next_token_str)
                yield next_token_str

        if return_generator:
            return generator(), retrieved_docs
        else:
            return "".join(generator()), retrieved_docs

    def naive_inference(
        self,
        histories: List[List[Dict]] | List[Dict],
        queries: List[str] | str,
        n_docs: int = 10,
        **kwargs,
    ) -> Tuple[
        List[str] | str, List[List[Tuple[str, float]]] | List[Tuple[str, float]]
    ]:
        """
        This method performs inference with the RAG model. It operates as follows:

        1. Accepts a query as input.
        2. Retrieves relevant passages based on the query.
        3. Creates a prompt with the retrieved passages and the query.
        4. Generates text based on the prompt.

        This method leverages the pipeline method of the Transformers library to perform inference.
        It is inserted here for comparison purposes with the autoregressive generation method.

        We are assuming that queries and histories are coherent in type.
        We support both batch inference and single queries, but we assume that if queries is a string then histories
        is a list of dictionaries containing the chat history.
        If queries is a list of strings then histories is a list of lists of dictionaries containing the chat history.

        :param histories: the chat history/histories (right now it is only used in the case RAG is not used)
        :param queries: the query(queries) for which to generate text
        :param n_docs: the number of documents to retrieve
        :param kwargs: additional arguments to pass to the LLM model

        :return: a tuple with the generated text and the retrieved documents
        """

        if isinstance(queries, list):
            updated_histories = []
            retrieved = []
            for history, query in zip(histories, queries):
                if self.use_rag is False:
                    updated_histories.append(join_messages_query_no_rag(history, query))
                else:
                    retrieved_now = self.faiss.search_text(query, n_neighbors=n_docs)
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
                retrieved = self.faiss.search_text(queries, n_neighbors=n_docs)
                # here we would do some preprocessing on the retrieved documents
                updated_histories = join_messages_query_rag(
                    histories, queries, retrieved
                )
        else:
            raise TypeError(
                "histories and queries must be either both strings or both lists of strings"
            )

        rag_config = deepcopy(self.llm_generation_config)
        if kwargs:
            rag_config.update(kwargs)
        response = self.llm.inference(updated_histories, rag_config)

        return response, retrieved

    @staticmethod
    def get_default_llm_config():
        return {
            "max_new_tokens": 500,
            "do_sample": False,
        }

    def add_arxiv_paper(self, paper):
        raise NotImplementedError

    # x = rag_handler.forward_batch_query_single_doc(
    #     {"query": ["ciao", "come", "stai"], "answer": ["bene", "molto", "adjpasd"]}
    # )

    # x = rag_handler.forward_single_query_multiple_docs("ciao", "bene")

    # def prepare_prompt(
    #     self, query: str, doc: str, answer: str
    # ) -> Tuple[Dict[str, torch.Tensor], int]:
    #     """
    #     Assemble a prompt with the context, query, and answer.
    #     Return a dictionary with keys "input_ids", "attention_mask", "query_plus_context_length".
    #     "input_ids" is a tensor of shape (1, seq_len) with the tokenized prompt.
    #     "attention_mask" is a tensor of shape (1, seq_len) with 1s in positions corresponding to tokens
    #     and 0s in positions corresponding to padding tokens.
    #     "query_plus_context_length" is the length of the query and context part of the prompt. To train,
    #     look only at the logits corresponding to the answer part of the prompt.
    #     """
    #     header = f"Context:\n{doc}\n\nQuery:\n{query}\n\nAnswer:\n"
    #     header_tokens = self.llm.tokenizer(header, return_tensors="pt", padding=False)
    #     query_plus_context_length = header_tokens["input_ids"].shape[1]
    #     answer_tokens = self.llm.tokenizer(answer, return_tensors="pt", padding=False)
    #     # if the tokenizer adds a <s> token at the beginning, we remove it
    #     if answer_tokens["input_ids"][0] == self.llm.tokenizer.encode("<s>"):
    #         answer_tokens["input_ids"] = answer_tokens["input_ids"][1:]
    #         answer_tokens["attention_mask"] = answer_tokens["attention_mask"][1:]
    #     input_ids = torch.cat(header_tokens["input_ids"], answer_tokens["input_ids"])
    #     attention_mask = torch.cat(
    #         header_tokens["attention_mask"], answer_tokens["attention_mask"]
    #     )
    #     return {
    #         "input_ids": input_ids,
    #         "attention_mask": attention_mask,
    #     }, query_plus_context_length

    # def replug_forward(
    #     self,
    #     query: str,
    #     answer: str,
    # ) -> Dict[str, torch.Tensor]:
    #     """
    #     logits: (num_docs, seq_len, vocab_size). The logits for each retrieved document, with
    #     the given query and answer.
    #     scores: (num_docs,). The scores of the retrieved documents, normalized.
    #     Used to average probabilities later.
    #     context_lengths: (num_docs,). The length of the query and context part of the prompt. These will
    #     be used to look only at the logits corresponding to the answer part of the prompt.
    #     """
    #     retrieved_docs = self.faiss.search_text(query)
    #     prompts = []
    #     scores = []
    #     context_lengths = []
    #     for doc, score in retrieved_docs:
    #         tokenized_prompt, query_plus_context_length = self.prepare_prompt(
    #             query, doc, answer
    #         )
    #         prompts.append(tokenized_prompt)
    #         scores.append(score)
    #         context_lengths.append(query_plus_context_length)
    #     # make a batch of prompts (get_logits expects a single dict with batched tensors as values)
    #     padding_id = self.llm.tokenizer.pad_token_id
    #     input_ids = [prompt["input_ids"] for prompt in prompts]
    #     padded_input_ids = pad_sequence(
    #         input_ids, batch_first=True, padding_value=padding_id
    #     )
    #     attention_mask = torch.stack([prompt["attention_mask"] for prompt in prompts])
    #     batch = {
    #         "input_ids": padded_input_ids,
    #         "attention_mask": attention_mask,
    #     }
    #     logits = self.llm.get_logits(batch)  # (num_docs, seq_len, vocab_size)
    #     scores = torch.tensor(scores)  # (num_docs,)
    #     scores /= scores.sum()
    #     context_lengths = torch.tensor(context_lengths)  # (num_docs,)
    #     return {
    #         "logits": logits,
    #         "scores": scores,
    #         "context_lengths": context_lengths,
    #     }

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

    # def forward_single_query_multiple_docs(
    #     self,
    #     query: str,
    #     answer: str,
    # ) -> Dict[str, torch.Tensor]:
    #     """
    #     Retrieve documents with faiss, craft a prompt for each document, and get the logits for each prompt.
    #     Return the logits and the scores of the retrieved documents, normalized. Also return
    #     the length of the answer part of the prompt.
    #     Output tensors are on the same device as the model.
    #     """
    #     retrieved_docs = self.faiss.search_text(query)
    #     headers = []  # strings
    #     scores = []  # floats
    #
    #     for doc, score in retrieved_docs:
    #         # WARNING THIS IS WRONG
    #         header = f"Context:\n{doc}\n\nQuery:\n{query}\n\nAnswer:\n"
    #         headers.append(header)
    #         scores.append(score)
    #
    #     tokenized_headers = self.llm.tokenizer(
    #         headers, return_tensors="pt", padding=True
    #     )  # (num_docs, seq_len)
    #     tokenized_answer = self.llm.tokenizer(
    #         answer, return_tensors="pt"
    #     )  # (1, seq_len)
    #     input_ids = torch.cat(
    #         [tokenized_headers["input_ids"], tokenized_answer["input_ids"]], dim=1
    #     )  # (num_docs, seq_len)
    #     attention_mask = torch.cat(
    #         [tokenized_headers["attention_mask"], tokenized_answer["attention_mask"]],
    #         dim=1,
    #     )  # (num_docs, seq_len)
    #     batch = {
    #         "input_ids": input_ids,
    #         "attention_mask": attention_mask,
    #     }
    #
    #     batch["input_ids"] = batch["input_ids"].to(self.llm.device)
    #     batch["attention_mask"] = batch["attention_mask"].to(self.llm.device)
    #     logits = self.llm.get_logits(batch)  # (num_docs, seq_len, vocab_size)
    #     scores = torch.tensor(scores)  # (num_docs,)
    #     scores /= scores.sum()
    #     answer_length = tokenized_answer["input_ids"].shape[1]
    #
    #     return {
    #         "logits": logits,
    #         "scores": scores,
    #         "answer_length": answer_length,
    #     }
