from typing import Optional, List, Dict, Tuple

from torch import nn
import torch
from transformers import BatchEncoding, pipeline, QuestionAnsweringPipeline

from backend.model.llm_handler import LLMHandler
from backend.vector_database.faiss_wrapper import FaissWrapper

REPLUG_GEN_MAX_LENGTH = 1000
DECODING_STRATEGIES = ["greedy", "top_k", "top_p"]
INFERENCE_TYPES = ["naive", "replug", "pipe"]
TOP_K = 50
TOP_P = 0.9

DEFAULT_LLM_GENERATION_CONFIG = {
    "max_new_tokens": 500,
    "do_sample": False,
}


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
        pretrained_model_path: Optional[str] = None,
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
            pretrained_model_path=pretrained_model_path,
        )
        self.use_rag = use_rag
        self.device = device
        self.llm_generation_config = DEFAULT_LLM_GENERATION_CONFIG

        if llm_generation_config is not None:
            self.llm_generation_config.update(llm_generation_config)
        self.pipe = pipeline(
            "text-generation",
            model=self.llm.model,
            tokenizer=self.llm.tokenizer,
        )

        self.llm.model.generation_config.max_length = 2500

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

    def forward(self, batch: dict) -> dict:
        # for use with RagTrainer
        return self.forward_batch_query_single_doc(batch)

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

        # eos_token = self.llm.tokenizer.eos_token
        # answers = [answer + eos_token for answer in answers]

        retrieved_docs = self.faiss.search_multiple_texts(queries, n_neighbors=1)
        headers: List[str] = []
        for query, doc in zip(queries, retrieved_docs):
            doc_content, _ = doc[0]
            # header = f"Context:\n{doc_content}\n\nQuery:\n{query}\n\nAnswer:\n"
            header = self._craft_single_doc_query(query, doc_content)
            headers.append(header)

        tokenized_headers: BatchEncoding = self.llm.tokenizer(
            headers, padding=False, return_tensors="pt"
        ).to(self.llm.device)

        if "targets" in batch:
            tokenized_answers = batch["targets"]
        else:
            tokenized_answers = self.llm.tokenizer(
                answers, padding=False, return_tensors="pt"
            ).to(self.llm.device)

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

    @torch.no_grad()
    def inference(
        self,
        query: str,
        n_docs_retrieved: int = 10,
        decoding_strategy: str = "greedy",
        inference_type: str = "replug",
        return_generator: bool = False,
        return_prompt: bool = False,
        history: Optional[List[Dict]] = None,
    ) -> (
        Tuple[str, List[Tuple[str, float]]]
        | Tuple[str, List[Tuple[str, float]], List[Dict]]
    ):
        """
        This method performs inference with the LLM model and optionally the RAG model.
        See the documentation of the specific inference methods for more details (no_rag_inference,
        naive_inference, replug_inference)

        :param query: the query for which to generate text
        :param n_docs_retrieved: the number of documents to retrieve
        :param decoding_strategy: the decoding strategy to use. Can be "greedy", "top_k", or "top_p"
        :param inference_type: the type of inference to perform. Can be "naive" or "replug"
        :param return_generator: whether to return a generator or the generated text
        :param return_prompt: if True, return the prompt used for generation as a third element of the
        returned tuple

        :return: a tuple with the generated text, the retrieved documents, and the prompt used for generation
                (if return_prompt is True)
        """
        assert decoding_strategy in DECODING_STRATEGIES, "Invalid decoding strategy"
        assert inference_type in INFERENCE_TYPES, "Invalid inference type"

        if not self.use_rag:
            if inference_type != "pipe":
                return self.no_rag_inference(
                    query,
                    decoding_strategy=decoding_strategy,
                    return_generator=return_generator,
                    return_prompt=return_prompt,
                    history=history,
                )
            else:
                self.llm.model.generation_config.max_length = 2500

                breakpoint()
                pipe_query = [
                    {
                        "role": "user",
                        "content": query,
                    }
                ]
                outputs = self.pipe(pipe_query, return_full_text=False)
                # if type([query][0]) == list:
                #     return [output[0]["generated_text"] for output in outputs]
                # else:
                return (outputs[0]["generated_text"],)
        else:
            if inference_type == "naive":
                return self.naive_inference(
                    query,
                    n_docs_retrieved=n_docs_retrieved,
                    decoding_strategy=decoding_strategy,
                    return_generator=return_generator,
                    return_prompt=return_prompt,
                )
            elif inference_type == "replug":
                return self.replug_inference(
                    query,
                    n_docs_retrieved=n_docs_retrieved,
                    decoding_strategy=decoding_strategy,
                    return_generator=return_generator,
                    return_prompt=return_prompt,
                )
            elif inference_type == "pipe":
                retrieved_docs = self.faiss.search_text(
                    query, n_neighbors=n_docs_retrieved
                )

                messages = self._craft_multiple_docs_query(query, retrieved_docs, do_prep=False)

                outputs = self.pipe(messages, return_full_text=False)
                return (outputs[0]["generated_text"],)

        raise ValueError("Bad parameters for inference method.")


    @torch.no_grad()
    def no_rag_inference(
        self,
        query: str,
        decoding_strategy: str = "greedy",
        return_generator: bool = False,
        return_prompt: bool = False,
        history: Optional[List[Dict]] = None,
    ) -> (
        Tuple[str, List[Tuple[str, float]]]
        | Tuple[str, List[Tuple[str, float]], List[Dict]]
    ):
        """
        This method performs inference with the LLM model only.
        This method reuses cached values of the past key values of the model to avoid recomputing them.

        :param query: the query for which to generate text
        :param decoding_strategy: the decoding strategy to use. Can be "greedy", "top_k", or "top_p"
        :param return_generator: whether to return a generator or the generated text
        :param return_prompt: if True, return the prompt used for generation as a third element of the

        :return: a tuple with the generated text, an empty list (to match the return type of the other methods),
        and the prompt used for generation
        """
        assert decoding_strategy in DECODING_STRATEGIES

        prompt = self._craft_no_docs_query(query, history)

        autoregressive_state = [
            {
                "past_key_values": None,
                "doc": None,
                "similarity": 1.0,
                "query": prompt,
                "tokenized_query": self.llm.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=False,
                )["input_ids"],
            }
        ]
        similarities = torch.tensor([1.0], device=self.device)

        generator = self._inference_generator(
            autoregressive_state,
            similarities,
            decoding_strategy=decoding_strategy,
        )

        output = generator if return_generator else "".join(generator)
        if return_prompt:
            return output, [], autoregressive_state[0]["query"]
        return output, []

    @torch.no_grad()
    def naive_inference(
        self,
        query: str,
        n_docs_retrieved: int = 10,
        decoding_strategy: str = "greedy",
        return_generator: bool = False,
        return_prompt: bool = False,
    ) -> (
        Tuple[str, List[Tuple[str, float]]]
        | Tuple[str, List[Tuple[str, float]], List[Dict]]
    ):
        """
        This method performs inference with the RAG model. It operates as follows:

        1. Accepts a query as input.
        2. Retrieves relevant passages based on the query.
        3. Creates a prompt with the retrieved passages and the query.
        4. Generates text based on the prompt.

        This method reuses cached values of the past key values of the model to avoid recomputing them.

        :param query: the query for which to generate text
        :param n_docs_retrieved: the number of documents to retrieve
        :param decoding_strategy: the decoding strategy to use. Can be "greedy", "top_k", or "top_p"
        :param return_generator: whether to return a generator or the generated text
        :param return_prompt: if True, return the prompt used for generation as a third element of the

        :return: a tuple with the generated text, the retrieved documents, and the prompt used for generation
        """

        assert decoding_strategy in DECODING_STRATEGIES

        retrieved_docs = self.faiss.search_text(query, n_neighbors=n_docs_retrieved)
        prompt = self._craft_multiple_docs_query(query, retrieved_docs)
        autoregressive_state = [
            {
                "past_key_values": None,
                "doc": None,
                "similarity": 1.0,
                "query": prompt,
                "tokenized_query": self.llm.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=False,
                )["input_ids"],
            }
        ]
        similarities = torch.tensor([1.0], device=self.device)

        generator = self._inference_generator(
            autoregressive_state,
            similarities,
            decoding_strategy=decoding_strategy,
        )

        output = generator if return_generator else "".join(generator)
        if return_prompt:
            return output, retrieved_docs, autoregressive_state[0]["query"]
        return output, retrieved_docs

    @torch.no_grad()
    def replug_inference(
        self,
        query: str,
        n_docs_retrieved: int = 10,
        decoding_strategy: str = "greedy",
        return_generator: bool = False,
        return_prompt: bool = False,
    ) -> (
        Tuple[str, List[Tuple[str, float]]]
        | Tuple[str, List[Tuple[str, float]], List[Dict]]
    ):
        """
        This method performs autoregressive generation using the REPLUG method. It operates as follows:

        1. Accepts a query as input.
        2. It retrieves relevant passages based on the query.
        3. For each passage separately, calculates the probability of the next token based on the query and the passage.
        4. It then computes the average of the probabilities weighted by the scores of the retrieved documents.
        6. It then applies a decoding strategy to determine the next token based on these probabilities.
        7. This process continues until either the maximum sequence length is reached or an end-of-sequence token is encountered.

        This method reuses cached values of the past key values of the model to avoid recomputing them.

        :param query: the query for which to generate text
        :param n_docs_retrieved: the number of documents to retrieve
        :param decoding_strategy: the decoding strategy to use. Can be "greedy", "top_k", or "top_p"
        :param return_generator: whether to return a generator or the generated text
        :param return_prompt: if True, return the prompt used for generation as a third element of the
        returned tuple

        :return: a tuple with the generated text, the retrieved documents, and the prompt used for generation
        """

        assert decoding_strategy in DECODING_STRATEGIES

        retrieved_docs = self.faiss.search_text(query, n_neighbors=n_docs_retrieved)
        autoregressive_state = [
            {
                "past_key_values": None,
                "doc": doc,
                "similarity": similarity,
                "query": self._craft_single_doc_query(query, doc),
                "tokenized_query": self.llm.tokenizer(
                    self._craft_single_doc_query(query, doc),
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

        generator = self._inference_generator(
            autoregressive_state,
            similarities,
            decoding_strategy=decoding_strategy,
        )

        output = generator if return_generator else "".join(generator)
        if return_prompt:
            return output, retrieved_docs, autoregressive_state
        return output, retrieved_docs

    def _craft_no_docs_query(
        self, query: str, history: Optional[List[Dict]] = None
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. You are specialized in answering STEM questions. "
                "You will be provided with a question to answer.",
            },
        ]
        if history is not None:
            messages.extend(history)

        messages.append(
            {
                "role": "user",
                "content": f"Question:\n{query}",
            },
        )

        mess_prep: str = self.llm.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return mess_prep

    def _craft_single_doc_query(self, query: str, doc: Dict) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. You are specialized in answering STEM questions. "
                "You will be provided with a context and a question to answer based on the context.",
            },
            {
                "role": "user",
                "content": f"Context:\n{doc['text']}\nQuestion:\n{query}",
            },
        ]

        mess_prep: str = self.llm.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return mess_prep

    def _craft_multiple_docs_query(
        self, query: str, docs: List[Tuple], do_prep=True
    ) -> str:
        user_prompt = ""
        user_prompt += "== Context ==\n"
        for i, (doc, _) in enumerate(docs):
            user_prompt += f"=== Document {i} ===\n"
            user_prompt += doc["text"] + "\n"
        user_prompt += f"== User Query ==\n{query}"

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. You are specialized in answering STEM questions."
                "You will be provided with a context consisting of multiple documents and a question to answer based on the context.",
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        if do_prep:
            mess_prep: str = self.llm.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            return mess_prep
        else:
            return messages

    @torch.no_grad()
    def _inference_generator(
        self,
        autoregressive_state: List[Dict],
        similarities: torch.Tensor,
        decoding_strategy: str = "greedy",
    ):
        answer = []
        while len(answer) < REPLUG_GEN_MAX_LENGTH:
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
