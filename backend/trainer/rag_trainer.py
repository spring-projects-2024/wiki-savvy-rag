from jepa.trainer.trainer import Trainer
from backend.model.rag_handler import RagHandler
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model
from peft.utils import prepare_model_for_kbit_training


# class RagCriterionOld(nn.Module):
#     """
#     For use with forward_single_query_multiple_docs and compute_probabilities_for_training.
#     """

#     def __init__(self):
#         super().__init__()
#         self.cross_entropy_from_log_proba = nn.NLLLoss(reduction="mean")

#     def forward(self, output: dict, batch: dict) -> dict:
#         """
#         :param output: dict with keys "probas", "answer_mask"
#         :param batch: dict with keys "answer_tokens"
#         """
#         probas = output["probas"]
#         log_probas = torch.log(probas)
#         target = batch["answer_tokens"]
#         loss = self.cross_entropy_from_log_proba(log_probas, target)  # TODO: check
#         return {"loss": loss}


class RagCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, output: dict, batch: dict) -> dict:
        """
        :param output: dict with keys "logits", "answer_lengths"
        :param batch: dict with keys "targets"
        """
        logits = output["logits"]  # (batch_size, max_len, vocab_size)
        targets = batch["targets"]  # list of tensors of different lengths
        answer_lengths = output["answer_lengths"]  # (batch_size,)
        loss = 0
        for logits_one_query, answer_length, answer_tokens in zip(
            logits, answer_lengths, targets
        ):
            assert len(answer_tokens) == answer_length
            loss += self.cross_entropy(
                logits_one_query[-answer_length:, :], answer_tokens
            )
        loss /= len(answer_lengths)
        return {"loss": loss}


class RagTrainer(Trainer):
    def __init__(self, model: RagHandler, **kwargs):
        super().__init__(model, **kwargs)
        torch.compile(model.llm.model)  # artigianale. commentalo per spegnerlo

    def train_step(self, batch: dict) -> dict:
        answers = batch["answer"]
        tokenized_answers = self.model.llm.tokenizer(
            answers, padding=False
        )  # BatchEncoding object
        batch["targets"] = tokenized_answers["input_ids"]
        return super().train_step(batch)


def prepare_for_qlora(model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=8,
        target_modules=[
            "q_proj",
            "o_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    return model
