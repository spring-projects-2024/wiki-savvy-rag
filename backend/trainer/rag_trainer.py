import os
from typing import Optional
from torch.optim import AdamW
from torch.utils.data import DataLoader

from backend.benchmark.utils import load_yahoo_answers, load_mmlu
from backend.vector_database.dataset import MockDataset
from jepa.trainer.trainer import Trainer
from backend.model.rag_handler import RagHandler
import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
    BatchEncoding,
)
from peft import LoraConfig, TaskType, get_peft_model
from peft.utils import prepare_model_for_kbit_training


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
            logits, answer_lengths, targets["input_ids"]
        ):
            assert (
                len(answer_tokens) == answer_length
            ), f"{len(answer_tokens)}  {answer_length}"
            loss += self.cross_entropy(
                logits_one_query[-answer_length - 1 : -1, :], answer_tokens
            )
        loss /= len(answer_lengths)
        return {"loss": loss}


class RagTrainer(Trainer):
    def __init__(
        self, model: RagHandler, checkpoint_interval_steps: Optional[int], **kwargs
    ):
        self.checkpoint_interval_steps = checkpoint_interval_steps
        super().__init__(model, **kwargs)
        # torch.compile(model.llm.model)  # artigianale. commentalo per spegnerlo

    def train_step(self, batch: dict) -> dict:
        if self.step != 0 and self.step % self.checkpoint_interval_steps == 0:
            self.make_checkpoint()
        answers = batch["answer"]
        tokenized_answers: BatchEncoding = self.model.llm.tokenizer(
            answers, padding=False, return_tensors="pt"
        )
        targets = {
            "input_ids": tokenized_answers["input_ids"],
            "attention_mask": tokenized_answers["attention_mask"],
        }
        batch["targets"] = targets
        return super().train_step(batch)

    def make_checkpoint(self):
        """
        We override this method because we are using qlora and we want to
        save the weights using huggingface methods rather than torch.save.
        """
        chkpt_dir = os.path.join(
            self.checkpoint_root_dir,
            "step" + str(self.step),
        )
        self.model.llm.save_weights(chkpt_dir)
        if self.log_to_wandb:
            artifact_name = f"chkpt-{self.step}"
            self.logger.log_checkpoint(chkpt_dir, artifact_name)


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
        lora_alpha=16,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, lora_config)
    return model


if __name__ == "__main__":
    print(torch.cuda.is_available())

    md = MockDataset(["ciao"])
    faiss_kwargs = {"embedder": None, "dataset": md, "index_str": "Flat"}
    rag_handler = RagHandler(
        model_name="Minami-su/Qwen1.5-0.5B-Chat_llamafy",
        device="cpu",
        use_qlora=False,
        llm_generation_config=None,
        llm_kwargs=None,
        # tokenizer_kwargs=tokenizer_kwargs,
        faiss_kwargs=faiss_kwargs,
    )

    rag_handler.faiss.train_from_text("ciao")
    rag_handler.faiss.add_text("ciao")

    batch_size = 1

    train_data = load_yahoo_answers(subset="stem")
    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_data = load_mmlu(split="validation", subset="stem")
    test_loader = DataLoader(test_data, batch_size=batch_size)
    train_metadata = {
        "id": "yahoo_answers",
        "use_as": "train",
        "num_samples": len(train_data),
    }

    optimizer = AdamW(rag_handler.parameters(recurse=True), lr=1e-5)

    # get parameters to optimize

    criterion = RagCriterion()
    num_training_steps = 20_000  # todo: change
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    train_config = {
        "model": rag_handler,
        "optimizer": optimizer,
        "criterion": criterion,
        "train_loader": train_loader,
        "train_metadata": train_metadata,
        "test_loader": test_loader,
        "max_epochs": 1,
        "device": "cpu",
        "scheduler": scheduler,
        "log_to_wandb": False,
        "log_interval": 1,
        "checkpoint_interval": 1,
        "checkpoint_root_dir": "../checkpoints",
        "seed": 42,
        "wandb_project": "ajdoasjda",
        "compile_model": False,
        "checkpoint_interval_steps": 1000,
    }

    print("Training...")

    rag_trainer = RagTrainer(**train_config)

    # print("Saving...")
    # rag_trainer.model.llm.save_weights("modello_salvato")
    # rag_trainer.train()
    #
    # print("Loading")

    rag_trainer.train_step(next(iter(train_loader)))

    for param in rag_trainer.model.llm.model.parameters():
        print(param.requires_grad)
        if param.requires_grad:
            print(param.grad)
