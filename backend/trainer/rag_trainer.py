from jepa.trainer.trainer import Trainer
import torch
from torch import nn


# TODO:
# 1. implement criterion for RAG as a nn.Module
# 2. implement train_step in RagTrainer
# 3. have a dataloader load data with the correct format
# 4. ensure RagHandler can be used as model in Trainer


class RagCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_from_log_proba = nn.NLLLoss()

    def forward(self, output: dict, batch: dict) -> dict:
        """
        :param output: dict with keys "probas", "answer_mask"
        :param batch: dict with keys "answer_tokens"
        """
        probas = output["probas"]
        log_probas = torch.log(probas)
        target = batch["answer_tokens"]
        loss = self.cross_entropy_from_log_proba(log_probas, target)  # TODO: check
        return {"loss": loss}


class RagTrainer(Trainer):
    pass
