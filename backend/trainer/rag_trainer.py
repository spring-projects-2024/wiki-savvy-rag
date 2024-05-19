from jepa.trainer.trainer import Trainer
from backend.model.rag_handler import RagHandler
import torch
from torch import nn


# TODO:
# 3. have a dataloader load data with the correct format
# 4. ensure RagHandler can be used as model in Trainer


class RagCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_from_log_proba = nn.NLLLoss(reduction="mean")

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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(
            self.model, RagHandler
        ), "RagTrainer expects a RagHandler model."
        assert isinstance(
            self.criterion, RagCriterion
        ), "RagTrainer expects a RagCriterion criterion."

    # def train_step(self, batch: dict) -> dict:
    #     for key in batch:
    #         if isinstance(batch[key], torch.Tensor):
    #             batch[key] = batch[key].to(self.device)
    #     output = self.model.replug_forward(batch)
