from jepa.logger import WandbLogger
from backend.model.rag_handler import RagHandler
from backend.trainer.rag_trainer import RagCriterion, RagTrainer, prepare_for_qlora
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml
from torch.optim import AdamW
from backend.benchmark.utils import load_yahoo_answers, load_mmlu
from transformers import get_linear_schedule_with_warmup


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, default="configs/training/prova.yaml"
    )
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    device = config["device"]
    model_name = config["model_name"]
    use_qlora = config["use_qlora"]
    optimizer_params = config["optimizer_params"]
    max_epochs = config["max_epochs"]
    batch_size = config["batch_size"]
    gradient_accumulation_steps = config["gradient_accumulation_steps"]
    log_to_wandb = config["log_to_wandb"]
    log_interval = config["log_interval"]
    checkpoint_interval_steps = config["checkpoint_interval_steps"]
    seed = config["seed"]
    wandb_project = config["wandb_project"]

    llm_generation_config = config.get("llm_generation_config", {})
    llm_kwargs = config.get("llm_kwargs", None)
    tokenizer_kwargs = config.get("tokenizer_kwargs", None)
    faiss_kwargs = config.get("faiss_kwargs", None)

    print("Preparing RAGHandler...")

    rag_handler = RagHandler(
        model_name=model_name,
        device=device,
        use_qlora=use_qlora,
        llm_generation_config=llm_generation_config,
        llm_kwargs=llm_kwargs,
        tokenizer_kwargs=tokenizer_kwargs,
        faiss_kwargs=faiss_kwargs,
    )

    # todo: check if this is necessary when reading from disk an already quantized model
    rag_handler.llm.model = prepare_for_qlora(rag_handler.llm.model)

    print("Preparing data...")

    # TODO: currently only batch_size=1 is supported. To have larger batch sizes,
    # the collate_fn in the DataLoader should be modified to pad the sequences.
    train_data = load_yahoo_answers(subset="stem")
    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_data = load_mmlu(split="validation", subset="stem")
    test_loader = DataLoader(test_data, batch_size=batch_size)
    train_metadata = {
        "id": "yahoo_answers",
        "use_as": "train",
        "num_samples": len(train_data),
    }

    WandbLogger.log_dataset(
        train_data, train_metadata, project="rag", entity="mattia-scardecchia"
    )

    print("Preparing training...")

    optimizer = AdamW(rag_handler.llm.model.parameters(), **optimizer_params)
    criterion = RagCriterion()
    num_training_steps = len(train_loader) * max_epochs
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
        "max_epochs": max_epochs,
        "device": device,
        "scheduler": scheduler,
        "log_to_wandb": log_to_wandb,
        "log_interval": log_interval,
        "checkpoint_root_dir": "../checkpoints",
        "seed": seed,
        "wandb_project": wandb_project,
        "compile_model": False,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "checkpoint_interval_steps": checkpoint_interval_steps,
    }

    print("Training...")

    rag_trainer = RagTrainer(**train_config)
    rag_trainer.train()


if __name__ == "__main__":
    main()
