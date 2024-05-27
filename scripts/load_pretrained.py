import torch
import yaml
import argparse
from backend.model.rag_handler import RagHandler

config_path = "configs/training/final.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

device = "cpu"
model_name = config["model_name"]
run_id = config["run_id"]
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
validation_interval = config["validation_interval"]
validation_samples = config["validation_samples"]
watch_model = config["watch_model"]

llm_generation_config = config.get("llm_generation_config", {})
llm_kwargs = config.get("llm_kwargs", None)
tokenizer_kwargs = config.get("tokenizer_kwargs", None)
faiss_kwargs = config.get("faiss_kwargs", None)
pretrained_model_path = "checkpoints/step100"

rag_handler = RagHandler(
    model_name=model_name,
    device=device,
    use_qlora=use_qlora,
    llm_generation_config=llm_generation_config,
    llm_kwargs=llm_kwargs,
    tokenizer_kwargs=tokenizer_kwargs,
    faiss_kwargs=faiss_kwargs,
    pretrained_model_path=pretrained_model_path,
)


query = {
    "input_ids": torch.tensor([[329, 3829, 190, 643]]),
    "attention_mask": torch.tensor([[1, 1, 1, 1]]),
}

# print logits
logits = rag_handler.llm.get_logits(query)

print(logits)

print("Loaded!")
