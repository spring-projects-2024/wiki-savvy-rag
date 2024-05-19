from typing import List, Iterable

import torch

from transformers import AutoModel, AutoTokenizer

model_path = "BAAI/bge-small-en-v1.5"


class EmbedderWrapper:
    def __init__(self, device):

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.embedder = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, device_map=device
        )
        self.device = device

    def to(self, device: str):
        self.embedder.to(device)
        self.device = device

    @torch.no_grad()  # todo: check if iterable is ok
    def get_embedding(self, text: Iterable[str] | str) -> torch.Tensor:
        batch_dict = self.tokenizer(
            text, max_length=512, padding=True, truncation=True, return_tensors="pt"
        )
        batch_dict = batch_dict.to(self.device)
        outputs = self.embedder(**batch_dict)
        embeddings = outputs.last_hidden_state[:, 0].clone()
        # np.random.seed(hash(text) % 1000)
        # w = np.random.standard_normal(self.d)
        # w /= np.linalg.norm(w)
        # return w
        return embeddings.cpu()

    def get_dimensionality(self):
        return self.embedder.config.hidden_size


if __name__ == "__main__":
    embedder = EmbedderWrapper("cpu")
    text = "The quick brown fox jumps over the lazy dog."
    embedding = embedder.get_embedding(text)
    print(embedding)
    print(embedding.shape)
