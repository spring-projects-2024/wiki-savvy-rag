import torch

from transformers import AutoModel, AutoTokenizer

model_path = "Alibaba-NLP/gte-base-en-v1.5"
revision = "269b9ac"


class EmbedderWrapper:
    def __init__(self, device):
        # todo: device for embedding

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.embedder = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, revision=revision, device_map=device
        )

    @torch.no_grad()
    def get_embedding(self, text) -> torch.Tensor:
        batch_dict = self.tokenizer(
            text, max_length=8192, padding=True, truncation=True, return_tensors="pt"
        )
        outputs = self.embedder(**batch_dict)
        embeddings = outputs.last_hidden_state[:, 0]

        # np.random.seed(hash(text) % 1000)
        # w = np.random.standard_normal(self.d)
        # w /= np.linalg.norm(w)
        # return w

        return embeddings

    def get_dimensionality(self):
        return self.embedder.config.hidden_size


if __name__ == "__main__":
    embedder = EmbedderWrapper("cpu")
    text = "The quick brown fox jumps over the lazy dog."
    embedding = embedder.get_embedding(text)
    print(embedding)
    print(embedding.shape)
