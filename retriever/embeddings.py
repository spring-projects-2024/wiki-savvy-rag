import numpy as np


class EmbedderWrapper:
    def __init__(self):
        self.embedder = None

    def get_embedding(self, text):
        np.random.seed(hash(text) % 1000)
        w = np.random.standard_normal(3)
        w /= np.linalg.norm(w)
        return w
