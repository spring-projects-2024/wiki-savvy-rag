import numpy as np


class EmbedderWrapper:
    def __init__(self, d=3):
        self.embedder = None
        self.d = d

    def get_embedding(self, text):
        np.random.seed(hash(text) % 1000)
        w = np.random.standard_normal(self.d)
        w /= np.linalg.norm(w)
        return w
