from typing import Dict, Tuple, List, Iterable, Optional

# keep in this specific order, otherwise it gives Segmentation Fault on Federico's pc
from backend.vector_database.embedder_wrapper import EmbedderWrapper
import faiss
import numpy as np
from backend.vector_database.dataset import DatasetSQL, MockDataset

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class FaissWrapper:
    """Wrapper class for the Faiss library. Handles the indexing and searching of vectors.

    Attributes:
    - device: The device on which the model is loaded.
    - dataset: The dataset to be used.
    - embedder: The EmbedderWrapper object to be used.
    - dim: The dimensionality of the embeddings.
    - _index: The Faiss index object.
    """

    def __init__(
        self,
        device,
        dataset: DatasetSQL | MockDataset | str,
        embedder: Optional[EmbedderWrapper] = None,
        *,
        index_path=None,
        index_str=None,
        nprobe=None,
    ):
        """
        Instantiate a FaissWrapper object.
        :param device: The device on which the model is loaded.
        :param dataset: The dataset to be used.
        :param embedder: The EmbedderWrapper object to be used.
        :param index_path: The path to the index file. If provided, index_str is ignored.
        :param index_str: The index string. if index_path is not provided, this is used to create the index.
        :param nprobe: The number of clusters to look at during search. Default is 10.
        """
        if embedder is None:
            embedder = EmbedderWrapper(device)
        self.device = device
        self.embedder = embedder

        if isinstance(dataset, str):
            self.dataset = DatasetSQL(db_path=dataset)
        elif isinstance(dataset, (DatasetSQL, MockDataset)):
            self.dataset = dataset

        self.dim = self.embedder.get_dimensionality()
        if index_path:
            self._index = faiss.read_index(index_path)
            assert self.dim == self._index.d
            if nprobe:
                self._index.nprobe = nprobe
        elif index_str:
            self._index = faiss.index_factory(
                self.dim, index_str, faiss.METRIC_INNER_PRODUCT
            )
            self._index.nprobe = nprobe if nprobe else 10

    def to(self, device: str = "cpu"):
        self.embedder.to(device)
        self.device = device

    def search_vectors(
        self, vectors: np.ndarray, n_neighbors: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for the nearest neighbors of n vectors.
        :param vectors: np.ndarray of shape (n, dim)
        :param n_neighbors: The number of neighbors to search for.
        :return: Index matrix I and distance matrix D
        """
        return self._index.search(vectors, n_neighbors)

    def _search_text_papers_chks(self, query: str, chunks):
        em_query = self.embedder.get_embedding(query).numpy()
        em_query = em_query.flatten()
        result = []
        for chunk in chunks:
            em_chunk = self.embedder.get_embedding(chunk["text"]).numpy()
            em_chunk = em_chunk.flatten()
            distance = np.dot(em_chunk, em_query)
            result.append((chunk, distance))
        return result

    def _search_vector(
        self, vector: np.ndarray, n_neighbors: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for the nearest neighbors of a vector.
        :param vector: np.ndarray of shape (dim)
        :param n_neighbors: The number of neighbors to search for.
        :return: Index matrix I and distance matrix D
        """
        return self.search_vectors(vector.reshape(1, -1), n_neighbors)

    def _search_text_get_I_D(
        self, text: str, n_neighbors: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the I and D matrices from a text. Wraps around _search_vector
        :param text: The text to search for.
        :param n_neighbors: The number of neighbors to search for.
        :return: Index matrix I and distance matrix D
        """
        vector = self.embedder.get_embedding(text).numpy()
        return self._search_vector(vector, n_neighbors)

    def _index_to_text(self, index: int) -> Dict:
        """
        :param index: The index of the text.
        :return: The text corresponding to the index.
        """
        return self.dataset.search_chunk(index)

    def search_text(self, text: str, n_neighbors=10) -> List[Tuple[Dict, float]]:
        """
        Search for the nearest neighbors of a text.
        :param text: The text to search for.
        :return: List of tuples of (text, distance)
        """

        I, D = self._search_text_get_I_D(text, n_neighbors)
        return [(self._index_to_text(i), j) for i, j in zip(D[0], I[0]) if i != -1]

    def search_text_with_docs(self, text: str, n_neighbors=10, other_docs=[]):
        wikis = self.search_text(text, n_neighbors)
        papers = self._search_text_papers_chks(text, other_docs)
        sorted_data = sorted(wikis + papers, key=lambda x: x[1], reverse=True)
        return [item for item in sorted_data[:n_neighbors]]

    def search_multiple_texts(
        self, texts: List[str], n_neighbors: int
    ) -> List[List[Tuple[Dict, float]]]:
        """
        Search for the nearest neighbors of multiple texts.
        :param texts: List of texts to search for.
        :param n_neighbors: The number of neighbors to search for.
        :return: List of lists of tuples of (text, similarity)
        """
        MAX_BATCH_SIZE = 250

        embeddings = []
        for i in range(0, len(texts), MAX_BATCH_SIZE):
            embeddings.append(
                self.embedder.get_embedding(texts[i : i + MAX_BATCH_SIZE]).numpy()
            )

        I, D = self.search_vectors(np.concatenate(embeddings), n_neighbors)

        return [
            [(self._index_to_text(i), j) for i, j in zip(D_i, I_i) if i != -1]
            for D_i, I_i in zip(D, I)
        ]

    def train_from_vectors(self, data):
        """
        Train the index on the input data.
        :param data: The data to train on.
        """

        self._index.train(data)

    def add_vectors(self, data):
        """Add vectors to the index."""
        self._index.add(data)

    def train_from_text(self, data: Iterable[str]):
        """
        Wrapper around train_from_vectors that takes text as input.
        :param data: The data to train on.
        """

        self.train_from_vectors(self.embedder.get_embedding(data))

    def add_text(self, data: Iterable[str]):
        """
        Wrapper around add_vectors that takes text as input.
        :param data: The data to add.
        """

        self.add_vectors(self.embedder.get_embedding(data))

    def save_to_disk(self, path):
        faiss.write_index(self._index, path)


if __name__ == "__main__":
    faiss_kwargs = {
        "index_path": "scripts/vector_database/data/PQ128.index",
        "embedder": None,
        "dataset": "scripts/dataset/data/dataset.db",
    }
    faiss = FaissWrapper("cpu", **faiss_kwargs)
    query = "What is the mechanism thanks to which aeroplanes can fly?"
    papers = [
        {"title": "A", "text": "aeroplanes job offers"},
        {"title": "B", "text": "thanks to which aeroplanes can fly?"},
    ]
    res = faiss.search_text_with_docs(query, 10, papers)
    print(res)


# INDEX_PATH = "backend/vector_database/data/PQ128.index"

# if __name__ == "__main__":
#     chunks = ["ciao", "sono", "mattia"]
#     dataset = MockDataset(chunks)
#     embedder = EmbedderWrapper("cpu")

#     fw = FaissWrapper(
#         index_str="Flat", dataset=dataset, device="cpu", embedder=embedder
#     )

#     fw.train_from_text(chunks)
#     fw.add_text(chunks)

#     fw.save_to_disk(INDEX_PATH)

#     res = fw.search_text("ciao")
#     print(res)

#     res = fw.search_text("mattia")
#     print(res)

#     res = fw.search_text("topo")
#     print(res)

#     del fw
#     print("--------testing--------")

#     fw = FaissWrapper(
#         index_path=INDEX_PATH,
#         dataset=dataset,
#         device="cpu",
#         embedder=embedder,
#     )

#     res = fw.search_text("ciao")
#     print(res)

#     res = fw.search_text("mattia")
#     print(res)

#     res = fw.search_text("topo")
#     print(res)
