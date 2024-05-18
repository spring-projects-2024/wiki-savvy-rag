from typing import Tuple, List, Iterable

# keep in this specific order, otherwise it gives Segmentation Fault on Federico's pc
from backend.vector_database.embedder_wrapper import EmbedderWrapper
import faiss
import numpy as np
from backend.vector_database.dataset import MockDataset


# TODO: allow processing queries in batches


class FaissWrapper:
    def __init__(
        self,
        device,
        dataset,
        embedder,
        *,
        index_path=None,
        index_str=None,
        nprobe=None,
    ):
        """
        Instantiate a FaissWrapper object.
        :param index_path: path to a saved index, optional and exclusive with index_str
        :param index_str: Faiss index string, optional and exclusive with index_path
        :param n_neighbors: parameter k for knn, default 10
        :param dataset:
        :param dim: dimensionality of the vectors (required if index_str is passed, ignored otherwise)
        :param nprobe: number of probes for search, ignored if index is loaded from file
        """

        if embedder is None:
            embedder = EmbedderWrapper(device)

        self.device = device
        self.embedder = embedder
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

    def search_vectors(
        self, vectors: np.ndarray, n_neighbors: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for the nearest neighbors of n vectors.
        :param vector: np.ndarray of shape (n, dim)
        :return: Index matrix I and distance matrix D
        """
        return self._index.search(vectors, n_neighbors)

    def _search_vector(self, vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for the nearest neighbors of a vector.
        :param vector: np.ndarray of shape (dim)
        :return: Index matrix I and distance matrix D
        """
        return self.search_vectors(vector.reshape(1, -1))

    def _search_text_get_I_D(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the I and D matrices from a text. Wraps around _search_vector
        :return: Index matrix I and distance matrix D
        """
        vector = self.embedder.get_embedding(text).numpy()
        return self._search_vector(vector)

    def _index_to_text(self, index: int) -> str:
        """
        todo: implement this as it depends on the structure of the dataset
        :param index:
        :return:
        """
        return self.dataset.search_chunk(index)

    def search_text(self, text: str) -> List[Tuple[str, float]]:
        """
        Search for the nearest neighbors of a text.
        :param text:
        :return: List of tuples of (text, distance)
        todo: check if they are sorted by distance, there is an index in faiss that re-orders results at the end
        """

        I, D = self._search_text_get_I_D(text)

        return [(self._index_to_text(i), j) for i, j in zip(D[0], I[0]) if i != -1]

    def train_from_vectors(self, data, train_on_gpu=False):
        if train_on_gpu:
            index_ivf = faiss.extract_index_ivf(self._index)
            clustering_index = faiss.index_cpu_to_all_gpus(
                faiss.IndexFlatL2(index_ivf.d)
            )
            index_ivf.clustering_index = clustering_index

        self._index.train(data)

    def add_vectors(self, data):

        self._index.add(data)

    def train_from_text(self, data: Iterable[str]):
        """
        Wrapper around train_from_vectors that takes text as input.
        :param data:
        """

        self.train_from_vectors(self.embedder.get_embedding(data))

    def add_text(self, data: Iterable[str]):
        """
        Wrapper around add_vectors that takes text as input.
        :param data:
        """

        self.add_vectors(self.embedder.get_embedding(data))

    def save_to_disk(self, path):
        faiss.write_index(self._index, path)


INDEX_PATH = "backend/vector_database/data/faiss.index"

if __name__ == "__main__":
    chunks = ["ciao", "sono", "mattia"]
    dataset = MockDataset(chunks)
    embedder = EmbedderWrapper("cpu")

    fw = FaissWrapper(
        index_str="Flat", dataset=dataset, device="cpu", embedder=embedder
    )

    fw.train_from_text(chunks)
    fw.add_text(chunks)

    fw.save_to_disk(INDEX_PATH)

    res = fw.search_text("ciao")
    print(res)

    res = fw.search_text("mattia")
    print(res)

    res = fw.search_text("topo")
    print(res)

    del fw
    print("--------testing--------")

    fw = FaissWrapper(
        index_path="index_faiss.index",
        dataset=dataset,
        device="cpu",
        embedder=embedder,
    )

    res = fw.search_text("ciao")
    print(res)

    res = fw.search_text("mattia")
    print(res)

    res = fw.search_text("topo")
    print(res)
