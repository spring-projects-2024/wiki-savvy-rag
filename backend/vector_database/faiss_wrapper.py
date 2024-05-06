from typing import Tuple, List, Iterable

# keep in this specific order, otherwise it gives Segmentation Fault on Federico's pc
from backend.vector_database.embedder_wrapper import EmbedderWrapper
import faiss
import numpy as np
from backend.vector_database.dataset import MockDataset

# TODO: allow processing queries in batches


# order of arguments: dimension, train set size, database (?) size, query set size
# df = datasets.SyntheticDataset(64, 4000, 10_000, 50, metric=faiss.METRIC_INNER_PRODUCT)
#
# print(faiss.MatrixStats(df.xb).comments)
#
#
# index_fs = faiss.index_factory(DIM, "OPQ16_64,IVF1000(IVF100,PQ32x4fs,RFlat),PQ16x4fsr,Refine(OPQ56_112,PQ56)",
#                                faiss.METRIC_INNER_PRODUCT)


class FaissWrapper:
    def __init__(
        self,
        device,
        dataset,
        embedder,
        *,
        index_path=None,
        index_str=None,
        n_neighbors=10,
    ):
        """
        Instantiate a FaissWrapper object.
        :param index_path: path to a saved index, optional and exclusive with index_str
        :param index_str: Faiss index string, optional and exclusive with index_path
        :param n_neighbors: parameter k for knn, default 10
        :param dataset:
        :param dim: dimensionality of the vectors (required if index_str is passed, ignored otherwise)
        """

        self.device = device
        self.embedder = embedder
        self.n_neighbors = n_neighbors
        self.dataset = dataset
        self.dim = self.embedder.get_dimensionality()

        if index_path:
            self._index = faiss.read_index(index_path)
            assert self.dim == self._index.d
        elif index_str:
            self._index = faiss.index_factory(
                self.dim, index_str, faiss.METRIC_INNER_PRODUCT
            )

    def _search_vector(self, vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for the nearest neighbors of a vector.
        :param vector: np.ndarray of shape (dim)
        :return: Index matrix I and distance matrix D
        """
        return self._index.search(vector.reshape(1, -1), self.n_neighbors)

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

    def train_from_vectors(self, data):
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
