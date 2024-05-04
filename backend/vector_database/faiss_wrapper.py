from typing import Tuple, List, Iterable

import faiss
import numpy as np
from faiss.contrib import datasets
import time

from backend.vector_database.embedder_wrapper import EmbedderWrapper


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
        self, *, device, index_path=None, index_str=None, n_neighbors=10, dataset=None
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
        self.embedder = EmbedderWrapper(device)
        self.dim = self.embedder.get_dimensionality()

        if index_path:
            self._index = faiss.read_index(index_path)
            assert self.dim == self._index.d
        elif index_str:
            self._index = faiss.index_factory(
                self.dim, index_str, faiss.METRIC_INNER_PRODUCT
            )

        self.n_neighbors = n_neighbors
        self.dataset = dataset

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
        if isinstance(self.dataset, list):
            return self.dataset[index]
        else:
            raise NotImplementedError

    def search_text(self, text: str) -> List[Tuple[str, float]]:
        """
        Search for the nearest neighbors of a text.
        :param text:
        :return: List of tuples of (text, distance)
        todo: check if they are sorted by distance, there is an index in faiss that re-orders results at the end
        """

        I, D = self._search_text_get_I_D(text)

        return [(self._index_to_text(i), j) for i, j in zip(D[0], I[0]) if i != -1]

    def train_and_add_index_from_vectors(self, train_data, add_data):
        self._index.train(train_data)

        # todo: check if it makes sense to make add_data iterable to avoid loading everything in ram
        # this todo propagates to train_and_add_index_from_text
        self._index.add(add_data)

    def train_and_add_index_from_text(
        self, train_data: Iterable[str], add_data: Iterable[str]
    ):
        """
        Wrapper around _train_and_add_index_from_vectors that takes text as input.
        :param train_data:
        :param add_data:
        """
        train_data = np.array(
            [self.embedder.get_embedding(text) for text in train_data]
        )
        add_data = np.array([self.embedder.get_embedding(text) for text in add_data])

        self.train_and_add_index_from_vectors(train_data, add_data)

    def save_to_disk(self, path):
        faiss.write_index(self._index, path)


if __name__ == "__main__":
    dataset = ["ciao", "sono", "mattia"]

    fw = FaissWrapper(dim=3, index_str="Flat", dataset=dataset)

    fw.train_and_add_index_from_text(dataset, dataset)

    fw.save_to_disk("index_faiss.index")

    res = fw.search_text("ciao")
    print(res)

    res = fw.search_text("mattia")
    print(res)

    res = fw.search_text("topo")
    print(res)

    del fw
    print("--------testing--------")

    fw = FaissWrapper(index_path="index_faiss.index", dataset=dataset)

    res = fw.search_text("ciao")
    print(res)

    res = fw.search_text("mattia")
    print(res)

    res = fw.search_text("topo")
    print(res)
