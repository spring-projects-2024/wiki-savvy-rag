import faiss
from faiss.contrib import datasets
import time

from faiss.contrib.evaluation import knn_intersection_measure

D = 768 // 2
M = 256
centroids = 1000
# best quantization is PQ{M}x4fsr but requires clustering

# HNSW is pretty good for accuracy
for nn in [16, 32, 64]:
    for pq in [12, 24]:
        # index_str = f"IVF{centroids}_HNSW32,{sq_type}"
        index_str = f"HNSW{nn}_PQ{pq}"
        index = faiss.index_factory(D, index_str, faiss.METRIC_INNER_PRODUCT)
        index.nprobe = 16


exit()
# Load the dataset
dataset = datasets.SyntheticDataset(D, 1, N, 1000)

# Train the index
print("Training the index...")
t = time.time()
index.train(dataset.xb)
print(f"Elapsed time: {time.time() - t:.2f}s")

index.add(dataset.xb)
print(f"Elapsed time: {time.time() - t:.2f}s")

# benchmark performance
t = time.time()
D, Iref = index.search(dataset.xq, 100)

print(f"Elapsed time: {time.time() - t:.2f}s")

faiss.write_index(index, "temp.index")

D, Inew = faiss.knn(dataset.xq, dataset.xb, 100, metric=faiss.METRIC_INNER_PRODUCT)

res = {
    rank: knn_intersection_measure(Inew[:, :rank], Iref[:, :rank])
    for rank in [1, 10, 100]
}

print(res)
