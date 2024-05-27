import os
import sys
import re
import json
from typing import List
import matplotlib.pyplot as plt

BENCHMARK_OUT_DIR = "out"

# load data from output folder
data = []

for filename in os.listdir(BENCHMARK_OUT_DIR):
    with open(os.path.join(BENCHMARK_OUT_DIR, filename)) as f:
        data.append(json.load(f))

for d in data:
    del d["metrics"]["answers"]

config_path = "configs/llm_vm.yaml"
k_shot_ok = [0]

# filter by ['args']['config_path']


new_data = {}

for d in data:
    cfpath = d["args"]["config_path"]
    if d["args"]["k_shot"] not in k_shot_ok:
        continue
    new_data[cfpath] = new_data.get(cfpath, [])

    if d["args"]["use_rag"] == False:
        new_data[cfpath].append((0, d["metrics"]["accuracy"]))
        continue

    new_data[cfpath].append((d["args"]["n_docs_retrieved"], d["metrics"]["accuracy"]))

for name, values in new_data.items():
    print(name, len(values))
    # sort by x
    values.sort(key=lambda x: x[0])

    x, y = zip(*values)
    plt.plot(x, y, label=name, marker="^")

plt.title("Accuracy (k_shot=0)")
plt.xlabel("n_docs_retrieved")
plt.ylabel("accuracy")
plt.grid()
# plt.legend()
plt.show()

# - accuracy vs number of documents retrieved, including 0 (can be done for a few chkpts, putting the curves in the same graph, as above. i would not use too many chkpts though)

# - accuracy vs training_step (using replug, kshot 0, fixed number of documents (e.g. 3))

n_docs_retrieved = 5
k_shot = 0
inference_type = "replug"

new_data = {}  # in the form x: y
for d in data:
    if d["args"]["n_docs_retrieved"] != n_docs_retrieved:
        continue
    if d["args"]["k_shot"] != k_shot:
        continue
    if d["args"]["inference_type"] != inference_type:
        continue
    if d["args"]["use_rag"] == False:
        continue

    cfpath = d["args"]["config_path"]

    # cfpath is like "configs/checkpoints/step1000.yaml"
    # we want to extract the step number

    m = re.search(r"step(\d+)", cfpath)
    if m:
        step = int(m.group(1))
    else:
        step = 0

    new_data[step] = d["metrics"]["accuracy"]

print(new_data)
x, y = zip(*sorted(new_data.items()))

plt.plot(x, y, marker="^")

plt.title(
    f"Accuracy (k_shot={k_shot}, n_docs_retrieved={n_docs_retrieved}, inference_type={inference_type})"
)
plt.xlabel("training_step")
plt.ylabel("accuracy")
plt.grid()
plt.show()

# accuracy vs kshots (using replug, fixed number of documents (e.g. 3), fixed training_step)

n_docs_retrieved = 3
config_path = "configs/checkpoints/step1000.yaml"
config_path = "configs/llm_vm.yaml"
inference_type = "replug"

new_data = {}  # in the form x: y

for d in data:
    if d["args"]["n_docs_retrieved"] != n_docs_retrieved:
        continue
    if d["args"]["config_path"] != config_path:
        continue
    if d["args"]["inference_type"] != inference_type:
        continue
    if d["args"]["use_rag"] == False:
        continue

    k_shot = d["args"]["k_shot"]

    new_data[k_shot] = d["metrics"]["accuracy"]

print(new_data)

x, y = zip(*sorted(new_data.items()))
plt.plot(x, y, marker="^")
plt.title(
    f"Accuracy (config_path={config_path}, n_docs_retrieved={n_docs_retrieved}, inference_type={inference_type})"
)
plt.xlabel("k_shot")
plt.ylabel("accuracy")
plt.grid()
plt.show()


