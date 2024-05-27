import datetime
import os
import sys
import re
import json
import time
from typing import List
import matplotlib.pyplot as plt

TRUST_AFTER = datetime.datetime(year=2024, month=5, day=27, hour=22, minute=9)

def extract_steps(name):
    m = re.search(r"step(\d+)", name)
    if m:
        return int(m.group(1))
    return 0


BENCHMARK_OUT_DIR = "out"
ALLOWED_STEPS = [0, 100, 200, 300, 500, 700, 1000, 1500, 3000, 5000, 10000, 60000, 30000]
# load data from output folder
data = []

for filename in os.listdir(BENCHMARK_OUT_DIR):
    # filename has formt "mmlu_2024-05-27-21-43-00.json"
    extract_date: datetime.datetime = datetime.datetime.strptime(
        filename.split("_")[1].split(".")[0], "%Y-%m-%d-%H-%M-%S"
    )

    with open(os.path.join(BENCHMARK_OUT_DIR, filename)) as f:
        temp = json.load(f)

    if temp["args"]["config_path"] == "configs/llm_vm.yaml":
        data.append(temp)
        continue

    if extract_date > TRUST_AFTER:
        data.append(temp)

nd = []
for d in data:
    del d["metrics"]["answers"]

    steps = extract_steps(d["args"]["config_path"])
    if steps in ALLOWED_STEPS:
        nd.append(d)

data = nd

config_path = "configs/llm_vm.yaml"
k_shot_ok = [0]

# filter by ['args']['config_path']


new_data = {}

for d in data:
    cfpath = d["args"]["config_path"]
    if d["args"]["k_shot"] not in k_shot_ok:
        continue
    if d["args"]["inference_type"] != "replug":
        continue


    new_data[cfpath] = new_data.get(cfpath, [])

    if d["args"]["use_rag"] == False:
        new_data[cfpath].append((0, d["metrics"]["accuracy"]))
        continue



    new_data[cfpath].append((d["args"]["n_docs_retrieved"], d["metrics"]["accuracy"]))

for name, values in new_data.items():
    # sort by x
    values.sort(key=lambda x: x[0])

    x, y = zip(*values)
    print(x, y)
    plt.plot(x, y, label=name, marker="^")

plt.title("Replug before finetuning (k_shot=0)")
plt.xlabel("n_docs_retrieved")
plt.ylabel("accuracy")
plt.grid()
# save to disk
plt.savefig("../../media/replug_before_finetuning.png")

plt.show()



# accuracy vs kshots (using replug, fixed number of documents (e.g. 3), fixed training_step)

n_docs_retrieved = 3
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
    # f"Baseline LLM with {n_docs_retrieved} documents and {inference_type}"
    "Rag before finetuning"
)
plt.xlabel("k_shot")
plt.ylabel("accuracy")
plt.grid()
plt.savefig("../../media/k_shot_ablation.png")


# find naive accuracy on config/llm_vm.yaml

for d in data:
    if d["args"]["config_path"] == "configs/llm_vm.yaml" and d["args"]["k_shot"] == 0 and \
        d["args"]["n_docs_retrieved"] == 3 and d["args"]["inference_type"] == "replug":
            print("replug:  ", d["metrics"]["accuracy"])

    if d["args"]["config_path"] == "configs/llm_vm.yaml" and d["args"]["k_shot"] == 0 and \
        d["args"]["n_docs_retrieved"] == 3 and d["args"]["inference_type"] == "naive":
            print("naive:   ", d["metrics"]["accuracy"])

