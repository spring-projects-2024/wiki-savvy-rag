import tqdm
import json
import pandas as pd
from collections import defaultdict


tree_path = "children_cats.csv"
roots_path = "roots.json"
max_depth = 3


def depth_first(data, key, visited, depth, max_depth):
    if depth > max_depth:
        return
    visited.add(key)
    if key in data:
        for child in data[key]:
            if child not in visited:
                depth_first(data, child, visited, depth + 1, max_depth)
    return visited


def process_category_tree(tree_path):
    df = pd.read_csv(tree_path)
    parent_to_children = defaultdict(list)
    child_to_parents = defaultdict(list)
    for index, row in tqdm.tqdm(df.iterrows()):
        parent = row["parent"].lower()
        children = [x.lower() for x in row["children"].split()]
        parent_to_children[parent].extend(children)
        for child in children:
            child_to_parents[child].append(parent)
    parent_to_children = {k: list(set(v)) for k, v in parent_to_children.items()}
    child_to_parents = {k: list(set(v)) for k, v in child_to_parents.items()}
    return parent_to_children, child_to_parents


if __name__ == "__main__":
    parent_to_children, child_to_parents = process_category_tree(tree_path)
    with open(roots_path, "r") as f:
        roots = json.load(f)
    selected = set()
    for root in roots:
        visited = depth_first(parent_to_children, root.lower(), set(), 0, max_depth)
        selected.update(visited)
    selected = list(selected)
    selected.sort()
    with open("selected_categories.json", "w") as f:
        json.dump(selected, f)
