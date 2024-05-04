import json
from backend.data_cleaning import utils
from backend.vector_database.faiss_wrapper import FaissWrapper

wiki_path = "wikidump_processing/data/subsample_chunked.xml"

dataset = []

with open(wiki_path, "r") as f:
    for page in utils.scroll_pages(f):
        page = utils.extract_tag(page, tag="page", add_tag=False)
        page = json.loads(page)
        for chunk in page:
            dataset.append(chunk["text"])
        if len(dataset) > 10000:
            break

fw = FaissWrapper(dim=8 * 16, index_str="IVF50,PQ8np", dataset=dataset, n_neighbors=1)
fw.train_and_add_index_from_text(dataset, dataset)

fw.save_to_disk("wikipedia_dummy.index")

res = fw.search_text("ciao")
print(res)

del fw

fw = FaissWrapper(index_path="wikipedia_dummy.index", dataset=dataset, n_neighbors=1)
res = fw.search_text("ciao")
print(res)
