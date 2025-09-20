import os
import numpy as np
import pandas as pd
import faiss

EMB_FILE = "../outputs/embeddings.npy"
META_FILE = "../outputs/metadata.parquet"
FAISS_INDEX_FILE = "../outputs/faiss_index.idx"

def load_embeddings():
    return np.load(EMB_FILE).astype("float32")

def build_index(emb_matrix, ids):
    print("Normalizing embeddings (L2)...")
    faiss.normalize_L2(emb_matrix)
    dim = emb_matrix.shape[1]
    print("Dimension:", dim)

    base_index = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap(base_index)
    index.add_with_ids(emb_matrix, ids.astype("int64"))
    return index

def main():
    os.makedirs("../outputs", exist_ok=True)

    if os.path.exists(FAISS_INDEX_FILE):
        print("Found existing FAISS index:", FAISS_INDEX_FILE)
        index = faiss.read_index(FAISS_INDEX_FILE)
        print("Loaded FAISS index with ntotal =", index.ntotal)
        return

    print("Loading metadata...")
    meta = pd.read_parquet(META_FILE)
    print("Loading embeddings...")
    emb = load_embeddings()

    if emb.shape[0] != meta.shape[0]:
        raise ValueError("Mismatch between embeddings and metadata!")

    ids = meta["id"].astype("int64").values
    index = build_index(emb, ids)

    print("Saving FAISS index to", FAISS_INDEX_FILE)
    faiss.write_index(index, FAISS_INDEX_FILE)
    print("Saved. ntotal =", index.ntotal)

if __name__ == "__main__":
    main()
