# scripts/vector_db_search.py
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import re, math

FAISS_INDEX_FILE = "../outputs/faiss_index.bin"
META_FILE = "../outputs/metadata.parquet"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 200
RETURN_K = 5

def normalize_query(q_vec):
    q_vec = q_vec.astype("float32")
    q_norm = np.linalg.norm(q_vec, axis=1, keepdims=True)
    q_norm[q_norm==0] = 1e-10
    return q_vec / q_norm

def parse_currency_to_number(s):
    # reuse a simple parser - you can import yours
    if s is None or s=="":
        return None
    s = str(s).lower().replace(",", "").replace("₹", "").strip()
    if "lakh" in s or "l" in s:
        nums = re.findall(r"\d+\.?\d*", s)
        return int(float(nums[0]) * 100000) if nums else None
    nums = re.findall(r"\d+\.?\d*", s)
    return int(nums[0]) if nums else None

def simple_extract_course(q):
    m = re.search(r'\b(mba|bba|bsc|msc|mca|btech|mtech|ba)\b', q.lower())
    return m.group(1).lower() if m else None

def main():
    print("Loading metadata...")
    meta = pd.read_parquet(META_FILE)
    print("Loading FAISS index...")
    index = faiss.read_index(FAISS_INDEX_FILE)
    print("Index loaded, ntotal =", index.ntotal)

    model = SentenceTransformer(MODEL_NAME)

    while True:
        q = input("\nQuery (or 'exit'): ").strip()
        if q.lower() in ("exit","quit"):
            break

        print("Embedding query...")
        q_vec = model.encode([q], convert_to_numpy=True)
        q_vec = normalize_query(q_vec)

        # ask index
        k = TOP_K if TOP_K <= index.ntotal else index.ntotal
        D, I = index.search(q_vec, k)

        # Map results (I contains external IDs because we used IndexIDMap)
        ids = I[0].tolist()
        scores = D[0].tolist()

        # Load metadata rows for those ids
        # meta['id'] should be same id dtype
        cand_meta = meta.set_index("id").loc[ids].reset_index()
        cand_meta["score"] = scores

        # example: filter by course keyword if present in query
        course_filter = simple_extract_course(q)
        if course_filter:
            cand_meta = cand_meta[cand_meta["Course Name"].str.lower().str.contains(course_filter, na=False) |
                                  cand_meta["combined_text"].str.lower().str.contains(course_filter, na=False)]

        # try to parse fees in metadata
        if "Course Fee (INR)" in cand_meta.columns:
            cand_meta["fee_num"] = cand_meta["Course Fee (INR)"].apply(parse_currency_to_number)
        else:
            cand_meta["fee_num"] = None

        # Sort by fee if available else by score
        cand_meta = cand_meta.copy()
        cand_meta["fee_sort"] = cand_meta["fee_num"].fillna(10**12)
        cand_meta = cand_meta.sort_values(by=["fee_sort", "score"], ascending=[True, False])

        print("\nTop results:")
        for i, row in cand_meta.head(RETURN_K).iterrows():
            print(f"{row['University Name']} - {row['Course Name']} | Fee: {row.get('Course Fee (INR)', 'N/A')} | score: {row['score']:.4f}")
            print("  snippet:", (row["combined_text"] or "")[:250] + ("..." if len(row["combined_text"])>250 else ""))

if __name__ == "__main__":
    main()
