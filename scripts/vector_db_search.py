

# # ----------------------
# # vector_db_search.py - robust version
# # ----------------------
# import faiss
# import numpy as np
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# import re

# FAISS_INDEX_FILE = "../outputs/faiss_index.idx"
# META_FILE = "../outputs/metadata.parquet"
# MODEL_NAME = "intfloat/e5-base"
# TOP_K = 200
# RETURN_K = 5

# # ----------------------
# # Normalization functions
# # ----------------------
# def normalize_course(text):
#     """Lowercase, remove dots, standardize course names."""
#     if not text:
#         return ""
#     text = text.lower().replace(".", "")
#     text = text.replace("master of science", "msc")
#     text = text.replace("bachelor of science", "bsc")
#     text = text.replace("bachelor of arts", "ba")
#     text = text.replace("master of business administration", "mba")
#     text = text.replace("bachelor of business administration", "bba")
#     # Add more standardizations as needed
#     return text

# def extract_course_from_query(q):
#     """Extract main course keyword from user query using exact word match."""
#     q_lower = q.lower().replace(".", "")
#     patterns = {
#         "msc": ["msc", "master of science"],
#         "bsc": ["bsc", "bachelor of science"],
#         "ba": ["ba", "bachelor of arts"],
#         "mba": ["mba", "master of business administration"],
#         "bba": ["bba", "bachelor of business administration"]
#     }
#     for key, keywords in patterns.items():
#         for kw in keywords:
#             if re.search(rf"\b{kw}\b", q_lower):
#                 return key
#     return None

# def parse_currency_to_number(s):
#     """Parse course fees like '₹2,50,000', '3 Lakh', '2.5–3 Lakh'."""
#     if not s:
#         return None
#     s = str(s).lower().replace(",", "").replace("₹", "").strip()
#     # Handle ranges like "2.5-3 lakh"
#     if "–" in s or "-" in s:
#         s = s.replace("–", "-")
#         parts = s.split("-")
#         s = parts[0]  # take lower end
#     # Handle "lakh"
#     if "lakh" in s or "l" in s:
#         nums = re.findall(r"\d+\.?\d*", s)
#         return int(float(nums[0]) * 100000) if nums else None
#     nums = re.findall(r"\d+\.?\d*", s)
#     return int(nums[0]) if nums else None

# def normalize_query_vec(q_vec):
#     q_vec = q_vec.astype("float32")
#     q_norm = np.linalg.norm(q_vec, axis=1, keepdims=True)
#     q_norm[q_norm == 0] = 1e-10
#     return q_vec / q_norm

# # ----------------------
# # Main search
# # ----------------------
# def main():
#     print("Loading metadata...")
#     meta = pd.read_parquet(META_FILE)

#     # Normalize course names in dataset
#     meta["Course Name_norm"] = meta["Course Name"].apply(normalize_course)
#     meta["combined_text_norm"] = meta["combined_text"].apply(normalize_course)

#     print("Loading FAISS index...")
#     index = faiss.read_index(FAISS_INDEX_FILE)
#     print("Index loaded, ntotal =", index.ntotal)

#     model = SentenceTransformer(MODEL_NAME, device="cpu")

#     while True:
#         q = input("\nQuery (or 'exit'): ").strip()
#         if q.lower() in ("exit", "quit"):
#             break

#         # Extract course keyword
#         course_filter = extract_course_from_query(q)
#         if not course_filter:
#             print("Could not detect course in query. Try again.")
#             continue

#         # Parse budget from query if present (optional)
#         max_fee = 300000  # Example hard-coded. Could parse dynamically if needed.

#         # Embed query
#         q_vec = model.encode([q], convert_to_numpy=True)
#         q_vec = normalize_query_vec(q_vec)

#         # Search top-k embeddings
#         k = TOP_K if TOP_K <= index.ntotal else index.ntotal
#         D, I = index.search(q_vec, k)

#         # Load candidate metadata
#         cand_meta = meta.set_index("id").loc[I[0]].reset_index()
#         cand_meta["score"] = D[0]

#         # Filter by course keyword using exact word boundaries
#         cand_meta = cand_meta[
#             cand_meta["Course Name_norm"].str.contains(rf"\b{course_filter}\b", na=False) |
#             cand_meta["combined_text_norm"].str.contains(rf"\b{course_filter}\b", na=False)
#         ]

#         # Parse fees and filter by max_fee
#         cand_meta["fee_num"] = cand_meta["Course Fee (INR)"].apply(parse_currency_to_number)
#         cand_meta = cand_meta[cand_meta["fee_num"].notna() & (cand_meta["fee_num"] <= max_fee)]

#         # Sort by fee first, then embedding score
#         cand_meta["fee_sort"] = cand_meta["fee_num"].fillna(10**12)
#         cand_meta = cand_meta.sort_values(by=["fee_sort", "score"], ascending=[True, False])

#         # Display top results
#         if cand_meta.empty:
#             print("No results found for this query.")
#             continue

#         print("\nTop results:")
#         for i, row in cand_meta.head(RETURN_K).iterrows():
#             print(f"{row['University Name']} - {row['Course Name']} | Fee: {row.get('Course Fee (INR)', 'N/A')} | score: {row['score']:.4f}")
#             snippet = (row["combined_text"] or "")[:250] + ("..." if len(row["combined_text"]) > 250 else "")
#             print("  snippet:", snippet)

# if __name__ == "__main__":
#     main()



# vector_db_search.py - robust version
# ----------------------
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import re

FAISS_INDEX_FILE = "../outputs/faiss_index.idx"
META_FILE = "../outputs/metadata.parquet"
MODEL_NAME = "intfloat/e5-base"
TOP_K = 200
RETURN_K = 5


# ----------------------
# Normalization functions
# ----------------------
def normalize_course(text):
    """Lowercase, remove dots, standardize course names."""
    if not text:
        return ""
    text = text.lower().replace(".", "")
    text = text.replace("master of science", "msc")
    text = text.replace("bachelor of science", "bsc")
    text = text.replace("bachelor of arts", "ba")
    text = text.replace("master of business administration", "mba")
    text = text.replace("bachelor of business administration", "bba")
    # Add more standardizations as needed
    return text


def extract_course_from_query(q):
    """Extract main course keyword from user query using exact word match."""
    q_lower = q.lower().replace(".", "")
    patterns = {
        "msc": ["msc", "master of science"],
        "bsc": ["bsc", "bachelor of science"],
        "ba": ["ba", "bachelor of arts"],
        "mba": ["mba", "master of business administration"],
        "bba": ["bba", "bachelor of business administration"],
    }
    for key, keywords in patterns.items():
        for kw in keywords:
            if re.search(rf"\b{kw}\b", q_lower):
                return key
    return None


def parse_currency_to_number(s):
    """Parse course fees like '₹2,50,000', '3 Lakh', '2.5–3 Lakh'."""
    if not s:
        return None
    s = str(s).lower().replace(",", "").replace("₹", "").strip()

    # Handle ranges like "2.5-3 lakh"
    if "–" in s or "-" in s:
        s = s.replace("–", "-")
        parts = s.split("-")
        s = parts[0]  # take lower end

    # Handle "lakh"
    if "lakh" in s or "l" in s:
        nums = re.findall(r"\d+\.?\d*", s)
        return int(float(nums[0]) * 100000) if nums else None

    nums = re.findall(r"\d+\.?\d*", s)
    return int(nums[0]) if nums else None


def normalize_query_vec(q_vec):
    q_vec = q_vec.astype("float32")
    q_norm = np.linalg.norm(q_vec, axis=1, keepdims=True)
    q_norm[q_norm == 0] = 1e-10
    return q_vec / q_norm


# ----------------------
# Main search
# ----------------------
def main():
    print("Loading metadata...")
    meta = pd.read_parquet(META_FILE)

    # Normalize course names in dataset
    meta["Course Name_norm"] = meta["Course Name"].apply(normalize_course)
    meta["combined_text_norm"] = meta["combined_text"].apply(normalize_course)

    print("Loading FAISS index...")
    index = faiss.read_index(FAISS_INDEX_FILE)
    print("Index loaded, ntotal =", index.ntotal)

    model = SentenceTransformer(MODEL_NAME, device="cpu")

    while True:
        q = input("\nQuery (or 'exit'): ").strip()
        if q.lower() in ("exit", "quit"):
            break

        # Extract course keyword
        course_filter = extract_course_from_query(q)
        if not course_filter:
            print("Could not detect course in query. Try again.")
            continue

        # Parse budget from query if present (optional)
        max_fee = 300000  # Example hard-coded. Could parse dynamically if needed.

        # Embed query
        q_vec = model.encode([q], convert_to_numpy=True)
        q_vec = normalize_query_vec(q_vec)

        # Search top-k embeddings
        k = TOP_K if TOP_K <= index.ntotal else index.ntotal
        D, I = index.search(q_vec, k)

        # Load candidate metadata
        cand_meta = meta.set_index("id").loc[I[0]].reset_index()
        cand_meta["score"] = D[0]

        # Filter by course keyword using exact word boundaries
        cand_meta = cand_meta[
            cand_meta["Course Name_norm"].str.contains(rf"\b{course_filter}\b", na=False)
            | cand_meta["combined_text_norm"].str.contains(rf"\b{course_filter}\b", na=False)
        ]

        # Parse fees and filter by max_fee
        cand_meta["fee_num"] = cand_meta["Course Fee (INR)"].apply(parse_currency_to_number)
        cand_meta = cand_meta[cand_meta["fee_num"].notna() & (cand_meta["fee_num"] <= max_fee)]

        # Sort by fee first, then embedding score
        cand_meta["fee_sort"] = cand_meta["fee_num"].fillna(10**12)
        cand_meta = cand_meta.sort_values(by=["fee_sort", "score"], ascending=[True, False])

        # Display top results
        if cand_meta.empty:
            print("No results found for this query.")
            continue

        print("\nTop results:")
        for i, row in cand_meta.head(RETURN_K).iterrows():
            print(
                f"{row['University Name']} - {row['Course Name']} "
                f"| Fee: {row.get('Course Fee (INR)', 'N/A')} "
                f"| score: {row['score']:.4f}"
            )
            snippet = (row["combined_text"] or "")[:250]
            if len(row["combined_text"]) > 250:
                snippet += "..."
            print(" snippet:", snippet)


if __name__ == "__main__":
    main()
