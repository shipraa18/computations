# # scripts/vector_db_search_fix.py
# import pandas as pd
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer
# import re
# import math
# from typing import Tuple

# DATA_FILE = "../outputs/processed_embeddings.csv"
# MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# FAISS_INDEX_FILE = "../outputs/faiss_index.bin"  # optional - we build in memory here
# TOP_K = 100        # retrieve a larger candidate set, then filter & sort
# RETURN_K = 5       # final number to return

# # ---------- helpers ----------
# def parse_currency_to_number(s: str) -> Tuple[int, int]:
#     """Return (min, max) in INR or (None,None). Handles formats like '₹1,50,000', '1.5 L', '1-2 L', '150k'."""
#     if s is None or (isinstance(s, float) and math.isnan(s)):
#         return None, None
#     s = str(s).strip().lower().replace(",", "").replace("inr", "").replace("rs.", "").replace("rs", "").replace("₹", "")
#     if s == "":
#         return None, None
#     # handle ranges "1-2 l", "1.5-2 l"
#     s = s.replace("–", "-").replace("—", "-")
#     multiplier = 1
#     if "lakh" in s or "lac" in s or "l " in s or "l." in s:
#         multiplier = 100000
#     elif "k" in s and not "lak" in s:
#         multiplier = 1000
#     # extract numbers
#     nums = re.findall(r"\d+\.?\d*", s)
#     if not nums:
#         return None, None
#     nums = [float(n) for n in nums]
#     # heuristics:
#     if multiplier != 1:
#         nums = [n * multiplier for n in nums]
#     # Now return min, max if range else same
#     if len(nums) == 1:
#         return int(nums[0]), int(nums[0])
#     else:
#         return int(min(nums)), int(max(nums))

# def normalize_mode(s):
#     if pd.isna(s) or s is None:
#         return ""
#     return str(s).strip().lower()

# def ensure_embeddings_array(x):
#     # input could be string representation like "[0.1, 0.2,...]" or list/ndarray
#     if isinstance(x, str):
#         arr = np.array(eval(x), dtype=np.float32)
#     elif isinstance(x, list):
#         arr = np.array(x, dtype=np.float32)
#     elif isinstance(x, np.ndarray):
#         arr = x.astype(np.float32)
#     else:
#         arr = np.array(x, dtype=np.float32)
#     return arr

# def l2_normalize_rows(mat):
#     norms = np.linalg.norm(mat, axis=1, keepdims=True)
#     norms[norms == 0] = 1e-10
#     return mat / norms

# # ---------- main ----------
# def main():
#     print("Loading processed data...")
#     df = pd.read_csv(DATA_FILE)

#     # Ensure we have a numeric fees column for filtering & sorting
#     if "fees_min" not in df.columns:
#         print("Parsing fees into numeric fees_min / fees_max...")
#         fees_min_list, fees_max_list = [], []
#         for s in df.get("Fees", df.get("fees", df.get("Course Fee (INR)", [""]*len(df)))):
#             mn, mx = parse_currency_to_number(s)
#             fees_min_list.append(mn)
#             fees_max_list.append(mx)
#         df["fees_min"] = fees_min_list
#         df["fees_max"] = fees_max_list

#     # Normalize mode column
#     if "Mode" in df.columns:
#         df["mode_norm"] = df["Mode"].apply(normalize_mode)
#     else:
#         df["mode_norm"] = df.get("Mode of Study", "").apply(normalize_mode) if "Mode of Study" in df.columns else ""
    
#     # Load and normalize embeddings
#     print("Loading embeddings and creating numpy array...")
#     emb_list = []
#     for raw in df["embedding"].values:
#         arr = ensure_embeddings_array(raw)
#         emb_list.append(arr)
#     embeddings = np.vstack(emb_list).astype("float32")  # shape (N, D)
#     print("Embeddings shape:", embeddings.shape)
#     embeddings = l2_normalize_rows(embeddings)  # normalize for cosine

#     # Build FAISS index (inner product on normalized vectors = cosine)
#     dim = embeddings.shape[1]
#     index = faiss.IndexFlatIP(dim)
#     index.add(embeddings)
#     print("FAISS index built with", index.ntotal, "vectors")

#     # Load model for query embeddings
#     model = SentenceTransformer(MODEL_NAME)

#     # Example query - you can replace this
#     query = "cheapest online MBA in India"
#     print("\nQuery:", query)
#     q_vec = model.encode([query]).astype("float32")
#     q_vec = q_vec / (np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-10)

#     # Search top_k candidates
#     D, I = index.search(q_vec, TOP_K)
#     dists = D[0]
#     idxs = I[0]

#     # Prepare candidate list, apply 'online' filter and require parsed fees
#     candidates = []
#     for rank_pos, (idx, score) in enumerate(zip(idxs, dists)):
#         if idx < 0:  # in case less than TOP_K vectors
#             continue
#         row = df.iloc[idx]
#         # mode filter: ensure it contains 'online' or 'distance'
#         mode_val = str(row.get("mode_norm", "")).lower()
#         if "online" not in mode_val and "distance" not in mode_val and "remote" not in mode_val:
#             continue
#         fee = row.get("fees_min")
#         if fee is None or (isinstance(fee, float) and np.isnan(fee)):
#             continue
#         candidates.append({
#             "idx": int(idx),
#             "score": float(score),
#             "fees_min": int(fee),
#             "university": row.get("University Name", ""),
#             "course": row.get("Course Name", ""),
#             "combined_text": row.get("combined_text", row.get("Course Name",""))
#         })

#     if not candidates:
#         print("No candidates matched mode='online' and with parsed fees. Consider lowering filters or ensure fees are present.")
#         return

#     # Sort by fees, then by score (you can change weighting)
#     candidates_sorted = sorted(candidates, key=lambda x: (x["fees_min"], -x["score"]))

#     print(f"\nTop {RETURN_K} results (filtered by online & sorted by fees):")
#     for i, c in enumerate(candidates_sorted[:RETURN_K], 1):
#         print(f"{i}. {c['university']} - {c['course']} | Fees: {c['fees_min']} | Score: {c['score']:.4f}")
#         print("   Text snippet:", (c["combined_text"][:200] + "...") if c["combined_text"] else "")

# if __name__ == "__main__":
#     main()


# vector_db_search_improved.py
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import re
import math
import ast
from typing import Tuple, Optional

# CONFIG
DATA_FILE = "../outputs/processed_embeddings.csv"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 500        # candidates to fetch from FAISS before filtering
RETURN_K = 5       # final results to return
MIN_CANDIDATES_AFTER_FILTER = 20  # warning threshold

# ---------- helpers ----------
def parse_currency_to_number(s: str) -> Tuple[Optional[int], Optional[int]]:
    """Return (min, max) in INR or (None,None). Handles formats like '₹1,50,000', '1.5 L', '1-2 L', '150k'."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return None, None
    s = str(s).strip().lower().replace(",", "").replace("inr", "").replace("rs.", "").replace("rs", "").replace("₹", "")
    if s == "":
        return None, None
    s = s.replace("–", "-").replace("—", "-")
    # determine multiplier
    multiplier = 1
    if re.search(r"\b(lakh|lac|l)\b", s):
        multiplier = 100000
    elif "k" in s and not re.search(r"\blak", s):
        multiplier = 1000
    # extract numbers
    nums = re.findall(r"\d+\.?\d*", s)
    if not nums:
        return None, None
    nums = [float(n) for n in nums]
    # apply multiplier heuristics only if unit present
    if multiplier != 1 and re.search(r"(lakh|lac|l|k)\b", s):
        nums = [n * multiplier for n in nums]
    # return min,max
    if len(nums) == 1:
        return int(nums[0]), int(nums[0])
    return int(min(nums)), int(max(nums))

def normalize_mode(s):
    if pd.isna(s) or s is None:
        return ""
    return str(s).strip().lower()

def ensure_embeddings_array(x):
    """
    Safe conversion of stored embedding -> numpy array.
    Handles list, numpy array, or string repr of list (using ast.literal_eval).
    """
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    if isinstance(x, list):
        return np.array(x, dtype=np.float32)
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            return np.array(parsed, dtype=np.float32)
        except Exception:
            # try to parse a comma separated string
            try:
                vals = [float(v) for v in re.findall(r"-?\d+\.?\d*", x)]
                return np.array(vals, dtype=np.float32)
            except Exception:
                raise ValueError("Unable to parse embedding from string.")
    # fallback: try to convert
    return np.array(x, dtype=np.float32)

def l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return mat / norms

def extract_filters_from_query(q: str):
    """
    Tries to auto-extract: course keyword (MBA, BBA, etc.) and max_fee.
    Returns: dict { 'course': 'mba' or None, 'max_fee': int or None, 'mode': 'online'/'offline'/None }
    """
    ql = q.lower()
    # Course keywords (extend this list as needed)
    course_match = re.search(r'\b(mba|bba|msc|ma|btech|mtech|bsc|ba|phd|mca)\b', ql)
    course = course_match.group(1).lower() if course_match else None

    # Mode keywords
    mode = None
    if re.search(r'\bonline\b|\bdistance\b|\bremote\b', ql):
        mode = "online"
    elif re.search(r'\boffline\b|\bcampus\b|\bregular\b', ql):
        mode = "offline"

    # max fee detection: "under 2 lakh", "below 150000", "less than 2L"
    m = re.search(r'(under|below|less than|<|<=)\s*([\d\.]+)\s*(lakh|lac|l|k)?', ql)
    max_fee = None
    if m:
        num = float(m.group(2))
        unit = (m.group(3) or "").lower()
        if unit.startswith('l'):
            max_fee = int(num * 100000)
        elif unit.startswith('k'):
            max_fee = int(num * 1000)
        else:
            max_fee = int(num)
    # also "under 200000" plain number
    if max_fee is None:
        m2 = re.search(r'under\s*([\d,]+)', ql)
        if m2:
            try:
                max_fee = int(m2.group(1).replace(',', ''))
            except:
                pass

    return {"course": course, "max_fee": max_fee, "mode": mode}

# ---------- main ----------
def main():
    print("Loading processed data...")
    df = pd.read_csv(DATA_FILE)

    # Ensure consistent column names by trimming whitespace
    df.columns = [c.strip() for c in df.columns]

    # Ensure numeric fee columns exist
    if "fees_min" not in df.columns:
        print("Parsing fees into numeric fees_min / fees_max...")
        fees_min_list, fees_max_list = [], []
        # try common fee columns, fallback to 'Fees' or variants
        fee_source_col = None
        for candidate in ["Fees", "fees", "Course Fee (INR)", "Course Fee", "fees_text", "Fee"]:
            if candidate in df.columns:
                fee_source_col = candidate
                break
        if fee_source_col is None:
            fee_source_col = df.columns[0]  # fallback to first col - not ideal
            print("Warning: couldn't find a fee column automatically. Using", fee_source_col)
        fee_series = df[fee_source_col].fillna("").astype(str).tolist()
        for s in fee_series:
            mn, mx = parse_currency_to_number(s)
            fees_min_list.append(mn)
            fees_max_list.append(mx)
        df["fees_min"] = fees_min_list
        df["fees_max"] = fees_max_list

    # Normalize mode column if present
    mode_col = None
    for cand in ["Mode", "mode", "Mode of Study", "mode_of_study"]:
        if cand in df.columns:
            mode_col = cand
            break
    if mode_col:
        df["mode_norm"] = df[mode_col].apply(normalize_mode)
    else:
        df["mode_norm"] = ""

    # Load and normalize embeddings
    print("Loading embeddings and creating numpy array...")
    emb_list = []
    for raw in df["embedding"].values:
        arr = ensure_embeddings_array(raw)
        emb_list.append(arr)
    embeddings = np.vstack(emb_list).astype("float32")  # shape (N, D)
    print("Embeddings shape:", embeddings.shape)

    # Normalize embeddings (L2) and build cosine-friendly index (Inner Product on normalized vectors)
    embeddings = l2_normalize_rows(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print("FAISS index built with", index.ntotal, "vectors")

    # Load model for query embeddings
    model = SentenceTransformer(MODEL_NAME)

    # Example interactive query - you can change this
    query = "cheapest online MBA in India under 2 lakh"
    print(f"\nQuery: {query}")

    # Extract filters automatically from query
    filters = extract_filters_from_query(query)
    print("Extracted filters:", filters)

    # Encode & normalize query vector
    q_vec = model.encode([query]).astype("float32")
    q_vec = q_vec / (np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-10)

    # Search top_k candidates
    TOP_K_LOCAL = TOP_K if TOP_K <= index.ntotal else index.ntotal
    D, I = index.search(q_vec, TOP_K_LOCAL)
    dists = D[0]
    idxs = I[0]

    # Prepare candidate list, apply mode filter and parsed fees and course filter
    candidates = []
    for idx, score in zip(idxs, dists):
        if idx < 0:
            continue
        row = df.iloc[int(idx)]
        # mode filter (if extracted)
        if filters["mode"]:
            mode_val = str(row.get("mode_norm", "")).lower()
            if filters["mode"] not in mode_val:
                continue

        # course filter (if extracted)
        if filters["course"]:
            course_name = str(row.get("Course Name", "") or row.get("course", "") or "").lower()
            combined_text = str(row.get("combined_text", "")).lower()
            # check in both course field and combined_text as fallback
            if filters["course"] not in course_name and filters["course"] not in combined_text:
                continue

        # fee filter
        fee_val = row.get("fees_min")
        if filters["max_fee"] is not None:
            if fee_val is None or (isinstance(fee_val, float) and math.isnan(fee_val)):
                continue
            if fee_val > filters["max_fee"]:
                continue

        candidates.append({
            "idx": int(idx),
            "score": float(score),
            "fees_min": int(fee_val) if fee_val is not None and not (isinstance(fee_val, float) and math.isnan(fee_val)) else None,
            "university": row.get("University Name", ""),
            "course": row.get("Course Name", ""),
            "combined_text": row.get("combined_text", row.get("Course Name",""))
        })

    if not candidates:
        print("No candidates matched the extracted filters. Consider increasing TOP_K or loosening filters.")
        return

    # Optional: group by (university+course) to avoid duplicates from chunking
    grouped = {}
    for c in candidates:
        key = (c["university"], c["course"])
        if key not in grouped:
            grouped[key] = c
        else:
            # keep the one with lower fees or higher score if fees equal/none
            existing = grouped[key]
            if c["fees_min"] is not None and (existing["fees_min"] is None or c["fees_min"] < existing["fees_min"]):
                grouped[key] = c
            elif c["fees_min"] == existing["fees_min"] and c["score"] > existing["score"]:
                grouped[key] = c

    candidates_unique = list(grouped.values())

    # Sort by fees then by similarity score (you can tune weights)
    candidates_sorted = sorted(
        candidates_unique,
        key=lambda x: ((x["fees_min"] if x["fees_min"] is not None else 10**12), -x["score"])
    )

    print(f"\nTop {RETURN_K} results (after filtering & grouping):")
    for i, c in enumerate(candidates_sorted[:RETURN_K], 1):
        print(f"{i}. {c['university']} - {c['course']} | Fees: {c['fees_min']} | Score: {c['score']:.4f}")
        snippet = (c["combined_text"] or "")[:300]
        print("   Snippet:", snippet + ("..." if len(snippet) >= 300 else ""))

    if len(candidates_sorted) < MIN_CANDIDATES_AFTER_FILTER:
        print("\n[Note] Low candidate count after filtering — consider increasing TOP_K, relaxing filters, or ensuring metadata (fees/mode/course) is filled for more rows.")

if __name__ == "__main__":
    main()
