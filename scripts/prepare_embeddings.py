# # scripts/prepare_embeddings.py

# import pandas as pd
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import os

# # ------------------------
# # CONFIG
# # ------------------------
# INPUT_FILE = "../data/final.csv"
# OUTPUT_FILE = "../outputs/processed_embeddings.csv"
# MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # HuggingFace embedding model
# COLUMNS_TO_USE = [
#     "University Name",
#     "Course Name",
#     "Specializations Available",
#     "Mode of Study",
#     "Eligibility",
#     "Accreditation/Approval",
#     "Course Fee (INR)",
#     "Course Duration",
#     "Career Outcomes",
#     "Placement Support",
#     "Average Salary After Graduation (INR)"
# ]
# # ------------------------

# def load_data(file_path):
#     ext = os.path.splitext(file_path)[1].lower()
#     if ext in [".xlsx", ".xls"]:
#         return pd.read_excel(file_path, sheet_name="Sheet1")
#     else:
#         return pd.read_csv(file_path)

# def prepare_text(row, columns):
#     """Combine selected columns into one text block"""
#     parts = []
#     for col in columns:
#         value = str(row[col]) if pd.notna(row[col]) else ""
#         if value and value != "nan":
#             parts.append(f"{col}: {value}")
#     return " | ".join(parts)

# def main():
#     # Load data
#     print("📂 Loading data...")
#     df = load_data(INPUT_FILE)

#     # Normalize column names: strip spaces, unify spacing
#     df.columns = [c.strip() for c in df.columns]

#     # Warn about missing columns
#     missing = [c for c in COLUMNS_TO_USE if c not in df.columns]
#     if missing:
#         print(f"⚠️ Missing columns will be skipped: {missing}")

#     # Prepare combined text
#     print("📝 Preparing text data...")
#     # Use only present columns to avoid KeyError
#     present_cols = [c for c in COLUMNS_TO_USE if c in df.columns]
#     df["combined_text"] = df.apply(lambda row: prepare_text(row, present_cols), axis=1)

#     # Load embedding model
#     print(f"🔎 Loading model: {MODEL_NAME}...")
#     model = SentenceTransformer(MODEL_NAME)

#     # Generate embeddings
#     print("⚡ Generating embeddings...")
#     embeddings = model.encode(df["combined_text"].tolist(), show_progress_bar=True)

#     # Save results
#     print("💾 Saving embeddings...")
#     df["embedding"] = embeddings.tolist()
#     df.to_csv(OUTPUT_FILE, index=False)

#     print(f"✅ Done! Saved to {OUTPUT_FILE}")

# if __name__ == "__main__":
#     os.makedirs("../outputs", exist_ok=True)
#     main()

# scripts/prepare_embeddings.py
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# CONFIG
INPUT_FILE = "../data/final.csv"
OUTPUT_METADATA = "../outputs/metadata.parquet"
OUTPUT_EMB_NPY = "../outputs/embeddings.npy"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLUMNS_TO_USE = [
    "University Name",
    "Course Name",
    "Specializations Available",
    "Mode of Study",
    "Eligibility",
    "Accreditation/Approval",
    "Course Fee (INR)",
    "Course Duration",
    "Career Outcomes",
    "Placement Support",
    "Average Salary After Graduation (INR)"
]

def load_data(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(file_path, sheet_name="Sheet1")
    else:
        return pd.read_csv(file_path)

def prepare_text(row, columns):
    parts = []
    for col in columns:
        if col in row and pd.notna(row[col]):
            parts.append(f"{col}: {row[col]}")
    return " | ".join(parts)

def main():
    os.makedirs("../outputs", exist_ok=True)
    print("Loading data...")
    df = load_data(INPUT_FILE)
    df.columns = [c.strip() for c in df.columns]

    present_cols = [c for c in COLUMNS_TO_USE if c in df.columns]
    print("Using columns:", present_cols)

    # create a stable integer id (FAISS expects int64 ids)
    # If you have an existing stable PK, use that column instead.
    df = df.reset_index(drop=True)
    df["id"] = df.index.astype("int64")

    print("Preparing combined text...")
    df["combined_text"] = df.apply(lambda r: prepare_text(r, present_cols), axis=1)

    print(f"Loading embedding model {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    texts = df["combined_text"].fillna("").tolist()
    print(f"Embedding {len(texts)} documents ...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Save embeddings as binary .npy and metadata as parquet (no embedding strings)
    print(f"Saving embeddings to {OUTPUT_EMB_NPY} (numpy .npy)...")
    np.save(OUTPUT_EMB_NPY, embeddings)  # shape (N, D)

    print(f"Saving metadata to {OUTPUT_METADATA} (parquet)...")
    # Keep key metadata and id -> this is the sidecar file for filters
    meta_cols = ["id", "combined_text"] + [c for c in present_cols if c not in ("combined_text",)]
    meta_df = df[["id", "combined_text"] + present_cols].copy()
    # Optionally, add parsed numeric fees column here (or keep for later)
    meta_df.to_parquet(OUTPUT_METADATA, index=False)

    print("Done. Artifacts:")
    print(" -", OUTPUT_EMB_NPY)
    print(" -", OUTPUT_METADATA)

if __name__ == "__main__":
    main()

