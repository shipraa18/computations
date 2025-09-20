
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# CONFIG
INPUT_FILE = "../data/dataset.csv"
OUTPUT_METADATA = "../outputs/metadata.parquet"
OUTPUT_EMB_NPY = "../outputs/embeddings.npy"
MODEL_NAME = "intfloat/e5-base"
COLUMNS_TO_USE = [
    "University Name",
    "University Type",
    "Course Name",
    "Course Level",
    "Mode of Study",
    "Course Duration",
    "Course Fee (INR)",
    "Eligibility(indian)",
    "Eligibility (foreign)",
    "Admission Process",
    "Application Deadline",
    "Accreditation/Approval",
    "Location",
    "Country",
    "Specializations Available",
    "Placement Support",
    "Scholarship Availability",
    "Medium of Instruction",
    "University Ranking (NIRF/Global)",
    "Career Outcomes",
    "Average Salary After Graduation (INR)",
    "Program Highlights",
    "Student Support Services",
    "Global Recognition",
    "Curriculum"
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
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    texts = df["combined_text"].fillna("").tolist()
    print(f"Embedding {len(texts)} documents ...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Save embeddings as binary .npy and metadata as parquet (no embedding strings)
    print(f"Saving embeddings to {OUTPUT_EMB_NPY} (numpy .npy)...")
    np.save(OUTPUT_EMB_NPY, embeddings)  # shape (N, D)

    print(f"Saving metadata to {OUTPUT_METADATA} (parquet)...")
    # Keep key metadata and id -> this is the sidecar file for filters
    meta_cols = (
        ["id", "combined_text"]
        + COLUMNS_TO_USE
        + ["Website URL", "Contact Email", "registration fees", "EMI available", "examination mode"]
    )

    meta_df = df[meta_cols].copy()
    meta_df.to_parquet(OUTPUT_METADATA, index=False)

    print("Done. Artifacts:")
    print(" -", OUTPUT_EMB_NPY)
    print(" -", OUTPUT_METADATA)

if __name__ == "__main__":
    main()

