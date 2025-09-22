import pandas as pd
import re
import os

INPUT_FILE = "../data/dataset.csv"
OUTPUT_FILE = "../outputs/cleaned_dataset.csv"

# ----------------------------
# Functions
# ----------------------------
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)  # collapse multiple spaces
    return text

def normalize_course_name(name):
    name = clean_text(name)
    name_map = {
        "master of business administration": "mba",
        "mba": "mba",
        "bachelor of business administration": "bba",
        "bba": "bba",
        "master of computer applications": "mca",
        "mca": "mca",
        "bachelor of computer applications": "bca",
        "bca": "bca",
    }
    for k, v in name_map.items():
        if k in name:
            return v
    return name  # return as is if no match

def parse_fee(fee_str):
    if pd.isna(fee_str):
        return None
    s = str(fee_str).lower().replace(",", "").replace("inr", "").strip()
    # handle range
    if "-" in s or "–" in s:
        s = s.replace("–", "-")
        s = s.split("-")[0]
    # handle lakh/l
    if "lakh" in s or "l" in s:
        nums = re.findall(r"\d+\.?\d*", s)
        return int(float(nums[0]) * 100000) if nums else None
    nums = re.findall(r"\d+\.?\d*", s)
    return int(nums[0]) if nums else None

def clean_specializations(spec):
    if pd.isna(spec):
        return ""
    spec = clean_text(spec)
    return spec

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv(INPUT_FILE)

# Normalize
df["Course Name_norm"] = df["Course Name"].apply(normalize_course_name)
df["Specializations_norm"] = df["Specializations Available"].apply(clean_specializations)
df["Course Fee_numeric"] = df["Course Fee (INR)"].apply(parse_fee)

# Optional: tag degrees
degrees = ["mba", "bba", "mca", "bca"]
for deg in degrees:
    df[deg] = df["Course Name_norm"].apply(lambda x: deg in x)

# Trim strings
for col in ["University Name", "Location", "Country", "Mode of Study"]:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: clean_text(x))

# Save cleaned dataset
os.makedirs("outputs", exist_ok=True)
df.to_csv(os.path.join("outputs", "cleaned_dataset.csv"), index=False)
print("✅ Cleaned dataset saved to outputs/cleaned_dataset.csv")
