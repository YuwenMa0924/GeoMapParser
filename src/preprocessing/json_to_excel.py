import json
import os
import pandas as pd

# ========== Path configuration ==========
JSON_DIR = "path/to/extraction_results"
OUTPUT_EXCEL = "path/to/maps_table_e1.xlsx"

# ========== Fields to export ==========
FIELDS = [
    "title",
    "scale",
    "scale_position",
    "bar_pixel_length",
    "projection",
    "publication_agency",
    "publication_date",
    "data_source"
]

def normalize_value(v):
    """
    Normalize the field format for convenient Excel-based analysis
    """
    if v is None:
        return ""
    if isinstance(v, list):
        return "; ".join([str(x) for x in v])
    return str(v).strip()

records = []

for fname in os.listdir(JSON_DIR):
    if not fname.endswith(".json"):
        continue

    fpath = os.path.join(JSON_DIR, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)

    row = {"file": fname}

    for field in FIELDS:
        row[field] = normalize_value(data.get(field))

    records.append(row)

df = pd.DataFrame(records)

# Write the output Excel file
df.to_excel(OUTPUT_EXCEL, index=False)

print(f"✔ Metadata table saved to: {OUTPUT_EXCEL}")
