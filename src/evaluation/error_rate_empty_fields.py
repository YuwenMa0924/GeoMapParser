import pandas as pd

# =========================
# 1. Read Excel
# =========================
file_path = "path/to/maps_table_e8.xlsx"
df = pd.read_excel(file_path)

N = len(df)

fields = [
    "title",
    "publication_agency",   # named agency in the source table (referred to as publication_agency here)
    "data_source",
    "projection",
    "scale",
    "publication_date"
]


# =========================
# 2. Empty-field test function
# =========================
def is_empty(x):
    if pd.isna(x):
        return True
    x = str(x).strip()
    return x == "" or x.lower() == "nan"


# =========================
# 3. Statistics
# =========================
results = []

for f in fields:
    empty_count = df[f].apply(is_empty).sum()
    empty_ratio = empty_count / N

    results.append({
        "field": f,
        "empty_count": int(empty_count),
        "empty_ratio": round(empty_ratio, 4),
        "empty_percentage": f"{empty_ratio * 100:.2f}%"
    })


result_df = pd.DataFrame(results)

# =========================
# 4. Output
# =========================
print("\n===== Empty Field Statistics =====")
print(result_df)

# Save
result_df.to_csv("empty_field_stats.csv", index=False, encoding="utf-8")