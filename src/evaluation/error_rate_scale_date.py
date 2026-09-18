import pandas as pd
import re

# =========================
# 1. Read Excel
# =========================
file_path = "path/to/maps_table_e8.xlsx"
df = pd.read_excel(file_path)

N = len(df)


# =========================
# 2. Rule functions
# =========================

# publication_date: must be a 4-digit year
def invalid_date(x):
    if pd.isna(x):
        return True
    x = str(x).strip()
    return not re.fullmatch(r"\d{4}", x)


# scale: must contain 1:xxx
def invalid_scale(x):
    if pd.isna(x):
        return True
    x = str(x).replace(" ", "")
    return "1:" not in x


# =========================
# 3. Statistics
# =========================

df["date_invalid"] = df["publication_date"].apply(invalid_date)
df["scale_invalid"] = df["scale"].apply(invalid_scale)

date_invalid_count = df["date_invalid"].sum()
scale_invalid_count = df["scale_invalid"].sum()

result = {
    "total_samples": N,

    "publication_date_invalid_count": int(date_invalid_count),
    "publication_date_invalid_ratio": float(date_invalid_count / N),

    "scale_invalid_count": int(scale_invalid_count),
    "scale_invalid_ratio": float(scale_invalid_count / N),
}

print("\n===== Field Quality Result =====")
for k, v in result.items():
    print(f"{k}: {v}")


# =========================
# 4. Save results
# =========================
pd.DataFrame([result]).to_csv("field_quality_result.csv", index=False)