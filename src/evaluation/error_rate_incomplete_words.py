import pandas as pd
from wordfreq import zipf_frequency

# ========== Configuration ==========
INPUT_EXCEL = "path/to/maps_table_e8.xlsx"

FIELDS = [
    "title",
    "publication_agency",
    "data_source"
]

# ========== Read data ==========
df = pd.read_excel(INPUT_EXCEL)

# ========== Utility functions ==========
def normalize_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def is_incomplete_words(text):
    """
    Determine whether incomplete words are present
    """
    if text.strip() == "":
        return False

    words = text.split()

    bad_words = 0
    total_words = 0

    for w in words:
        if len(w) < 3:
            continue

        total_words += 1

        # Fully capitalized tokens are usually legitimate abbreviations (not flagged as errors)
        if w.isupper():
            continue

        # Language-frequency test (a frequently observed OCR-error signature)
        if zipf_frequency(w.lower(), "en") < 1.0:
            bad_words += 1

    if total_words == 0:
        return False

    return bad_words >= 1


# ========== Per-field statistics ==========
print("\n===== Incomplete Words Statistics =====\n")

total_all = 0
error_all = 0

for field in FIELDS:
    if field not in df.columns:
        print(f"[WARN] missing field: {field}")
        continue

    sub = df[[field]].copy()
    sub[field] = sub[field].apply(normalize_text)

    sub["is_incomplete"] = sub[field].apply(is_incomplete_words)

    total = len(sub)
    error = sub["is_incomplete"].sum()
    ratio = error / total if total > 0 else 0

    total_all += total
    error_all += error

    print(f"{field}: {error}/{total} = {ratio:.4f}")

# ========== overall ==========
overall_ratio = error_all / total_all if total_all > 0 else 0

print("\n===== OVERALL =====")
print(f"total: {total_all}")
print(f"incomplete_words: {error_all}")
print(f"ratio: {overall_ratio:.4f}")