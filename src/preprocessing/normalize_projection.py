import pandas as pd
from rapidfuzz import process, fuzz

INPUT_EXCEL = "path/to/maps_table_e4.xlsx"
OUTPUT_EXCEL = "path/to/maps_table_e5.xlsx"
COLUMN = "projection"

# Preserve the original canonical capitalization (camel-case style)
CANONICAL_PROJECTIONS = [
    "Mercator Projection",
    "Lambert Conformal Conic Projection",
    "Equidistant Cylindrical Projection",
    "Transverse Mercator Projection",
    "Polar Stereographic Projection",
    "Orthographic Projection",
    "Lambert Projection",
    "Mercator Projection and Polar Stereographic Projection",
    "Robinson Pseudo Cylindrical Projection",
    "Modified Stereographic Projection"
]

# Build a lowercase version of the canonical list for matching (the original list is left unchanged)
CANONICAL_LOWER = [p.lower() for p in CANONICAL_PROJECTIONS]

SIM_THRESHOLD = 30  # Lower the threshold to accommodate variants

df = pd.read_excel(INPUT_EXCEL)

def normalize_projection(text):
    if pd.isna(text):
        return None

    text = str(text).strip()
    if text == "":
        return text

    # Convert the input text to lowercase for matching
    text_lower = text.lower()

    # Perform fuzzy matching against the lowercase canonical list
    result = process.extractOne(
        text_lower,
        CANONICAL_LOWER,
        scorer=fuzz.token_sort_ratio
    )

    if result is None:
        return text

    match_lower, score, _ = result
    if score >= SIM_THRESHOLD:
        # Locate the matched lowercase string in the original canonical list
        # Note: duplicates are possible in principle; we assume none and retrieve the original capitalization directly by index
        idx = CANONICAL_LOWER.index(match_lower)
        return CANONICAL_PROJECTIONS[idx]
    else:
        return text

df["projection_norm"] = df[COLUMN].apply(normalize_projection)

df.to_excel(OUTPUT_EXCEL, index=False)
print(f"✔ projection normalized → {OUTPUT_EXCEL}")