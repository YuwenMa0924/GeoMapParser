import pandas as pd
import re

# ================================
# Paths
# ================================

INPUT_EXCEL = "path/to/maps_table_e7.xlsx"
OUTPUT_EXCEL = "path/to/maps_table_e8.xlsx"

# ================================
# Read the data
# ================================

df = pd.read_excel(INPUT_EXCEL)

# ================================
# SCALE cleaning function (enhanced version)
# ================================

def parse_scale(text):
    """
    Parse a scale value into a unified form:

    Supported:
    1:1,000,000
    1:1000000
    1:1 000 000
    1:1 004 000
    1 : 1,000,000
    scale 1:1,000,000
    """

    if pd.isna(text):
        return None

    text = str(text)

    # Capture the numeric part of the scale (spaces and commas allowed)
    match = re.search(r"1\s*[:：]\s*([\d,\s]+)", text)

    if match:

        num = match.group(1)

        # Remove all non-digit characters
        num = re.sub(r"\D", "", num)

        try:
            return int(num)
        except:
            return None

    return None


# ================================
# Parse the scale
# ================================

df["scale_value"] = df["scale"].apply(parse_scale)

# ================================
# Scale classification
# ================================

def classify_scale(v):

    if pd.isna(v):
        return None

    if v <= 250000:
        return "Large-scale"

    elif v <= 1000000:
        return "Medium-scale"

    elif v <= 5000000:
        return "Small-scale"

    else:
        return "Very small-scale"


df["scale_class"] = df["scale_value"].apply(classify_scale)

# ================================
# scale display field
# ================================

def format_scale(v):

    if pd.isna(v):
        return None

    return f"1:{int(v):,}"


df["scale_label"] = df["scale_value"].apply(format_scale)
# ================================
# Output
# ================================

df.to_excel(OUTPUT_EXCEL, index=False)

print("✔ Scale normalization completed")
print("✔ scale_value column created")
print("✔ scale_class column created")
print("✔ Output saved to:", OUTPUT_EXCEL)