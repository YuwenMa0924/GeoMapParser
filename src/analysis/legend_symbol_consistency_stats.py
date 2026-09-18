import os
import json
import pandas as pd
import numpy as np
from collections import Counter
import re

# ================= Configuration =================
JSON_DIR = "path/to/extraction_results"

STOPWORDS = {
    "of","and","the","with","unit","units","area","areas",
    "showing","show","region","regions"
}

# ================= Utility functions =================
def tokenize(text):
    if not isinstance(text,str):
        return []
    words = re.findall(r"[A-Za-z]+",text.lower())
    return [w for w in words if w not in STOPWORDS and len(w)>2]

def entropy(probs):
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

# ================= Data loading =================
symbol_semantic = []
legend_complexity = []

for fname in os.listdir(JSON_DIR):

    if not fname.lower().endswith(".json"):
        continue

    with open(os.path.join(JSON_DIR,fname),"r",encoding="utf-8") as f:
        data = json.load(f)

    legend = data.get("legend_data")

    if not isinstance(legend,list):
        continue

    legend_complexity.append(len(legend))

    for item in legend:
        value = item.get("value","")
        key_type = item.get("key_type","unknown")

        symbol_semantic.append({
            "value": value,
            "key_type": key_type
        })

df_symbol = pd.DataFrame(symbol_semantic)

# ================= (5) Symbol consistency quantification =================

symbol_matrix = (
    df_symbol.groupby(["value","key_type"])
    .size()
    .unstack(fill_value=0)
)

# Normalization
symbol_matrix_norm = symbol_matrix.div(symbol_matrix.sum(axis=1), axis=0)

results = []

for value, row in symbol_matrix_norm.iterrows():
    probs = row.values
    max_prob = probs.max()
    ent = entropy(probs)

    if max_prob >= 0.8:
        consistency = "strong"
    elif max_prob >= 0.5:
        consistency = "medium"
    else:
        consistency = "weak"

    results.append({
        "value": value,
        "max_prob": max_prob,
        "entropy": ent,
        "consistency": consistency
    })

df_consistency = pd.DataFrame(results)

# Compute shares
consistency_ratio = df_consistency["consistency"].value_counts(normalize=True) * 100

print("\n========== Symbol consistency statistics ==========")
print(consistency_ratio)

# Identify weakly consistent units (directly quotable in the paper)
weak_units = df_consistency[df_consistency["consistency"]=="weak"]
print("\nExamples of weakly consistent units:")
print(weak_units.sort_values("entropy",ascending=False).head(10))


# ================= (6)(7) Information density quantification =================

legend_lengths = np.array(legend_complexity)

total_maps = len(legend_lengths)

low = np.sum((legend_lengths >= 0) & (legend_lengths <= 10))
mid = np.sum((legend_lengths > 10) & (legend_lengths <= 50))
high = np.sum(legend_lengths > 50)
extreme = np.sum(legend_lengths > 100)

print("\n========== Information density statistics ==========")
print(f"0-10 share: {low/total_maps*100:.2f}%")
print(f"10-50 share: {mid/total_maps*100:.2f}%")
print(f">50 share: {high/total_maps*100:.2f}%")
print(f">100 share: {extreme/total_maps*100:.2f}%")

# ================= Export =================
df_consistency.to_excel("symbol_consistency.xlsx", index=False)

print("\n✅ Exported: symbol_consistency.xlsx")