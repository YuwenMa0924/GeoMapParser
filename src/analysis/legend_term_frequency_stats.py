import os
import json
import pandas as pd
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
    words = re.findall(r"[A-Za-z]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]

# ================= Data loading =================
all_tokens = []

for fname in os.listdir(JSON_DIR):

    if not fname.lower().endswith(".json"):
        continue

    with open(os.path.join(JSON_DIR, fname), "r", encoding="utf-8") as f:
        data = json.load(f)

    legend = data.get("legend_data")

    if not isinstance(legend, list):
        continue

    for item in legend:
        value = item.get("value", "")
        all_tokens.extend(tokenize(value))

# ================= Core statistics =================

# Total token count
total_tokens = len(all_tokens)

# Term frequency counts
token_freq = Counter(all_tokens)

# Convert to DataFrame (full list)
df_all = pd.DataFrame(token_freq.items(), columns=["term", "frequency"])
df_all = df_all.sort_values("frequency", ascending=False)

# Compute share
df_all["percentage"] = df_all["frequency"] / total_tokens * 100

# ================= Top-k share =================
def top_k_ratio(df, k):
    return df.head(k)["frequency"].sum() / total_tokens * 100

top5_ratio = top_k_ratio(df_all, 5)
top10_ratio = top_k_ratio(df_all, 10)
top20_ratio = top_k_ratio(df_all, 20)

# ================= Output =================
print("\n========== TOKEN STATS ==========")
print(f"Total tokens: {total_tokens}")

print("\nTop-20 terms (exact values):")
print(df_all.head(20))

print("\n========== Share statistics ==========")
print(f"Top-5 share: {top5_ratio:.2f}%")
print(f"Top-10 share: {top10_ratio:.2f}%")
print(f"Top-20 share: {top20_ratio:.2f}%")

# ================= Export =================
df_all.to_excel("term_frequency_full.xlsx", index=False)

print("\n✅ Exported: term_frequency_full.xlsx")