import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import re

# ================= font =================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 14,
    "legend.fontsize": 10,
    "axes.unicode_minus": False,
    "figure.dpi": 100,
    "savefig.dpi": 300
})

# ================= configuration =================
JSON_DIR = "path/to/map_extraction_results"
STOPWORDS = {
    "of", "and", "the", "with", "unit", "units", "area", "areas",
    "showing", "show", "region", "regions"
}

# ================= tool=================
def tokenize(text):
    if not isinstance(text, str):
        return []
    words = re.findall(r"[A-Za-z]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]

# ================= read data =================
all_tokens = []
symbol_semantic = []
legend_complexity = []

for fname in os.listdir(JSON_DIR):
    if not fname.endswith(".json"):
        continue
    with open(os.path.join(JSON_DIR, fname), "r", encoding="utf-8") as f:
        data = json.load(f)
    legend = data.get("legend_data")
    if not isinstance(legend, list):
        continue
    legend_complexity.append({"file": fname, "legend_length": len(legend)})
    for item in legend:
        value = item.get("value", "")
        key_type = item.get("key_type", "unknown")
        all_tokens.extend(tokenize(value))
        symbol_semantic.append({"value": value, "key_type": key_type})


token_freq = Counter(all_tokens)
top_tokens = token_freq.most_common(20)

df_tokens = pd.DataFrame(top_tokens, columns=["term", "frequency"])


fig, ax = plt.subplots(figsize=(16, 10))


sns.barplot(
    data=df_tokens,
    x="frequency",
    y="term",
    color="#1f77b4",
    ax=ax,
    height=0.8,
    edgecolor="none",
    order=df_tokens.sort_values(by="frequency", ascending=False)["term"]
)


ax.yaxis.set_tick_params(pad=8)
ax.set_ylim(-0.5, 29.5)


ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.title("High-frequency Geological Terms in Legend Data", pad=15)
plt.xlabel("Frequency", labelpad=12)
plt.ylabel("Geological Term", labelpad=12)


ax.set_xticks(range(0, 1600, 200))

plt.tight_layout()
plt.savefig("figure12a_legend_terms.png", dpi=300, bbox_inches="tight")
plt.show()
# =====================================================
# 2️⃣ Symbol–Semantic Consistency Analysis
# =====================================================
# =====================================================
# =====================================================
# 2️⃣ Symbol–Semantic Consistency Analysis
# =====================================================

df_symbol = pd.DataFrame(symbol_semantic)


top_values = df_symbol["value"].value_counts().head(10).index
df_symbol = df_symbol[df_symbol["value"].isin(top_values)]


top_keys = df_symbol["key_type"].value_counts().head(5).index
df_symbol = df_symbol[df_symbol["key_type"].isin(top_keys)]

symbol_matrix = df_symbol.groupby(["value","key_type"]).size().unstack(fill_value=0)

symbol_matrix_norm = symbol_matrix.div(symbol_matrix.sum(axis=1), axis=0)

ssc = symbol_matrix.max(axis=1) / symbol_matrix.sum(axis=1)
ssc = ssc.sort_values()

symbol_distribution = df_symbol["key_type"].value_counts()

# ================= Figure =================

fig = plt.figure(figsize=(14,6), dpi=300)

gs = fig.add_gridspec(
    1,3,
    width_ratios=[2,1,1],
    wspace=0.8
)

# --------------------------------
# (a) Heatmap
# --------------------------------

ax1 = fig.add_subplot(gs[0])

hm = sns.heatmap(
    symbol_matrix_norm,
    cmap="YlGnBu",
    annot=True,
    fmt=".2f",
    mask=symbol_matrix_norm < 0.02,
    linewidths=0.4,
    annot_kws={"size":4},
    cbar_kws={
        "label":"Probability",
        "shrink":0.7      # shorten the colorbar
    },
    ax=ax1
)

ax1.set_title("(a) Symbol–Semantic Mapping", fontsize=4)

ax1.set_xlabel("Symbol Type", fontsize=4)
ax1.set_ylabel("Geological Unit", fontsize=4)

ax1.tick_params(axis='x', labelsize=4, rotation=30)
ax1.tick_params(axis='y', labelsize=4)

# colorbar
cbar = hm.collections[0].colorbar
cbar.ax.tick_params(labelsize=4)
cbar.set_label("Probability", fontsize=4)

# --------------------------------
# (b) SSC
# --------------------------------

ax2 = fig.add_subplot(gs[1])

sns.barplot(
    x=ssc.values,
    y=ssc.index,
    color="#4C72B0",
    ax=ax2
)

ax2.set_xlim(0,1)

ax2.set_title("(b) Consistency Score", fontsize=4)

ax2.set_xlabel("SSC", fontsize=4)
ax2.set_ylabel("")

ax2.tick_params(axis='x', labelsize=4)
ax2.tick_params(axis='y', labelsize=4)

ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# --------------------------------
# (c) Symbol distribution
# --------------------------------

ax3 = fig.add_subplot(gs[2])

sns.barplot(
    x=symbol_distribution.values,
    y=symbol_distribution.index,
    color="#55A868",
    ax=ax3
)

ax3.set_title("(c) Symbol Usage", fontsize=4)

ax3.set_xlabel("Frequency", fontsize=4)
ax3.set_ylabel("")

ax3.tick_params(axis='x', labelsize=4)
ax3.tick_params(axis='y', labelsize=4)

ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

# --------------------------------
# Main title
# --------------------------------

fig.suptitle(
    "Symbol–Semantic Consistency of Geological Map Legends",
    fontsize=10,
    y=1.02
)

plt.tight_layout(pad=2.0)
plt.savefig("figure12b_symbol_semantic_consistency.png", dpi=300, bbox_inches="tight")
plt.show()
# =====================================================
# 3️⃣ Geological-information complexity analysis (keeping the previous optimization)
# =====================================================
df_complexity = pd.DataFrame(legend_complexity)

fig, ax = plt.subplots(figsize=(11, 5))
sns.histplot(
    df_complexity["legend_length"],
    bins=25,
    color="#1f77b4",
    kde=False,
    ax=ax,
    edgecolor="black",
    linewidth=0.5
)

ax.set_xlim(0, 210)
ax.set_ylim(0, 650)
ax.set_xticks(range(0, 220, 40))
ax.set_yticks(range(0, 700, 200))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_xlabel("Number of Legend Items", labelpad=8)
ax.set_ylabel("Number of Maps", labelpad=8)
ax.set_title("Distribution of Geological Information Density", pad=10)
plt.tight_layout()
plt.savefig("figure12c_information_density.png", dpi=300, bbox_inches="tight")
plt.show()

print("✔ The first figure replicates the example style 1:1")
