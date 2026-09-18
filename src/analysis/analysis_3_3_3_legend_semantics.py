import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import re
import numpy as np

# ==================== Colorblind-friendly palette ====================
COLORBLIND_SAFE = [
    "#332288","#88CCEE","#44AA99","#117733",
    "#999933","#DDCC77","#CC6677","#882255",
    "#AA4499","#DDDDDD","#6699CC","#FFB000"
]

def get_cb_colors(n):
    if n <= len(COLORBLIND_SAFE):
        return COLORBLIND_SAFE[:n]
    else:
        import matplotlib.cm as cm
        cmap = cm.get_cmap('tab20')
        return [cmap(i) for i in np.linspace(0, 1, n)]

# ==================== Global style ====================
TEXT_DARK  = '#2E2E2E'
AXIS_GRAY  = '#4A4A4A'
GRID_GRAY  = '#D9D9D9'

# ================= Original global style (kept) =================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9
})

# ================= Added new style (without overriding the original fonts) =================
plt.rcParams.update({
    "axes.linewidth": 0.8,
    "axes.edgecolor": AXIS_GRAY,
    "xtick.color": AXIS_GRAY,
    "ytick.color": AXIS_GRAY,
    "text.color": TEXT_DARK
})

sns.set_style("white")
plt.rcParams["font.family"] = "Times New Roman"

# ================= Configuration =================
JSON_DIR = "path/to/yolov8n_extraction_results/all_results"

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

# ================= Data reading =================
all_tokens = []
symbol_semantic = []
legend_complexity = []

for fname in os.listdir(JSON_DIR):
    if not fname.endswith(".json"):
        continue
    with open(os.path.join(JSON_DIR,fname),"r",encoding="utf-8") as f:
        data = json.load(f)
    legend = data.get("legend_data")
    if not isinstance(legend,list):
        continue
    legend_complexity.append({
        "file":fname,
        "legend_length":len(legend)
    })
    for item in legend:
        value = item.get("value","")
        key_type = item.get("key_type","unknown")
        all_tokens.extend(tokenize(value))
        symbol_semantic.append({
            "value":value,
            "key_type":key_type
        })

# ================= Data preparation =================
# High-frequency terms
token_freq = Counter(all_tokens)
top_tokens = token_freq.most_common(20)
df_tokens = pd.DataFrame(top_tokens,columns=["term","frequency"])

# symbol semantic
df_symbol = pd.DataFrame(symbol_semantic)
top_values = df_symbol["value"].value_counts().head(8).index
top_keys = df_symbol["key_type"].value_counts().head(5).index
df_symbol = df_symbol[
    (df_symbol["value"].isin(top_values)) &
    (df_symbol["key_type"].isin(top_keys))
]
symbol_matrix = (
    df_symbol.groupby(["value","key_type"])
    .size()
    .unstack(fill_value=0)
)
symbol_matrix_norm = symbol_matrix.div(symbol_matrix.sum(axis=1),axis=0)

# complexity
df_complexity = pd.DataFrame(legend_complexity)

# ================= Separate figure (a) =================
plt.figure(figsize=(6,5))
sns.barplot(
    data=df_tokens,
    x="frequency",
    y="term",
    color=get_cb_colors(1)[0]
)
plt.title("(a)High-frequency Geological Terms")
plt.xlabel("Frequency")
plt.ylabel("Term")
sns.despine(top=True, right=True)
plt.tight_layout()
plt.show()

# ================= Separate figure (b) =================
plt.figure(figsize=(7,5))
sns.heatmap(
    symbol_matrix_norm,
    cmap=sns.light_palette("#44AA99", as_cmap=True),  # Changed: light gradient
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={"label":"Probability"}
)
plt.title("(b)Symbol–Semantic Mapping")
plt.xlabel("Symbol Type")
plt.ylabel("Geological Unit")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# ================= Separate figure (c) =================
plt.figure(figsize=(6,4))
sns.histplot(
    df_complexity["legend_length"],
    bins=25,
    color=get_cb_colors(1)[0],
    kde=False
)
plt.title("(c)Geological Information Density")
plt.xlabel("Number of Legend Items")
plt.ylabel("Number of Maps")
sns.despine(top=True, right=True)
plt.tight_layout()
plt.show()

# ================= Combined figure (three subplots side by side) =================
fig = plt.figure(figsize=(20,6))
gs = fig.add_gridspec(1,3, width_ratios=[1.2,1.3,1])

# (a) High-frequency terms
ax1 = fig.add_subplot(gs[0])
sns.barplot(
    data=df_tokens,
    x="frequency",
    y="term",
    color=get_cb_colors(1)[0],
    ax=ax1
)
ax1.set_title("(a)High-frequency Geological Terms")
ax1.set_xlabel("Frequency")
ax1.set_ylabel("Term")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# (b) Symbol-Semantic heatmap
ax2 = fig.add_subplot(gs[1])
sns.heatmap(
    symbol_matrix_norm,
    cmap=sns.light_palette("#44AA99", as_cmap=True),  # Changed: light gradient
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={"label":"Probability"},
    ax=ax2
)
ax2.set_title("(b)Symbol–Semantic Mapping")
ax2.set_xlabel("Symbol Type")
ax2.set_ylabel("Geological Unit")
ax2.tick_params(axis='x', rotation=30)

# (c) Complexity distribution
ax3 = fig.add_subplot(gs[2])
sns.histplot(
    df_complexity["legend_length"],
    bins=25,
    color=get_cb_colors(1)[0],
    kde=False,
    ax=ax3
)
ax3.set_title("(c)Geological Information Density")
ax3.set_xlabel("Number of Legend Items")
ax3.set_ylabel("Number of Maps")
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()