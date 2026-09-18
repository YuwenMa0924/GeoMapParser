import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ================= Path configuration =================
INPUT_EXCEL = "path/to/maps_table_e6.xlsx"

# ================= Read data =================
df = pd.read_excel(INPUT_EXCEL)

# ================= Basic cleaning =================
df["publication_date"] = pd.to_numeric(df["publication_date"], errors="coerce")

df = df[(df["publication_date"] >= 1960) & (df["publication_date"] <= 2021)]

def parse_scale(x):
    if isinstance(x, str):
        m = re.search(r"1\s*[:：]\s*([\d,]+)", x)
        if m:
            return int(m.group(1).replace(",", ""))
    return None

df["scale_value"] = df["scale"].apply(parse_scale)

# =====================================================
# 2️⃣ Journal-quality projection evolution (final layout)
# =====================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# ---------- User-specified palette and style ----------
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

TEXT_DARK  = '#2E2E2E'
AXIS_GRAY  = '#4A4A4A'
GRID_GRAY  = '#D9D9D9'

# ---------- Merge the existing font settings with the user style ----------
# plt.rcParams.update({
#     "font.family": "Times New Roman",
#     "axes.labelsize": 10,
#     "axes.titlesize": 11,
#     "xtick.labelsize": 9,
#     "ytick.labelsize": 9
# })
# plt.rcParams.update({
#     "font.size": 12,
#     "axes.linewidth": 0.8,
#     "axes.edgecolor": AXIS_GRAY,
#     "xtick.color": AXIS_GRAY,
#     "ytick.color": AXIS_GRAY,
#     "text.color": TEXT_DARK
# })
# sns.set_style("white")
sns.set_style("white")
sns.set(font='Times New Roman')   # Key: set the font via seaborn

plt.rcParams.update({
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.size": 12,
    "axes.linewidth": 0.8,
    "axes.edgecolor": AXIS_GRAY,
    "xtick.color": AXIS_GRAY,
    "ytick.color": AXIS_GRAY,
    "text.color": TEXT_DARK
})

df_proj = df.dropna(subset=["publication_date", "projection_norm", "target_body"]).copy()
df_proj["decade"] = (df_proj["publication_date"] // 10) * 10

# projection types
all_projections = sorted(df_proj["projection_norm"].unique())

# unified decades
all_decades = list(range(1960, 2021, 10))

# ---------- Sort by number of maps ----------
body_counts = df_proj["target_body"].value_counts()
bodies_sorted = body_counts.index.tolist()

# ---------- Use the colorblind-friendly palette ----------
palette = get_cb_colors(len(all_projections))

# ---------- 3×3 layout ----------
fig, axes = plt.subplots(
    3, 3,
    figsize=(16, 11),
    dpi=150,
    sharex=True,
    sharey=True
)

axes = axes.flatten()

subplot_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)"]

# ---------- Plotting ----------
for i, body in enumerate(bodies_sorted):

    ax = axes[i]

    sub = df_proj[df_proj["target_body"] == body]

    proj_counts = (
        sub.groupby(["decade", "projection_norm"])
        .size()
        .unstack()
    )

    proj_counts = proj_counts.reindex(columns=all_projections, fill_value=0)
    proj_counts = proj_counts.reindex(all_decades, fill_value=0)

    proj_ratio = proj_counts.div(proj_counts.sum(axis=1), axis=0).fillna(0)

    bottom = np.zeros(len(proj_ratio))

    for j, proj in enumerate(all_projections):

        ax.bar(
            proj_ratio.index,
            proj_ratio[proj],
            bottom=bottom,
            width=7,
            color=palette[j],
            edgecolor="white",
            linewidth=0.3
        )

        bottom += proj_ratio[proj].values

    # Subplot title (remove the n)
    ax.set_title(f"{subplot_labels[i]} {body}", loc="left")

    ax.set_ylim(0, 1)

    ax.set_xticks(all_decades)
    ax.set_xticklabels(all_decades, rotation=45)

    ax.set_ylabel("Proportion")

    ax.grid(axis="y", linestyle="--", alpha=0.4)

# ---------- Put the legend in the empty subplot ----------
legend_ax = axes[7]   # the 8th subplot
legend_ax.axis("off")

legend_handles = [
    Patch(facecolor=palette[i], edgecolor="none", label=all_projections[i])
    for i in range(len(all_projections))
]

legend_ax.legend(
    handles=legend_handles,
    loc="center left",
    frameon=True,
    edgecolor="black",
    title="Projection Type",
    fontsize=8,
    ncol=2
)

# Hide the last empty subplot
axes[8].axis("off")

# ---------- Title ----------
fig.suptitle(
    "Evolution of Projection Types in Planetary Geological Mapping (1960–2021)",
    fontsize=14,
    y=0.97
)

plt.tight_layout(rect=[0, 0.02, 1, 0.95])

plt.show()

# ---------- Export ----------
fig.savefig(
    "projection_evolution_planetary_maps.tiff",
    dpi=1000,
    bbox_inches="tight",
    pad_inches=0.15
)

fig.savefig(
    "projection_evolution_planetary_maps.pdf",
    bbox_inches="tight",
    pad_inches=0.15
)

print("✔ Final journal-quality figure exported")

# The projection dominance summary below is kept unchanged...
print("\n===== Projection Dominance Summary =====")

for body in bodies_sorted:
    print(f"\n--- {body} ---")
    sub = df_proj[df_proj["target_body"] == body]
    proj_counts = (
        sub.groupby(["decade", "projection_norm"])
        .size()
        .unstack()
    )
    proj_counts = proj_counts.reindex(columns=all_projections, fill_value=0)
    proj_counts = proj_counts.reindex(all_decades, fill_value=0)
    proj_ratio = proj_counts.div(proj_counts.sum(axis=1), axis=0).fillna(0)
    for decade in all_decades:
        row = proj_ratio.loc[decade]
        if row.sum() == 0:
            continue
        top_proj = row.idxmax()
        top_ratio = row.max()
        print(f"{decade}s: {top_proj} ({top_ratio:.2f})")