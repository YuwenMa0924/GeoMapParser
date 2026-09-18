import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import re
import numpy as np

# ==================== 1. Color palette ====================
COLORBLIND_SAFE = [
    "#332288",  # deep blue
    "#88CCEE",  # light blue
    "#44AA99",  # teal
    "#117733",  # green
    "#999933",  # olive
    "#DDCC77",  # sand
    "#CC6677",  # rose
    "#882255",  # wine
    "#AA4499",  # purple
    "#DDDDDD",  # light gray
    "#6699CC",  # steel blue
    "#FFB000"   # orange
]

def get_cb_colors(n):
    if n <= len(COLORBLIND_SAFE):
        return COLORBLIND_SAFE[:n]
    else:
        # Automatic interpolation expansion (rarely used)
        import matplotlib.cm as cm
        cmap = cm.get_cmap('tab20')
        return [cmap(i) for i in np.linspace(0, 1, n)]

# ==================== 2. Global font settings ====================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 5,
    "axes.titlesize": 5,
    "axes.labelsize": 5,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "legend.fontsize": 5
})

# ========== Paths ==========
INPUT_EXCEL = "path/to/maps_table_e5.xlsx"

# ========== Read data ==========
df = pd.read_excel(INPUT_EXCEL)
df["publication_date"] = pd.to_numeric(df["publication_date"], errors="coerce")

START_YEAR = 1960
END_YEAR = 2021

df = df[
    (df["publication_date"] >= START_YEAR) &
    (df["publication_date"] <= END_YEAR)
]

# Extract the scale
def parse_scale(x):
    if isinstance(x, str):
        m = re.search(r"1\s*[:：]\s*([\d,]+)", x)
        if m:
            return int(m.group(1).replace(",", ""))
    return None

df["scale_value"] = df["scale_label"].apply(parse_scale)

# =====================================================
# Data computation
# =====================================================

year_counts = (
    df.dropna(subset=["publication_date"])
      .groupby("publication_date")
      .size()
)

scale_trend = (
    df.dropna(subset=["publication_date", "scale_value"])
      .groupby("publication_date")["scale_value"]
      .mean()
)

df_proj = df.dropna(subset=["publication_date", "projection_norm"]).copy()
df_proj["decade"] = (df_proj["publication_date"] // 10) * 10

proj_counts = (
    df_proj.groupby(["decade", "projection_norm"])
           .size()
           .unstack(fill_value=0)
)

proj_ratio = proj_counts.div(proj_counts.sum(axis=1), axis=0)

# ⭐⭐⭐ Core: sort by importance + automatic colors ⭐⭐⭐
proj_ratio = proj_ratio[proj_ratio.mean().sort_values(ascending=False).index]
colors = get_cb_colors(len(proj_ratio.columns))

# =====================================================
# Create the layout
# =====================================================

fig = plt.figure(figsize=(12, 8), dpi=300)
gs = gridspec.GridSpec(2, 2,
                       width_ratios=[1, 1.3],
                       height_ratios=[1, 1],
                       wspace=0.15,
                       hspace=0.3)

# Subplot (a) (dark main line)
ax_a = fig.add_subplot(gs[0, 0])
ax_a.plot(year_counts.index, year_counts.values,
          color="#2F3E46", linewidth=1.2)
ax_a.set_ylabel("Number of Maps", labelpad=2)
ax_a.set_title("(a)", pad=3)
ax_a.set_xlim(START_YEAR, END_YEAR)
ax_a.tick_params(axis='both', which='major', pad=1)

# Subplot (b) (auxiliary color)
ax_b = fig.add_subplot(gs[1, 0])
ax_b.plot(scale_trend.index, scale_trend.values,
          color="#52796F", linewidth=1.2)
ax_b.set_xlabel("Publication Year", labelpad=2)
ax_b.set_ylabel("Average Scale Denominator (1 : x)", labelpad=2)
ax_b.set_title("(b)", pad=3)
ax_b.set_xlim(START_YEAR, END_YEAR)
ax_b.tick_params(axis='both', which='major', pad=1)

# Subplot (c) (key change: colorblind-friendly palette)
ax_c = fig.add_subplot(gs[:, 1])
proj_ratio.plot(
    kind="area",
    stacked=True,
    ax=ax_c,
    color=colors,   # ⭐replaced
    linewidth=0,
    legend=False
)

ax_c.set_xlabel("Decade", labelpad=2)
ax_c.set_ylabel("Proportion", labelpad=2)
ax_c.set_title("(c)", pad=3)
ax_c.set_ylim(0, 1)
ax_c.tick_params(axis='both', which='major', pad=1)

# Legend
ax_c.legend(
    loc="upper right",
    bbox_to_anchor=(0.98, 0.98),
    fontsize=3.3,
    ncol=2,
    frameon=False,
    columnspacing=0.8,
    handletextpad=0.3
)

# =====================================================
plt.subplots_adjust(
    top=0.9,
    bottom=0.1,
    left=0.08,
    right=0.95
)

plt.savefig(
    "map_temporal_evolution_colorblind.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()