import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import re
import warnings
from scipy.stats import chi2_contingency
from scipy.spatial.distance import jensenshannon
import numpy as np

warnings.filterwarnings("ignore")

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
        cmap = plt.cm.get_cmap('tab20')
        return [cmap(i) for i in np.linspace(0, 1, n)]

# ==================== Global style ====================
TEXT_DARK  = '#2E2E2E'
AXIS_GRAY  = '#4A4A4A'
GRID_GRAY  = '#D9D9D9'

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 3.5,
    "axes.titlesize": 3.5,
    "axes.labelsize": 3.5,
    "xtick.labelsize": 3.5,
    "ytick.labelsize": 3,
    "legend.fontsize": 3,
    "axes.grid": False,
    "figure.dpi": 600
})

plt.rcParams.update({
    "axes.linewidth": 0.8,
    "axes.edgecolor": AXIS_GRAY,
    "xtick.color": AXIS_GRAY,
    "ytick.color": AXIS_GRAY,
    "text.color": TEXT_DARK
})

sns.set_style("white")
plt.rcParams["font.family"] = "Times New Roman"

# =========================
# Read data
# =========================
INPUT_EXCEL = "path/to/maps_table_e6.xlsx"
df = pd.read_excel(INPUT_EXCEL)

# =====================================================
# Utility functions (original logic preserved)
# =====================================================
def parse_scale_value(x):
    if isinstance(x, str):
        m = re.search(r"1\s*[:：]\s*([\d,]+)", x)
        if m:
            return int(m.group(1).replace(",", ""))
    return None

def scale_bin(val):
    if val is None:
        return None
    if val <= 250_000:
        return "Large-scale"
    elif val <= 1_000_000:
        return "Medium-scale"
    elif val <= 5_000_000:
        return "Small-scale"
    else:
        return "Very small-scale"

# =====================================================
# Data preprocessing (original logic preserved)
# =====================================================
df["scale_value"] = df["scale"].apply(parse_scale_value)
df["scale_bin"] = df["scale_value"].apply(scale_bin)

# =====================================================
# 1 Projection preference
# =====================================================
proj_pref = (
    df.dropna(subset=["target_body", "projection_norm"])
      .groupby(["target_body", "projection_norm"])
      .size()
      .reset_index(name="count")
)

# =====================================================
# 2 Scale distribution
# =====================================================
scale_long = (
    df.dropna(subset=["target_body", "scale_bin"])
      .groupby(["target_body", "scale_bin"])
      .size()
      .reset_index(name="count")
)

# =====================================================
# 3 Heatmap data — show only the 5 most common positions
# =====================================================
df_agency = df.dropna(subset=["agency_norm", "scale_position"]).copy()
df_agency["agency_norm"] = df_agency["agency_norm"].astype(str)
df_agency = df_agency.assign(
    agency_list=df_agency["agency_norm"].str.split(";")
).explode("agency_list")
df_agency["agency_list"] = df_agency["agency_list"].str.strip()

heatmap_raw = (
    df_agency.groupby(["agency_list", "scale_position"])
             .size()
             .unstack(fill_value=0)
)

col_totals = heatmap_raw.sum(axis=0).sort_values(ascending=False)
top5_positions = col_totals.head(5).index.tolist()
heatmap_data = heatmap_raw[top5_positions]
row_totals = heatmap_data.sum(axis=1).sort_values(ascending=False)
heatmap_data = heatmap_data.loc[row_totals.index]

# =====================================================
# 4. Statistical test: are there significant differences in projection preference (exclude small-sample target bodies + merge low-frequency projections)
# =====================================================
print("\n========== Statistical test results ==========")

MIN_BODY_COUNT = 20
MIN_PROJ_COUNT = 50

proj_data = df.dropna(subset=["target_body", "projection_norm"])
body_counts = proj_data["target_body"].value_counts()
valid_bodies = body_counts[body_counts >= MIN_BODY_COUNT].index.tolist()
print(f"Target bodies included in the test (map count ≥ {MIN_BODY_COUNT}): {valid_bodies}")
excluded_bodies = body_counts[body_counts < MIN_BODY_COUNT].index.tolist()
if excluded_bodies:
    print(f"Excluded small-sample target bodies: {excluded_bodies}")

proj_data_filtered = proj_data[proj_data["target_body"].isin(valid_bodies)]
proj_counts = proj_data_filtered["projection_norm"].value_counts()
rare_projs = proj_counts[proj_counts < MIN_PROJ_COUNT].index.tolist()
print(f"\nProjection types merged into 'Other' (total frequency < {MIN_PROJ_COUNT}): {rare_projs}")
if not rare_projs:
    print("No low-frequency projection types; no merging needed.")

proj_data_filtered = proj_data_filtered.copy()
proj_data_filtered["proj_group"] = proj_data_filtered["projection_norm"].apply(
    lambda x: "Other" if x in rare_projs else x
)

print("\nMerged projection categories and frequencies:")
print(proj_data_filtered["proj_group"].value_counts())

contingency = pd.crosstab(proj_data_filtered["target_body"],
                          proj_data_filtered["proj_group"])
print("\nContingency table (rows = target bodies, columns = projection categories):")
print(contingency)

chi2, p, dof, expected = chi2_contingency(contingency)
n = contingency.sum().sum()
phi2 = chi2 / n
r, k = contingency.shape
cramer_v = np.sqrt(phi2 / min(k-1, r-1))

print("\n=== Chi-square test (after filtering + merging low-frequency projections)===")
print(f"Chi-square value = {chi2:.4f}")
print(f"Degrees of freedom = {dof}")
print(f"p value = {p:.6e}")
print(f"Cramer's V = {cramer_v:.4f}")
print(f"Minimum expected frequency = {expected.min():.4f}")

expected_lt5 = (expected < 5).sum()
expected_lt1 = (expected < 1).sum()
total_cells = expected.size
print(f"Proportion of cells with expected frequency < 5: {expected_lt5}/{total_cells} ({100*expected_lt5/total_cells:.1f}%)")
print(f"Proportion of cells with expected frequency < 1: {expected_lt1}/{total_cells} ({100*expected_lt1/total_cells:.1f}%)")

if expected_lt1 > 0:
    print("Warning: cells with expected frequency < 1 exist; the chi-square result may be unreliable. Consider increasing MIN_PROJ_COUNT.")
elif expected_lt5 / total_cells > 0.2:
    print("Warning: more than 20% of cells have expected frequency < 5; the chi-square approximation may be inaccurate.")
else:
    print("Expected frequencies meet the basic requirements of the chi-square test (no <1 cells, and the <5 proportion ≤20%).")

if p < 0.001:
    print("\nConclusion: the projection type distribution differs highly significantly across target bodies (p < 0.001)")
elif p < 0.05:
    print("Conclusion: the projection type distribution differs significantly across target bodies (p < 0.05)")
else:
    print("Conclusion: no significant difference found (p >= 0.05)")

# 4.3 JS divergence (using the merged projection groups)
all_groups = proj_data_filtered["proj_group"].unique()
bodies = valid_bodies
prob_matrix = []
for body in bodies:
    subset = proj_data_filtered[proj_data_filtered["target_body"] == body]
    counts = subset["proj_group"].value_counts().reindex(all_groups, fill_value=0)
    prob = counts / counts.sum()
    prob_matrix.append(prob)

prob_df = pd.DataFrame(prob_matrix, index=bodies, columns=all_groups)
n_bodies = len(bodies)
js_matrix = np.zeros((n_bodies, n_bodies))
for i in range(n_bodies):
    for j in range(i+1, n_bodies):
        js = jensenshannon(prob_df.iloc[i], prob_df.iloc[j], base=2)
        js_matrix[i, j] = js
        js_matrix[j, i] = js

# Plot the JS divergence heatmap (switched to a light gradient)
plt.figure(figsize=(8, 6))
sns.heatmap(js_matrix, annot=True, fmt=".3f",
            xticklabels=bodies, yticklabels=bodies,
            cmap=sns.light_palette("#44AA99", as_cmap=True),
            cbar_kws={"label": "Jensen–Shannon divergence"})
plt.title(f"Projection Distribution Dissimilarity (bodies ≥{MIN_BODY_COUNT}, rare projections merged)")
plt.tight_layout()
plt.savefig("projection_js_divergence_merged.png", dpi=300)
plt.show()

# =====================================================
# Layout: two stacked figures in the left column + one figure in the right column (original layout fully preserved, only the palette changed)
# =====================================================
fig = plt.figure(figsize=(12, 8), dpi=300)
gs = gridspec.GridSpec(
    2, 2,
    width_ratios=[1, 1.2],
    height_ratios=[1, 1],
    wspace=0.25,
    hspace=0.4
)

# (a) Projection preference (palette replaced with the colorblind-friendly one)
ax_a = fig.add_subplot(gs[0, 0])
sns.barplot(
    data=proj_pref,
    x="target_body",
    y="count",
    hue="projection_norm",
    ax=ax_a,
    palette=get_cb_colors(len(proj_pref["projection_norm"].unique()))
)
ax_a.set_title("(a) Projection Preferences", pad=1)
ax_a.set_xlabel("Planetary Body", labelpad=1)
ax_a.set_ylabel("Number of Maps", labelpad=1)
ax_a.tick_params(axis="x", rotation=45, pad=0.5)
ax_a.legend(
    title="Projection",
    fontsize=2,
    frameon=False,
    loc="upper left",
    ncol=2,
    bbox_to_anchor=(0.02, 0.98),
    handlelength=0.8,
    handletextpad=0.2
)

# (b) Scale distribution (palette replaced with the colorblind-friendly one)
ax_b = fig.add_subplot(gs[1, 0])
sns.barplot(
    data=scale_long,
    x="target_body",
    y="count",
    hue="scale_bin",
    ax=ax_b,
    palette=get_cb_colors(len(scale_long["scale_bin"].unique()))
)
ax_b.set_title("(b) Scale Level Distribution", pad=1)
ax_b.set_xlabel("Planetary Body", labelpad=1)
ax_b.set_ylabel("Number of Maps", labelpad=1)
ax_b.tick_params(axis="x", rotation=45, pad=0.5)
ax_b.legend(
    title="Scale Level",
    fontsize=3,
    frameon=False,
    loc="upper right",
    handlelength=0.8,
    handletextpad=0.2
)

# (c) Heatmap (switched to a light gradient)
ax_c = fig.add_subplot(gs[:, 1])
sns.heatmap(
    heatmap_data,
    cmap=sns.light_palette("#44AA99", as_cmap=True),
    annot=True,
    fmt="d",
    annot_kws={"size": 3},
    cbar_kws={"shrink": 0.8, "label": ""},
    ax=ax_c
)
ax_c.set_title("(c) Scale Block Position by Agency", pad=1)
ax_c.set_xlabel("Scale Block Position", labelpad=1)
ax_c.set_ylabel("Agency", labelpad=1)
ax_c.tick_params(axis="y", rotation=0, pad=0.3)

plt.subplots_adjust(top=0.9, bottom=0.08, left=0.06, right=0.94)
plt.savefig("projection_scale_heatmap_layout_final.png", dpi=300, bbox_inches="tight")
plt.show()
print("Minimum expected frequency:", expected.min())