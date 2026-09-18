import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np

warnings.filterwarnings('ignore')

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

sns.set_style("white")
sns.set(font='Times New Roman')

plt.rcParams.update({
    "font.size": 5,
    "axes.titlesize": 5,
    "axes.labelsize": 5,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "legend.fontsize": 5,
    "axes.linewidth": 0.8,
    "axes.edgecolor": AXIS_GRAY,
    "xtick.color": AXIS_GRAY,
    "ytick.color": AXIS_GRAY,
    "text.color": TEXT_DARK
})

# =========================
# Path configuration
# =========================
INPUT_EXCEL = "path/to/maps_table_e6.xlsx"

# =========================
# Read data
# =========================
df = pd.read_excel(INPUT_EXCEL)

# =========================================================
# ① Agency output share (pie chart)
# =========================================================
df_agency = df.dropna(subset=["agency_norm"]).copy()
df_agency["agency_list"] = df_agency["agency_norm"].astype(str).str.split(";")
df_agency = df_agency.explode("agency_list")
df_agency["agency_list"] = df_agency["agency_list"].str.strip()
df_agency = df_agency[~df_agency["agency_list"].str.upper().eq("0")]

agency_counts = df_agency["agency_list"].value_counts()
top_n = 16
top_agencies = agency_counts.iloc[:top_n]
others_sum = agency_counts.iloc[top_n:].sum()
agency_plot = top_agencies.copy()
if others_sum > 0:
    agency_plot["Others"] = others_sum

def autopct_func(pct):
    return f"{pct:.1f}%" if pct >= 1 else ""

colors = get_cb_colors(len(agency_plot))

fig, ax = plt.subplots(figsize=(12, 10), dpi=100)
wedges, texts, autotexts = ax.pie(
    agency_plot.values,
    autopct=autopct_func,
    startangle=140,
    pctdistance=0.8,
    wedgeprops=dict(edgecolor='white', linewidth=1),
    colors=colors
)

ax.legend(wedges, agency_plot.index,
          loc="center left",
          bbox_to_anchor=(1.0, 0.5),
          fontsize=9)

plt.setp(autotexts, size=10, weight="bold")
ax.set_title("Institutional Contribution Share of Planetary Geologic Maps", fontsize=14, pad=20)
plt.tight_layout()
plt.show()

# =========================================================
# Mission / Product / Sensor – Target Body (Combined Figure)
# =========================================================
df_mission = df.dropna(subset=["mission_norm", "target_body"]).copy()
df_mission["mission_norm"] = df_mission["mission_norm"].astype(str).str.split(";")
df_mission["target_body"] = df_mission["target_body"].astype(str).str.split(r"[;,]")
df_mission = df_mission.explode("mission_norm").explode("target_body")
df_mission["mission_norm"] = df_mission["mission_norm"].str.strip()
df_mission["target_body"] = df_mission["target_body"].str.strip()

mission_target_matrix = (
    df_mission.groupby(["mission_norm", "target_body"])
              .size()
              .unstack(fill_value=0)
)

df_product = df.dropna(subset=["product", "target_body"]).copy()
df_product["product"] = df_product["product"].astype(str).str.split(";")
df_product["target_body"] = df_product["target_body"].astype(str).str.split(r"[;,]")
df_product = df_product.explode("product").explode("target_body")
df_product["product"] = df_product["product"].str.strip()
df_product["target_body"] = df_product["target_body"].str.strip()

product_target_matrix = (
    df_product.groupby(["product", "target_body"])
              .size()
              .unstack(fill_value=0)
)

df_sensor = df.dropna(subset=["sensor", "target_body"]).copy()
df_sensor["sensor"] = df_sensor["sensor"].astype(str).str.split(";")
df_sensor["target_body"] = df_sensor["target_body"].astype(str).str.split(r"[;,]")
df_sensor = df_sensor.explode("sensor").explode("target_body")
df_sensor["sensor"] = df_sensor["sensor"].str.strip()
df_sensor["target_body"] = df_sensor["target_body"].str.strip()

sensor_target_matrix = (
    df_sensor.groupby(["sensor", "target_body"])
              .size()
              .unstack(fill_value=0)
)

def print_top_distribution(matrix, name):
    print(f"\n===== {name} Distribution (count + share)=====")
    for body in matrix.columns:
        col = matrix[body]
        total = col.sum()
        if total == 0:
            continue
        df_stat = pd.DataFrame({
            "count": col,
            "percentage": col / total * 100
        }).sort_values("count", ascending=False)
        print(f"\n--- {body} ---")
        print(f"Total = {total}")
        print(df_stat.head(10).round(2))

print_top_distribution(mission_target_matrix, "Mission")
print_top_distribution(product_target_matrix, "Product")
print_top_distribution(sensor_target_matrix, "Sensor")

# =========================================================
# Combined figure (heatmaps) — switched to a light gradient palette (no dark colors)
# =========================================================
# Generate a light colormap from white to soft blue-green
light_cmap = sns.light_palette("#44AA99", as_cmap=True)

fig, axes = plt.subplots(1, 3, figsize=(22, 8), dpi=300)

sns.heatmap(mission_target_matrix, cmap=light_cmap, linewidths=0.5, cbar=True, annot=False, ax=axes[0])
axes[0].set_title("(a) Mission–Target Body Mapping", fontsize=5)
axes[0].set_xlabel("Target Body", fontsize=4)
axes[0].set_ylabel("Mission", fontsize=3)
axes[0].tick_params(axis='x', labelsize=3, rotation=0)
axes[0].tick_params(axis='y', labelsize=3, rotation=0)

sns.heatmap(product_target_matrix, cmap=light_cmap, linewidths=0.5, cbar=True, annot=False, ax=axes[1])
axes[1].set_title("(b) Product–Target Body Mapping", fontsize=5)
axes[1].set_xlabel("Target Body", fontsize=4)
axes[1].set_ylabel("Product", fontsize=3)
axes[1].tick_params(axis='x', labelsize=3, rotation=0)
axes[1].tick_params(axis='y', labelsize=3, rotation=0)

sns.heatmap(sensor_target_matrix, cmap=light_cmap, linewidths=0.5, cbar=True, annot=False, ax=axes[2])
axes[2].set_title("(c) Sensor–Target Body Mapping", fontsize=5)
axes[2].set_xlabel("Target Body", fontsize=4)
axes[2].set_ylabel("Sensor", fontsize=3)
axes[2].tick_params(axis='x', labelsize=3, rotation=0)
axes[2].tick_params(axis='y', labelsize=3, rotation=0)

plt.tight_layout()
plt.show()

# ===============================
# Collaboration analysis (bar chart)
# ===============================
df_collab = df.dropna(subset=["agency_norm"]).copy()
df_collab = df_collab[~df_collab["agency_norm"].str.upper().eq("UNKNOWN")]

df_collab_explode = df_collab.copy()
df_collab_explode["agency_list"] = df_collab_explode["agency_norm"].astype(str).str.split(";")
df_collab_explode = df_collab_explode.explode("agency_list")
df_collab_explode["agency_list"] = df_collab_explode["agency_list"].str.strip()
df_collab_explode = df_collab_explode[df_collab_explode["agency_list"] != ""]
df_collab_explode = df_collab_explode[~df_collab_explode["agency_list"].str.upper().eq("UNKNOWN")]

agency_count_clean = df_collab_explode.groupby(df_collab_explode.index)["agency_list"].count()
df_collab["agency_count"] = agency_count_clean.fillna(0)
df_collab = df_collab[df_collab["agency_count"] > 0]

df_collab["collaboration_type"] = df_collab["agency_count"].apply(
    lambda x: "Multi-institution Collaboration" if x > 1 else "Single Institution"
)
collab_counts = df_collab["collaboration_type"].value_counts()

plt.figure(figsize=(8, 6), dpi=100)
bars = plt.bar(collab_counts.index, collab_counts.values, color=get_cb_colors(2))
plt.ylabel("Number of Maps", fontsize=11, labelpad=10)
plt.title("Inter-institutional Collaboration in Planetary Geologic Mapping", fontsize=14, pad=15)
plt.xticks(rotation=0)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{int(height)}",
             ha='center', va='bottom', fontsize=11, weight='bold')
plt.ylim(0, max(collab_counts.values) * 1.1)
plt.tight_layout()
plt.show()