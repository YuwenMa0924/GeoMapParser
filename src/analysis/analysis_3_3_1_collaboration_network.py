import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import re
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

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

def get_contrast_text_color(hex_color):
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(hex_color)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return 'white' if luminance < 0.5 else 'black'

# ==================== Global style ====================
TEXT_DARK  = '#2E2E2E'
AXIS_GRAY  = '#4A4A4A'
GRID_GRAY  = '#D9D9D9'

sns.set_style("white")
sns.set(font='Times New Roman')

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.linewidth": 0.8,
    "axes.edgecolor": AXIS_GRAY,
    "xtick.color": AXIS_GRAY,
    "ytick.color": AXIS_GRAY,
    "text.color": TEXT_DARK
})

# =========================
# Data reading and network construction (original logic fully preserved)
# =========================
INPUT_EXCEL = "path/to/maps_table_e5.xlsx"
df = pd.read_excel(INPUT_EXCEL)

edges = []
for agencies in df["agency_norm"].dropna():
    agency_list = re.split(r"[;,/]", str(agencies))
    agency_list = [a.strip().upper() for a in agency_list]
    agency_list = [a for a in agency_list if a not in ["OTHER", "UNKNOWN", ""]]
    if len(agency_list) < 2:
        continue
    for a, b in itertools.combinations(sorted(set(agency_list)), 2):
        edges.append((a, b))

edges_df = pd.DataFrame(edges, columns=["source", "target"])
edge_weights = edges_df.value_counts().reset_index(name="weight")

G = nx.Graph()
for _, row in edge_weights.iterrows():
    G.add_edge(row["source"], row["target"], weight=row["weight"])

# Node colors (colorblind-friendly)
unique_nodes = list(G.nodes())
color_palette = get_cb_colors(len(unique_nodes))
node_color_dict = {node: color_palette[i] for i, node in enumerate(unique_nodes)}
node_colors = [node_color_dict[node] for node in G.nodes()]

# Edge widths (original logic)
edge_weights_list = [d["weight"] for (_, _, d) in G.edges(data=True)]
edge_widths = [np.log(w + 1) * 0.6 for w in edge_weights_list]
edge_widths = [max(w, 0.3) for w in edge_widths]

# Dynamic node sizes (original logic)
node_weighted_degree = dict(G.degree(weight='weight'))
min_size = 200
max_size = 1000
min_degree = min(node_weighted_degree.values())
max_degree = max(node_weighted_degree.values())
node_sizes = [
    min_size + (node_weighted_degree[node] - min_degree) / (max_degree - min_degree) * (max_size - min_size)
    for node in G.nodes()
]

# ===============================
# Plot: draw only the network graph (no table)
# ===============================
plt.figure(figsize=(12, 9), dpi=300)  # slightly enlarged to fit the legend

pos = nx.spring_layout(G, seed=42, k=0.5, iterations=500)

# Draw nodes
nx.draw_networkx_nodes(
    G, pos,
    node_size=node_sizes,
    node_color=node_colors,
    edgecolors="black",
    linewidths=0.5
)

# Draw edges
nx.draw_networkx_edges(
    G, pos,
    width=edge_widths,
    edge_color="#888888",
    alpha=0.7
)

# Adaptive labels (to fix the too-dark color issue)
for node, (x, y) in pos.items():
    node_color = node_color_dict[node]
    text_color = get_contrast_text_color(node_color)
    plt.text(x, y, node, fontsize=9, fontweight='bold',
             ha='center', va='center', color=text_color,
             bbox=dict(boxstyle='round,pad=0.2', facecolor=node_color, edgecolor='none', alpha=0.7))

plt.title("Institution Collaboration Network", fontsize=14, pad=15)

# ========== Revised legend (placed outside on the right, not disturbing the network plot) ==========
legend_elements = [
    Patch(facecolor='#DDDDDD', edgecolor='black', label='Nodes (Institutions)'),
    Line2D([0], [0], color='#888888', lw=2, label='Edges (Collaboration)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=6, label='Node size(Weighted degree)'),
    Line2D([0], [0], color='#888888', lw=4, label='Edge width(Frequency)')
]

# Place the legend outside on the right and adjust the margins
plt.legend(
    handles=legend_elements,
    bbox_to_anchor=(1.02, 1),   # placed outside on the right of the figure
    loc='upper left',
    fontsize=6,
    frameon=True,
    edgecolor='black',
    handlelength=1.2,
    handletextpad=0.5,
    borderaxespad=0
)

# Make room for the right-side legend (25% whitespace on the right)
plt.subplots_adjust(right=0.75)

plt.axis("off")
plt.tight_layout()  # Note: tight_layout may ignore subplots_adjust, but it does not conflict here

# ========== Print node information (original logic restored) ==========
print("\n===== Node Quantitative Information =====")
print("Node | Color | Weighted Degree | Node Size")
print("-" * 60)
for node in G.nodes():
    color_hex = node_color_dict[node]
    wdeg = node_weighted_degree[node]
    size = node_sizes[list(G.nodes()).index(node)]
    print(f"{node:15} | {color_hex:10} | {wdeg:15} | {size:10.0f}")

plt.show()