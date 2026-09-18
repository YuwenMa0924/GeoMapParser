import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import re
import numpy as np

from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# ============================================================
# 1. Path Configuration
# ============================================================

INPUT_EXCEL = (
    r"path/to/maps-table/maps_table_e8.xlsx"
)

OUTPUT_DIR = (
    r"path/to/output_figures"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

OUTPUT_PNG = os.path.join(
    OUTPUT_DIR,
    "Figure8_Institutional_Collaboration_Network.png"
)

OUTPUT_PDF = os.path.join(
    OUTPUT_DIR,
    "Figure8_Institutional_Collaboration_Network.pdf"
)

OUTPUT_SVG = os.path.join(
    OUTPUT_DIR,
    "Figure8_Institutional_Collaboration_Network.svg"
)

AGENCY_COLUMN = "agency_norm"


# ============================================================
# 2. basic parameters
# ============================================================

FIGSIZE = (13, 9)
DPI = 300


SPRING_K = 0.42
SPRING_ITERATIONS = 1200
SPRING_SEED = 42


NODE_SIZE_MIN = 450
NODE_SIZE_MAX = 2200


EDGE_WIDTH_MIN = 0.7
EDGE_WIDTH_MAX = 4.0


# ============================================================
# 3. color
# ============================================================

COMMUNITY_COLORS = [
    "#332288",
    "#117733",
    "#CC6677",
    "#88CCEE",
    "#DDCC77",
    "#44AA99",
    "#882255",
    "#AA4499",
    "#6699CC",
    "#E69F00"
]


# ============================================================
# 4. font
# ============================================================

plt.rcParams.update({
    "font.family": "Times New Roman",

    "font.size": 11,

    "axes.titlesize": 14,
    "axes.labelsize": 11,

    "xtick.labelsize": 10,
    "ytick.labelsize": 10,

    "legend.fontsize": 9,

    # Keep PDF/SVG fonts editable
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none"
})


# ============================================================
# 5. tool
# ============================================================

def normalize_agency_name(text):


    text = str(text).strip()

    if not text:
        return ""

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.upper()


def linear_scale(values, out_min, out_max):


    values = np.asarray(
        values,
        dtype=float
    )

    vmin = values.min()
    vmax = values.max()

    if np.isclose(vmax, vmin):

        return np.full(
            len(values),
            (out_min + out_max) / 2
        )

    return (
        out_min
        +
        (
            (values - vmin)
            /
            (vmax - vmin)
        )
        *
        (out_max - out_min)
    )


# ============================================================
# 6. read data
# ============================================================

print("=" * 70)
print("Institutional Collaboration Network")
print("=" * 70)

print("\nReading data...")

df = pd.read_excel(
    INPUT_EXCEL
)

if AGENCY_COLUMN not in df.columns:

    raise ValueError(
        f"does not find: {AGENCY_COLUMN}"
    )

print(
    f"✔ total: {len(df)}"
)


# ============================================================
# 7. Building collaboration network
# ============================================================

print(
    "\nBuilding collaboration network..."
)

edges = []

for agencies in df[
    AGENCY_COLUMN
].dropna():

    # support ; , /
    agency_list = re.split(
        r"[;,/]",
        str(agencies)
    )

    agency_list = [
        normalize_agency_name(a)
        for a in agency_list
    ]

    # Removing invalid items
    agency_list = [
        a
        for a in agency_list
        if a not in {
            "",
            "OTHER",
            "UNKNOWN",
            "NONE",
            "NAN"
        }
    ]

    # Keep only unique institutions within the same map
    agency_list = sorted(
        set(agency_list)
    )

    if len(agency_list) < 2:
        continue

    # Generate pairwise combinations of institutions
    for a, b in itertools.combinations(
        agency_list,
        2
    ):

        edges.append(
            (a, b)
        )


if not edges:

    raise ValueError(
        "no networking, please check agency_norm data."
    )


edges_df = pd.DataFrame(
    edges,
    columns=[
        "source",
        "target"
    ]
)


# ============================================================
# 8. Count cooperation frequency
# ============================================================

edge_weights = (
    edges_df
    .value_counts()
    .reset_index(
        name="weight"
    )
)

print(
    f"✔ Count cooperation relation quantity: {len(edge_weights)}"
)


# ============================================================
# 9. create NetworkX graph
# ============================================================

G = nx.Graph()

for _, row in edge_weights.iterrows():

    G.add_edge(
        row["source"],
        row["target"],
        weight=int(
            row["weight"]
        )
    )

print(
    f"✔ number of node: {G.number_of_nodes()}"
)

print(
    f"✔ Calculate edge count: {G.number_of_edges()}"
)


# ============================================================
# 10. Weighted Degree
# ============================================================

weighted_degree = dict(
    G.degree(
        weight="weight"
    )
)


# ============================================================
# 11. Community Detection
# ============================================================

print(
    "\nDetecting network communities..."
)

communities = list(
    nx.community.greedy_modularity_communities(
        G,
        weight="weight"
    )
)

print(
    f"✔ Community : {len(communities)}"
)


node_community = {}

for cid, community in enumerate(
    communities,
    start=1
):

    for node in community:

        node_community[
            node
        ] = cid

    print(
        f"  Community {cid}: "
        f"{len(community)} nodes"
    )


# ============================================================
# 12. Community color
# ============================================================

community_color_dict = {}

for cid, community in enumerate(
    communities
):

    color = COMMUNITY_COLORS[
        cid % len(COMMUNITY_COLORS)
    ]

    for node in community:

        community_color_dict[
            node
        ] = color


# ============================================================
# 13. node size
# ============================================================

nodes = list(
    G.nodes()
)

degree_values = [
    weighted_degree[node]
    for node in nodes
]

node_sizes = linear_scale(
    degree_values,
    NODE_SIZE_MIN,
    NODE_SIZE_MAX
)

node_size_dict = dict(
    zip(
        nodes,
        node_sizes
    )
)


# ============================================================
# 14. edge
# ============================================================

edge_items = list(
    G.edges(
        data=True
    )
)

edge_weight_values = [
    data["weight"]
    for _, _, data in edge_items
]

# log scaling
log_edge_weights = np.log1p(
    edge_weight_values
)

edge_widths = linear_scale(
    log_edge_weights,
    EDGE_WIDTH_MIN,
    EDGE_WIDTH_MAX
)


# ============================================================
# 15. layout
# ============================================================

print(
    "\nCalculating network layout..."
)

pos = nx.spring_layout(
    G,
    seed=SPRING_SEED,
    k=SPRING_K,
    iterations=SPRING_ITERATIONS,
    weight="weight"
)


# ============================================================
# 16. create Figure
# ============================================================

fig, ax = plt.subplots(
    figsize=FIGSIZE,
    dpi=DPI
)


# ============================================================
# 17. Draw graph edges
# ============================================================

nx.draw_networkx_edges(
    G,
    pos,
    ax=ax,
    width=edge_widths,
    edge_color="#A6A6A6",
    alpha=0.55,
    style="solid"
)


# ============================================================
# 18. Draw nodes # White nodes with colored community borders
# ============================================================

node_edge_colors = [
    community_color_dict[node]
    for node in nodes
]

nx.draw_networkx_nodes(
    G,
    pos,
    ax=ax,

    node_size=node_sizes,


    node_color="#FFFFFF",


    edgecolors=node_edge_colors,

    linewidths=2.4,

    alpha=1.0
)


# ============================================================
# 19. Node labels: placed directly at the center of circles
# ============================================================

for node in nodes:

    x, y = pos[node]

    # Adjust the font size according to the name length
    name_length = len(node)

    if name_length <= 4:
        fontsize = 10

    elif name_length <= 6:
        fontsize = 9.5

    elif name_length <= 8:
        fontsize = 9

    else:
        fontsize = 8.5

    ax.text(
        x,
        y,
        node,

        fontsize=fontsize,

        fontweight="bold",

        color="#222222",

        ha="center",
        va="center",


        zorder=10
    )


# ============================================================
# 20. Build a concise unified legend
# ============================================================

legend_handles = []


# ------------------------------------------------------------
# 20.1 Node size
# ------------------------------------------------------------

degree_min = min(
    degree_values
)

degree_mid = np.median(
    degree_values
)

degree_max = max(
    degree_values
)

legend_sizes = linear_scale(
    [
        degree_min,
        degree_mid,
        degree_max
    ],
    NODE_SIZE_MIN,
    NODE_SIZE_MAX
)


legend_handles.extend([
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markerfacecolor="white",
        markeredgecolor="#555555",
        markeredgewidth=1.5,
        markersize=np.sqrt(
            legend_sizes[0]
        ) / 2.2,
        label=f"Node size = weighted degree ({degree_min:.0f})"
    ),

    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markerfacecolor="white",
        markeredgecolor="#555555",
        markeredgewidth=1.5,
        markersize=np.sqrt(
            legend_sizes[1]
        ) / 2.2,
        label=f"Node size = weighted degree ({degree_mid:.0f})"
    ),

    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markerfacecolor="white",
        markeredgecolor="#555555",
        markeredgewidth=1.5,
        markersize=np.sqrt(
            legend_sizes[2]
        ) / 2.2,
        label=f"Node size = weighted degree ({degree_max:.0f})"
    )
])


# ------------------------------------------------------------
# 20.2 Edge width
# ------------------------------------------------------------

edge_min = min(
    edge_weight_values
)

edge_mid = int(
    np.median(
        edge_weight_values
    )
)

edge_max = max(
    edge_weight_values
)

legend_handles.extend([
    Line2D(
        [0],
        [0],
        color="#A6A6A6",
        lw=1.0,
        label=f"Edge width = collaboration frequency ({edge_min})"
    ),

    Line2D(
        [0],
        [0],
        color="#A6A6A6",
        lw=2.5,
        label=f"Edge width = collaboration frequency ({edge_mid})"
    ),

    Line2D(
        [0],
        [0],
        color="#A6A6A6",
        lw=4.0,
        label=f"Edge width = collaboration frequency ({edge_max})"
    )
])


# ------------------------------------------------------------
# 20.3 Community
# ------------------------------------------------------------

for cid, community in enumerate(
    communities,
    start=1
):

    color = community_color_dict[
        next(iter(community))
    ]

    legend_handles.append(
        Patch(
            facecolor="white",
            edgecolor=color,
            linewidth=2.4,
            label=f"Community {cid}"
        )
    )


# ============================================================
#21. Fix legend on figure right side # Do not rely on bbox_inches for axes-legend layout
# ============================================================

fig.legend(
    handles=legend_handles,

    title="Network attributes",

    loc="center right",

    bbox_to_anchor=(
        0.985,
        0.50
    ),

    frameon=True,

    framealpha=0.97,

    edgecolor="#BDBDBD",

    facecolor="white",

    fontsize=8.5,

    title_fontsize=10,

    borderpad=0.9,

    labelspacing=0.75,

    handlelength=2.5,

    handletextpad=0.7
)


# ============================================================
# 22. Remove axes
# ============================================================

ax.axis(
    "off"
)




# ============================================================
# 23. adjust Figure
# ============================================================

plt.subplots_adjust(
    left=0.02,
    right=0.78,
    top=0.98,
    bottom=0.03
)


# ============================================================
# 25. save as PNG / PDF / SVG
# ============================================================

print(
    "\nSaving figures..."
)

fig.savefig(
    OUTPUT_PNG,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    OUTPUT_PDF,
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    OUTPUT_SVG,
    bbox_inches="tight",
    facecolor="white"
)


print(
    f"✔ PNG saved:\n{OUTPUT_PNG}"
)

print(
    f"✔ PDF saved:\n{OUTPUT_PDF}"
)

print(
    f"✔ SVG saved:\n{OUTPUT_SVG}"
)


# ============================================================
# 26. output check
# ============================================================

print(
    "\n===== Output Check ====="
)

for path in [
    OUTPUT_PNG,
    OUTPUT_PDF,
    OUTPUT_SVG
]:

    if os.path.exists(path):

        print(
            f"✔ {path}"
        )

    else:

        print(
            f"✘ Warning: File not generated: {path}"
        )


# ============================================================
# 27. Output node information
# ============================================================

print(
    "\n===== Node Quantitative Information ====="
)

print(
    "Node | Weighted Degree | Community | Node Size"
)

print(
    "-" * 85
)

for node in sorted(
    nodes,
    key=lambda n:
    weighted_degree[n],
    reverse=True
):

    print(
        f"{node:15s} | "
        f"{weighted_degree[node]:15.0f} | "
        f"Community "
        f"{node_community[node]:2d} | "
        f"{node_size_dict[node]:10.0f}"
    )


# ============================================================
# 28. output Top collaboration pairs
# ============================================================

print(
    "\n===== Top Collaboration Pairs ====="
)

top_edges = sorted(
    edge_items,
    key=lambda x:
    x[2]["weight"],
    reverse=True
)

print(
    "Source | Target | Frequency"
)

print(
    "-" * 80
)

for u, v, data in top_edges[:20]:

    print(
        f"{u:15s} | "
        f"{v:15s} | "
        f"{data['weight']:8d}"
    )


# ============================================================
# 29. show
# ============================================================

plt.show()