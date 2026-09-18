import os
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


warnings.filterwarnings("ignore")


# ============================================================
# 1. Path configuration
# ============================================================

INPUT_EXCEL = (
    r"path/to/maps-table/"
    r"maps_table_e8.xlsx"
)

OUTPUT_DIR = (
    r"path/to/output_figures/"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


OUTPUT_PNG = os.path.join(
    OUTPUT_DIR,
    "Figure_Mission_Product_Sensor_TargetBody_Heatmaps.png"
)

OUTPUT_PDF = os.path.join(
    OUTPUT_DIR,
    "Figure_Mission_Product_Sensor_TargetBody_Heatmaps.pdf"
)

OUTPUT_SVG = os.path.join(
    OUTPUT_DIR,
    "Figure_Mission_Product_Sensor_TargetBody_Heatmaps.svg"
)


# ============================================================
# 2. Field configuration
# ============================================================

MISSION_COLUMN = "mission_norm"
PRODUCT_COLUMN = "product"
SENSOR_COLUMN = "sensor"
BODY_COLUMN = "target_body"


# ============================================================
# 3. Plotting parameters
# ============================================================

DPI = 300

FIG_W = 17
FIG_H = 8

FONT_FAMILY = "Times New Roman"

TITLE_SIZE = 12
AXIS_LABEL_SIZE = 10
TICK_SIZE = 8.5
COLORBAR_SIZE = 8.5


# ============================================================
# 4. Unified continuous colormap
#
# Matches the heatmaps in Figures 9–11:
# light → teal → dark teal
# ============================================================

HEATMAP_CMAP = sns.light_palette(
    "#44AA99",
    as_cmap=True
)


# ============================================================
# 5. Global plotting style
# ============================================================

plt.rcParams.update({
    "font.family": FONT_FAMILY,

    "font.size": 10,

    "axes.titlesize": TITLE_SIZE,

    "axes.labelsize": AXIS_LABEL_SIZE,

    "xtick.labelsize": TICK_SIZE,

    "ytick.labelsize": TICK_SIZE,

    "pdf.fonttype": 42,

    "ps.fonttype": 42,

    "svg.fonttype": "none",

    "axes.unicode_minus": False,

    "axes.linewidth": 0.8,

    "axes.edgecolor": "#555555",

    "xtick.color": "#555555",

    "ytick.color": "#555555",

    "text.color": "#2E2E2E"
})


# ============================================================
# 6. Read data
# ============================================================

print("=" * 80)
print(
    "Mission / Product / Sensor – Target Body Heatmaps"
)
print("=" * 80)

print(
    "\nReading Excel..."
)

df = pd.read_excel(
    INPUT_EXCEL
)

print(
    f"✔ Raw record count: {len(df)}"
)


# ============================================================
# 7. Field checks
# ============================================================

required_columns = [
    MISSION_COLUMN,
    PRODUCT_COLUMN,
    SENSOR_COLUMN,
    BODY_COLUMN
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing the following fields:\n"
        +
        "\n".join(
            missing_columns
        )
    )


# ============================================================
# 8. Utility functions
# ============================================================

def build_matrix(
    data,
    category_column,
    body_column
):
    """
    Build the
    category × target_body
    frequency matrix.

    Multiple categories in one record are
    supported and split on semicolons.
    """

    subset = data.dropna(
        subset=[
            category_column,
            body_column
        ]
    ).copy()

    if subset.empty:

        return pd.DataFrame()


    # Split categories
    subset[
        category_column
    ] = (
        subset[
            category_column
        ]
        .astype(str)
        .str.split(";")
    )

    # Split target bodies
    subset[
        body_column
    ] = (
        subset[
            body_column
        ]
        .astype(str)
        .str.split(
            r"[;,]"
        )
    )

    # explode
    subset = (
        subset
        .explode(
            category_column
        )
        .explode(
            body_column
        )
    )

    # Strip whitespace
    subset[
        category_column
    ] = (
        subset[
            category_column
        ]
        .astype(str)
        .str.strip()
    )

    subset[
        body_column
    ] = (
        subset[
            body_column
        ]
        .astype(str)
        .str.strip()
    )


    # Drop invalid values
    invalid = {
        "",
        "NONE",
        "NAN",
        "UNKNOWN",
        "OTHER"
    }

    subset = subset[
        ~subset[
            category_column
        ]
        .str.upper()
        .isin(invalid)
    ].copy()

    subset = subset[
        ~subset[
            body_column
        ]
        .str.upper()
        .isin(invalid)
    ].copy()


    # Build the frequency matrix
    matrix = (
        subset
        .groupby(
            [
                category_column,
                body_column
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    return matrix


# ============================================================
# 9. Build the three heatmap matrices
# ============================================================

print(
    "\nBuilding matrices..."
)


mission_target_matrix = build_matrix(
    df,
    MISSION_COLUMN,
    BODY_COLUMN
)

product_target_matrix = build_matrix(
    df,
    PRODUCT_COLUMN,
    BODY_COLUMN
)

sensor_target_matrix = build_matrix(
    df,
    SENSOR_COLUMN,
    BODY_COLUMN
)


print(
    f"✔ Mission matrix: "
    f"{mission_target_matrix.shape}"
)

print(
    f"✔ Product matrix: "
    f"{product_target_matrix.shape}"
)

print(
    f"✔ Sensor matrix: "
    f"{sensor_target_matrix.shape}"
)


# ============================================================
# 10. Unify the Target Body order across the three heatmaps
# ============================================================

all_bodies = sorted(
    set(
        mission_target_matrix.columns
    )
    |
    set(
        product_target_matrix.columns
    )
    |
    set(
        sensor_target_matrix.columns
    )
)


mission_target_matrix = (
    mission_target_matrix
    .reindex(
        columns=all_bodies,
        fill_value=0
    )
)

product_target_matrix = (
    product_target_matrix
    .reindex(
        columns=all_bodies,
        fill_value=0
    )
)

sensor_target_matrix = (
    sensor_target_matrix
    .reindex(
        columns=all_bodies,
        fill_value=0
    )
)


# ============================================================
# 11. Create the combined figure
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(
        FIG_W,
        FIG_H
    ),
    gridspec_kw={
        "width_ratios": [
            1,
            1,
            1
        ]
    }
)


# ============================================================
# 12. Heatmap drawing function
# ============================================================

def draw_heatmap(
    ax,
    matrix,
    title,
    ylabel
):

    # If the matrix is empty
    if matrix.empty:

        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            fontsize=10
        )

        ax.axis(
            "off"
        )

        return


    # Maximum value, used to unify the color scale across the three heatmaps
    sns.heatmap(
        matrix,

        ax=ax,

        cmap=HEATMAP_CMAP,

        vmin=0,

        vmax=global_vmax,

        linewidths=0.5,

        linecolor="white",

        annot=False,

        cbar=False
    )


    ax.set_title(
        title,

        loc="left",

        fontsize=TITLE_SIZE,

        fontweight="bold",

        pad=8
    )


    ax.set_xlabel(
        "Target Body",

        fontsize=AXIS_LABEL_SIZE,

        labelpad=6
    )


    ax.set_ylabel(
        ylabel,

        fontsize=AXIS_LABEL_SIZE,

        labelpad=6
    )


    ax.tick_params(
        axis="x",

        rotation=45,

        labelsize=TICK_SIZE,

        pad=3
    )


    ax.tick_params(
        axis="y",

        rotation=0,

        labelsize=TICK_SIZE,

        pad=3
    )


    # Frame
    for spine in ax.spines.values():

        spine.set_visible(
            True
        )

        spine.set_linewidth(
            0.8
        )

        spine.set_edgecolor(
            "#777777"
        )


# ============================================================
# 13. Unified color scale for the three matrices
# ============================================================

global_vmax = max(
    mission_target_matrix.to_numpy().max()
    if not mission_target_matrix.empty
    else 0,

    product_target_matrix.to_numpy().max()
    if not product_target_matrix.empty
    else 0,

    sensor_target_matrix.to_numpy().max()
    if not sensor_target_matrix.empty
    else 0
)


# Guard against a maximum of 0
if global_vmax <= 0:

    global_vmax = 1


# ============================================================
# 14. Draw the Mission heatmap
# ============================================================

draw_heatmap(
    axes[0],

    mission_target_matrix,

    "(a) Mission–Target Body Mapping",

    "Mission"
)


# ============================================================
# 15. Draw the Product heatmap
# ============================================================

draw_heatmap(
    axes[1],

    product_target_matrix,

    "(b) Product–Target Body Mapping",

    "Product"
)


# ============================================================
# 16. Draw the Sensor heatmap
# ============================================================

draw_heatmap(
    axes[2],

    sensor_target_matrix,

    "(c) Sensor–Target Body Mapping",

    "Sensor"
)


# ============================================================
# 17. Add a shared colorbar
# ============================================================

# Use an extra colorbar axes
cbar_ax = fig.add_axes(
    [
        0.92,
        0.18,
        0.015,
        0.64
    ]
)


sm = plt.cm.ScalarMappable(
    cmap=HEATMAP_CMAP,

    norm=plt.Normalize(
        vmin=0,
        vmax=global_vmax
    )
)

sm.set_array([])


cbar = fig.colorbar(
    sm,

    cax=cbar_ax
)


cbar.set_label(
    "Number of Maps",

    fontsize=10,

    labelpad=8
)


cbar.ax.tick_params(
    labelsize=8.5
)


# ============================================================
# 18. Overall title
# ============================================================

fig.suptitle(
    "Mission, Product, and Sensor Distributions across Planetary Bodies",

    fontsize=14,

    fontweight="bold",

    y=0.98
)


# ============================================================
# 19. Layout
# ============================================================

plt.subplots_adjust(
    left=0.055,
    right=0.90,
    top=0.88,
    bottom=0.13,

    wspace=0.30
)


# ============================================================
# 20. Save PNG / PDF / SVG
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


# ============================================================
# 21. Output checks
# ============================================================

print(
    "\n===== Output Check ====="
)


for output_path in [
    OUTPUT_PNG,
    OUTPUT_PDF,
    OUTPUT_SVG
]:

    if os.path.exists(
        output_path
    ):

        print(
            f"✔ {output_path}"
        )

    else:

        print(
            f"✘ File not generated: "
            f"{output_path}"
        )


print(
    "\n✔ Three heatmaps generated successfully."
)


# ============================================================
# 22. Show
# ============================================================

plt.show()