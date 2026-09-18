import os
import re
import warnings
import string

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ============================================================
# 1. Path Configuration
# ============================================================

INPUT_EXCEL = (
    r"path/to/maps-table/"
    r"maps_table_e8.xlsx"
)

OUTPUT_DIR = (
    r"path/to/GMIE_data/"
    r"plot_0816/"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# 2. output path
# ============================================================

FIG13_PNG = os.path.join(
    OUTPUT_DIR,
    "Figure13_Mapping_Coverage_Gap_Heatmap.png"
)

FIG13_PDF = os.path.join(
    OUTPUT_DIR,
    "Figure13_Mapping_Coverage_Gap_Heatmap.pdf"
)

FIG13_SVG = os.path.join(
    OUTPUT_DIR,
    "Figure13_Mapping_Coverage_Gap_Heatmap.svg"
)

FIG14_PNG = os.path.join(
    OUTPUT_DIR,
    "Figure14_Mapping_Gap_by_Body_and_Decade.png"
)

FIG14_PDF = os.path.join(
    OUTPUT_DIR,
    "Figure14_Mapping_Gap_by_Body_and_Decade.pdf"
)

FIG14_SVG = os.path.join(
    OUTPUT_DIR,
    "Figure14_Mapping_Gap_by_Body_and_Decade.svg"
)

# ============================================================
# 3. field configuration
# ============================================================

BODY_COLUMN = "target_body"
DATE_COLUMN = "publication_date"

# Prefer the final normalized scale_label
SCALE_CANDIDATES = [
    "scale_label",
    "scale"
]

# ============================================================
# 4. painting parameters
# ============================================================

DPI = 300

FONT_FAMILY = "Times New Roman"

TITLE_SIZE = 13
AXIS_LABEL_SIZE = 10.5
TICK_SIZE = 9
ANNOTATION_SIZE = 8

# ============================================================
# 5. color
# ============================================================

HEATMAP_CMAP = sns.light_palette(
    "#44AA99",
    as_cmap=True
)

# ============================================================
# 6. Matplotlib style
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

    "text.color": "#2E2E2E",
})


# ============================================================
# 7. tool
# ============================================================

def parse_scale(text):
    """
    Extract the scale denominator N from a 1:N string.

    Supports:
        1:100000
        1:100,000
        1:1 000 000
        1:100000 (full-width colon)
    """

    if pd.isna(text):
        return np.nan

    text = str(text)

    match = re.search(
        r"1\s*[:：]\s*([\d,\s]+)",
        text
    )

    if match is None:
        return np.nan

    num = re.sub(
        r"\D",
        "",
        match.group(1)
    )

    if not num:
        return np.nan

    try:
        return int(num)

    except ValueError:
        return np.nan


def classify_scale(value):
    """
    Final four-level scale classification.

    Large-scale:
        N <= 250,000

    Medium-scale:
        250,000 < N <= 1,000,000

    Small-scale:
        1,000,000 < N <= 5,000,000

    Very small-scale:
        N > 5,000,000
    """

    if pd.isna(value):
        return np.nan

    if value <= 250_000:
        return "Large-scale"

    elif value <= 1_000_000:
        return "Medium-scale"

    elif value <= 5_000_000:
        return "Small-scale"

    else:
        return "Very small-scale"


def get_scale_column(df):
    """
    Automatically detect scale column.
    """

    for col in SCALE_CANDIDATES:

        if col in df.columns:
            print(
                f"✔ Scale column: {col}"
            )

            return col

    raise ValueError(
        "Neither scale_label nor scale field was found."
    )


def save_figure(
        fig,
        png_path,
        pdf_path,
        svg_path
):
    """
    Save publication-quality figures.
    """

    fig.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white"
    )

    fig.savefig(
        svg_path,
        bbox_inches="tight",
        facecolor="white"
    )

    print(
        f"✔ PNG: {png_path}"
    )

    print(
        f"✔ PDF: {pdf_path}"
    )

    print(
        f"✔ SVG: {svg_path}"
    )


# ============================================================
# 8. Read data
# ============================================================

print("=" * 80)

print(
    "3.3.4 Mapping Coverage Gap Analysis"
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
# 9. Field checks
# ============================================================

required_columns = [
    BODY_COLUMN,
    DATE_COLUMN
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "The Excel file is missing the following fields:\n"
        + "\n".join(
            missing_columns
        )
    )

SCALE_COLUMN = get_scale_column(
    df
)

# ============================================================
# 10. Basic preprocessing
# ============================================================

df[DATE_COLUMN] = pd.to_numeric(
    df[DATE_COLUMN],
    errors="coerce"
)

# Keep the same data range as the original 3.3.4
df = df[
    (df[DATE_COLUMN] >= 1960)
    &
    (df[DATE_COLUMN] <= 2026)
    ].copy()

df["scale_value"] = (
    df[SCALE_COLUMN]
    .apply(parse_scale)
)

# Four-class scale_class
df["scale_class"] = (
    df["scale_value"]
    .apply(classify_scale)
)

df[BODY_COLUMN] = (
    df[BODY_COLUMN]
    .astype(str)
    .str.strip()
)

# Drop invalid planetary bodies
invalid_bodies = {
    "",
    "UNKNOWN",
    "OTHER",
    "NONE",
    "NAN"
}

df = df[
    ~df[BODY_COLUMN]
    .str.upper()
    .isin(
        invalid_bodies
    )
].copy()

# decade
df["decade"] = (
        df[DATE_COLUMN]
        // 10
        * 10
)

# ============================================================
# 11. Build the 3.3.4 analysis dataset
# ============================================================

df_gap = df.dropna(
    subset=[
        BODY_COLUMN,
        "scale_class",
        DATE_COLUMN
    ]
).copy()

print(
    f"✔ Valid coverage-gap records: "
    f"{len(df_gap)}"
)

# ============================================================
# 12. Fix the order of the four scale classes
# ============================================================

scale_classes = [
    "Large-scale",
    "Medium-scale",
    "Small-scale",
    "Very small-scale"
]

# ============================================================
# 13. Fix the decade order
# ============================================================

decades = sorted(
    df_gap[
        "decade"
    ]
    .dropna()
    .unique()
)

decades = [
    int(d)
    for d in decades
]

# ============================================================
# 14. Planetary bodies
# ============================================================

bodies = sorted(
    df_gap[
        BODY_COLUMN
    ]
    .unique()
)

print(
    "\n===== Planetary Bodies ====="
)

for body in bodies:
    n = (
        df_gap[
            df_gap[
                BODY_COLUMN
            ] == body
            ]
        .shape[0]
    )

    print(
        f"{body}: n={n}"
    )

print(
    "\n===== Scale Classes ====="
)

for sc in scale_classes:
    count = (
        df_gap[
            df_gap[
                "scale_class"
            ] == sc
            ]
        .shape[0]
    )

    print(
        f"{sc}: n={count}"
    )

# ============================================================
# 15. Tabulate Body × Scale Class × Decade
# ============================================================

count_3d = (
    df_gap
    .groupby(
        [
            BODY_COLUMN,
            "scale_class",
            "decade"
        ]
    )
    .size()
    .reset_index(
        name="count"
    )
)

# ============================================================
# 16. Build the complete three-dimensional grid
# ============================================================

full_index = (
    pd.MultiIndex.from_product(
        [
            bodies,
            scale_classes,
            decades
        ],
        names=[
            BODY_COLUMN,
            "scale_class",
            "decade"
        ]
    )
)

full_df = (
    pd.DataFrame(
        index=full_index
    )
    .reset_index()
)

merged = (
    full_df
    .merge(
        count_3d,
        on=[
            BODY_COLUMN,
            "scale_class",
            "decade"
        ],
        how="left"
    )
)

merged["count"] = (
    merged["count"]
    .fillna(0)
    .astype(int)
)

# ============================================================
# ============================================================
# FIGURE 13
# Coverage rate: Planetary Body × Scale Class
# ============================================================
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "Figure 13: Mapping Coverage Rate by Planetary Body and Scale Class"
)

print(
    "=" * 80
)

# For each body × scale_class:
# number of decades with map coverage / total number of decades
coverage = (
    merged
    .groupby(
        [
            BODY_COLUMN,
            "scale_class"
        ]
    )
    .apply(
        lambda g:
        (
                g["count"] > 0
        ).sum()
        /
        len(decades)
    )
    .reset_index(
        name="coverage_rate"
    )
)

pivot_coverage = (
    coverage
    .pivot(
        index=BODY_COLUMN,
        columns="scale_class",
        values="coverage_rate"
    )
    .reindex(
        index=bodies,
        columns=scale_classes
    )
)

print(
    "\n===== Coverage Rate ====="
)

print(
    pivot_coverage.round(3)
)

fig13, ax13 = plt.subplots(
    figsize=(9, 6.5)
)

sns.heatmap(
    pivot_coverage,

    ax=ax13,

    cmap=HEATMAP_CMAP,

    vmin=0,

    vmax=1,

    annot=True,

    fmt=".2f",

    annot_kws={
        "fontsize": ANNOTATION_SIZE,
        "color": "#222222"
    },

    linewidths=0.8,

    linecolor="white",

    cbar_kws={
        "label": "Coverage Rate",
        "shrink": 0.85
    }
)

ax13.set_title(
    "Figure 13. Mapping Coverage Rate by Planetary Body and Scale Class",

    loc="left",

    fontsize=TITLE_SIZE,

    fontweight="bold",

    pad=10
)

ax13.set_xlabel(
    "Scale Class",
    fontsize=AXIS_LABEL_SIZE,
    labelpad=8
)

ax13.set_ylabel(
    "Planetary Body",
    fontsize=AXIS_LABEL_SIZE,
    labelpad=8
)

ax13.tick_params(
    axis="x",
    rotation=0,
    labelsize=TICK_SIZE
)

ax13.tick_params(
    axis="y",
    rotation=0,
    labelsize=TICK_SIZE
)

for spine in ax13.spines.values():
    spine.set_visible(
        True
    )

    spine.set_linewidth(
        0.8
    )

    spine.set_edgecolor(
        "#777777"
    )

plt.tight_layout()

save_figure(
    fig13,
    FIG13_PNG,
    FIG13_PDF,
    FIG13_SVG
)

plt.show()

# ============================================================
# 17. Output the completely missing combinations
# ============================================================

no_maps = (
    merged[
        merged["count"] == 0
        ]
)

print(
    "\n===== No-map body × scale-class × decade combinations ====="
)

if len(no_maps) > 0:

    print(
        no_maps[
            [
                BODY_COLUMN,
                "scale_class",
                "decade"
            ]
        ]
        .to_string(
            index=False
        )
    )

else:

    print(
        "No complete missing combinations detected."
    )

# ============================================================
# ============================================================
# FIGURE 14
# Mapping Gap per Body
# ============================================================
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "Figure 14: Mapping Gap per Planetary Body"
)

print(
    "=" * 80
)

n_bodies = len(
    bodies
)

n_cols = int(
    np.ceil(
        np.sqrt(
            n_bodies
        )
    )
)

n_rows = int(
    np.ceil(
        n_bodies / n_cols
    )
)

fig14, axes14 = plt.subplots(
    n_rows,
    n_cols,

    figsize=(
        n_cols * 4.2,
        n_rows * 3.4
    ),

    squeeze=False
)

fig14.suptitle(
    "Figure 14. Mapping Coverage Gaps across Time and Scale",

    fontsize=14,

    fontweight="bold",

    y=0.995
)

subplot_labels = list(
    string.ascii_lowercase
)

# Same color scale across all subplots
global_max_count = (
    merged["count"]
    .max()
)

if global_max_count <= 0:
    global_max_count = 1

# ============================================================
# 18. One heatmap per planetary body
# ============================================================

for idx, body in enumerate(
        bodies
):

    row = (
            idx // n_cols
    )

    col = (
            idx % n_cols
    )

    ax = axes14[
        row,
        col
    ]

    subset = merged[
        merged[
            BODY_COLUMN
        ] == body
        ].copy()

    pivot_body = (
        subset
        .pivot(
            index="decade",
            columns="scale_class",
            values="count"
        )
        .reindex(
            index=decades,
            columns=scale_classes,
            fill_value=0
        )
        .fillna(0)
    )

    sns.heatmap(
        pivot_body,

        ax=ax,

        cmap=HEATMAP_CMAP,

        vmin=0,

        vmax=global_max_count,

        annot=True,

        fmt="d",

        annot_kws={
            "fontsize": 8,
            "color": "#222222"
        },

        cbar=False,

        linewidths=0.6,

        linecolor="white"
    )

    ax.set_title(
        f"({subplot_labels[idx]}) {body}",

        loc="left",

        fontsize=11,

        fontweight="bold",

        pad=7
    )

    ax.set_xlabel(
        "Scale Class",
        fontsize=9.5
    )

    ax.set_ylabel(
        "Decade",
        fontsize=9.5
    )

    ax.tick_params(
        axis="x",
        rotation=0,
        labelsize=8
    )

    ax.tick_params(
        axis="y",
        rotation=0,
        labelsize=8
    )

    # --------------------------------------------------------
    # Red border marks missing cells
    # --------------------------------------------------------

    for i in range(
            len(pivot_body)
    ):

        for j in range(
                len(pivot_body.columns)
        ):

            if (
                    pivot_body.iloc[i, j]
                    == 0
            ):
                ax.add_patch(
                    plt.Rectangle(
                        (
                            j,
                            i
                        ),
                        1,
                        1,

                        fill=False,

                        edgecolor="#CC6677",

                        linewidth=1.5
                    )
                )

# ============================================================
# 19. Hide the surplus subplots
# ============================================================

for idx in range(
        len(bodies),
        n_rows * n_cols
):
    row = (
            idx // n_cols
    )

    col = (
            idx % n_cols
    )

    axes14[
        row,
        col
    ].axis(
        "off"
    )

# ============================================================
# 20. Shared colorbar for Figure 14
# ============================================================

cbar_ax = fig14.add_axes(
    [
        0.92,
        0.20,
        0.015,
        0.60
    ]
)

sm = plt.cm.ScalarMappable(
    cmap=HEATMAP_CMAP,

    norm=plt.Normalize(
        vmin=0,
        vmax=global_max_count
    )
)

sm.set_array([])

cbar14 = fig14.colorbar(
    sm,

    cax=cbar_ax
)

cbar14.set_label(
    "Number of Maps",
    fontsize=9.5,
    labelpad=7
)

cbar14.ax.tick_params(
    labelsize=8
)

# ============================================================
# 21. Figure 14 layout
# ============================================================

plt.subplots_adjust(
    left=0.06,
    right=0.90,
    top=0.91,
    bottom=0.08,

    wspace=0.30,

    hspace=0.45
)

save_figure(
    fig14,
    FIG14_PNG,
    FIG14_PDF,
    FIG14_SVG
)

plt.show()

# ============================================================
# 22. Summary: which body × scale class combinations are entirely missing over the study period
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "Overall Missing Coverage by Planetary Body and Scale Class"
)

print(
    "=" * 80
)

for body in bodies:

    for scale_class in scale_classes:

        total_maps = (
            merged[
                (
                        merged[
                            BODY_COLUMN
                        ]
                        == body
                )
                &
                (
                        merged[
                            "scale_class"
                        ]
                        == scale_class
                )
                ]["count"]
            .sum()
        )

        if total_maps == 0:
            print(
                f"⚠ {body} has no maps "
                f"in the {scale_class} category "
                f"across the entire study period."
            )

print(
    "\n✔ Figure 13 and Figure 14 generated successfully."
)