import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency
from scipy.spatial.distance import jensenshannon


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

FIG9_PNG = os.path.join(
    OUTPUT_DIR,
    "Figure9_Institutional_Mapping_Scale.png"
)

FIG9_PDF = os.path.join(
    OUTPUT_DIR,
    "Figure9_Institutional_Mapping_Scale.pdf"
)

FIG9_SVG = os.path.join(
    OUTPUT_DIR,
    "Figure9_Institutional_Mapping_Scale.svg"
)


FIG10_PNG = os.path.join(
    OUTPUT_DIR,
    "Figure10_Planetary_Cartographic_Design.png"
)

FIG10_PDF = os.path.join(
    OUTPUT_DIR,
    "Figure10_Planetary_Cartographic_Design.pdf"
)

FIG10_SVG = os.path.join(
    OUTPUT_DIR,
    "Figure10_Planetary_Cartographic_Design.svg"
)


FIG11_PNG = os.path.join(
    OUTPUT_DIR,
    "Figure11_JSD_Projection_Distributions.png"
)

FIG11_PDF = os.path.join(
    OUTPUT_DIR,
    "Figure11_JSD_Projection_Distributions.pdf"
)

FIG11_SVG = os.path.join(
    OUTPUT_DIR,
    "Figure11_JSD_Projection_Distributions.svg"
)


# ============================================================
# 3. Field configuration
# ============================================================

AGENCY_COLUMN = "agency_norm"
BODY_COLUMN = "target_body"
PROJECTION_COLUMN = "projection_norm"

SCALE_CANDIDATES = [
    "scale_label",
    "scale"
]

DATE_COLUMN = "publication_date"


# ============================================================
# 4. basic parameters
# ============================================================

DPI = 300

FIG_FONT = "Times New Roman"

TITLE_SIZE = 13
AXIS_LABEL_SIZE = 10.5
TICK_SIZE = 9
LEGEND_SIZE = 9
ANNOTATION_SIZE = 8

# Figure 9(d)

MIN_AGENCY_TEMPORAL_COUNT = 20

# Figure 11

MIN_BODY_COUNT_FOR_TEST = 20

# Figure 11

MIN_PROJECTION_COUNT = 50


# ============================================================
# 5. color
# ============================================================

COLORBLIND_SAFE = [
    "#332288",
    "#88CCEE",
    "#44AA99",
    "#117733",
    "#999933",
    "#DDCC77",
    "#CC6677",
    "#882255",
    "#AA4499",
    "#6699CC",
    "#E69F00",
    "#56B4E9",
]


# ============================================================
# scale color
# ============================================================

SCALE_COLORS = {
    "Large-scale": "#332288",
    "Medium-scale": "#44AA99",
    "Small-scale": "#DDCC77",
    "Very small-scale": "#CC6677",
}


# Figure 10 target body colors
BODY_COLORS = [
    "#332288",
    "#117733",
    "#CC6677",
    "#0072B2",
    "#E69F00",
    "#AA4499",
    "#56B4E9",
    "#999933",
]


# Figure 10 projection colors
PROJECTION_COLORS = [
    "#332288",
    "#88CCEE",
    "#44AA99",
    "#117733",
    "#CC6677",
    "#DDCC77",
    "#AA4499",
    "#6699CC",
]


# ============================================================
# 6. Matplotlib
# ============================================================

plt.rcParams.update({
    "font.family": FIG_FONT,

    "font.size": 10,

    "axes.titlesize": TITLE_SIZE,

    "axes.labelsize": AXIS_LABEL_SIZE,

    "xtick.labelsize": TICK_SIZE,

    "ytick.labelsize": TICK_SIZE,

    "legend.fontsize": LEGEND_SIZE,

    "axes.unicode_minus": False,

    "axes.linewidth": 0.8,

    "axes.edgecolor": "#555555",

    "xtick.color": "#555555",

    "ytick.color": "#555555",

    "text.color": "#2E2E2E",

    "pdf.fonttype": 42,

    "ps.fonttype": 42,

    "svg.fonttype": "none",
})


# ============================================================
# 7. tool
# ============================================================

def parse_scale(text):
    """
    Extract N from a 1:N scale string.

    Supports:
        1:100000
        1:100,000
        1:1 000 000
        1:100000 (full-width colon)
        scale 1:1,000,000
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

    try:

        num = re.sub(
            r"\D",
            "",
            match.group(1)
        )

        if not num:
            return np.nan

        return int(num)

    except ValueError:
        return np.nan


def classify_scale(scale_value):
    """
    Map scale denominator to FOUR scale classes.

    Large-scale:
        N <= 250,000

    Medium-scale:
        250,000 < N <= 1,000,000

    Small-scale:
        1,000,000 < N <= 5,000,000

    Very small-scale:
        N > 5,000,000
    """

    if pd.isna(scale_value):
        return np.nan

    if scale_value <= 250_000:

        return "Large-scale"

    elif scale_value <= 1_000_000:

        return "Medium-scale"

    elif scale_value <= 5_000_000:

        return "Small-scale"

    else:

        return "Very small-scale"


def find_scale_column(df):
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
    Save publication-quality PNG/PDF/SVG.
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


def clean_agency_value(text):
    """
    Clean agency names for plotting only.
    Does NOT perform additional semantic normalization.
    """

    if pd.isna(text):
        return ""

    text = str(text).strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


def add_panel_title(
    ax,
    text
):
    """
    Standard panel title.
    """

    ax.set_title(
        text,
        loc="left",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        pad=8
    )


# ============================================================
# 8. read data
# ============================================================

print("=" * 80)
print(
    "Cartographic Parameter Space Analysis"
)
print("=" * 80)

print(
    "\nReading Excel..."
)

df = pd.read_excel(
    INPUT_EXCEL
)

print(
    f"✔ total raw recorded number: {len(df)}"
)


# ============================================================
# 9. field check
# ============================================================

required_columns = [
    AGENCY_COLUMN,
    BODY_COLUMN,
    PROJECTION_COLUMN,
    DATE_COLUMN,
]

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing:

    raise ValueError(
        "Excel Warning: Missing field:\n"
        + "\n".join(missing)
    )


SCALE_COLUMN = find_scale_column(
    df
)


# ============================================================
# 10. Basic data processing
# ============================================================

df[DATE_COLUMN] = pd.to_numeric(
    df[DATE_COLUMN],
    errors="coerce"
)

df["scale_value"] = (
    df[SCALE_COLUMN]
    .apply(parse_scale)
)



# ============================================================

df["scale_class"] = (
    df["scale_value"]
    .apply(classify_scale)
)


df[BODY_COLUMN] = (
    df[BODY_COLUMN]
    .astype(str)
    .str.strip()
)

df[PROJECTION_COLUMN] = (
    df[PROJECTION_COLUMN]
    .astype(str)
    .str.strip()
)

df[AGENCY_COLUMN] = (
    df[AGENCY_COLUMN]
    .astype(str)
    .str.strip()
)


# ============================================================
# Check scale classification result
# ============================================================

print(
    "\n===== Scale Class Distribution ====="
)

print(
    df["scale_class"]
    .value_counts(
        dropna=False
    )
)


# ============================================================
# ============================================================
# FIGURE 9
# Institutional Differences in Cartographic Scale
# ============================================================
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "Figure 9: Institutional Differences in Cartographic Scale"
)

print(
    "=" * 80
)


# ------------------------------------------------------------
#11. Institution-scale data
# ------------------------------------------------------------

df_scale = df.dropna(
    subset=[
        AGENCY_COLUMN,
        "scale_value"
    ]
).copy()


# Split multi-agency entries
df_scale["agency"] = (
    df_scale[
        AGENCY_COLUMN
    ]
    .str.split(";")
)

df_scale = (
    df_scale
    .explode("agency")
)

df_scale["agency"] = (
    df_scale["agency"]
    .apply(clean_agency_value)
)



invalid_agencies = {
    "",
    "UNKNOWN",
    "OTHER",
    "NAN",
    "NONE"
}

df_scale = df_scale[
    ~df_scale[
        "agency"
    ]
    .str.upper()
    .isin(
        invalid_agencies
    )
].copy()



df_scale = (
    df_scale
    .reset_index(
        names="map_row"
    )
)

df_scale = (
    df_scale
    .drop_duplicates(
        subset=[
            "map_row",
            "agency"
        ]
    )
)


print(
    f"✔ Institutional scale records: "
    f"{len(df_scale)}"
)


# ------------------------------------------------------------
# 12. Agency order by median scale denominator
# ------------------------------------------------------------

agency_median = (
    df_scale
    .groupby("agency")[
        "scale_value"
    ]
    .median()
    .sort_values()
)

agency_order = (
    agency_median
    .index
    .tolist()
)

agency_n = (
    df_scale
    .groupby("agency")[
        "scale_value"
    ]
    .count()
)

print(
    f"✔ Institutions represented: "
    f"{len(agency_order)}"
)


# ============================================================
# FIGURE 9(a)
# Institutional scale distribution
# ============================================================

fig9, axes9 = plt.subplots(
    2,
    2,
    figsize=(15, 11)
)


ax = axes9[0, 0]


box_data = []

box_labels = []

for agency in agency_order:

    values = (
        df_scale[
            df_scale[
                "agency"
            ] == agency
        ]["scale_value"]
        .values
    )

    box_data.append(
        values
    )

    box_labels.append(
        f"{agency}\n(n={len(values)})"
    )


bp = ax.boxplot(
    box_data,
    vert=True,
    patch_artist=True,

    showfliers=True,

    medianprops={
        "color": "#332288",
        "linewidth": 2
    },

    whiskerprops={
        "color": "#666666",
        "linewidth": 1
    },

    capprops={
        "color": "#666666",
        "linewidth": 1
    },

    boxprops={
        "facecolor": "#D9EDF7",
        "edgecolor": "#555555",
        "linewidth": 0.8
    },

    flierprops={
        "marker": "o",
        "markersize": 2.5,
        "markerfacecolor": "#999999",
        "markeredgecolor": "#777777",
        "alpha": 0.45
    }
)


ax.set_yscale(
    "log"
)

add_panel_title(
    ax,
    "(a) Scale Denominator Distribution by Institution"
)

ax.set_xlabel(
    "Institution"
)

ax.set_ylabel(
    "Scale Denominator (log scale)"
)

ax.set_xticks(
    np.arange(
        1,
        len(agency_order) + 1
    )
)

ax.set_xticklabels(
    box_labels,
    rotation=45,
    ha="right",
    fontsize=8
)

ax.grid(
    axis="y",
    which="major",
    alpha=0.20,
    linewidth=0.6
)

ax.spines[
    ["top", "right"]
].set_visible(False)


# ============================================================
# FIGURE 9(b)
# Median scale denominator
# ============================================================

ax = axes9[0, 1]


median_values = (
    agency_median.values
)

y_positions = np.arange(
    len(agency_order)
)


ax.barh(
    y_positions,
    median_values,
    color="#6699CC",
    edgecolor="white",
    linewidth=0.5
)

ax.set_xscale(
    "log"
)

add_panel_title(
    ax,
    "(b) Median Mapping Scale Denominator by Institution"
)

ax.set_xlabel(
    "Median Scale Denominator (log scale)"
)

ax.set_ylabel(
    "Institution"
)

ax.set_yticks(
    y_positions
)

ax.set_yticklabels(
    agency_order,
    fontsize=8.5
)

ax.invert_yaxis()

ax.grid(
    axis="x",
    which="major",
    alpha=0.20,
    linewidth=0.6
)

ax.spines[
    ["top", "right"]
].set_visible(False
)

# Add a median label to each bar
for i, value in enumerate(
    median_values
):

    ax.text(
        value * 1.05,
        i,
        f"1:{value:,.0f}",
        va="center",
        fontsize=7.5,
        color="#444444"
    )


# ============================================================
# FIGURE 9(c)
# Scale-class composition
# ============================================================

ax = axes9[1, 0]


scale_dist = (
    df_scale
    .groupby(
        [
            "agency",
            "scale_class"
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)


scale_dist = (
    scale_dist
    .reindex(
        agency_order
    )
    .fillna(0)
)


# ============================================================
# Revised:
# four-class scale classification
# ============================================================

scale_order = [
    "Large-scale",
    "Medium-scale",
    "Small-scale",
    "Very small-scale"
]


scale_dist = (
    scale_dist
    .reindex(
        columns=scale_order,
        fill_value=0
    )
)


scale_prop = (
    scale_dist
    .div(
        scale_dist.sum(
            axis=1
        ),
        axis=0
    )
    *
    100
)


left = np.zeros(
    len(scale_prop)
)


for scale_class in scale_order:

    values = (
        scale_prop[
            scale_class
        ]
        .values
    )

    ax.barh(
        y_positions,
        values,
        left=left,
        color=SCALE_COLORS[
            scale_class
        ],
        edgecolor="white",
        linewidth=0.5,
        label=scale_class
    )

    left += values


add_panel_title(
    ax,
    "(c) Scale-Class Composition across Institutions"
)

ax.set_xlabel(
    "Proportion of Maps (%)"
)

ax.set_ylabel(
    "Institution"
)

ax.set_xlim(
    0,
    100
)

ax.set_yticks(
    y_positions
)

ax.set_yticklabels(
    agency_order,
    fontsize=8.5
)

ax.invert_yaxis()

ax.grid(
    axis="x",
    alpha=0.20,
    linewidth=0.6
)

ax.spines[
    ["top", "right"]
].set_visible(False)

ax.legend(
    title="Scale class",
    loc="lower right",
    frameon=False,
    fontsize=8
)


# ============================================================
# FIGURE 9(d)
# Decadal Evolution of Institutional Mapping Scale
# ============================================================

ax = axes9[1, 1]


temporal_df = df_scale.copy()

temporal_df[DATE_COLUMN] = pd.to_numeric(
    temporal_df[DATE_COLUMN],
    errors="coerce"
)

temporal_df = temporal_df[
    (temporal_df[DATE_COLUMN] >= 1960)
    &
    (temporal_df[DATE_COLUMN] <= 2021)
].copy()


temporal_df["decade"] = (
    temporal_df[DATE_COLUMN] // 10
) * 10


agency_total_counts = (
    temporal_df["agency"]
    .value_counts()
)


temporal_agencies = (
    agency_total_counts[
        agency_total_counts
        >= MIN_AGENCY_TEMPORAL_COUNT
    ]
    .index
    .tolist()
)


temporal_df = temporal_df[
    temporal_df["agency"].isin(
        temporal_agencies
    )
].copy()


temporal_stats = (
    temporal_df
    .groupby(
        [
            "agency",
            "decade"
        ]
    )["scale_value"]
    .agg(
        median="median",
        q1=lambda x:
            x.quantile(0.25),
        q3=lambda x:
            x.quantile(0.75),
        n="count"
    )
    .reset_index()
)


print(
    "\n===== Figure 9(d) Decadal Median Scale ====="
)

print(
    temporal_stats[
        [
            "agency",
            "decade",
            "median",
            "q1",
            "q3",
            "n"
        ]
    ]
    .sort_values(
        [
            "agency",
            "decade"
        ]
    )
    .to_string(
        index=False
    )
)


print(
    "\n===== Figure 9(d) Temporal Agencies ====="
)

for agency in temporal_agencies:

    print(
        f"{agency}: "
        f"n={agency_total_counts[agency]}"
    )


temporal_colors = {}

for i, agency in enumerate(
    temporal_agencies
):

    temporal_colors[
        agency
    ] = COLORBLIND_SAFE[
        i % len(
            COLORBLIND_SAFE
        )
    ]


for agency in temporal_agencies:

    sub = (
        temporal_stats[
            temporal_stats[
                "agency"
            ] == agency
        ]
        .sort_values(
            "decade"
        )
    )

    if sub.empty:
        continue

    x = (
        sub["decade"]
        .astype(int)
        .values
    )

    median = (
        sub["median"]
        .values
    )

    q1 = (
        sub["q1"]
        .values
    )

    q3 = (
        sub["q3"]
        .values
    )

    color = temporal_colors[
        agency
    ]


    ax.fill_between(
        x,
        q1,
        q3,
        color=color,
        alpha=0.08,
        linewidth=0
    )


    ax.plot(
        x,
        median,

        color=color,

        linewidth=1.8,

        marker="o",

        markersize=5,

        markerfacecolor="white",

        markeredgecolor=color,

        markeredgewidth=1.2,

        label=agency,

        zorder=5
    )


ax.set_yscale(
    "log"
)


all_decades = np.arange(
    1960,
    2021,
    10
)

ax.set_xticks(
    all_decades
)

ax.set_xticklabels(
    [
        f"{d}s"
        for d in all_decades
    ]
)

ax.set_xlim(
    1958,
    2023
)


ax.set_title(
    "(d) Decadal Evolution of Institutional Mapping Scale",
    loc="left",
    fontsize=TITLE_SIZE,
    fontweight="bold",
    pad=10
)

ax.set_xlabel(
    "Decade"
)

ax.set_ylabel(
    "Median Scale Denominator (log scale)"
)


ax.grid(
    axis="y",
    which="major",
    alpha=0.20,
    linewidth=0.6
)

ax.grid(
    axis="x",
    which="major",
    alpha=0.08,
    linewidth=0.5
)


ax.spines[
    ["top", "right"]
].set_visible(False)


ax.legend(
    title="Institution",

    loc="upper left",

    bbox_to_anchor=(
        1.01,
        1.00
    ),

    frameon=False,

    fontsize=7.5,

    title_fontsize=8.5
)


# ============================================================
# Figure 9 overall layout
# ============================================================

fig9.suptitle(
    "Figure 9. Institutional Differences and Temporal Evolution in Mapping Scale",
    fontsize=14,
    fontweight="bold",
    y=0.995
)


fig9.tight_layout(
    rect=[
        0,
        0,
        0.87,
        0.965
    ],
    h_pad=2.0,
    w_pad=1.5
)

fig9.subplots_adjust(
    hspace=0.42
)


save_figure(
    fig9,
    FIG9_PNG,
    FIG9_PDF,
    FIG9_SVG
)

plt.show()


# ============================================================
# ============================================================
# FIGURE 10
# Cartographic Design Patterns across Planetary Bodies
# ============================================================
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "Figure 10: Planetary Cartographic Design Patterns"
)

print(
    "=" * 80
)


body_scale_data = df.dropna(
    subset=[
        BODY_COLUMN,
        "scale_class"
    ]
).copy()


body_projection_data = df.dropna(
    subset=[
        BODY_COLUMN,
        PROJECTION_COLUMN
    ]
).copy()


body_counts = (
    df[
        df[BODY_COLUMN]
        .notna()
    ][
        BODY_COLUMN
    ]
    .value_counts()
)


body_order = (
    body_counts
    .sort_values()
    .index
    .tolist()
)


print(
    "\nPlanetary bodies:"
)

for body in reversed(
    body_order
):

    print(
        f"  {body}: "
        f"n={body_counts[body]}"
    )


# ============================================================
# FIGURE 10(a)
# Projection composition
# ============================================================

projection_counts = (
    body_projection_data
    .groupby(
        [
            BODY_COLUMN,
            PROJECTION_COLUMN
        ]
    )
    .size()
    .reset_index(
        name="count"
    )
)


global_projection_counts = (
    projection_counts
    .groupby(
        PROJECTION_COLUMN
    )["count"]
    .sum()
    .sort_values(
        ascending=False
    )
)


TOP_PROJECTION_NUMBER = 5

top_projection_types = (
    global_projection_counts
    .head(
        TOP_PROJECTION_NUMBER
    )
    .index
    .tolist()
)


projection_counts[
    "projection_group"
] = (
    projection_counts[
        PROJECTION_COLUMN
    ]
    .apply(
        lambda x:
        x
        if x in top_projection_types
        else "Other"
    )
)


projection_counts = (
    projection_counts
    .groupby(
        [
            BODY_COLUMN,
            "projection_group"
        ]
    )["count"]
    .sum()
    .reset_index()
)


projection_counts[
    "proportion"
] = (
    projection_counts
    .groupby(
        BODY_COLUMN
    )["count"]
    .transform(
        lambda x:
        x / x.sum()
    )
    *
    100
)


projection_order = (
    top_projection_types
    +
    (
        ["Other"]
        if "Other"
        in projection_counts[
            "projection_group"
        ].values
        else []
    )
)


projection_color_dict = {}

for i, proj in enumerate(
    projection_order
):

    projection_color_dict[
        proj
    ] = PROJECTION_COLORS[
        i % len(
            PROJECTION_COLORS
        )
    ]


# ============================================================
# FIGURE 10(b)
# Scale-class composition
# ============================================================

scale_counts_body = (
    body_scale_data
    .groupby(
        [
            BODY_COLUMN,
            "scale_class"
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)


scale_counts_body = (
    scale_counts_body
    .reindex(
        body_order
    )
    .fillna(0)
)


# ============================================================
# Revised:
# four-class scale classification
# ============================================================

scale_counts_body = (
    scale_counts_body
    .reindex(
        columns=scale_order,
        fill_value=0
    )
)


scale_prop_body = (
    scale_counts_body
    .div(
        scale_counts_body.sum(
            axis=1
        ),
        axis=0
    )
    *
    100
)


# Print the Figure 10(b) values for easy cross-checking against the manuscript
print(
    "\n===== Figure 10(b) Scale-Class Composition (%) ====="
)

print(
    scale_prop_body.round(2)
)


# ============================================================
# Figure 10 canvas
# ============================================================

fig10, axes10 = plt.subplots(
    1,
    2,
    figsize=(15, 8)
)


# ============================================================
# Figure 10(a)
# ============================================================

ax = axes10[0]

body_y = np.arange(
    len(body_order)
)

left = np.zeros(
    len(body_order)
)


for projection in projection_order:

    values = []

    for body in body_order:

        row = projection_counts[
            (
                projection_counts[
                    BODY_COLUMN
                ] == body
            )
            &
            (
                projection_counts[
                    "projection_group"
                ] == projection
            )
        ]

        if len(row) > 0:

            values.append(
                row[
                    "proportion"
                ].iloc[0]
            )

        else:

            values.append(
                0.0
            )

    values = np.asarray(
        values
    )

    ax.barh(
        body_y,
        values,
        left=left,
        color=projection_color_dict[
            projection
        ],
        edgecolor="white",
        linewidth=0.5,
        label=projection
    )

    left += values


add_panel_title(
    ax,
    "(a) Projection Composition by Planetary Body"
)

ax.set_xlabel(
    "Proportion of Maps (%)"
)

ax.set_ylabel(
    "Planetary Body"
)

ax.set_xlim(
    0,
    100
)

ax.set_yticks(
    body_y
)

ax.set_yticklabels(
    [
        f"{body} (n={body_counts[body]})"
        for body in body_order
    ],
    fontsize=9
)

ax.invert_yaxis()

ax.grid(
    axis="x",
    alpha=0.20,
    linewidth=0.6
)

ax.spines[
    ["top", "right"]
].set_visible(False)

ax.legend(
    title="Projection",
    loc="upper left",
    bbox_to_anchor=(
        1.01,
        1
    ),
    frameon=False,
    fontsize=8
)


# ============================================================
# Figure 10(b)
# ============================================================

ax = axes10[1]

left = np.zeros(
    len(body_order)
)


for scale_class in scale_order:

    values = (
        scale_prop_body[
            scale_class
        ]
        .values
    )

    ax.barh(
        body_y,
        values,
        left=left,
        color=SCALE_COLORS[
            scale_class
        ],
        edgecolor="white",
        linewidth=0.5,
        label=scale_class
    )

    left += values


add_panel_title(
    ax,
    "(b) Scale-Class Composition by Planetary Body"
)

ax.set_xlabel(
    "Proportion of Maps (%)"
)

ax.set_ylabel(
    "Planetary Body"
)

ax.set_xlim(
    0,
    100
)

ax.set_yticks(
    body_y
)

ax.set_yticklabels(
    [
        f"{body} (n={body_counts[body]})"
        for body in body_order
    ],
    fontsize=9
)

ax.invert_yaxis()

ax.grid(
    axis="x",
    alpha=0.20,
    linewidth=0.6
)

ax.spines[
    ["top", "right"]
].set_visible(False)

ax.legend(
    title="Scale class",
    loc="upper left",
    bbox_to_anchor=(
        1.01,
        1
    ),
    frameon=False,
    fontsize=8
)


fig10.suptitle(
    "Figure 10. Cartographic Design Patterns across Planetary Bodies",
    fontsize=14,
    fontweight="bold",
    y=0.995
)


fig10.tight_layout(
    rect=[
        0,
        0,
        0.84,
        0.97
    ]
)


save_figure(
    fig10,
    FIG10_PNG,
    FIG10_PDF,
    FIG10_SVG
)

plt.show()


# ============================================================
# ============================================================
# FIGURE 11
# Jensen–Shannon Divergence
# ============================================================
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "Figure 11: Jensen–Shannon Divergence"
)

print(
    "=" * 80
)


projection_test_data = df.dropna(
    subset=[
        BODY_COLUMN,
        PROJECTION_COLUMN
    ]
).copy()


body_counts_test = (
    projection_test_data[
        BODY_COLUMN
    ]
    .value_counts()
)


valid_bodies = (
    body_counts_test[
        body_counts_test
        >= MIN_BODY_COUNT_FOR_TEST
    ]
    .index
    .tolist()
)


projection_test_data = (
    projection_test_data[
        projection_test_data[
            BODY_COLUMN
        ].isin(
            valid_bodies
        )
    ]
    .copy()
)


print(
    "\nBodies included in statistical test:"
)

print(
    valid_bodies
)


# ============================================================
# Projection grouping
# ============================================================

raw_projection_counts = (
    projection_test_data[
        PROJECTION_COLUMN
    ]
    .value_counts()
    .sort_values(
        ascending=False
    )
)


# Categories retained for the statistical test: those with at least
# MIN_PROJECTION_COUNT maps; rarer projections are merged into "Other"
# (matches the rule described in the SI, section S9.8).
TOP3_PROJECTIONS = (
    raw_projection_counts[
        raw_projection_counts >= MIN_PROJECTION_COUNT
    ]
    .index
    .tolist()
)


projection_test_data[
    "projection_group"
] = (
    projection_test_data[
        PROJECTION_COLUMN
    ]
    .apply(
        lambda x:
        x
        if x in TOP3_PROJECTIONS
        else "Other"
    )
)


projection_order_test = (
    TOP3_PROJECTIONS
    + ["Other"]
)


print(
    "\nFinal projection categories for statistical test:"
)

print(
    projection_order_test
)


# ============================================================
# Contingency table
# ============================================================

contingency = pd.crosstab(
    projection_test_data[
        BODY_COLUMN
    ],
    projection_test_data[
        "projection_group"
    ]
)

print(
    "\nContingency table:"
)

print(
    contingency
)


# ============================================================
# Chi-square test
# ============================================================

chi2, p, dof, expected = (
    chi2_contingency(
        contingency
    )
)

n = contingency.values.sum()

rows, cols = contingency.shape

phi2 = chi2 / n

cramer_v = np.sqrt(
    phi2
    /
    min(
        cols - 1,
        rows - 1
    )
)


expected_lt5 = (
    expected < 5
).sum()

expected_lt1 = (
    expected < 1
).sum()

total_cells = (
    expected.size
)


print(
    "\n===== Chi-square test ====="
)

print(
    f"Chi-square = {chi2:.4f}"
)

print(
    f"df = {dof}"
)

print(
    f"p = {p:.6e}"
)

print(
    f"Cramer's V = {cramer_v:.4f}"
)

print(
    f"Minimum expected frequency = "
    f"{expected.min():.4f}"
)

print(
    f"Expected < 5: "
    f"{expected_lt5}/{total_cells} "
    f"({100 * expected_lt5 / total_cells:.1f}%)"
)

print(
    f"Expected < 1: "
    f"{expected_lt1}/{total_cells} "
    f"({100 * expected_lt1 / total_cells:.1f}%)"
)


# ============================================================
# Probability distributions
# ============================================================

prob_matrix = []

test_body_order = sorted(
    valid_bodies
)

for body in test_body_order:

    subset = (
        projection_test_data[
            projection_test_data[
                BODY_COLUMN
            ] == body
        ]
    )

    counts = (
        subset[
            "projection_group"
        ]
        .value_counts()
        .reindex(
            projection_order_test,
            fill_value=0
        )
    )

    probabilities = (
        counts
        /
        counts.sum()
    )

    prob_matrix.append(
        probabilities.values
    )


prob_df = pd.DataFrame(
    prob_matrix,
    index=test_body_order,
    columns=projection_order_test
)


# ============================================================
# JSD matrix
# ============================================================

n_bodies = len(
    test_body_order
)

js_matrix = np.zeros(
    (
        n_bodies,
        n_bodies
    )
)


for i in range(
    n_bodies
):

    for j in range(
        i + 1,
        n_bodies
    ):

        js = jensenshannon(
            prob_df.iloc[i],
            prob_df.iloc[j],
            base=2
        )

        js_matrix[
            i,
            j
        ] = js

        js_matrix[
            j,
            i
        ] = js


js_df = pd.DataFrame(
    js_matrix,
    index=test_body_order,
    columns=test_body_order
)


print(
    "\n===== Jensen-Shannon divergence matrix ====="
)

print(
    js_df.round(3)
)


# ============================================================
# Figure 11
# ============================================================

fig11, ax = plt.subplots(
    figsize=(8, 7)
)


max_js = max(
    0.5,
    js_matrix.max()
)


image = ax.imshow(
    js_matrix,

    cmap="YlGnBu",

    vmin=0,

    vmax=max_js,

    aspect="equal"
)


add_panel_title(
    ax,
    "Jensen–Shannon Divergence of Projection Distributions"
)


ax.set_xlabel(
    "Planetary Body"
)

ax.set_ylabel(
    "Planetary Body"
)


ax.set_xticks(
    np.arange(
        n_bodies
    )
)

ax.set_yticks(
    np.arange(
        n_bodies
    )
)

ax.set_xticklabels(
    [
        f"{body}\n(n={body_counts_test[body]})"
        for body in test_body_order
    ],
    fontsize=9
)

ax.set_yticklabels(
    [
        f"{body}\n(n={body_counts_test[body]})"
        for body in test_body_order
    ],
    fontsize=9
)


for i in range(
    n_bodies
):

    for j in range(
        n_bodies
    ):

        value = js_matrix[
            i,
            j
        ]

        ax.text(
            j,
            i,
            f"{value:.3f}",
            ha="center",
            va="center",
            fontsize=9,
            color="#222222"
        )


cbar = fig11.colorbar(
    image,
    ax=ax,
    fraction=0.045,
    pad=0.04
)

cbar.set_label(
    "Jensen–Shannon divergence",
    fontsize=10
)


fig11.suptitle(
    "Figure 11. Pairwise Dissimilarity of Projection Distributions among Planetary Bodies",
    fontsize=14,
    fontweight="bold",
    y=0.98
)


fig11.tight_layout(
    rect=[
        0,
        0,
        0.98,
        0.94
    ]
)


save_figure(
    fig11,
    FIG11_PNG,
    FIG11_PDF,
    FIG11_SVG
)

plt.show()


# ============================================================
# 21. Output summary
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "ALL FIGURES 9–11 GENERATED SUCCESSFULLY"
)

print(
    "=" * 80
)

print(
    "\nFigure 9:"
)

print(
    FIG9_PNG
)

print(
    FIG9_PDF
)

print(
    FIG9_SVG
)

print(
    "\nFigure 10:"
)

print(
    FIG10_PNG
)

print(
    FIG10_PDF
)

print(
    FIG10_SVG
)

print(
    "\nFigure 11:"
)

print(
    FIG11_PNG
)

print(
    FIG11_PDF
)

print(
    FIG11_SVG
)

print(
    "\n✔ Institutional scale analysis"
)

print(
    "✔ Planetary projection/scale analysis"
)

print(
    "✔ Chi-square + Cramer's V"
)

print(
    "✔ Jensen–Shannon divergence"
)

print(
    "✔ Four-level scale classification:"
)

print(
    "   Large-scale: N <= 250,000"
)

print(
    "   Medium-scale: 250,000 < N <= 1,000,000"
)

print(
    "   Small-scale: 1,000,000 < N <= 5,000,000"
)

print(
    "   Very small-scale: N > 5,000,000"
)

print(
    "✔ Scale block position analysis removed"
)

print(
    "✔ Publication-quality PNG/PDF/SVG"
)