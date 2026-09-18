# Paper figure mapping: panels from this script correspond to main-text
# Figure 4(a)(b), Figure 9(a)(b) and Figure 10(a) - see README, section
# "Script-to-Figure Map".
import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency
from scipy.spatial.distance import jensenshannon


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


# ============================================================
# 2. Output files
# ============================================================

FIG4_PNG = os.path.join(
    OUTPUT_DIR,
    "Figure4_Temporal_Evolution.png"
)

FIG4_PDF = os.path.join(
    OUTPUT_DIR,
    "Figure4_Temporal_Evolution.pdf"
)

FIG4_SVG = os.path.join(
    OUTPUT_DIR,
    "Figure4_Temporal_Evolution.svg"
)


FIG5_PNG = os.path.join(
    OUTPUT_DIR,
    "Figure5_Institutional_Scale_Differences.png"
)

FIG5_PDF = os.path.join(
    OUTPUT_DIR,
    "Figure5_Institutional_Scale_Differences.pdf"
)

FIG5_SVG = os.path.join(
    OUTPUT_DIR,
    "Figure5_Institutional_Scale_Differences.svg"
)


FIG6_PNG = os.path.join(
    OUTPUT_DIR,
    "Figure6_Projection_Preferences.png"
)

FIG6_PDF = os.path.join(
    OUTPUT_DIR,
    "Figure6_Projection_Preferences.pdf"
)

FIG6_SVG = os.path.join(
    OUTPUT_DIR,
    "Figure6_Projection_Preferences.svg"
)


# ============================================================
# 3. Global plotting parameters
# ============================================================

FONT_FAMILY = "Times New Roman"

TITLE_SIZE = 13
AXIS_LABEL_SIZE = 11
TICK_SIZE = 9
LEGEND_SIZE = 9
ANNOTATION_SIZE = 8

DPI = 300


# ============================================================
# 4. Colorblind-safe palette
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


# Figure 4c: target body colors
BODY_COLORS = [
    "#332288",
    "#117733",
    "#CC6677",
    "#0072B2",
    "#E69F00",
    "#AA4499",
]


# Figure 5: scale classes (four-level scheme, as defined in the paper)
SCALE_CLASS_COLORS = {
    "Large": "#332288",
    "Medium": "#44AA99",
    "Small": "#DDCC77",
    "Very small": "#CC6677",
}


# Figure 6: projection colors
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
# 5. Global Matplotlib settings
# ============================================================

plt.rcParams.update({
    "font.family": FONT_FAMILY,

    "font.size": 10,

    "axes.titlesize": TITLE_SIZE,

    "axes.labelsize": AXIS_LABEL_SIZE,

    "xtick.labelsize": TICK_SIZE,

    "ytick.labelsize": TICK_SIZE,

    "legend.fontsize": LEGEND_SIZE,

    "axes.unicode_minus": False,

    "pdf.fonttype": 42,

    "ps.fonttype": 42,

    "svg.fonttype": "none",

    "axes.linewidth": 0.8,

    "axes.edgecolor": "#555555",

    "xtick.color": "#555555",

    "ytick.color": "#555555",

    "text.color": "#2E2E2E",
})


# ============================================================
# 6. Utility functions
# ============================================================

def parse_scale(x):
    """
    Extract N from a 1:N scale string.
    Supports 1:100000, 1:100,000, and the full-width colon variant.
    """

    if pd.isna(x):
        return np.nan

    text = str(x)

    match = re.search(
        r"1\s*[:：]\s*([\d,]+)",
        text
    )

    if match:

        try:

            return int(
                match.group(1)
                .replace(",", "")
            )

        except ValueError:

            return np.nan

    return np.nan


def get_scale_column(df):
    """
    Prefer scale_label; fall back to scale.
    """

    candidates = [
        "scale_label",
        "scale"
    ]

    for col in candidates:

        if col in df.columns:

            print(
                f"✔ Scale field in use: {col}"
            )

            return col

    raise ValueError(
        "Neither scale_label nor scale field was found."
    )


def normalize_projection(text):
    """
    Normalize the projection text.
    """

    if pd.isna(text):
        return np.nan

    text = str(text).strip()

    if not text:
        return np.nan

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


def classify_scale(value):
    """
    Scale class (four-level scheme).

    Large:
        denominator <= 250,000

    Medium:
        250,000 < denominator <= 1,000,000

    Small:
        1,000,000 < denominator <= 5,000,000

    Very small:
        denominator > 5,000,000
    """

    if pd.isna(value):
        return np.nan

    if value <= 250_000:

        return "Large"

    elif value <= 1_000_000:

        return "Medium"

    elif value <= 5_000_000:

        return "Small"

    else:

        return "Very small"


def choose_top_categories(
    series,
    top_n=5,
    other_label="Other"
):
    """
    Return the top_n categories; the rest are merged into Other.
    """

    counts = (
        series
        .value_counts()
    )

    top_categories = (
        counts
        .head(top_n)
        .index
        .tolist()
    )

    return top_categories


def save_figure(
    fig,
    png_path,
    pdf_path,
    svg_path
):

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
# 7. Read data
# ============================================================

print("=" * 80)
print("Planetary Geological Mapping Statistical Analysis")
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
# 8. Required-field checks
# ============================================================

required_columns = [
    "publication_date",
    "target_body",
    "projection_norm",
    "agency_norm",
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing the following fields:\n"
        + "\n".join(missing_columns)
    )


SCALE_COLUMN = get_scale_column(
    df
)


# ============================================================
# 9. Basic preprocessing
# ============================================================

# publication year
df["publication_date"] = pd.to_numeric(
    df["publication_date"],
    errors="coerce"
)

# Keep plausible years
df = df[
    (df["publication_date"] >= 1960)
    &
    (df["publication_date"] <= 2021)
].copy()

# year
df["year"] = (
    df["publication_date"]
    .round()
    .astype("Int64")
)

# decade
df["decade"] = (
    (df["year"] // 10) * 10
)

# scale
df["scale_value"] = (
    df[SCALE_COLUMN]
    .apply(parse_scale)
)

df["scale_class"] = (
    df["scale_value"]
    .apply(classify_scale)
)

# projection
df["projection_norm"] = (
    df["projection_norm"]
    .apply(normalize_projection)
)

# agency
df["agency_norm"] = (
    df["agency_norm"]
    .astype(str)
    .str.strip()
)


print(
    f"✔ Records with a valid time range: {len(df)}"
)


# ============================================================
# ============================================================
# FIGURE 4
# Temporal Evolution of Planetary Geological Mapping
# ============================================================
# ============================================================

print(
    "\n"
    + "=" * 80
)
print(
    "Figure 4: Temporal Evolution"
)
print(
    "=" * 80
)


# ------------------------------------------------------------
# Figure 4(a)
# Annual map count
# ------------------------------------------------------------

annual_count = (
    df.dropna(
        subset=["year"]
    )
    .groupby("year")
    .size()
)

annual_years = (
    annual_count
    .index
    .astype(int)
)

annual_values = (
    annual_count
    .values
)


# ------------------------------------------------------------
# Figure 4(b)
# Annual median + IQR + sample size
# ------------------------------------------------------------

scale_yearly = (
    df.dropna(
        subset=[
            "year",
            "scale_value"
        ]
    )
    .groupby("year")[
        "scale_value"
    ]
    .agg(
        median="median",
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75),
        n="count"
    )
    .reset_index()
)
# ------------------------------------------------------------
# Decadal scale statistics for manuscript reporting
# ------------------------------------------------------------

def get_year_of_min(group):
    idx = group["scale_value"].idxmin()
    return int(group.loc[idx, "year"])


def get_year_of_max(group):
    idx = group["scale_value"].idxmax()
    return int(group.loc[idx, "year"])


scale_decadal = (
    df.dropna(
        subset=[
            "decade",
            "scale_value",
            "year"
        ]
    )
    .groupby("decade")
    .apply(
        lambda g: pd.Series({
            "median": g["scale_value"].median(),
            "q1": g["scale_value"].quantile(0.25),
            "q3": g["scale_value"].quantile(0.75),
            "n": g["scale_value"].count(),

            "minimum": g["scale_value"].min(),
            "minimum_year": get_year_of_min(g),

            "maximum": g["scale_value"].max(),
            "maximum_year": get_year_of_max(g)
        })
    )
    .reset_index()
)

scale_decadal["n"] = (
    scale_decadal["n"]
    .astype(int)
)

print(
    "\n"
    + "=" * 100
)

print(
    "Decadal Scale Statistics for Manuscript Reporting"
)

print(
    "=" * 100
)

print(
    scale_decadal.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Output that can be copied directly into the manuscript
# ------------------------------------------------------------

print(
    "\n===== Manuscript-ready decadal results ====="
)

for _, row in scale_decadal.iterrows():

    decade = int(
        row["decade"]
    )

    median = int(
        round(
            row["median"]
        )
    )

    minimum = int(
        round(
            row["minimum"]
        )
    )

    minimum_year = int(
        row["minimum_year"]
    )

    maximum = int(
        round(
            row["maximum"]
        )
    )

    maximum_year = int(
        row["maximum_year"]
    )

    n = int(
        row["n"]
    )

    print(
        f"{decade}s: "
        f"median=1:{median:,}; "
        f"min=1:{minimum:,} ({minimum_year}); "
        f"max=1:{maximum:,} ({maximum_year}); "
        f"n={n}"
    )


# ------------------------------------------------------------
# Figure 4(c)
# Decadal target-body composition
# ------------------------------------------------------------

body_counts = (
    df.dropna(
        subset=[
            "decade",
            "target_body"
        ]
    )
    .groupby(
        ["decade", "target_body"]
    )
    .size()
    .reset_index(
        name="count"
    )
)

top_bodies = choose_top_categories(
    body_counts["target_body"],
    top_n=5
)

body_counts["body_group"] = (
    body_counts["target_body"]
    .apply(
        lambda x:
        x
        if x in top_bodies
        else "Other"
    )
)

body_composition = (
    body_counts
    .groupby(
        ["decade", "body_group"]
    )["count"]
    .sum()
    .reset_index()
)

body_composition["proportion"] = (
    body_composition
    .groupby("decade")["count"]
    .transform(
        lambda x:
        x / x.sum()
    )
)


decades = sorted(
    body_composition[
        "decade"
    ].dropna().unique()
)


body_order = (
    top_bodies
    + ["Other"]
)

body_colors = {
    body: BODY_COLORS[i % len(BODY_COLORS)]
    for i, body in enumerate(body_order)
}


# ------------------------------------------------------------
# Figure 4 canvas
# ------------------------------------------------------------

fig4, axes4 = plt.subplots(
    3,
    1,
    figsize=(10, 11),
    gridspec_kw={
        "height_ratios": [
            1,
            1.5,
            1
        ]
    }
)


# ============================================================
# Figure 4(a)
# ============================================================

ax = axes4[0]

ax.bar(
    annual_years,
    annual_values,
    width=0.8,
    color="#6699CC",
    edgecolor="white",
    linewidth=0.3
)

ax.set_title(
    "(a) Annual Number of Published Maps",
    loc="left",
    pad=8,
    fontweight="bold"
)

ax.set_xlabel(
    "Publication Year"
)

ax.set_ylabel(
    "Number of Maps"
)

ax.grid(
    axis="y",
    alpha=0.2,
    linewidth=0.6
)

ax.spines[
    ["top", "right"]
].set_visible(False)


# ============================================================
# Figure 4(b)
# ============================================================

ax = axes4[1]

years = (
    scale_yearly["year"]
    .astype(int)
    .values
)

median_values = (
    scale_yearly["median"]
    .values
)

q1_values = (
    scale_yearly["q1"]
    .values
)

q3_values = (
    scale_yearly["q3"]
    .values
)

n_values = (
    scale_yearly["n"]
    .values
)


# IQR
ax.fill_between(
    years,
    q1_values,
    q3_values,
    color="#88CCEE",
    alpha=0.35,
    label="Interquartile range (IQR)"
)

# Median
ax.plot(
    years,
    median_values,
    color="#332288",
    linewidth=2.0,
    marker="o",
    markersize=3.5,
    label="Annual median"
)


# Marker size = annual sample size
# ============================================================
# Sample size encoded by marker size
# ============================================================

marker_sizes = (
    25
    +
    np.sqrt(n_values) * 8
)

ax.scatter(
    years,
    median_values,
    s=marker_sizes,
    facecolor="white",
    edgecolor="#332288",
    linewidth=1.0,
    zorder=5
)


ax.set_yscale(
    "log"
)

ax.set_title(
    "(b) Annual Median Map Scale Denominator",
    loc="left",
    pad=8,
    fontweight="bold"
)

ax.set_xlabel(
    "Publication Year"
)

ax.set_ylabel(
    "Scale Denominator (log scale)"
)

ax.grid(
    axis="y",
    which="major",
    alpha=0.2,
    linewidth=0.6
)

ax.spines[
    ["top", "right"]
].set_visible(False)


# ============================================================
# Sample-size legend
# Marker size represents annual sample size
# ============================================================

sample_legend_values = [
    1,
    5,
    20
]

sample_legend_handles = []

for n_example in sample_legend_values:

    size_example = (
        25
        +
        np.sqrt(n_example) * 8
    )

    sample_legend_handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="#332288",
            markeredgewidth=1.0,
            markersize=np.sqrt(
                size_example
            ),
            label=f"n = {n_example}"
        )
    )


# Original IQR + median legend
iqr_handle = plt.Line2D(
    [0],
    [0],
    color="#88CCEE",
    linewidth=8,
    alpha=0.35,
    label="Interquartile range (IQR)"
)

median_handle = plt.Line2D(
    [0],
    [0],
    color="#332288",
    linewidth=2.0,
    marker="o",
    markersize=4,
    label="Annual median"
)


ax.legend(
    handles=[
        median_handle,
        iqr_handle,
        *sample_legend_handles
    ],
    loc="upper right",
    frameon=False,
    title="Sample size encoded by marker size",
    title_fontsize=8.5
)


# Only annotate n for low-sample years
for x, y, n in zip(
    years,
    median_values,
    n_values
):

    if n <= 3:

        ax.annotate(
            f"n={int(n)}",
            xy=(x, y),
            xytext=(0, 8),
            textcoords="offset points",
            fontsize=7.5,
            ha="center",
            color="#555555"
        )


ax.legend(
    loc="upper right",
    frameon=False
)


# ============================================================
# Figure 4(c)
# ============================================================

ax = axes4[2]

bottom = np.zeros(
    len(decades)
)

for body in body_order:

    subset = (
        body_composition[
            body_composition[
                "body_group"
            ] == body
        ]
    )

    values = []

    for decade in decades:

        row = subset[
            subset[
                "decade"
            ] == decade
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

    ax.bar(
        decades,
        values * 100,
        bottom=bottom * 100,
        width=6.5,
        color=body_colors[body],
        edgecolor="white",
        linewidth=0.4,
        label=body
    )

    bottom += values


ax.set_title(
    "(c) Decadal Composition of Target Bodies",
    loc="left",
    pad=8,
    fontweight="bold"
)

ax.set_xlabel(
    "Decade"
)

ax.set_ylabel(
    "Share of Maps (%)"
)

ax.set_ylim(
    0,
    100
)

ax.set_xticks(
    decades
)

ax.grid(
    axis="y",
    alpha=0.2,
    linewidth=0.6
)

ax.spines[
    ["top", "right"]
].set_visible(False)

ax.legend(
    title="Target Body",
    loc="upper left",
    bbox_to_anchor=(1.01, 1.0),
    frameon=False
)


fig4.suptitle(
    "Figure 4. Temporal Evolution of Planetary Geological Mapping",
    fontsize=14,
    fontweight="bold",
    y=0.995
)

fig4.tight_layout(
    rect=[
        0,
        0,
        0.88,
        0.98
    ]
)


save_figure(
    fig4,
    FIG4_PNG,
    FIG4_PDF,
    FIG4_SVG
)

plt.show()


# ============================================================
# ============================================================
# FIGURE 5
# Institutional Differences in Cartographic Scale
# ============================================================
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "Figure 5: Institutional Differences"
)

print(
    "=" * 80
)


# ------------------------------------------------------------
# Data
# ------------------------------------------------------------

df_scale = df.dropna(
    subset=[
        "agency_norm",
        "scale_value"
    ]
).copy()


# Drop invalid agencies
invalid_agencies = {
    "",
    "UNKNOWN",
    "OTHER",
    "NAN"
}

df_scale = df_scale[
    ~df_scale[
        "agency_norm"
    ]
    .str.upper()
    .isin(
        invalid_agencies
    )
].copy()


# Split multi-agency entries
df_scale["agency"] = (
    df_scale["agency_norm"]
    .str.split(";")
)

df_scale = (
    df_scale
    .explode("agency")
)

df_scale["agency"] = (
    df_scale["agency"]
    .astype(str)
    .str.strip()
)


# Drop empty values
df_scale = df_scale[
    df_scale["agency"] != ""
]


# ------------------------------------------------------------
# Order agencies by median
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


agency_counts = (
    df_scale
    .groupby("agency")
    .size()
)


# ------------------------------------------------------------
# Figure 5
# ------------------------------------------------------------

fig5, axes5 = plt.subplots(
    2,
    1,
    figsize=(11, 10),
    gridspec_kw={
        "height_ratios": [
            1.2,
            0.8
        ]
    }
)


# ============================================================
# Figure 5(a)
# Scale denominator distribution
# ============================================================

ax = axes5[0]


# Use matplotlib boxplot to keep the order consistent with the sorting
box_data = [
    df_scale[
        df_scale[
            "agency"
        ] == agency
    ]["scale_value"]
    .values
    for agency in agency_order
]


bp = ax.boxplot(
    box_data,
    vert=False,
    patch_artist=True,
    labels=[
        f"{agency} (n={agency_counts[agency]})"
        for agency in agency_order
    ],
    showfliers=True,
    flierprops={
        "marker": "o",
        "markersize": 2.5,
        "markerfacecolor": "#BBBBBB",
        "markeredgecolor": "#888888",
        "alpha": 0.5
    },
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
    }
)


ax.set_xscale(
    "log"
)

ax.set_title(
    "(a) Scale Denominator Distribution by Institution",
    loc="left",
    pad=8,
    fontweight="bold"
)

ax.set_xlabel(
    "Scale Denominator (log scale)"
)

ax.set_ylabel(
    "Institution"
)

ax.grid(
    axis="x",
    which="major",
    alpha=0.2,
    linewidth=0.6
)

ax.spines[
    ["top", "right"]
].set_visible(False)

ax.tick_params(
    axis="y",
    labelsize=8.5
)


# ============================================================
# Figure 5(b)
# Scale-class composition
# ============================================================

scale_class_dist = (
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

scale_class_dist = (
    scale_class_dist
    .reindex(
        agency_order
    )
    .fillna(0)
)


scale_class_dist = (
    scale_class_dist
    .reindex(
        columns=[
            "Large",
            "Medium",
            "Small",
            "Very small"
        ],
        fill_value=0
    )
)


scale_class_prop = (
    scale_class_dist
    .div(
        scale_class_dist.sum(
            axis=1
        ),
        axis=0
    )
)


ax = axes5[1]

left = np.zeros(
    len(scale_class_prop)
)

y_positions = np.arange(
    len(scale_class_prop)
)

for scale_class in [
    "Large",
    "Medium",
    "Small",
    "Very small"
]:

    values = (
        scale_class_prop[
            scale_class
        ]
        .values
        *
        100
    )

    ax.barh(
        y_positions,
        values,
        left=left,
        color=SCALE_CLASS_COLORS[
            scale_class
        ],
        edgecolor="white",
        linewidth=0.5,
        label=scale_class
    )

    left += values


ax.set_title(
    "(b) Scale-Class Composition by Institution",
    loc="left",
    pad=8,
    fontweight="bold"
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
    [
        f"{agency}"
        for agency in agency_order
    ],
    fontsize=8.5
)

ax.invert_yaxis()

ax.grid(
    axis="x",
    alpha=0.2,
    linewidth=0.6
)

ax.spines[
    ["top", "right"]
].set_visible(False)

ax.legend(
    title="Scale Class",
    loc="upper left",
    bbox_to_anchor=(1.01, 1.0),
    frameon=False
)


fig5.suptitle(
    "Figure 5. Institutional Differences in Cartographic Scale",
    fontsize=14,
    fontweight="bold",
    y=0.995
)

fig5.tight_layout(
    rect=[
        0,
        0,
        0.87,
        0.98
    ]
)


save_figure(
    fig5,
    FIG5_PNG,
    FIG5_PDF,
    FIG5_SVG
)

plt.show()


# ============================================================
# ============================================================
# FIGURE 6
# Projection Preferences across Planetary Bodies
# ============================================================
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "Figure 6: Projection Preferences"
)

print(
    "=" * 80
)


# ------------------------------------------------------------
# 1. Filter out small-sample bodies
# ------------------------------------------------------------

MIN_BODY_COUNT = 20

projection_data = df.dropna(
    subset=[
        "target_body",
        "projection_norm"
    ]
).copy()


body_counts = (
    projection_data[
        "target_body"
    ]
    .value_counts()
)


valid_bodies = (
    body_counts[
        body_counts >= MIN_BODY_COUNT
    ]
    .index
    .tolist()
)


projection_data = projection_data[
    projection_data[
        "target_body"
    ].isin(
        valid_bodies
    )
].copy()


print(
    f"✔ Bodies included in the projection analysis: {valid_bodies}"
)


# ------------------------------------------------------------
# 2. Merge low-frequency projections
# ------------------------------------------------------------

MIN_PROJ_COUNT = 50

projection_counts = (
    projection_data[
        "projection_norm"
    ]
    .value_counts()
)


rare_projections = (
    projection_counts[
        projection_counts < MIN_PROJ_COUNT
    ]
    .index
    .tolist()
)


projection_data[
    "projection_group"
] = (
    projection_data[
        "projection_norm"
    ]
    .apply(
        lambda x:
        "Other"
        if x in rare_projections
        else x
    )
)


# If there are still many categories, keep only the top 5
projection_group_counts = (
    projection_data[
        "projection_group"
    ]
    .value_counts()
)

top_projections = (
    projection_group_counts
    .head(5)
    .index
    .tolist()
)

projection_data[
    "projection_group"
] = (
    projection_data[
        "projection_group"
    ]
    .apply(
        lambda x:
        x
        if x in top_projections
        else "Other"
    )
)


# Ensure Other comes last
projection_order = (
    top_projections
    + (
        ["Other"]
        if "Other"
        in projection_data[
            "projection_group"
        ].unique()
        else []
    )
)


# ============================================================
# Figure 6(a)
# ============================================================

projection_composition = (
    projection_data
    .groupby(
        [
            "target_body",
            "projection_group"
        ]
    )
    .size()
    .reset_index(
        name="count"
    )
)


projection_composition[
    "proportion"
] = (
    projection_composition
    .groupby(
        "target_body"
    )["count"]
    .transform(
        lambda x:
        x / x.sum()
    )
)


# Sort by the total number of maps per body
body_order = (
    projection_data[
        "target_body"
    ]
    .value_counts()
    .sort_values()
    .index
    .tolist()
)


fig6, axes6 = plt.subplots(
    1,
    2,
    figsize=(14, 7),
    gridspec_kw={
        "width_ratios": [
            1.15,
            1
        ]
    }
)


ax = axes6[0]

left = np.zeros(
    len(body_order)
)

for i, projection in enumerate(
    projection_order
):

    values = []

    for body in body_order:

        row = projection_composition[
            (
                projection_composition[
                    "target_body"
                ] == body
            )
            &
            (
                projection_composition[
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

    values = (
        np.asarray(values)
        *
        100
    )

    ax.barh(
        body_order,
        values,
        left=left,
        color=PROJECTION_COLORS[
            i % len(
                PROJECTION_COLORS
            )
        ],
        edgecolor="white",
        linewidth=0.5,
        label=projection
    )

    left += values


ax.set_title(
    "(a) Projection Composition by Planetary Body",
    loc="left",
    pad=8,
    fontweight="bold"
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

ax.grid(
    axis="x",
    alpha=0.2,
    linewidth=0.6
)

ax.spines[
    ["top", "right"]
].set_visible(False)

ax.legend(
    title="Projection",
    loc="upper left",
    bbox_to_anchor=(1.01, 1.0),
    frameon=False
)


# ============================================================
# 3. Chi-square test
# ============================================================

contingency = pd.crosstab(
    projection_data[
        "target_body"
    ],
    projection_data[
        "projection_group"
    ]
)

chi2, p, dof, expected = (
    chi2_contingency(
        contingency
    )
)

n = contingency.values.sum()

r, k = contingency.shape

phi2 = chi2 / n

cramer_v = np.sqrt(
    phi2 /
    min(
        k - 1,
        r - 1
    )
)


print(
    "\n===== Projection Chi-square Test ====="
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


# ============================================================
# 4. Jensen–Shannon divergence
# ============================================================

all_projection_groups = projection_order

prob_matrix = []

for body in valid_bodies:

    subset = projection_data[
        projection_data[
            "target_body"
        ] == body
    ]

    counts = (
        subset[
            "projection_group"
        ]
        .value_counts()
        .reindex(
            all_projection_groups,
            fill_value=0
        )
    )

    prob = (
        counts /
        counts.sum()
    )

    prob_matrix.append(
        prob.values
    )


prob_df = pd.DataFrame(
    prob_matrix,
    index=valid_bodies,
    columns=all_projection_groups
)


n_bodies = len(
    valid_bodies
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


# ============================================================
# Figure 6(b)
# ============================================================

ax = axes6[1]

im = ax.imshow(
    js_matrix,
    cmap="YlGnBu",
    vmin=0,
    vmax=max(
        0.5,
        js_matrix.max()
    ),
    aspect="auto"
)


ax.set_title(
    "(b) Jensen–Shannon Divergence of Projection Distributions",
    loc="left",
    pad=8,
    fontweight="bold"
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
    valid_bodies,
    rotation=45,
    ha="right"
)

ax.set_yticklabels(
    valid_bodies
)


# Numeric annotations
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
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color="#222222"
        )


cbar = fig6.colorbar(
    im,
    ax=ax,
    fraction=0.046,
    pad=0.04
)

cbar.set_label(
    "Jensen–Shannon divergence",
    fontsize=10
)


fig6.suptitle(
    "Figure 6. Projection Preferences across Planetary Bodies",
    fontsize=14,
    fontweight="bold",
    y=0.99
)

fig6.tight_layout(
    rect=[
        0,
        0,
        0.98,
        0.95
    ]
)


save_figure(
    fig6,
    FIG6_PNG,
    FIG6_PDF,
    FIG6_SVG
)

plt.show()


# ============================================================
# 11. Final output summary
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "ALL FIGURES GENERATED SUCCESSFULLY"
)

print(
    "=" * 80
)

print(
    "\nFigure 4:"
)

print(
    FIG4_PNG
)

print(
    FIG4_PDF
)

print(
    FIG4_SVG
)

print(
    "\nFigure 5:"
)

print(
    FIG5_PNG
)

print(
    FIG5_PDF
)

print(
    FIG5_SVG
)

print(
    "\nFigure 6:"
)

print(
    FIG6_PNG
)

print(
    FIG6_PDF
)

print(
    FIG6_SVG
)

print(
    "\n✔ Temporal → Institution → Planetary body"
)

print(
    "✔ Median + IQR + sample-size encoding"
)

print(
    "✔ Logarithmic scale for scale denominator"
)

print(
    "✔ 100% stacked composition"
)

print(
    "✔ Colorblind-safe palette"
)

print(
    "✔ Publication-quality PNG/PDF/SVG"
)