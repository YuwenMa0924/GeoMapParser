import os

# ============================================================
# HuggingFace mirror
# ============================================================

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_ENDPOINT"] = "https://hf-mirror.com"

# ============================================================
# Imports
# ============================================================

import json
import re
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
from wordfreq import zipf_frequency


# ============================================================
# Path configuration
# ============================================================

# JSON result directories for the three OCR models
MODEL_DIRS = {
    "PaddleOCR": r"path/to/example_outputs",
    "EasyOCR": r"path/to/easyocr_results",
    "TrOCR": r"path/to/trocr_results",
}

# Output Excel
OUTPUT_EXCEL = (
    r"path/to/semantic_analysis_results/"
    r"OCR_quality_sensitivity_analysis.xlsx"
)

# Local SciBERT
SCIBERT_MODEL = (
    r"path/to/models/scibert_scivocab_uncased"
)


# ============================================================
# Field configuration
# ============================================================

FIELDS = [
    "title",
    "scale",
    "projection",
    "publication_agency",
    "publication_date",
    "data_source"
]

# Fields that use semantic embeddings
SEMANTIC_FIELDS = {
    "title",
    "projection",
    "publication_agency",
    "data_source"
}

# Fields that use the OCR incomplete-word check
OCR_FIELDS = {
    "title",
    "projection",
    "publication_agency",
    "data_source"
}


# ============================================================
# Sensitivity analysis parameters
# ============================================================

# semantic transition center
Z0_VALUES = [
    2.0,
    2.5,
    3.0
]

# sigmoid slope
K_VALUES = [
    0.5,
    1.0,
    2.0
]

# Baseline parameters finally adopted in the paper
BASE_Z0 = 2.5
BASE_K = 1.0


# ============================================================
# Text normalization
# ============================================================

def normalize_text(x):

    if x is None:
        return ""

    if isinstance(x, list):
        return "; ".join(x)

    return str(x).strip()


# ============================================================
# Rule penalty
# ============================================================

def rule_violation(field, text):

    """
    Rule consistency penalty

    Pr = 0:
        rule satisfied

    Pr = 1:
        rule violated
    """

    if text is None or text.strip() == "":
        return 1.0, ["empty_field"]

    # --------------------------------------------------------
    # Scale
    # --------------------------------------------------------

    if field == "scale":

        if re.search(
            r"1\s*[:：]\s*\d[\d,]*",
            text.strip()
        ):
            return 0.0, []

        return 1.0, [
            "invalid_scale_format"
        ]

    # --------------------------------------------------------
    # Publication date
    # --------------------------------------------------------

    if field == "publication_date":

        if re.fullmatch(
            r"\d{4}",
            text.strip()
        ):
            return 0.0, []

        return 1.0, [
            "invalid_date_format"
        ]

    # --------------------------------------------------------
    # Other fields
    # --------------------------------------------------------

    return 0.0, []


# ============================================================
# OCR incomplete-word penalty
# ============================================================

def ocr_penalty(field, text):

    """
    OCR incomplete-word penalty

    Po = abnormal token ratio
    """

    # Fields that are not part of the OCR completeness evaluation
    if field not in OCR_FIELDS:
        return 0.0, []

    # Empty field
    if text is None or text.strip() == "":
        return 1.0, ["empty_field"]

    words = text.split()

    if len(words) == 0:
        return 0.0, []

    bad_words = []

    for w in words:

        # Strip punctuation and digits
        clean_word = re.sub(
            r"[^a-zA-Z]",
            "",
            w
        )

        # Do not evaluate tokens that are too short
        if len(clean_word) < 4:
            continue

        # Do not evaluate all-uppercase abbreviations
        if clean_word.isupper():
            continue

        freq = zipf_frequency(
            clean_word.lower(),
            "en"
        )

        # Low-frequency tokens are treated as potential OCR anomalies
        if freq < 1.0:
            bad_words.append(clean_word)

    ratio = len(bad_words) / len(words)

    reasons = []

    if len(bad_words) > 0:
        reasons.append("incomplete_words")

    return ratio, reasons


# ============================================================
# Semantic penalty
# ============================================================

def semantic_penalty(z, z0, k):

    """
    Continuous sigmoid semantic penalty

        Ps = 1 / (1 + exp(-k * (z - z0)))
    """

    # Prevent exponential overflow
    exponent = np.clip(
        -k * (z - z0),
        -60,
        60
    )

    penalty = (
        1.0 /
        (
            1.0 +
            np.exp(exponent)
        )
    )

    return penalty


# ============================================================
# Load the JSON of a single OCR model
# ============================================================

def load_model_records(json_dir):

    records = []

    if not os.path.isdir(json_dir):
        raise FileNotFoundError(
            f"JSON directory does not exist:\n{json_dir}"
        )

    files = [
        f for f in os.listdir(json_dir)
        if f.lower().endswith(".json")
    ]

    print(
        f"JSON directory: {json_dir}"
    )

    print(
        f"Number of JSON files: {len(files)}"
    )

    if len(files) == 0:
        raise ValueError(
            f"No JSON files in the directory:\n{json_dir}"
        )

    for fname in sorted(files):

        path = os.path.join(
            json_dir,
            fname
        )

        try:

            with open(
                path,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

        except Exception as e:

            print(
                f"⚠ Skipping file {fname}: {e}"
            )

            continue

        for field in FIELDS:

            records.append(
                {
                    "file": fname,
                    "field": field,
                    "text": normalize_text(
                        data.get(field)
                    )
                }
            )

    if len(records) == 0:

        raise ValueError(
            f"No valid records were read:\n{json_dir}"
        )

    return pd.DataFrame(records)


# ============================================================
# Compute semantic z-score
# ============================================================

def calculate_semantic_z(df, embedder):

    print(
        "  → Computing SciBERT embeddings..."
    )

    embeddings = embedder.encode(
        df["text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True
    )

    df["embedding"] = list(embeddings)

    semantic_z = []

    print(
        "  → Computing field-wise semantic z-score..."
    )

    for field in FIELDS:

        sub = df[
            df["field"] == field
        ]

        # Non-semantic fields: z = 0
        if field not in SEMANTIC_FIELDS:

            semantic_z.extend(
                [0.0] * len(sub)
            )

            continue

        X = np.vstack(
            sub["embedding"].values
        )

        # Semantic centroid of the current field
        center = X.mean(
            axis=0,
            keepdims=True
        )

        # cosine distance
        distances = cosine_distances(
            X,
            center
        ).flatten()

        # standardized deviation
        z = (
            distances -
            distances.mean()
        ) / (
            distances.std() +
            1e-6
        )

        semantic_z.extend(
            z.tolist()
        )

    df["semantic_z"] = semantic_z

    return df


# ============================================================
# Compute rule penalty and OCR penalty
# ============================================================

def calculate_nonsemantic_penalties(df):

    print(
        "  → Computing rule penalty..."
    )

    rule_results = df.apply(
        lambda row:
        rule_violation(
            row["field"],
            row["text"]
        ),
        axis=1
    )

    df["rule_score"] = rule_results.apply(
        lambda x: x[0]
    )

    df["rule_reasons"] = rule_results.apply(
        lambda x: x[1]
    )

    print(
        "  → Computing OCR incompleteness penalty..."
    )

    ocr_results = df.apply(
        lambda row:
        ocr_penalty(
            row["field"],
            row["text"]
        ),
        axis=1
    )

    df["ocr_score"] = ocr_results.apply(
        lambda x: x[0]
    )

    df["ocr_reasons"] = ocr_results.apply(
        lambda x: x[1]
    )

    return df


# ============================================================
# Compute the quality score for one parameter combination
# ============================================================

def calculate_quality_scores(df, z0, k):

    quality_scores = []

    for _, row in df.iterrows():

        # ----------------------------------------------------
        # Semantic penalty
        # ----------------------------------------------------

        if row["field"] in SEMANTIC_FIELDS:

            ps = semantic_penalty(
                row["semantic_z"],
                z0,
                k
            )

        else:

            ps = 0.0

        # ----------------------------------------------------
        # Rule penalty
        # ----------------------------------------------------

        pr = min(
            1.0,
            float(row["rule_score"])
        )

        # ----------------------------------------------------
        # OCR penalty
        # ----------------------------------------------------

        po = min(
            1.0,
            float(row["ocr_score"])
        )

        # ----------------------------------------------------
        # Final Q
        # ----------------------------------------------------

        q = (
            (1.0 - ps) *
            (1.0 - pr) *
            (1.0 - po)
        )

        q = max(
            0.0,
            q
        )

        quality_scores.append(q)

    return np.array(
        quality_scores
    )


# ============================================================
# Full sensitivity analysis for one model
# ============================================================

def analyze_model(model_name, df):

    print("\n")
    print("=" * 70)
    print(
        f"Analyzing model: {model_name}"
    )
    print("=" * 70)

    summary_rows = []
    detail_rows = []

    # --------------------------------------------------------
    # All z0 × k combinations
    # --------------------------------------------------------

    for z0 in Z0_VALUES:

        for k in K_VALUES:

            print(
                f"  Parameter combination: Z0={z0}, K={k}"
            )

            q_values = calculate_quality_scores(
                df,
                z0,
                k
            )

            # Append to a temporary DataFrame
            temp = df.copy()

            temp["quality_score"] = q_values

            # ------------------------------------------------
            # Overall mean quality score
            # ------------------------------------------------

            overall_quality = (
                temp["quality_score"].mean()
            )

            # ------------------------------------------------
            # Per-field mean quality score
            # ------------------------------------------------

            field_means = (
                temp
                .groupby("field")[
                    "quality_score"
                ]
                .mean()
                .to_dict()
            )

            summary_row = {
                "Model": model_name,
                "Z0": z0,
                "K": k,
                "Overall_Textual_Quality": round(
                    overall_quality,
                    6
                )
            }

            # Add per-field scores
            for field in FIELDS:

                summary_row[
                    f"{field}_Q"
                ] = round(
                    field_means.get(
                        field,
                        np.nan
                    ),
                    6
                )

            # Whether these are the baseline parameters
            summary_row["Is_Baseline"] = (
                z0 == BASE_Z0 and
                k == BASE_K
            )

            summary_rows.append(
                summary_row
            )

            # ------------------------------------------------
            # Save all per-record details
            # ------------------------------------------------

            for idx, row in temp.iterrows():

                detail_rows.append(
                    {
                        "Model": model_name,
                        "file": row["file"],
                        "field": row["field"],
                        "text": row["text"],
                        "semantic_z": row["semantic_z"],
                        "rule_score": row["rule_score"],
                        "ocr_score": row["ocr_score"],
                        "Z0": z0,
                        "K": k,
                        "quality_score": q_values[idx]
                    }
                )

    summary_df = pd.DataFrame(
        summary_rows
    )

    detail_df = pd.DataFrame(
        detail_rows
    )

    return summary_df, detail_df


# ============================================================
# Main program
# ============================================================

def main():

    print("=" * 70)
    print(
        "OCR Textual Quality Score Sensitivity Analysis"
    )
    print("=" * 70)

    print("\nParameter range:")
    print(
        f"Z0 = {Z0_VALUES}"
    )
    print(
        f"K  = {K_VALUES}"
    )

    print(
        f"\nBaseline: Z0={BASE_Z0}, K={BASE_K}"
    )

    # --------------------------------------------------------
    # Load the model
    # --------------------------------------------------------

    print("\nLoading SciBERT...")

    embedder = SentenceTransformer(
        SCIBERT_MODEL
    )

    print(
        "✔ SciBERT loaded."
    )

    all_summary = []
    all_detail = []

    # --------------------------------------------------------
    # Analyze the three OCR models one after another
    # --------------------------------------------------------

    model_data = {}

    for model_name, json_dir in MODEL_DIRS.items():

        df = load_model_records(
            json_dir
        )

        print(
            f"  Number of field records: {len(df)}"
        )

        # Ensure 6 × number of maps
        print(
            f"  Number of maps: {df['file'].nunique()}"
        )

        df = calculate_semantic_z(
            df,
            embedder
        )

        df = calculate_nonsemantic_penalties(
            df
        )

        model_data[
            model_name
        ] = df

        summary_df, detail_df = analyze_model(
            model_name,
            df
        )

        all_summary.append(
            summary_df
        )

        all_detail.append(
            detail_df
        )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    summary_all = pd.concat(
        all_summary,
        ignore_index=True
    )

    detail_all = pd.concat(
        all_detail,
        ignore_index=True
    )

    # ========================================================
    # Determine the model ranking under each parameter combination
    # ========================================================

    ranking_rows = []

    for z0 in Z0_VALUES:

        for k in K_VALUES:

            sub = summary_all[
                (summary_all["Z0"] == z0) &
                (summary_all["K"] == k)
            ].copy()

            sub = sub.sort_values(
                "Overall_Textual_Quality",
                ascending=False
            )

            ranking = (
                " > ".join(
                    sub["Model"].tolist()
                )
            )

            scores = dict(
                zip(
                    sub["Model"],
                    sub["Overall_Textual_Quality"]
                )
            )

            ranking_rows.append(
                {
                    "Z0": z0,
                    "K": k,
                    "Ranking": ranking,
                    "PaddleOCR_Q": scores.get(
                        "PaddleOCR",
                        np.nan
                    ),
                    "EasyOCR_Q": scores.get(
                        "EasyOCR",
                        np.nan
                    ),
                    "TrOCR_Q": scores.get(
                        "TrOCR",
                        np.nan
                    ),
                    "Ranking_Stable": (
                        ranking ==
                        "PaddleOCR > EasyOCR > TrOCR"
                    )
                }
            )

    ranking_df = pd.DataFrame(
        ranking_rows
    )

    # ========================================================
    # Baseline vs parameter combinations
    # ========================================================

    baseline_df = summary_all[
        summary_all["Is_Baseline"]
    ].copy()

    # Q range for each model
    range_rows = []

    for model in MODEL_DIRS.keys():

        sub = summary_all[
            summary_all["Model"] == model
        ]

        q_min = sub[
            "Overall_Textual_Quality"
        ].min()

        q_max = sub[
            "Overall_Textual_Quality"
        ].max()

        q_baseline = sub.loc[
            (
                (sub["Z0"] == BASE_Z0) &
                (sub["K"] == BASE_K)
            ),
            "Overall_Textual_Quality"
        ]

        if len(q_baseline) > 0:
            q_baseline = q_baseline.iloc[0]
        else:
            q_baseline = np.nan

        range_rows.append(
            {
                "Model": model,
                "Baseline_Q": round(
                    q_baseline,
                    6
                ),
                "Minimum_Q": round(
                    q_min,
                    6
                ),
                "Maximum_Q": round(
                    q_max,
                    6
                ),
                "Range": round(
                    q_max - q_min,
                    6
                )
            }
        )

    range_df = pd.DataFrame(
        range_rows
    )

    # ========================================================
    # Overall stability
    # ========================================================

    stable_count = ranking_df[
        "Ranking_Stable"
    ].sum()

    total_count = len(
        ranking_df
    )

    stability_ratio = (
        stable_count /
        total_count
        if total_count > 0
        else 0
    )

    stability_summary = pd.DataFrame(
        [
            {
                "Total_parameter_combinations":
                    total_count,
                "Stable_rankings":
                    stable_count,
                "Stability_ratio":
                    round(
                        stability_ratio,
                        4
                    )
            }
        ]
    )

    # ========================================================
    # Output Excel
    # ========================================================

    print("\nWriting Excel...")

    with pd.ExcelWriter(
        OUTPUT_EXCEL,
        engine="openpyxl"
    ) as writer:

        # Overall and per-field Q for all parameter combinations
        summary_all.to_excel(
            writer,
            sheet_name="All_Summary",
            index=False
        )

        # Model ranking
        ranking_df.to_excel(
            writer,
            sheet_name="Ranking_Stability",
            index=False
        )

        # Q range
        range_df.to_excel(
            writer,
            sheet_name="Score_Range",
            index=False
        )

        # Baseline parameters
        baseline_df.to_excel(
            writer,
            sheet_name="Baseline_Results",
            index=False
        )

        # Stability summary
        stability_summary.to_excel(
            writer,
            sheet_name="Stability_Summary",
            index=False
        )

        # All record-level detailed results
        detail_all.to_excel(
            writer,
            sheet_name="Detailed_Results",
            index=False
        )

    # ========================================================
    # Console output
    # ========================================================

    print("\n")
    print("=" * 70)
    print("Sensitivity analysis completed")
    print("=" * 70)

    print(
        f"\nOutput file:\n{OUTPUT_EXCEL}"
    )

    print(
        "\nTotal number of parameter combinations:",
        total_count
    )

    print(
        "Number of combinations preserving PaddleOCR > EasyOCR > TrOCR:",
        stable_count
    )

    print(
        "Ranking stability ratio:",
        f"{stability_ratio:.2%}"
    )

    print("\n===== Quality score range per model =====")

    print(
        range_df.to_string(
            index=False
        )
    )

    print(
        "\n===== Model ranking per parameter combination ====="
    )

    print(
        ranking_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Final verdict
    # --------------------------------------------------------

    if stability_ratio == 1.0:

        print(
            "\n✅ Conclusion: "
            "under all parameter combinations, "
            "the OCR model ranking remains "
            "PaddleOCR > EasyOCR > TrOCR."
        )

    else:

        print(
            "\n⚠ Note: "
            "under some parameter combinations the model ranking changes; "
            "please check parameter sensitivity further."
        )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()