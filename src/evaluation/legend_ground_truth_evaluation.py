import pandas as pd
from Levenshtein import ratio  # pip install python-Levenshtein

# ================= Parameter configuration =================
GT_FILE = "path/to/legend_gt.xlsx"
MODEL_FILES = {
    "ChatGPT": "path/to/legend_GPT.xlsx",
    "Qwen": "path/to/legend_QWEN.xlsx",
    "InternVL": "path/to/legend_IVL.xlsx"
}
VALUE_THRESH_COLOR = 0.6 # Threshold for the "color" type
VALUE_THRESH_DEFAULT = 0.6  # Threshold for the remaining types
DEBUG = True  # Print matching debug information


# ================= Utility functions =================
def normalize_text(s):
    if not isinstance(s, str):
        return ""
    import re
    s = s.lower().strip()
    s = re.sub(r"[.,;:]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def fuzzy_match(s1, s2):
    return ratio(normalize_text(s1), normalize_text(s2))


# ================= Column-name adaptation =================
def adapt_columns(df, mapping):
    cols_lower = {c.lower(): c for c in df.columns}
    new_cols = {}
    for target_col, candidates in mapping.items():
        found = False
        for c in candidates:
            if c.lower() in cols_lower:
                new_cols[cols_lower[c.lower()]] = target_col
                found = True
                break
        if not found:
            df[target_col] = ""
    df.rename(columns=new_cols, inplace=True)
    return df


# ================= Evaluation function =================
def evaluate_legend(gt_df, pred_df, debug=False):
    # Convert all map_id values to strings and strip whitespace
    gt_df['map_id'] = gt_df['map_id'].astype(str).str.strip()
    pred_df['map_id'] = pred_df['map_id'].astype(str).str.strip()

    # Fill missing values
    gt_df[['key_type_gt', 'value_gt', 'key_data_gt']] = gt_df[['key_type_gt', 'value_gt', 'key_data_gt']].fillna(
        "").astype(str)
    pred_df[['key_type', 'value', 'key_data']] = pred_df[['key_type', 'value', 'key_data']].fillna("").astype(str)

    map_ids = gt_df['map_id'].unique()
    per_map_results = []
    total_TP = total_FP = total_FN = 0

    pred_grouped = pred_df.groupby("map_id") if not pred_df.empty else {}

    for map_id in map_ids:
        gt_items = gt_df[gt_df['map_id'] == map_id]
        preds = pred_grouped.get_group(map_id) if map_id in pred_grouped.groups else pd.DataFrame()
        preds = preds.copy()
        used_pred_idx = set()

        TP = FP = FN = 0

        for gt_idx, gt in gt_items.iterrows():
            matched = False
            best_score = 0
            best_p_idx = None
            for p_idx, p in preds.iterrows():
                if p_idx in used_pred_idx:
                    continue

                # Strict key_type match
                if normalize_text(gt["key_type_gt"]) != normalize_text(p["key_type"]):
                    continue

                # Determine the threshold and the matching logic
                if normalize_text(gt["key_type_gt"]) == "color":
                    min_score = fuzzy_match(gt["value_gt"], p["value"])
                    threshold = VALUE_THRESH_COLOR
                else:
                    val_score = fuzzy_match(gt["value_gt"], p["value"])
                    key_score = fuzzy_match(gt["key_data_gt"], p["key_data"])
                    min_score = min(val_score, key_score)
                    threshold = VALUE_THRESH_DEFAULT

                if min_score >= threshold and min_score > best_score:
                    best_score = min_score
                    best_p_idx = p_idx

                if debug:
                    print(f"[DEBUG] map_id={map_id}, GT idx={gt_idx} vs Pred idx={p_idx}: "
                          f"type OK, "
                          f"value_score={fuzzy_match(gt['value_gt'], p['value']):.3f}, "
                          f"{'' if normalize_text(gt['key_type_gt']) == 'color' else f'key_score={key_score:.3f}'}")

            if best_p_idx is not None:
                TP += 1
                used_pred_idx.add(best_p_idx)
                matched = True
                if debug:
                    print(f"[MATCHED] GT idx={gt_idx} matched with Pred idx={best_p_idx}, score={best_score:.3f}")
            if not matched:
                FN += 1
                if debug:
                    print(f"[FN] GT idx={gt_idx} not matched")

        FP += len(preds) - len(used_pred_idx)

        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        per_map_results.append({
            "map_id": map_id,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        })

        total_TP += TP
        total_FP += FP
        total_FN += FN

    overall_precision = total_TP / (total_TP + total_FP) if total_TP + total_FP > 0 else 0
    overall_recall = total_TP / (total_TP + total_FN) if total_TP + total_FN > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (
                overall_precision + overall_recall) if overall_precision + overall_recall > 0 else 0

    overall_metrics = {
        "TP": total_TP,
        "FP": total_FP,
        "FN": total_FN,
        "Precision": overall_precision,
        "Recall": overall_recall,
        "F1": overall_f1
    }

    return pd.DataFrame(per_map_results), overall_metrics


# ================= Main program =================
if __name__ == "__main__":
    gt_mapping = {
        "map_id": ["map_id", "map id", "MapID"],
        "key_type_gt": ["key_type_gt", "key type gt", "Type_GT"],
        "key_data_gt": ["key_data_gt", "key data gt", "Data_GT"],
        "value_gt": ["value_gt", "value", "Value_GT"]
    }
    pred_mapping = {
        "map_id": ["map_id", "map id", "MapID"],
        "key_type": ["key_type", "key type", "Type"],
        "key_data": ["key_data", "key data", "Data"],
        "value": ["value", "Value"]
    }

    gt_df = pd.read_excel(GT_FILE)
    gt_df = adapt_columns(gt_df, gt_mapping)

    all_results = []
    all_overall = []  # Added: stores the overall metrics of each model

    for model_name, model_file in MODEL_FILES.items():
        pred_df = pd.read_excel(model_file)
        pred_df = adapt_columns(pred_df, pred_mapping)

        per_map_df, overall = evaluate_legend(gt_df, pred_df, debug=DEBUG)
        per_map_df["Model"] = model_name
        all_results.append(per_map_df)

        # Save the overall metrics
        overall_copy = overall.copy()
        overall_copy["Model"] = model_name
        all_overall.append(overall_copy)

        print(f"\n=== {model_name} overall metrics ===")
        print(overall)

    # Aggregate the overall metrics of all models and print a comparison
    overall_df = pd.DataFrame(all_overall)
    overall_df = overall_df[["Model", "TP", "FP", "FN", "Precision", "Recall", "F1"]]
    print("\n=== Overall metrics comparison across all models ===")
    print(overall_df.round(4).to_string(index=False))

    # Save the per-map results
    all_maps_df = pd.concat(all_results, ignore_index=True)
    all_maps_df.to_excel("path/to/legend_per_map_debug.xlsx", index=False)
    print("\n✅ Exported the per-map_id evaluation table: MLLM_legend_per_map_debug_color_fixed.xlsx")