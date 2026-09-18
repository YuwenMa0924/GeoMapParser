import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import re

# ================= Configuration =================
FILE_PATH = "path/to/ablation/check.xlsx"  # Replace with the path to your Excel file
OUTPUT_FILE = "path/to/ablation/ocr_prf_evaluation.xlsx"  # Output file name; set to None to only print results

# Column-name prefixes
PRED_PREFIX = "extracted_"
GT_PREFIX = "actual_"

# ================= Read data =================
df = pd.read_excel(FILE_PATH)

# Automatically identify all field names (prefix removed)
pred_cols = [col for col in df.columns if col.startswith(PRED_PREFIX)]
gt_cols = [col for col in df.columns if col.startswith(GT_PREFIX)]

# Extract the shared field names (identical once the prefix is removed)
fields = set([col[len(PRED_PREFIX):] for col in pred_cols]) & \
         set([col[len(GT_PREFIX):] for col in gt_cols])

if not fields:
    print("No matching extracted and actual columns were found; please check the column-name format.")
    exit()

# ================= Compute metrics =================
results = []

for field in sorted(fields):
    pred_col = PRED_PREFIX + field
    gt_col = GT_PREFIX + field

    # Ensure the columns are numeric (0/1)
    y_pred = pd.to_numeric(df[pred_col], errors='coerce').fillna(0).astype(int)
    y_true = pd.to_numeric(df[gt_col], errors='coerce').fillna(0).astype(int)

    # Compute the confusion-matrix elements (optional, for debugging)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    # Compute the metrics with scikit-learn (handles all-zero or all-one cases)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    results.append({
        "field": field,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })

# ================= Output results =================
results_df = pd.DataFrame(results)

if OUTPUT_FILE:
    results_df.to_excel(OUTPUT_FILE, index=False)
    print(f"Evaluation results saved to {OUTPUT_FILE}")

# Print to the console
print("\n=== Field-level extraction quality evaluation ===\n")
print(results_df.round(4).to_string(index=False))

# Compute the macro averages (optional)
macro_precision = results_df["Precision"].mean()
macro_recall = results_df["Recall"].mean()
macro_f1 = results_df["F1"].mean()

print("\n=== Macro average (mean over fields) ===")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall:    {macro_recall:.4f}")
print(f"Macro F1:        {macro_f1:.4f}")
# Print the P/R/F1 of each field (more compact)
print("\n=== P/R/F1 by field ===")
for _, row in results_df.iterrows():
    print(f"{row['field']}: P={row['Precision']:.4f}, R={row['Recall']:.4f}, F1={row['F1']:.4f}")