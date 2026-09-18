import json
import os
import numpy as np
import pandas as pd

# ---------------------- Step 1: Read the data and compute the area of each box ----------------------
def calculate_box_area(box_coords):
    """Compute the area of a single detection box"""
    x1, y1, x2, y2 = box_coords
    width = x2 - x1
    height = y2 - y1
    area = abs(width * height)
    return area

# Store the label and area of every detection box
all_box_info = []
# Replace with the path to your folder of JSON files
json_folder = "path/to/extraction_results"
json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]

# Read in batch and compute areas
for json_file in json_files:
    file_path = os.path.join(json_folder, json_file)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for box in data["detection_boxes_detail"]:
        actual_area = calculate_box_area(box["box_coordinates"])
        all_box_info.append({
            "label": box["label"],
            "area": actual_area
        })

# ---------------------- Step 2: Aggregate macro-level area statistics for each class ----------------------
df = pd.DataFrame(all_box_info)
# Aggregate by class: compute mean area, median area, and total box count (mean area is the key reference)
label_stats = df.groupby("label").agg(
    mean_area=("area", "mean"),
    median_area=("area", "median"),
    box_count=("area", "count"),
    total_area=("area", "sum")
).reset_index()

# ---------------------- Step 3: Classify into large/medium/small by tertiles of the class mean area ----------------------
# Extract the mean area of all classes and compute the tertiles (the key thresholds for the large/medium/small split)
category_avg_areas = label_stats["mean_area"].values
# Tertiles: 33.3% (small → medium), 66.6% (medium → large)
small_medium_threshold = np.percentile(category_avg_areas, 33.3)
medium_large_threshold = np.percentile(category_avg_areas, 66.6)

# Assign an area-based feature category to each class
def judge_area_type(avg_area):
    if avg_area >= medium_large_threshold:
        return "Large-area feature"
    elif avg_area >= small_medium_threshold:
        return "Medium-area feature"
    else:
        return "Small-area feature"

# New column: area-based feature category
label_stats["area_feature_type"] = label_stats["mean_area"].apply(judge_area_type)

# ---------------------- Step 4: Output the final conclusions (core results) ----------------------
print("="*80)
print("[888 maps - final area-based feature category for each class]")
print("="*80)
# Formatted output, rounded to 2 decimals
result_df = label_stats[["label", "mean_area", "median_area", "box_count", "area_feature_type"]].round(2)
# Sort by area-based feature category for easier viewing
result_df = result_df.sort_values(by="mean_area", ascending=False)

# Print the final conclusion table
print(result_df.to_string(index=False))

# ---------------------- Step 5: Save the results (Excel + text summary) ----------------------
# 1. Save the Excel file (the core file for review)
with pd.ExcelWriter("path/to/bbox_area_classification.xlsx", engine="openpyxl") as writer:
    # Core conclusion table
    result_df.to_excel(writer, sheet_name="final_classification_results", index=False)
    # Supplementary threshold description table
    threshold_df = pd.DataFrame({
        "threshold_type": ["small-to-medium area threshold", "medium-to-large area threshold"],
        "threshold_pixel2": [small_medium_threshold, medium_large_threshold],
        "description": ["category mean area < this value = small-area feature", "category mean area ≥ this value = large-area feature"]
    }).round(2)
    threshold_df.to_excel(writer, sheet_name="threshold_definitions", index=False)

# 2. Generate a text version of the conclusions (ready to copy for reporting)
print("\n" + "="*80)
print("[Text version of the conclusions (ready for reporting)]")
print("="*80)
for _, row in result_df.iterrows():
    print(f"• {row['label']}: belongs to [{row['area_feature_type']}] (mean area: {row['mean_area']:.2f} pixel²)")

# 3. Print the classification-threshold description
print("\n[Classification rule description]")
print(f"- Small-area feature: category mean area < {small_medium_threshold:.2f} pixel²")
print(f"- Medium-area feature: {small_medium_threshold:.2f} ≤ category mean area < {medium_large_threshold:.2f} pixel²")
print(f"- Large-area feature: category mean area ≥ {medium_large_threshold:.2f} pixel²")

print("\n✅ Results saved to an Excel file, containing the final classification results and the threshold definitions!")