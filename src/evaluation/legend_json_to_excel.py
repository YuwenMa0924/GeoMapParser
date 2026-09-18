import os
import json
import pandas as pd

def extract_legend_to_excel(json_folder, output_excel):
    records = []

    # Iterate over the folder
    for filename in os.listdir(json_folder):
        if filename.lower().endswith('.json'):  # Accept .json / .JSON
            file_path = os.path.join(json_folder, filename)

            # map_id = file name (suffix removed)
            map_id = os.path.splitext(filename)[0]

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                legend_list = data.get("legend_data", [])

                # Skip entries without legend_data
                if not isinstance(legend_list, list):
                    print(f"[WARN] {filename} legend_data is not a list; skipping")
                    continue

                # Iterate over the legend items
                for idx, item in enumerate(legend_list, start=1):
                    value = item.get("value", "")
                    key_type = item.get("key_type", "")
                    key_data = item.get("key_data", "")

                    records.append({
                        "map_id": map_id,
                        "legend_id": idx,
                        "value_gt": value,
                        "key_type_gt": key_type,
                        "key_data_gt": key_data
                    })

            except Exception as e:
                print(f"[ERROR] Failed to process file: {filename}, error: {e}")

    # Convert to a DataFrame
    df = pd.DataFrame(records)

    # Sort by map_id + legend_id
    df = df.sort_values(by=["map_id", "legend_id"])

    # Save the Excel file
    df.to_excel(output_excel, index=False)

    print(f"✅ GT Excel generated: {output_excel}")
    print(f"{len(df)} legend records in total")


# ======================
# Usage example
# ======================
if __name__ == "__main__":
    json_folder = r"path/to/comparison_internvl2/results"   # Path to your JSON folder
    output_excel = r"path/to/legend_IVL.xlsx"

    extract_legend_to_excel(json_folder, output_excel)