import os
import json
import pandas as pd


def process_geologic_json_files(json_dir, output_excel):
    """
    Batch-process geologic JSON files, count whether each field has content, and generate an Excel statistics table
    :param json_dir: path to the directory containing the JSON files
    :param output_excel: path to the output Excel file
    """
    # Define the fields to count (one-to-one with the 7 categories in the requirements)
    fields = [
        "title", "scale", "projection",
        "publication_agency", "publication_date",
        "data_source", "legend_data"
    ]
    # Store the statistics for all files
    result_list = []

    # Iterate over all files in the specified directory
    for filename in os.listdir(json_dir):
        # Only process files with the .json extension
        if not filename.lower().endswith(".json"):
            continue

        # Extract the file ID (e.g. 1.json -> 1, 12.json -> 12)
        file_id = filename.split(".")[0]
        # Build the full file path
        file_path = os.path.join(json_dir, filename)

        # Initialize the statistics for the current file (all zeros by default)
        file_stats = {"ID": file_id}
        for field in fields:
            file_stats[field] = 0

        try:
            # Read and parse the JSON file
            with open(file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

            # Check each field for content; set it to 1 if present
            # 1. Handle string-type fields (title/scale/projection/publication_date/data_source)
            for field in ["title", "scale", "projection", "publication_date", "data_source"]:
                # Condition: field present + not null + non-empty string
                if field in json_data and json_data[field] is not None and str(json_data[field]).strip() != "":
                    file_stats[field] = 1

            # 2. Handle list-type fields (publication_agency/legend_data)
            for field in ["publication_agency", "legend_data"]:
                # Condition: field present + is a list + list non-empty
                if field in json_data and isinstance(json_data[field], list) and len(json_data[field]) > 0:
                    file_stats[field] = 1

        except Exception as e:
            # Catch file read/parse exceptions and print a notice (without stopping the overall flow)
            print(f"Error while processing file {filename}: {str(e)}")

        # Append the current file's statistics to the list
        result_list.append(file_stats)

    # Convert the statistics into a DataFrame (convenient for Excel generation)
    df = pd.DataFrame(result_list)
    # Sort by ID (optional; adjust as needed)
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce")
    df = df.sort_values("ID").fillna({"ID": "unknown"})
    df["ID"] = df["ID"].astype(str)

    # Generate the Excel file (without the default pandas index)
    df.to_excel(output_excel, index=False, engine="openpyxl")
    print(f"Statistics completed! Results saved to: {output_excel}")


# -------------------------- Script usage example --------------------------
if __name__ == "__main__":
    # Please change the following two paths to your actual paths
    # 1. Directory containing the JSON files (e.g. D:/geologic_json)
    JSON_DIRECTORY = "path/to/ablation_D/extraction_results"
    # 2. Path to the output Excel file (e.g. D:/geologic_stats.xlsx)
    OUTPUT_EXCEL_PATH = "path/to/ablation_D/field_presence.xlsx"

    # Run the statistics and generate the Excel file
    process_geologic_json_files(JSON_DIRECTORY, OUTPUT_EXCEL_PATH)