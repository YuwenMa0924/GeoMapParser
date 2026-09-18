import pandas as pd


def calculate_field_existence_rate(input_excel, output_excel):
    """
    Compute field presence rates:
    1. Overwrite the existing "field_presence_rate" column (overall consistency ratio across the 7 fields)
    2. Additionally compute the presence rate of each individual field (consistency ratio between the extracted value and the actual value of a single field)
    :param input_excel: input Excel path (the original check.xlsx)
    :param output_excel: output Excel path (a new file; the original is not overwritten)
    """
    # Step 1: read the original Excel file
    try:
        df = pd.read_excel(input_excel)
        print(f"✅ Successfully read file: {input_excel}")
        print(f"📊 Data size: {len(df)} rows, {len(df.columns)} columns")
        total_rows = len(df)  # Get the actual total row count (more general than a hard-coded 888)
    except Exception as e:
        print(f"❌ Failed to read file: {str(e)}")
        return

    # Step 2: define the 7 core fields to compare (corresponding to the "extracted_XX" / "actual_XX" columns)
    core_fields = [
        "title", "scale", "projection",
        "publication_agency", "publication_date",
        "data_source", "legend_data"
    ]

    # Step 3: verify that the required columns exist (to avoid column-name mismatches)
    required_columns = []
    for field in core_fields:
        required_columns.extend([f"extracted_{field}", f"actual_{field}"])  # Generate the 14 paired column names
    required_columns.append("field_presence_rate")  # Confirm that the target column exists

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"❌ Table is missing required columns: {missing_cols}")
        print("💡 Please make sure the column names are formatted as 'extracted_<field>' and 'actual_<field>' and that the 'field_presence_rate' column exists")
        return

    # Step 4: define the per-row computation (clean data + compare consistency) - original overall presence rate logic
    def compute_single_rate(row):
        consistent_count = 0  # Count the consistent field pairs
        for field in core_fields:
            # Get the "extracted value" and the "actual value" of the current field and clean abnormal data
            extract_val = row[f"extracted_{field}"]
            actual_val = row[f"actual_{field}"]

            # Data cleaning: empty values and non-1/0 values are treated as 0 (per the "empty means 0" rule)
            extract_val = 0 if pd.isna(extract_val) or extract_val not in [1, 0] else extract_val
            actual_val = 0 if pd.isna(actual_val) or actual_val not in [1, 0] else actual_val

            # Increment the count if the values match
            if extract_val == actual_val:
                consistent_count += 1

        # Compute the presence rate (rounded to 3 decimals, e.g. 6/7=0.857)
        return round(consistent_count / 7, 3)

    # Step 5: compute in batch and overwrite the "field_presence_rate" column (original logic)
    df["field_presence_rate"] = df.apply(compute_single_rate, axis=1)
    print(f"🔢 Overall field presence rate computed; the existing 'field_presence_rate' column was overwritten")

    # Step 6: added - compute the presence rate of each individual field
    single_field_rates = {}  # Store the presence rate of each field
    print("\n📋 Individual field presence rate results:")
    for field in core_fields:
        # Clean row by row and check consistency
        consistent_rows = 0
        for idx, row in df.iterrows():
            extract_val = row[f"extracted_{field}"]
            actual_val = row[f"actual_{field}"]

            # Data cleaning (consistent with the overall logic)
            extract_val = 0 if pd.isna(extract_val) or extract_val not in [1, 0] else extract_val
            actual_val = 0 if pd.isna(actual_val) or actual_val not in [1, 0] else actual_val

            # Increment the count if consistent
            if extract_val == actual_val:
                consistent_rows += 1

        # Compute the individual field presence rate (rounded to 3 decimals)
        rate = round(consistent_rows / total_rows, 3)
        single_field_rates[field] = rate
        print(f"   - {field}: {rate} (consistent rows: {consistent_rows}/{total_rows})")

    # Step 7: save the results (original data + individual field presence rate summary)
    try:
        # Create an Excel writer that supports multiple sheets
        with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
            # Save the original data (including the overall field presence rate)
            df.to_excel(writer, sheet_name="raw_data_overall_presence_rate", index=False)
            # Save the individual field presence rate summary
            rate_df = pd.DataFrame(
                list(single_field_rates.items()),
                columns=["field_name", "individual_presence_rate"]
            )
            rate_df.to_excel(writer, sheet_name="individual_field_presence_rates", index=False)

        # Print a statistics summary
        rate_stats = df["field_presence_rate"].describe()
        print(f"\n✅ Results saved to: {output_excel}")
        print(f"📈 Overall presence rate statistics summary:")
        print(f"   - Max: {rate_stats['max']}")
        print(f"   - Min: {rate_stats['min']}")
        print(f"   - Mean: {round(rate_stats['mean'], 3)}")
        print(f"   - Median: {rate_stats['50%']}")
    except Exception as e:
        print(f"❌ Failed to save file: {str(e)}")
        return


# -------------------------- Script entry point --------------------------
if __name__ == "__main__":
    # -------------------------- Please modify the following 2 paths --------------------------
    INPUT_FILE_PATH = "path/to/ablation_E/ocr_field_consistency.xlsx"  # Original file path
    OUTPUT_FILE_PATH = "path/to/ablation_E/ocr_field_consistency_eachfield.xlsx"  # New file save path
    # ----------------------------------------------------------------------

    # Run the computation
    calculate_field_existence_rate(INPUT_FILE_PATH, OUTPUT_FILE_PATH)