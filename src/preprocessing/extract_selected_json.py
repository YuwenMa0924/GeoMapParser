import os
import pandas as pd
import shutil

# ===================== User configuration =====================
EXCEL_PATH = "path/to/maps_table_e10.xlsx"          # Path to the Excel file
JSON_FOLDER = "path/to/yolov8n_extraction_results"            # Path to the JSON folder
OUTPUT_FOLDER = "selected_jsons"               # Output folder name (will be created in the current directory)
FILE_COLUMN = "file"                           # Column name storing file names in the Excel file
SELECTED_COLUMN = "is_selected"                # Column name of the selection flag
# ===================================================

def main():
    # 1. Read the Excel file
    try:
        df = pd.read_excel(EXCEL_PATH)
    except Exception as e:
        print(f"❌ Failed to read the Excel file: {e}")
        return

    # Check that the required columns exist
    if FILE_COLUMN not in df.columns:
        print(f"❌ Column '{FILE_COLUMN}' not found in the Excel file; available columns: {df.columns.tolist()}")
        return
    if SELECTED_COLUMN not in df.columns:
        print(f"❌ Column '{SELECTED_COLUMN}' not found in the Excel file; available columns: {df.columns.tolist()}")
        return

    # 2. Filter rows with is_selected == True (compatible with booleans, strings, numbers)
    # Convert various possible values into a boolean
    def parse_bool(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return val == 1
        if isinstance(val, str):
            return val.strip().lower() in ("true", "1", "yes", "y")
        return False

    df['_selected'] = df[SELECTED_COLUMN].apply(parse_bool)
    df_selected = df[df['_selected'] == True]
    print(f"📊 Total rows: {len(df)}; after filtering (is_selected=True): {len(df_selected)} rows")

    if df_selected.empty:
        print("⚠️ No data left after filtering; exiting.")
        return

    # 3. Get the file column: drop duplicates and missing values, convert to string
    file_names = df_selected[FILE_COLUMN].dropna().astype(str).str.strip()
    file_names = file_names[file_names != ""]
    unique_files = set(file_names)
    print(f"📄 Unique JSON file names to extract: {len(unique_files)}")

    # 4. Check whether the JSON folder exists
    if not os.path.isdir(JSON_FOLDER):
        print(f"❌ JSON folder does not exist: {JSON_FOLDER}")
        return

    # 5. Create the output folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 6. Iterate over the file names and copy the files
    success_count = 0
    missing_count = 0

    for fname in sorted(unique_files):
        # Generate candidate file names (with and without .json)
        candidates = []
        if fname.endswith('.json'):
            candidates.append(fname)
            candidates.append(fname[:-5])  # Strip .json
        else:
            candidates.append(fname)
            candidates.append(fname + '.json')

        # Deduplicate while preserving order
        seen = set()
        candidates = [c for c in candidates if not (c in seen or seen.add(c))]

        copied = False
        for cand in candidates:
            src_path = os.path.join(JSON_FOLDER, cand)
            if os.path.isfile(src_path):
                dst_path = os.path.join(OUTPUT_FOLDER, cand)
                shutil.copy2(src_path, dst_path)
                print(f"✅ Copied successfully: {cand}")
                success_count += 1
                copied = True
                break

        if not copied:
            print(f"⚠️ File not found: {fname} (tried {candidates})")
            missing_count += 1

    # 7. Summary
    print("\n" + "=" * 50)
    print("📋 Extraction complete!")
    print(f"  Successfully copied: {success_count}")
    print(f"  Files not found: {missing_count}")
    print(f"  Output folder: {os.path.abspath(OUTPUT_FOLDER)}")

if __name__ == "__main__":
    main()