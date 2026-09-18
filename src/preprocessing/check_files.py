import os
import pandas as pd

# ==================== User configuration ====================
EXCEL_PATH = "path/to/maps_table_e10.xlsx"        # Path to the Excel file
FOLDER_PATH = "path/to/mllm_internvl_results"     # Path to the folder containing JSON files
FILE_COLUMN = "file"                             # Column name storing file names in the Excel file
SELECTED_COLUMN = "is_selected"                  # Column name of the selection flag
EXTENSION = ".json"                              # File extension to check
IGNORE_EXT = True                                # If True, ignore the extension when comparing names (recommended)
# ===========================================================

def get_base_name(filename, ext):
    """If IGNORE_EXT is True, strip the extension; otherwise return the filename unchanged."""
    if IGNORE_EXT and filename.lower().endswith(ext.lower()):
        return filename[:-len(ext)]
    return filename

def parse_boolean(value):
    """Convert various possible values into a boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "y")
    return False

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

    # 2. Filter rows with is_selected == True
    # Convert the column to boolean first (compatible with multiple formats)
    df['_selected_bool'] = df[SELECTED_COLUMN].apply(parse_boolean)
    df_filtered = df[df['_selected_bool'] == True].copy()
    print(f"📊 Total rows: {len(df)}; after filtering (is_selected=True): {len(df_filtered)} rows")

    if df_filtered.empty:
        print("⚠️ No rows left after filtering; exiting.")
        return

    # 3. Get the file column: drop duplicates and missing values, convert to string
    excel_files = set(df_filtered[FILE_COLUMN].dropna().astype(str).str.strip())
    if not excel_files:
        print("⚠️ The 'file' column is empty after filtering; nothing to compare.")
        return

    # 4. Read the JSON files in the folder
    if not os.path.isdir(FOLDER_PATH):
        print(f"❌ Folder does not exist: {FOLDER_PATH}")
        return

    all_files = os.listdir(FOLDER_PATH)
    json_files = [f for f in all_files if f.lower().endswith(EXTENSION.lower())]

    # 5. Normalize file names (optionally ignoring the extension)
    if IGNORE_EXT:
        folder_set = {get_base_name(f, EXTENSION) for f in json_files}
        excel_set = {get_base_name(f, EXTENSION) for f in excel_files}
    else:
        folder_set = set(json_files)
        excel_set = set(excel_files)

    # 6. Compute the intersection and differences
    both = excel_set & folder_set
    only_excel = excel_set - folder_set
    only_folder = folder_set - excel_set

    # 7. Print the results
    print("\n" + "="*50)
    print(f"📊 Comparison results (filtered by is_selected=True; extension {'ignored' if IGNORE_EXT else 'kept'})")
    print("="*50)
    print(f"Entries in Excel after filtering (deduplicated): {len(excel_set)}")
    print(f"JSON files in folder: {len(folder_set)}")
    print(f"✅ Present in both: {len(both)}")
    if both:
        print("   " + ", ".join(sorted(both)[:20]) + (" ..." if len(both) > 20 else ""))

    print(f"\n📌 Only in Excel (missing files): {len(only_excel)}")
    if only_excel:
        for name in sorted(only_excel):
            print(f"   - {name}")

    print(f"\n📁 Only in folder (extra files): {len(only_folder)}")
    if only_folder:
        for name in sorted(only_folder):
            print(f"   - {name}")

    # Optional: save the list of missing file names
    if only_excel:
        with open("missing_files.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(only_excel)))
        print("\n💾 Missing file names saved to missing_files.txt")

if __name__ == "__main__":
    main()
