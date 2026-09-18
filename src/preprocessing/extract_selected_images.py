import os
import shutil
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ==================== User configuration ====================
EXCEL_PATH = "path/to/maps_table_e10.xlsx"  # Path to the Excel file
SOURCE_FOLDER = "path/to/planetary_map_images"  # Path to the source image folder (with subfolders)
TARGET_FOLDER = "representative_100"  # Target folder name (will be created in the current directory)
FILE_COLUMN = "file"  # Column name storing file names in the Excel file (e.g. "abc.json")
SELECTED_COLUMN = "is_selected"  # Column name of the selection flag
# Supported image extensions (add or remove as needed)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
# Operation mode: 'copy' to copy, 'move' to move (use with caution)
OPERATION = 'copy'  # Or 'move'


# =================================================

def parse_boolean(value):
    """Convert various possible values into a boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "y")
    return False


def build_image_index(root_folder, extensions):
    """
    Recursively traverse root_folder and build an index of image files.
    Returns a dict: {file name (with extension): (full path, name of the top-level subfolder relative to root_folder)}
    If the file is directly under the root folder, the top-level subfolder name is the empty string ''.
    """
    index = {}
    root_path = Path(root_folder)
    for file_path in root_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            rel_path = file_path.relative_to(root_path)
            # Get the top-level subfolder name
            if str(rel_path.parent) == '.':
                sub_dir = ''  # Root directory
            else:
                sub_dir = rel_path.parent.parts[0]  # Top-level subfolder name
            fname = file_path.name  # Full file name with extension
            if fname in index:
                print(f"⚠️ Warning: duplicate file name '{fname}', already present in '{index[fname][1]}'; ignoring '{file_path}'")
            else:
                index[fname] = (str(file_path), sub_dir)
    return index


def get_candidate_names(fname):
    """
    Generate candidate image file names from a file name in the Excel file (which may contain .json).
    Returns a list of candidate file names (with extension).
    """
    candidates = []
    # 1. Try the name as-is
    candidates.append(fname)

    # 2. If it ends with .json, strip .json and try the base name
    if fname.lower().endswith('.json'):
        base = fname[:-5]  # Strip .json
        candidates.append(base)  # No-extension case (image extensions are appended below)
        # Generate a candidate for each image extension
        for ext in IMAGE_EXTENSIONS:
            candidates.append(base + ext)
    else:
        # If there is no extension at all, try appending the image extensions
        if '.' not in fname:
            for ext in IMAGE_EXTENSIONS:
                candidates.append(fname + ext)
    # Deduplicate (preserving order)
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)
    return unique_candidates


def main():
    # 1. Read the Excel file
    try:
        df = pd.read_excel(EXCEL_PATH)
    except Exception as e:
        print(f"❌ Failed to read the Excel file: {e}")
        return

    if FILE_COLUMN not in df.columns:
        print(f"❌ Column '{FILE_COLUMN}' not found in the Excel file; available columns: {df.columns.tolist()}")
        return
    if SELECTED_COLUMN not in df.columns:
        print(f"❌ Column '{SELECTED_COLUMN}' not found in the Excel file; available columns: {df.columns.tolist()}")
        return

    # 2. Filter rows with is_selected == True
    df['_selected_bool'] = df[SELECTED_COLUMN].apply(parse_boolean)
    df_selected = df[df['_selected_bool'] == True].copy()
    print(f"📊 Total rows: {len(df)}; after filtering (is_selected=True): {len(df_selected)} rows")

    if df_selected.empty:
        print("⚠️ No data left after filtering; exiting.")
        return

    selected_files = df_selected[FILE_COLUMN].dropna().astype(str).str.strip()
    selected_files = selected_files[selected_files != ""]
    unique_files = set(selected_files)
    print(f"📄 Unique image file names to process (from Excel): {len(unique_files)}")
    # Show the first few examples
    if unique_files:
        print(f"   Examples: {', '.join(list(unique_files)[:3])}")

    # 3. Create the target folder
    target_path = Path(TARGET_FOLDER)
    target_path.mkdir(parents=True, exist_ok=True)

    # 4. Check the source folder and build the index
    if not os.path.isdir(SOURCE_FOLDER):
        print(f"❌ Source folder does not exist: {SOURCE_FOLDER}")
        return

    print("🔍 Scanning the source folder and its subfolders to build the image index...")
    image_index = build_image_index(SOURCE_FOLDER, IMAGE_EXTENSIONS)
    print(f"✅ Found {len(image_index)} image files in total.")

    # 5. Iterate over each file name, find and copy/move it
    success_count = 0
    fail_count = 0
    missing_count = 0
    source_stats = defaultdict(int)

    for fname in sorted(unique_files):
        # Generate the list of candidate file names
        candidates = get_candidate_names(fname)
        found_path = None
        found_subdir = None
        for candidate in candidates:
            if candidate in image_index:
                found_path, found_subdir = image_index[candidate]
                break  # Use the first match

        if found_path is None:
            print(f"⚠️ Image not found: {fname} (tried {candidates})")
            missing_count += 1
            continue

        # Destination path (using the found file name, i.e. with extension)
        dest_path = target_path / os.path.basename(found_path)
        if dest_path.exists():
            base, ext = os.path.splitext(dest_path.name)
            counter = 1
            while True:
                new_name = f"{base}_{counter}{ext}"
                new_dest = target_path / new_name
                if not new_dest.exists():
                    dest_path = new_dest
                    break
                counter += 1

        try:
            if OPERATION == 'copy':
                shutil.copy2(found_path, dest_path)
            elif OPERATION == 'move':
                shutil.move(found_path, dest_path)
            else:
                print(f"❌ Unknown operation: {OPERATION}; use 'copy' or 'move'")
                return
            print(f"✅ {OPERATION} succeeded: {os.path.basename(found_path)} -> {dest_path}")
            success_count += 1
            source_stats[found_subdir if found_subdir else 'root'] += 1
        except Exception as e:
            print(f"❌ Error while processing {fname}: {e}")
            fail_count += 1

    # 6. Summary
    print("\n" + "=" * 50)
    print("📋 Processing complete!")
    print(f"  Successfully processed: {success_count}")
    print(f"  Files not found: {missing_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Target folder: {target_path.absolute()}")

    if source_stats:
        print("\n📁 Number of files found per source folder:")
        for subdir, count in sorted(source_stats.items(), key=lambda x: (x[0] == 'root', x[0])):
            if subdir:
                print(f"   Subfolder '{subdir}': {count} files")
            else:
                print(f"   Root directory: {count} files")


if __name__ == "__main__":
    main()