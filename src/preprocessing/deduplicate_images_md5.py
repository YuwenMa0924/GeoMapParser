import os
import hashlib

def get_file_md5(file_path):
    """Compute the MD5 checksum of a file."""
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        while chunk := f.read(4096):
            md5.update(chunk)
    return md5.hexdigest()

def remove_duplicate_images(folder_path):
    """Remove duplicate images from a folder."""
    md5_dict = {}
    duplicate_count = 0
    # Walk through all image files in the folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                file_path = os.path.join(root, file)
                # Compute the MD5 checksum
                file_md5 = get_file_md5(file_path)
                if file_md5 in md5_dict:
                    # Remove the duplicate
                    os.remove(file_path)
                    duplicate_count += 1
                    print(f"Removing duplicate image: {file_path}")
                else:
                    md5_dict[file_md5] = file_path
    print(f"Total duplicate images removed: {duplicate_count}")

# Run the function; replace with the path to your image folder
if __name__ == '__main__':
    remove_duplicate_images("path/to/yolo_dataset/images")