import os
import json
import cv2
from tqdm import tqdm

# ========= Required changes =========
classes = ["diagram","legend","map_frame","metadata_block","north_arrow","scale_block","title"]  # ⚠️ Set this to your own class names
root_dir = "path/to/rtdetr"  # Dataset root directory
# =======================

image_exts = (".jpg", ".jpeg", ".png")


def yolo_to_coco_bbox(box, img_w, img_h):
    x_c, y_c, w, h = box
    x_min = (x_c - w / 2) * img_w
    y_min = (y_c - h / 2) * img_h
    return [x_min, y_min, w * img_w, h * img_h]


def convert_split(split):
    image_dir = os.path.join(root_dir, "images", split)
    label_dir = os.path.join(root_dir, "labels", split)
    output_json = os.path.join(root_dir, "annotations", f"{split}.json")

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    images = []
    annotations = []
    ann_id = 0
    img_id = 0

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(image_exts)]

    for img_file in tqdm(image_files, desc=f"Processing {split}"):

        img_path = os.path.join(image_dir, img_file)

        # Automatically match the label file (compatible with jpg/png/jpeg)
        base_name = os.path.splitext(img_file)[0]
        label_path = os.path.join(label_dir, base_name + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Failed to read image: {img_path}")
            continue

        h, w = img.shape[:2]

        images.append({
            "id": img_id,
            "file_name": img_file,
            "width": w,
            "height": h
        })

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()

                if len(parts) != 5:
                    print(f"[WARN] Invalid annotation line: {label_path}")
                    continue

                cls_id = int(parts[0])
                box = list(map(float, parts[1:]))

                bbox = yolo_to_coco_bbox(box, w, h)
                x, y, bw, bh = bbox

                # Clip to prevent out-of-bounds coordinates
                x = max(0, x)
                y = max(0, y)
                bw = max(0, bw)
                bh = max(0, bh)

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls_id,
                    "bbox": [x, y, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0
                })

                ann_id += 1

        img_id += 1

    categories = [{"id": i, "name": name} for i, name in enumerate(classes)]

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_json, "w") as f:
        json.dump(coco_dict, f, indent=4)

    print(f"✅ {split} conversion completed: {output_json}")
    print(f"   Number of images: {len(images)}")
    print(f"   Number of annotations: {len(annotations)}")


def main():
    for split in ["train", "val", "test"]:
        convert_split(split)


if __name__ == "__main__":
    main()