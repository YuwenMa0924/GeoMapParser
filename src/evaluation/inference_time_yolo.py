import os
import time
import torch
import cv2

from ultralytics import YOLO

# ===== Configuration =====
IMAGE_DIR = "path/to/representative_100"
MODEL_PATH = "path/to/rtdetr/output/rtdetr_r18vd_6x_coco/eval/latest.pth"

torch.set_num_threads(1)

# ===== Load the image list =====
image_paths = [
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
]

# ===== Load the model =====
model = YOLO(MODEL_PATH)
model.to("cpu")

# ===== Warm-up =====
for _ in range(10):
    _ = model(cv2.imread(image_paths[0]))

# ===== Timing =====
times = []

for path in image_paths:
    img = cv2.imread(path)

    start = time.perf_counter()
    _ = model(img)
    end = time.perf_counter()

    times.append(end - start)

# ===== Statistics =====
avg = sum(times) / len(times)

print(f"Average inference time: {avg*1000:.2f} ms/img")
print(f"FPS: {1/avg:.2f}")