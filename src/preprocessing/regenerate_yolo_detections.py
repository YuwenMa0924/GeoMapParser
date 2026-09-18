from ultralytics import YOLO
import os
import json
from tqdm import tqdm


# ======================
# PATH
# ======================

MODEL_PATH = r"path/to/weights/best.pt"

IMAGE_DIR = r"path/to/input_images"

OUTPUT_DIR = r"path/to/yolo_json"


CONF = 0.25


# ======================
# Load model
# ======================

model = YOLO(MODEL_PATH)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ======================
# Image list
# ======================

images = []

for f in os.listdir(IMAGE_DIR):

    if f.lower().endswith(
        (".png",".jpg",".jpeg",".tif")
    ):
        images.append(f)


print(
    f"Found images: {len(images)}"
)


# ======================
# inference
# ======================

for img_name in tqdm(images):

    img_path = os.path.join(
        IMAGE_DIR,
        img_name
    )


    results = model.predict(
        img_path,
        conf=CONF,
        verbose=False
    )


    result = results[0]


    detections=[]


    if result.boxes is not None:

        boxes=result.boxes


        for i in range(len(boxes)):

            xyxy = boxes.xyxy[i].cpu().numpy().tolist()

            conf = float(
                boxes.conf[i].cpu().numpy()
            )

            cls = int(
                boxes.cls[i].cpu().numpy()
            )


            detections.append(
                {
                    "class":cls,
                    "confidence":conf,
                    "bbox":xyxy
                }
            )


    output={

        "file":img_name,

        "total_detection_boxes":
            len(detections),

        "detections":
            detections
    }


    json_name = os.path.splitext(img_name)[0]+".json"


    save_path=os.path.join(
        OUTPUT_DIR,
        json_name
    )


    with open(
        save_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=4,
            ensure_ascii=False
        )


print("Done")