import os
import tempfile          # added for temporary files
os.environ['FLAGS_use_onednn'] = '1'
os.environ['FLAGS_enable_onednn_fusion'] = '0'
os.environ['FLAGS_onednn_dynamic_graph_fusion'] = '0'
os.environ['FLAGS_onednn_fusion_op_types'] = 'fused_conv2d'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import json
from PIL import Image
from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import dashscope
from http import HTTPStatus
import ast
import re

Image.MAX_IMAGE_PIXELS = None

# -------------------------- 0. Batch configuration --------------------------
MODEL_PATH = "path/to/weights/best.pt"
IMAGE_INPUT_DIR = "path/to/input_images"
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png']

TEMP_LEGEND_PATH = "temp_legend.jpg"
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")  # set your DashScope (Qwen) API key, e.g. export DASHSCOPE_API_KEY=<your-key>
dashscope.api_key = DASHSCOPE_API_KEY

EXPORT_JSON_DIR = "path/to/ablation_B/extraction_results"
METADATA_DIR = "path/to/ablation_B/metadata_json"

# Initialize the model and OCR
model = YOLO(MODEL_PATH)
ocr_engine = PaddleOCR(use_textline_orientation=True, lang='en')

CONF_THRESHOLDS = {
    "map_frame": 0.7,
    "title": 0.5,
    "legend": 0.5,
    "diagram": 0.5,
    "metadata_block": 0.5,
    "scale_block": 0.1,
    "north_arrow": 0.05
}
DEFAULT_CONF = 0.1

# -------------------------- Image compression function (unified preprocessing) --------------------------
def compress_image(img, max_size_mb=20, quality=85):
    """Compress a PIL image to the given maximum file size (MB); returns the temporary file path."""
    img_copy = img.copy()
    max_dim = 3000   # cap the longest side in pixels
    if max(img_copy.size) > max_dim:
        ratio = max_dim / max(img_copy.size)
        new_size = (int(img_copy.size[0] * ratio), int(img_copy.size[1] * ratio))
        img_copy = img_copy.resize(new_size, Image.Resampling.LANCZOS)

    # create a unique temporary file (suffix .jpg)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        temp_path = tmp.name

    # initial save
    img_copy.save(temp_path, "JPEG", quality=quality, optimize=True)
    # if the file is still too large, lower the quality step by step
    while os.path.getsize(temp_path) > max_size_mb * 1024 * 1024 and quality > 20:
        quality -= 10
        img_copy.save(temp_path, "JPEG", quality=quality, optimize=True)
    return temp_path

# -------------------------- Utility functions --------------------------
def get_image_list(input_dir):
    image_list = []
    if not os.path.exists(input_dir):
        print(f"Error: image input folder {input_dir} does not exist!")
        return image_list
    for file_name in os.listdir(input_dir):
        file_suffix = os.path.splitext(file_name)[1].lower()
        if file_suffix in SUPPORTED_FORMATS:
            file_path = os.path.join(input_dir, file_name)
            file_name_no_suffix = os.path.splitext(file_name)[0]
            image_list.append((file_path, file_name_no_suffix))
    image_list.sort()
    print(f"Successfully found {len(image_list)} images to process:")
    for idx, (path, name) in enumerate(image_list):
        print(f"  {idx + 1}. {name}{os.path.splitext(path)[1]} - {path}")
    return image_list

# -------------------------- Core function --------------------------
def process_single_image(image_path, image_name):
    print(f"\n{'='*50}\nProcessing image: {image_name} - {image_path}\n{'='*50}")
    compressed_path = None
    try:
        # ---------- 1. Load the original image and compress it (unified preprocessing) ----------
        original_img = Image.open(image_path)
        if original_img.mode != "RGB":
            original_img = original_img.convert("RGB")
        compressed_path = compress_image(original_img, max_size_mb=20, quality=85)
        print(f"[{image_name}] Image compressed to: {compressed_path}")

        # ---------- 2. YOLO detection (on the compressed image) ----------
        results = model(compressed_path, conf=0.05)
        detection_boxes = []
        class_names = model.names

        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                coords = box.xyxy[0].astype(int)
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                conf_thresh = CONF_THRESHOLDS.get(label_name, DEFAULT_CONF)
                if confidence >= conf_thresh:
                    detection_boxes.append({
                        "label": label_name,
                        "box": coords.tolist(),
                        "confidence": round(confidence, 3)
                    })

        # Distinguish the main map from inset maps
        map_frame_boxes = [box for box in detection_boxes if box['label'] == 'map_frame']
        def distinguish_maps(detection_boxes_inner):
            map_frames = []
            other_boxes = []
            for box_info in detection_boxes_inner:
                if box_info['label'] == 'map_frame':
                    box = box_info['box']
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    box_info['area'] = area
                    map_frames.append(box_info)
                else:
                    other_boxes.append(box_info)
            if not map_frames:
                return detection_boxes_inner
            map_frames.sort(key=lambda x: x['area'], reverse=True)
            main_map = map_frames[0]
            main_map['label'] = 'main_map'
            for inset_map in map_frames[1:]:
                inset_map['label'] = 'inset_map'
            return [main_map] + map_frames[1:] + other_boxes

        updated_boxes = distinguish_maps(detection_boxes)

        # ---------- 3. OCR text extraction (on the compressed image) ----------
        text_boxes = [box for box in updated_boxes if box['label'] in ['title', 'metadata_block', 'scale_block']]
        def task_extract_text_inner(image_path_inner, text_boxes_inner):
            try:
                original_image = Image.open(image_path_inner)
                ocr_results = {}
                for idx, box_info in enumerate(text_boxes_inner):
                    label = box_info['label']
                    x1, y1, x2, y2 = box_info['box']
                    w, h = original_image.size
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    cropped_image = original_image.crop((x1, y1, x2, y2))
                    if cropped_image.mode != "RGB":
                        cropped_image = cropped_image.convert("RGB")
                    cropped_array = np.array(cropped_image)
                    result = ocr_engine.predict(cropped_array)
                    extracted_text = ""
                    if isinstance(result, list) and result:
                        for item in result:
                            if isinstance(item, dict):
                                texts = item.get("rec_texts", [])
                                extracted_text += "\n".join(texts) + "\n"
                            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                                text = item[1][0] if isinstance(item[1], (list, tuple)) else str(item[1])
                                extracted_text += text + "\n"
                    ocr_results[f"{label}_{idx+1}"] = extracted_text.strip()
                return ocr_results
            except Exception as e:
                print(f"[OCR error]: {e}")
                return {"error": str(e)}

        # ---------- 4. CV scale analysis (on the compressed image) ----------
        scale_block_boxes = [box for box in updated_boxes if box['label'] == 'scale_block']
        def task_analyze_graphics_inner(image_path_inner, scale_region_boxes_inner):
            if not scale_region_boxes_inner:
                return {"scale_cv": {"exists": False}}
            box = max(scale_region_boxes_inner, key=lambda b: (b['box'][2]-b['box'][0])*(b['box'][3]-b['box'][1]))['box']
            img = cv2.imread(image_path_inner)
            if img is None:
                return {"scale_cv": {"exists": True, "error": "image_read_failed"}}
            x1, y1, x2, y2 = box
            scale_crop = img[y1:y2, x1:x2]
            h_crop, w_crop = scale_crop.shape[:2]
            aspect_ratio = round(w_crop / max(h_crop, 1), 2)
            img_h, img_w = img.shape[:2]
            center_x, center_y = (x1 + x2)/2, (y1 + y2)/2
            vertical_position = "middle"
            if center_y < 0.3 * img_h: vertical_position="top"
            elif center_y > 0.7 * img_h: vertical_position="bottom"
            horizontal_position = "center"
            if center_x < 0.3 * img_w: horizontal_position="left"
            elif center_x > 0.7 * img_w: horizontal_position="right"
            position = f"{vertical_position}_{horizontal_position}"
            has_bar, bar_pixel_length = False, None
            gray = cv2.cvtColor(scale_crop, cv2.COLOR_BGR2GRAY)
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30,3))
            opened = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bar_candidates = [(x,y,w,h) for cnt in contours for x,y,w,h in [cv2.boundingRect(cnt)] if w/h>5 and w*h>500]
            if bar_candidates:
                has_bar = True
                bar_pixel_length = max(bar_candidates, key=lambda b:b[2])[2]
            return {"scale_cv": {"exists": True, "aspect_ratio": aspect_ratio, "relative_position": position, "area": w_crop*h_crop, "has_bar": has_bar, "bar_pixel_length": bar_pixel_length}}

        # ---------- 5. MLLM legend analysis (on the compressed image) ----------
        legend_boxes = [box for box in updated_boxes if box['label']=='legend']

        def task_analyze_legend_mllm_inner(image_path_inner, legend_boxes_inner):
            if not legend_boxes_inner:
                print(f"[{image_name}] No legend box detected")
                return {"legend_data": []}

            print(f"[{image_name}] legend_boxes:", legend_boxes_inner)
            final_legend_data = []

            img = cv2.imread(image_path_inner)
            if img is None:
                return {"error": "failed to read the image"}

            h, w = img.shape[:2]

            for idx, box_dict in enumerate(legend_boxes_inner):
                try:
                    x1, y1, x2, y2 = box_dict['box']
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    if x2 <= x1 or y2 <= y1:
                        print(f"[{image_name}] [WARN] legend box {idx} is invalid: {x1},{y1},{x2},{y2}")
                        continue

                    legend_img = img[y1:y2, x1:x2]

                    # limit the size
                    max_side = 1024
                    lh, lw = legend_img.shape[:2]
                    scale = min(max_side / max(lh, lw), 1.0)
                    if scale < 1.0:
                        legend_img = cv2.resize(
                            legend_img,
                            (int(lw * scale), int(lh * scale)),
                            interpolation=cv2.INTER_AREA
                        )

                    cv2.imwrite(TEMP_LEGEND_PATH, legend_img)

                    # Prompt
                    prompt = """
         You are a professional map legend analyzer. Carefully analyze this legend image and return a JSON array.
                Each object in the array represents one legend item and must contain the following fields:
                1. 'value': the text description of the item (e.g. "forest", "river","crater").
                2. 'key_type': the symbol type (e.g. "color", "icon", "line", "pattern").
                3. 'key_data': the specific data of the symbol (if 'color', give the HEX color value; if 'icon' or 'line', give a short English description such as "black_triangle" or "dashed_line").
        """

                    local_image_path = f"file://{os.path.abspath(TEMP_LEGEND_PATH)}"
                    messages = [
                        {"role": "user", "content": [{"image": local_image_path}]},
                        {"role": "user", "content": [{"text": prompt}]}
                    ]

                    print(f"[{image_name}] Calling the Qwen API to analyze legend box {idx}...")
                    response = dashscope.MultiModalConversation.call(
                        model='qwen-vl-plus',
                        messages=messages
                    )

                    if response.status_code != HTTPStatus.OK:
                        print(f"[{image_name}] [WARN] Failed to analyze legend box {idx}: {response.message}")
                        continue

                    content = response.output.choices[0].message.content
                    if isinstance(content, list):
                        content = "".join(map(str, content))
                    content = str(content).strip()

                    print(f"[{image_name}] [DEBUG] raw content repr of box {idx}:", repr(content)[:1000])

                    legend_json = []

                    parsed_inner = None
                    try:
                        if content.startswith("{") and ("'text'" in content or '"text"' in content):
                            py = ast.literal_eval(content)
                            if isinstance(py, dict):
                                parsed_inner = py.get('text') or py.get('content') or next(iter(py.values()))
                    except Exception:
                        parsed_inner = None

                    if parsed_inner is None:
                        parsed_inner = content

                    if isinstance(parsed_inner, str) and (
                            '\\n' in parsed_inner or '\\t' in parsed_inner or '\\\\"' in parsed_inner):
                        try:
                            parsed_inner = parsed_inner.encode('utf-8').decode('unicode_escape')
                        except Exception:
                            pass

                    json_text = None
                    m = re.search(r"```json\s*(.*?)\s*```", parsed_inner, re.S)
                    if m:
                        json_text = m.group(1)
                    else:
                        m2 = re.search(r"(\[.*\])", parsed_inner, re.S)
                        if m2:
                            json_text = m2.group(1)
                        else:
                            json_text = parsed_inner

                    json_text = json_text.strip()
                    if (json_text.startswith("'") and json_text.endswith("'")) or (
                            json_text.startswith('"') and json_text.endswith('"')):
                        json_text = json_text[1:-1].strip()

                    print(f"[{image_name}] [DEBUG] json_text snippet:",
                          (json_text[:400] + '...') if len(json_text) > 400 else json_text)

                    try:
                        legend_json = json.loads(json_text)
                        if not isinstance(legend_json, list):
                            raise ValueError("parsed JSON is not a list")
                    except Exception as e1:
                        try:
                            legend_json = ast.literal_eval(json_text)
                            if not isinstance(legend_json, list):
                                legend_json = []
                        except Exception as e2:
                            print(f"[{image_name}] [WARN] Failed to parse JSON for legend box {idx}:", e1, "| fallback error:", e2)
                            print("Raw content repr (truncated):", repr(content)[:2000])
                            legend_json = []

                    final_legend_data.extend(legend_json)

                    for j, item in enumerate(legend_json):
                        try:
                            print(
                                f"[{image_name}] [Legend {idx + 1}.{j + 1}] value: {item.get('value')}, key_type: {item.get('key_type')}, key_data: {item.get('key_data')}")
                        except Exception:
                            print(f"[{image_name}] [Legend {idx + 1}.{j + 1}] (unable to read the standard fields of this item)",
                                  repr(item)[:200])

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"[{image_name}] [WARN] Exception while processing legend box {idx}: {e}")
                    continue

            return {"legend_data": final_legend_data}

        # run the three steps (all based on the compressed image path)
        ocr_output = task_extract_text_inner(compressed_path, text_boxes)
        cv_output = task_analyze_graphics_inner(compressed_path, scale_block_boxes)
        legend_output = task_analyze_legend_mllm_inner(compressed_path, legend_boxes)

        # -------------------------- Ablation experiment B: direct concatenation output --------------------------
        final_json = {
            "ocr_results": ocr_output,
            "scale_analysis": cv_output,
            "legend_data": legend_output.get("legend_data", [])
        }

        # Write JSON
        if not os.path.exists(EXPORT_JSON_DIR):
            os.makedirs(EXPORT_JSON_DIR, exist_ok=True)
        json_path = os.path.join(EXPORT_JSON_DIR, f"{image_name}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(final_json, f, ensure_ascii=False, indent=2)
        print(f"{image_name} JSON saved: {json_path}")

        # Metadata
        try:
            metadata_info = {
                "image_name": image_name,
                "image_path": image_path,
                "compressed_image_path": compressed_path,
                "total_detection_boxes": len(updated_boxes),
                "detection_boxes_detail": [{"label": box["label"], "box_coordinates": box["box"], "confidence": box.get("confidence", 0.0), "area": box.get("area", 0)} for box in updated_boxes],
                "main_map_count": len([box for box in updated_boxes if box["label"] == "main_map"]),
                "inset_map_count": len([box for box in updated_boxes if box["label"] == "inset_map"])
            }
            if not os.path.exists(METADATA_DIR):
                os.makedirs(METADATA_DIR, exist_ok=True)
            metadata_json_path = os.path.join(METADATA_DIR, f"{image_name}_metadata.json")
            with open(metadata_json_path, "w", encoding="utf-8") as f:
                json.dump(metadata_info, f, ensure_ascii=False, indent=2)
            print(f"{image_name} metadata saved: {metadata_json_path}")
        except Exception as e:
            print(f"{image_name} failed to save metadata: {e}")

        return True

    except Exception as e:
        print(f"[{image_name}] Processing exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # clean up the temporary compressed image file
        if compressed_path and os.path.exists(compressed_path):
            try:
                os.remove(compressed_path)
                print(f"[{image_name}] Deleted temporary compressed file: {compressed_path}")
            except:
                pass
        # clean up the temporary legend file
        if os.path.exists(TEMP_LEGEND_PATH):
            try:
                os.remove(TEMP_LEGEND_PATH)
            except:
                pass

# -------------------------- Main program --------------------------
if __name__ == "__main__":
    image_tasks = get_image_list(IMAGE_INPUT_DIR)
    if not image_tasks:
        print("No images to process; exiting!")
        exit()

    if not os.path.exists(EXPORT_JSON_DIR):
        os.makedirs(EXPORT_JSON_DIR, exist_ok=True)
    if not os.path.exists(METADATA_DIR):
        os.makedirs(METADATA_DIR, exist_ok=True)

    success_count, fail_count = 0, 0
    for image_path, image_name in image_tasks:
        if process_single_image(image_path, image_name):
            success_count += 1
        else:
            fail_count += 1

    print(f"\n{'='*60}\nBatch processing complete! Total: {len(image_tasks)} | Success: {success_count} | Failed: {fail_count}")
    print(f"JSON files saved to: {EXPORT_JSON_DIR}\nMetadata saved to: {METADATA_DIR}\n{'='*60}")