import os
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

Image.MAX_IMAGE_PIXELS = None
import ast
import re

# -------------------------- 0. Batch configuration --------------------------
MODEL_PATH = "path/to/weights/best.pt"
IMAGE_INPUT_DIR = "path/to/input_images"
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png']

TEMP_LEGEND_PATH = "temp_legend.jpg"
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")  # set your DashScope (Qwen) API key, e.g. export DASHSCOPE_API_KEY=<your-key>
dashscope.api_key = DASHSCOPE_API_KEY
EXPORT_JSON_DIR = "path/to/extraction_results"
METADATA_DIR = "path/to/metadata_json"


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


# -------------------------- Tool function: Retrieve the list of images for batch processing. --------------------------
def get_image_list(input_dir):
    image_list = []
    if not os.path.exists(input_dir):
        print(f"Error: Image input folder {input_dir} dose not exist!")
        return image_list
    for file_name in os.listdir(input_dir):
        file_suffix = os.path.splitext(file_name)[1].lower()
        if file_suffix in SUPPORTED_FORMATS:
            file_path = os.path.join(input_dir, file_name)
            file_name_no_suffix = os.path.splitext(file_name)[0]
            image_list.append((file_path, file_name_no_suffix))
    image_list.sort()
    print(f"Successfully obtained {len(image_list)} images pending processing:")
    for idx, (path, name) in enumerate(image_list):
        print(f"  {idx + 1}. {name}{os.path.splitext(path)[1]} - {path}")
    return image_list


# -------------------------- Core function: Process a single image.--------------------------
def process_single_image(image_path, image_name):
    print(f"\n" + "=" * 50)
    print(f"Processing image: {image_name} - {image_path}")
    print("=" * 50)
    try:
        # Step 1: Layout analysis
        results = model(image_path, conf=0.05)
        detection_boxes = []
        class_names = model.names

        print(f"\n[{image_name}]YOLOv8n performance:")
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                coords = box.xyxy[0].astype(int)
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                label_name = label_name.lstrip()
                conf_thresh = CONF_THRESHOLDS.get(label_name, DEFAULT_CONF)
                if confidence >= conf_thresh:
                    detection_boxes.append({
                        "label": label_name,
                        "box": coords.tolist(),
                        "confidence": round(confidence, 3)
                    })
                    print(f"  {label_name}: {coords.tolist()} (confidence: {confidence:.2f}, thresh: {conf_thresh})")

        print(f"\n[{image_name}]The classes contained in the YOLO model: {class_names}")

        # Step 2: Logic classification (Main image / Sub-image)
        map_frame_boxes = [box for box in detection_boxes if box['label'] == 'map_frame']
        print(f"[{image_name}]Number of detected map_frame: {len(map_frame_boxes)}")

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
                print(f"[{image_name}]Warning: No map_frame detected, returning all detection boxes")
                return detection_boxes_inner
            map_frames.sort(key=lambda x: x['area'], reverse=True)
            main_map = map_frames[0]
            main_map['label'] = 'main_map'
            inset_maps = map_frames[1:]
            for inset_map in inset_maps:
                inset_map['label'] = 'inset_map'
            return [main_map] + inset_maps + other_boxes

        updated_boxes = distinguish_maps(detection_boxes)
        print(f"\n[{image_name}]Total detected {len(updated_boxes)} elements:")
        for i, box in enumerate(updated_boxes):
            print(f"  {i + 1}. {box['label']}: {box['box']}")

        # Step 3: Parallel extraction
        text_boxes = [box for box in updated_boxes if box['label'] in ['title', 'metadata_block', 'scale_block']]
        scale_block_boxes = [box for box in updated_boxes if box['label'] == 'scale_block']
        legend_boxes = [box for box in updated_boxes if box['label'] == 'legend']

        print(f"\n[{image_name}]Number of text boxes requiring OCR processing: {len(text_boxes)}")
        for i, box in enumerate(text_boxes):
            print(f"  OCR box{i + 1}: {box['label']} - coordinates: {box['box']}")

        # Sub-function: OCR text extraction
        def task_extract_text_inner(image_path_inner, text_boxes_inner):
            try:
                original_image = Image.open(image_path_inner)
                ocr_results = {}
                for idx, box_info in enumerate(text_boxes_inner):
                    label = box_info['label']
                    box = box_info['box']
                    x1, y1, x2, y2 = box
                    w, h = original_image.size
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 <= x1 or y2 <= y1:
                        print(f"[{image_name}][warning] OCR box {label}_{idx + 1} Invalid coordinates, skipped")
                        continue
                    cropped_image = original_image.crop((x1, y1, x2, y2))
                    if cropped_image.mode != "RGB":
                        cropped_image = cropped_image.convert("RGB")
                    cropped_array = np.array(cropped_image)
                    print(f"[{image_name}]Cropping image shape ({label}_{idx + 1}):", cropped_array.shape)
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
                    extracted_text = extracted_text.strip()
                    ocr_results[f"{label}_{idx + 1}"] = extracted_text
                return ocr_results
            except Exception as e:
                print(f"[{image_name}][OCR error]: {e}")
                return {"error": str(e)}

        # Sub-function: Scale bar CV analysis
        def task_analyze_graphics_inner(image_path_inner, scale_region_boxes_inner):
            if not scale_region_boxes_inner:
                return {"scale_cv": {"exists": False}}
            box = max(scale_region_boxes_inner, key=lambda b: (b['box'][2] - b['box'][0]) * (b['box'][3] - b['box'][1]))['box']
            img = cv2.imread(image_path_inner)
            if img is None:
                return {"scale_cv": {"exists": True, "error": "image_read_failed"}}
            x1, y1, x2, y2 = box
            scale_crop = img[y1:y2, x1:x2]
            w = x2 - x1
            h = y2 - y1
            aspect_ratio = round(w / max(h, 1), 2)
            img_height, img_width = img.shape[:2]
            box_center_x = (x1 + x2) / 2
            box_center_y = (y1 + y2) / 2
            vertical_position = "middle"
            if box_center_y < 0.3 * img_height:
                vertical_position = "top"
            elif box_center_y > 0.7 * img_height:
                vertical_position = "bottom"
            horizontal_position = "center"
            if box_center_x < 0.3 * img_width:
                horizontal_position = "left"
            elif box_center_x > 0.7 * img_width:
                horizontal_position = "right"
            position = f"{vertical_position}_{horizontal_position}"
            has_bar = False
            bar_pixel_length = None
            gray = cv2.cvtColor(scale_crop, cv2.COLOR_BGR2GRAY)
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
            opened = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bar_candidates = []
            for cnt in contours:
                x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
                aspect = w_cnt / max(h_cnt, 1)
                area_cnt = w_cnt * h_cnt
                if aspect > 5 and area_cnt > 500:
                    bar_candidates.append((x, y, w_cnt, h_cnt))
            if bar_candidates:
                has_bar = True
                bar = max(bar_candidates, key=lambda b: b[2])
                bar_pixel_length = int(bar[2])
            return {
                "scale_cv": {
                    "exists": True,
                    "aspect_ratio": aspect_ratio,
                    "relative_position": position,
                    "area": w * h,
                    "has_bar": has_bar,
                    "bar_pixel_length": bar_pixel_length,
                }
            }

        # Sub-function: Legend MLLM analysis
        def task_analyze_legend_mllm_inner(image_path_inner, legend_boxes_inner):
            if not legend_boxes_inner:
                print(f"[{image_name}]No legend box detected")
                return {"legend_data": []}
            print(f"[{image_name}]legend_boxes:", legend_boxes_inner)
            final_legend_data = []
            img = cv2.imread(image_path_inner)
            if img is None:
                return {"error": "Warning: Failed to read image"}
            h, w = img.shape[:2]
            for idx, box_dict in enumerate(legend_boxes_inner):
                try:
                    x1, y1, x2, y2 = box_dict['box']
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 <= x1 or y2 <= y1:
                        print(f"[{image_name}][WARN] number {idx} legend box(es) invalid: {x1},{y1},{x2},{y2}")
                        continue
                    legend_img = img[y1:y2, x1:x2]
                    max_side = 1024
                    lh, lw = legend_img.shape[:2]
                    scale = min(max_side / max(lh, lw), 1.0)
                    if scale < 1.0:
                        legend_img = cv2.resize(legend_img, (int(lw * scale), int(lh * scale)), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(TEMP_LEGEND_PATH, legend_img)
                    prompt = """
You are a professional map-legend analyzer. Carefully analyze this legend image and return a JSON array.
Each object in the array represents one legend item and must contain the following fields:
1. "value": text description of this item (e.g. "forest", "river", "crater").
2. "key_type": type of symbol (e.g. "color", "icon", "line", "pattern").
3. "key_data": specific data for the symbol. If "color", provide the HEX color value. If "icon" or "line", give a short English description such as "black_triangle" or "dashed_line".

"""
                    local_image_path = f"file://{os.path.abspath(TEMP_LEGEND_PATH)}"
                    messages = [
                        {"role": "user", "content": [{"image": local_image_path}]},
                        {"role": "user", "content": [{"text": prompt}]}
                    ]
                    print(f"[{image_name}]Calling Qwen-API to analyze legend box {idx} ...")
                    response = dashscope.MultiModalConversation.call(model='qwen-vl-plus', messages=messages)
                    if response.status_code != HTTPStatus.OK:
                        print(f"[{image_name}][WARN]  {idx} legend box(es) failed to analyze:{response.message}")
                        continue
                    content = response.output.choices[0].message.content
                    if isinstance(content, list):
                        content = "".join(map(str, content))
                    content = str(content).strip()
                    print(f"[{image_name}][DEBUG]  {idx}  box raw content repr:", repr(content)[:1000])
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
                    if isinstance(parsed_inner, str) and ('\\n' in parsed_inner or '\\t' in parsed_inner or '\\\\"' in parsed_inner):
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
                    if (json_text.startswith("'") and json_text.endswith("'")) or (json_text.startswith('"') and json_text.endswith('"')):
                        json_text = json_text[1:-1].strip()
                    print(f"[{image_name}][DEBUG] json_text snippet:", (json_text[:400] + '...') if len(json_text) > 400 else json_text)
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
                            print(f"[{image_name}][WARN]  {idx} legend box(es) failed JSON parsing:", e1, "| fallback error:", e2)
                            print("raw content repr (truncated):", repr(content)[:2000])
                            legend_json = []
                    final_legend_data.extend(legend_json)
                    for j, item in enumerate(legend_json):
                        try:
                            print(f"[{image_name}][Legend {idx + 1}.{j + 1}] value: {item.get('value')}, key_type: {item.get('key_type')}, key_data: {item.get('key_data')}")
                        except Exception:
                            print(f"[{image_name}][Legend {idx + 1}.{j + 1}] (Failed to read standard fields for this item)", repr(item)[:200])
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"[{image_name}][WARN]  {idx} legend box(es) encountered processing exceptions:{e}")
                    continue
            return {"legend_data": final_legend_data}

        print(f"\n[{image_name}]Starting OCR text extraction...")
        ocr_output = task_extract_text_inner(image_path, text_boxes)
        cv_output = task_analyze_graphics_inner(image_path, scale_block_boxes)
        legend_output = task_analyze_legend_mllm_inner(image_path, legend_boxes)

        print(f"\n[{image_name}]OCR output: {ocr_output}")
        print(f"[{image_name}]CV output: {cv_output}")
        print(f"[{image_name}]legend output: {legend_output}")

        # Step 4: Final aggregation (LLM generates target JSON)
        def final_summary_inner(ocr_output_inner, cv_output_inner, legend_output_inner):
            ocr_summary = {}
            if isinstance(ocr_output_inner, dict) and "error" not in ocr_output_inner:
                title_texts = []
                metadata_texts = []
                scale_texts = []
                for key, text in ocr_output_inner.items():
                    if key.startswith('title_'):
                        title_texts.append(text)
                    elif key.startswith('metadata_block_'):
                        metadata_texts.append(text)
                    elif key.startswith('scale_block_'):
                        scale_texts.append(text)
                ocr_summary = {
                    "title": "\n".join(title_texts) if title_texts else None,
                    "metadata_block": "\n".join(metadata_texts) if metadata_texts else None,
                    "scale_block": "\n".join(scale_texts) if scale_texts else None
                }
            else:
                ocr_summary = ocr_output_inner
            raw_data_for_llm = {
                "ocr_results": ocr_summary,
                "scale_analysis": cv_output_inner,
                "legend_data": legend_output_inner.get("legend_data", "[]")
            }
            final_prompt = f"""
### System Instruction ###
You are a professional cartographic-information extraction assistant.
Your task is to analyze the raw data extracted from a map provided below, and strictly populate a JSON structure based on these data.

### Data Extraction Rules ###
1. title: Extract from the `title` field of `ocr_results`. Merge multiple titles after filtering if present.
2. scale: Extract from the `scale_block` field of `ocr_results`. Look for patterns such as "1:100,000", "Scale: 1:50,000". If not found, extract from the `title` or `metadata_block` field of `ocr_results`.
3. projection: Extract from the `scale_block` field of `ocr_results`. Look for patterns such as "Mercator projection", "Lambert conformal conic". If not found, attempt extraction from the `metadata_block` or `title` field of `ocr_results`.
4. publication_agency: Extract from the `metadata_block` or `title` field of `ocr_results`. Look for agency names such as "USGS", "NASA", "Geological Survey".
5. publication_date: Extract from the `title` field of `ocr_results`. Look for year-formatted values such as "1975", "2022".
6. data_source: Extract relevant information from the `metadata_block` field of `ocr_results`; this may be a specific mission, sensor, etc.
7. legend_data: Use the provided `legend_data` directly.
8. If `scale_cv.exists` is true, treat it as supporting evidence even if no textual scale value is detected. If no scale information is extracted but `scale_cv.exists` is true, output: "Scale bar exists, but no specific numerical value detected".
9. scale_position: Extract from `scale_analysis.scale_cv.relative_position`; use null if unavailable.
10. bar_pixel_length: Extract from `scale_analysis.scale_cv.bar_pixel_length`; use null if unavailable.



### Output JSON template ###
{{
  "title": null,
  "scale": null,
  "scale_position": null,
  "bar_pixel_length": null,
  "projection": null,
  "publication_agency": [],
  "publication_date": null,
  "data_source": null,
  "legend_data": []
}}

### Input data (from Step 3) ###
{json.dumps(raw_data_for_llm, ensure_ascii=False, indent=2)}

Please output JSON strictly according to the template. Only use the input data provided above. Fill in null where data is unavailable.

"""
            try:
                print(f"\n[{image_name}]Invoking Qwen for final aggregation...")
                response = dashscope.Generation.call(model='qwen-turbo', prompt=final_prompt)
                if response.status_code == HTTPStatus.OK:
                    return response.output.text
                else:
                    print(f"[{image_name}]LLM aggregation error: {response.message}")
                    return None
            except Exception as e:
                print(f"[{image_name}][LLM aggregation error]: {e}")
                import traceback
                traceback.print_exc()
                return None

        # Generating final JSON result (LLM will be called only once)
        final_text = final_summary_inner(ocr_output, cv_output, legend_output)

        print(f"\n[{image_name}]--- final output ---")
        if final_text:
            print(final_text)

        print(f"\n[{image_name}]--- final output(RAW)---")
        print(final_text)

        if not final_text or not final_text.strip():
            print(f"[{image_name}][ERROR] Warning: LLM returned no content")
            return False

        # Light-weight JSON validation
        try:
            final_json = json.loads(final_text)
        except Exception as e:
            print(f"[{image_name}][ERROR] Warning: JSON parsing failed, skipping this image")
            print("RAW OUTPUT repr (truncated):")
            print(repr(final_text[:1000]))
            return False

        # Writing standardized JSON
        final_json_name = f"{image_name}.json"
        final_json_path = os.path.join(EXPORT_JSON_DIR, final_json_name)
        try:
            with open(final_json_path, "w", encoding="utf-8") as f:
                json.dump(final_json, f, ensure_ascii=False, indent=2)
            print(f"[{image_name}]JSON Successfully written: {final_json_path}")
        except Exception as e:
            print(f"[{image_name}]JSON File write failed: {e}")
            return False

        # Generating and saving metadata JSON
        try:
            metadata_info = {
                "image_name": image_name,
                "image_path": image_path,
                "total_detection_boxes": len(updated_boxes),
                "detection_boxes_detail": [
                    {
                        "label": box["label"],
                        "box_coordinates": box["box"],
                        "confidence": box.get("confidence", 0.0),
                        "area": box.get("area", 0)
                    } for box in updated_boxes
                ],
                "main_map_count": len([box for box in updated_boxes if box["label"] == "main_map"]),
                "inset_map_count": len([box for box in updated_boxes if box["label"] == "inset_map"])
            }
            if not os.path.exists(METADATA_DIR):
                os.makedirs(METADATA_DIR, exist_ok=True)
            metadata_json_name = f"{image_name}_metadata.json"
            metadata_json_path = os.path.join(METADATA_DIR, metadata_json_name)
            with open(metadata_json_path, "w", encoding="utf-8") as f:
                json.dump(metadata_info, f, ensure_ascii=False, indent=2)
            print(f"[{image_name}]Metadata JSON has been successfully written: {metadata_json_path}")
        except Exception as e:
            print(f"[{image_name}]Metadata JSON write failed: {e}")
            return False

        return True

    except Exception as e:
        print(f"\n[{image_name}]Overall image processing exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if os.path.exists(TEMP_LEGEND_PATH):
            try:
                os.remove(TEMP_LEGEND_PATH)
            except:
                pass


# -------------------------- main --------------------------
if __name__ == "__main__":
    image_tasks = get_image_list(IMAGE_INPUT_DIR)
    if not image_tasks:
        print("No images to process, program exiting!")
        exit()

    if not os.path.exists(EXPORT_JSON_DIR):
        os.makedirs(EXPORT_JSON_DIR, exist_ok=True)
        print(f"\n`JSON export directory pre-created`: {EXPORT_JSON_DIR}")

    if not os.path.exists(METADATA_DIR):
        os.makedirs(METADATA_DIR, exist_ok=True)
        print(f"`Metadata directory pre-created`: {METADATA_DIR}")

    success_count = 0
    fail_count = 0
    for image_path, image_name in image_tasks:
        if process_single_image(image_path, image_name):
            success_count += 1
        else:
            fail_count += 1

    print(f"\n" + "=" * 60)
    print(f"Batch processing completed! Total:{len(image_tasks)}  | Success:{success_count}  | Failed:{fail_count} ")
    print(f"All successfully generated JSON files are saved in:{EXPORT_JSON_DIR}")
    print(f"All successfully generated metadata JSON files are saved in:{METADATA_DIR}")
    print("=" * 60)