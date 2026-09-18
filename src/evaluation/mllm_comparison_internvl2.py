import os

os.environ['FLAGS_use_onednn'] = '1'  # Keep OneDNN acceleration enabled
os.environ['FLAGS_enable_onednn_fusion'] = '0'  # Disable global fusion
os.environ['FLAGS_onednn_dynamic_graph_fusion'] = '0'
os.environ['FLAGS_onednn_fusion_op_types'] = 'fused_conv2d'  # Disabled
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Resolve the OpenMP conflict (does not affect other packages)
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

# -------------------------- 0. Batch configuration (core change: batch-related settings added, single-image path commented out) --------------------------
MODEL_PATH = "path/to/weights/best.pt"
# Comment out the original single-image path; switch to a folder of images for batch input
# IMAGE_PATH = "path/to/test/813.jpg"
IMAGE_INPUT_DIR = "path/to/representative_100"  # Folder of images for batch processing (change to your own path)
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png']  # Supported image formats

TEMP_LEGEND_PATH = "temp_legend.jpg"
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")  # set your API key via environment variable
dashscope.api_key = DASHSCOPE_API_KEY
# JSON export folder (configuration kept from before; may be modified)
EXPORT_JSON_DIR = "path/to/comparison_internvl2/results"
# New: metadata folder configuration (change 1)
METADATA_DIR = "path/to/comparison_internvl2/metadata"  # Path where metadata JSON files are saved

# Initialize global tools once (avoids repeated initialization during batch processing and improves efficiency)
model = YOLO(MODEL_PATH)  # Global YOLO model
ocr_engine = PaddleOCR(use_textline_orientation=True, lang='en')  # Global OCR engine
# Per-class confidence thresholds (defined globally to avoid repeated construction)
CONF_THRESHOLDS = {
    "map_frame": 0.7,
    "title": 0.5,
    "legend": 0.5,
    "diagram": 0.5,
    "metadata_block": 0.5,
    "scale_block": 0.1,
    "north_arrow": 0.05
}
DEFAULT_CONF = 0.1  # Fallback threshold


# -------------------------- Tool function: Retrieve the list of images for batch processing --------------------------
def get_image_list(input_dir):
    """Iterate over the input folder and return the paths and extension-less names of all images in supported formats."""
    image_list = []
    if not os.path.exists(input_dir):
        print(f"Error: Image input folder {input_dir} does not exist!")
        return image_list
    # Iterate over all files in the folder
    for file_name in os.listdir(input_dir):
        file_suffix = os.path.splitext(file_name)[1].lower()
        # Keep only supported image formats
        if file_suffix in SUPPORTED_FORMATS:
            file_path = os.path.join(input_dir, file_name)  # Absolute image path
            file_name_no_suffix = os.path.splitext(file_name)[0]  # File name without extension (used to name the corresponding JSON)
            image_list.append((file_path, file_name_no_suffix))
    # Sort to guarantee a deterministic processing order
    image_list.sort()
    print(f"Successfully obtained {len(image_list)} images pending processing:")
    for idx, (path, name) in enumerate(image_list):
        print(f"  {idx + 1}. {name}{os.path.splitext(path)[1]} - {path}")
    return image_list


# -------------------------- Core function: Process a single image (all original logic encapsulated here) --------------------------
def process_single_image(image_path, image_name):
    """
    Process a single image and generate the corresponding JSON result.
    :param image_path: absolute path of the image
    :param image_name: extension-less image name (used to name the JSON file)
    :return: True on success, False on failure
    """
    print(f"\n" + "=" * 50)
    print(f"Processing image: {image_name} - {image_path}")
    print("=" * 50)
    try:
        # -------------------------- Step 1: Layout analysis (extract information-region boxes) --------------------------
        results = model(image_path, conf=0.05)
        detection_boxes = []
        class_names = model.names

        print(f"\n[{image_name}]YOLOv8n detection results:")
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                coords = box.xyxy[0].astype(int)
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                label_name = (
                    class_names[class_id]
                    if class_id < len(class_names)
                    else f"class_{class_id}"
                ).lstrip()

                # Retrieve the threshold for the corresponding class
                conf_thresh = CONF_THRESHOLDS.get(label_name, DEFAULT_CONF)

                if confidence >= conf_thresh:
                    detection_boxes.append({
                        "label": label_name,
                        "box": coords.tolist(),
                        "confidence": round(confidence, 3)
                    })
                    print(
                        f"  {label_name}: {coords.tolist()} "
                        f"(confidence: {confidence:.2f}, thresh: {conf_thresh})"
                    )

        # Print all classes contained in the YOLO model
        print(f"\n[{image_name}]The classes contained in the YOLO model: {class_names}")

        # -------------------------- Step 2: Logic classification (main map / inset map) --------------------------
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

        # -------------------------- Step 3: Parallel extraction (OCR / CV / MLLM) --------------------------
        # Select the boxes that require OCR processing
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

                    # 1. Crop the image
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

                    # 2. Convert to a numpy array
                    cropped_array = np.array(cropped_image)
                    print(f"[{image_name}]Cropping image shape ({label}_{idx + 1}):", cropped_array.shape)

                    # 3. Run OCR
                    result = ocr_engine.predict(cropped_array)

                    # 4. Safely extract the text
                    extracted_text = ""
                    if isinstance(result, list) and result:
                        # PaddleX OCR output may be a dict or a list
                        for item in result:
                            if isinstance(item, dict):
                                # Use rec_texts
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

        # Sub-function: Scale-bar CV analysis
        def task_analyze_graphics_inner(image_path_inner, scale_region_boxes_inner):
            if not scale_region_boxes_inner:
                return {"scale_cv": {"exists": False}}

            # Take the largest scale_block
            box = max(
                scale_region_boxes_inner,
                key=lambda b: (b['box'][2] - b['box'][0]) * (b['box'][3] - b['box'][1])
            )['box']
            # === Crop the scale_block image ===
            img = cv2.imread(image_path_inner)
            if img is None:
                return {"scale_cv": {"exists": True, "error": "image_read_failed"}}

            x1, y1, x2, y2 = box
            scale_crop = img[y1:y2, x1:x2]

            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1

            aspect_ratio = round(w / max(h, 1), 2)

            # # Relative-position estimation
            # === Refined position-determination logic (core modification) ===
            # 1. Read the overall image size first (reuse the img already loaded by CV2 to avoid reopening it)
            img_height, img_width = img.shape[:2]
            # 2. Compute the center coordinates of the scale-bar box (a more precise basis for position estimation)
            box_center_x = (x1 + x2) / 2
            box_center_y = (y1 + y2) / 2

            # 3. Determine the vertical position (top/bottom/middle)
            vertical_position = "middle"
            if box_center_y < 0.3 * img_height:  # Upper third of the image
                vertical_position = "top"
            elif box_center_y > 0.7 * img_height:  # Lower third of the image
                vertical_position = "bottom"

            # 4. Determine the horizontal position (left/right/center)
            horizontal_position = "center"
            if box_center_x < 0.3 * img_width:  # Left third of the image
                horizontal_position = "left"
            elif box_center_x > 0.7 * img_width:  # Right third of the image
                horizontal_position = "right"

            # 5. Combine into the final position (e.g. bottom_left, top_right, middle_center)
            position = f"{vertical_position}_{horizontal_position}"
            # position = "unknown"
            # if y1 > 0.6 * Image.open(image_path_inner).size[1]:
            #     position = "bottom"
            # if x1 < 0.3 * Image.open(image_path_inner).size[0]:
            #     position += "_left"

            # === Attempt to detect the scale bar (horizontal elongated strip) ===
            has_bar = False
            bar_pixel_length = None
            bar_segments = None

            gray = cv2.cvtColor(scale_crop, cv2.COLOR_BGR2GRAY)
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Horizontal structuring element, tailored to the scale bar
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
                # Take the longest bar
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

        # -------------------------- Legend MLLM analysis (InternVL2 / BLIP2 latest API) --------------------------
        # -------------------------- Legend MLLM analysis (fixed version) --------------------------
        from huggingface_hub import InferenceClient
        import cv2, json, ast

        HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")  # set your Hugging Face token via environment variable
        # Initialize the client globally to avoid repeated construction
        hf_client = InferenceClient(token=HF_API_TOKEN)

        def task_analyze_legend_mllm_inner(image_path_inner, legend_boxes_inner):
            """
            Fixed version: analyze the legend with the Hugging Face VQA interface; compatible with the BLIP2 model.
            """
            if not legend_boxes_inner:
                return {"legend_data": []}

            img = cv2.imread(image_path_inner)
            if img is None:
                return {"error": "Failed to read the image"}

            final_legend_data = []

            for idx, box_dict in enumerate(legend_boxes_inner):
                x1, y1, x2, y2 = box_dict['box']
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                # Crop the legend image and write it to the constant path
                legend_img = img[y1:y2, x1:x2]
                cv2.imwrite(TEMP_LEGEND_PATH, legend_img)

                with open(TEMP_LEGEND_PATH, "rb") as f:
                    img_bytes = f.read()

                # Simplified prompt adapted to BLIP2 output (forces the return of a JSON array)
                prompt = """Analyze the map legend, return a JSON array. Each object has value, key_type, key_data. Only return JSON."""

                try:
                    # Core fix: switch to the VQA interface (the standard interface supported by BLIP2)
                    response = hf_client.visual_question_answering(
                        model="Salesforce/blip2-flan-t5-xl",
                        image=img_bytes,
                        question=prompt
                    )

                    # Parse the returned result
                    content = str(response).strip()
                    print(f"[Legend {idx + 1}] raw output: {content[:100]}...")

                    # Safely parse the JSON
                    try:
                        legend_json = json.loads(content)
                        if isinstance(legend_json, list):
                            final_legend_data.extend(legend_json)
                    except:
                        try:
                            legend_json = ast.literal_eval(content)
                            if isinstance(legend_json, list):
                                final_legend_data.extend(legend_json)
                        except:
                            print(f"[Legend {idx + 1}] Parsing failed, skipped")
                            continue

                except Exception as e:
                    print(f"[Legend MLLM error] legend index {idx}: {str(e)}")
                    continue

            return {"legend_data": final_legend_data}

        # Run the three-step extraction
        print(f"\n[{image_name}]Starting OCR text extraction...")
        ocr_output = task_extract_text_inner(image_path, text_boxes)
        cv_output = task_analyze_graphics_inner(image_path, scale_block_boxes)
        legend_output = task_analyze_legend_mllm_inner(image_path, legend_boxes)

        print(f"\n[{image_name}]OCR output: {ocr_output}")
        print(f"[{image_name}]CV output: {cv_output}")
        print(f"[{image_name}]legend output: {legend_output}")

        # -------------------------- Step 4: Final aggregation (LLM generates the target JSON) --------------------------
        def final_summary_inner(ocr_output_inner, cv_output_inner, legend_output_inner):
            # Prepare the OCR data
            ocr_summary = {}
            if isinstance(ocr_output_inner, dict) and "error" not in ocr_output_inner:
                # Merge texts by label
                title_texts = []
                metadata_texts = []
                scale_texts = []

                for key, text in ocr_output_inner.items():
                    # Extract the label name
                    if key.startswith('title_'):
                        title_texts.append(text)
                    elif key.startswith('metadata_block_'):
                        metadata_texts.append(text)
                    elif key.startswith('scale_block_'):
                        scale_texts.append(text)

                # Build the structured data
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

        # Generate the final JSON result
        final_result = final_summary_inner(ocr_output, cv_output, legend_output)
        print(f"\n[{image_name}]--- final output ---")
        if final_result:
            print(final_result)

        final_text = final_summary_inner(ocr_output, cv_output, legend_output)

        print(f"\n[{image_name}]--- final output (RAW) ---")
        print(final_text)

        if not final_text or not final_text.strip():
            print(f"[{image_name}][ERROR] LLM returned no content")
            return False

        # ========= Lightweight JSON validation =========
        try:
            final_json = json.loads(final_text)
        except Exception as e:
            print(f"[{image_name}][ERROR] JSON parsing failed, skipping this map")
            print("RAW OUTPUT repr (truncated):")
            print(repr(final_text[:1000]))
            return False

        # ========= Write the standardized JSON =========
        final_json_name = f"{image_name}.json"
        final_json_path = os.path.join(EXPORT_JSON_DIR, final_json_name)

        try:
            with open(final_json_path, "w", encoding="utf-8") as f:
                json.dump(final_json, f, ensure_ascii=False, indent=2)
            print(f"[{image_name}]JSON successfully written: {final_json_path}")
        except Exception as e:
            print(f"[{image_name}]JSON file write failed: {e}")
            return False

        # ========= New: generate and save the metadata JSON (change 2) =========
        try:
            # 1. Gather the core metadata information
            metadata_info = {
                "image_name": image_name,
                "image_path": image_path,
                "total_detection_boxes": len(updated_boxes),  # Total number of boxes detected by YOLO
                "detection_boxes_detail": [  # Detailed information for each box
                    {
                        "label": box["label"],
                        "box_coordinates": box["box"],
                        "confidence": box.get("confidence", 0.0),
                        "area": box.get("area", 0) if "area" in box else 0
                    } for box in updated_boxes
                ],
                "main_map_count": len([box for box in updated_boxes if box["label"] == "main_map"]),  # Number of main maps
                "inset_map_count": len([box for box in updated_boxes if box["label"] == "inset_map"])  # Number of inset maps
            }
            # 2. Create the metadata folder
            if not os.path.exists(METADATA_DIR):
                os.makedirs(METADATA_DIR, exist_ok=True)
            # 3. Save the metadata JSON
            metadata_json_name = f"{image_name}_metadata.json"
            metadata_json_path = os.path.join(METADATA_DIR, metadata_json_name)
            with open(metadata_json_path, "w", encoding="utf-8") as f:
                json.dump(metadata_info, f, ensure_ascii=False, indent=2)
            print(f"[{image_name}]Metadata JSON successfully written: {metadata_json_path}")
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
        # Clean up the temporary legend file regardless of success or failure, to avoid leftovers
        if os.path.exists(TEMP_LEGEND_PATH):
            try:
                os.remove(TEMP_LEGEND_PATH)
            except:
                pass


# -------------------------- Main program: batch execution entry point --------------------------
if __name__ == "__main__":
    # 1. Retrieve the list of images to process
    image_tasks = get_image_list(IMAGE_INPUT_DIR)
    if not image_tasks:
        print("No images to process, program exiting!")
        exit()

    # 2. Automatically create the JSON export folder
    if not os.path.exists(EXPORT_JSON_DIR):
        os.makedirs(EXPORT_JSON_DIR, exist_ok=True)
        print(f"\nJSON export folder pre-created: {EXPORT_JSON_DIR}")

    # New: pre-create the metadata folder (change 3)
    if not os.path.exists(METADATA_DIR):
        os.makedirs(METADATA_DIR, exist_ok=True)
        print(f"Metadata folder pre-created: {METADATA_DIR}")

    # 3. Loop over and process each image
    success_count = 0
    fail_count = 0
    for image_path, image_name in image_tasks:
        if process_single_image(image_path, image_name):
            success_count += 1
        else:
            fail_count += 1

    # 4. Print the batch-processing summary
    print(f"\n" + "=" * 60)
    print(f"Batch processing completed! Total: {len(image_tasks)} | Success: {success_count} | Failed: {fail_count}")
    print(f"All successfully generated JSON files are saved in: {EXPORT_JSON_DIR}")
    print(f"All successfully generated metadata JSON files are saved in: {METADATA_DIR}")  # change 4
    print("=" * 60)