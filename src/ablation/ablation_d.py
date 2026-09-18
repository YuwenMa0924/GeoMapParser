import os
import tempfile          # Added, for temporary files
os.environ['FLAGS_use_onednn'] = '1'
os.environ['FLAGS_enable_onednn_fusion'] = '0'
os.environ['FLAGS_onednn_dynamic_graph_fusion'] = '0'
os.environ['FLAGS_onednn_fusion_op_types'] = 'fused_conv2d'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import json
from PIL import Image
from paddleocr import PaddleOCR
import cv2
import dashscope
from http import HTTPStatus
import re

Image.MAX_IMAGE_PIXELS = None

# -------------------------- 0. Batch configuration --------------------------
IMAGE_INPUT_DIR = "path/to/ablation_D/input_images"  # Folder containing the batch images
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png']

TEMP_IMAGE_PATH = "temp_image.jpg"
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")  # set your DashScope (Qwen) API key, e.g. export DASHSCOPE_API_KEY=<your-key>
dashscope.api_key = DASHSCOPE_API_KEY

EXPORT_JSON_DIR = "path/to/ablation_D/extraction_results"
METADATA_DIR = "path/to/ablation_D/metadata_json"

# OCR engine
ocr_engine = PaddleOCR(use_textline_orientation=True, lang='en')

# -------------------------- Image compression function (unified preprocessing) --------------------------
def compress_image(img, max_size_mb=20, quality=85):
    """Compress a PIL image to the specified maximum file size (MB) and return the temporary file path"""
    img_copy = img.copy()
    max_dim = 3000   # Limit on the maximum pixels of the longest edge
    if max(img_copy.size) > max_dim:
        ratio = max_dim / max(img_copy.size)
        new_size = (int(img_copy.size[0] * ratio), int(img_copy.size[1] * ratio))
        img_copy = img_copy.resize(new_size, Image.Resampling.LANCZOS)

    # Create a unique temporary file (suffix .jpg)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        temp_path = tmp.name

    # Initial save
    img_copy.save(temp_path, "JPEG", quality=quality, optimize=True)
    # If the file is still too large, progressively lower the quality
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
    print(f"Successfully obtained {len(image_list)} images pending processing:")
    for idx, (path, name) in enumerate(image_list):
        print(f"  {idx + 1}. {name}{os.path.splitext(path)[1]} - {path}")
    return image_list

# -------------------------- Core function: process a single image --------------------------
def process_single_image(image_path, image_name):
    print(f"\n{'='*50}\nProcessing image: {image_name} - {image_path}\n{'='*50}")
    compressed_path = None
    try:
        # ---------- 1. Load the original image and compress it (unified preprocessing) ----------
        original_img = Image.open(image_path)
        if original_img.mode != "RGB":
            original_img = original_img.convert("RGB")
        compressed_path = compress_image(original_img, max_size_mb=20, quality=85)
        print(f"[{image_name}]Image compressed to: {compressed_path}")

        # Load the compressed image for OCR (convert to a numpy array)
        compressed_img = Image.open(compressed_path)
        image_array = np.asarray(compressed_img, dtype=np.uint8)
        image_array = np.ascontiguousarray(image_array)

        # ---------- 2. OCR extraction (based on the compressed image) ----------
        extracted_text = ""
        try:
            print(f"[{image_name}]Starting OCR extraction...")
            ocr_result = ocr_engine.predict(image_array)
            if isinstance(ocr_result, list):
                for item in ocr_result:
                    if isinstance(item, dict):
                        texts = item.get("rec_texts", [])
                        extracted_text += "\n".join(texts) + "\n"
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        text = item[1][0] if isinstance(item[1], (list, tuple)) else str(item[1])
                        extracted_text += text + "\n"
            extracted_text = extracted_text.strip()
            print(f"[{image_name}]OCR finished, text length: {len(extracted_text)}")
        except Exception as e:
            print(f"[{image_name}]OCR crashed: {e}")
            extracted_text = ""

        ocr_output = {"full_image_text": extracted_text}

        # ---------- 3. CV analysis (scale bar detection, based on the compressed image) ----------
        cv_output = {"scale_cv": {"exists": False, "has_bar": False, "bar_pixel_length": None}}
        try:
            print(f"[{image_name}]Starting CV analysis...")
            img_cv = cv2.imread(compressed_path)
            if img_cv is not None:
                h_cv, w_cv = img_cv.shape[:2]
                if h_cv > 0 and w_cv > 0:
                    roi_y1 = int(h_cv * 2 / 3)
                    roi = img_cv[roi_y1:h_cv, 0:w_cv]
                    if roi.size > 0:
                        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
                        opened = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
                        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        has_bar = False
                        bar_pixel_length = None
                        for cnt in contours:
                            x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
                            aspect = w_cnt / max(h_cnt, 1)
                            area_cnt = w_cnt * h_cnt
                            if aspect > 5 and area_cnt > 500:
                                has_bar = True
                                bar_pixel_length = int(w_cnt)
                                break
                        cv_output = {
                            "scale_cv": {
                                "exists": True,
                                "aspect_ratio": round(w_cv / max(h_cv, 1), 2),
                                "relative_position": "bottom_center",
                                "area": w_cv * h_cv,
                                "has_bar": has_bar,
                                "bar_pixel_length": bar_pixel_length
                            }
                        }
                    else:
                        print(f"[{image_name}]ROI region is empty")
                else:
                    print(f"[{image_name}]Invalid image size: {w_cv}x{h_cv}")
            else:
                print(f"[{image_name}]OpenCV failed to read the image")
        except Exception as e:
            print(f"[{image_name}]Exception during CV analysis: {e}")

        # ---------- 4. LLM summarization ----------
        raw_data_for_llm = {
            "ocr_results": ocr_output,
            "scale_analysis": cv_output,
            "legend_data": []
        }

        final_prompt = f"""
### System instructions ###
You are a professional map cartographic information extraction assistant.
Based on the OCR results (title/metadata/scale/legend) and the scale analysis, output the following JSON exactly.
### Data extraction rules ###
1. Title (title): extract from the title field of ocr_results. If there are multiple titles, filter and merge them. Pay attention to the semantic accuracy of the title: no garbled text, no incoherent meaning; please check carefully.
2. Scale (scale): extract from the scale_block field of ocr_results. Look for patterns such as "1:100,000", "Scale: 1:50,000", etc. If none is found, extract it from the title or metadata_block field of ocr_results.
3. Projection (projection): extract from the scale_block field of ocr_results. Look for patterns such as "Mercator projection", "Lambert conformal conic", etc. If none is found, try extracting it from the metadata_block or title field of ocr_results.
4. Publication agency (publication_agency): extract from the metadata_block or title field of ocr_results. Look for agency names such as "USGS", "NASA", "Geological Survey", etc.
5. Publication date (publication_date): extract from the title field of ocr_results. Look for year formats such as "1975", "2022", etc.
6. Data source (data_source): extract the relevant information from the metadata_block field of ocr_results; it may be a specific mission, sensor, etc.
7. Legend data (legend_data): all legend information is provided by OCR; return a JSON array.
8. If scale_cv.exists is true, even if no numeric scale text is detected, it can still be treated as corroborating evidence. If no scale information is extracted but scale_cv.exists is true, output: "Scale exists, but no specific value was detected".
9. Scale position (scale_position): extract from scale_analysis.scale_cv.relative_position; null if absent.
10. Bar pixel length of the graphic scale (bar_pixel_length): extract from scale_analysis.scale_cv.bar_pixel_length; null if absent.

JSON template:
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
### Input data ###
{json.dumps(raw_data_for_llm,ensure_ascii=False,indent=2)}
"""
        final_text = None
        try:
            print(f"[{image_name}]Calling the LLM for final summarization...")
            response = dashscope.Generation.call(model='qwen-turbo', prompt=final_prompt)
            if response.status_code == HTTPStatus.OK:
                final_text = response.output.text
            else:
                print(f"[{image_name}]LLM summarization error: {response.message}")
        except Exception as e:
            print(f"[{image_name}]LLM call exception: {e}")

        if not final_text or not final_text.strip():
            print(f"[{image_name}][ERROR] LLM did not return valid content")
            return False

        # ---------- 5. Parse and save the JSON ----------
        try:
            final_json = json.loads(final_text)
        except json.JSONDecodeError as e:
            print(f"[{image_name}]JSON parsing failed: {e}")
            print("Raw response content:", final_text[:500])
            return False

        final_json_path = os.path.join(EXPORT_JSON_DIR, f"{image_name}.json")
        os.makedirs(EXPORT_JSON_DIR, exist_ok=True)
        with open(final_json_path, "w", encoding="utf-8") as f:
            json.dump(final_json, f, ensure_ascii=False, indent=2)
        print(f"[{image_name}]JSON saved: {final_json_path}")

        # Save metadata
        metadata_info = {
            "image_name": image_name,
            "image_path": image_path,
            "compressed_image_path": compressed_path,
            "ocr_text_length": len(extracted_text)
        }
        os.makedirs(METADATA_DIR, exist_ok=True)
        metadata_json_path = os.path.join(METADATA_DIR, f"{image_name}_metadata.json")
        with open(metadata_json_path, "w", encoding="utf-8") as f:
            json.dump(metadata_info, f, ensure_ascii=False, indent=2)
        print(f"[{image_name}]Metadata saved: {metadata_json_path}")

        # Explicitly release large objects (helps memory reclamation)
        del image_array
        del compressed_img
        return True

    except Exception as e:
        print(f"[{image_name}]Top-level exception caught: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up the compressed image temporary file
        if compressed_path and os.path.exists(compressed_path):
            try:
                os.remove(compressed_path)
                print(f"[{image_name}]Deleted temporary compressed file: {compressed_path}")
            except:
                pass

# -------------------------- Main program --------------------------
if __name__ == "__main__":
    image_tasks = get_image_list(IMAGE_INPUT_DIR)
    if not image_tasks:
        print("No images to process, exiting the program!")
        exit()

    os.makedirs(EXPORT_JSON_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)

    success_count, fail_count = 0, 0
    for image_path, image_name in image_tasks:
        if process_single_image(image_path, image_name):
            success_count += 1
        else:
            fail_count += 1

    print(f"\n{'='*60}")
    print(f"Batch processing complete! Total: {len(image_tasks)} images | Succeeded: {success_count} | Failed: {fail_count}")
    print(f"JSON export path: {EXPORT_JSON_DIR}")
    print(f"Metadata export path: {METADATA_DIR}")
    print(f"{'='*60}")