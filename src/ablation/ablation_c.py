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
IMAGE_INPUT_DIR = "path/to/ablation_C/input_images"
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png']
TEMP_LEGEND_PATH = "temp_legend.jpg"
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")  # set your DashScope (Qwen) API key, e.g. export DASHSCOPE_API_KEY=<your-key>
dashscope.api_key = DASHSCOPE_API_KEY
EXPORT_JSON_DIR = "path/to/ablation_C/extraction_results"
METADATA_DIR = "path/to/ablation_C/metadata_json"

# Initialize global tools
model = YOLO(MODEL_PATH)
ocr_engine = PaddleOCR(use_textline_orientation=True, lang='en')

# confidence threshold for each category
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

# -------------------------- Image compression function --------------------------
def compress_image(img, max_size_mb=20, quality=85):
    """Compress a PIL image to the given maximum file size (MB); returns the temporary file path."""
    img_copy = img.copy()
    max_dim = 3000  # cap the longest side in pixels
    if max(img_copy.size) > max_dim:
        ratio = max_dim / max(img_copy.size)
        new_size = (int(img_copy.size[0] * ratio), int(img_copy.size[1] * ratio))
        img_copy = img_copy.resize(new_size, Image.Resampling.LANCZOS)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        temp_path = tmp.name

    img_copy.save(temp_path, "JPEG", quality=quality, optimize=True)
    while os.path.getsize(temp_path) > max_size_mb * 1024 * 1024 and quality > 20:
        quality -= 10
        img_copy.save(temp_path, "JPEG", quality=quality, optimize=True)
    return temp_path

# -------------------------- Utility function: get the batch image list --------------------------
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

# -------------------------- Core function: process a single image --------------------------
def process_single_image(image_path, image_name):
    print(f"\n{'='*50}\nProcessing image: {image_name} - {image_path}\n{'='*50}")
    compressed_path = None
    try:
        # ---------- 1. Load the original image and compress it ----------
        print(f"[{image_name}] Loading original image...")
        original_img = Image.open(image_path)
        if original_img.mode != "RGB":
            original_img = original_img.convert("RGB")
        compressed_path = compress_image(original_img, max_size_mb=5, quality=85)
        print(f"[{image_name}] Image compressed to: {compressed_path}")

        # ---------- 2. YOLO detection (on the compressed image) ----------
        results = model(compressed_path, conf=0.05)
        detection_boxes = []
        class_names = model.names

        print(f"\n[{image_name}] YOLOv8n detection results:")
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
                    print(f"  {label_name}: {coords.tolist()} (confidence: {confidence:.2f}, threshold: {conf_thresh})")

        # -------------------------- Distinguish main map / inset maps --------------------------
        map_frame_boxes = [box for box in detection_boxes if box['label'] == 'map_frame']
        print(f"[{image_name}] Number of detected map_frame boxes: {len(map_frame_boxes)}")

        def distinguish_maps(detection_boxes_inner):
            map_frames = []
            other_boxes = []
            for box_info in detection_boxes_inner:
                if box_info['label'] == 'map_frame':
                    box = box_info['box']
                    area = (box[2]-box[0])*(box[3]-box[1])
                    box_info['area'] = area
                    map_frames.append(box_info)
                else:
                    other_boxes.append(box_info)
            if not map_frames:
                print(f"[{image_name}] Warning: no map_frame detected; returning all detection boxes")
                return detection_boxes_inner
            map_frames.sort(key=lambda x: x['area'], reverse=True)
            main_map = map_frames[0]
            main_map['label'] = 'main_map'
            for inset_map in map_frames[1:]:
                inset_map['label'] = 'inset_map'
            return [main_map]+map_frames[1:]+other_boxes

        updated_boxes = distinguish_maps(detection_boxes)
        print(f"\n[{image_name}] {len(updated_boxes)} elements detected in total:")
        for i, box in enumerate(updated_boxes):
            print(f"  {i+1}. {box['label']}: {box['box']}")

        # -------------------------- OCR text extraction (including legend) --------------------------
        text_labels = ['title', 'metadata_block', 'scale_block', 'legend']
        text_boxes = [box for box in updated_boxes if box['label'] in text_labels]
        scale_block_boxes = [box for box in updated_boxes if box['label']=='scale_block']

        print(f"\n[{image_name}] Number of text boxes to process with OCR: {len(text_boxes)}")
        for i, box in enumerate(text_boxes):
            print(f"  OCR box {i+1}: {box['label']} - coords: {box['box']}")

        def task_extract_text_inner(image_path_inner, text_boxes_inner):
            try:
                original_image = Image.open(image_path_inner)  # compressed image
                ocr_results = {}
                for idx, box_info in enumerate(text_boxes_inner):
                    label = box_info['label']
                    box = box_info['box']
                    x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(original_image.width, box[2]), min(original_image.height, box[3])
                    if x2 <= x1 or y2 <= y1:
                        print(f"[{image_name}] [WARN] OCR box {label}_{idx+1} has invalid coordinates; skipping")
                        continue
                    cropped_image = original_image.crop((x1, y1, x2, y2)).convert("RGB")
                    cropped_array = np.array(cropped_image)
                    result = ocr_engine.predict(cropped_array)
                    extracted_text = ""
                    if isinstance(result, list) and result:
                        for item in result:
                            if isinstance(item, dict):
                                extracted_text += "\n".join(item.get("rec_texts", [])) + "\n"
                            elif isinstance(item, (list, tuple)) and len(item)>=2:
                                extracted_text += str(item[1][0]) + "\n"
                    ocr_results[f"{label}_{idx+1}"] = extracted_text.strip()
                return ocr_results
            except Exception as e:
                print(f"[{image_name}] [OCR error]: {e}")
                return {"error": str(e)}

        def task_analyze_graphics_inner(image_path_inner, scale_region_boxes_inner):
            if not scale_region_boxes_inner:
                return {"scale_cv": {"exists": False}}
            box = max(scale_region_boxes_inner, key=lambda b: (b['box'][2]-b['box'][0])*(b['box'][3]-b['box'][1]))['box']
            img = cv2.imread(image_path_inner)
            if img is None:
                return {"scale_cv": {"exists": True, "error": "image_read_failed"}}
            x1, y1, x2, y2 = box
            scale_crop = img[y1:y2, x1:x2]
            w, h = x2-x1, y2-y1
            aspect_ratio = round(w/max(h,1),2)
            img_h, img_w = img.shape[:2]
            center_x, center_y = (x1+x2)/2, (y1+y2)/2
            vertical = "middle"
            if center_y<0.3*img_h: vertical="top"
            elif center_y>0.7*img_h: vertical="bottom"
            horizontal="center"
            if center_x<0.3*img_w: horizontal="left"
            elif center_x>0.7*img_w: horizontal="right"
            position = f"{vertical}_{horizontal}"
            # detect the scale bar
            has_bar=False; bar_pixel_length=None
            gray=cv2.cvtColor(scale_crop, cv2.COLOR_BGR2GRAY)
            _, bw=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(30,3))
            opened=cv2.morphologyEx(bw, cv2.MORPH_OPEN,kernel)
            contours,_=cv2.findContours(opened,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            bars=[]
            for cnt in contours:
                x,y,w_cnt,h_cnt=cv2.boundingRect(cnt)
                if w_cnt/h_cnt>5 and w_cnt*h_cnt>500: bars.append((x,y,w_cnt,h_cnt))
            if bars:
                has_bar=True
                bar_pixel_length=int(max(bars,key=lambda b:b[2])[2])
            return {"scale_cv":{"exists":True,"aspect_ratio":aspect_ratio,"relative_position":position,"area":w*h,"has_bar":has_bar,"bar_pixel_length":bar_pixel_length}}

        # run OCR and CV analysis (using the compressed image path)
        ocr_output = task_extract_text_inner(compressed_path, text_boxes)
        cv_output = task_analyze_graphics_inner(compressed_path, scale_block_boxes)

        # -------------------------- Final summary (LLM) --------------------------
        def final_summary_inner(ocr_output_inner, cv_output_inner):
            ocr_summary = {}
            if isinstance(ocr_output_inner, dict) and "error" not in ocr_output_inner:
                title_texts=[]
                metadata_texts=[]
                scale_texts=[]
                legend_texts=[]
                for key,text in ocr_output_inner.items():
                    if key.startswith("title_"): title_texts.append(text)
                    elif key.startswith("metadata_block_"): metadata_texts.append(text)
                    elif key.startswith("scale_block_"): scale_texts.append(text)
                    elif key.startswith("legend_"): legend_texts.append(text)
                ocr_summary={"title":"\n".join(title_texts) if title_texts else None,
                             "metadata_block":"\n".join(metadata_texts) if metadata_texts else None,
                             "scale_block":"\n".join(scale_texts) if scale_texts else None,
                             "legend_block":"\n".join(legend_texts) if legend_texts else None}
            else: ocr_summary=ocr_output_inner

            raw_data={"ocr_results":ocr_summary,"scale_analysis":cv_output_inner}
            final_prompt=f"""
### System instructions ###
You are a professional map cartographic information extraction assistant.
Based on the OCR results (title/metadata/scale/legend) and the scale analysis, strictly output the following JSON.
### Data extraction rules ###
1. Title (title): extract from the title field of ocr_results. If there are multiple titles, filter and merge them. Pay attention to the semantic accuracy of the title; avoid garbled or incoherent text, and check carefully.
2. Scale (scale): extract from the scale_block field of ocr_results. Look for patterns such as "1:100,000", "Scale: 1:50,000". If not found, extract from the title or metadata_block field of ocr_results.
3. Projection (projection): extract from the scale_block field of ocr_results. Look for patterns such as "Mercator projection", "Lambert conformal conic"; if not found, try the metadata_block or title field of ocr_results.
4. Publication agency (publication_agency): extract from the metadata_block or title field of ocr_results. Look for agency names such as "USGS", "NASA", "Geological Survey".
5. Publication date (publication_date): extract from the title field of ocr_results. Look for year formats such as "1975", "2022".
6. Data source (data_source): extract the relevant information from the metadata_block field of ocr_results; it may be a specific mission, sensor, etc.
7. Legend data (legend_data): the legend information is provided entirely by OCR; return a JSON array.
8. If scale_cv.exists is true, it can be treated as supporting evidence even if no numerical scale text is detected. If no scale information is extracted but scale_cv.exists is true, output: "Scale exists, but no specific value was detected".
9. Scale position (scale_position): extract from scale_analysis.scale_cv.relative_position; null if absent.
10. Scale bar pixel length (bar_pixel_length): extract from scale_analysis.scale_cv.bar_pixel_length; null if absent.

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
{json.dumps(raw_data,ensure_ascii=False,indent=2)}
"""
            try:
                response=dashscope.Generation.call(model='qwen-turbo',prompt=final_prompt)
                if response.status_code==HTTPStatus.OK:
                    return response.output.text
                else:
                    print(f"LLM summary error: {response.message}")
                    return None
            except Exception as e:
                print(f"[LLM summary error]: {e}")
                return None

        final_text=final_summary_inner(ocr_output, cv_output)
        print(f"\n[{image_name}] --- Final output ---\n{final_text}")

        if not final_text or not final_text.strip(): return False

        try:
            final_json=json.loads(final_text)
        except Exception as e:
            print(f"[ERROR] Failed to parse JSON: {e}")
            return False

        final_json_name=f"{image_name}.json"
        final_json_path=os.path.join(EXPORT_JSON_DIR, final_json_name)
        with open(final_json_path,"w",encoding="utf-8") as f: json.dump(final_json,f,ensure_ascii=False,indent=2)
        print(f"[{image_name}] JSON successfully written to: {final_json_path}")

        # ------------------- Metadata -------------------
        metadata_info={"image_name":image_name,"image_path":image_path,
                       "total_detection_boxes":len(updated_boxes),
                       "detection_boxes_detail":[{"label":b["label"],"box_coordinates":b["box"],"confidence":b.get("confidence",0.0),"area":b.get("area",0) if "area" in b else 0} for b in updated_boxes],
                       "main_map_count":len([b for b in updated_boxes if b["label"]=="main_map"]),
                       "inset_map_count":len([b for b in updated_boxes if b["label"]=="inset_map"])}
        os.makedirs(METADATA_DIR,exist_ok=True)
        metadata_json_path=os.path.join(METADATA_DIR,f"{image_name}_metadata.json")
        with open(metadata_json_path,"w",encoding="utf-8") as f: json.dump(metadata_info,f,ensure_ascii=False,indent=2)
        print(f"[{image_name}] Metadata JSON successfully written to: {metadata_json_path}")

        return True

    except Exception as e:
        print(f"[{image_name}] Processing exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # clean up the temporary compressed image file
        if compressed_path and os.path.exists(compressed_path):
            try: os.remove(compressed_path)
            except: pass
        # clean up the temporary legend file (not used in this experiment, but kept just in case)
        if os.path.exists(TEMP_LEGEND_PATH):
            try: os.remove(TEMP_LEGEND_PATH)
            except: pass

# -------------------------- Main program --------------------------
if __name__=="__main__":
    image_tasks=get_image_list(IMAGE_INPUT_DIR)
    if not image_tasks: exit()
    os.makedirs(EXPORT_JSON_DIR,exist_ok=True)
    os.makedirs(METADATA_DIR,exist_ok=True)
    success_count=0; fail_count=0
    for image_path,image_name in image_tasks:
        if process_single_image(image_path,image_name): success_count+=1
        else: fail_count+=1
    print(f"\n{'='*60}\nBatch processing complete! Total: {len(image_tasks)} | Success: {success_count} | Failed: {fail_count}\nJSON output: {EXPORT_JSON_DIR}\nMetadata output: {METADATA_DIR}\n{'='*60}")