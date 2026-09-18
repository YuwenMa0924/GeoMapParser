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
# Keep GPT for the legend MLLM; add Qwen for the final LLM aggregation
from openai import OpenAI
import base64
import dashscope
from http import HTTPStatus

Image.MAX_IMAGE_PIXELS = None
import ast
import re

# -------------------------- 0. Batch configuration --------------------------
MODEL_PATH = "path/to/weights/best.pt"
IMAGE_INPUT_DIR = "path/to/representative_100"
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png']

# GPT configuration (used only for legend MLLM visual analysis)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")  # set your API key via environment variable
client = OpenAI(api_key=OPENAI_API_KEY)
# Qwen configuration (used only for the final text-LLM aggregation; kept uniform across the comparison experiment)
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")  # set your API key via environment variable
dashscope.api_key = DASHSCOPE_API_KEY

TEMP_LEGEND_PATH = "temp_legend.jpg"

# Output paths (dedicated to the MLLM comparison experiment)
EXPORT_JSON_DIR = "path/to/comparison_gpt4o/results"
METADATA_DIR = "path/to/comparison_gpt4o/metadata"

# Initialize global tools
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

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Kept only: GPT-4o image analysis (legend MLLM, core of the comparison experiment)
def call_gpt4o_image(image_path, prompt):
    base64_image = encode_image(image_path)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        temperature=0
    )
    return response.choices[0].message.content

# -------------------------- Tool function --------------------------
def get_image_list(input_dir):
    image_list = []
    if not os.path.exists(input_dir):
        print(f"Error: Image input folder {input_dir} does not exist!")
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

# -------------------------- Core processing function --------------------------
def process_single_image(image_path, image_name):
    print(f"\n" + "=" * 50)
    print(f"Processing image: {image_name} - {image_path}")
    print("=" * 50)
    try:
        # Step 1: YOLO layout analysis
        results = model(image_path, conf=0.05)
        detection_boxes = []
        class_names = model.names

        print(f"\n[{image_name}]YOLO detection results:")
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                coords = box.xyxy[0].astype(int)
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                conf_thresh = CONF_THRESHOLDS.get(label_name, DEFAULT_CONF)
                if confidence >= conf_thresh:
                    detection_boxes.append({"label": label_name, "box": coords.tolist(), "confidence": round(confidence, 3)})
                    print(f"  {label_name}: {coords.tolist()} (confidence: {confidence:.2f}, thresh: {conf_thresh})")

        # Step 2: Main map / inset map classification
        def distinguish_maps(detection_boxes_inner):
            map_frames, other_boxes = [], []
            for box_info in detection_boxes_inner:
                if box_info['label'] == 'map_frame':
                    box = box_info['box']
                    box_info['area'] = (box[2]-box[0])*(box[3]-box[1])
                    map_frames.append(box_info)
                else:
                    other_boxes.append(box_info)
            if not map_frames:
                return detection_boxes_inner
            map_frames.sort(key=lambda x: x['area'], reverse=True)
            main_map = map_frames[0]
            main_map['label'] = 'main_map'
            inset_maps = [b for b in map_frames[1:]]
            for b in inset_maps: b['label'] = 'inset_map'
            return [main_map] + inset_maps + other_boxes

        updated_boxes = distinguish_maps(detection_boxes)
        print(f"\n[{image_name}]Total detected {len(updated_boxes)} elements")

        # Step 3: Extraction modules
        text_boxes = [b for b in updated_boxes if b['label'] in ['title','metadata_block','scale_block']]
        scale_block_boxes = [b for b in updated_boxes if b['label'] == 'scale_block']
        legend_boxes = [b for b in updated_boxes if b['label'] == 'legend']

        # OCR extraction
        def task_extract_text_inner(image_path_inner, text_boxes_inner):
            try:
                original_image = Image.open(image_path_inner)
                ocr_results = {}
                for idx, box_info in enumerate(text_boxes_inner):
                    label, box = box_info['label'], box_info['box']
                    x1,y1,x2,y2 = box
                    x1,y1 = max(0,x1), max(0,y1)
                    x2,y2 = min(original_image.size[0],x2), min(original_image.size[1],y2)
                    if x2<=x1 or y2<=y1: continue
                    cropped = original_image.crop((x1,y1,x2,y2)).convert("RGB")
                    result = ocr_engine.predict(np.array(cropped))
                    extracted_text = ""
                    if result:
                        for item in result:
                            if isinstance(item,dict): extracted_text += "\n".join(item.get("rec_texts",[])) + "\n"
                            elif isinstance(item,(list,tuple)) and len(item)>=2: extracted_text += str(item[1][0]) + "\n"
                    ocr_results[f"{label}_{idx+1}"] = extracted_text.strip()
                return ocr_results
            except Exception as e:
                print(f"[{image_name}][OCR error]: {e}")
                return {"error":str(e)}

        # Scale-bar CV analysis
        def task_analyze_graphics_inner(image_path_inner, scale_region_boxes_inner):
            if not scale_region_boxes_inner: return {"scale_cv":{"exists":False}}
            box = max(scale_region_boxes_inner, key=lambda b: (b['box'][2]-b['box'][0])*(b['box'][3]-b['box'][1]))['box']
            img = cv2.imread(image_path_inner)
            if img is None: return {"scale_cv":{"exists":True,"error":"image_read_failed"}}
            x1,y1,x2,y2 = box
            scale_crop = img[y1:y2, x1:x2]
            w,h = x2-x1, y2-y1
            aspect_ratio = round(w/max(h,1),2)
            # Position estimation
            img_h, img_w = img.shape[:2]
            cx, cy = (x1+x2)/2, (y1+y2)/2
            v_pos = "top" if cy<0.3*img_h else "bottom" if cy>0.7*img_h else "middle"
            h_pos = "left" if cx<0.3*img_w else "right" if cx>0.7*img_w else "center"
            position = f"{v_pos}_{h_pos}"
            # Scale-bar detection
            gray = cv2.cvtColor(scale_crop, cv2.COLOR_BGR2GRAY)
            _, bw = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(30,3))
            opened = cv2.morphologyEx(bw,cv2.MORPH_OPEN,kernel)
            contours,_ = cv2.findContours(opened,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            bar_candidates = [cv2.boundingRect(c) for c in contours if (cv2.boundingRect(c)[2]/max(cv2.boundingRect(c)[3],1))>5 and cv2.boundingRect(c)[2]*cv2.boundingRect(c)[3]>500]
            has_bar, bar_len = (True, max(bar_candidates, key=lambda b:b[2])[2]) if bar_candidates else (False, None)
            return {"scale_cv":{"exists":True,"aspect_ratio":aspect_ratio,"relative_position":position,"area":w*h,"has_bar":has_bar,"bar_pixel_length":bar_len}}

        # Legend MLLM: keep GPT-4o (the comparison-experiment variable)
        def task_analyze_legend_mllm_inner(image_path_inner, legend_boxes_inner):
            if not legend_boxes_inner: return {"legend_data":[]}
            final_legend_data = []
            img = cv2.imread(image_path_inner)
            h,w = img.shape[:2]
            for idx, box_dict in enumerate(legend_boxes_inner):
                try:
                    x1,y1,x2,y2 = box_dict['box']
                    x1,y1 = max(0,x1), max(0,y1)
                    x2,y2 = min(w,x2), min(h,y2)
                    if x2<=x1 or y2<=y1: continue
                    legend_img = img[y1:y2, x1:x2]
                    max_side = 1024
                    lh,lw = legend_img.shape[:2]
                    scale = min(max_side/max(lh,lw),1.0)
                    if scale<1.0: legend_img = cv2.resize(legend_img,(int(lw*scale),int(lh*scale)),interpolation=cv2.INTER_AREA)
                    cv2.imwrite(TEMP_LEGEND_PATH, legend_img)
                    prompt = """You are a professional map-legend analyzer. Carefully analyze this legend image and return a JSON array.
        Each object in the array represents one legend item and must contain the following fields:
        1. "value": text description of this item (e.g. "forest", "river", "crater").
        2. "key_type": type of symbol (e.g. "color", "icon", "line", "pattern").
        3. "key_data": specific data for the symbol. If "color", provide the HEX color value. If "icon" or "line", give a short English description such as "black_triangle" or "dashed_line"."""
                    print(f"[{image_name}]Calling GPT-4o to analyze the legend...")
                    content = call_gpt4o_image(TEMP_LEGEND_PATH, prompt).strip()
                    # JSON parsing
                    try:
                        json_text = re.search(r"```json\s*(.*?)\s*```", content, re.S).group(1) if re.search(r"```json\s*(.*?)\s*```", content, re.S) else re.search(r"(\[.*\])", content, re.S).group(1) if re.search(r"(\[.*\])", content, re.S) else content
                        legend_json = json.loads(json_text.strip())
                    except:
                        legend_json = []
                    final_legend_data.extend(legend_json)
                except Exception as e:
                    print(f"[{image_name}]Legend processing exception: {e}")
            return {"legend_data":final_legend_data}

        # Run the extraction steps
        ocr_output = task_extract_text_inner(image_path, text_boxes)
        cv_output = task_analyze_graphics_inner(image_path, scale_block_boxes)
        legend_output = task_analyze_legend_mllm_inner(image_path, legend_boxes)

        # ===================== Core change: the final aggregation LLM is switched back to Qwen-turbo (uniform comparison) =====================
        def final_summary_inner(ocr_output_inner, cv_output_inner, legend_output_inner):
            ocr_summary = {}
            if isinstance(ocr_output_inner, dict) and "error" not in ocr_output_inner:
                title_texts = [t for k,t in ocr_output_inner.items() if k.startswith('title_')]
                metadata_texts = [t for k,t in ocr_output_inner.items() if k.startswith('metadata_block_')]
                scale_texts = [t for k,t in ocr_output_inner.items() if k.startswith('scale_block_')]
                ocr_summary = {"title":"\n".join(title_texts) if title_texts else None,"metadata_block":"\n".join(metadata_texts) if metadata_texts else None,"scale_block":"\n".join(scale_texts) if scale_texts else None}
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

            # Invoke Qwen (the unified LLM)
            try:
                print(f"[{image_name}]Invoking Qwen-turbo for final aggregation...")
                response = dashscope.Generation.call(model='qwen-turbo', prompt=final_prompt)
                if response.status_code == HTTPStatus.OK:
                    return response.output.text
                else:
                    print(f"LLM error: {response.message}")
                    return None
            except Exception as e:
                print(f"LLM exception: {e}")
                return None

        # Generate the result
        final_text = final_summary_inner(ocr_output, cv_output, legend_output)
        if not final_text: return False

        # Parse and save the JSON
        final_json = json.loads(final_text)
        os.makedirs(EXPORT_JSON_DIR, exist_ok=True)
        os.makedirs(METADATA_DIR, exist_ok=True)
        with open(os.path.join(EXPORT_JSON_DIR, f"{image_name}.json"), "w", encoding="utf-8") as f:
            json.dump(final_json, f, ensure_ascii=False, indent=2)
        # Save the metadata
        metadata = {"image_name":image_name,"image_path":image_path,"total_detection_boxes":len(updated_boxes),"detection_boxes_detail":[{"label":b["label"],"box":b["box"],"conf":b.get("confidence",0)} for b in updated_boxes]}
        with open(os.path.join(METADATA_DIR, f"{image_name}_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return True

    except Exception as e:
        print(f"[{image_name}]Processing failed: {e}")
        return False
    finally:
        if os.path.exists(TEMP_LEGEND_PATH): os.remove(TEMP_LEGEND_PATH)

# -------------------------- Main program --------------------------
if __name__ == "__main__":
    image_tasks = get_image_list(IMAGE_INPUT_DIR)
    if not image_tasks: exit()
    success, fail = 0,0
    for path, name in image_tasks:
        if process_single_image(path, name): success+=1
        else: fail+=1
    print(f"\nBatch processing completed | Total: {len(image_tasks)} | Success: {success} | Failed: {fail}")