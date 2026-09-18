import os
import json
import tempfile          # Added, for temporary files
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from ultralytics import YOLO  # Kept but unused (consistent with the structure of the A code)
import dashscope
from http import HTTPStatus
import re
import ast

os.environ['FLAGS_use_onednn'] = '1'
os.environ['FLAGS_enable_onednn_fusion'] = '0'
os.environ['FLAGS_onednn_dynamic_graph_fusion'] = '0'
os.environ['FLAGS_onednn_fusion_op_types'] = 'fused_conv2d'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -------------------------- Global configuration --------------------------
TEMP_LEGEND_PATH = "temp_full_image.jpg"   # Kept but no longer used (consistent with the A code)
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")  # set your DashScope (Qwen) API key, e.g. export DASHSCOPE_API_KEY=<your-key>
dashscope.api_key = DASHSCOPE_API_KEY

IMAGE_INPUT_DIR = "path/to/ablation_E/input_images"  # Keep the original directory
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png']
EXPORT_JSON_DIR = "path/to/ablation_E/extraction_results"
METADATA_DIR = "path/to/ablation_E/metadata_json"

# -------------------------- Image compression function (consistent with the E experiment) --------------------------
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
        ext = os.path.splitext(file_name)[1].lower()
        if ext in SUPPORTED_FORMATS:
            path = os.path.join(input_dir, file_name)
            name_no_suffix = os.path.splitext(file_name)[0]
            image_list.append((path, name_no_suffix))
    image_list.sort()
    print(f"Successfully obtained {len(image_list)} images")
    return image_list

# -------------------------- Core function --------------------------
def process_full_image(image_path, image_name):
    print(f"\nProcessing image: {image_name}")
    compressed_path = None
    try:
        # Load the original image and compress it
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        compressed_path = compress_image(img, max_size_mb=5, quality=85)
        print(f"Image compressed to temporary file: {compressed_path}")

        prompt = f"""
You are a professional map cartographic information extraction assistant.

Please analyze this full map image and directly identify the following information:
1. Title (title): if there are multiple titles, filter and merge them. Pay attention to the semantic accuracy of the title: no garbled text, no incoherent meaning; please check carefully.
2. Scale (scale): look for patterns such as "1:100,000", "Scale: 1:50,000", etc.
3. Projection (projection): look for patterns such as "Mercator projection", "Lambert conformal conic", etc.
4. Publication agency (publication_agency): look for agency names such as "USGS", "NASA", "Geological Survey", etc.
5. Publication date (publication_date): look for year formats such as "1975", "2022", etc.
6. Data source (data_source): extract the relevant information; it may be a specific mission, sensor, etc.
7. Legend data (legend_data): you are a professional map legend analyzer. Please carefully analyze this legend image and return a JSON array.
        Each object in the array represents one legend entry and must contain the following fields:
        1. 'value': the textual description of the entry (e.g. "forest", "river","crater").
        2. 'key_type': the type of the symbol (e.g. "color", "icon", "line", "pattern").
        3. 'key_data': the concrete data of the symbol (for 'color', give the HEX color value; for 'icon' or 'line', give a short English description, such as "black_triangle" or "dashed_line").
8. Scale information: determine whether a scale bar exists and compute its pixel length.
9. Scale position (scale_position): detect the position of the scale bar and describe its location (e.g. bottom_left, top_right, middle_center ).
Please output strictly according to the following JSON template, filling in only what was identified; use null when there is no data:

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
"""

        # Use the compressed image path
        local_image_path = f"file://{os.path.abspath(compressed_path)}"
        messages = [
            {"role": "user", "content": [{"image": local_image_path}]},
            {"role": "user", "content": [{"text": prompt}]}
        ]

        print(f"Calling Qwen to analyze the full image...")
        response = dashscope.MultiModalConversation.call(
            model='qwen-vl-plus',
            messages=messages
        )

        if response.status_code != HTTPStatus.OK:
            print(f"[WARN] MLLM analysis failed: {response.message}")
            return False

        raw_content = response.output.choices[0].message.content

        # ---------- Robust JSON extraction function (consistent with the A code) ----------
        def robust_json_parse(content):
            if isinstance(content, list):
                if len(content) > 0:
                    return robust_json_parse(content[0])
                else:
                    raise ValueError("Empty list in content")
            if isinstance(content, dict):
                if "text" in content:
                    content = content["text"]
                else:
                    return content
            if not isinstance(content, str):
                content = str(content)
            content = content.strip()
            import re
            code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
            if code_block:
                json_str = code_block.group(1).strip()
            else:
                start = content.find('{')
                end = content.rfind('}')
                if start != -1 and end != -1 and start < end:
                    json_str = content[start:end+1]
                else:
                    json_str = content
            # Clean control characters
            import re as re2
            def clean_control_chars(s):
                return re2.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', s)
            json_str = clean_control_chars(json_str)
            def try_fix_incomplete_json(s):
                if not s:
                    return None
                s = s.rstrip()
                if not s.endswith('}') and not s.endswith(']'):
                    quote_count = s.count('"') + s.count("'")
                    if quote_count % 2 == 1:
                        s = s + '"'
                    s = s + '}'
                try:
                    return json.loads(s)
                except json.JSONDecodeError:
                    for _ in range(5):
                        s = s + '}'
                        try:
                            return json.loads(s)
                        except:
                            continue
                    return None
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                try:
                    clean2 = re2.sub(r'[^\x20-\x7E\u4e00-\u9fff]', ' ', json_str)
                    return json.loads(clean2)
                except:
                    pass
                fixed = try_fix_incomplete_json(json_str)
                if fixed is not None:
                    return fixed
                try:
                    import ast
                    if (json_str.startswith('"') and json_str.endswith('"')) or (json_str.startswith("'") and json_str.endswith("'")):
                        json_str = ast.literal_eval(json_str)
                    result = ast.literal_eval(json_str)
                    return result
                except Exception as e2:
                    json_str = json_str.replace('\\n', '\n').replace('\\t', '\t')
                    try:
                        return json.loads(json_str)
                    except:
                        raise ValueError(f"Failed to parse JSON: {e}\nRaw content: {content[:500]}")
        # ---------- Invoke parsing ----------
        try:
            final_json = robust_json_parse(raw_content)
        except Exception as e:
            print(f"[ERROR] JSON parsing failed: {e}")
            print("RAW content repr:", repr(raw_content)[:1000])
            return False

        # Save the final JSON
        if not os.path.exists(EXPORT_JSON_DIR):
            os.makedirs(EXPORT_JSON_DIR, exist_ok=True)
        final_json_path = os.path.join(EXPORT_JSON_DIR, f"{image_name}.json")
        with open(final_json_path, "w", encoding="utf-8") as f:
            json.dump(final_json, f, ensure_ascii=False, indent=2)
        print(f"JSON saved: {final_json_path}")

        # Save metadata
        if not os.path.exists(METADATA_DIR):
            os.makedirs(METADATA_DIR, exist_ok=True)
        metadata_info = {
            "image_name": image_name,
            "image_path": image_path,
        }
        metadata_json_path = os.path.join(METADATA_DIR, f"{image_name}_metadata.json")
        with open(metadata_json_path, "w", encoding="utf-8") as f:
            json.dump(metadata_info, f, ensure_ascii=False, indent=2)
        print(f"Metadata saved: {metadata_json_path}")

        return True

    except Exception as e:
        print(f"[ERROR] Processing exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up the compressed image temporary file
        if compressed_path and os.path.exists(compressed_path):
            try:
                os.remove(compressed_path)
                print(f"Deleted temporary compressed file: {compressed_path}")
            except:
                pass
        # Clean up other temporary files (if any)
        if os.path.exists(TEMP_LEGEND_PATH):
            try:
                os.remove(TEMP_LEGEND_PATH)
            except:
                pass

# -------------------------- Main program --------------------------
if __name__ == "__main__":
    image_tasks = get_image_list(IMAGE_INPUT_DIR)
    if not image_tasks:
        print("No images to process, exiting the program!")
        exit()

    success_count = 0
    fail_count = 0
    for path, name in image_tasks:
        if process_full_image(path, name):
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "="*60)
    print(f"Batch processing complete, total {len(image_tasks)} images | succeeded: {success_count} | failed: {fail_count}")
    print(f"JSON output: {EXPORT_JSON_DIR}")
    print(f"Metadata output: {METADATA_DIR}")
    print("="*60)