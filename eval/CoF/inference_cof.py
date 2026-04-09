"""
Inference script for Chain-of-Focus (CoF) model
"""

import os
import json
import re
import base64
import argparse
from io import BytesIO
from math import ceil

from PIL import Image
from openai import OpenAI
from tqdm import tqdm
import requests


# ──────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def encode_pil_image_to_base64(pil_image):
    buffered = BytesIO()
    if pil_image.mode in ("RGBA", "P"):
        pil_image = pil_image.convert("RGB")
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


# ──────────────────────────────────────────────
# Image crop/resize
# ──────────────────────────────────────────────

def smart_resize_simple(height, width, factor=28, min_pixels=4*28*28, max_pixels=3840*3840):
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    if h_bar * w_bar > max_pixels:
        beta = (max_pixels / (height * width)) ** 0.5
        h_bar = int(height * beta / factor) * factor
        w_bar = int(width * beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = (min_pixels / (height * width)) ** 0.5
        h_bar = max(factor, ceil(height * beta / factor) * factor)
        w_bar = max(factor, ceil(width * beta / factor) * factor)

    return h_bar, w_bar


def resize_image(original_image, factor=28, min_pixels=4*28*28, max_pixels=3840*3840):
    if isinstance(original_image, Image.Image):
        original_width, original_height = original_image.size
        new_height, new_width = smart_resize_simple(
            height=original_height, width=original_width,
            factor=factor, min_pixels=min_pixels, max_pixels=max_pixels
        )
        return original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    elif isinstance(original_image, tuple):
        original_width, original_height = original_image[0], original_image[1]
        new_height, new_width = smart_resize_simple(
            height=original_height, width=original_width,
            factor=factor, min_pixels=min_pixels, max_pixels=max_pixels
        )
        return (new_width, new_height)


def check_absolute_bbox_format(bbox, w, h):
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False, f"[WARNING] Invalid bbox format: {bbox}"
    if not all(isinstance(coord, (int, float)) for coord in bbox):
        return False, f"[WARNING] Non-numeric bbox values: {bbox}"

    x0, y0, x1, y1 = bbox
    if not (0 <= x0 < w and 0 <= y0 < h and 0 < x1 <= w and 0 < y1 <= h):
        return False, f"[WARNING] Bbox out of image bounds [0, 0, {w}, {h}]: {bbox}"
    if x1 <= x0 or y1 <= y0:
        return False, f"[WARNING] Invalid bbox (x1<=x0 or y1<=y0): {bbox}"

    return True, "Valid"


def find_absolute_bboxes(outputs_string, image, enlarge_factor=1):
    """Extract bbox_2d from <tool_call> tag and validate it."""
    W, H = image.size

    tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
    tool_call_match = re.search(tool_call_pattern, outputs_string, re.DOTALL)

    if not tool_call_match:
        print("[WARNING] No tool_call found in outputs_string.")
        return None

    tool_call_content = tool_call_match.group(1).strip()

    try:
        tool_call_data = json.loads(tool_call_content)
        bbox_content = tool_call_data["arguments"]["bbox_2d"]
        x0, y0, x1, y1 = bbox_content
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"[WARNING] Failed to parse tool_call: {e}")
        return None

    is_valid, error_msg = check_absolute_bbox_format([x0, y0, x1, y1], W, H)
    if not is_valid:
        print(f"[WARNING] {error_msg}")
        return None

    if enlarge_factor != 1:
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        bw = (x1 - x0) * enlarge_factor
        bh = (y1 - y0) * enlarge_factor
        x0 = max(0, cx - bw / 2)
        y0 = max(0, cy - bh / 2)
        x1 = min(W, cx + bw / 2)
        y1 = min(H, cy + bh / 2)

    return [int(x0), int(y0), int(x1), int(y1)]


def do_crop(image, decoded_text, scaleup_factor=2, enlarge_factor=1.5, min_pixels=4*28*28):
    """Parse bbox from tool_call and crop/resize the image."""
    bbox = find_absolute_bboxes(decoded_text, image, enlarge_factor)
    if bbox is None:
        print("[WARNING] Failed to extract bbox, returning original image.")
        return image, None

    try:
        cropped = image.crop(bbox)
        resize_target = (cropped.size[0] * scaleup_factor, cropped.size[1] * scaleup_factor)
        new_width, new_height = resize_image(resize_target, min_pixels=min_pixels)
        cropped = cropped.resize((new_width, new_height), Image.Resampling.LANCZOS).convert("RGB").copy()
        return cropped, bbox
    except Exception as e:
        print(f"[ERROR] Error cropping image: {e}")
        import traceback
        traceback.print_exc()
        return image, None


# ──────────────────────────────────────────────
# SYSTEM/USER PROMPT
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox_2d) and an optional object label.","parameters":{"properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox_2d"], "type":"object"},"args_format": "Format the arguments as a JSON object."}}
</tools>

For the function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

USER_PROMPT = "\nThink in the mind first, and then decide whether to call tools one or more times OR provide final answer. Format strictly as: <think>...</think> <tool_call>...</tool_call> <tool_call>...</tool_call> (if any tools needed) OR <answer>...</answer> (if no tools needed)."


# ──────────────────────────────────────────────
# Single sample inference
# ──────────────────────────────────────────────

def run_inference_single(client, model_name, image_path, question_text,
                         max_turns=6, max_new_tokens=512,
                         scaleup_factor=2, enlarge_factor=1.5, min_pixels=4*28*28):
    """Run multi-turn inference for a single CoF sample."""

    original_image = Image.open(image_path).convert('RGB')
    resized_image = resize_image(original_image, min_pixels=min_pixels)
    base64_image = encode_pil_image_to_base64(resized_image)

    formatted_question = f"Question: {question_text}{USER_PROMPT}"

    chat_message = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                {"type": "text", "text": formatted_question},
            ],
        }
    ]

    status = 'success'
    turn_idx = 0
    response_message = ""
    turn_history = []

    try:
        while turn_idx < max_turns:
            print(f"  [INFO] turn {turn_idx}")

            params = {
                "model": model_name,
                "messages": chat_message,
                "temperature": 0.0,
                "max_tokens": max_new_tokens,
            }

            response = client.chat.completions.create(**params)
            response_message = response.choices[0].message.content or ""

            chat_message.append({
                "role": "assistant",
                "content": [{"type": "text", "text": response_message}],
            })

            turn_data = {
                "turn": turn_idx,
                "assistant_response": response_message,
                "action_type": None,
                "bbox": None,
            }

            if '<tool_call>' in response_message:
                turn_data["action_type"] = "tool_call"

                cropped_image, bbox = do_crop(
                    resized_image, response_message,
                    scaleup_factor=scaleup_factor,
                    enlarge_factor=enlarge_factor,
                    min_pixels=min_pixels,
                )
                turn_data["bbox"] = bbox

                base64_cropped = encode_pil_image_to_base64(cropped_image)

                chat_message.append({
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_cropped}"}},
                        {"type": "text", "text": USER_PROMPT},
                    ],
                })
            else:
                turn_data["action_type"] = "answer"
                turn_history.append(turn_data)
                turn_idx += 1
                break

            turn_history.append(turn_data)
            turn_idx += 1

        if turn_idx >= max_turns and ('<answer>' not in response_message and '<tool_call>' in response_message):
            status = 'max_turns_exceeded'

    except Exception as e:
        status = 'error'
        import traceback
        traceback.print_exc()
        return {
            'status': status,
            'response': response_message,
            'error': str(e),
            'turns': turn_idx,
            'turn_history': turn_history,
        }

    return {
        'status': status,
        'response': response_message,
        'turns': turn_idx,
        'turn_history': turn_history,
    }


# ──────────────────────────────────────────────
# Path resolution
# ──────────────────────────────────────────────

def resolve_image_path(verl_image_path, image_base_dir):
    """Convert verl-format image path (/workspace/data/Dataset/...) to actual path."""
    rel_path = verl_image_path
    for prefix in ['/workspace/data/Dataset/', '/workspace/data/']:
        if rel_path.startswith(prefix):
            rel_path = rel_path[len(prefix):]
            break
    return os.path.join(image_base_dir, rel_path)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Run CoF multi-turn inference with image zoom tool')
    parser.add_argument('--data_path', type=str,
                        default='/workspace/data/Dataset/combined_test_500_verl.json',
                        help='Path to verl-format JSON file')
    parser.add_argument('--image_base_dir', type=str,
                        default='/workspace/data/Dataset',
                        help='Base directory for resolving image paths')
    parser.add_argument('--output_path', type=str,
                        default='results_cof.jsonl',
                        help='Path to save inference results')
    parser.add_argument('--api_base', type=str, default='http://localhost:8001/v1',
                        help='OpenAI API base URL')
    parser.add_argument('--max_turns', type=int, default=5,
                        help='Maximum number of assistant turns')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Maximum number of generated tokens per turn (matches vllm_inference.py)')
    parser.add_argument('--scaleup_factor', type=float, default=2,
                        help='Factor to scale up cropped region')
    parser.add_argument('--enlarge_factor', type=float, default=1.5,
                        help='Factor to enlarge bbox before cropping')
    parser.add_argument('--min_resolution', type=int, default=112,
                        help='Minimum resolution for image resize (min_pixels = min_resolution^2)')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=None)

    args = parser.parse_args()

    min_pixels = args.min_resolution ** 2 if args.min_resolution != 0 else 4 * 28 * 28
    print(f"[INFO] min_pixels: {min_pixels}")

    # Initialize OpenAI client
    client = OpenAI(
        api_key="EMPTY",
        base_url=args.api_base,
    )

    # Get model name from vLLM server
    response = requests.get(f"{args.api_base}/models")
    models = response.json()
    model_name = models['data'][0]['id']
    print(f"[INFO] Using model: {model_name}")

    # Load dataset (verl format)
    with open(args.data_path, 'r') as f:
        dataset = json.load(f)
    print(f"[INFO] Loaded {len(dataset)} samples from {args.data_path}")

    # Determine processing range
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx is not None else len(dataset)
    dataset = dataset[start_idx:end_idx]
    print(f"[INFO] Processing samples {start_idx} to {end_idx}")

    # Check for already processed samples (resume support)
    processed_indices = set()
    if os.path.exists(args.output_path):
        with open(args.output_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    processed_indices.add(entry['index'])
                except Exception:
                    pass
        print(f"[INFO] Found {len(processed_indices)} already processed samples, will skip them")

    results = []
    for idx, sample in enumerate(tqdm(dataset, desc="CoF inference")):
        global_idx = start_idx + idx

        if global_idx in processed_indices:
            continue

        verl_image_path = sample['images'][0]['image']
        image_path = resolve_image_path(verl_image_path, args.image_base_dir)

        if not os.path.exists(image_path):
            print(f"[WARNING] Image not found: {image_path}")
            result_entry = {
                'index': global_idx,
                'question': sample['extra_info']['question'],
                'ground_truth': sample['extra_info']['answer'],
                'is_unanswerable': sample['extra_info']['is_unanswerable'],
                'data_source': sample['data_source'],
                'status': 'error',
                'error': 'Image not found',
                'model_answer': '',
                'turns': 0,
            }
            results.append(result_entry)
            with open(args.output_path, 'a') as f:
                f.write(json.dumps(result_entry, ensure_ascii=False) + '\n')
            continue

        question_text = sample.get('extra_info', {}).get('question', '')
        if not question_text:
            for msg in sample.get('prompt', []):
                if msg.get('role') == 'user':
                    question_text = msg.get('content', '')
                    if question_text.startswith('<image>\n'):
                        question_text = question_text[len('<image>\n'):]
                    break

        result = run_inference_single(
            client=client,
            model_name=model_name,
            image_path=image_path,
            question_text=question_text,
            max_turns=args.max_turns,
            max_new_tokens=args.max_new_tokens,
            scaleup_factor=args.scaleup_factor,
            enlarge_factor=args.enlarge_factor,
            min_pixels=min_pixels,
        )

        # Extract model answer from <answer> tags
        model_answer_raw = result.get('response', '')
        answer_match = re.search(r'<answer>(.*?)</answer>', model_answer_raw, re.DOTALL)
        model_answer = answer_match.group(1).strip() if answer_match else ""

        result_entry = {
            'index': global_idx,
            'question': sample['extra_info']['question'],
            'ground_truth': sample['extra_info']['answer'],
            'is_unanswerable': sample['extra_info']['is_unanswerable'],
            'data_source': sample['data_source'],
            'model_answer': model_answer,
            'model_raw_response': model_answer_raw,
            'model_name': model_name,
            'status': result['status'],
            'turns': result['turns'],
            'turn_history': result.get('turn_history', []),
        }

        if 'error' in result:
            result_entry['error'] = result['error']

        results.append(result_entry)

        # Save incrementally
        with open(args.output_path, 'a') as f:
            f.write(json.dumps(result_entry, ensure_ascii=False) + '\n')



if __name__ == "__main__":
    main()
