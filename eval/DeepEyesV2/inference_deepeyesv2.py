import os
import json
from openai import OpenAI
import base64
from io import BytesIO
from PIL import Image
import requests
import time
import re
import random
from math import ceil
import textwrap
import autopep8
from tqdm import tqdm
import argparse


def generate_session_id():
    salted_str = str(int(time.time())) + str(random.randint(10000, 99999))
    salted_hash_str = str(hex(hash(salted_str.encode('utf-8')))).split('0x')[-1]
    return salted_hash_str


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def base64_to_pil_image(base64_string: str) -> Image.Image:
    if ',' in base64_string and base64_string.startswith('data:'):
        base64_string = base64_string.split(',', 1)[1]
    base64_string = base64_string.strip()
    missing_padding = len(base64_string) % 4
    if missing_padding:
        base64_string += '=' * (4 - missing_padding)
    image_data = base64.b64decode(base64_string)
    if len(image_data) == 0:
        raise ValueError("Decoded image data is empty")
    image_buffer = BytesIO(image_data)
    try:
        image = Image.open(image_buffer)
        image.load()
        return image
    except Exception as e:
        raise ValueError(f"Invalid image data: {e}. Data length: {len(image_data)} bytes")


def maybe_resize_image(image):
    height, width = image.height, image.width
    if max(height, width) / min(height, width) > 200:
        max_val = max(height, width)
        min_val = min(height, width)
        old_scale = max_val / min_val
        max_ratio = min(150, old_scale / 2)
        target_max = int(min_val * max_ratio)
        if height > width:
            new_height = target_max
            new_width = int(width * old_scale / max_ratio)
        else:
            new_width = target_max
            new_height = int(height * old_scale / max_ratio)
        image = image.resize((int(new_width), int(new_height)), Image.LANCZOS)
        height, width = image.height, image.width

    if min(height, width) >= 32:
        return image

    ratio = 32 / min(height, width)
    new_height = ceil(height * ratio)
    new_width = ceil(width * ratio)
    new_image = image.resize((new_width, new_height), Image.LANCZOS)
    return new_image


def fix_python_indentation(code):
    try:
        lines = [line.strip() for line in code.split('\n') if line.strip()]
        fixed_lines = []
        indent = 0
        for line in lines:
            if any(line.startswith(kw) for kw in ['except', 'elif', 'else', 'finally']):
                indent = max(0, indent - 1)
            fixed_lines.append('    ' * indent + line)
            if line.endswith(':'):
                indent += 1
        temp_code = '\n'.join(fixed_lines)
        dedented_code = textwrap.dedent(temp_code).strip()
        formatted_code = autopep8.fix_code(dedented_code, options={'aggressive': 2})
        return formatted_code
    except Exception as e:
        print('Code Format Error:', e)
        return code


def extract_python_code(action_string: str) -> str:
    tool_call_match = re.findall(r'<code>(.*?)</code>', action_string, re.DOTALL)
    if not tool_call_match:
        return None
    last_code_block = tool_call_match[-1]
    pattern = r'```python\s*\n(.*?)\n```'
    code_match = re.findall(pattern, last_code_block, re.DOTALL)
    if not code_match:
        return None
    return code_match[-1]


def request_jupyter_execution(code_string, code_sandbox_url, session_id, code_timeout=200, request_timeout=240):
    try:
        response = requests.post(
            code_sandbox_url,
            json={
                "session_id": session_id,
                "code": code_string,
                "timeout": code_timeout
            },
            timeout=request_timeout
        )
        if response.status_code != 200:
            print(f' [ERROR code] Jupyter sandbox returned status code {response.status_code}')
            return None
        try:
            resjson = response.json()
        except ValueError as json_err:
            print(f' [ERROR code] Failed to parse JSON response: {json_err}')
            return None
        result_dict = resjson.get('output')
        if result_dict is None:
            print(f' [ERROR code] No output field in response')
            return None
    except requests.exceptions.Timeout:
        print(f' [ERROR code] Request to Jupyter sandbox timed out after {request_timeout}s')
        return None
    except requests.exceptions.ConnectionError as conn_err:
        print(f' [ERROR code] Connection to Jupyter sandbox failed: {conn_err}')
        return None
    except Exception as err:
        print(f' [ERROR code] Request to Jupyter sandbox failed: {err}')
        return None

    image_pil_list = []
    image_base64_list = result_dict.get("images", [])
    for idx, img in enumerate(image_base64_list):
        try:
            if not img or len(img) == 0:
                continue
            img_pil = base64_to_pil_image(img)
            img_pil = maybe_resize_image(img_pil)
            image_pil_list.append(img_pil)
        except Exception as err:
            print(f' [ERROR code] Failed to decode image {idx}: {err}')
            continue

    return dict(
        status=resjson.get("status", "error"),
        execution_time=resjson.get("execution_time", -1.0),
        result=result_dict.get("result", ""),
        stdout=result_dict.get("stdout", ""),
        stderr=result_dict.get("stderr", ""),
        images=image_pil_list,
    )


RETURN_PROMPT = """Code execution result:
stdout:
```
{stdout}
```

stderr:
```
{stderr}
```

{image}
"""

INITIALIZATION_CODE_TEMPLATE = """
from PIL import Image
import base64
from io import BytesIO

_img_base64 = "{base64_image}"
image_1 = Image.open(BytesIO(base64.b64decode(_img_base64)))
"""


def run_inference_single(client, model_name, image_path, system_prompt, user_prompt_text,
                         code_sandbox_url, session_id, max_turns=5):
    base64_image = encode_image_to_base64(image_path)

    chat_message = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                {"type": "text", "text": user_prompt_text},
            ],
        }
    ]

    status = 'success'
    try_count = 0
    turn_idx = 0
    response_message = ""
    turn_history = []

    init_code_string = INITIALIZATION_CODE_TEMPLATE.format(base64_image=base64_image)
    init_ret = request_jupyter_execution(init_code_string, code_sandbox_url, session_id)

    if not init_ret or init_ret['status'] != 'success':
        return {
            'status': 'error',
            'response': '',
            'error': 'Initialization failed',
            'turns': 0
        }

    try:
        while True:
            if try_count >= max_turns:
                status = 'max_turns_exceeded'
                break

            params = {
                "model": model_name,
                "messages": chat_message,
                "temperature": 0.0,
                "max_tokens": 4096,
                "stop": ["</answer>", "</code>"],
            }

            response = client.chat.completions.create(**params)
            response_message = response.choices[0].message.content or ""

            finish_reason = response.choices[0].finish_reason
            if finish_reason == 'stop':
                if '<code>' in response_message and '</code>' not in response_message:
                    response_message += '</code>'
                elif '<answer>' in response_message and '</answer>' not in response_message:
                    response_message += '</answer>'

            chat_message.append({
                "role": "assistant",
                "content": response_message
            })

            turn_data = {
                "turn": turn_idx,
                "assistant_response": response_message,
                "action_type": None,
                "code_executed": None,
                "code_output": None
            }

            if '</answer>' in response_message and '<answer>' in response_message:
                turn_data["action_type"] = "answer"
                turn_history.append(turn_data)
                turn_idx += 1
                try_count += 1
                break

            if '<code>' in response_message:
                turn_data["action_type"] = "code"
                code_string = extract_python_code(response_message)
                if not code_string:
                    status = 'error'
                    turn_data["code_output"] = {"error": "Failed to extract code"}
                    turn_history.append(turn_data)
                    break

                code_string = fix_python_indentation(code_string)
                turn_data["code_executed"] = code_string

                exec_ret = request_jupyter_execution(code_string, code_sandbox_url, session_id)

                if not exec_ret:
                    status = 'error'
                    turn_data["code_output"] = {"error": "Execution failed"}
                    turn_history.append(turn_data)
                    break

                turn_data["code_output"] = {
                    "status": exec_ret.get("status"),
                    "stdout": exec_ret.get("stdout", ""),
                    "stderr": exec_ret.get("stderr", ""),
                    "num_images": len(exec_ret.get("images", []))
                }

                image_list = exec_ret.get('images', [])
                image_list = image_list[:10]
                code_result_string = RETURN_PROMPT.format(
                    stdout=exec_ret.get('stdout', ''),
                    stderr=exec_ret.get('stderr', ''),
                    image="Images:\n" + "<image>" * len(image_list) if len(image_list) > 0 else "",
                ).strip()

                add_message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": code_result_string},
                    ],
                }
                if image_list:
                    for img in image_list:
                        buffered = BytesIO()
                        if img.mode in ("RGBA", "P"):
                            img = img.convert("RGB")
                        img.save(buffered, format="JPEG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                        add_message["content"].append(
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                        )
                chat_message.append(add_message)

            turn_history.append(turn_data)
            turn_idx += 1
            try_count += 1

    except Exception as e:
        status = 'error'
        return {
            'status': status,
            'response': response_message,
            'error': str(e),
            'turns': turn_idx,
            'turn_history': turn_history
        }

    return {
        'status': status,
        'response': response_message,
        'turns': turn_idx,
        'turn_history': turn_history
    }


def resolve_image_path(verl_image_path, image_base_dir):
    rel_path = verl_image_path
    for prefix in ['/workspace/data/Dataset/', '/workspace/data/']:
        if rel_path.startswith(prefix):
            rel_path = rel_path[len(prefix):]
            break
    return os.path.join(image_base_dir, rel_path)


def main():
    parser = argparse.ArgumentParser(description='Run inference on combined_test_500_verl.json')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to verl-format JSON file')
    parser.add_argument('--image_base_dir', type=str, required=True,
                        help='Base directory for resolving image paths')
    parser.add_argument('--output_path', type=str,
                        default='results_test_500.jsonl',
                        help='Path to save inference results')
    parser.add_argument('--api_base', type=str, default='http://localhost:8001/v1',
                        help='OpenAI API base URL')
    parser.add_argument('--code_sandbox_url', type=str, default='http://127.0.0.1:18903/run_jupyter',
                        help='Code sandbox URL')
    parser.add_argument('--max_turns', type=int, default=5,
                        help='Maximum number of assistant turns (matches training max_assistant_turns=5)')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Start index for processing')
    parser.add_argument('--end_idx', type=int, default=None,
                        help='End index for processing')

    args = parser.parse_args()

    # Initialize OpenAI client
    client = OpenAI(
        api_key="EMPTY",
        base_url=args.api_base,
    )

    # Get model name from vLLM server
    response = requests.get(f"{args.api_base}/models")
    models = response.json()
    model_name = models['data'][0]['id']
    print(f"Using model: {model_name}")

    # Load dataset (verl format)
    with open(args.data_path, 'r') as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} samples from {args.data_path}")

    # Determine processing range
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx is not None else len(dataset)
    dataset = dataset[start_idx:end_idx]
    print(f"Processing samples {start_idx} to {end_idx}")

    # Check for already processed samples
    processed_indices = set()
    if os.path.exists(args.output_path):
        with open(args.output_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    processed_indices.add(entry['index'])
                except:
                    pass
        print(f"Found {len(processed_indices)} already processed samples, will skip them")

    # Process dataset
    results = []
    for idx, sample in enumerate(tqdm(dataset, desc="Processing combined test")):
        global_idx = start_idx + idx

        if global_idx in processed_indices:
            continue

        session_id = generate_session_id()

        # Resolve image path from verl format
        verl_image_path = sample['images'][0]['image']
        image_path = resolve_image_path(verl_image_path, args.image_base_dir)

        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            result_entry = {
                'index': global_idx,
                'question': sample['extra_info']['question'],
                'ground_truth': sample['extra_info']['answer'],
                'is_unanswerable': sample['extra_info']['is_unanswerable'],
                'data_source': sample['data_source'],
                'status': 'error',
                'error': 'Image not found',
                'model_answer': '',
                'turns': 0
            }
            results.append(result_entry)
            with open(args.output_path, 'a') as f:
                f.write(json.dumps(result_entry, ensure_ascii=False) + '\n')
            continue

        system_prompt = ""
        user_prompt_text = ""
        for msg in sample['prompt']:
            if msg['role'] == 'system':
                system_prompt = msg['content']
            elif msg['role'] == 'user':
                user_prompt_text = msg['content']
                if user_prompt_text.startswith('<image>\n'):
                    user_prompt_text = user_prompt_text[len('<image>\n'):]

        result = run_inference_single(
            client=client,
            model_name=model_name,
            image_path=image_path,
            system_prompt=system_prompt,
            user_prompt_text=user_prompt_text,
            code_sandbox_url=args.code_sandbox_url,
            session_id=session_id,
            max_turns=args.max_turns
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
