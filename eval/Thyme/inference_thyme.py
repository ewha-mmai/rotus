"""
Inference script for Thyme
"""

import os
import sys
import json
import copy
import pickle
from openai import OpenAI
import base64
from PIL import Image
import requests
import re
from tqdm import tqdm
import argparse


MAX_ITERATIONS = 5
MAX_NEW_TOKENS_PER_STEP = 2048
SPECIAL_STRING_LIST = ["</code>", "</answer>"]

CODE_REGEX = re.compile(
    r'<code>\s*(?:```\s*)?(?:python\s*)?([\s\S]*?)\s*(?:```\s*)?</code>',
    re.IGNORECASE
)


# ──────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────

def remove_unpickable_values(dictionary):
    if dictionary is None:
        return {}
    result = {}
    for key, value in dictionary.items():
        try:
            pickle.dumps(value)
            result[key] = value
        except (pickle.PicklingError, TypeError, AttributeError):
            pass
    return result


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def run_inference_single(client, model_name, image_path, system_prompt, user_prompt_text,
                         execute_code_in_sandbox, temp_output_dir,
                         max_turns=MAX_ITERATIONS, verbose=False):
    base64_image = encode_image_to_base64(image_path)
    image_path_for_code = image_path 

    # Initialize conversation history with system + user messages
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

    full_response = ""
    intermediate_steps = []
    previous_execution_context = {}  # maintains multi-turn execution state

    try:
        for iteration in range(max_turns):
            generated_content = []

            if verbose:
                print(f"\n--- Iteration {iteration + 1} ---")

            last_execution_context = copy.deepcopy(remove_unpickable_values(previous_execution_context))

            # vLLM API call
            response = client.chat.completions.create(
                model=model_name,
                messages=chat_message,
                max_tokens=MAX_NEW_TOKENS_PER_STEP,
                temperature=0.6,
                stop=SPECIAL_STRING_LIST,
            )

            generated_text_segment = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason

            if finish_reason == "stop":
                if "</code>" not in generated_text_segment and "<code>" in generated_text_segment:
                    generated_text_segment += "</code>"
                elif "</answer>" not in generated_text_segment and "<answer>" in generated_text_segment:
                    generated_text_segment += "</answer>"

            full_response += generated_text_segment

            if verbose:
                print(f"LLM (segment {iteration+1}):\n{generated_text_segment}")

            # Step info
            step_info = {
                'iteration': iteration + 1,
                'output': generated_text_segment,
                'has_code': '<code>' in generated_text_segment,
                'has_answer': '</answer>' in generated_text_segment,
                'code_executed': False,
                'code_success': None,
                'code_error': None,
            }
            intermediate_steps.append(step_info)

            if "</answer>" in generated_text_segment:
                generated_content.append(
                    {"type": "text", "text": generated_text_segment},
                )

            code_match = CODE_REGEX.search(generated_text_segment)

            if code_match:
                code_to_execute = code_match.group(1).strip()

                if verbose:
                    print(f"\033[31m--- Found Code Block ---\n{code_to_execute}\n-------------------------\033[0m")

                step_info['code_executed'] = True

                processed_img_paths, _, error_msg, current_execution_context = execute_code_in_sandbox(
                    code_to_execute, image_path_for_code,
                    temp_output_dir=temp_output_dir,
                    previous_execution_context=previous_execution_context
                )

                if not processed_img_paths:
                    previous_execution_context = last_execution_context
                    step_info['code_success'] = False
                    step_info['code_error'] = error_msg
                    if verbose:
                        print(f"Code execution failed: {error_msg}")
                    continue

                # Build sandbox output content
                has_valid_images = False
                generated_content += [
                    {"type": "text", "text": generated_text_segment},
                    {"type": "text", "text": "<sandbox_output>"},
                ]

                first_path = processed_img_paths[0]
                if os.path.exists(first_path):
                    # Convert output image paths to base64 and append
                    for img_path in processed_img_paths:
                        if os.path.exists(img_path):
                            if not has_valid_images:
                                has_valid_images = True
                            img_b64 = encode_image_to_base64(img_path)
                            generated_content.append(
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                            )
                else:
                    generated_content.append({"type": "text", "text": first_path})

                if has_valid_images or not os.path.exists(first_path):
                    generated_content.append({"type": "text", "text": "</sandbox_output>"})
                    step_info['code_success'] = True
                    previous_execution_context = current_execution_context

                    if verbose:
                        if has_valid_images:
                            print(f"Code executed successfully, output: {processed_img_paths}")
                        else:
                            print(f"Code executed successfully, stdout: {first_path}")
                else:
                    if verbose:
                        print('skip this generation due to error')
                    continue
            else:
                if "</answer>" not in generated_text_segment:
                    if verbose:
                        print('Warning: no code, no </answer>')
                    break

            if chat_message[-1]["role"] == "user":
                chat_message.append({"role": "assistant", "content": generated_content})
            elif chat_message[-1]["role"] == "assistant":
                last_content = chat_message[-1]["content"]
                if isinstance(last_content, list):
                    last_content.extend(generated_content)
                else:
                    chat_message[-1]["content"] = [
                        {"type": "text", "text": last_content}
                    ] + generated_content

            if "</answer>" in generated_text_segment:
                if verbose:
                    print("\033[32m--- Final answer tag found. ---\033[0m")
                break

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'full_response': full_response,
            'error': str(e),
            'turns': iteration + 1 if 'iteration' in dir() else 0,
            'intermediate_steps': intermediate_steps,
        }

    # Determine status
    status = 'success'
    if '</answer>' not in full_response:
        if iteration + 1 >= max_turns:
            status = 'max_turns_exceeded'
        else:
            status = 'no_answer'

    return {
        'status': status,
        'full_response': full_response,
        'turns': iteration + 1,
        'intermediate_steps': intermediate_steps,
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
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.join(image_base_dir, rel_path)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Run Thyme multi-turn inference with local sandbox execution')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to verl-format JSON file')
    parser.add_argument('--image_base_dir', type=str, required=True,
                        help='Base directory for resolving image paths')
    parser.add_argument('--output_path', type=str,
                        default='results_thyme.jsonl',
                        help='Path to save inference results')
    parser.add_argument('--api_base', type=str, default='http://localhost:8001/v1',
                        help='OpenAI API base URL')
    parser.add_argument('--thyme_path', type=str, required=True,
                        help='Path to Thyme model repo (for importing swift.trainers.sandbox)')
    parser.add_argument('--temp_output_dir', type=str,
                        default='./temp_processed_images',
                        help='Temporary directory for sandbox processed images')
    parser.add_argument('--max_turns', type=int, default=MAX_ITERATIONS,
                        help=f'Maximum number of iterations (default: {MAX_ITERATIONS})')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=None)
    parser.add_argument('--verbose', action='store_true',
                        help='Print intermediate reasoning steps')

    args = parser.parse_args()

    # Import execute_code_in_sandbox from Thyme's swift
    if args.thyme_path not in sys.path:
        sys.path.insert(0, args.thyme_path)
    from swift.trainers.sandbox import execute_code_in_sandbox

    # Create temp output dir
    os.makedirs(args.temp_output_dir, exist_ok=True)

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

    # Check for already processed samples
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

    # Process dataset
    results = []
    for idx, sample in enumerate(tqdm(dataset, desc="Thyme inference")):
        global_idx = start_idx + idx

        if global_idx in processed_indices:
            continue

        # Resolve image path from verl format
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

        # Extract system prompt from verl format
        system_prompt = ""
        raw_question = sample['extra_info']['question']
        for msg in sample['prompt']:
            if msg['role'] == 'system':
                system_prompt = msg['content']

        # Regenerate user prompt with actual image path and size
        try:
            with Image.open(image_path) as img:
                image_size = f"{img.width}x{img.height}"
        except Exception:
            image_size = "Unable to determine (error reading image)"

        user_prompt_text = f"""<image>
{raw_question}

### User Image Path:** "{image_path}"
### User Image Size:** "{image_size}"

### **Output Format (strict adherence required):**

<think>Your detailed reasoning process, including any code, should go here.</think>
<answer>Your final answer to the user's question goes here.</answer>
"""

        result = run_inference_single(
            client=client,
            model_name=model_name,
            image_path=image_path,
            system_prompt=system_prompt,
            user_prompt_text=user_prompt_text,
            execute_code_in_sandbox=execute_code_in_sandbox,
            temp_output_dir=args.temp_output_dir,
            max_turns=args.max_turns,
            verbose=args.verbose,
        )

        # Extract model answer from <answer> tags
        full_response = result.get('full_response', '')
        answer_match = re.search(r'<answer>(.*?)</answer>', full_response, re.DOTALL)
        model_answer = answer_match.group(1).strip() if answer_match else ''

        result_entry = {
            'index': global_idx,
            'question': sample['extra_info']['question'],
            'ground_truth': sample['extra_info']['answer'],
            'is_unanswerable': sample['extra_info']['is_unanswerable'],
            'data_source': sample['data_source'],
            'model_answer': model_answer,
            'full_response': full_response,
            'model_name': model_name,
            'status': result['status'],
            'turns': result['turns'],
            'intermediate_steps': result.get('intermediate_steps', []),
        }

        if 'error' in result:
            result_entry['error'] = result['error']

        results.append(result_entry)

        # Save incrementally
        with open(args.output_path, 'a') as f:
            f.write(json.dumps(result_entry, ensure_ascii=False) + '\n')

    # Print summary
    print(f"\n{'='*50}")
    print(f"Inference completed! Results saved to {args.output_path}")
    print(f"Total samples: {len(results)}")
    if results:
        success_count = sum(1 for r in results if r['status'] == 'success')
        max_turns_count = sum(1 for r in results if r['status'] == 'max_turns_exceeded')
        error_count = sum(1 for r in results if r['status'] == 'error')
        print(f"  Success: {success_count}")
        print(f"  Max turns exceeded: {max_turns_count}")
        print(f"  Error: {error_count}")

        answerable_count = sum(1 for r in results if not r['is_unanswerable'])
        unanswerable_count = sum(1 for r in results if r['is_unanswerable'])
        print(f"  Answerable: {answerable_count}, Unanswerable: {unanswerable_count}")

        # Code execution stats
        total_code_calls = sum(
            sum(1 for s in r.get('intermediate_steps', []) if s.get('code_executed'))
            for r in results
        )
        avg_turns = sum(r['turns'] for r in results) / len(results)
        print(f"  Total code executions: {total_code_calls}")
        print(f"  Avg turns per sample: {avg_turns:.2f}")


if __name__ == "__main__":
    main()
