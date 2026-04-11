"""
Convert dataset to VeRL training format for Chain-of-Focus (CoF) model.

Input format (e.g., HARDBench):
    {
        "question": "...",
        "answer": "...",
        "image": "TallyQA/VG_100K/2360883.jpg",   # relative path from image_root
        "data_source": "tallyqa",
        "is_unanswerable": false
    }

Usage:
    python scripts/convert_cof.py \
        --input /path/to/data.json \
        --output /path/to/output.json \
        --image_root /workspace/data/Dataset

    # Override prompts if needed:
        --system_prompt "Your custom system prompt"
        --user_template "Your custom user template with {question} placeholder"
"""

import json
import argparse

DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant.

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

DEFAULT_USER_TEMPLATE = """<image>
Question: {question}
Think in the mind first, and then decide whether to call tools one or more times OR provide final answer. Format strictly as: <think>...</think> <tool_call>...</tool_call> <tool_call>...</tool_call> (if any tools needed) OR <answer>...</answer> (if no tools needed)."""


def convert(input_path, output_path, image_root, system_prompt, user_template):
    with open(input_path) as f:
        data = json.load(f)

    converted = []
    for item in data:
        image_path = f"{image_root}/{item['image']}"
        user_content = user_template.format(question=item['question'])
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        entry = {
            "images": [{"image": image_path}],
            "data_source": item['data_source'],
            "prompt": prompt,
            "reward_model": {
                "ground_truth": str(item['answer']),
                "style": "rule",
            },
            "env_name": "cof",
            "extra_info": {
                "answer": str(item['answer']),
                "question": item['question'],
                "is_unanswerable": item['is_unanswerable'],
            },
        }
        converted.append(entry)

    with open(output_path, 'w') as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)

    n_ans = sum(1 for x in converted if not x['extra_info']['is_unanswerable'])
    n_unans = sum(1 for x in converted if x['extra_info']['is_unanswerable'])
    print(f"Converted {len(converted)} entries → {output_path}")
    print(f"  Answerable: {n_ans}, Unanswerable: {n_unans}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input JSON file path')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    parser.add_argument('--image_root', default='/workspace/data/Dataset',
                        help='Root path prepended to image relative paths')
    parser.add_argument('--system_prompt', default=None)
    parser.add_argument('--user_template', default=None,
                        help='User prompt template with {question} placeholder')
    args = parser.parse_args()

    convert(
        args.input, args.output, args.image_root,
        system_prompt=args.system_prompt or DEFAULT_SYSTEM_PROMPT,
        user_template=args.user_template or DEFAULT_USER_TEMPLATE,
    )
