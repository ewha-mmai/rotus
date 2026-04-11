"""
Convert dataset to VeRL training format for Thyme model.

Input format (e.g., HARDBench):
    {
        "question": "...",
        "answer": "...",
        "image": "TallyQA/VG_100K/2360883.jpg",   # relative path from image_root
        "data_source": "tallyqa",
        "is_unanswerable": false
    }

Usage:
    python scripts/convert_thyme.py \
        --input /path/to/data.json \
        --output /path/to/output.json \
        --image_root /workspace/data/Dataset

    # If images are stored locally at a different path (for size detection):
        --local_image_root /data1/hyemin/rotus/data/Dataset

    # Override prompts if needed:
        --system_prompt "Your custom system prompt"
        --user_template "Your custom user template with {question}, {image_path}, {image_size} placeholders"
"""

import json
import argparse
from PIL import Image


DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant.

Solve the following problem step by step, and optionally write Python code for image manipulation to enhance your reasoning process. The Python code will be executed by an external sandbox, and the processed image or result (wrapped in <sandbox_output></sandbox_output>) can be returned to aid your reasoning and help you arrive at the final answer.

**Reasoning & Image Manipulation (Optional but Encouraged):**
    * You have the capability to write executable Python code to perform image manipulations (e.g., cropping to a Region of Interest (ROI), resizing, rotation, adjusting contrast) or perform calculation for better reasoning.
    * The code will be executed in a secure sandbox, and its output will be provided back to you for further analysis.
    * All Python code snippets **must** be wrapped as follows:
    <code>
    ```python
    # your code.
    ```
    </code>
    * At the end of the code, print the path of the processed image (processed_path) or the result for further processing in a sandbox environment."""

DEFAULT_USER_TEMPLATE = """<image>
{question}

### User Image Path:** "{image_path}"
### User Image Size:** "{image_size}"

### **Output Format (strict adherence required):**

At each step, output one of the following:
- To run code: <think>Your reasoning here.</think><code>```python
# your code
```</code>
- To give the final answer: <think>Your reasoning here.</think><answer>Your final answer to the user's question goes here.</answer>
"""


def get_image_size(image_path):
    try:
        with Image.open(image_path) as img:
            return f"{img.width}x{img.height}"
    except Exception:
        return "Unknown"


def convert(input_path, output_path, image_root, system_prompt, user_template, local_image_root=None):
    with open(input_path) as f:
        data = json.load(f)

    converted = []
    for item in data:
        image_path = f"{image_root}/{item['image']}"
        local_path = f"{local_image_root}/{item['image']}" if local_image_root else image_path
        image_size = get_image_size(local_path)

        user_content = user_template.format(
            question=item['question'],
            image_path=image_path,
            image_size=image_size,
        )
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
            "env_name": "thyme",
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
    parser.add_argument('--local_image_root', default=None,
                        help='Local root for reading image size (if different from image_root)')
    parser.add_argument('--system_prompt', default=None)
    parser.add_argument('--user_template', default=None,
                        help='User prompt template with {question}, {image_path}, {image_size} placeholders')
    args = parser.parse_args()

    convert(
        args.input, args.output, args.image_root,
        system_prompt=args.system_prompt or DEFAULT_SYSTEM_PROMPT,
        user_template=args.user_template or DEFAULT_USER_TEMPLATE,
        local_image_root=args.local_image_root,
    )
