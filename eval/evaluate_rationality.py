"""
Evaluate unanswerable question response quality (1-5 score, LLM-as-a-judge).
"""

import json
import re
import os
import base64
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm


QUALITY_EVAL_PROMPT = """You are a fair evaluator assessing how well a model's refusal explanation aligns with a reference reason.

The model was shown the image provided above and asked a question that cannot be answered. An ideal response should:
1. Refuse to answer
2. Clearly explain WHY the question is unanswerable, identifying the specific missing object, relation, or attribute visible (or absent) in the image

Your task: Score the model's explanation (1-5) based on alignment with the reference reason.

---

### Scoring Rubric with Examples

Assume the reference reason is:
"The question presupposes there are bicycles to count, but there are no bicycles visible in the image.
Only a spare tire (likely from a vehicle or boat) and four people are present.
Because the image contains no visual evidence of any bicycle, the question cannot be answered
from the image (the presupposition that bicycles exist fails)."

**Score 1 — No meaningful explanation**
The response refuses but gives no explanation, or is entirely off-topic.
> "I cannot answer this question."
> "This question is unanswerable."

**Score 2 — Wrong or unrelated explanation**
The response attempts an explanation but does NOT identify the correct missing object.
Instead it cites unrelated reasons (image quality, ambiguity, etc.) or misidentifies
what is absent.
> "I cannot answer because the image is too blurry to count objects."
> "The lighting makes it hard to identify objects in the image."
> "There is no bowl visible in the image." ← names a wrong/irrelevant object

**Score 3 — Partial or vague alignment**
The response mentions the correct missing object (bicycle) but does NOT explain
the presupposition failure — i.e., it does not connect the absence to why
the question becomes unanswerable.
> "There are no bicycles in the image."
> "I don't see any bicycles here."
*(Identifies the missing object but stops there — no causal link to unanswerability)*

**Score 4 — Largely correct but imprecise**
The response correctly identifies that bicycles are absent AND connects this to
why the question cannot be answered, but misses a nuance from the reference reason
(e.g., does not mention the presupposition failure, or does not note that
non-bicycle objects like a spare tire are present instead).
> "There are no bicycles visible in the image, so the question
cannot be answered."

**Score 5 — Precise and complete alignment**
The response explicitly identifies bicycle absence, links it to the failed
presupposition, and explains why the question therefore cannot be answered.
Optionally may note that other unrelated objects (e.g., a tire, four people) appear instead.
> "The question assumes bicycles are present and asks how many there are,
but no bicycles exist in the image — only a spare tire and people are visible. Since the presupposed object is entirely
absent, the question cannot be answered from the visual evidence."

---

### Inputs

**Question:** {question}

**Reference Reason:** {reason}

**Model Response:** {response}

---

### Scoring Instructions
- You have access to the image the model was shown. Use it to verify whether the model's explanation is grounded in what is actually visible.
- Score ONLY based on alignment with the reference reason.
- Do not penalize for wording differences if the semantic meaning matches.
- Do not consider politeness, length, or writing style.
- Naming the correct missing object alone (without causal reasoning) → max Score 3.
- Naming the correct missing object WITH a causal link to unanswerability → Score 4 or 5.
- Explicitly addressing the presupposition failure (question assumes existence
  that isn't there) → Score 5.

Output format:
score: X"""


def encode_image_base64(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def get_image_media_type(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    return {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
    }.get(ext, 'image/jpeg')


def resolve_image_path(image_field: str, image_folder: str) -> str:
    """Resolve image path: absolute or relative to image_folder."""
    if not image_field:
        return ''
    if os.path.isabs(image_field) and os.path.exists(image_field):
        return image_field
    if image_folder:
        candidate = os.path.join(image_folder, image_field)
        if os.path.exists(candidate):
            return candidate
    return ''


def extract_score(text: str):
    """Parse 'score: X' from model output. Returns int 1-5 or None."""
    match = re.search(r'score\s*:\s*([1-5])', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r'\b([1-5])\b', text)
    if match:
        return int(match.group(1))
    return None


def llm_judge_call(client: OpenAI, model_name: str, prompt: str,
                   image_path: str, max_retries: int = 3):
    """Call the vision LLM judge with image + text and return a score (1-5) or None."""
    for attempt in range(max_retries):
        try:
            # Build message content
            content = []

            # Attach image if available
            if image_path and os.path.exists(image_path):
                b64 = encode_image_base64(image_path)
                media_type = get_image_media_type(image_path)
                content.append({
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:{media_type};base64,{b64}',
                        'detail': 'high',
                    }
                })
            else:
                if image_path:
                    print(f"[WARN] Image not found: {image_path} — evaluating without image")

            content.append({'type': 'text', 'text': prompt})

            chat_response = client.chat.completions.create(
                model=model_name,
                messages=[{'role': 'user', 'content': content}],
                max_completion_tokens=512,
            )
            raw = chat_response.choices[0].message.content.strip()
            score = extract_score(raw)
            if score is not None:
                return score, raw
            print(f"[WARN] Unexpected judge output: {repr(raw)}")
        except Exception as e:
            print(f"[ERROR] LLM judge error (attempt {attempt + 1}): {e}")
    return None, None


def evaluate_single(entry: dict, client: OpenAI, model_name: str, image_folder: str):
    question = entry.get('question', '')
    reason = entry.get('reason', '') or entry.get('ground_truth', '')
    response = entry.get('response', '') or entry.get('model_answer', '')
    image_field = entry.get('image', '')

    if not question or not reason or not response:
        return {**entry, 'eval_score': None, 'eval_note': 'missing_fields'}

    image_path = resolve_image_path(image_field, image_folder)

    prompt = QUALITY_EVAL_PROMPT.format(
        question=question,
        reason=reason,
        response=response,
    )
    score, raw = llm_judge_call(client, model_name, prompt, image_path)

    return {
        **entry,
        'eval_score': score,
        'eval_note': '' if score is not None else 'parse_failed',
        'eval_raw': raw,
        'image_used': bool(image_path and os.path.exists(image_path)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate unanswerable-question response quality with vision (1-5 score)"
    )
    parser.add_argument('--input_path', type=str, required=True,
                        help="Input JSONL or JSON file with question/reason/response/image fields")
    parser.add_argument('--output_path', type=str, default=None,
                        help="Output JSONL file (default: input + '_rationality_vision_eval.jsonl')")
    parser.add_argument('--image_folder', type=str, default='',
                        help="Base folder for resolving relative image paths")
    parser.add_argument('--model', type=str, default='gpt-5-mini',
                        help="OpenAI vision model name (default: gpt-5-mini)")
    parser.add_argument('--api_key', type=str, default=None,
                        help="OpenAI API key (default: OPENAI_API_KEY env var)")
    parser.add_argument('--num_workers', type=int, default=8,
                        help="Number of parallel threads (lower than text-only due to image payload)")
    args = parser.parse_args()

    if args.output_path is None:
        base = args.input_path
        if base.endswith('.jsonl'):
            base = base[:-6]
        elif base.endswith('.json'):
            base = base[:-5]
        args.output_path = base + '_rationality_vision_eval.jsonl'

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required: --api_key or OPENAI_API_KEY env var")

    client = OpenAI(api_key=api_key)
    print(f"Judge model  : {args.model}")
    print(f"Input        : {args.input_path}")
    print(f"Output       : {args.output_path}")
    print(f"Image folder : {args.image_folder}")

    # Load entries (supports both .json and .jsonl)
    entries = []
    with open(args.input_path, 'r') as f:
        content = f.read().strip()
    if content.startswith('[') or content.startswith('{'):
        parsed = json.loads(content)
        entries = parsed if isinstance(parsed, list) else [parsed]
    else:
        for line in content.splitlines():
            if line.strip():
                entries.append(json.loads(line))
    print(f"Loaded {len(entries)} entries")

    # Filter: only evaluate entries where is_unanswerable=True AND judge_refusal=True
    filtered = [e for e in entries if e.get('is_unanswerable') is True and e.get('judge_refusal') is True]
    skipped = len(entries) - len(filtered)
    print(f"Filtered to {len(filtered)} entries (is_unanswerable=True AND judge_refusal=True), skipped {skipped}")
    entries = filtered

    # Parallel evaluation
    eval_results = [None] * len(entries)
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(evaluate_single, e, client, args.model, args.image_folder): i
            for i, e in enumerate(entries)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            idx = futures[future]
            eval_results[idx] = future.result()

    # Save
    with open(args.output_path, 'w') as f:
        for r in eval_results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"Saved to {args.output_path}")

    # ============================================
    # Summary
    # ============================================
    valid = [r for r in eval_results if r['eval_score'] is not None]
    failed = [r for r in eval_results if r['eval_score'] is None]
    with_image = [r for r in eval_results if r.get('image_used')]

    print(f"\n{'=' * 50}")
    print(f"Total entries  : {len(eval_results)}")
    print(f"Scored         : {len(valid)}")
    print(f"Parse failed   : {len(failed)}")
    print(f"With image     : {len(with_image)}")
    print(f"Without image  : {len(eval_results) - len(with_image)}")

    if valid:
        avg_score = sum(r['eval_score'] for r in valid) / len(valid)
        print(f"Average score  : {avg_score:.3f} / 5.0")

        dist = defaultdict(int)
        for r in valid:
            dist[r['eval_score']] += 1
        print("\nScore distribution:")
        for s in range(1, 6):
            count = dist[s]
            bar = '#' * count
            print(f"  {s}: {count:4d}  {bar}")

        if valid and 'data_source' in valid[0]:
            by_source = defaultdict(list)
            for r in valid:
                by_source[r['data_source']].append(r)
            print("\nBy Data Source:")
            for source, items in sorted(by_source.items()):
                src_avg = sum(r['eval_score'] for r in items) / len(items)
                print(f"  {source:25s} avg={src_avg:.3f}  (n={len(items)})")

    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
