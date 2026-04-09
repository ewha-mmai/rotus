import argparse
import base64
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List


def load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list.")
    return data


def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def read_image_as_data_url(image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/png"
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def extract_question(item: Dict[str, Any]) -> str:
    extra_info = item.get("extra_info")
    if isinstance(extra_info, dict):
        q = extra_info.get("question")
        if isinstance(q, str) and q.strip():
            return q.strip()

    prompt = item.get("prompt", "")
    if isinstance(prompt, str) and prompt:
        pattern = r"<\|vision_end\|>(.*?)<\|im_end\|>"
        match = re.search(pattern, prompt, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        return prompt.strip()

    return ""


def post_chat_completion(
    api_base: str,
    model_name: str,
    question: str,
    image_data_url: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
) -> str:
    url = f"{api_base.rstrip('/')}/chat/completions"

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    },
                    {
                        "type": "text",
                        "text": question,
                    },
                ],
            },
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer EMPTY",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            raw = res.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code}: {err_body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Connection error: {repr(e)}") from e

    obj = json.loads(raw)
    choices = obj.get("choices")
    if not choices:
        raise RuntimeError(f"No choices in response: {obj}")

    message = choices[0].get("message") or {}
    content = message.get("content")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        chunks = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                chunks.append(c.get("text", ""))
        return "".join(chunks)

    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_base", default="http://127.0.0.1:8011/v1")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--request_timeout", type=float, default=120.0)
    args = parser.parse_args()

    data = load_json_list(args.data_path)
    n = len(data)
    start = max(0, args.start_idx)
    end = n if args.end_idx < 0 else min(n, args.end_idx)

    if start >= end:
        raise ValueError(f"Invalid range: start={start}, end={end}, n={n}")

    ensure_dir(args.output_path)

    # Resume support: collect already processed indices
    processed_indices = set()
    if os.path.exists(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        processed_indices.add(json.loads(line)["index"])
                    except Exception:
                        pass
        print(f"[INFO] Found {len(processed_indices)} already processed samples, will skip them")

    with open(args.output_path, "a", encoding="utf-8") as out_f:
        for idx in range(start, end):
            if idx in processed_indices:
                continue

            item = data[idx]
            image_path = item["images"][0]["image"]
            question = extract_question(item)
            extra_info = item.get("extra_info", {})
            ground_truth = extra_info.get("answer", "")
            is_unanswerable = extra_info.get("is_unanswerable", False)

            t0 = time.time()
            try:
                image_data_url = read_image_as_data_url(image_path)
                pred = post_chat_completion(
                    api_base=args.api_base,
                    model_name=args.model_name,
                    question=question,
                    image_data_url=image_data_url,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    timeout=args.request_timeout,
                )

                rec: Dict[str, Any] = {
                    "index": idx,
                    "id": item.get("id"),
                    "data_source": item.get("data_source"),
                    "image": image_path,
                    "question": question,
                    "ground_truth": ground_truth,
                    "is_unanswerable": is_unanswerable,
                    "model_answer": pred,
                    "model_name": args.model_name,
                    "latency_sec": round(time.time() - t0, 4),
                }
            except Exception as e:
                rec = {
                    "index": idx,
                    "id": item.get("id"),
                    "data_source": item.get("data_source"),
                    "image": image_path,
                    "question": question,
                    "ground_truth": ground_truth,
                    "is_unanswerable": is_unanswerable,
                    "model_answer": "",
                    "model_name": args.model_name,
                    "error": repr(e),
                    "latency_sec": round(time.time() - t0, 4),
                }

            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()


if __name__ == "__main__":
    main()
