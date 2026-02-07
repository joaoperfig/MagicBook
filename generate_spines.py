#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from pathlib import Path

from PIL import Image

import spine_renderer


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROMPT_DIR = BASE_DIR / "prompts"
PIC_DIR = BASE_DIR / "pic"
OUTPUT_DIR = PIC_DIR / "spines"

CURATED_CSV = DATA_DIR / "booknames_curated.csv"
PROMPT_FILE = PROMPT_DIR / "spine_design_prompt.txt"
JSON_PROMPT_FILE = PROMPT_DIR / "spine_json_prompt.txt"
DEBUG_LAYOUT_JSONL = DATA_DIR / "spine_layout_debug.jsonl"

DEFAULT_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-5.2")
DEFAULT_LAYOUT_MODEL = os.getenv("OPENAI_LAYOUT_MODEL", "gpt-5.2-codex")
OUTPUT_SIZE = (480, 800)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com")


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def load_prompt() -> str:
    with PROMPT_FILE.open("r", encoding="utf-8") as handle:
        return handle.read()


def load_json_prompt() -> str:
    with JSON_PROMPT_FILE.open("r", encoding="utf-8") as handle:
        return handle.read()


def parse_time_label(label: str) -> int:
    parts = label.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid time format: {label}")
    hour = int(parts[0])
    minute = int(parts[1])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Time out of range: {label}")
    return hour * 60 + minute


def build_api_url(path: str) -> str:
    base = OPENAI_BASE_URL.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}{path}"
    return f"{base}/v1{path}"


def extract_response_text(response_json: dict) -> str:
    if isinstance(response_json.get("output_text"), str):
        return response_json["output_text"].strip()
    outputs = response_json.get("output") or []
    for output in outputs:
        for content in output.get("content") or []:
            if content.get("type") == "output_text" and content.get("text"):
                return str(content["text"]).strip()
    raise ValueError("No text content found in response.")


def post_json(url: str, payload: dict) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


def openai_chat_json(prompt: str, model: str) -> list[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    responses_payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Return strict JSON only.\n\n" + prompt,
                    }
                ],
            }
        ],
    }

    chat_payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Return strict JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
    }

    try:
        response_json = post_json(build_api_url("/responses"), responses_payload)
        content = extract_response_text(response_json)
    except RuntimeError as exc:
        if "HTTP 404" not in str(exc):
            raise
        response_json = post_json(build_api_url("/chat/completions"), chat_payload)
        content = response_json["choices"][0]["message"]["content"].strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("[")
        end = content.rfind("]")
        if start == -1 or end == -1:
            raise
        data = json.loads(content[start : end + 1])
    if not isinstance(data, list):
        raise ValueError("Expected JSON array of strings.")
    descriptions = [str(item).strip() for item in data if str(item).strip()]
    if len(descriptions) < 3:
        raise ValueError("Expected at least 3 descriptions.")
    return descriptions[:3]


def build_image_prompt(
    template: str, descriptions: list[str], title: str, author: str
) -> str:
    descriptions_block = "\n".join(
        f"Description {idx + 1}: {desc}" for idx, desc in enumerate(descriptions)
    ).strip()
    return (
        template.replace("{title}", title)
        .replace("{author}", author)
        .replace("{descriptions}", descriptions_block)
        .strip()
    )


def openai_chat_json_layouts(prompt: str, model: str) -> list[dict]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    responses_payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Return strict JSON only.\n\n" + prompt,
                    }
                ],
            }
        ],
    }

    chat_payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Return strict JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
    }

    try:
        response_json = post_json(build_api_url("/responses"), responses_payload)
        content = extract_response_text(response_json)
    except RuntimeError as exc:
        if "HTTP 404" not in str(exc):
            raise
        response_json = post_json(build_api_url("/chat/completions"), chat_payload)
        content = response_json["choices"][0]["message"]["content"].strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1:
            raise
        data = json.loads(content[start : end + 1])
    if isinstance(data, list):
        layouts = data
    elif isinstance(data, dict) and isinstance(data.get("layouts"), list):
        layouts = data["layouts"]
    else:
        raise ValueError("Expected JSON object with layouts array for layouts.")
    if len(layouts) < 3:
        raise ValueError("Expected at least 3 layouts.")
    return layouts[:3]


def render_layout(layout: dict) -> Image.Image:
    return spine_renderer.render_layout(layout)


def save_layout_image(image: Image.Image, output_path: Path) -> None:
    image = image.convert("RGBA")
    canvas = Image.new("RGBA", OUTPUT_SIZE, "#FF0000")
    paste_x = int(round((OUTPUT_SIZE[0] - image.width) / 2))
    canvas.alpha_composite(image, (paste_x, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")


def generate_and_save_images(
    output_paths: list[Path],
    layouts: list[dict],
) -> list[tuple[Path, dict]]:
    results: list[tuple[Path, dict]] = []
    for output_path, layout in zip(output_paths, layouts, strict=True):
        image = render_layout(layout)
        save_layout_image(image, output_path)
        results.append((output_path, layout))
    return results


def load_curated_rows() -> list[dict]:
    if not CURATED_CSV.exists():
        raise FileNotFoundError(f"Missing curated file: {CURATED_CSV}")
    with CURATED_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate book spine images for curated score=3 books."
    )
    parser.add_argument(
        "--start",
        help="Optional start time HH:MM to limit processing.",
    )
    parser.add_argument(
        "--end",
        help="Optional end time HH:MM to limit processing.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep seconds between books.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite images if they already exist.",
    )
    parser.add_argument(
        "--text-model",
        default=DEFAULT_TEXT_MODEL,
        help="OpenAI model for description generation.",
    )
    parser.add_argument(
        "--layout-model",
        default=DEFAULT_LAYOUT_MODEL,
        help="OpenAI model for JSON layout generation.",
    )
    return parser


def main() -> None:
    load_env_file(BASE_DIR / ".env")
    args = build_parser().parse_args()

    start_minutes = parse_time_label(args.start) if args.start else None
    end_minutes = parse_time_label(args.end) if args.end else None

    prompt_template = load_prompt()
    json_prompt_template = load_json_prompt()
    rows = load_curated_rows()
    score3_rows = [row for row in rows if row.get("score") == "3"]

    print(f"[init] score=3 rows: {len(score3_rows)}", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_LAYOUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    DEBUG_LAYOUT_JSONL.write_text("", encoding="utf-8")

    filtered_rows: list[dict] = []
    for row in score3_rows:
        time_label = row.get("time", "")
        if not time_label:
            continue
        total_minutes = parse_time_label(time_label)
        if start_minutes is not None and total_minutes < start_minutes:
            continue
        if end_minutes is not None and total_minutes > end_minutes:
            continue
        filtered_rows.append(row)

    print(f"[init] filtered rows: {len(filtered_rows)}", flush=True)

    design_queue_workers = 10
    json_queue_workers = 10
    suffixes = ["a", "b", "c"]

    with ThreadPoolExecutor(max_workers=design_queue_workers) as design_executor, ThreadPoolExecutor(
        max_workers=json_queue_workers
    ) as json_executor:
        design_futures: dict = {}

        for index, row in enumerate(filtered_rows, start=1):
            time_label = row.get("time", "")
            title = (row.get("title") or "").strip()
            author = (row.get("author") or "").strip()
            time_slug = time_label.replace(":", "")

            prompt = prompt_template.format(
                title=title,
                author=author,
                time_label=time_label,
            ).strip()

            output_paths: list[Path] = []
            task_meta: dict[Path, str] = {}
            for suffix in suffixes:
                output_path = OUTPUT_DIR / f"{time_slug}_{suffix}.png"
                if output_path.exists() and not args.overwrite:
                    print(f"[skip] {output_path.name} exists", flush=True)
                    continue
                output_paths.append(output_path)

            if not output_paths:
                continue

            print(
                f"[queue {index}/{len(filtered_rows)}] {time_label} {title} "
                f"({len(output_paths)} images)",
                flush=True,
            )

            book_context = {
                "time": time_label,
                "title": title,
                "author": author,
                "time_slug": time_slug,
                "output_paths": output_paths,
                "task_meta": task_meta,
            }
            future = design_executor.submit(openai_chat_json, prompt, args.text_model)
            design_futures[future] = book_context

        total_design = len(design_futures)
        total_images = sum(
            len(context["output_paths"]) for context in design_futures.values()
        )
        print(
            f"[init] queued books: {total_design}, queued images: {total_images}",
            flush=True,
        )

        json_futures: dict = {}
        pending_design = set(design_futures.keys())
        design_done = 0
        json_done = 0
        json_total = 0
        render_done = 0

        while pending_design or json_futures:
            pending = pending_design.union(json_futures.keys())
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                if future in pending_design:
                    pending_design.remove(future)
                    book_context = design_futures[future]
                    time_label = book_context["time"]
                    title = book_context["title"]
                    author = book_context["author"]
                    time_slug = book_context["time_slug"]
                    output_paths = book_context["output_paths"]
                    task_meta = book_context["task_meta"]

                    try:
                        descriptions = future.result()
                    except Exception as exc:
                        print(f"[error] description generation failed: {exc}", flush=True)
                        continue

                    design_done += 1
                    print(
                        f"[design {design_done}/{total_design}] {time_label} {title}",
                        flush=True,
                    )

                    for suffix, description in zip(suffixes, descriptions, strict=True):
                        output_path = OUTPUT_DIR / f"{time_slug}_{suffix}.png"
                        if output_path in output_paths:
                            task_meta[output_path] = description

                    json_prompt = build_image_prompt(
                        json_prompt_template,
                        descriptions,
                        title,
                        author,
                    )
                    json_future = json_executor.submit(
                        openai_chat_json_layouts, json_prompt, args.layout_model
                    )
                    json_futures[json_future] = (book_context, descriptions)
                    json_total += 1
                else:
                    book_context, descriptions = json_futures.pop(future)
                    time_label = book_context["time"]
                    title = book_context["title"]
                    author = book_context["author"]
                    time_slug = book_context["time_slug"]
                    output_paths = book_context["output_paths"]
                    task_meta = book_context["task_meta"]

                    try:
                        layouts = future.result()
                    except Exception as exc:
                        print(f"[error] layout generation failed: {exc}", flush=True)
                        continue

                    json_done += 1
                    print(
                        f"[json {json_done}/{json_total}] {time_label} {title}",
                        flush=True,
                    )

                    layouts_by_path = dict(
                        zip(
                            [OUTPUT_DIR / f"{time_slug}_{suffix}.png" for suffix in suffixes],
                            layouts,
                            strict=True,
                        )
                    )
                    to_render: list[tuple[Path, dict]] = [
                        (path, layouts_by_path[path]) for path in output_paths
                    ]

                    with ThreadPoolExecutor(max_workers=3) as executor:
                        future_map = {
                            executor.submit(generate_and_save_images, [path], [layout]): path
                            for path, layout in to_render
                        }
                        for render_future in as_completed(future_map):
                            output_path = future_map[render_future]
                            try:
                                saved = render_future.result()
                                saved_path, layout = saved[0]
                                original_description = task_meta.get(saved_path, "")
                                debug_entry = {
                                    "time": time_label,
                                    "title": title,
                                    "author": author,
                                    "description": original_description,
                                    "output": str(saved_path),
                                    "layout": layout,
                                }
                                with DEBUG_LAYOUT_JSONL.open("a", encoding="utf-8") as handle:
                                    handle.write(
                                        json.dumps(debug_entry, ensure_ascii=False) + "\n"
                                    )
                                render_done += 1
                                print(
                                    f"[saved {render_done}/{total_images}] {output_path.name}",
                                    flush=True,
                                )
                            except Exception as exc:
                                print(
                                    f"[error] {output_path.name} generation failed: {exc}",
                                    flush=True,
                                )

            if args.sleep:
                time.sleep(args.sleep)


if __name__ == "__main__":
    main()
