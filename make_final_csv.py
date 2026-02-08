#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

import generate_spines as gs
from image_to_epd_bmps import convert_image, parse_size


@dataclass(frozen=True)
class BookEntry:
    time_label: str
    time_slug: str
    title: str
    author: str
    image_path: Path


def parse_time_label(label: str) -> int:
    return gs.parse_time_label(label)


def format_time_label(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def format_hhmmss(seconds: int) -> str:
    seconds = seconds % 86400
    hour = seconds // 3600
    minute = (seconds % 3600) // 60
    sec = seconds % 60
    return f"{hour:02d}:{minute:02d}:{sec:02d}"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_png_metadata(path: Path) -> dict[str, str]:
    try:
        image = Image.open(path)
    except OSError:
        return {}
    metadata: dict[str, str] = {}
    info = getattr(image, "info", {}) or {}
    for key, value in info.items():
        if isinstance(value, str):
            metadata[str(key).lower()] = value.strip()
    text_data = getattr(image, "text", {}) or {}
    for key, value in text_data.items():
        if isinstance(value, str):
            metadata[str(key).lower()] = value.strip()
    return metadata


def build_schedule(times_seconds: list[int]) -> dict[int, tuple[int, int]]:
    if not times_seconds:
        return {}
    ordered = sorted(times_seconds)
    total = len(ordered)
    schedule: dict[int, tuple[int, int]] = {}
    for idx, current in enumerate(ordered):
        prev_time = ordered[idx - 1] if idx > 0 else ordered[-1] - 86400
        next_time = ordered[idx + 1] if idx < total - 1 else ordered[0] + 86400
        start_boundary = int((prev_time + current) / 2)
        end_boundary = int((current + next_time) / 2)
        start = start_boundary % 86400
        end = (end_boundary - 1) % 86400
        schedule[current] = (start, end)
    return schedule


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build final curated CSV with schedule spans."
    )
    parser.add_argument(
        "--output",
        default=str(gs.DATA_DIR / "booknames_curated_final.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--bmp-dir",
        default=str(gs.PIC_DIR / "final_bmp"),
        help="Output directory for BMP layers.",
    )
    parser.add_argument(
        "--bmp-size",
        default="480x800",
        help="BMP output size like 800x480 (default: 480x800).",
    )
    parser.add_argument(
        "--bmp-fit",
        choices=("contain", "cover", "none"),
        default="contain",
        help="How to fit the image into the output size (default: contain).",
    )
    args = parser.parse_args()

    rows = gs.load_curated_rows()
    rows_by_time: dict[str, dict] = {}
    for row in rows:
        time_label = (row.get("time") or "").strip()
        if not time_label:
            continue
        rows_by_time[time_label] = row

    score3_times = {
        time_label
        for time_label, row in rows_by_time.items()
        if str(row.get("score", "")).strip() == "3"
    }

    final_dir = gs.PIC_DIR / "final"
    final_paths = sorted(final_dir.glob("*.png"))
    if not final_paths:
        raise FileNotFoundError(f"No final images found in {final_dir}")

    images_by_time: dict[str, Path] = {}
    for path in final_paths:
        time_slug = path.stem
        if len(time_slug) != 4 or not time_slug.isdigit():
            continue
        time_label = f"{time_slug[:2]}:{time_slug[2:]}"
        images_by_time[time_label] = path

    curated_times = sorted(
        [time for time in images_by_time.keys() if time in score3_times],
        key=parse_time_label,
    )

    if not curated_times:
        raise RuntimeError("No curated score=3 times found with final images.")

    hashes_by_time = {
        time: sha256_path(images_by_time[time]) for time in curated_times
    }

    entries: list[BookEntry] = []
    for time_label in curated_times:
        row = rows_by_time.get(time_label, {})
        time_slug = time_label.replace(":", "")
        title = (row.get("title") or "").strip()
        author = (row.get("author") or "").strip()

        total_minutes = parse_time_label(time_label)
        if total_minutes >= 12 * 60:
            source_minutes = total_minutes - 12 * 60
            source_label = format_time_label(source_minutes)
            source_row = rows_by_time.get(source_label)
            if (
                source_label in images_by_time
                and hashes_by_time.get(source_label) == hashes_by_time.get(time_label)
                and source_row
            ):
                title = (source_row.get("title") or "").strip()
                author = (source_row.get("author") or "").strip()

        if not title or not author:
            metadata = load_png_metadata(images_by_time[time_label])
            title = title or metadata.get("title", "")
            author = author or metadata.get("author", "")

        entries.append(
            BookEntry(
                time_label=time_label,
                time_slug=time_slug,
                title=title,
                author=author,
                image_path=images_by_time[time_label],
            )
        )

    time_seconds = [parse_time_label(entry.time_label) * 60 for entry in entries]
    schedule = build_schedule(time_seconds)
    ordered_times = sorted(time_seconds)

    bmp_output_dir = Path(args.bmp_dir)
    bmp_size = parse_size(args.bmp_size) if args.bmp_size else None

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "time",
                "start",
                "end",
                "title",
                "author",
                "image",
                "image_black",
                "image_red",
                "image_preview",
            ],
        )
        writer.writeheader()
        for entry in entries:
            seconds = parse_time_label(entry.time_label) * 60
            start, end = schedule[seconds]
            if seconds == ordered_times[0]:
                start = 0
            if seconds == ordered_times[-1]:
                end = 86399

            bmp_black = bmp_output_dir / f"{entry.time_slug}_b.bmp"
            bmp_red = bmp_output_dir / f"{entry.time_slug}_r.bmp"
            bmp_preview = bmp_output_dir / f"{entry.time_slug}_preview.bmp"
            if not (bmp_black.exists() and bmp_red.exists() and bmp_preview.exists()):
                bmp_black, bmp_red, bmp_preview = convert_image(
                    entry.image_path, bmp_output_dir, bmp_size, args.bmp_fit
                )
            writer.writerow(
                {
                    "time": entry.time_label,
                    "start": format_hhmmss(start),
                    "end": format_hhmmss(end),
                    "title": entry.title,
                    "author": entry.author,
                    "image": entry.image_path.name,
                    "image_black": bmp_black.name,
                    "image_red": bmp_red.name,
                    "image_preview": bmp_preview.name,
                }
            )


if __name__ == "__main__":
    main()
