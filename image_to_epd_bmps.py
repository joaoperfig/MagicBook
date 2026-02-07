#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps


BLACK_VALUE_THRESHOLD = 0.5
RED_SATURATION_THRESHOLD = 0.5


def parse_size(size_text: str) -> tuple[int, int]:
    if "x" not in size_text:
        raise ValueError("Size must be in the form WIDTHxHEIGHT.")
    width_text, height_text = size_text.lower().split("x", 1)
    return int(width_text), int(height_text)


def resize_image(image: Image.Image, size: tuple[int, int], fit: str) -> Image.Image:
    if fit == "none":
        return image
    if fit == "contain":
        resized = ImageOps.contain(image, size)
        canvas = Image.new("RGB", size, "white")
        offset = ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2)
        canvas.paste(resized, offset)
        return canvas
    if fit == "cover":
        return ImageOps.fit(image, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    raise ValueError(f"Unknown fit mode: {fit}")


def convert_image(
    input_path: Path,
    output_dir: Path | None,
    size: tuple[int, int] | None,
    fit: str,
) -> tuple[Path, Path, Path]:
    image = Image.open(input_path).convert("RGB")
    if size:
        image = resize_image(image, size, fit)
    hsv = image.convert("HSV")
    hsv_pixels = hsv.load()

    width, height = image.size
    black_layer = Image.new("1", (width, height), 1)
    red_layer = Image.new("1", (width, height), 1)
    preview = Image.new("RGB", (width, height), "white")

    for y in range(height):
        for x in range(width):
            h, s, v = hsv_pixels[x, y]
            value = v / 255.0
            saturation = s / 255.0

            if value < BLACK_VALUE_THRESHOLD:
                black_layer.putpixel((x, y), 0)
                preview.putpixel((x, y), (0, 0, 0))
            else:
                if saturation >= RED_SATURATION_THRESHOLD:
                    red_layer.putpixel((x, y), 0)
                    preview.putpixel((x, y), (255, 0, 0))
                else:
                    preview.putpixel((x, y), (255, 255, 255))

    output_root = output_dir or input_path.parent
    output_root.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    black_path = output_root / f"{stem}_b.bmp"
    red_path = output_root / f"{stem}_r.bmp"
    preview_path = output_root / f"{stem}_preview.bmp"

    black_layer.save(black_path)
    red_layer.save(red_path)
    preview.save(preview_path)

    return black_path, red_path, preview_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert PNG/JPG into Waveshare black/red BMP layers and preview.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to input PNG or JPG image.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory for generated BMP files.",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="480x800",
        help="Optional output size like 800x480 (default: 480x800).",
    )
    parser.add_argument(
        "--fit",
        choices=("contain", "cover", "none"),
        default="contain",
        help="How to fit the image into the output size (default: contain).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    size = parse_size(args.size) if args.size else None
    black_path, red_path, preview_path = convert_image(
        args.input,
        args.output_dir,
        size,
        args.fit,
    )
    print(f"Wrote {black_path}")
    print(f"Wrote {red_path}")
    print(f"Wrote {preview_path}")


if __name__ == "__main__":
    main()
