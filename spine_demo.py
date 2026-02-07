#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(__file__).resolve().parent
PIC_DIR = BASE_DIR / "pic"
LIB_DIR = BASE_DIR / "lib"
if LIB_DIR.exists():
    sys.path.append(str(LIB_DIR))

FRAME_WIDTH = 480
FRAME_HEIGHT = 800
SPINE_WIDTH = 300
SPINE_HEIGHT = 800

TITLE_TEXT = "1984"
AUTHOR_TEXT = "George Orwell"


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_path = PIC_DIR / "Font.ttc"
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size)
    return ImageFont.load_default()


def draw_rotated_text(
    base: Image.Image,
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    center: tuple[int, int],
    angle: int = 90,
) -> None:
    dummy = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    draw_dummy = ImageDraw.Draw(dummy)
    bbox = draw_dummy.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    text_img = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))
    draw_text = ImageDraw.Draw(text_img)
    draw_text.text((0, 0), text, font=font, fill=fill)

    rotated = text_img.rotate(angle, expand=True)
    paste_x = center[0] - rotated.width // 2
    paste_y = center[1] - rotated.height // 2
    base.alpha_composite(rotated, (paste_x, paste_y))


def main() -> None:
    frame = Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), "white")
    frame_rgba = frame.convert("RGBA")

    spine_left = (FRAME_WIDTH - SPINE_WIDTH) // 2
    spine_top = (FRAME_HEIGHT - SPINE_HEIGHT) // 2
    spine_right = spine_left + SPINE_WIDTH
    spine_bottom = spine_top + SPINE_HEIGHT

    draw = ImageDraw.Draw(frame_rgba)
    draw.rectangle(
        [spine_left, spine_top, spine_right, spine_bottom],
        fill="red",
    )

    title_font = load_font(56)
    author_font = load_font(36)

    spine_center_x = FRAME_WIDTH // 2
    title_center_y = FRAME_HEIGHT // 2 - 120
    author_center_y = FRAME_HEIGHT // 2 + 120

    draw_rotated_text(
        frame_rgba,
        TITLE_TEXT,
        title_font,
        fill=(255, 255, 255),
        center=(spine_center_x, title_center_y),
        angle=90,
    )
    draw_rotated_text(
        frame_rgba,
        AUTHOR_TEXT,
        author_font,
        fill=(0, 0, 0),
        center=(spine_center_x, author_center_y),
        angle=90,
    )

    output_path = BASE_DIR / "spine_1984.bmp"
    spine_image = frame_rgba.convert("RGB")
    spine_image.save(output_path)
    print(f"Wrote {output_path}")

    try:
        from waveshare_epd import epd7in5b_V2
    except ImportError as exc:
        raise SystemExit(
            "Waveshare library not found. Make sure ./lib is available on the Raspberry Pi."
        ) from exc

    epd = epd7in5b_V2.EPD()
    epd.init()
    epd.Clear()
    # Keep default data-interval settings; use border command if supported.
    epd.send_command(0x3C)
    epd.send_data(0x03)  # red border on panels that support VBD

    # Create 1-bit buffers for black and red channels.
    black_layer = Image.new("1", (FRAME_WIDTH, FRAME_HEIGHT), 1)
    red_layer = Image.new("1", (FRAME_WIDTH, FRAME_HEIGHT), 1)

    pixels = spine_image.load()
    for y in range(FRAME_HEIGHT):
        for x in range(FRAME_WIDTH):
            r, g, b = pixels[x, y]
            if r > 200 and g < 80 and b < 80:
                red_layer.putpixel((x, y), 0)
            elif r < 60 and g < 60 and b < 60:
                black_layer.putpixel((x, y), 0)

    epd.display(epd.getbuffer(black_layer), epd.getbuffer(red_layer))
    epd.sleep()


if __name__ == "__main__":
    main()
