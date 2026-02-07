#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
PIC_DIR = BASE_DIR / "pic"
LIB_DIR = BASE_DIR / "lib"
if LIB_DIR.exists():
    sys.path.append(str(LIB_DIR))

BORDER_COLOR = "red"
BLACK_BMP = PIC_DIR / "1984_test_b.bmp"
RED_BMP = PIC_DIR / "1984_test_r.bmp"
PREVIEW_BMP = PIC_DIR / "1984_test_preview.bmp"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Display spine BMP layers on the e-paper.")
    parser.add_argument(
        "--swap-colors",
        action="store_true",
        help="Swap the red and black layers before display.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not BLACK_BMP.exists() or not RED_BMP.exists():
        raise SystemExit(
            "Missing BMP layers. Run image_to_epd_bmps.py to generate *_b.bmp and *_r.bmp."
        )

    black_layer = Image.open(BLACK_BMP)
    red_layer = Image.open(RED_BMP)
    if args.swap_colors:
        black_layer, red_layer = red_layer, black_layer

    if PREVIEW_BMP.exists():
        print(f"Using preview {PREVIEW_BMP}")
    print(f"Using black layer {BLACK_BMP}")
    print(f"Using red layer {RED_BMP}")

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
    border_map = {
        "white": 0x01,
        "black": 0x02,
        "red": 0x03,
    }
    border_value = border_map.get(BORDER_COLOR.lower(), 0x03)
    epd.send_command(0x3C)
    epd.send_data(border_value)

    epd.display(epd.getbuffer(black_layer), epd.getbuffer(red_layer))
    epd.sleep()


if __name__ == "__main__":
    main()
