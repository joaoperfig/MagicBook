#!/usr/bin/env python3
"""
render_spine.py
Compile a spine_boxes_v1 JSON layout into a PNG.

Usage:
  python render_spine.py input.json output.png
"""

import json
import math
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent
FONTS_DIR = BASE_DIR / "fonts"


# -----------------------------
# Config: colors + font mapping
# -----------------------------

COLOR_MAP: Dict[str, Tuple[int, int, int]] = {
    "red": (255, 0, 0),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}

# You can customize these paths per machine.
# The renderer will fall back to Pillow's default font if none found.
FONT_CANDIDATES: Dict[str, List[str]] = {
    # Serif-ish
    "Baskerville": [
        str(FONTS_DIR / "Baskerville.ttf"),
        "/Library/Fonts/Baskerville.ttf",
        "/System/Library/Fonts/Supplemental/Baskerville.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "DejaVuSerif.ttf",
    ],
    "Garamond": [
        str(FONTS_DIR / "Garamond.ttf"),
        "/Library/Fonts/Adobe Garamond Pro.ttf",
        "/System/Library/Fonts/Supplemental/Garamond.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "DejaVuSerif.ttf",
    ],
    "Georgia": [
        str(FONTS_DIR / "Georgia.ttf"),
        "/Library/Fonts/Georgia.ttf",
        "/System/Library/Fonts/Supplemental/Georgia.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "DejaVuSerif.ttf",
    ],
    "Times": [
        str(FONTS_DIR / "Times.ttf"),
        "/Library/Fonts/Times New Roman.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "DejaVuSerif.ttf",
    ],
    "Didot": [
        str(FONTS_DIR / "Didot.ttf"),
        "/Library/Fonts/Didot.ttf",
        "/System/Library/Fonts/Supplemental/Didot.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "DejaVuSerif.ttf",
    ],
    # Sans-ish
    "Helvetica": [
        str(FONTS_DIR / "Helvetica.ttf"),
        "/Library/Fonts/Helvetica.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "DejaVuSans.ttf",
    ],
    "Arial": [
        str(FONTS_DIR / "Arial.ttf"),
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "DejaVuSans.ttf",
    ],
    "Futura": [
        str(FONTS_DIR / "Futura.ttf"),
        "/Library/Fonts/Futura.ttf",
        "/System/Library/Fonts/Supplemental/Futura.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "DejaVuSans.ttf",
    ],
    "GillSans": [
        str(FONTS_DIR / "GillSans.ttf"),
        "/Library/Fonts/GillSans.ttc",
        "/System/Library/Fonts/Supplemental/GillSans.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "DejaVuSans.ttf",
    ],
    # Mono-ish
    "Courier": [
        str(FONTS_DIR / "Courier.ttf"),
        "/Library/Fonts/Courier New.ttf",
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "DejaVuSansMono.ttf",
    ],
}


# -----------------------------
# Helpers
# -----------------------------

@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def w(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def h(self) -> int:
        return max(0, self.y2 - self.y1)

    def clamp(self, W: int, H: int) -> "Box":
        x1 = min(max(self.x1, 0), W)
        y1 = min(max(self.y1, 0), H)
        x2 = min(max(self.x2, 0), W)
        y2 = min(max(self.y2, 0), H)
        # ensure ordering
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        return Box(x1, y1, x2, y2)


def parse_color(name: str) -> Tuple[int, int, int]:
    if name not in COLOR_MAP:
        raise ValueError(f"Unsupported color: {name}")
    return COLOR_MAP[name]


def load_font(font_name: str, size: int) -> ImageFont.FreeTypeFont:
    candidates = FONT_CANDIDATES.get(font_name, [])
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    # last resort: default bitmap font (limited metrics)
    return ImageFont.load_default()


def text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    # Returns (w,h) for multiline text using the same per-line layout as rendering.
    lines = text.split("\n")
    if not lines:
        lines = [""]
    max_w = 0
    total_h = 0
    for line in lines:
        sample = line if line else " "
        lb = draw.textbbox((0, 0), sample, font=font)
        w = lb[2] - lb[0]
        h = lb[3] - lb[1]
        max_w = max(max_w, w)
        total_h += h
    return max(1, max_w), max(1, total_h)


def fit_font_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_name: str,
    box_w: int,
    box_h: int,
    min_size: int = 1,
    max_size: int = 2000,
) -> int:
    """
    Find the largest font size that fits within (box_w, box_h) for the given text.
    Uses binary search.
    """
    lo, hi = min_size, max_size
    best = min_size
    while lo <= hi:
        mid = (lo + hi) // 2
        font = load_font(font_name, mid)
        w, h = text_bbox(draw, text, font)
        if w <= box_w and h <= box_h:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def justify_line(draw: ImageDraw.ImageDraw, line: str, font: ImageFont.ImageFont, target_w: int) -> List[Tuple[str, int]]:
    """
    Returns list of (word, extra_px_after) for a justified line.
    Simple space expansion: distribute extra pixels across spaces.
    """
    words = line.split(" ")
    if len(words) <= 1:
        return [(line, 0)]

    # Measure base width with single spaces
    base = " ".join(words)
    base_w = draw.textlength(base, font=font)
    extra = max(0, int(round(target_w - base_w)))
    gaps = len(words) - 1
    if gaps <= 0 or extra <= 0:
        return [(base, 0)]

    per = extra // gaps
    rem = extra % gaps
    out = []
    for i, w in enumerate(words):
        if i < gaps:
            out.append((w, per + (1 if i < rem else 0)))
        else:
            out.append((w, 0))
    return out


def render_text_block(
    text: str,
    font_name: str,
    color: Tuple[int, int, int],
    box_w: int,
    box_h: int,
    align: str,
    fit_mode: str,
    allow_nonlinear_scale: bool,
) -> Image.Image:
    """
    Render multiline text into an RGBA image, sized exactly (box_w, box_h).
    - Auto-fits font size to box using 'contain' (max size that fits).
    - If fit_mode == 'cover', uses size that covers the box then crops center.
    - If allow_nonlinear_scale, renders at contain size and stretches to fill box.
    """
    if box_w <= 0 or box_h <= 0:
        return Image.new("RGBA", (max(1, box_w), max(1, box_h)), (0, 0, 0, 0))

    # Temporary draw context for measuring
    tmp = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)

    # Choose a base font size that "contains"
    contain_size = fit_font_size(d, text, font_name, box_w, box_h, min_size=1, max_size=3000)

    if fit_mode not in ("contain", "cover"):
        fit_mode = "contain"

    if fit_mode == "contain":
        font_size = contain_size
        font = load_font(font_name, font_size)
        text_w, text_h = text_bbox(d, text, font)
        render_w, render_h = max(1, text_w), max(1, text_h)

        # render tightly then place into box with alignment
        tight = Image.new("RGBA", (render_w, render_h), (0, 0, 0, 0))
        td = ImageDraw.Draw(tight)

        # Draw line-by-line to support justify cleanly
        lines = text.split("\n")
        y = 0
        for i, line in enumerate(lines):
            sample = line if line else " "
            lb = td.textbbox((0, 0), sample, font=font)
            line_h = (lb[3] - lb[1])
            y_line = y - lb[1]
            if align == "justify" and i != len(lines) - 1 and " " in line:
                # justify line (not last line)
                parts = justify_line(td, line, font, render_w)
                x = 0
                for word, extra_px in parts:
                    td.text((x, y_line), word, font=font, fill=color)
                    x += int(td.textlength(word, font=font)) + int(td.textlength(" ", font=font)) + extra_px
            else:
                # normal alignment within tight box width
                if align == "center":
                    lw = td.textlength(line, font=font)
                    x = (render_w - lw) / 2
                elif align == "right":
                    lw = td.textlength(line, font=font)
                    x = render_w - lw
                else:
                    x = 0
                td.text((x, y_line), line, font=font, fill=color)
            y += line_h

        if allow_nonlinear_scale:
            # Stretch tight render to fill the box exactly (can distort)
            stretched = tight.resize((box_w, box_h), resample=Image.Resampling.BICUBIC)
            return stretched

        # Place tight render into final box with centering
        out = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
        ox = (box_w - render_w) // 2
        oy = (box_h - render_h) // 2
        out.alpha_composite(tight, (ox, oy))
        return out

    # cover mode: pick size that covers the box, render, then crop
    # We approximate by scaling contain render up so that it covers box
    font = load_font(font_name, max(1, contain_size))
    text_w, text_h = text_bbox(d, text, font)
    text_w = max(1, text_w)
    text_h = max(1, text_h)

    scale = max(box_w / text_w, box_h / text_h)
    # Convert scale to a larger font size
    font_size = max(1, int(round(contain_size * scale)))
    font = load_font(font_name, font_size)

    # Render again tightly at this font size
    tmp2 = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    d2 = ImageDraw.Draw(tmp2)
    text_w2, text_h2 = text_bbox(d2, text, font)
    render_w, render_h = max(1, text_w2), max(1, text_h2)

    tight = Image.new("RGBA", (render_w, render_h), (0, 0, 0, 0))
    td = ImageDraw.Draw(tight)

    lines = text.split("\n")
    y = 0
    for i, line in enumerate(lines):
        sample = line if line else " "
        lb = td.textbbox((0, 0), sample, font=font)
        line_h = (lb[3] - lb[1])
        y_line = y - lb[1]
        if align == "justify" and i != len(lines) - 1 and " " in line:
            parts = justify_line(td, line, font, render_w)
            x = 0
            for word, extra_px in parts:
                td.text((x, y_line), word, font=font, fill=color)
                x += int(td.textlength(word, font=font)) + int(td.textlength(" ", font=font)) + extra_px
        else:
            if align == "center":
                lw = td.textlength(line, font=font)
                x = (render_w - lw) / 2
            elif align == "right":
                lw = td.textlength(line, font=font)
                x = render_w - lw
            else:
                x = 0
            td.text((x, y_line), line, font=font, fill=color)
        y += line_h

    if allow_nonlinear_scale:
        # In cover mode, nonlinear scaling just fills the box anyway
        return tight.resize((box_w, box_h), resample=Image.Resampling.BICUBIC)

    # Crop center to box size
    cx = (render_w - box_w) // 2
    cy = (render_h - box_h) // 2
    cx = max(0, cx)
    cy = max(0, cy)
    cropped = tight.crop((cx, cy, cx + box_w, cy + box_h))
    return cropped


def render_layout(layout: dict) -> Image.Image:
    if layout.get("schema_version") != "spine_boxes_v1":
        raise ValueError("Unsupported schema_version (expected spine_boxes_v1)")

    spine = layout.get("spine", {})
    W = int(spine.get("width", 240))
    H = int(spine.get("height", 800))
    bg_name = spine.get("background", "red")
    bg = parse_color(bg_name)

    img = Image.new("RGBA", (W, H), (*bg, 255))
    draw = ImageDraw.Draw(img)

    elements = layout.get("elements", [])
    if not isinstance(elements, list):
        raise ValueError("elements must be a list")

    for el in elements:
        et = el.get("type")
        color = parse_color(el.get("color", "black"))

        box = Box(
            int(round(el.get("x1", 0))),
            int(round(el.get("y1", 0))),
            int(round(el.get("x2", 0))),
            int(round(el.get("y2", 0))),
        ).clamp(W, H)

        if et == "rect":
            if box.w > 0 and box.h > 0:
                draw.rectangle([box.x1, box.y1, box.x2, box.y2], fill=(*color, 255))

        elif et == "circle":
            if box.w > 0 and box.h > 0:
                draw.ellipse([box.x1, box.y1, box.x2, box.y2], fill=(*color, 255))

        elif et == "text":
            text = el.get("text", "")
            if not isinstance(text, str):
                raise ValueError("text element requires string 'text'")

            align = el.get("align", "center")
            if align not in ("left", "center", "right", "justify"):
                align = "center"

            direction = el.get("direction", "horizontal")
            if direction not in ("horizontal", "vertical_rotated", "vertical_stacked"):
                direction = "horizontal"

            font_name = el.get("font", "Helvetica")
            if font_name not in FONT_CANDIDATES:
                font_name = "Helvetica"

            fit_mode = el.get("fit", "contain")
            if fit_mode not in ("contain", "cover"):
                fit_mode = "contain"

            allow_nonlinear_scale = bool(el.get("allow_nonlinear_scale", False))

            # Prepare text content for stacked direction:
            # If vertical_stacked, we assume text already includes \n for stacking.
            # (So the "compiler" is deterministic and simple.)
            text_to_render = text

            # Render according to direction
            if direction == "horizontal":
                block = render_text_block(
                    text=text_to_render,
                    font_name=font_name,
                    color=color,
                    box_w=box.w,
                    box_h=box.h,
                    align=align,
                    fit_mode=fit_mode,
                    allow_nonlinear_scale=allow_nonlinear_scale,
                )
                img.alpha_composite(block, (box.x1, box.y1))

            elif direction == "vertical_stacked":
                # No rotation; text expected to have explicit \n stacking.
                block = render_text_block(
                    text=text_to_render,
                    font_name=font_name,
                    color=color,
                    box_w=box.w,
                    box_h=box.h,
                    align=align,
                    fit_mode=fit_mode,
                    allow_nonlinear_scale=allow_nonlinear_scale,
                )
                img.alpha_composite(block, (box.x1, box.y1))

            else:  # vertical_rotated
                # For rotated: fit text into a swapped box (h,w), render, rotate, then place.
                # This keeps "fit to box" intuitive.
                block = render_text_block(
                    text=text_to_render,
                    font_name=font_name,
                    color=color,
                    box_w=box.h,   # swapped
                    box_h=box.w,   # swapped
                    align=align,
                    fit_mode=fit_mode,
                    allow_nonlinear_scale=allow_nonlinear_scale,
                )
                # Rotate clockwise 90:
                rot = block.rotate(-90, expand=True, resample=Image.Resampling.BICUBIC)

                # rot should now be roughly (box.w, box.h). If not, center it.
                out = Image.new("RGBA", (box.w, box.h), (0, 0, 0, 0))
                ox = (box.w - rot.size[0]) // 2
                oy = (box.h - rot.size[1]) // 2
                out.alpha_composite(rot, (ox, oy))
                img.alpha_composite(out, (box.x1, box.y1))

        else:
            raise ValueError(f"Unsupported element type: {et}")

    return img


def main():
    if len(sys.argv) != 3:
        print("Usage: python render_spine.py input.json output.png", file=sys.stderr)
        sys.exit(2)

    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path, "r", encoding="utf-8") as f:
        layout = json.load(f)

    img = render_layout(layout)
    # Save as PNG
    img.convert("RGBA").save(out_path, "PNG")


if __name__ == "__main__":
    main()
