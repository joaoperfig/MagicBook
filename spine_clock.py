#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
DATA_CSV = BASE_DIR / "data" / "booknames_curated_final.csv"
BMP_DIR = BASE_DIR / "pic" / "final_bmp"
LIB_DIR = BASE_DIR / "lib"
if LIB_DIR.exists():
    sys.path.append(str(LIB_DIR))

BORDER_COLOR = "red"


@dataclass(frozen=True)
class TimeSlot:
    time_label: str
    start_sec: int
    end_sec: int
    black_path: Path
    red_path: Path


def parse_hhmmss(value: str) -> int:
    parts = value.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid time format: {value!r}")
    hours, minutes, seconds = (int(part) for part in parts)
    return hours * 3600 + minutes * 60 + seconds


def is_in_range(start_sec: int, end_sec: int, now_sec: int) -> bool:
    if start_sec <= end_sec:
        return start_sec <= now_sec <= end_sec
    return now_sec >= start_sec or now_sec <= end_sec


def seconds_until(target_sec: int, now_sec: int) -> int:
    delta = (target_sec - now_sec) % 86400
    return delta if delta > 0 else 86400


def load_schedule(csv_path: Path, bmp_dir: Path) -> list[TimeSlot]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not bmp_dir.exists():
        raise FileNotFoundError(f"BMP directory not found: {bmp_dir}")

    slots: list[TimeSlot] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            start = (row.get("start") or "").strip()
            end = (row.get("end") or "").strip()
            time_label = (row.get("time") or "").strip()
            black_name = (row.get("image_black") or "").strip()
            red_name = (row.get("image_red") or "").strip()
            if not start or not end or not black_name or not red_name:
                continue
            black_path = bmp_dir / black_name
            red_path = bmp_dir / red_name
            if not black_path.exists() or not red_path.exists():
                raise FileNotFoundError(
                    f"Missing BMP layers for {time_label}: {black_path}, {red_path}"
                )
            slots.append(
                TimeSlot(
                    time_label=time_label,
                    start_sec=parse_hhmmss(start),
                    end_sec=parse_hhmmss(end),
                    black_path=black_path,
                    red_path=red_path,
                )
            )

    if not slots:
        raise RuntimeError("No schedule entries found in the CSV.")
    return slots


def find_current_slot(slots: list[TimeSlot], now_sec: int) -> TimeSlot | None:
    for slot in slots:
        if is_in_range(slot.start_sec, slot.end_sec, now_sec):
            return slot
    return None


def set_border(epd) -> None:
    border_map = {
        "white": 0x01,
        "black": 0x02,
        "red": 0x03,
    }
    border_value = border_map.get(BORDER_COLOR.lower(), 0x03)
    epd.send_command(0x3C)
    epd.send_data(border_value)


def show_slot(epd, slot: TimeSlot, fast: bool) -> None:
    if fast:
        epd.init_Fast()
    else:
        epd.init()
        epd.Clear()
    set_border(epd)
    black_layer = Image.open(slot.black_path)
    red_layer = Image.open(slot.red_path)
    epd.display(epd.getbuffer(black_layer), epd.getbuffer(red_layer))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    try:
        from waveshare_epd import epd7in5b_V2
    except ImportError as exc:
        raise SystemExit(
            "Waveshare library not found. Make sure ./lib is available on the Raspberry Pi."
        ) from exc

    epd = epd7in5b_V2.EPD()
    slots = load_schedule(DATA_CSV, BMP_DIR)

    current_label: str | None = None
    since_full = 9  # force a full refresh on first display

    try:
        while True:
            now = datetime.now()
            now_sec = now.hour * 3600 + now.minute * 60 + now.second
            slot = find_current_slot(slots, now_sec)
            if slot is None:
                logging.warning("No matching slot for current time; retrying soon.")
                time.sleep(30)
                continue

            if slot.time_label != current_label:
                use_full = since_full >= 9
                show_slot(epd, slot, fast=not use_full)
                if use_full:
                    since_full = 0
                    logging.info("Displayed %s with full refresh", slot.time_label)
                else:
                    since_full += 1
                    logging.info("Displayed %s with fast refresh", slot.time_label)
                current_label = slot.time_label

            next_change = (slot.end_sec + 1) % 86400
            sleep_for = seconds_until(next_change, now_sec)
            time.sleep(sleep_for)
    except KeyboardInterrupt:
        logging.info("Exiting.")
    finally:
        epd.sleep()
        epd7in5b_V2.epdconfig.module_exit(cleanup=True)


if __name__ == "__main__":
    main()
