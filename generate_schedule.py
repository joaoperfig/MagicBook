#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate a human-readable HTML schedule from the book schedule CSV.
"""

import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_CSV = BASE_DIR / "data" / "booknames_curated_final.csv"
OUTPUT_HTML = BASE_DIR / "schedule.html"
GITHUB_BASE_URL = "https://raw.githubusercontent.com/joaoperfig/MagicBook/refs/heads/main/pic/final_bmp/"


def format_time_range(start: str, end: str) -> str:
    """Format time range in a human-readable way."""
    return f"from {start} to {end}"


def generate_html_schedule():
    """Generate HTML schedule from CSV data."""
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"CSV not found: {DATA_CSV}")
    
    rows = []
    with DATA_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            time_label = row.get("time", "").strip()
            start = row.get("start", "").strip()
            end = row.get("end", "").strip()
            title = row.get("title", "").strip()
            author = row.get("author", "").strip()
            preview_image = row.get("image_preview", "").strip()
            
            if not all([time_label, start, end, title, author]):
                continue
            
            # Build GitHub URL for preview image
            image_url = f"{GITHUB_BASE_URL}{preview_image}" if preview_image else ""
            
            rows.append({
                "time": time_label,
                "start": start,
                "end": end,
                "title": title,
                "author": author,
                "image_url": image_url,
            })
    
    # Generate HTML
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Magic Book Schedule</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
        }
        .schedule-table {
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }
        .schedule-table thead {
            background-color: #34495e;
            color: white;
        }
        .schedule-table th {
            padding: 15px;
            text-align: left;
            font-weight: 600;
            font-size: 0.95em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .schedule-table td {
            padding: 12px 15px;
            border-bottom: 1px solid #e0e0e0;
        }
        .schedule-table tbody tr:hover {
            background-color: #f8f9fa;
        }
        .schedule-table tbody tr:last-child td {
            border-bottom: none;
        }
        .time-col {
            font-weight: 600;
            color: #2c3e50;
            width: 80px;
        }
        .title-col {
            font-weight: 500;
            color: #2c3e50;
        }
        .author-col {
            color: #7f8c8d;
            font-style: italic;
        }
        .time-range-col {
            color: #555;
            font-size: 0.95em;
        }
        .view-link {
            color: #3498db;
            text-decoration: none;
            font-weight: 500;
        }
        .view-link:hover {
            text-decoration: underline;
            color: #2980b9;
        }
        @media print {
            body {
                background-color: white;
                padding: 0;
            }
            .schedule-table {
                box-shadow: none;
            }
            .schedule-table thead {
                background-color: #34495e !important;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }
        }
    </style>
</head>
<body>
    <h1>Magic Book Schedule</h1>
    <table class="schedule-table">
        <thead>
            <tr>
                <th>Time</th>
                <th>Book Title</th>
                <th>Author</th>
                <th>Time Range</th>
                <th>Preview</th>
            </tr>
        </thead>
        <tbody>
"""
    
    for row in rows:
        time_range = format_time_range(row["start"], row["end"])
        view_link = f'<a href="{row["image_url"]}" class="view-link" target="_blank">view</a>' if row["image_url"] else ""
        
        html_content += f"""            <tr>
                <td class="time-col">{row["time"]}</td>
                <td class="title-col">{row["title"]}</td>
                <td class="author-col">{row["author"]}</td>
                <td class="time-range-col">{time_range}</td>
                <td>{view_link}</td>
            </tr>
"""
    
    html_content += """        </tbody>
    </table>
</body>
</html>
"""
    
    OUTPUT_HTML.write_text(html_content, encoding="utf-8")
    print(f"Schedule generated: {OUTPUT_HTML}")
    print(f"Open this file in your browser and print to PDF (Ctrl+P -> Save as PDF)")


if __name__ == "__main__":
    generate_html_schedule()
