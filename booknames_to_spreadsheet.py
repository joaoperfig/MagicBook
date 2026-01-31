import argparse
import json
import sys
from pathlib import Path

LOCAL_DEPS = Path(__file__).resolve().parent / ".deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))


def load_booknames(input_path: Path) -> dict:
    with input_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(output_path: Path, rows: list[dict]) -> None:
    import csv

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["time", "title", "author", "score"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_xlsx(output_path: Path, rows: list[dict]) -> None:
    try:
        from openpyxl import Workbook
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency 'openpyxl'. Install it with: python -m pip install openpyxl"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Booknames"
    sheet.append(["time", "title", "author", "score"])
    for row in rows:
        sheet.append([row["time"], row["title"], row["author"], row["score"]])
    workbook.save(output_path)


def convert_file(
    input_path: Path,
    output_path: Path | None,
    output_format: str,
) -> Path:
    data = load_booknames(input_path)
    rows = []
    for time_key in sorted(data.keys()):
        entry = data.get(time_key, {})
        rows.append(
            {
                "time": time_key,
                "title": entry.get("title", ""),
                "author": entry.get("author", ""),
                "score": entry.get("score", ""),
            }
        )

    resolved_output = output_path or input_path.with_suffix(f".{output_format}")
    if output_format == "csv":
        write_csv(resolved_output, rows)
    else:
        write_xlsx(resolved_output, rows)
    return resolved_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert booknames.json into a spreadsheet.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Path(s) to booknames.json file(s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path (only valid when a single input is used).",
    )
    parser.add_argument(
        "--format",
        choices=("xlsx", "csv"),
        default="xlsx",
        help="Output file format (default: xlsx).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.output and len(args.inputs) != 1:
        parser.error("--output can only be used with a single input file.")

    for input_path in args.inputs:
        output_path = args.output if len(args.inputs) == 1 else None
        resolved_output = convert_file(input_path, output_path, args.format)
        print(f"Wrote {resolved_output}")


if __name__ == "__main__":
    main()
