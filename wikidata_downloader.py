#!/usr/bin/env python3
"""
WDQS exporter (literary works) -> NDJSON with:
- tqdm progress bar (estimate total)
- robust JSON/XML parsing (handles WDQS returning XML sometimes)
- checkpoint/resume (offset + rows written)
- fallback resume from existing NDJSON by counting lines if checkpoint missing
- better retry handling (Retry-After, higher retry caps)
- adaptive LIMIT reduction after repeated failures

Output:
  data/literary_works.ndjson
Checkpoint:
  data/literary_works.ndjson.checkpoint.json
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import requests
from tqdm import tqdm


WDQS_URL = "https://query.wikidata.org/sparql"
USER_AGENT = "magicbook-literary-export/1.2 (mailto:you@example.com)"

# Your measured estimate (COUNT query you ran)
TOTAL_ESTIMATE = 1_358_801

QUERY_TEMPLATE = """
SELECT ?work ?workLabel ?author ?authorLabel ?pubDate ?workType ?workTypeLabel WHERE {{
  ?work wdt:P31/wdt:P279* wd:Q7725634 .
  OPTIONAL {{ ?work wdt:P50 ?author . }}
  OPTIONAL {{ ?work wdt:P577 ?pubDate . }}
  OPTIONAL {{ ?work wdt:P31 ?workType . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lang},en". }}
}}
LIMIT {limit}
OFFSET {offset}
"""


@dataclass
class Checkpoint:
    offset: int = 0          # OFFSET for next page
    rows_written: int = 0    # total NDJSON lines written so far


def load_checkpoint(path: str) -> Checkpoint:
    if not os.path.exists(path):
        return Checkpoint()
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return Checkpoint(offset=int(d.get("offset", 0)), rows_written=int(d.get("rows_written", 0)))
    except Exception as e:
        print(f"[warn] failed to read checkpoint {path}: {e}", file=sys.stderr)
        return Checkpoint()


def save_checkpoint(path: str, cp: Checkpoint) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"offset": cp.offset, "rows_written": cp.rows_written}, f)
    os.replace(tmp, path)


def count_lines_fast(path: str) -> int:
    """
    Count lines without loading the file into memory.
    """
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def infer_checkpoint_from_file(out_path: str, limit: int) -> Checkpoint:
    """
    If checkpoint is missing but NDJSON exists, infer progress from line count.

    NOTE: This is conservative:
    - offset is rounded DOWN to last full page boundary (n//limit)*limit
      to avoid skipping data.
    - rows_written is the real line count (for tqdm initial position).
    """
    if not os.path.exists(out_path):
        return Checkpoint()

    n = count_lines_fast(out_path)
    inferred_offset = (n // limit) * limit
    return Checkpoint(offset=inferred_offset, rows_written=n)


def request_wdqs(
    session: requests.Session,
    query: str,
    timeout: int = 120,
    max_retries: int = 20,
    base_backoff: float = 2.0,
) -> requests.Response:
    headers = {
        "User-Agent": USER_AGENT,
        # Prefer JSON, but allow XML (WDQS sometimes returns it anyway)
        "Accept": "application/sparql-results+json, application/sparql-results+xml;q=0.9",
    }

    last_err: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            r = session.get(
                WDQS_URL,
                params={"query": query, "format": "json"},  # strongly request JSON
                headers=headers,
                timeout=timeout,
            )

            if r.status_code == 200:
                return r

            if r.status_code in (429, 500, 502, 503, 504):
                retry_after = r.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    sleep_s = float(retry_after) + random.random()
                else:
                    # Exponential backoff with cap + jitter
                    sleep_s = base_backoff * (2 ** min(attempt, 8)) + random.random() * 2

                print(f"[warn] WDQS {r.status_code}, retrying in {sleep_s:.1f}s", file=sys.stderr)
                time.sleep(sleep_s)
                continue

            r.raise_for_status()

        except requests.RequestException as e:
            last_err = e
            if attempt >= max_retries:
                break
            sleep_s = base_backoff * (2 ** min(attempt, 8)) + random.random() * 2
            print(f"[warn] network/error: {e}, retrying in {sleep_s:.1f}s", file=sys.stderr)
            time.sleep(sleep_s)

    raise RuntimeError(f"WDQS failed after retries (last error: {last_err})")


def parse_bindings(resp: requests.Response):
    """
    Return a list[dict] SPARQL bindings, regardless of JSON or XML response.
    """
    ctype = (resp.headers.get("Content-Type") or "").lower()

    if "json" in ctype:
        return resp.json()["results"]["bindings"]

    if "xml" in ctype:
        ns = {"s": "http://www.w3.org/2005/sparql-results#"}
        root = ET.fromstring(resp.text)
        rows = []
        for result in root.findall(".//s:result", ns):
            row = {}
            for binding in result.findall("s:binding", ns):
                name = binding.attrib.get("name")
                if not name:
                    continue
                literal = binding.find("s:literal", ns)
                uri = binding.find("s:uri", ns)
                if literal is not None and literal.text is not None:
                    row[name] = {"value": literal.text}
                elif uri is not None and uri.text is not None:
                    row[name] = {"value": uri.text}
            rows.append(row)
        return rows

    # Rare: HTML error with 200, or unexpected content type
    snippet = resp.text[:300].replace("\n", " ")
    raise ValueError(f"Unexpected Content-Type: {ctype}. Body starts: {snippet!r}")


def export_ndjson(
    out_path: str,
    checkpoint_path: str | None = None,
    limit: int = 5000,
    lang: str = "en",
    polite_sleep_s: float = 0.8,
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if checkpoint_path is None:
        checkpoint_path = out_path + ".checkpoint.json"

    cp = load_checkpoint(checkpoint_path)

    # Fallback: infer progress from existing file if checkpoint is missing/empty.
    if cp.offset == 0 and cp.rows_written == 0 and os.path.exists(out_path):
        inferred = infer_checkpoint_from_file(out_path, limit)
        cp = inferred
        save_checkpoint(checkpoint_path, cp)
        print(
            f"[info] inferred progress from file: rows_written={cp.rows_written}, offset={cp.offset} (limit={limit})",
            file=sys.stderr,
        )

    session = requests.Session()

    # Append mode so resumes don't overwrite.
    with open(out_path, "a", encoding="utf-8") as f:
        pbar = tqdm(
            total=TOTAL_ESTIMATE,
            initial=min(cp.rows_written, TOTAL_ESTIMATE),
            unit="rows",
            desc="Wikidata literary works",
            dynamic_ncols=True,
        )

        failures_in_a_row = 0
        current_limit = limit  # allow adaptive reduction without breaking "limit" param

        while True:
            query = QUERY_TEMPLATE.format(limit=current_limit, offset=cp.offset, lang=lang)

            try:
                resp = request_wdqs(session, query)
                bindings = parse_bindings(resp)
                failures_in_a_row = 0
            except Exception as e:
                failures_in_a_row += 1
                print(f"[warn] page failed at offset={cp.offset}: {e}", file=sys.stderr)

                # If WDQS is struggling, reduce page size.
                if failures_in_a_row >= 3 and current_limit > 1000:
                    current_limit = max(1000, current_limit // 2)
                    print(f"[warn] reducing LIMIT to {current_limit}", file=sys.stderr)

                # Save checkpoint before exiting so rerun can resume.
                save_checkpoint(checkpoint_path, cp)
                pbar.close()
                raise

            if not bindings:
                save_checkpoint(checkpoint_path, cp)
                pbar.close()
                print("[info] no more rows returned; done.", file=sys.stderr)
                return

            # Write rows
            for b in bindings:
                row = {k: v.get("value") for k, v in b.items()}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            written = len(bindings)

            # Update checkpoint
            cp.rows_written += written
            cp.offset += current_limit  # move to next page using the *current* limit
            save_checkpoint(checkpoint_path, cp)

            # Update tqdm
            pbar.update(written)

            # Be polite to WDQS
            if polite_sleep_s:
                time.sleep(polite_sleep_s)


if __name__ == "__main__":
    export_ndjson("data/literary_works.ndjson")
