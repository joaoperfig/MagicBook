#!/usr/bin/env python3
"""
Export Wikidata literary works (Q7725634) via WDQS to NDJSON
with tqdm progress bar and robust JSON/XML handling.
"""

from __future__ import annotations

import json
import random
import sys
import time
import xml.etree.ElementTree as ET

import requests
from tqdm import tqdm


WDQS_URL = "https://query.wikidata.org/sparql"
USER_AGENT = "magicbook-literary-export/1.0 (mailto:you@example.com)"

TOTAL_ESTIMATE = 1_358_801  # as of 31/01/2026

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


def request_wdqs(session, query, timeout=120, max_retries=6, base_backoff=2.0):
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/sparql-results+json, application/sparql-results+xml;q=0.9",
    }

    for attempt in range(max_retries + 1):
        try:
            r = session.get(
                WDQS_URL,
                params={"query": query, "format": "json"},
                headers=headers,
                timeout=timeout,
            )
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(base_backoff * (2 ** attempt) + random.random())
                continue
            r.raise_for_status()
        except requests.RequestException:
            if attempt >= max_retries:
                raise
            time.sleep(base_backoff * (2 ** attempt) + random.random())

    raise RuntimeError("WDQS failed after retries")


def parse_bindings(resp: requests.Response):
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
                name = binding.attrib["name"]
                literal = binding.find("s:literal", ns)
                uri = binding.find("s:uri", ns)
                if literal is not None:
                    row[name] = {"value": literal.text}
                elif uri is not None:
                    row[name] = {"value": uri.text}
            rows.append(row)
        return rows

    raise ValueError(f"Unexpected Content-Type: {ctype}\n{resp.text[:300]}")


def export_ndjson(
    out_path: str,
    limit: int = 5000,
    lang: str = "en",
    sleep_s: float = 0.5,
):
    session = requests.Session()
    offset = 0

    pbar = tqdm(
        total=TOTAL_ESTIMATE,
        unit="rows",
        desc="Wikidata literary works",
        dynamic_ncols=True,
    )

    with open(out_path, "w", encoding="utf-8") as f:
        while True:
            query = QUERY_TEMPLATE.format(limit=limit, offset=offset, lang=lang)
            resp = request_wdqs(session, query)
            bindings = parse_bindings(resp)

            if not bindings:
                break

            for b in bindings:
                row = {k: v.get("value") for k, v in b.items()}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            pbar.update(len(bindings))
            offset += limit

            if sleep_s:
                time.sleep(sleep_s)

    pbar.close()


if __name__ == "__main__":
    export_ndjson("data/literary_works.ndjson")
