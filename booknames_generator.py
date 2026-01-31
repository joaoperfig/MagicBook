import json
import os
import random
import time
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROMPT_DIR = BASE_DIR / "prompts"
TIME_CANDIDATES_PATH = DATA_DIR / "time_candidates.json"
SELECTION_CACHE_PATH = DATA_DIR / "time_selection.json"
OUTPUT_PATH = DATA_DIR / "booknames.json"

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
PROMPT_CACHE: dict[str, str] = {}
FAKE_AUTHOR_POOL = [
    "Agatha Christie",
    "Stephen King",
    "George Orwell",
    "Jane Austen",
    "Charles Dickens",
    "Leo Tolstoy",
    "Fyodor Dostoevsky",
    "J.R.R. Tolkien",
    "Gabriel Garcia Marquez",
    "Ernest Hemingway",
    "Virginia Woolf",
    "Mark Twain",
    "Oscar Wilde",
    "Arthur Conan Doyle",
    "J.K. Rowling",
    "Harper Lee",
    "Toni Morrison",
    "Mary Shelley",
    "H.G. Wells",
    "Jules Verne",
    "Aldous Huxley",
    "Ray Bradbury",
    "Kurt Vonnegut",
    "Isaac Asimov",
    "Ursula K. Le Guin",
    "Neil Gaiman",
    "Margaret Atwood",
    "C.S. Lewis",
    "Herman Melville",
    "Emily Bronte",
]


ONES = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}
TENS = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
}


def num_to_words(num: int) -> str:
    if num < 20:
        return ONES[num]
    tens = (num // 10) * 10
    remainder = num % 10
    if remainder == 0:
        return TENS[tens]
    return f"{TENS[tens]} {ONES[remainder]}"


def minute_word_variants(minute: int) -> list[str]:
    base = num_to_words(minute)
    variants = {base}
    if minute < 10:
        variants.add(f"oh {base}")
        variants.add(f"o {base}")
        variants.add(f"zero {base}")
    return sorted(variants)


def hour_to_words(hour24: int) -> tuple[str, str]:
    hour12 = hour24 % 12
    if hour12 == 0:
        hour12 = 12
    return num_to_words(hour24), num_to_words(hour12)


def generate_time_queries(hour24: int, minute: int) -> list[str]:
    hour12 = hour24 % 12 or 12
    hh = f"{hour24:02d}"
    mm = f"{minute:02d}"

    queries = set()

    # Numeric time forms.
    queries.add(f"{hh}:{mm}")
    queries.add(f"{hour24}:{mm}")
    queries.add(f"{hour12}:{mm}")
    queries.add(f"{hh}{mm}")
    if hour24 != 0:
        queries.add(f"{hour24}{mm}")
    queries.add(f"{hour12}{mm}")

    hour24_words, hour12_words = hour_to_words(hour24)
    minute_words = minute_word_variants(minute)

    # Worded time forms.
    if minute == 0:
        queries.add(f"{hour12_words} o'clock")
        queries.add(f"{hour12_words} o clock")
        if hour24 != 0:
            queries.add(f"{hour24_words} hundred")
        if hour12 != hour24:
            queries.add(f"{hour12_words} hundred")
    else:
        for mw in minute_words:
            queries.add(f"{hour24_words} {mw}")
            queries.add(f"{hour12_words} {mw}")
            queries.add(f"{hour24_words} hundred and {mw}")
            queries.add(f"{hour12_words} hundred and {mw}")

    # Relative phrases around the hour.
    if minute in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 30}:
        past = num_to_words(minute)
        for hour_words in (hour12_words, hour24_words):
            queries.add(f"{past} past {hour_words}")
            queries.add(f"{past} after {hour_words}")
            queries.add(f"{past} minutes past {hour_words}")
            queries.add(f"{past} minutes after {hour_words}")
            queries.add(f"{past} minute past {hour_words}")
            queries.add(f"{past} minute after {hour_words}")
    if 50 <= minute <= 59:
        to_next = num_to_words(60 - minute)
        next_hour12 = (hour12 % 12) + 1
        next_hour_words = num_to_words(next_hour12)
        queries.add(f"{to_next} to {next_hour_words}")
        queries.add(f"{to_next} before {next_hour_words}")
        queries.add(f"{to_next} minutes before {next_hour_words}")
        queries.add(f"{to_next} minutes to {next_hour_words}")
        queries.add(f"{to_next} minute before {next_hour_words}")
        queries.add(f"{to_next} minute to {next_hour_words}")
        if hour24 == 23:
            queries.add(f"{to_next} minutes before midnight")
            queries.add(f"{to_next} minutes to midnight")
            queries.add(f"{to_next} minute before midnight")
            queries.add(f"{to_next} minute to midnight")
        if hour24 == 11:
            queries.add(f"{to_next} minutes before noon")
            queries.add(f"{to_next} minutes to noon")

    # Classic phrasing.
    if minute == 15:
        queries.add(f"quarter past {hour12_words}")
    if minute == 30:
        queries.add(f"half past {hour12_words}")
    if minute == 45:
        next_hour12 = (hour12 % 12) + 1
        queries.add(f"quarter to {num_to_words(next_hour12)}")

    # Midday / midnight specials.
    if hour24 == 0 and minute == 0:
        queries.update({"midnight", "twelve midnight"})
    if hour24 == 12 and minute == 0:
        queries.update({"noon", "midday", "mid-day", "twelve noon"})

    return sorted(q for q in queries if q.strip())


def build_primary_queries(hour24: int, minute: int) -> list[str]:
    hour12 = hour24 % 12 or 12
    mm = f"{minute:02d}"
    hour24_words, hour12_words = hour_to_words(hour24)
    minute_words = minute_word_variants(minute)

    primary = []
    primary.append(f"{hour24:02d}:{mm}")
    primary.append(f"{hour12}:{mm}")
    if hour24 != 0:
        primary.append(f"{hour24}{mm}")
    primary.append(f"{hour12}{mm}")

    if minute == 0:
        primary.append(f"{hour12_words} o'clock")
        if hour24 != 0:
            primary.append(f"{hour24_words} hundred")
        if hour24 == 0:
            primary.append("midnight")
        if hour24 == 12:
            primary.append("noon")
            primary.append("midday")
    else:
        primary.append(f"{hour12_words} {minute_words[0]}")
        primary.append(f"{hour24_words} {minute_words[0]}")
        if hour24 == 0 and minute in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 30}:
            past = num_to_words(minute)
            primary.append(f"{past} past midnight")
            primary.append(f"{past} after midnight")

    return [q for q in primary if q]


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)


def load_prompt(name: str) -> str:
    if name in PROMPT_CACHE:
        return PROMPT_CACHE[name]
    path = PROMPT_DIR / name
    with path.open("r", encoding="utf-8") as handle:
        content = handle.read()
    PROMPT_CACHE[name] = content
    return content


def fetch_json_with_retry(url: str, request_sleep: float, verbose: bool) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if request_sleep:
            time.sleep(request_sleep)
        return payload
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        if verbose:
            print(f"[search] request error, skipping: {exc}", flush=True)
        if request_sleep:
            time.sleep(request_sleep)
        return {}


def google_books_search(
    query: str,
    max_results: int = 5,
    request_sleep: float = 0.1,
    verbose: bool = False,
) -> list[dict]:
    api_key = os.getenv("GOOGLE_BOOKS_API_KEY")
    if api_key:
        params = {
            "q": query,
            "maxResults": str(max_results),
            "printType": "books",
            "key": api_key,
        }
        url = "https://www.googleapis.com/books/v1/volumes?" + urllib.parse.urlencode(
            params
        )
        payload = fetch_json_with_retry(url, request_sleep, verbose)
        results = []
        for item in payload.get("items", []):
            info = item.get("volumeInfo", {})
            title = info.get("title")
            authors = info.get("authors") or []
            if not title or not authors:
                continue
            results.append(
                {
                    "title": title.strip(),
                    "author": ", ".join(a.strip() for a in authors if a.strip()),
                }
            )
        return results

    try:
        from google_books_api_wrapper.api import GoogleBooksAPI
    except ImportError:  # pragma: no cover - optional dependency
        GoogleBooksAPI = None

    if GoogleBooksAPI is not None:
        client = GoogleBooksAPI()
        result_set = client.search_book(search_term=query)
        books = result_set.get_all_results()
        results = []
        for book in books[:max_results]:
            if not book.title or not book.authors:
                continue
            results.append(
                {
                    "title": book.title.strip(),
                    "author": ", ".join(a.strip() for a in book.authors if a.strip()),
                }
            )
        return results

    # Final fallback to direct API without a key.
    params = {"q": query, "maxResults": str(max_results), "printType": "books"}
    url = "https://www.googleapis.com/books/v1/volumes?" + urllib.parse.urlencode(params)
    payload = fetch_json_with_retry(url, request_sleep, verbose)

    results = []
    for item in payload.get("items", []):
        info = item.get("volumeInfo", {})
        title = info.get("title")
        authors = info.get("authors") or []
        if not title or not authors:
            continue
        results.append(
            {
                "title": title.strip(),
                "author": ", ".join(a.strip() for a in authors if a.strip()),
            }
        )
    return results


def open_library_search(
    query: str,
    max_results: int = 5,
    request_sleep: float = 0.1,
    verbose: bool = False,
) -> list[dict]:
    params = {"title": query, "limit": str(max_results)}
    url = "https://openlibrary.org/search.json?" + urllib.parse.urlencode(params)
    payload = fetch_json_with_retry(url, request_sleep, verbose)
    results = []
    for doc in payload.get("docs", []):
        title = doc.get("title")
        authors = doc.get("author_name") or []
        if not title or not authors:
            continue
        results.append(
            {
                "title": title.strip(),
                "author": ", ".join(a.strip() for a in authors if a.strip()),
            }
        )
    return results


def build_source_queries(hour24: int, minute: int, queries: list[str]) -> dict:
    primary = build_primary_queries(hour24, minute)
    google_terms = [f'intitle:"{term}"' for term in primary]
    if not google_terms and queries:
        google_terms = [f'"{queries[0]}"']

    openlibrary_terms = primary if primary else queries[:4]

    return {
        "google": google_terms,
        "openlibrary": openlibrary_terms,
    }


def collect_candidates_for_time(
    hour24: int,
    minute: int,
    max_results_total: int,
    max_results_per_query: int,
    request_sleep: float,
    verbose: bool,
    dedupe_authors: bool = False,
) -> tuple[list[str], list[dict], dict, dict, dict]:
    queries = generate_time_queries(hour24, minute)
    source_queries = build_source_queries(hour24, minute, queries)
    all_pairs = {}
    source_counts = {"google": 0, "openlibrary": 0}
    source_results = {"google": [], "openlibrary": []}

    per_query_limit = max(1, max_results_per_query)

    google_author_seen = set()
    for google_query in source_queries["google"]:
        for item in google_books_search(
            google_query,
            max_results=per_query_limit,
            request_sleep=request_sleep,
            verbose=verbose,
        ):
            if dedupe_authors:
                author_key = normalize_author_key(item.get("author"))
                if author_key in google_author_seen:
                    continue
                google_author_seen.add(author_key)
            key = f"{item['title']}|{item['author']}"
            if key not in all_pairs:
                all_pairs[key] = item
            source_counts["google"] += 1
            source_results["google"].append(item)
            if len(all_pairs) >= max_results_total:
                return queries, list(all_pairs.values()), source_counts, source_results, source_queries

    openlibrary_author_seen = set()
    for openlibrary_query in source_queries["openlibrary"]:
        for item in open_library_search(
            openlibrary_query,
            max_results=per_query_limit,
            request_sleep=request_sleep,
            verbose=verbose,
        ):
            if dedupe_authors:
                author_key = normalize_author_key(item.get("author"))
                if author_key in openlibrary_author_seen:
                    continue
                openlibrary_author_seen.add(author_key)
            key = f"{item['title']}|{item['author']}"
            if key not in all_pairs:
                all_pairs[key] = item
            source_counts["openlibrary"] += 1
            source_results["openlibrary"].append(item)
            if len(all_pairs) >= max_results_total:
                return queries, list(all_pairs.values()), source_counts, source_results, source_queries

    return queries, list(all_pairs.values()), source_counts, source_results, source_queries


def load_env_file(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def openai_select_book(
    time_label: str,
    queries: list[str],
    candidates: list[dict],
    recent_selections: list[dict],
    extra_note: str | None = None,
) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    candidate_lines = "\n".join(
        f'- "{item["title"]}" — {item["author"]}' for item in candidates
    )
    if not candidate_lines:
        candidate_lines = "(no candidates found)"

    extra_note_text = extra_note or ""
    examples = load_prompt("examples.txt").strip()
    template = load_prompt("select_prompt.txt")
    hour, minute = time_label.split(":")
    hour_int = int(hour)
    am_pm = "AM" if hour_int < 12 else "PM"
    am_pm_context = f"{time_label} {am_pm}"
    fake_author_hint = random.choice(FAKE_AUTHOR_POOL)
    recent_lines = "\n".join(
        f'- {item["time"]}: "{item["title"]}" — {item["author"]} (score {item["score"]})'
        for item in recent_selections
    )
    if not recent_lines:
        recent_lines = "(none)"

    prompt = template.format(
        time_label=time_label,
        am_pm=am_pm,
        am_pm_context=am_pm_context,
        extra_note=extra_note_text,
        examples=examples,
        fake_author_hint=fake_author_hint,
        queries=", ".join(queries),
        candidate_lines=candidate_lines,
        recent_selections=recent_lines,
    ).strip()

    payload = {
        "model": DEFAULT_MODEL,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": "You are a precise JSON-only responder.",
            },
            {"role": "user", "content": prompt},
        ],
    }

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=60) as response:
        response_json = json.loads(response.read().decode("utf-8"))

    content = response_json["choices"][0]["message"]["content"].strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1:
            return json.loads(content[start : end + 1])
        raise


def iter_times(start_minutes: int | None, end_minutes: int | None):
    for hour in range(24):
        for minute in range(60):
            total = hour * 60 + minute
            if start_minutes is not None and total < start_minutes:
                continue
            if end_minutes is not None and total > end_minutes:
                continue
            yield hour, minute


def parse_time_label(label: str) -> int:
    parts = label.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid time format: {label}")
    hour = int(parts[0])
    minute = int(parts[1])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Time out of range: {label}")
    return hour * 60 + minute


STOP_WORDS = {
    "a",
    "an",
    "and",
    "the",
    "of",
    "for",
    "to",
    "in",
    "on",
    "at",
    "by",
    "with",
    "from",
    "into",
    "over",
    "under",
    "about",
}


def normalize_title(title: str) -> str:
    lowered = (title or "").lower()
    cleaned = []
    for ch in lowered:
        if ch.isalnum() or ch.isspace():
            cleaned.append(ch)
        else:
            cleaned.append(" ")
    words = [w for w in "".join(cleaned).split() if w and w not in STOP_WORDS]
    return " ".join(words)


def normalize_author_key(author: str) -> str:
    return (author or "").strip().lower()


def selection_key(selection: dict) -> str:
    return normalize_title(selection.get("title") or "")


def allow_am_pm_duplicate(current_time: str, previous_time: str) -> bool:
    current_total = parse_time_label(current_time)
    prev_total = parse_time_label(previous_time)
    current_hour, current_min = divmod(current_total, 60)
    prev_hour, prev_min = divmod(prev_total, 60)
    if current_min != prev_min:
        return False
    if (current_hour % 12) != (prev_hour % 12):
        return False
    return current_hour != prev_hour


def openai_verify_batch(
    batch: list[dict],
    extra_note: str | None = None,
) -> list[dict]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    lines = "\n".join(
        f'- {item["time"]}: "{item["title"]}" — {item["author"]} (score {item["score"]})'
        for item in batch
    )
    extra_note_text = extra_note or ""
    template = load_prompt("verify_prompt.txt")
    prompt = template.format(
        extra_note=extra_note_text,
        batch_lines=lines,
    ).strip()

    payload = {
        "model": DEFAULT_MODEL,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": "You are a precise JSON-only responder.",
            },
            {"role": "user", "content": prompt},
        ],
    }

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=60) as response:
        response_json = json.loads(response.read().decode("utf-8"))

    content = response_json["choices"][0]["message"]["content"].strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("[")
        end = content.rfind("]")
        if start != -1 and end != -1:
            return json.loads(content[start : end + 1])
        raise


def duplicate_conflicts(batch: list[dict], selections: dict) -> list[str]:
    conflicts = []
    seen = {}
    for item in batch:
        key = selection_key(item)
        if key in seen:
            other_time = seen[key]
            if not allow_am_pm_duplicate(item["time"], other_time):
                conflicts.append(
                    f'"{item["title"]}" — {item["author"]} already used for {other_time}'
                )
        else:
            seen[key] = item["time"]

    for item in batch:
        key = selection_key(item)
        for prev_time, prev_sel in selections.items():
            if prev_time == item["time"]:
                continue
            if selection_key(prev_sel) != key:
                continue
            if allow_am_pm_duplicate(item["time"], prev_time):
                continue
            conflicts.append(
                f'"{item["title"]}" — {item["author"]} already used for {prev_time}'
            )
            break
    return conflicts


def main():
    load_env_file(BASE_DIR / ".env")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    import argparse

    parser = argparse.ArgumentParser(
        description="Generate time-indexed book titles for a book-cover clock."
    )
    parser.add_argument(
        "--mode",
        choices=["search", "select", "all"],
        default="all",
        help="Run only the search step, selection step, or both.",
    )
    parser.add_argument(
        "--start",
        help="Optional start time HH:MM to limit processing.",
    )
    parser.add_argument(
        "--end",
        help="Optional end time HH:MM to limit processing.",
    )
    parser.add_argument(
        "--max-results-total",
        type=int,
        default=500,
        help="Max total results to keep per time.",
    )
    parser.add_argument(
        "--max-results-per-query",
        type=int,
        default=5,
        help="Max results to keep per individual query.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep seconds between external search requests.",
    )
    parser.add_argument(
        "--request-sleep",
        type=float,
        default=0.1,
        help="Sleep seconds between individual API requests.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear cached results before running.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-source search results for each time.",
    )
    parser.add_argument(
        "--no-dedupe-authors",
        action="store_false",
        dest="dedupe_authors",
        default=True,
        help="Disable author de-duplication within each source.",
    )
    args = parser.parse_args()

    start_minutes = parse_time_label(args.start) if args.start else None
    end_minutes = parse_time_label(args.end) if args.end else None

    if args.reset:
        for path in (TIME_CANDIDATES_PATH, SELECTION_CACHE_PATH, OUTPUT_PATH):
            if path.exists():
                path.unlink()
        time_candidates = {}
        selections = {}
    else:
        time_candidates = load_json(TIME_CANDIDATES_PATH, {})
        selections = load_json(SELECTION_CACHE_PATH, {})

    new_selected_times: list[str] = []

    for hour, minute in iter_times(start_minutes, end_minutes):
        time_label = f"{hour:02d}:{minute:02d}"

        if args.mode in {"search", "all"}:
            if time_label not in time_candidates:
                queries, candidates, source_counts, source_results, source_queries = (
                    collect_candidates_for_time(
                        hour,
                        minute,
                        max_results_total=args.max_results_total,
                        max_results_per_query=args.max_results_per_query,
                        request_sleep=args.request_sleep,
                        verbose=args.verbose,
                        dedupe_authors=args.dedupe_authors,
                    )
                )
                time_candidates[time_label] = {
                    "queries": queries,
                    "candidates": candidates,
                }
                save_json(TIME_CANDIDATES_PATH, time_candidates)
                print(
                    f"[search] {time_label} -> {len(queries)} queries, "
                    f"{len(candidates)} candidate books "
                    f"(google {source_counts['google']}, "
                    f"openlibrary {source_counts['openlibrary']})",
                    flush=True,
                )
                if args.verbose:
                    print(f"[search-results] {time_label}", flush=True)
                    print(
                        f"queries: google={source_queries['google']}, "
                        f"openlibrary={source_queries['openlibrary']}",
                        flush=True,
                    )
                    for source_name in ("google", "openlibrary"):
                        print(f"{source_name}:", flush=True)
                        for item in source_results[source_name]:
                            print(
                                f'- "{item["title"]}" — {item["author"]}',
                                flush=True,
                            )
                        print("", flush=True)

        if args.mode in {"select", "all"}:
            if time_label not in selections:
                info = time_candidates.get(time_label) or {"queries": [], "candidates": []}
                last_times = sorted(selections.keys())[-10:]
                recent_selections = [
                    {
                        "time": t,
                        "title": selections[t].get("title"),
                        "author": selections[t].get("author"),
                        "score": selections[t].get("score"),
                    }
                    for t in last_times
                ]
                extra_note = None
                attempt = 0
                while True:
                    try:
                        selection = openai_select_book(
                            time_label,
                            info["queries"],
                            info["candidates"],
                            recent_selections,
                            extra_note=extra_note,
                        )
                    except Exception as exc:
                        print(
                            f"[error] selection failed for {time_label}: {exc}. "
                            "Using fallback title and continuing.",
                            flush=True,
                        )
                        selection = {
                            "title": f"{time_label}",
                            "author": random.choice(FAKE_AUTHOR_POOL),
                            "score": 0,
                        }
                        break
                    sel_key = selection_key(selection)
                    duplicates = [
                        prev_time
                        for prev_time, prev_sel in selections.items()
                        if selection_key(prev_sel) == sel_key
                    ]
                    if not duplicates:
                        break
                    if any(
                        allow_am_pm_duplicate(time_label, prev_time)
                        for prev_time in duplicates
                    ):
                        break
                    attempt += 1
                    if attempt >= 6:
                        print(
                            f"[warn] duplicate selection persisted for {time_label}: "
                            f"{selection.get('title')} — {selection.get('author')}. "
                            "Keeping it and continuing.",
                            flush=True,
                        )
                        break
                    extra_note = (
                        "IMPORTANT: The selected title/author has already been used "
                        f"for {', '.join(duplicates)}. You MUST pick a different book "
                        "for this time (or invent a new one if needed)."
                    )
                selections[time_label] = selection
                save_json(SELECTION_CACHE_PATH, selections)
                save_json(OUTPUT_PATH, selections)
                print(
                    f"[select] {time_label} -> {selection.get('title')} "
                    f"— {selection.get('author')} (score {selection.get('score')})",
                    flush=True,
                )
                new_selected_times.append(time_label)

                if len(new_selected_times) % 10 == 0:
                    batch_times = new_selected_times[-10:]
                    batch = [
                        {
                            "time": t,
                            "title": selections[t].get("title"),
                            "author": selections[t].get("author"),
                            "score": selections[t].get("score"),
                        }
                        for t in batch_times
                    ]
                    attempt = 0
                    extra_note = None
                    verified_map = None
                    while True:
                        try:
                            verified = openai_verify_batch(batch, extra_note=extra_note)
                        except Exception as exc:
                            print(
                                f"[warn] batch verify failed for {batch_times[0]}–{batch_times[-1]}: "
                                f"{exc}. Keeping existing picks.",
                                flush=True,
                            )
                            verified_map = None
                            break
                        verified_map = {item.get("time"): item for item in verified}
                        if set(verified_map.keys()) != set(batch_times):
                            print(
                                "[warn] batch verify returned unexpected times. "
                                "Keeping existing picks.",
                                flush=True,
                            )
                            verified_map = None
                            break
                        prior_selections = {
                            k: v for k, v in selections.items() if k not in batch_times
                        }
                        conflicts = duplicate_conflicts(
                            [verified_map[t] for t in batch_times],
                            prior_selections,
                        )
                        if not conflicts:
                            break
                        attempt += 1
                        if attempt >= 4:
                            print(
                                "[warn] repeated duplicate violations during batch verify. "
                                "Keeping existing picks.",
                                flush=True,
                            )
                            verified_map = None
                            break
                        extra_note = (
                            "IMPORTANT: Duplicate title/author pairs are not allowed "
                            "across times (except AM/PM pairs). Fix these conflicts: "
                            + "; ".join(conflicts)
                        )

                    if verified_map is not None:
                        for t in batch_times:
                            item = verified_map[t]
                            previous = selections.get(t, {})
                            old_title = previous.get("title")
                            old_author = previous.get("author")
                            old_score = previous.get("score")
                            if (
                                old_title != item.get("title")
                                or old_author != item.get("author")
                                or old_score != item.get("score")
                            ):
                                print(
                                    f"[verify-change] {t}: "
                                    f'"{old_title}" — {old_author} (score {old_score}) '
                                    "-> "
                                    f'"{item.get("title")}" — {item.get("author")} '
                                    f'(score {item.get("score")})',
                                    flush=True,
                                )
                            selections[t] = {
                                "title": item.get("title"),
                                "author": item.get("author"),
                                "score": item.get("score"),
                            }
                        save_json(SELECTION_CACHE_PATH, selections)
                        save_json(OUTPUT_PATH, selections)
                        print(
                            f"[verify] {batch_times[0]}–{batch_times[-1]} -> batch cleaned",
                            flush=True,
                        )
                if args.sleep:
                    time.sleep(args.sleep)

    save_json(OUTPUT_PATH, selections)


if __name__ == "__main__":
    main()
