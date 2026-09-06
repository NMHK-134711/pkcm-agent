"""Index the team-building articles a pokedb search lists. For hk to run.

    python scripts/fetch_articles.py --season-start 3 --season-end 5
    python scripts/fetch_articles.py --from-files "saved page 1.html"

Each result card already carries the party's six species and their items, the
author's rank and rating, and -- the reason to collect it -- a link to the
article itself, where the moves and the spreads are. Those links are the input
to reading the articles; this script only makes the list.

Same rule as scripts/fetch_usage.py: champs.pokedb.tokyo/robots.txt names
ClaudeBot and Claude-SearchBot and disallows them everywhere, the server
enforces it, so the fetching is hk's. The default User-Agent says who it is
rather than pretending to be Chrome, there is a second and a half between
requests, every page is cached, and --from-files parses pages saved by hand
without touching the network.

The search has no page count, only a "next" link, so this walks until a page
comes back with no cards on it.

**What comes out.** ``data/champions/articles.json``: one row per article with
its link, its host, rank and rating, and the six ``dex-forme`` keys with item
ids. Reading the linked articles is a separate job with a separate policy --
pokesol and hatena and note allow it, blog.naver names ClaudeBot and does not.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SEARCH = "https://champs.pokedb.tokyo/article/search"
AGENT = ("pkcm-agent/0.1 (personal Pokemon Champions research; "
         "contact khg950520@gmail.com)")
CARD = r'(?=<article class="card article-card">)'


def strip(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", text)).strip()


def parse_page(page: str) -> list[dict]:
    """Every article card on one search page."""
    out = []
    for card in re.split(CARD, page)[1:]:
        link = re.search(r'href="(https?://(?!champs\.pokedb)[^"]+)"', card)
        if not link:
            continue
        url = html.unescape(link.group(1))
        host = urllib.parse.urlparse(url).netloc
        text = strip(card)
        rank = re.search(r"(\d+)\s*[位위]", text)
        rating = re.search(r"\b(\d{3,4})\s*\.?\s*(\d{1,3})\b", text)
        mons = re.findall(r'dex-(\d{4}-\d{2})-96" title="([^"]*)"', card)
        items = re.findall(r'item-icon-24 item-(\d+)" title="([^"]*)"', card)
        out.append({
            "url": url,
            "host": host,
            "rank": int(rank.group(1)) if rank else None,
            "rating": float(f"{rating.group(1)}.{rating.group(2)}") if rating else None,
            "team": [{"dex": key, "name": name} for key, name in mons],
            "items": [{"id": int(ident), "name": name} for ident, name in items],
        })
    return out


def search_url(args, page: int) -> str:
    query = {
        "rule": args.rule,
        "season_start": args.season_start,
        "season_end": args.season_end,
        "sort": "default",
        "per_page": args.per_page,
        "page": page,
    }
    return SEARCH + "?" + urllib.parse.urlencode(query)


def fetch(url: str, saved: Path, agent: str, delay: float) -> str | None:
    if saved.exists():
        return saved.read_text(encoding="utf-8", errors="replace")
    request = urllib.request.Request(url, headers={
        "User-Agent": agent,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "ja,en;q=0.8",
    })
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        print(f"  HTTP {error.code} on {url[:90]}", flush=True)
        return None
    except Exception as error:  # noqa: BLE001
        print(f"  {type(error).__name__} {error}", flush=True)
        return None
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_text(body, encoding="utf-8")
    time.sleep(delay)
    return body


def main() -> int:
    for out in (sys.stdout, sys.stderr):
        try:
            out.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season-start", type=int, default=3)
    parser.add_argument("--season-end", type=int, default=5)
    parser.add_argument("--rule", type=int, default=0, help="0 is singles")
    parser.add_argument("--per-page", type=int, default=30)
    parser.add_argument("--max-pages", type=int, default=200,
                        help="a stop in case the walk never sees an empty page")
    parser.add_argument("--delay", type=float, default=1.5)
    parser.add_argument("--agent", default=AGENT)
    parser.add_argument("--from-files", nargs="*", default=None,
                        help="parse pages saved by hand and fetch nothing")
    parser.add_argument("--cache", default=str(ROOT / "data" / "raw" / "pokedb_articles"))
    parser.add_argument("--out", default=str(ROOT / "data" / "champions" / "articles.json"))
    args = parser.parse_args()

    rows: list[dict] = []
    if args.from_files is not None:
        for one in args.from_files:
            path = Path(one)
            found = parse_page(path.read_text(encoding="utf-8", errors="replace"))
            print(f"  {path.name[:60]:62} {len(found)} articles")
            rows.extend(found)
    else:
        cache = Path(args.cache)
        print(f"seasons M-{args.season_start}..M-{args.season_end}, "
              f"{args.per_page} a page, {args.delay}s between requests")
        for page in range(1, args.max_pages + 1):
            saved = cache / (f"s{args.season_start}-{args.season_end}"
                             f"-r{args.rule}-p{page}.html")
            body = fetch(search_url(args, page), saved, args.agent, args.delay)
            if body is None:
                break
            found = parse_page(body)
            print(f"  page {page:3}  {len(found):3} articles", flush=True)
            if not found:
                break
            rows.extend(found)

    seen, unique = set(), []
    for row in rows:
        if row["url"] in seen:
            continue
        seen.add(row["url"])
        unique.append(row)

    hosts = Counter(row["host"] for row in unique)
    full = sum(1 for row in unique if len(row["team"]) == 6)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"seasons": [args.season_start, args.season_end],
                    "rule": args.rule, "articles": unique},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{len(unique)} articles ({full} with all six shown) -> {args.out}")
    print("by host:")
    for host, count in hosts.most_common():
        print(f"   {count:4}  {host}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
