"""Fetch the articles the index lists, from the hosts that allow it.

    python scripts/fetch_article_bodies.py --host pokesol.app
    python scripts/fetch_article_bodies.py --host hatena --delay 4

``data/champions/articles.json`` has 666 links across 233 hosts. The party's
six species and items are already in that index; the article is where the
four move slots and the spread are, which nothing else has.

**Every host is asked first.** Its robots.txt is fetched and read for this
agent *and* for ClaudeBot, and a host that disallows either is skipped with
its rule printed. blog.naver.com names ClaudeBot and says in a comment that
access for AI training is prohibited, so its articles are never fetched here
however the flags are set -- they are for a person to open.

Polite by construction: three seconds between requests by default, one host
at a time, every page cached, and a resume that skips what is already down.
These are people's personal blogs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from urllib.robotparser import RobotFileParser

ROOT = Path(__file__).resolve().parent.parent

AGENT = ("pkcm-agent/0.1 (personal Pokemon Champions research; "
         "contact khg950520@gmail.com)")
#: Asked as well as our own name. A site that shuts out ClaudeBot has said
#: what it thinks about this, and the fact that a different string gets served
#: is not permission.
ALSO = "ClaudeBot"


def robots_for(host: str, agent: str, timeout: int = 20) -> tuple[bool, str]:
    """(may we fetch this host, why not)."""
    url = f"https://{host}/robots.txt"
    parser = RobotFileParser()
    try:
        request = urllib.request.Request(url, headers={"User-Agent": agent})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            return False, f"robots.txt returned {error.code}"
        return True, "no robots.txt"
    except Exception as error:  # noqa: BLE001
        return False, f"robots.txt unreachable: {type(error).__name__}"
    parser.parse(text.splitlines())
    probe = f"https://{host}/"
    for who in (agent, ALSO):
        if not parser.can_fetch(who, probe):
            return False, f"robots.txt disallows {who}"
    if re.search(r"(?i)claude", text) and not parser.can_fetch(ALSO, probe):
        return False, "robots.txt names Claude"
    return True, ""


def slug_for(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    tail = re.sub(r"[^A-Za-z0-9._-]", "_", (parsed.path or "/").strip("/")) or "index"
    return f"{tail[:120]}.html"


def main() -> int:
    for out in (sys.stdout, sys.stderr):
        try:
            out.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", default=str(ROOT / "data" / "champions" / "articles.json"))
    parser.add_argument("--host", default=None,
                        help="substring of the host to take, e.g. pokesol.app")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--agent", default=AGENT)
    parser.add_argument("--cache", default=str(ROOT / "data" / "raw" / "articles"))
    args = parser.parse_args()

    index = json.loads(Path(args.index).read_text(encoding="utf-8"))
    wanted = [row for row in index["articles"]
              if args.host is None or args.host in row["host"]]
    if args.limit:
        wanted = wanted[:args.limit]
    print(f"{len(wanted)} articles across {len({r['host'] for r in wanted})} hosts")

    allowed: dict[str, tuple[bool, str]] = {}
    cache = Path(args.cache)
    got = skipped = refused = failed = 0
    for row in wanted:
        host = row["host"]
        if host not in allowed:
            allowed[host] = robots_for(host, args.agent)
            ok, why = allowed[host]
            print(f"  {host:38} {'allowed' if ok else 'SKIPPED -- ' + why}", flush=True)
            time.sleep(1.0)
        ok, _ = allowed[host]
        if not ok:
            refused += 1
            continue

        saved = cache / host / slug_for(row["url"])
        if saved.exists():
            skipped += 1
            continue
        request = urllib.request.Request(row["url"], headers={
            "User-Agent": args.agent,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "ja,en;q=0.8",
        })
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read().decode("utf-8", "replace")
        except Exception as error:  # noqa: BLE001 - one dead link is not a failure
            print(f"    {type(error).__name__} {getattr(error, 'code', '')} "
                  f"{row['url'][:70]}", flush=True)
            failed += 1
            time.sleep(args.delay)
            continue
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_text(body, encoding="utf-8")
        got += 1
        if got % 25 == 0:
            print(f"    {got} fetched", flush=True)
        time.sleep(args.delay)

    print(f"\nfetched {got}, already had {skipped}, refused by robots {refused}, "
          f"failed {failed}")
    print(f"cache {cache}")
    blocked = Counter(why for ok, why in allowed.values() if not ok)
    for why, count in blocked.most_common():
        print(f"   {count:3} hosts: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
