"""Save each article's pictures into a folder of its own, for a person to read.

    python scripts/fetch_article_images.py --host note.com
    python scripts/fetch_article_images.py --host hatena --delay 4

note and the hatena blogs put their parties in screenshots, and every author
writes the surrounding text their own way, so there is nothing to parse
reliably. This does not try. It collects the pictures, one directory per
article, with a manifest naming the article, its rank and rating, and the six
species the search index already knows it used -- so whoever reads the images
knows what they should be looking at.

Every host is asked first, the page's host and the image's, and a host that
disallows this agent or ClaudeBot is skipped with its rule printed --
blog.naver.com does both. Three seconds between requests by default. These are
people's blogs.

Furniture is left behind: avatars, buttons, favicons, share icons and
thumbnails asking for a width of a couple of hundred pixels. What is kept is
what a party screenshot looks like -- a large image from the platform's own
content CDN.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from fetch_article_bodies import AGENT, robots_for, slug_for  # noqa: E402

#: Where each platform serves the pictures inside a post.
CONTENT_HOSTS = ("assets.st-note.com", "st-hatena.com", "hatena.ne.jp",
                 "pokesol.app", "livedoor.blogimg.jp", "blogimg.goo.ne.jp")
#: Avatars, buttons and the rest, by what their URL says about them.
FURNITURE = re.compile(
    r"(?i)(favicon|avatar|profile|icon|logo|banner|badge|button|share|ogp"
    r"|preset_user_image|/poc-image/|spacer|blank)")


def wanted_images(page: str, base: str) -> list[str]:
    """Content pictures, largest form, in the order they appear."""
    found: list[str] = []
    for match in re.finditer(r'<img[^>]+?src="([^"]+)"', page):
        url = urllib.parse.urljoin(base, match.group(1))
        host = urllib.parse.urlparse(url).netloc
        if not any(one in host for one in CONTENT_HOSTS):
            continue
        if FURNITURE.search(url):
            continue
        # A thumbnail asks for a small width; the same path without it is the
        # picture itself, and that is what is worth keeping.
        small = re.search(r"[?&]width=(\d+)", url)
        if small and int(small.group(1)) < 600:
            url = re.sub(r"[?&]width=\d+", "", url)
        if url not in found:
            found.append(url)
    return found


def get(url: str, agent: str, timeout: int = 60) -> bytes | None:
    request = urllib.request.Request(url, headers={
        "User-Agent": agent,
        "Accept": "*/*",
        "Accept-Language": "ja,en;q=0.8",
    })
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except Exception as error:  # noqa: BLE001
        print(f"      {type(error).__name__} {getattr(error, 'code', '')} "
              f"{url[:70]}", flush=True)
        return None


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
    parser.add_argument("--host", default=None, help="substring of the host to take")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--max-images", type=int, default=20,
                        help="per article; a long post can carry dozens of "
                             "screenshots that are not the party")
    parser.add_argument("--agent", default=AGENT)
    parser.add_argument("--out", default=str(ROOT / "data" / "raw" / "article_images"))
    args = parser.parse_args()

    index = json.loads(Path(args.index).read_text(encoding="utf-8"))
    wanted = [row for row in index["articles"]
              if args.host is None or args.host in row["host"]]
    if args.limit:
        wanted = wanted[:args.limit]
    print(f"{len(wanted)} articles across {len({r['host'] for r in wanted})} hosts")

    allowed: dict[str, tuple[bool, str]] = {}

    def may(host: str) -> bool:
        if host not in allowed:
            allowed[host] = robots_for(host, args.agent)
            ok, why = allowed[host]
            if not ok:
                print(f"  {host:38} SKIPPED -- {why}", flush=True)
            time.sleep(0.5)
        return allowed[host][0]

    root = Path(args.out)
    articles = images = skipped = refused = 0
    for row in wanted:
        if not may(row["host"]):
            refused += 1
            continue
        folder = root / row["host"] / slug_for(row["url"]).replace(".html", "")
        manifest = folder / "article.json"
        if manifest.exists():
            skipped += 1
            continue

        page = get(row["url"], args.agent)
        time.sleep(args.delay)
        if page is None:
            continue
        text = page.decode("utf-8", "replace")
        title = re.search(r"<title>(.*?)</title>", text, re.S)
        picks = wanted_images(text, row["url"])[:args.max_images]

        folder.mkdir(parents=True, exist_ok=True)
        saved = []
        for number, url in enumerate(picks, 1):
            if not may(urllib.parse.urlparse(url).netloc):
                continue
            blob = get(url, args.agent)
            time.sleep(args.delay)
            if not blob or len(blob) < 8000:      # an 8KB "picture" is a button
                continue
            suffix = re.search(r"\.(jpe?g|png|gif|webp)", url.lower())
            name = f"{number:02d}.{suffix.group(1) if suffix else 'jpg'}"
            (folder / name).write_bytes(blob)
            saved.append({"file": name, "url": url, "bytes": len(blob)})
            images += 1

        manifest.write_text(json.dumps({
            "url": row["url"], "host": row["host"],
            "title": re.sub(r"\s+", " ", title.group(1)).strip() if title else None,
            "rank": row["rank"], "rating": row["rating"],
            "team": [t["dex"] for t in row["team"]],
            "team_names": [t["name"] for t in row["team"]],
            "items": [i["name"] for i in row["items"]],
            "images": saved,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        articles += 1
        print(f"  [{articles:4}] {len(saved):2} images  {row['rank'] or '?':>5}  "
              f"{(title.group(1)[:44] if title else row['url'][-44:])}", flush=True)

    print(f"\n{articles} articles, {images} images, {skipped} already had, "
          f"{refused} refused by robots")
    print(f"into {root}")
    for why, count in Counter(why for ok, why in allowed.values() if not ok).most_common():
        print(f"   {count:3} hosts: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
