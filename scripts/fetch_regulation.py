"""Download a Bulbapedia regulation-set page as raw wikitext.

The rendered HTML is template soup, but ``action=raw`` gives us the source, where
every legal entry is one tidy ``{{CPCard|...}}`` call. Output lands in
``data/raw/regulation_<slug>.wikitext`` for ``build_champions_data.py`` to parse.

Usage:
    python scripts/fetch_regulation.py m_b
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
URL = "https://bulbapedia.bulbagarden.net/w/index.php?title=Regulation_Set_{name}&action=raw"
USER_AGENT = "Mozilla/5.0 (compatible; pkcm-agent/0.1; personal research project)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slug", nargs="?", default="m_b", help="regulation slug, e.g. m_b")
    args = parser.parse_args()

    page = args.slug.upper().replace("_", "-")
    request = urllib.request.Request(URL.format(name=page), headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        text = response.read().decode("utf-8")

    if "CPCard" not in text:
        print(f"page {page} has no CPCard templates; is the slug right?", file=sys.stderr)
        return 1

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"regulation_{args.slug}.wikitext"
    out_path.write_text(text, encoding="utf-8")
    print(f"fetched Regulation Set {page}: {len(text)} bytes -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
