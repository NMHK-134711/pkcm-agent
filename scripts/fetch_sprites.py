"""Download the sprites the browser UI draws, and only those.

https://github.com/PokeAPI/sprites indexes by PokeAPI's *pokemon* id, not by
national number: Mega Venusaur is 10033, Hisuian Samurott 10236. So the join
needs PokeAPI's ``pokemon.csv``, whose ``identifier`` column -- ``venusaur-mega``,
``samurott-hisui`` -- is our species id with the hyphens taken out.

That covers 293 of the 311 species Champions fields, including 74 of its 76
Megas: PokeAPI carries even the ones Champions invented, ``meganium-mega`` and
``glimmora-mega`` among them. The eighteen that miss are forms PokeAPI spells
differently (``aegislash-shield`` for our ``aegislash``) or that only Champions
has (Meowstic's two Megas), and they fall back to their species' default sprite
with the substitution reported. **Reported, because a silently wrong sprite is
a silently wrong sprite** -- the fallback is a picture of the right Pokemon in
the wrong forme, which is much better than nothing and still not the truth.

Front and back, because a battle shows one of each.

Output goes to ``data/raw/sprites/`` (gitignored) with a MANIFEST recording the
mapping, so the UI can look a species up without the CSV.

Usage:
    python scripts/fetch_sprites.py [--force]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pkcm.data.dex import load_dex  # noqa: E402

RAW_DIR = ROOT / "data" / "raw" / "sprites"
POKEMON_CSV = "https://raw.githubusercontent.com/PokeAPI/pokeapi/master/data/v2/csv/pokemon.csv"
SPRITES = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/"
USER_AGENT = "pkcm-agent/0.1 (personal project; sprite cache for a local UI)"

#: front-facing goes on the far side, back-facing on ours.
VIEWS = {"front": "", "back": "back/"}


def ident(value: str) -> str:
    return "".join(c for c in (value or "").lower() if c.isalnum())


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def pokeapi_ids() -> tuple[dict[str, int], dict[int, int]]:
    """identifier -> pokemon id, and species (national) number -> its default."""
    rows = list(csv.DictReader(io.StringIO(fetch(POKEMON_CSV).decode("utf-8"))))
    by_ident = {ident(row["identifier"]): int(row["id"]) for row in rows}
    default = {int(row["species_id"]): int(row["id"])
               for row in rows if row["is_default"] == "1"}
    return by_ident, default


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):  # pragma: no cover
            pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="re-download even where the file is already here")
    args = parser.parse_args()

    dex = load_dex()
    regulation = dex.regulation("m_b")
    fieldable = sorted(set(regulation.legal_species) | set(regulation.legal_megas))

    print("resolving PokeAPI ids...", flush=True)
    by_ident, default = pokeapi_ids()

    mapping: dict[str, int] = {}
    substituted: list[str] = []
    for species_id in fieldable:
        found = by_ident.get(species_id)
        if found is None:
            found = default.get(dex.species[species_id].dex_num)
            if found is not None:
                substituted.append(species_id)
        if found is not None:
            mapping[species_id] = found
    print(f"  {len(mapping)}/{len(fieldable)} species resolved, "
          f"{len(substituted)} using their species' default sprite")
    if substituted:
        print(f"  substituted: {', '.join(substituted)}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    wanted = {pokemon_id for pokemon_id in mapping.values()}
    missing: list[str] = []
    downloaded = cached = 0
    for view, prefix in VIEWS.items():
        target_dir = RAW_DIR / view
        target_dir.mkdir(exist_ok=True)
        for index, pokemon_id in enumerate(sorted(wanted), 1):
            target = target_dir / f"{pokemon_id}.png"
            if target.exists() and not args.force:
                cached += 1
                continue
            try:
                target.write_bytes(fetch(f"{SPRITES}{prefix}{pokemon_id}.png"))
                downloaded += 1
            except Exception as error:  # pragma: no cover - network
                missing.append(f"{view}/{pokemon_id}: {error}")
            if index % 50 == 0:
                print(f"  {view}: {index}/{len(wanted)}", flush=True)

    print(f"\n{downloaded} downloaded, {cached} already here")
    if missing:
        print(f"{len(missing)} could not be fetched:")
        for line in missing[:10]:
            print(f"  {line}")

    (RAW_DIR / "MANIFEST.json").write_text(json.dumps({
        "source": SPRITES,
        "note": ("species id -> PokeAPI pokemon id. Files are <id>.png under "
                 "front/ and back/."),
        "substituted": substituted,
        "species": mapping,
        "_fetched": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=1), encoding="utf-8")
    print(f"manifest -> {RAW_DIR / 'MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
