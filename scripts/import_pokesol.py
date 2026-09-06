"""Read a party out of a saved pokesol article. No picture involved.

    python scripts/import_pokesol.py article.html --out runs/imported.json
    python scripts/import_pokesol.py saved/ --append data/champions/parties_hand.json

The 46 hand parties were entered by reading screenshots. pokesol's article
cards are not screenshots: the species is the dex number in a sprite
filename, the ability, nature, item, four moves and the SP row are text, and
the printed stat line is a checksum on all of it. About a quarter of the
articles a pokedb search lists are pokesol ones, and they carry exactly what
the ranked-team archive does not -- what is in the four move slots and where
the points went.

**The stat line is the point.** Base stats plus SP plus nature determine the
six numbers exactly, so a parse either reproduces them or is wrong. That check
runs on every Pokemon and a card that fails it is refused rather than
imported: a wrong move slipped into the field silently, and one did, is worth
more damage than a party we did not take.

It also settles the nature without trusting a name table. Only one of the
twenty-one reproduces a given stat line, so the arithmetic names it and the
Japanese label is the cross-check -- if the two disagree, the card is held
back for a person to look at.

Fetching is not done here. Save the pages from a browser and point this at
them; see scripts/fetch_usage.py for why the crawling is hk's to run.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pkcm.data.dex import Stat, load_dex  # noqa: E402
from pkcm.engine.legality import team_errors  # noqa: E402
from pkcm.engine.pokemon import PokemonSet  # noqa: E402
from pkcm.engine.stats import NATURES, compute_stats, get_nature  # noqa: E402

CARD = r'(?=<img alt="[^"]*" loading="lazy" width="110")'
ORDER = (Stat.HP, Stat.ATK, Stat.DEF, Stat.SPA, Stat.SPD, Stat.SPE)


def strip(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", text)).strip()


def tables() -> tuple[dict, dict, dict, dict]:
    raw = ROOT / "data" / "raw" / "pokechams"

    def english(row) -> str:
        return re.sub(r"[^a-z0-9]", "", row["nameEn"].lower())

    species = {}
    for row in json.loads((raw / "champions_pokemon.json").read_text(encoding="utf-8")):
        key = f"{int(row['nationalDex']):04d}-{int(row['formIndex']):02d}"
        species[key] = row
    moves = {r["nameJa"]: english(r)
             for r in json.loads((raw / "moves.json").read_text(encoding="utf-8"))}
    items = {r["nameJa"]: english(r)
             for r in json.loads((raw / "items.json").read_text(encoding="utf-8"))}
    abilities = {r["nameJa"]: english(r)
                 for r in json.loads((raw / "abilities.json").read_text(encoding="utf-8"))}
    return species, moves, items, abilities


#: Only used to cross-check what the stat line already decided.
NATURE_JA = {
    "いじっぱり": "adamant", "ゆうかん": "brave", "さみしがり": "lonely",
    "やんちゃ": "naughty", "ずぶとい": "bold", "のんき": "relaxed",
    "わんぱく": "impish", "のうてんき": "lax", "ひかえめ": "modest",
    "れいせい": "quiet", "おっとり": "mild", "うっかりや": "rash",
    "おだやか": "calm", "なまいき": "sassy", "おとなしい": "gentle",
    "しんちょう": "careful", "ようき": "jolly", "むじゃき": "naive",
    "おくびょう": "timid", "せっかち": "hasty", "まじめ": "serious",
}


def parse_card(card: str) -> dict | None:
    """One Pokemon's card, as the page has it, before any mapping."""
    # Mega sprites are 0212_01.png; the base forme is 0212.png.
    sprite = re.search(r'width="110"[^>]*src="[^"]*?(\d{4})(?:[-_](\d+))?\.png"', card)
    name = re.search(r'<h3 class="truncate[^"]*"[^>]*>(.*?)</h3>', card, re.S)
    if not (sprite and name):
        return None
    labels = re.findall(
        r'<span class="inline-flex h-6 items-center rounded'
        r'(?: border border-border| bg-muted)? px-1\.5 text-xs[^"]*"[^>]*>(.*?)</span>',
        card, re.S)
    item = re.search(r'<img alt="([^"]*)" loading="lazy" width="24"', card)
    moves = [strip(one) for one in
             re.findall(r'<span class="truncate"[^>]*>(.*?)</span>', card, re.S)]
    table = re.search(r"<table[^>]*>(.*?)</table>", card, re.S)
    stats, points = [], []
    if table:
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table.group(1), re.S)
        if len(rows) >= 3:
            stats = [strip(c) for c in re.findall(r"<td[^>]*>(.*?)</td>", rows[1], re.S)]
            points = [strip(c) for c in re.findall(r"<td[^>]*>(.*?)</td>", rows[2], re.S)]
    return {
        "dex": sprite.group(1),
        "form": sprite.group(2) or "0",
        "name_ja": strip(name.group(1)),
        "ability_ja": strip(labels[0]) if labels else None,
        "nature_ja": strip(labels[1]) if len(labels) > 1 else None,
        "item_ja": html.unescape(item.group(1)) if item else None,
        "moves_ja": moves[:4],
        "stats": [int(one) for one in stats if one.isdigit()],
        "sp": [0 if one in ("—", "-", "") else int(one)
               for one in points if one in ("—", "-", "") or one.isdigit()],
    }


def build(raw: dict, dex, species, moves_ja, items_ja, abilities_ja) -> tuple:
    """A PokemonSet plus every complaint about it. Empty complaints means clean."""
    problems: list[str] = []
    key = f"{int(raw['dex']):04d}-{int(raw['form']):02d}"
    row = species.get(key)
    if row is None:
        return None, [f"{raw['name_ja']}: no species for {key}"]
    name = row["slug"].replace("-", "")
    if name not in dex.species:
        name = row["slug"]
    if name not in dex.species:
        return None, [f"{raw['name_ja']}: {row['slug']} is not in our dex"]

    # A Mega card names the Mega, and the stat line printed on it is the
    # Mega's. The set registers the base holding the stone, so the checksum
    # has to be read against the forme the numbers came from.
    shown = name
    base = dex.species[name]
    if base.is_mega and base.base_species:
        name = base.base_species

    moves = []
    for one in raw["moves_ja"]:
        found = moves_ja.get(one)
        if found is None or found not in dex.moves:
            problems.append(f"{raw['name_ja']}: move {one!r} did not map")
        else:
            moves.append(found)
    item = items_ja.get(raw["item_ja"] or "")
    if raw["item_ja"] and item is None:
        problems.append(f"{raw['name_ja']}: item {raw['item_ja']!r} did not map")
    ability = abilities_ja.get(raw["ability_ja"] or "")
    if raw["ability_ja"] and ability is None:
        problems.append(f"{raw['name_ja']}: ability {raw['ability_ja']!r} did not map")

    sp = tuple(raw["sp"]) if len(raw["sp"]) == 6 else None
    if sp is None:
        problems.append(f"{raw['name_ja']}: no SP row")
        return None, problems

    # The stat line names the nature: exactly one of the twenty-one reproduces
    # it. The Japanese label is then a cross-check on the parse, not its source.
    printed = raw["stats"]
    fits = []
    if len(printed) == 6:
        for candidate in sorted(NATURES):
            got = compute_stats(dex.species[shown].base_stats, sp, candidate)
            if [got[stat] for stat in ORDER] == printed:
                fits.append(candidate)
    labelled = NATURE_JA.get(raw["nature_ja"] or "")
    if len(fits) == 1:
        nature = fits[0]
        if labelled and labelled != nature:
            problems.append(f"{raw['name_ja']}: stats say {nature}, "
                            f"label says {labelled}")
    elif labelled:
        nature = labelled
        if printed:
            problems.append(f"{raw['name_ja']}: no nature reproduces the stat line "
                            f"{printed}; taking the label {labelled}")
    else:
        problems.append(f"{raw['name_ja']}: nature undetermined")
        return None, problems

    # A writer details the Pokemon the article is about and leaves the rest of
    # the card blank, and a blank card is legal -- four empty move slots and no
    # points break no rule. It is still not a set. Say so here rather than let
    # a one-move Corviknight into the field wearing a tick.
    if len(moves) < 4:
        problems.append(f"{raw['name_ja']}: only {len(moves)} of 4 moves given")
    if not any(sp):
        problems.append(f"{raw['name_ja']}: no points spent, the card is blank")

    built = PokemonSet(species=name, ability=ability or "__none__",
                       moves=tuple(moves), item=item, nature=nature, sp=sp)
    return built, problems


def read(path: Path, dex, species, moves_ja, items_ja, abilities_ja) -> dict:
    page = path.read_text(encoding="utf-8", errors="replace")
    title = re.search(r"<title>(.*?)</title>", page, re.S)
    team, problems = [], []
    for card in re.split(CARD, page)[1:]:
        raw = parse_card(card)
        if raw is None:
            continue
        built, complaints = build(raw, dex, species, moves_ja, items_ja, abilities_ja)
        problems.extend(complaints)
        if built is not None:
            team.append(built)
    return {"title": strip(title.group(1)) if title else path.stem,
            "source": path.name, "team": team, "problems": problems}


def main() -> int:
    for out in (sys.stdout, sys.stderr):
        try:
            out.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="saved .html files, or directories")
    parser.add_argument("--regulation", default="m_b")
    parser.add_argument("--out", default=str(ROOT / "runs" / "pokesol_import.json"))
    parser.add_argument("--strict", action="store_true",
                        help="keep only parties with no complaints at all")
    args = parser.parse_args()

    dex = load_dex()
    allowed = dex.regulation(args.regulation)
    species, moves_ja, items_ja, abilities_ja = tables()

    files: list[Path] = []
    for one in args.paths:
        path = Path(one)
        files.extend(sorted(path.glob("*.html")) if path.is_dir() else [path])

    kept, held = [], []
    for path in files:
        party = read(path, dex, species, moves_ja, items_ja, abilities_ja)
        errors = list(party["problems"])
        if len(party["team"]) != 6:
            errors.append(f"read {len(party['team'])} of 6")
        else:
            errors.extend(team_errors(dex, allowed, tuple(party["team"])))
        row = {"title": party["title"], "source": party["source"],
               "team": [{"species": one.species, "ability": one.ability,
                         "moves": list(one.moves), "item": one.item,
                         "nature": one.nature, "sp": list(one.sp)}
                        for one in party["team"]],
               "problems": errors}
        (kept if not errors else held).append(row)
        mark = "ok  " if not errors else "HELD"
        print(f"{mark} {party['title'][:44]:46} {len(party['team'])}/6")
        for complaint in errors:
            print(f"       {complaint}")

    payload = {"parties": kept, "held": [] if args.strict else held}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print(f"\n{len(kept)} clean, {len(held)} held -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
