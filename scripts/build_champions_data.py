"""Parse the Bulbapedia Regulation Set wikitext into a machine-readable legality file.

The wiki page lists every legal entry as a single template call::

    {{CPCard|0026|Raichu}}
    {{CPCard|0026|Raichu|ig=-Alola|name=[[...]]}}
    {{CPCard|0006|Charizard|ig=-Mega X|name=[[...]]}}

``ig`` is the in-game form suffix, and concatenating ``species + ig`` happens to
produce exactly Showdown's forme naming, so ``to_id(species + ig)`` gives us the
Showdown pokedex key for free.

The output also carries a coverage report: entries whose key is absent from
Showdown's pokedex are Champions-original (new Mega Evolutions) and will need
hand-authored stats in the override layer.

Usage:
    python scripts/build_champions_data.py [--regulation m_b]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "champions"

CARD_RE = re.compile(r"\{\{CPCard\|(?P<num>\d+)\|(?P<species>[^|}]+?)(?P<args>\|[^}]*)?\}\}")
IG_RE = re.compile(r"\|ig=(?P<ig>[^|}]*)")

# The in-game form suffix usually matches Showdown's forme naming exactly. These
# are the cases where it does not. One wiki card may expand to several Showdown
# entries (Champions treats Meowstic's Mega as a single option; Showdown splits
# it by gender), so the value is always a list.
FORM_ALIASES: dict[str, list[str]] = {
    "meowsticfemale": ["meowsticf"],
    "basculegionfemale": ["basculegionf"],
    "gourgeistjumbo": ["gourgeistsuper"],  # "Jumbo Variety" is Showdown's "Super"
    "meowsticmega": ["meowsticmmega", "meowsticfmega"],
}


def to_id(text: str) -> str:
    """Showdown's toID(): lowercase, drop everything that is not [a-z0-9]."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def parse_cards(section: str) -> list[dict]:
    entries = []
    for match in CARD_RE.finditer(section):
        species = match.group("species").strip()
        args = match.group("args") or ""
        ig_match = IG_RE.search(args)
        form = ig_match.group("ig").strip() if ig_match else ""
        raw_id = to_id(species + form)
        entries.append(
            {
                "dex_num": int(match.group("num")),
                "species": species,
                "form": form,
                "showdown_ids": FORM_ALIASES.get(raw_id, [raw_id]),
            }
        )
    return entries


def split_sections(text: str) -> tuple[str, str]:
    """Return (eligible-species section, mega section)."""
    mega_match = re.search(r"^===\s*Mega Evolutions\s*===\s*$", text, re.M)
    if mega_match is None:
        raise ValueError("could not locate the '===Mega Evolutions===' heading")

    eligible_match = re.search(r"^==\s*Eligible Pok.mon\s*==\s*$", text, re.M)
    if eligible_match is None:
        raise ValueError("could not locate the '==Eligible Pokemon==' heading")

    after_match = re.search(r"^==\s*Related articles\s*==\s*$", text, re.M)
    end = after_match.start() if after_match else len(text)

    return text[eligible_match.end():mega_match.start()], text[mega_match.end():end]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regulation", default="m_b", help="regulation slug, e.g. m_b")
    args = parser.parse_args()

    source = RAW_DIR / f"regulation_{args.regulation}.wikitext"
    if not source.exists():
        print(f"missing {source}; run scripts/fetch_regulation.py first", file=sys.stderr)
        return 1

    text = source.read_text(encoding="utf-8")
    eligible_section, mega_section = split_sections(text)
    eligible = parse_cards(eligible_section)
    megas = parse_cards(mega_section)

    pokedex = json.loads((RAW_DIR / "pokedex.json").read_text(encoding="utf-8"))

    for entry in eligible + megas:
        entry["missing"] = [i for i in entry["showdown_ids"] if i not in pokedex]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"regulation_{args.regulation}.json"
    out_path.write_text(
        json.dumps(
            {
                "regulation": args.regulation.upper().replace("_", "-"),
                "source": "https://bulbapedia.bulbagarden.net/wiki/Regulation_Set_"
                + args.regulation.upper().replace("_", "-"),
                "eligible": eligible,
                "megas": megas,
            },
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    missing_eligible = [e for e in eligible if e["missing"]]
    missing_megas = [e for e in megas if e["missing"]]

    n_eligible = sum(len(e["showdown_ids"]) for e in eligible)
    n_megas = sum(len(e["showdown_ids"]) for e in megas)
    print(f"eligible species : {len(eligible):4} cards -> {n_eligible:4} formes  ({len(missing_eligible)} unresolved)")
    print(f"legal megas      : {len(megas):4} cards -> {n_megas:4} formes  ({len(missing_megas)} unresolved)")
    print(f"written          : {out_path.relative_to(ROOT)}")

    if missing_eligible:
        print("\nUNRESOLVED SPECIES (fix the name mapping):")
        for e in missing_eligible:
            print(f"  {e['dex_num']:4} {e['species']}{e['form']}  -> {e['missing']}")

    if missing_megas:
        print(f"\nCHAMPIONS-ORIGINAL MEGAS needing hand-authored stats ({len(missing_megas)}):")
        for e in missing_megas:
            print(f"  {e['dex_num']:4} {e['species']}{e['form']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
