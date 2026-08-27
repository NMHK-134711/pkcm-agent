"""Render the pkmnchamps archive into party sample text files.

Input is whatever ``scripts/fetch_pkmnchamps.py`` cached. Output is one plain
text file per party in the layout of ``party_samples/메가아쿠스타_비파티.txt``
-- which is a copy-paste of a party page's slot cards, so that is what this
reproduces, field for field:

    Mega메가아쿠스타        <- "Mega" is the badge over the sprite, then the name
    아쿠스타나이트          <- item
    고집                    <- nature
    ↑공격 ↓특공             <- what the nature moves
    체력32 공격14 특방20    <- SP, non-zero only, in HP/A/B/C/D/S order
    아쿠아브레이크          <- four moves
    사이코커터
    아이스스피너
    아쿠아제트
                            <- blank line between slots

The ability is deliberately absent: the site card shows it, the sample does not.
``--with-ability`` puts it back on the line under the name.

Names come from the site's own dex, not from ``data/champions/names.json``,
because the samples are meant to read the way the site does and the two disagree
on formes (메가 플라엣테(영원의 꽃) vs 메가플라엣테).

Three slug repairs are needed on the way, all because the archive is stitched
together from Pokepaste exports that spell things their own way:

* ``basculegion`` and ``aegislash`` have no base-forme entry -- the dex only
  knows ``basculegion-male`` and ``aegislash-shield``.
* Mega stones arrive under half a dozen spellings (``starminite`` for
  ``starmienite``, three different Dragonite stones, once a bare species slug).
  A mega slot holds its own stone, so the stone map settles it. The site itself
  gives up here and renders no item at all; ``--strict`` matches that instead.
* A few parties name the base species and let the stone imply the mega. Those
  are left as the base species, which is what the site's slot card shows.

Anything still unresolved is written as its English slug and listed in the run
summary, so a bad name is visible rather than quietly plausible.

Usage:
    python scripts/export_party_samples.py [--format single|double|all]
                                           [--out DIR] [--with-ability] [--strict]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "pkmnchamps"
OUT_DIR = ROOT / "party_samples" / "pkmnchamps"

#: SP order and labels as the site prints them.
STATS = (("hp", "체력"), ("atk", "공격"), ("def", "방어"),
         ("spa", "특공"), ("spd", "특방"), ("spe", "속도"))
STAT_KO = dict(STATS)

#: The site's cards are copy-pasted out of a browser, so CRLF and no final
#: newline -- matching the hand-made sample exactly.
NEWLINE = "\r\n"

#: Formes the archive names by their base species, which the dex does not carry.
#: Both are cases where the game has one battle forme and the dex names it.
BASE_FORME = {"basculegion": "basculegion-male", "aegislash": "aegislash-shield"}

UNSAFE = re.compile(r"[^\w가-힣ぁ-んァ-ン一-龥.-]+")


def _flat(slug: str) -> str:
    return slug.replace("-", "").replace("'", "").replace(".", "").lower()


class Names:
    """The site's dex, with the repairs the archive data needs to land on it."""

    def __init__(self, tables: dict):
        self.species = tables["species"]
        self.moves = tables["moves"]
        self.items = tables["items"]
        self.abilities = tables["abilities"]
        self.natures = tables["natures"]
        self.stones = tables["mega_stones"]
        self.species_abilities = tables["species_abilities"]
        self.mega_of = {stone: forme for forme, stone in self.stones.items()}
        self.unresolved: Counter[tuple[str, str]] = Counter()
        self.loose = {kind: {_flat(k): k for k in table}
                      for kind, table in (("move", self.moves),
                                          ("item", self.items),
                                          ("ability", self.abilities))}

    def look_up(self, kind: str, table: dict, slug: str) -> str:
        """Table lookup, tolerating a slug that only differs in its hyphens.

        The exports are hand-typed in places -- ``galewings`` for ``gale-wings``.
        Ignoring the separators is still an exact match, not a guess.
        """
        if slug in table:
            return table[slug]
        loose = self.loose[kind].get(_flat(slug))
        return table[loose] if loose else self.miss(kind, slug)

    def miss(self, kind: str, slug: str) -> str:
        self.unresolved[(kind, slug)] += 1
        return slug

    def species_slug(self, slug: str) -> str:
        """The dex key for a slot's species, or the raw slug if there is none."""
        if slug in self.species:
            return slug
        if slug in BASE_FORME and BASE_FORME[slug] in self.species:
            return BASE_FORME[slug]
        for suffix in ("-mega-x", "-mega-y", "-mega"):
            if not slug.endswith(suffix):
                continue
            base = slug[: -len(suffix)]
            # floette-mega is really floette-eternal-mega: same base, same mega
            # suffix, an intermediate forme name the export dropped.
            wider = sorted(k for k in self.species
                           if k.startswith(base + "-") and k.endswith(suffix))
            if len(wider) == 1:
                return wider[0]
        return slug

    def stone_slug(self, slug: str | None, species: str, strict: bool) -> str | None:
        """The item the slot really holds, in the dex's spelling."""
        if not slug:
            return None
        canonical = self.loose["item"].get(_flat(slug))
        if canonical:
            return canonical
        if strict:
            return None  # what the site does: an item it cannot read is not shown
        # A mega slot holds its own stone whatever the export called it -- try
        # the slot's forme, then the mega of its base species, which covers the
        # parties that name the base and let the stone imply the mega.
        stone = self.stones.get(species) or self.stones.get(f"{species}-mega")
        return stone if stone in self.items else self.miss("item", slug)

    def resolve_slot(self, slot: dict, strict: bool) -> tuple[str, str | None]:
        """Return the slot's (species key, item key), mega evolution applied.

        A party may name the base species and leave the mega to the stone --
        ``glimmora`` holding ``glimmorite`` is Mega Glimmora, and the site's
        card says so. That inference is the stone's alone: a stone the dex
        cannot read leaves the slot un-evolved, which is why the misspellings
        matter beyond the item line.
        """
        species = self.species_slug(slot["pokemon"])
        item = self.stone_slug(slot.get("item"), species, strict)
        forme = self.mega_of.get(item or "")
        if forme and forme.startswith(species):
            species = forme
        return species, item

    def move_name(self, slug: str) -> str:
        return self.look_up("move", self.moves, slug)

    def ability_name(self, slug: str | None, species: str) -> str | None:
        """The ability the slot actually battles with.

        A mega's ability comes with the forme -- Mega Starmie is 천하장사 no
        matter that the slot was exported holding Starmie's natural-cure -- so
        where the dex gives the forme exactly one ability, that one wins.
        """
        forme = self.species_abilities.get(species) or []
        if len(forme) == 1:
            slug = forme[0]
        if not slug:
            return None
        return self.look_up("ability", self.abilities, slug)

    def nature_lines(self, slug: str | None) -> list[str]:
        if not slug:
            return []
        nature = self.natures.get(slug)
        if nature is None:
            return [self.miss("nature", slug)]
        if nature["up"] == nature["down"]:
            return [nature["ko"]]  # hardy, docile and friends move nothing
        return [nature["ko"],
                f"↑{STAT_KO[nature['up']]} ↓{STAT_KO[nature['down']]}"]


def render_slot(slot: dict, names: Names, with_ability: bool,
                strict: bool) -> list[str]:
    species, item = names.resolve_slot(slot, strict)
    name = names.species.get(species) or names.miss("species", species)
    lines = [f"Mega{name}" if "-mega" in species else name]

    if with_ability:
        ability = names.ability_name(slot.get("ability"), species)
        if ability:
            lines.append(ability)

    if item:
        lines.append(names.look_up("item", names.items, item))

    lines += names.nature_lines(slot.get("nature"))

    evs = slot.get("evs") or {}
    spread = " ".join(f"{label}{evs[key]}" for key, label in STATS if evs.get(key))
    if spread:
        lines.append(spread)

    lines += [names.move_name(move) for move in slot["moves"] if move]
    return lines


def render_party(party: dict, names: Names, with_ability: bool,
                 strict: bool) -> str:
    blocks = [NEWLINE.join(render_slot(slot, names, with_ability, strict))
              for slot in party["showdown_slots"] if slot]
    return (NEWLINE * 2).join(blocks)


def is_complete(party: dict) -> bool:
    slots = [s for s in party.get("showdown_slots") or [] if s]
    return len(slots) == 6 and all(
        s.get("nature") and s.get("evs") and any((s.get("evs") or {}).values())
        for s in slots
    )


def filename(party: dict, taken: set[str]) -> str:
    """A stable, sortable, filesystem-safe name: rate first, then the title.

    Rate first because that is the only ordering the archive itself asserts, and
    the strongest thing to know about a party before opening it.
    """
    rate = party.get("rate")
    head = f"{rate}_" if rate else ""
    stem = UNSAFE.sub("_", (party.get("title") or party["id"])).strip("_")[:80]
    candidate = f"{head}{stem}"
    if candidate in taken:  # two teams, one title -- the uuid separates them
        candidate = f"{candidate}_{party['id'][:8]}"
    taken.add(candidate)
    return f"{candidate}.txt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", default="all",
                        choices=("single", "double", "all"),
                        help="battle format to export (default: all)")
    parser.add_argument("--out", type=Path, default=OUT_DIR,
                        help=f"output directory (default: {OUT_DIR})")
    parser.add_argument("--with-ability", action="store_true",
                        help="include the ability line the site card shows")
    parser.add_argument("--strict", action="store_true",
                        help="do not repair misspelled mega stones; drop the "
                             "item the way the site itself does")
    args = parser.parse_args()

    parties_path = RAW_DIR / "archive_parties.json"
    names_path = RAW_DIR / "names_ko.json"
    if not parties_path.exists() or not names_path.exists():
        print("no cache -- run: python scripts/fetch_pkmnchamps.py",
              file=sys.stderr)
        return 1

    parties = json.loads(parties_path.read_text(encoding="utf-8"))
    names = Names(json.loads(names_path.read_text(encoding="utf-8")))

    wanted = [p for p in parties
              if args.format == "all" or p.get("battle_format") == args.format]
    if not wanted:
        print(f"no {args.format} parties in the cache", file=sys.stderr)
        return 1

    index = ["\t".join(("file", "format", "rate", "rank", "source",
                        "owner", "complete", "url"))]
    counts: Counter[str] = Counter()
    taken: dict[str, set[str]] = {}
    for party in wanted:
        fmt = party.get("battle_format") or "unknown"
        out_dir = args.out / fmt
        out_dir.mkdir(parents=True, exist_ok=True)
        name = filename(party, taken.setdefault(fmt, set()))
        text = render_party(party, names, args.with_ability, args.strict)
        (out_dir / name).write_text(text, encoding="utf-8", newline="")
        counts[fmt] += 1
        index.append("\t".join((
            f"{fmt}/{name}", fmt,
            str(party.get("rate") or ""), str(party.get("rank") or ""),
            party.get("source") or "", party.get("owner_name") or "",
            "yes" if is_complete(party) else "no",
            f"https://pkmnchamps.com/parties/{party['id']}",
        )))

    (args.out / "index.tsv").write_text("\n".join(index) + "\n", encoding="utf-8")

    for fmt, count in sorted(counts.items()):
        print(f"  {fmt:8} {count:>4} parties -> {args.out / fmt}")
    partial = [p for p in wanted if not is_complete(p)]
    if partial:
        print(f"\n  {len(partial)}파티는 사이트에도 성격/노력치가 없다 "
              f"(index.tsv의 complete=no):")
        for party in partial:
            print(f"    {party.get('title', party['id'])[:60]}")
    if names.unresolved:
        print("\n  이름을 못 찾아 영문 슬러그로 적은 것 "
              "(사이트 도감에 없음):", file=sys.stderr)
        for (kind, slug), n in names.unresolved.most_common():
            print(f"    {kind:8} {slug:24} x{n}", file=sys.stderr)
    print(f"\nwrote {args.out / 'index.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
