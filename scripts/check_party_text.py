"""Proof-read hand-transcribed parties before they become training data.

    python scripts/check_party_text.py party_samples/pkmnchamps/single
    python scripts/check_party_text.py <dir> --json data/champions/parties_hand.json

A party typed out of a screenshot fails differently from one pulled off an API.
Nothing is malformed -- every line is a plausible line -- and one wrong syllable
produces a Pokemon that is legal, playable, and not the one the ladder brought.
So every field is resolved against the game's own name tables and refused if it
does not land. Nothing here guesses.

Two layouts are accepted, both as the sites render them.

**Korean**, five lines per Pokemon, or six with a nature after the ability::

    팬텀
    저주받은바디
    팬텀나이트
    셰도볼/오물웨이브/길동무/방어
    12/0/4/21/0/29

Without a nature the set is recorded neutral, which is a real loss -- a nature
moves two stats by ten percent -- so the files missing one are listed at the end.

**Japanese**, the pokedb layout::

    ドヒドイデ @ たべのこし
    特性: さいせいりょく
    性格: しんちょう
    157(32)-83-186(14)-65-200(20+)-55
    アクアブレイク / どくどく / くろいきり / じこさいせい

That layout carries its own checksum. The stat line prints the *computed* stat
beside the SP that bought it, so both are recomputed from the base stats and the
nature. A mistyped SP moves the stat and the two stop agreeing -- which catches
the errors no name table can, because a wrong number is still a number.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pkcm.data.dex import load_dex  # noqa: E402
from pkcm.engine.legality import (PokemonSet, set_errors,  # noqa: E402
                                  team_errors)
from pkcm.engine.stats import (NATURES, SP_PER_STAT_CAP, SP_TOTAL,  # noqa: E402
                               Stat, compute_stat, get_nature)

STAT_ORDER = ("HP", "ATK", "DEF", "SPA", "SPD", "SPE")

#: pokedb writes a Mega Stone generically. A species has at most one, so the
#: species names the stone and nothing is being guessed.
GENERIC_STONE = {"メガストーン", "메가스톤", "메가스톤아이템", "megastone"}

#: A raised stat and a lowered one name exactly one Stat Alignment, so a
#: transcription that marks them does not have to name it -- and cannot
#: misspell it. Twenty of the twenty-one are reachable this way; the neutral
#: one has nothing to mark.
BY_SHIFT = {(nature.boosted, nature.hindered): key
            for key, nature in NATURES.items()}

#: The pokechams tables carry no natures, and there are only twenty-five.
JAPANESE_NATURES = {
    "がんばりや": "hardy", "さみしがり": "lonely", "ゆうかん": "brave",
    "いじっぱり": "adamant", "やんちゃ": "naughty", "ずぶとい": "bold",
    "すなお": "docile", "のんき": "relaxed", "わんぱく": "impish",
    "のうてんき": "lax", "おくびょう": "timid", "せっかち": "hasty",
    "まじめ": "serious", "ようき": "jolly", "むじゃき": "naive",
    "ひかえめ": "modest", "おっとり": "mild", "れいせい": "quiet",
    "てれや": "bashful", "うっかりや": "rash", "おだやか": "calm",
    "おとなしい": "gentle", "なまいき": "sassy", "しんちょう": "careful",
    "きまぐれ": "quirky",
}


def loose(value: str) -> str:
    """A key that survives spacing, punctuation and full-width Roman letters.

    ``메가라이츄 Y`` and ``メガライチュウＹ`` both have to reach ``raichumegay``,
    and the second writes its Y in the full-width block.
    """
    out = []
    for ch in value or "":
        if ch in " \t·・/(),-'’.":
            continue
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:
            ch = chr(code - 0xFEE0)
        out.append(ch.lower())
    return "".join(out)


def ident(value: str) -> str:
    return "".join(c for c in (value or "").lower() if c.isalnum())


class Complaint(Exception):
    """Something in the text does not resolve. Never repaired, only reported."""


class Tables:
    """Korean and Japanese names to our ids, for every field a slot carries."""

    def __init__(self, dex) -> None:
        self.shown: dict[int, dict[str, str]] = {}
        #: Names accepted after a rearrangement, reported rather than hidden.
        self.normalised: list[tuple[str, str]] = []
        korean = json.loads((ROOT / "data/champions/names.json").read_text("utf-8"))
        self.species = self._flip(korean["species"])
        self.items = self._flip(korean["items"])
        self.abilities = self._flip(korean["abilities"])
        self.moves = self._flip(korean["moves"])
        self.natures = self._flip(korean["natures"])
        for field, table in (("species", self.species), ("items", self.items),
                             ("abilities", self.abilities), ("moves", self.moves),
                             ("natures", self.natures)):
            self.shown[id(table)] = {loose(name): name
                                     for name in korean[field].values()}
        self._add_japanese(dex)

    @staticmethod
    def _flip(table: dict) -> dict:
        out: dict[str, str] = {}
        for key, name in table.items():
            out.setdefault(loose(name), key)
        return out

    def _remember(self, table: dict, shown: str, landed: str) -> None:
        """Keep the name as written, so a miss can be answered with a guess."""
        self.shown.setdefault(id(table), {})[loose(shown)] = shown
        table.setdefault(loose(shown), landed)

    def _add_japanese(self, dex) -> None:
        raw = ROOT / "data/raw/pokechams"
        for stem, target, table in (("champions_pokemon", self.species, None),
                                    ("items", self.items, dex.items),
                                    ("abilities", self.abilities, dex.abilities),
                                    ("moves", self.moves, dex.moves)):
            path = raw / f"{stem}.json"
            if not path.exists():
                continue
            for entry in json.loads(path.read_text("utf-8")):
                japanese = loose(entry.get("nameJa") or "")
                if not japanese:
                    continue
                if table is None:
                    landed = self._species_from(dex, entry)
                else:
                    key = ident(entry.get("nameEn") or "")
                    landed = key if key in table else None
                if landed:
                    self._remember(target, entry["nameJa"], landed)
        for japanese, english in JAPANESE_NATURES.items():
            self.natures.setdefault(loose(japanese), english)

    @staticmethod
    def _species_from(dex, entry) -> str | None:
        """Their national dex number and form index as our species id."""
        number = entry.get("nationalDex")
        family = [s for s in dex.species.values() if s.dex_num == number]
        if not family:
            return None
        english = ident(entry.get("nameEn") or "")
        exact = next((s for s in family if ident(s.name) == english), None)
        if exact is not None:
            return exact.id
        if not entry.get("formIndex"):
            base = next((s for s in family if s.id == s.base_species), None)
            return base.id if base is not None else family[0].id
        # Their wording puts the forme in a parenthetical and reorders the
        # words -- "Charizard (Mega Charizard X)" for our "Charizard-Mega-X",
        # "Samurott (Hisuian Form)" for our "Samurott-Hisui". Match word by
        # word, by prefix so hisui reaches hisuian, and never fall back to the
        # base form: an unresolved Hisuian Samurott must refuse, not quietly
        # become the Unovan one.
        theirs = set(re.findall(r"[a-z0-9]+", (entry.get("nameEn") or "").lower()))
        fits = []
        for other in family:
            if other.id == other.base_species:
                continue
            ours = re.findall(r"[a-z0-9]+", other.name.lower())
            if all(any(word.startswith(token) or token.startswith(word)
                       for word in theirs) for token in ours):
                fits.append((len(ours), other.id))
        if not fits:
            return None
        best = max(count for count, _ in fits)
        winners = [name for count, name in fits if count == best]
        return winners[0] if len(winners) == 1 else None

    def look(self, table: dict, name: str, what: str) -> str:
        """Resolve, or refuse with the nearest names it could have been.

        A transcription miss is nearly always one syllable off -- 셰도볼 for
        섀도볼 -- and "unknown move" alone sends the reader back to the source
        to find out which. The suggestions are never applied; they are there so
        a person can confirm one in a second.
        """
        key = loose(name)
        if key in table:
            return table[key]
        display = self.shown.get(id(table), {})

        # Both sites write forms in an order ours does not: 대검귀(히스이) for
        # 히스이 대검귀, ダイケンキ(ヒスイ) for ヒスイダイケンキ. Once the
        # separators are gone these are the same characters in a different
        # order, so an *unambiguous* rearrangement is accepted -- and said out
        # loud, because a silent rename is how a wrong Pokemon gets in.
        signature = "".join(sorted(key))
        same = [k for k in display if "".join(sorted(k)) == signature]
        if len(same) == 1:
            self.normalised.append((name.strip(), display[same[0]]))
            return table[same[0]]
        inside = [k for k in display if key and key in k]
        if len(inside) == 1:
            self.normalised.append((name.strip(), display[inside[0]]))
            return table[inside[0]]

        near = difflib.get_close_matches(key, display, n=3, cutoff=0.6)
        hint = ""
        if near:
            hint = " -- did you mean " + " / ".join(display[k] for k in near) + "?"
        raise Complaint(f"unknown {what}: {name.strip()!r}{hint}")


def read_spread(line: str) -> tuple[list[int], list[str]]:
    """Six SP values, each optionally marked ``+`` or ``-`` by the nature."""
    cells = [one for one in re.split(r"[/\s]+", line.strip()) if one]
    if len(cells) != 6:
        raise Complaint(f"SP line has {len(cells)} entries, expected 6: {line.strip()!r}")
    sp, marks = [], []
    for cell in cells:
        # The mark lands on either side of the number depending on who typed
        # it -- ``0-`` and ``-0`` both mean a lowered stat with nothing in it.
        match = re.fullmatch(r"([+-]?)\s*(\d+)\s*([+-]?)", cell)
        if match is None:
            raise Complaint(f"cannot read SP entry {cell!r}")
        before, digits, after = match.groups()
        if before and after and before != after:
            raise Complaint(f"SP entry {cell!r} is marked both ways")
        sp.append(int(digits))
        marks.append(after or before)
    return sp, marks


def derive_nature(marks: list[str]) -> str | None:
    """The Stat Alignment its own marks name, or ``None`` if unmarked."""
    up = [Stat(index) for index, mark in enumerate(marks) if mark == "+"]
    down = [Stat(index) for index, mark in enumerate(marks) if mark == "-"]
    if not up and not down:
        return None
    if len(up) != 1 or len(down) != 1:
        raise Complaint(f"expected one + and one -, found {len(up)} and {len(down)}")
    if Stat.HP in (up[0], down[0]):
        raise Complaint("HP is never raised or lowered by a Stat Alignment")
    found = BY_SHIFT.get((up[0], down[0]))
    if found is None:
        raise Complaint(f"no Stat Alignment raises {up[0].name} and lowers "
                        f"{down[0].name}")
    return found


def parse_korean(block: list[str], tables: Tables) -> tuple[dict, list[int] | None]:
    if len(block) not in (5, 6):
        raise Complaint(f"expected 5 or 6 lines, found {len(block)}")
    nature = None
    if len(block) == 6:
        species, ability, nature_line, item, moves, spread = block
        nature = tables.look(tables.natures, nature_line, "nature")
    else:
        species, ability, item, moves, spread = block

    # ``위험예지 (메가진화 전: 위협)`` names what it plays as and what it
    # registered with. The roster wants the one it registered with.
    if "(" in ability and ":" in ability:
        ability = ability.split(":", 1)[1].rstrip(") ")

    sp, marks = read_spread(spread)
    shift = derive_nature(marks)
    if shift is not None:
        if nature is not None and nature != shift:
            raise Complaint(f"the nature line says {nature}, but the marks on "
                            f"the SP line say {shift}")
        nature = shift
    return {
        "pokemon": tables.look(tables.species, species, "species"),
        "ability": tables.look(tables.abilities, ability, "ability"),
        "item": (None if loose(item) in {loose(one) for one in GENERIC_STONE}
                 else tables.look(tables.items, item, "item")),
        "moves": [tables.look(tables.moves, one, "move")
                  for one in moves.split("/") if one.strip()],
        "nature": nature,
        "sp": sp,
    }, None


def parse_japanese(block: list[str], tables: Tables) -> tuple[dict, list[int]]:
    if len(block) != 5:
        raise Complaint(f"expected 5 lines, found {len(block)}")
    head, ability_line, nature_line, stat_line, move_line = block
    species, _, item = head.partition("@")
    cells = [c for c in stat_line.split("-") if c.strip()]
    if len(cells) != 6:
        raise Complaint(f"stat line has {len(cells)} cells, expected 6")
    sp, shown = [], []
    for cell in cells:
        match = re.fullmatch(r"\s*(\d+)\s*(?:\(\s*(\d*)\s*([+-]?)\s*\))?\s*", cell)
        if match is None:
            raise Complaint(f"cannot read stat cell {cell!r}")
        shown.append(int(match.group(1)))
        sp.append(int(match.group(2) or 0))
    return {
        "pokemon": tables.look(tables.species, species, "species"),
        "ability": tables.look(tables.abilities,
                               ability_line.split(":", 1)[-1], "ability"),
        "item": (None if loose(item) in {loose(one) for one in GENERIC_STONE}
                 else tables.look(tables.items, item, "item")),
        "moves": [tables.look(tables.moves, one, "move")
                  for one in move_line.split("/") if one.strip()],
        "nature": tables.look(tables.natures,
                              nature_line.split(":", 1)[-1], "nature"),
        "sp": sp,
    }, shown


def audit(dex, regulation, slot: dict, shown: list[int] | None) -> list[str]:
    """What the transcription got wrong, as far as the game's tables can say.

    Legality is not re-implemented here. ``set_errors`` is the one checker the
    engine and the importers already trust, and a second opinion written by
    hand is a second chance to be wrong -- the first draft of this function
    rolled its own learnset lookup and reported that Garchomp cannot learn
    Earthquake. What is left here is what ``set_errors`` cannot know: how these
    two sites *write* a Pokemon down.
    """
    said: list[str] = []
    written = dex.species[slot["pokemon"]]

    # A Mega is never registered: the team brings the base and the stone. Both
    # sites name the Mega, because that is what the battle shows.
    registered = dex.species[written.base_species] if written.is_mega else written
    # ...and the base of a Mega is not always the form the format fields.
    # Floette registers as Floette-Eternal here; plain Floette is not eligible,
    # and only the Eternal form has the Mega at all.
    if registered.id not in regulation.legal_species:
        fieldable = [dex.species[f] for f in registered.other_formes
                     if f in regulation.legal_species
                     and not dex.species[f].is_mega]
        if len(fieldable) == 1:
            registered = fieldable[0]
    slot["pokemon"] = registered.id
    # The Mega hangs off the family, not off the form that registers it:
    # Floette-Eternal's Mega is listed under plain Floette's formes.
    megas = [s for s in dex.species.values()
             if s.is_mega and s.base_species == registered.base_species]

    # ...and they name the ability it plays with, which for a Mega is the
    # Mega's. The roster wants the one it registered with.
    ability = slot["ability"]
    if ability not in registered.abilities:
        owner = next((m for m in megas if ability in m.abilities), None)
        if owner is not None:
            slot["ability"] = registered.abilities[0]
            said.append(f"{dex.abilities[ability].name} is {owner.name}'s; "
                        f"registered as {dex.abilities[slot['ability']].name}")

    stones = [m.required_item for m in megas if m.required_item]
    if slot["item"] is None and stones:
        # The site said "Mega Stone" and the species names which one.
        if len(set(stones)) == 1:
            slot["item"] = stones[0]
        else:
            said.append(f"ERROR {registered.name} has {len(set(stones))} Mega "
                        f"Stones; the text does not say which")
    elif slot["item"] is None:
        said.append("ERROR holds a Mega Stone, but this species has none")

    item = slot["item"]
    if item and dex.items[item].mega_stone:
        holders = dex.mega_stones.get(item, ())
        if not any(dex.species[m].base_species == registered.base_species
                   for m in holders):
            owner = dex.species[holders[0]].name if holders else "no species here"
            said.append(f"ERROR holds {dex.items[item].name}, "
                        f"which is {owner}'s stone")

    # The Japanese layout prints the stat its SP bought, so both can be checked
    # against each other. This is the only handle on a mistyped number: a wrong
    # SP is still a legal SP, and every other check would pass it.
    if shown is not None and slot.get("nature"):
        nature = get_nature(slot["nature"])
        for index, stat in enumerate(Stat):
            want = compute_stat(written.base_stats[index], slot["sp"][index],
                                nature, stat)
            if want != shown[index]:
                said.append(f"ERROR {STAT_ORDER[index]} reads {shown[index]}, "
                            f"but {slot['sp'][index]} SP on {written.name} "
                            f"({slot['nature']}) gives {want}")

    if len(slot["moves"]) != 4:
        said.append(f"{len(slot['moves'])} moves, expected 4")

    built = PokemonSet(species=registered.id, ability=slot["ability"],
                       moves=tuple(slot["moves"]), item=item,
                       nature=slot["nature"] or "serious", sp=tuple(slot["sp"]))
    said.extend(f"ERROR {line.split(': ', 1)[-1]}"
                for line in set_errors(dex, regulation, built))
    return said


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(
        allow_abbrev=False, description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--format", default="singles", choices=("singles", "doubles"))
    parser.add_argument("--json", type=Path, default=None,
                        help="write the parties that pass, in the importer's shape")
    args = parser.parse_args()

    dex = load_dex()
    regulation = dex.regulation("m_b")
    tables = Tables(dex)

    files = sorted(p for p in args.directory.glob("*.txt")
                   if re.fullmatch(r"\d+", p.stem))
    print(f"{len(files)} hand-entered files in {args.directory.as_posix()}\n")

    clean = 0
    parties: list[dict] = []
    natureless: list[str] = []
    for path in files:
        text = path.read_text("utf-8")
        japanese = "特性" in text
        blocks = [[line.strip() for line in chunk.splitlines() if line.strip()]
                  for chunk in re.split(r"\n\s*\n", text.strip()) if chunk.strip()]

        notes: list[str] = []
        built: list[dict] = []
        tables.normalised.clear()
        for index, block in enumerate(blocks, 1):
            label = block[0] if block else "(empty)"
            try:
                slot, shown = (parse_japanese(block, tables) if japanese
                               else parse_korean(block, tables))
            except Complaint as error:
                notes.append(f"  #{index} {label}: {error}")
                continue
            for line in audit(dex, regulation, slot, shown):
                notes.append(f"  #{index} {dex.species[slot['pokemon']].name}: {line}")
            built.append(slot)

        for written, canonical in tables.normalised:
            notes.append(f"  read {written!r} as {canonical!r}")
        if len(blocks) != 6:
            notes.append(f"  {len(blocks)} Pokemon, expected 6")
        if built and not any(one.get("nature") for one in built):
            natureless.append(path.stem)

        if len(built) == 6:
            team = tuple(PokemonSet(species=one["pokemon"], ability=one["ability"],
                                    moves=tuple(one["moves"]), item=one["item"],
                                    nature=one["nature"] or "serious",
                                    sp=tuple(one["sp"]))
                         for one in built)
            for line in team_errors(dex, regulation, team, args.format):
                notes.append(f"  team: {line}")
            parties.append({
                "id": path.stem, "title": f"pokedb {path.stem}",
                "team": [{"species": one.species, "ability": one.ability,
                          "moves": list(one.moves), "item": one.item,
                          "nature": one.nature, "sp": list(one.sp)}
                         for one in team],
            })

        if notes:
            print(path.name)
            for line in notes:
                print(line)
            print()
        else:
            clean += 1

    print(f"  {clean} clean, {len(files) - clean} with something to look at")
    if natureless:
        print(f"  {len(natureless)} carry no nature and were recorded neutral: "
              f"{', '.join(natureless)}")
    if args.json and parties:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(parties, ensure_ascii=False, indent=1),
                             encoding="utf-8")
        print(f"  {len(parties)} parties written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
