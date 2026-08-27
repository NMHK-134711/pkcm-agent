"""Read the pkmnchamps ranker-party archive into the engine's ``PokemonSet``.

Random teams are the null distribution, and measured, they are a long way from
the game: 37.7% of randomly generated Pokemon carry no same-type attack at all,
and items are dealt without reference to who holds them. Everything the agent
has been trained and measured on so far sits on that distribution.

``scripts/fetch_pkmnchamps.py`` caches the archive, and every party in it
carries ``showdown_slots`` -- English slugs for species, ability, item, nature
and moves, plus the SP spread. **Read that, not the rendered team sheets.** The
sheets are written for people: they are in Korean, they leave the ability off
entirely, and their species wording matches neither ``names.json`` nor the
포케챔스 dex. An earlier version of this script parsed them and spent all its
time on spelling.

    python scripts/fetch_pkmnchamps.py
    python scripts/import_parties.py --out data/champions/parties_m_b.json

Nothing here invents a value. A slot that does not resolve, or a team the
regulation refuses, is reported and dropped: a guessed ability changes what the
team is, and a team that quietly became legal is worse than one left out.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pkcm.data import champout  # noqa: E402
from pkcm.data.dex import Stat, load_dex  # noqa: E402
from pkcm.engine.legality import team_errors  # noqa: E402
from pkcm.engine.pokemon import PokemonSet  # noqa: E402
from pkcm.engine.stats import NATURES, SP_PER_STAT_CAP, SP_TOTAL  # noqa: E402

ARCHIVE = ROOT / "data" / "raw" / "pkmnchamps" / "archive_parties.json"

#: Their EV keys, in the engine's stat order.
EV_ORDER = ("hp", "atk", "def", "spa", "spd", "spe")


class SlotError(Exception):
    """A slot we will not guess at."""


def ident(value: str | None) -> str:
    """Their slug as our id: ``rough-skin`` -> ``roughskin``."""
    return "".join(c for c in (value or "").lower() if c.isalnum())


def resolve_species(dex, slug: str | None, fieldable: set[str]) -> str:
    """Their species slug as ours, and it has to be one this format fields.

    Their slugs carry form words ours drop -- ``aegislash-shield``,
    ``basculegion-male``, ``floette-eternal-mega`` -- so trailing segments come
    off one at a time until something resolves. First hit wins, which keeps
    ``floette-eternal-mega`` at ``floetteeternal`` rather than falling through
    to plain ``floette``.

    A slug that lands on a form the roster does not carry -- ``vivillon`` when
    the roster has ``vivillonfancy`` -- then resolves to the one fieldable form
    of that species, and only when there is exactly one. With two it is
    genuinely ambiguous and stays refused. docs/HANDOFF.md records Vivillon.

    A Mega is never registered: the team brings the base and the stone. So
    ``-mega`` comes off with the rest and the stone does the work.
    """
    parts = (slug or "").split("-")
    while parts:
        key = ident("-".join(parts))
        if key in fieldable:
            return key
        if key in dex.species:
            base = dex.species[key].base_species
            forms = [other for other in fieldable
                     if dex.species[other].base_species == base
                     and not dex.species[other].is_mega]
            if len(forms) == 1:
                return forms[0]
            break
        parts = parts[:-1]
    raise SlotError(f"species {slug!r} is not in this format")


#: Pairs the site's slug conversion confuses. Both are switch-and-hit moves and
#: Korean names them 유턴 and 퀵턴; the site's own name table has both right, and
#: only the English slug it derives comes out wrong.
#:
#: Evidence, over the 19 archive slots carrying ``u-turn``: the 17 on species
#: that learn U-turn are fine, and the 2 on species that cannot learn it --
#: 대쓰여너 and 메가아쿠스타 -- both learn Flip Turn. hk confirmed the party.
CONFUSABLE_MOVES = {"uturn": "flipturn", "flipturn": "uturn"}


def repair_move(rom, species: str, move_id: str) -> str:
    """Fix a move the species cannot learn, when the fix is not a guess.

    Only swaps within ``CONFUSABLE_MOVES``, only when the species cannot learn
    what was written, and only when it *can* learn the other one. Anything else
    is left alone to be refused: a party that quietly became legal is worse than
    one that was left out.
    """
    learnable = rom.learnset.get(species)
    if learnable is None or move_id in learnable:
        return move_id
    other = CONFUSABLE_MOVES.get(move_id)
    if other and other in learnable:
        return other
    return move_id


def form_for(rom, dex, species: str, fieldable: set[str], slot: dict) -> str:
    """Which form of this species the slot is actually describing.

    The archive drops the form: 히스이 대검귀 arrives as ``samurott`` carrying
    비검천중파 and 날카로운칼날, neither of which the original form has. The
    ROM's per-form learnset and ability table answer it, and **only when exactly
    one form fits** -- otherwise the slot stays as written and is refused.
    """
    base = dex.species[species].base_species
    forms = [other for other in fieldable
             if dex.species[other].base_species == base
             and not dex.species[other].is_mega and other in rom.learnset]
    if len(forms) < 2:
        return species

    ability = ident(slot.get("ability"))
    moves = [ident(name) for name in slot.get("moves") or ()]
    fits = [other for other in forms
            if (not ability or ability in rom.abilities.get(other, ()))
            and all(move in rom.learnset[other]
                    or CONFUSABLE_MOVES.get(move) in rom.learnset[other]
                    for move in moves)]
    return fits[0] if len(fits) == 1 else species


def mega_of(dex, regulation, species: str):
    """The one Mega this species may become here, or ``None``."""
    found = [key for key in regulation.legal_megas
             if dex.species[key].base_species == dex.species[species].base_species]
    return dex.species[found[0]] if len(found) == 1 else None


def parse_sp(evs: dict) -> tuple[int, ...]:
    spread = [0] * len(Stat)
    for index, key in enumerate(EV_ORDER):
        spread[index] = int(evs.get(key, 0) or 0)
    if sum(spread) > SP_TOTAL:
        raise SlotError(f"SP over budget: {sum(spread)} > {SP_TOTAL}")
    if any(value > SP_PER_STAT_CAP for value in spread):
        raise SlotError(f"SP over the per-stat cap of {SP_PER_STAT_CAP}")
    return tuple(spread)


def parse_slot(dex, regulation, slot: dict, fieldable: set[str],
               rom=None) -> PokemonSet:
    species = resolve_species(dex, slot.get("pokemon"), fieldable)
    if rom is not None:
        species = form_for(rom, dex, species, fieldable, slot)

    ability = ident(slot.get("ability"))
    if not ability:
        raise SlotError("ability missing")
    if ability not in dex.abilities:
        raise SlotError(f"unknown ability {slot.get('ability')!r}")
    # The archive lists what the slot *plays as*, which for a Mega is the
    # Mega's ability -- Shadow Tag on a Gengar that registers with Cursed Body.
    # A team registers the base form, and the Mega's ability comes with the
    # Mega, so it is put back where the roster expects it.
    if ability not in dex.species[species].abilities:
        mega = mega_of(dex, regulation, species)
        if mega is not None and ability in mega.abilities:
            ability = dex.species[species].abilities[0]

    item = ident(slot.get("item")) or None
    if item and item not in dex.items:
        # Their spelling of a Champions-original Mega Stone: ``starmienite``
        # against our ``starminite``, ``glimmorite`` against ``glimmoranite``.
        # Nothing is being guessed -- the species has exactly one Mega here and
        # that Mega names the only stone it can hold.
        mega = mega_of(dex, regulation, species)
        if mega is not None and mega.required_item and item.endswith("ite"):
            item = mega.required_item
        else:
            raise SlotError(f"unknown item {slot.get('item')!r}")

    nature = ident(slot.get("nature")) or "serious"
    if nature not in NATURES:
        raise SlotError(f"unknown nature {slot.get('nature')!r}")

    moves = []
    for name in slot.get("moves") or ():
        move_id = ident(name)
        if move_id not in dex.moves:
            raise SlotError(f"unknown move {name!r}")
        if rom is not None:
            move_id = repair_move(rom, species, move_id)
        moves.append(move_id)
    if not moves:
        raise SlotError("no moves")

    return PokemonSet(species=species, ability=ability, moves=tuple(moves),
                      item=item, nature=nature, sp=parse_sp(slot.get("evs") or {}))


def as_dict(pokemon: PokemonSet) -> dict:
    if is_dataclass(pokemon):
        return asdict(pokemon)
    return {name: getattr(pokemon, name) for name in pokemon.__slots__}


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):  # pragma: no cover
            pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--format", default="singles", choices=("singles", "doubles"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not args.archive.exists():
        print(f"no archive at {args.archive} -- run scripts/fetch_pkmnchamps.py")
        return 1

    dex = load_dex()
    regulation = dex.regulation("m_b")
    fieldable = set(regulation.legal_species)
    archive = json.loads(args.archive.read_text(encoding="utf-8"))
    try:
        rom = champout.load(dex, fieldable | set(regulation.legal_megas))
    except champout.MissingDump as error:
        print(f"  ({error}; forms and move slugs will not be repaired)")
        rom = None

    wanted = {"singles": {"single", "singles"},
              "doubles": {"double", "doubles"}}[args.format]

    teams: list[dict] = []
    refused: Counter = Counter()
    illegal: Counter = Counter()
    skipped: Counter = Counter()

    for party in archive:
        if (party.get("battle_format") or "").lower() not in wanted:
            skipped["a different format"] += 1
            continue
        slots = party.get("showdown_slots")
        if not slots:
            skipped["no showdown_slots"] += 1
            continue

        built = []
        for slot in slots:
            try:
                built.append(parse_slot(dex, regulation, slot, fieldable, rom))
            except SlotError as error:
                refused[str(error)] += 1
                break
        else:
            complaints = team_errors(dex, regulation, tuple(built), args.format)
            if complaints:
                for line in complaints:
                    illegal[line[:70]] += 1
                continue
            teams.append({
                "id": party.get("id"),
                "title": party.get("title"),
                "rate": party.get("rate"),
                "rank": party.get("rank"),
                "team": [as_dict(pokemon) for pokemon in built],
            })

    print(f"{len(archive)} parties in the archive")
    for reason, count in skipped.most_common():
        print(f"  {count:4} skipped -- {reason}")
    print(f"  {len(teams):4} imported and legal in {args.format}")

    if refused:
        print(f"\n{sum(refused.values())} teams refused on a slot:")
        for reason, count in refused.most_common(12):
            print(f"  x{count:<3} {reason}")
    if illegal:
        print(f"\n{sum(illegal.values())} legality complaints on parsed teams:")
        for reason, count in illegal.most_common(12):
            print(f"  x{count:<3} {reason}")

    rated = sorted(team["rate"] for team in teams if team.get("rate"))
    if rated:
        print(f"\nladder rating: {rated[0]}-{rated[-1]}, "
              f"median {rated[len(rated) // 2]}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(teams, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        print(f"\nwrote {len(teams)} teams -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
