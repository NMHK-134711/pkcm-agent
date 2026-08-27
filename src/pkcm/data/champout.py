"""The Switch ROM's own tables, lined up with our ids.

``scripts/fetch_champout.py`` caches the dumps from projectpokemon/champout.
They are the game's tables, which puts them above every other source here:
Showdown's champions mod is a mechanics reference written by people reading the
game, and the 포케챔스 dex and pkmnchamps archive are sites reporting it.

The join is the awkward part. The ROM keys species as ``0503001`` -- national
number, then form -- and names nothing in English, so there is no id to match
on. What it does carry is base stats, types and the national number, and those
identify a form: **matching on the numbers is a check, where matching on a
name-mangling rule would be a guess.** The three that stay ambiguous after that
are separated by gender, which the ROM does record.

Moves and items join on their national numbers, which our dex already carries.

Everything here is read-only and lazy: nothing loads until something asks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from pkcm.data.dex import Dex, Stat

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "champout"

#: The ROM's type numbering, in its own order. Ours are Showdown's lowercase
#: names, so this is the translation between them.
ROM_TYPES = (
    "normal", "fighting", "flying", "poison", "ground", "rock", "bug", "ghost",
    "steel", "fire", "water", "grass", "electric", "psychic", "ice", "dragon",
    "dark", "fairy",
)

_STAT_KEYS = ("hp", "atk", "def", "spatk", "spdef", "agi")


class MissingDump(FileNotFoundError):
    """The cache is not there. Loud, because silently skipping the ROM would
    let a worse source win by default."""


@dataclass(frozen=True)
class Champout:
    """The ROM tables, keyed by our ids where a join exists."""

    #: our species id -> the ROM's ``id`` string (``0503001``)
    species: dict[str, str]
    #: our species id -> the move ids it can learn, **filtered to the moves the
    #: ROM marks available**.
    #:
    #: ``waza_learn.json`` is the raw table and carries rows for moves that ship
    #: disabled: 561 distinct moves appear across it, against 497 flagged
    #: ``available``. Taking it unfiltered hands Tsareena Magical Leaf and Rotom
    #: Thunder Shock, neither of which is in the game -- and the 포케챔스 dex
    #: independently counts 497 too.
    learnset: dict[str, frozenset[str]]
    #: our species id -> its ability ids, in the ROM's slot order
    abilities: dict[str, tuple[str, ...]]
    #: move id -> whether the ROM marks it available in Champions
    available: dict[str, bool]
    #: ROM rows nothing in our dex matched, for reporting
    unmatched: tuple[str, ...] = field(default=())
    #: The learnset rows before the availability filter, for comparing the two.
    learnset_raw: dict[str, frozenset[str]] = field(default_factory=dict)


def _read(name: str):
    path = RAW_DIR / name
    if not path.exists():
        raise MissingDump(
            f"{path} is not there -- run scripts/fetch_champout.py")
    return json.loads(path.read_text(encoding="utf-8"))


def _rom_key(entry: dict) -> tuple[int, tuple[int, ...], tuple[str, ...]]:
    types = tuple(sorted({ROM_TYPES[int(entry["type1"])],
                          ROM_TYPES[int(entry["type2"])]}))
    return (int(entry["no"]),
            tuple(int(entry[key]) for key in _STAT_KEYS),
            types)


def _our_key(dex: Dex, species_id: str) -> tuple[int, tuple[int, ...], tuple[str, ...]]:
    entry = dex.species[species_id]
    return (entry.dex_num,
            tuple(entry.base_stats[stat] for stat in Stat),
            tuple(sorted(set(entry.types))))


@lru_cache(maxsize=1)
def _load(dex_id: int) -> Champout:  # pragma: no cover - keyed by id(dex)
    raise RuntimeError("call load() instead")


def load(dex: Dex, fieldable: set[str] | None = None) -> Champout:
    """Join the ROM tables onto our ids. Raises ``MissingDump`` without a cache.

    ``fieldable`` breaks the ties that stats cannot: a Gmax forme has its base
    forme's numbers exactly, so ``charizard`` and ``charizardgmax`` are the same
    row to this join. Only one of them is in the format, and that is the one the
    ROM row is about.
    """
    personal = _read("personal.json")
    learn = {row["id"]: row["waza"] for row in _read("waza_learn.json")}
    waza = _read("waza.json")

    by_number: dict[int, str] = {}
    for move in dex.moves.values():
        by_number.setdefault(move.num, move.id)

    ours: dict[tuple, list[str]] = {}
    for species_id in dex.species:
        ours.setdefault(_our_key(dex, species_id), []).append(species_id)

    species: dict[str, str] = {}
    abilities: dict[str, tuple[str, ...]] = {}
    learnset: dict[str, frozenset[str]] = {}
    unmatched: list[str] = []

    ability_by_number = _ability_numbers(dex)

    for entry in personal:
        if entry.get("is_valid") != "1":
            continue
        found = ours.get(_rom_key(entry), [])
        if len(found) > 1 and fieldable:
            narrowed = [s for s in found if s in fieldable]
            if narrowed:
                found = narrowed
        if len(found) > 1:
            # Same number, stats and types: the ROM separates these by gender
            # and so must we. ``sex`` is 0 male-only, 254 female-only here.
            wanted = {"0": "M", "254": "F"}.get(entry.get("sex", ""))
            if wanted:
                narrowed = [s for s in found if dex.species[s].gender == wanted]
                if narrowed:
                    found = narrowed
        if len(found) != 1:
            unmatched.append(entry["id"])
            continue

        species_id = found[0]
        species[species_id] = entry["id"]
        # The ROM fills both regular slots even when a species has one ability,
        # so ``toku0`` and ``toku1`` repeat. Order is kept; repeats are not.
        seen: list[str] = []
        for slot in ("toku0", "toku1", "toku2"):
            number = entry.get(slot)
            found_ability = ability_by_number.get(int(number)) if number else None
            if found_ability and found_ability not in seen:
                seen.append(found_ability)
        abilities[species_id] = tuple(seen)
        numbers = (learn.get(entry["id"]) or "").split(",")
        learnset[species_id] = frozenset(
            by_number[int(number)] for number in numbers
            if number and int(number) in by_number)

    available = {}
    for row in waza:
        move_id = by_number.get(int(row["id"]))
        if move_id is not None:
            available[move_id] = row.get("available") == "1"

    enabled = {move_id for move_id, yes in available.items() if yes}
    return Champout(species=species,
                    learnset={key: value & enabled
                              for key, value in learnset.items()},
                    learnset_raw=learnset,
                    abilities=abilities, available=available,
                    unmatched=tuple(unmatched))


def _ability_numbers(dex: Dex) -> dict[int, str]:
    """The ROM numbers abilities; our dex knows the same numbers."""
    found: dict[int, str] = {}
    for ability_id, ability in dex.abilities.items():
        number = getattr(ability, "num", None)
        if number is None:
            number = (getattr(ability, "raw", {}) or {}).get("num")
        if number is not None:
            found.setdefault(int(number), ability_id)
    return found
