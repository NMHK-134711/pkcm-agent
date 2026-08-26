"""Typed, read-only view over the game data.

Two layers stack here:

1. ``data/raw/`` - Pokemon Showdown's client data, normalized to JSON by
   ``scripts/fetch_showdown_data.py``. Showdown already carries Champions'
   new Mega Evolutions and abilities, so this covers the numbers.
2. ``data/champions/`` - our curated layer: which species and Megas a
   regulation actually allows, the ruleset constants, and **the Champions
   override layer**.

The override layer is not optional. Showdown's base data is Scarlet/Violet;
Champions removes 194 of the moves it calls standard, re-enables 9 more, changes
the numbers on ~30, and caps every move's base PP at 20. Reading the base data
alone gives a move pool that is 28% moves the game does not have.

Everything here is frozen and cached: the engine treats the dex as a constant
and never mutates it, which is what keeps battle states cheap to clone.

Mechanics are deliberately *not* modeled yet. Each ``Move``/``Ability``/``Item``
keeps its untouched Showdown dict in ``.raw``; fields graduate into typed
attributes as the engine grows to actually use them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import IntEnum
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
CHAMPIONS_DIR = ROOT / "data" / "champions"


class Stat(IntEnum):
    """Index into a stat tuple. Order matches Showdown's ``baseStats``."""

    HP = 0
    ATK = 1
    DEF = 2
    SPA = 3
    SPD = 4
    SPE = 5


STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")

#: The 18 battle types, lowercased ids. Showdown's chart also lists "stellar",
#: which only matters for Terastallization and so is out of scope here.
TYPES = (
    "normal", "fire", "water", "electric", "grass", "ice",
    "fighting", "poison", "ground", "flying", "psychic", "bug",
    "rock", "ghost", "dragon", "dark", "steel", "fairy",
)

#: Showdown's damageTaken encoding: 0 neutral, 1 weak, 2 resist, 3 immune.
_DAMAGE_TAKEN = {0: 1.0, 1: 2.0, 2: 0.5, 3: 0.0}

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def to_id(text: str) -> str:
    """Showdown's ``toID``: lowercase and strip everything but ``[a-z0-9]``."""
    return _NON_ALNUM.sub("", text.lower())


@dataclass(frozen=True, slots=True)
class Species:
    id: str
    name: str
    dex_num: int
    types: tuple[str, ...]
    base_stats: tuple[int, int, int, int, int, int]
    abilities: tuple[str, ...]
    weight_kg: float
    base_species: str
    forme: str
    required_item: str | None
    other_formes: tuple[str, ...]
    #: "M", "F" or "N". Fixed-gender species declare it; the rest are decided
    #: per Pokemon, so a set carries its own.
    gender: str | None
    #: Set when this forme is *derived* from another at the same base species
    #: (Rotom's appliances, Necrozma's fusions). Showdown stores only the
    #: signature move on such formes; the rest is inherited from the source.
    changes_from: str | None
    raw: dict[str, Any] = field(repr=False, compare=False, default_factory=dict)

    @property
    def is_mega(self) -> bool:
        return self.forme.startswith("Mega")


@dataclass(frozen=True, slots=True)
class Move:
    id: str
    name: str
    num: int
    type: str
    category: str  # "Physical" | "Special" | "Status"
    base_power: int
    accuracy: int | None  # None means the move cannot miss
    pp: int
    priority: int
    target: str
    flags: frozenset[str]
    raw: dict[str, Any] = field(repr=False, compare=False, default_factory=dict)

    @property
    def is_status(self) -> bool:
        return self.category == "Status"


@dataclass(frozen=True, slots=True)
class Ability:
    id: str
    name: str
    num: int
    raw: dict[str, Any] = field(repr=False, compare=False, default_factory=dict)


@dataclass(frozen=True, slots=True)
class Item:
    id: str
    name: str
    num: int
    #: ``(base species id, mega species id)`` pairs if this is a Mega Stone.
    #: A stone may serve several base formes (Magearna and Magearna-Original
    #: share Magearnite), hence a tuple rather than a single pair.
    mega_stone: tuple[tuple[str, str], ...]
    raw: dict[str, Any] = field(repr=False, compare=False, default_factory=dict)


class TypeChart:
    """Attacking-type against defending-type multipliers."""

    __slots__ = ("_matrix",)

    def __init__(self, raw: dict[str, Any]) -> None:
        matrix: dict[str, dict[str, float]] = {}
        for attacking in TYPES:
            row = {}
            for defending in TYPES:
                taken = raw[defending]["damageTaken"][attacking.capitalize()]
                row[defending] = _DAMAGE_TAKEN[taken]
            matrix[attacking] = row
        self._matrix = matrix

    def multiplier(self, attacking: str, defending: tuple[str, ...]) -> float:
        row = self._matrix[attacking]
        result = 1.0
        for defending_type in defending:
            result *= row[defending_type]
        return result


@dataclass(frozen=True, slots=True)
class Regulation:
    """Which species and Megas a regulation allows, plus its rule constants."""

    name: str
    legal_species: frozenset[str]
    legal_megas: frozenset[str]
    #: Mega Stone item id -> mega species ids, restricted to legal Megas.
    legal_mega_stones: dict[str, tuple[str, ...]]
    rules: dict[str, Any]

    @property
    def team_size(self) -> int:
        return self.rules["confirmed"]["team_size"]

    @property
    def level(self) -> int:
        return self.rules["confirmed"]["level_rule"]

    def bring_select(self, battle_format: str) -> tuple[int, int]:
        """``(team size, number brought into battle)`` for ``singles``/``doubles``."""
        for section in ("confirmed", "assumed"):
            if battle_format in self.rules[section]:
                entry = self.rules[section][battle_format]
                return entry["bring"], entry["select"]
        raise KeyError(f"unknown battle format {battle_format!r}")


#: Champions caps every move's base PP at 20 (mods/champions/scripts.ts, init()).
CHAMPIONS_MAX_BASE_PP = 20


def _apply_overrides(raw: dict[str, Any], changes: dict[str, dict[str, Any]]) -> int:
    """Fold Champions' deltas into the base data. Returns how many entries changed.

    ``isNonstandard: null`` in the mod means "Champions re-enables this", so the
    marker is removed rather than set. ``"Past"`` means the entry does not exist
    in Champions at all -- 194 of the moves the base data calls standard.
    """
    applied = 0
    for key, fields in changes.items():
        entry = raw.get(key)
        if entry is None:
            continue
        for field, value in fields.items():
            if value is None:
                entry.pop(field, None)
            else:
                entry[field] = value
        applied += 1
    return applied


def _load_overrides() -> dict[str, Any]:
    path = CHAMPIONS_DIR / "overrides.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run: python scripts/build_champions_overrides.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _load_raw(stem: str) -> dict[str, Any]:
    path = RAW_DIR / f"{stem}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run: python scripts/fetch_showdown_data.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


class Dex:
    """Immutable bundle of every lookup table the engine needs."""

    def __init__(self) -> None:
        overrides = _load_overrides()

        pokedex = _load_raw("pokedex")
        self.species: dict[str, Species] = {
            key: _build_species(key, value, pokedex) for key, value in pokedex.items()
        }

        raw_moves = _load_raw("moves")
        self.override_counts = {
            "moves": _apply_overrides(raw_moves, overrides["moves"]["changes"]),
        }
        for entry in raw_moves.values():
            entry["pp"] = min(entry["pp"], CHAMPIONS_MAX_BASE_PP)
        self.moves: dict[str, Move] = {
            key: _build_move(key, value) for key, value in raw_moves.items()
        }

        raw_abilities = _load_raw("abilities")
        self.override_counts["abilities"] = _apply_overrides(
            raw_abilities, overrides["abilities"]["changes"]
        )
        self.abilities: dict[str, Ability] = {
            key: Ability(key, value.get("name", key), value.get("num", -1), value)
            for key, value in raw_abilities.items()
        }

        raw_items = _load_raw("items")
        self.override_counts["items"] = _apply_overrides(raw_items, overrides["items"]["changes"])
        self.items: dict[str, Item] = {
            key: _build_item(key, value) for key, value in raw_items.items()
        }

        self.type_chart = TypeChart(_load_raw("typechart"))

    def exists_in_champions(self, entry) -> bool:
        """Whether a move / item / ability is part of Champions at all."""
        return entry.raw.get("isNonstandard") is None

    @cached_property
    def learnsets(self) -> dict[str, Any]:
        """Lazy: 3 MB that only team building needs, never the battle loop."""
        return _load_raw("learnsets")

    @cached_property
    def mega_stones(self) -> dict[str, tuple[str, ...]]:
        """Every Mega Stone item id -> the mega species ids it can produce."""
        return {
            item.id: tuple(dict.fromkeys(mega for _, mega in item.mega_stone))
            for item in self.items.values()
            if item.mega_stone
        }

    @cached_property
    def _mega_lookup(self) -> dict[tuple[str, str], str]:
        return {
            (base, item.id): mega
            for item in self.items.values()
            for base, mega in item.mega_stone
        }

    def mega_evolution(self, species_id: str, item_id: str | None) -> str | None:
        """The mega forme ``species_id`` reaches while holding ``item_id``, if any."""
        if item_id is None:
            return None
        return self._mega_lookup.get((species_id, item_id))

    def regulation(self, slug: str = "m_b") -> Regulation:
        legality_path = CHAMPIONS_DIR / f"regulation_{slug}.json"
        rules_path = CHAMPIONS_DIR / f"ruleset_{slug}.json"
        if not legality_path.exists():
            raise FileNotFoundError(
                f"{legality_path} is missing. Run: python scripts/build_champions_data.py"
            )
        legality = json.loads(legality_path.read_text(encoding="utf-8"))
        rules = json.loads(rules_path.read_text(encoding="utf-8"))

        legal_species = frozenset(i for e in legality["eligible"] for i in e["showdown_ids"])
        legal_megas = frozenset(i for e in legality["megas"] for i in e["showdown_ids"])
        legal_mega_stones = {
            item_id: legal
            for item_id, mega_ids in self.mega_stones.items()
            if (legal := tuple(m for m in mega_ids if m in legal_megas))
        }
        return Regulation(
            name=legality["regulation"],
            legal_species=legal_species,
            legal_megas=legal_megas,
            legal_mega_stones=legal_mega_stones,
            rules=rules,
        )


#: Battle-relevant fields a cosmetic forme (Alcremie's swirls, Vivillon's
#: patterns, ...) omits because it is mechanically identical to its base forme.
_INHERITED_FIELDS = ("num", "baseStats", "types", "abilities", "weightkg", "heightm")


def _build_species(key: str, raw: dict[str, Any], pokedex: dict[str, Any]) -> Species:
    if "baseStats" not in raw:
        base = pokedex.get(to_id(raw.get("baseSpecies", "")), {})
        raw = {**raw, **{f: base[f] for f in _INHERITED_FIELDS if f in base}}

    stats = raw["baseStats"]
    abilities = tuple(
        to_id(raw["abilities"][slot])
        for slot in ("0", "1", "H", "S")
        if slot in raw["abilities"]
    )
    required_item = raw.get("requiredItem")
    changes_from = raw.get("changesFrom")
    gender = raw.get("gender")
    return Species(
        id=key,
        name=raw["name"],
        dex_num=raw["num"],
        types=tuple(t.lower() for t in raw["types"]),
        base_stats=tuple(stats[k] for k in STAT_KEYS),  # type: ignore[arg-type]
        abilities=abilities,
        weight_kg=float(raw.get("weightkg", 0.0)),
        base_species=to_id(raw.get("baseSpecies", raw["name"])),
        forme=raw.get("forme", ""),
        required_item=to_id(required_item) if required_item else None,
        other_formes=tuple(to_id(f) for f in raw.get("otherFormes", ())),
        changes_from=to_id(changes_from) if changes_from else None,
        gender=gender,
        raw=raw,
    )


def _build_move(key: str, raw: dict[str, Any]) -> Move:
    accuracy = raw["accuracy"]
    return Move(
        id=key,
        name=raw["name"],
        num=raw.get("num", -1),
        type=raw["type"].lower(),
        category=raw["category"],
        base_power=raw["basePower"],
        accuracy=None if accuracy is True else int(accuracy),
        pp=raw["pp"],
        priority=raw.get("priority", 0),
        target=raw["target"],
        flags=frozenset(raw.get("flags", {})),
        raw=raw,
    )


def _build_item(key: str, raw: dict[str, Any]) -> Item:
    mega_stone = tuple(
        (to_id(base), to_id(mega)) for base, mega in raw.get("megaStone", {}).items()
    )
    return Item(
        id=key,
        name=raw.get("name", key),
        num=raw.get("num", -1),
        mega_stone=mega_stone,
        raw=raw,
    )


@lru_cache(maxsize=1)
def load_dex() -> Dex:
    """Process-wide cached dex. Safe to call from anywhere."""
    return Dex()
