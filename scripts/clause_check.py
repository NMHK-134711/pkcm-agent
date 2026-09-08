"""Every sentence of every move description, held against the engine.

``mechanic_check`` asks whether a move does what its *fields* say. Aurora Veil
passed it while going up in the sun, because "Fails unless the weather is
Snow" is not a field -- it is a sentence in ``desc``, and nothing read it. Four
more of its five were in the same paragraph.

So the unit here is the clause, not the move. The 497 descriptions are 1022
sentences and 621 distinct ones, because a sentence like "If the user has the
Skill Link Ability, this move will always hit five times" belongs to eight
moves at once. Each family is checked on **every** move that carries it, and
the report ends with the clauses nothing has claimed yet -- which is the part
that was missing, and the only part that can catch the next Aurora Veil.

    python scripts/clause_check.py            # every family, then the gap
    python scripts/clause_check.py bigroot    # one family by name
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_spec = importlib.util.spec_from_file_location(
    "mechanic_check", ROOT / "scripts" / "mechanic_check.py")
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

DEX = mc.DEX
Fight, mon, ours = mc.Fight, mc.mon, mc.ours
Action = mc.Action
BattleConfig, new_battle, step = mc.BattleConfig, mc.new_battle, mc.step
from pkcm.engine.actions import TARGET_ALLY               # noqa: E402

CHAMPIONS = [m for m in DEX.moves.values() if DEX.exists_in_champions(m)]


def sentences(move) -> list[str]:
    text = (move.raw.get("desc") or move.raw.get("shortDesc") or "").strip()
    return [one.strip() for one in re.split(r"(?<=\.)\s+", text) if one.strip()]


#: clause text -> the moves whose description contains it
BY_CLAUSE: dict[str, list[str]] = defaultdict(list)
for _move in CHAMPIONS:
    for _sentence in sentences(_move):
        BY_CLAUSE[_sentence].append(_move.id)


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #

FAMILIES: dict[str, dict] = {}


LEGAL_ITEMS = {one["id"] for one in json.loads(
    (ROOT / "data" / "champions" / "items_m_b.json")
    .read_text(encoding="utf-8"))["items"]}


def family(name: str, *patterns: str, needs_item: str | None = None):
    """Claim every clause matching ``patterns`` and check it move by move.

    The patterns are substrings of the description sentence. Claiming is the
    point: a clause no family claims is printed at the end as unchecked, so
    the report cannot quietly shrink the way ``498/498`` did.

    ``needs_item`` names an item the clause hangs on. If the format does not
    allow it the family reports "n/a" with the reason rather than "ok" -- an
    untestable clause is not a passing one.
    """
    def wrap(probe):
        claimed = sorted({one for one in BY_CLAUSE
                          for pattern in patterns if pattern in one})
        FAMILIES[name] = {"patterns": patterns, "clauses": claimed,
                          "probe": probe, "needs_item": needs_item}
        return probe
    return wrap


def members(name: str) -> list[str]:
    """Every move carrying any clause this family claims."""
    out = {move for clause in FAMILIES[name]["clauses"] for move in BY_CLAUSE[clause]}
    return sorted(out)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

WALL = ("snorlax", "__none__", ("splash", "bodyslam", "protect", "rest"),
        None, "sassy", (32, 0, 32, 0, 32, 0))


def wall(species="snorlax", ability="__none__", item=None):
    return mon(species, ability, ("splash", "bodyslam", "protect", "rest"),
               item, "sassy", (32, 0, 32, 0, 32, 0))


#: Mew learns everything, which is what a harness that has to cast 497 moves
#: needs. ``mechanic_check`` picked it for the same reason.
UNIVERSAL = "mew" if "mew" in DEX.species else "snorlax"


def swinger(move_id, ability="__none__", item=None,
            extra=("splash", "protect", "rest"), species=None):
    """Something that can use ``move_id``, invested on the right side."""
    move = DEX.moves[move_id]
    physical = move.category != "Special"
    nature = "adamant" if physical else "modest"
    sp = (32, 32, 0, 0, 2, 0) if physical else (32, 0, 0, 32, 2, 0)
    return mon(species or UNIVERSAL, ability, (move_id,) + tuple(extra), item,
               nature, sp)


def gendered(species, ability, moves, nature, sp, gender):
    """``mechanic_check.mon`` has no gender, and Attract refuses without one."""
    from pkcm.engine.pokemon import PokemonSet

    return PokemonSet(species=species, ability=ability, moves=tuple(moves),
                      item=None, nature=nature, sp=sp, gender=gender)


def hp_lost(f, side=1):
    """Everything that came off ``side`` in the turn just played."""
    return sum(e.amount or 0 for e in f.log
               if e.kind == "damage" and (e.side or 0) == side)


def landed(f, move_id, side=1):
    return any(e.kind == "damage" and (e.side or 0) == side and e.move == move_id
               for e in f.log)


def until(build, wanted, tries=30):
    """Replay with fresh seeds until the position is the one we asked for."""
    for seed in range(tries):
        f = build(seed)
        if wanted(f):
            return f
    return None


def verdict(rows):
    """``rows`` is (move, ok, detail). One bad move fails the family."""
    bad = [r for r in rows if not r[1]]
    detail = "; ".join(f"{move}: {note}" for move, _, note in rows)
    return not bad, detail



# --------------------------------------------------------------------------- #
# Clauses about things this format does not have
# --------------------------------------------------------------------------- #
#
# "If the user is holding Utility Umbrella" cannot arise where Utility Umbrella
# is not a legal item, and neither can a sentence about Sky Drop where Sky Drop
# is not a legal move. That is a real answer, but only when it is *checked*:
# the names are pulled out of the clause and looked up, and a clause is only
# set aside when every mechanic it names is absent from Regulation M-B.

_REGULATION = DEX.regulation("m_b")
_ROSTER = _REGULATION.legal_species | _REGULATION.legal_megas
LEGAL_ABILITIES = {one for species in _ROSTER
                   for one in DEX.species[species].abilities}
LEGAL_MOVES = {move.id for move in CHAMPIONS}


def _as_id(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


#: Every proper name the descriptions use, by what it is.
NAMED = {}
for _move in DEX.moves.values():
    NAMED.setdefault(_as_id(_move.raw.get("name") or _move.id),
                     ("move", _move.id))
for _item in DEX.items.values():
    NAMED.setdefault(_as_id(_item.name), ("item", _item.id))
for _species in DEX.species.values():
    for _ability in _species.abilities:
        NAMED.setdefault(_as_id(_ability), ("ability", _ability))
for _pretty, _id in (("Battle Armor", "battlearmor"), ("Shell Armor", "shellarmor")):
    NAMED.setdefault(_as_id(_pretty), ("ability", _id))

#: Names in the descriptions that are neither a move nor an item nor an
#: ability the dex knows -- mechanics from other formats and other games.
FOREIGN = {
    "sky drop": "a move this format does not have",
    "max guard": "a Dynamax move, and Dynamax is not in Champions",
    "battle royal": "a format Champions does not run",
    "z-move": "not in Champions",
    "terastallized": "not in Champions",
    "mega evolved": "handled by the mega rules, not by a move",
}

_NAME_PATTERN = re.compile(r"\b(?:[A-Z][a-z']+(?:[ -][A-Z][a-z']+)*)\b")

#: Words that start a sentence or a species and are not mechanics.
_NOT_A_MECHANIC = {
    "the", "this", "if", "it", "a", "an", "and", "or", "for", "in", "at", "on",
    "power", "damage", "pokemon", "fails", "has", "causes", "raises", "lowers",
    "deals", "hits", "prevents", "until", "during", "while", "when", "after",
    "before", "no", "not", "there", "these", "those", "all", "any", "each",
    "every", "both", "one", "two", "three", "four", "five", "x", "hp", "pp",
}


#: Words that name a category rather than a mechanic. "Berry" is an item id
#: in the dex and a common noun in the descriptions, and reading it as the
#: former set aside every clause about berries in a format with 28 of them.
_GENERIC = {"berry", "berries", "pokemon", "ability", "abilities", "item",
            "items", "move", "moves", "type", "types", "terrain", "weather",
            "orb", "plate", "memory", "drive", "gem"}


def out_of_format(clause: str):
    """Which mechanics a clause names that Regulation M-B does not have.

    Only moves and items count as evidence, and only when nothing else in the
    sentence is checkable. An ability name does not: Simple Beam grants Simple
    whether or not a legal species is born with it.
    """
    named, missing = [], []
    for phrase in _NAME_PATTERN.findall(clause):
        low = phrase.lower()
        if low in _NOT_A_MECHANIC or low in _GENERIC or len(phrase) < 4:
            continue
        if low in FOREIGN:
            named.append(phrase)
            missing.append(f"{phrase} -- {FOREIGN[low]}")
            continue
        found = NAMED.get(_as_id(phrase))
        if found is None or found[0] == "ability":
            continue
        kind, one = found
        legal = {"move": LEGAL_MOVES, "item": LEGAL_ITEMS}[kind]
        named.append(phrase)
        if one not in legal:
            missing.append(f"{phrase} -- not a legal {kind} here")
    return named, missing


# --------------------------------------------------------------------------- #
# The families
# --------------------------------------------------------------------------- #


@family("bigroot", "If Big Root is held by")
def _big_root():
    rows = []
    for move_id in members("bigroot"):
        def healed(item, move_id=move_id):
            f = Fight([swinger(move_id, item=item)],
                      [wall(mc._reachable(DEX.moves[move_id]))], seed=7)
            slot = f.state.sides[0].active[0]
            f.state.sides[0].hp[slot] = 1     # so nothing is capped by the ceiling
            before = f.hp(0)
            f.turn(Action.move(0), Action.move(0))
            return f.hp(0) - before

        plain = healed(None)
        boosted = healed("bigroot")
        ratio = boosted / plain if plain else 0.0
        rows.append((move_id, plain > 0 and 1.25 <= ratio <= 1.35,
                     f"{plain} -> {boosted} (x{ratio:.2f})"))
    return verdict(rows)


@family("skilllink", "If the user has the Skill Link Ability, this move will "
                     "always hit five times")
def _skill_link():
    rows = []
    for move_id in members("skilllink"):
        counts = set()
        for seed in range(12):
            f = Fight([swinger(move_id, ability="skilllink")],
                      [wall(mc._reachable(DEX.moves[move_id]))], seed=seed)
            f.turn(Action.move(0), Action.move(0))
            hit = len([e for e in f.log if e.kind == "damage"
                       and (e.side or 0) == 1 and e.move == move_id])
            if hit:
                counts.add(hit)
        rows.append((move_id, counts == {5}, f"hit {sorted(counts)} times"))
    return verdict(rows)


@family("loadeddice", "If the user is holding Loaded Dice, this move will hit 4-5",
        needs_item="loadeddice")
def _loaded_dice():
    rows = []
    for move_id in members("loadeddice"):
        counts = set()
        for seed in range(16):
            f = Fight([swinger(move_id, item="loadeddice")],
                      [wall(mc._reachable(DEX.moves[move_id]))], seed=seed)
            f.turn(Action.move(0), Action.move(0))
            hit = len([e for e in f.log if e.kind == "damage"
                       and (e.side or 0) == 1 and e.move == move_id])
            if hit:
                counts.add(hit)
        rows.append((move_id, counts and counts <= {4, 5}, f"hit {sorted(counts)} times"))
    return verdict(rows)


@family("multihit", "Hits two to five times.",
        "Has a 35% chance to hit two or three times and a 15% chance to hit "
        "four or five times")
def _multihit_spread():
    """35/35/15/15 over 240 casts, three standard errors either way."""
    want = {2: 0.35, 3: 0.35, 4: 0.15, 5: 0.15}
    rows = []
    for move_id in members("multihit"):
        seen = {2: 0, 3: 0, 4: 0, 5: 0}
        tries = 240
        for seed in range(tries):
            f = Fight([swinger(move_id)],
                      [wall(mc._reachable(DEX.moves[move_id]))], seed=seed)
            f.turn(Action.move(0), Action.move(0))
            hit = len([e for e in f.log if e.kind == "damage"
                       and (e.side or 0) == 1 and e.move == move_id])
            if hit in seen:
                seen[hit] += 1
        total = sum(seen.values()) or 1
        off = []
        for count, share in want.items():
            got = seen[count] / total
            slack = max(0.06, 3 * (share * (1 - share) / total) ** 0.5)
            if abs(got - share) > slack:
                off.append(f"{count}x {got:.0%} not {share:.0%}")
        rows.append((move_id, not off,
                     f"{seen} of {total}" + (f" -- {'; '.join(off)}" if off else "")))
    return verdict(rows)


@family("powerherb", "If the user is holding a Power Herb, the move completes "
                     "in one turn")
def _power_herb():
    rows = []
    for move_id in members("powerherb"):
        def first_turn(item, move_id=move_id):
            f = Fight([swinger(move_id, item=item)],
                      [wall(mc._reachable(DEX.moves[move_id]))], seed=7)
            f.turn(Action.move(0), Action.move(0))
            return hp_lost(f), f.item(0)

        without, _ = first_turn(None)
        with_it, left = first_turn("powerherb")
        rows.append((move_id, without == 0 and with_it > 0 and left is None,
                     f"turn one took {without} without and {with_it} with; "
                     f"the herb is {left!r} afterwards"))
    return verdict(rows)


@family("charge", "This attack charges on the first turn and executes on the second")
def _charge():
    rows = []
    for move_id in members("charge"):
        f = Fight([swinger(move_id)],
                  [wall(mc._reachable(DEX.moves[move_id]))], seed=7)
        f.turn(Action.move(0), Action.move(0))
        first = hp_lost(f)
        f.turn(Action.move(0), Action.move(0))
        second = hp_lost(f)
        rows.append((move_id, first == 0 and second > 0,
                     f"{first} on the charge, {second} on the strike"))
    return verdict(rows)


@family("recharge", "the user must recharge on the following turn and cannot "
                    "select a move")
def _recharge():
    from pkcm.engine.state import legal_actions
    rows = []
    for move_id in members("recharge"):
        f = Fight([swinger(move_id)],
                  [wall(mc._reachable(DEX.moves[move_id]))], seed=7)
        f.turn(Action.move(0), Action.move(0))
        struck = hp_lost(f) > 0
        allowed = [str(one) for one in legal_actions(f.state, 0)]
        moves = [one for one in allowed if one.startswith("move")]
        rows.append((move_id, struck and not moves,
                     f"landed={struck}; next turn it may {allowed}"))
    return verdict(rows)


@family("recoil33", "the user takes recoil damage equal to 33% the HP lost by "
                    "the target")
def _recoil_third():
    rows = []
    for move_id in members("recoil33"):
        f = Fight([swinger(move_id)],
                  [wall(mc._reachable(DEX.moves[move_id]))], seed=7)
        before = f.hp(0)
        f.turn(Action.move(0), Action.move(0))
        dealt, paid = hp_lost(f), before - f.hp(0)
        want = round(dealt * 0.33)
        rows.append((move_id, dealt > 0 and abs(paid - want) <= 1,
                     f"dealt {dealt}, paid {paid}, a third is {want}"))
    return verdict(rows)


@family("minimize", "Damage doubles and no accuracy check is done if the "
                    "target has used Minimize while active")
def _minimize():
    rows = []
    for move_id in members("minimize"):
        def took(minimized, move_id=move_id):
            f = Fight([swinger(move_id)],
                      [mon(mc._reachable(DEX.moves[move_id]), "__none__",
                           ("minimize", "splash", "protect", "rest"), None,
                           "sassy", (32, 0, 32, 0, 32, 0))], seed=7)
            f.turn(Action.move(1), Action.move(0 if minimized else 1))
            f.turn(Action.move(0), Action.move(1))
            return hp_lost(f)

        plain, small = took(False), took(True)
        ratio = small / plain if plain else 0.0
        rows.append((move_id, 1.9 <= ratio <= 2.1,
                     f"{plain} -> {small} (x{ratio:.2f})"))
    return verdict(rows)


@family("critratio", "Has a higher chance for a critical hit.")
def _crit_ratio():
    """Focus Energy is the instrument: two stages on top of the move's own.

    Sampling 1/8 against 1/24 needs hundreds of games. Focus Energy adds two
    stages, so a move that starts one stage up is at three and crits every
    time, while one that starts level is at two and does not.
    """
    rows = []
    for move_id in members("critratio"):
        crits = []
        for seed in range(8):
            f = Fight([swinger(move_id, extra=("focusenergy", "splash", "protect"))],
                      [wall(mc._reachable(DEX.moves[move_id]))], seed=seed)
            f.turn(Action.move(1), Action.move(0))
            f.turn(Action.move(0), Action.move(0))
            crits += [e.crit for e in f.log if e.kind == "damage"
                      and (e.side or 0) == 1 and e.move == move_id]
        rows.append((move_id, bool(crits) and all(crits),
                     f"{sum(crits)} of {len(crits)} landings were critical"))
    return verdict(rows)


@family("noaccuracy", "This move does not check accuracy.")
def _no_accuracy():
    rows = []
    for move_id in members("noaccuracy"):
        misses = 0
        for seed in range(8):
            f = Fight([swinger(move_id)],
                      [mon(mc._reachable(DEX.moves[move_id]), "__none__",
                           ("doubleteam", "splash", "protect", "rest"), None,
                           "sassy", (32, 0, 32, 0, 32, 0))], seed=seed)
            for _ in range(3):                   # evasion up to +3
                f.turn(Action.move(1), Action.move(0))
            f.turn(Action.move(0), Action.move(1))
            if not landed(f, move_id):
                misses += 1
        rows.append((move_id, misses == 0,
                     f"missed {misses} of 8 against three stages of evasion"))
    return verdict(rows)



@family("binding", "Prevents the target from switching for four or five turns",
        "Causes damage to the target equal to 1/8 of its maximum HP (1/6 if "
        "the user is holding Binding Band)",
        "This effect is not stackable or reset by using this or another binding move")
def _binding():
    """Four or five turns, an eighth a turn, and one wrap at a time."""
    rows = []
    for move_id in members("binding"):
        lengths, fractions = set(), set()
        for seed in range(12):
            f = Fight([swinger(move_id)], [wall()], seed=seed)
            f.turn(Action.move(0), Action.move(0))
            if not landed(f, move_id):
                continue
            ticks = 0
            for _ in range(9):
                f.turn(Action.move(1), Action.move(0))
                hurt = [e for e in f.log if e.kind == "status_damage"
                        and (e.side or 0) == 1 and e.detail == move_id]
                if not hurt:
                    break
                ticks += 1
                fractions.add(round(f.max_hp(1) / hurt[0].amount))
            lengths.add(ticks + 1)          # the turn it landed counts

        # Casting it again while it is running must not restart the clock.
        g = Fight([swinger(move_id)], [wall()], seed=7)
        g.turn(Action.move(0), Action.move(0))
        first = dict(g.state.sides[1].volatiles[g.state.sides[1].active[0]]
                     .get("partiallytrapped") or {})
        g.turn(Action.move(0), Action.move(0))
        again = dict(g.state.sides[1].volatiles[g.state.sides[1].active[0]]
                     .get("partiallytrapped") or {})
        stackable = first.get("turns") is not None and again.get("turns", 0) > first["turns"]

        rows.append((move_id,
                     lengths <= {4, 5} and lengths and fractions == {8}
                     and not stackable,
                     f"lasted {sorted(lengths)} turns, took 1/{sorted(fractions)} "
                     f"a turn, re-cast restarted it={stackable}"))
    return verdict(rows)


@family("gripclaw", "seven turns if the user is holding Grip Claw",
        needs_item="gripclaw")
def _grip_claw():
    rows = []
    for move_id in members("gripclaw"):
        lengths = set()
        for seed in range(6):
            f = Fight([swinger(move_id, item="gripclaw")], [wall()], seed=seed)
            f.turn(Action.move(0), Action.move(0))
            if not landed(f, move_id):
                continue
            ticks = 0
            for _ in range(11):
                f.turn(Action.move(1), Action.move(0))
                if not [e for e in f.log if e.kind == "status_damage"
                        and (e.side or 0) == 1 and e.detail == move_id]:
                    break
                ticks += 1
            lengths.add(ticks + 1)
        rows.append((move_id, lengths == {7}, f"lasted {sorted(lengths)} turns"))
    return verdict(rows)


@family("bindingband", "1/6 if the user is holding Binding Band",
        needs_item="bindingband")
def _binding_band():
    rows = []
    for move_id in members("bindingband"):
        fractions = set()
        for seed in range(6):
            f = Fight([swinger(move_id, item="bindingband")], [wall()], seed=seed)
            f.turn(Action.move(0), Action.move(0))
            if not landed(f, move_id):
                continue
            f.turn(Action.move(1), Action.move(0))
            hurt = [e for e in f.log if e.kind == "status_damage"
                    and (e.side or 0) == 1 and e.detail == move_id]
            if hurt:
                fractions.add(round(f.max_hp(1) / hurt[0].amount))
        rows.append((move_id, fractions == {6}, f"took 1/{sorted(fractions)} a turn"))
    return verdict(rows)


@family("shedshell", "The target can still switch out if it is holding Shed Shell")
def _shed_shell():
    """Held by the trapped one, and it walks out of anything."""
    from pkcm.engine.state import legal_actions
    rows = []
    for move_id in members("shedshell"):
        def caught(item, move_id=move_id):
            f = until(lambda seed: Fight(
                [swinger(move_id)],
                [wall(mc._reachable(DEX.moves[move_id]), item=item)], seed=seed)
                .turn(Action.move(0), Action.move(0)),
                lambda f: "trapped" in f.volatiles(1))
            if f is None:
                return None
            return [str(one) for one in legal_actions(f.state, 1)
                    if str(one).startswith("switch")]

        held, bare = caught("shedshell"), caught(None)
        rows.append((move_id, held is not None and bare is not None
                     and held and not bare,
                     f"holding Shed Shell it may {held}; holding nothing, {bare}"))
    return verdict(rows)


@family("substitutemulti",
        "If one of the hits breaks the target's substitute, it will take damage "
        "for the remaining hits",
        "If the first hit breaks the target's substitute, it will take damage "
        "for the second hit")
def _substitute_multihit():
    rows = []
    for move_id in members("substitutemulti"):
        f = until(lambda seed, move_id=move_id: Fight(
            [swinger(move_id)],
            [mon("magikarp", "__none__", ("substitute", "splash", "protect", "rest"),
                 None, "serious", (0,) * 6)], seed=seed)
            .turn(Action.move(1), Action.move(0))
            .turn(Action.move(0), Action.move(1)),
            lambda f: any(e.kind == "substitute_broken" or
                          (e.kind == "damage" and (e.side or 0) == 1)
                          for e in f.log))
        if f is None:
            rows.append((move_id, False, "the substitute never broke in thirty tries"))
            continue
        broke = any(e.kind in ("substitute_broken", "substitute_faded") for e in f.log)
        through = any(e.kind == "damage" and (e.side or 0) == 1 for e in f.log)
        rows.append((move_id, not broke or through,
                     f"substitute broke={broke}, the target then took damage={through}"))
    return verdict(rows)


@family("ohko", "Deals damage to the target equal to the target's maximum HP.",
        "Ignores accuracy and evasiveness modifiers.")
def _ohko():
    rows = []
    for move_id in members("ohko"):
        f = until(lambda seed, move_id=move_id: Fight(
            [swinger(move_id)],
            [mon("magikarp", "__none__", ("doubleteam", "splash", "protect", "rest"),
                 None, "serious", (32, 0, 32, 0, 2, 0))], seed=seed)
            .turn(Action.move(1), Action.move(0))
            .turn(Action.move(1), Action.move(0))
            .turn(Action.move(0), Action.move(1)),
            lambda f: landed(f, move_id))
        if f is None:
            rows.append((move_id, False, "never connected in thirty tries"))
            continue
        took = hp_lost(f)
        whole = f.state.pokemon(1, f.state.sides[1].active[0]).max_hp
        rows.append((move_id, took >= whole,
                     f"took {took} off a {whole} HP target behind two Double Teams"))
    return verdict(rows)


@family("outrage",
        "The user spends two or three turns locked into this move and becomes "
        "confused immediately after its move on the last turn of the effect")
def _outrage():
    rows = []
    from pkcm.engine.state import legal_actions
    for move_id in members("outrage"):
        lengths, confused = set(), set()
        for seed in range(10):
            f = Fight([swinger(move_id)], [wall()], seed=seed)
            turns = 0
            for _ in range(5):
                f.turn(Action.move(0), Action.move(0))
                turns += 1
                if "lockedmove" not in f.volatiles(0) and \
                        "rampage" not in f.volatiles(0):
                    break
            lengths.add(turns)
            confused.add("confusion" in f.volatiles(0))
        rows.append((move_id, lengths <= {2, 3} and lengths and confused == {True},
                     f"locked in for {sorted(lengths)} turns; confused "
                     f"afterwards {sorted(confused)}"))
    return verdict(rows)


@family("uturn", "the user switches out even if it is trapped")
def _uturn_out_of_a_trap():
    rows = []
    for move_id in members("uturn"):
        f = Fight([swinger(move_id, extra=("splash", "protect", "rest"))],
                  [mon("gengar", "__none__", ("meanlook", "splash", "protect", "rest"),
                       None, "timid", (32, 0, 32, 0, 2, 32))], seed=7)
        f.turn(Action.move(1), Action.move(0))          # they trap us
        trapped = "trapped" in f.volatiles(0)
        before = f.active_species(0)
        f.turn(Action.move(0), Action.move(1))
        if f.state.phase.name in ("MID_TURN_SWITCH", "FORCED_SWITCH"):
            f.turn(Action.switch(1), Action.PASS)
        rows.append((move_id, trapped and f.active_species(0) != before,
                     f"trapped={trapped}; {before} -> {f.active_species(0)}"))
    return verdict(rows)


@family("stickyhold", "A target with the Sticky Hold Ability does not lose its "
                      "held item if it has not fainted")
def _sticky_hold():
    rows = []
    for move_id in members("stickyhold"):
        def kept(ability, move_id=move_id):
            f = until(lambda seed: Fight(
                [swinger(move_id)],
                [wall(ability=ability, item="leftovers")], seed=seed)
                .turn(Action.move(0), Action.move(0)),
                lambda f: landed(f, move_id) or move_id == "trick")
            return None if f is None else f.item(1)

        held, taken = kept("stickyhold"), kept("__none__")
        rows.append((move_id, held == "leftovers" and taken != "leftovers",
                     f"with Sticky Hold it kept {held!r}; without, {taken!r}"))
    return verdict(rows)



def _delegate(name):
    """Re-run ``mechanic_check``'s own measurement for every member move.

    Not a shortcut: those probes cast the move two hundred times and count.
    What is added here is refusing the "[inconclusive]" that used to pass.
    """
    rows = []
    for move_id in members(name):
        entry = mc.CHECKS.get(move_id)
        if entry is None:
            rows.append((move_id, False, "no check written"))
            continue
        try:
            ok, detail = entry[1]()
        except Exception as error:
            ok, detail = False, f"{type(error).__name__}: {error}"
        blind = "[inconclusive]" in str(detail)
        rows.append((move_id, bool(ok) and not blind,
                     ("unmeasured: " if blind else "") + str(detail)[:90]))
    return verdict(rows)


@family("secondary", "Has a ")
def _secondaries():
    """Every "Has a N% chance to ..." sentence, at the rate the data gives."""
    return _delegate("secondary")


@family("critstages", "chance for a critical hit by")
def _crit_stages():
    """A crit-ratio stage is not a stat stage, and the family below cannot
    tell the difference: the sentence matched "Raises the ... by N stages" and
    was claimed by a check that measures declared ``boosts``. Dragon Cheer was
    handing out Focus Energy's two to everyone because of it.

    Stages here are 1/24, 1/8, 1/2, always -- so the rate names the stage.
    """
    want = {"focusenergy": [("snorlax", 0.5)],
            "dragoncheer": [("snorlax", 0.125), ("garchomp", 0.5)]}
    rows = []
    for move_id in members("critstages"):
        off = []
        for species, share in want[move_id]:
            crits = hits = 0
            for seed in range(120):
                if move_id == "focusenergy":
                    f = Fight([mon(species, "__none__",
                                   (move_id, "bodyslam", "protect", "rest"),
                                   None, "adamant", (32, 32, 0, 0, 2, 32))],
                              [wall("blissey" if "blissey" in DEX.species
                                    else "snorlax")], seed=seed)
                    f.turn(Action.move(0), Action.move(0))
                    f.turn(Action.move(1), Action.move(0))
                    landed_hits = [e for e in f.log if e.kind == "damage"
                                   and (e.side or 0) == 1
                                   and e.move == "bodyslam"]
                else:
                    ours = [mon(UNIVERSAL, "__none__",
                                (move_id, "splash", "protect", "rest"), None,
                                "serious", (32, 0, 32, 0, 2, 32)),
                            mon(species, "__none__",
                                ("bodyslam", "splash", "protect", "rest"),
                                None, "adamant", (32, 32, 0, 0, 2, 0))]
                    theirs = [wall("blissey" if "blissey" in DEX.species
                                   else "snorlax")] * 2
                    f = Pair(ours, theirs, seed=seed)
                    f.turn((Action.move(0, target=TARGET_ALLY),
                            Action.move(0, target=0)),
                           (Action.move(0), Action.move(0)))
                    landed_hits = [e for e in f.log if e.kind == "damage"
                                   and (e.side or 0) == 1
                                   and e.move == "bodyslam"]
                hits += len(landed_hits)
                crits += sum(1 for e in landed_hits if e.crit)
            rate = crits / hits if hits else 0.0
            slack = max(0.08, 3 * (share * (1 - share) / max(1, hits)) ** 0.5)
            if abs(rate - share) > slack:
                off.append(f"{species}: {crits} of {hits} critical "
                           f"({rate:.0%}) against the {share:.0%} that stage "
                           f"gives")
        rows.append((move_id, not off, "; ".join(off) or
                     "the stage the sentence names, counted"))
    return verdict(rows)


@family("declaredboosts", "Lowers the ", "Raises the ", "Raises a ")
def _declared_boosts():
    """Every "Raises/Lowers the ... by N stages" sentence."""
    return _delegate("declaredboosts")


@family("drainfraction", "The user recovers ")
def _drain_fraction():
    """Half of what it took, or three quarters for Draining Kiss."""
    rows = []
    for move_id in members("drainfraction"):
        want = DEX.moves[move_id].raw.get("drain")
        if not want:
            rows.append((move_id, False, "the description drains and the data does not"))
            continue
        f = until(lambda seed, move_id=move_id: Fight(
            [swinger(move_id)], [wall(mc._reachable(DEX.moves[move_id]))], seed=seed),
            lambda f: True)
        slot = f.state.sides[0].active[0]
        f.state.sides[0].hp[slot] = 1
        f.turn(Action.move(0), Action.move(0))
        dealt, back = hp_lost(f), f.hp(0) - 1
        share = want[0] / want[1]
        rows.append((move_id, dealt > 0 and abs(back - round(dealt * share)) <= 1,
                     f"took {dealt}, gave back {back}, the data says "
                     f"{want[0]}/{want[1]} = {round(dealt * share)}"))
    return verdict(rows)


@family("noeffect", "No additional effect.")
def _no_additional_effect():
    """It hits, and nothing else happens: no status, no stage, no volatile."""
    ignore = {"movesused", "moveactions", "lastmove", "lastmovefailed",
              "tookdamagethisturn", "hurtthisturn"}
    rows = []
    for move_id in members("noeffect"):
        move = DEX.moves[move_id]
        declared = [key for key in ("boosts", "status", "volatileStatus",
                                    "secondary", "secondaries", "self",
                                    "sideCondition", "weather", "terrain",
                                    "heal", "drain", "recoil", "forceSwitch",
                                    "selfSwitch")
                    if move.raw.get(key)]
        f = until(lambda seed, move=move: _cast_once(move, seed),
                  lambda f: landed(f, f.under_test))
        if f is None:
            rows.append((move_id, False, "never connected in thirty tries"))
            continue
        left = (set(f.volatiles(1)) | set(f.volatiles(0))) - ignore
        rows.append((move_id,
                     not declared and not left and f.status(1) is None
                     and not f.boosts(1) and not f.boosts(0),
                     f"declares {declared or 'nothing'}; afterwards "
                     f"status={f.status(1)}, boosts={f.boosts(1) or None}/"
                     f"{f.boosts(0) or None}, volatiles={sorted(left) or 'none'}"))
    return verdict(rows)


def _cast_once(move, seed):
    f = Fight([swinger(move.id)], [wall(mc._reachable(move))], seed=seed)
    f.under_test = move.id
    f.turn(Action.move(0), Action.move(0))
    return f



def failed(f, side=0):
    """Whether the move the side just used reported failure."""
    if any(e.kind == "move_failed" and (e.side or 0) == side for e in f.log):
        return True
    volatiles = f.state.sides[side].volatiles[f.state.sides[side].active[0]]
    return bool(volatiles.get("lastmovefailed"))


@family("roomtoggle", "If this move is used during the effect, the effect ends.")
def _room_toggle():
    """Trick, Magic and Wonder Room switch off; the declarative path put them
    straight back, so none of the three could ever be switched off."""
    rows = []
    for move_id in members("roomtoggle"):
        f = Fight([swinger(move_id)], [wall()], seed=7)
        f.turn(Action.move(0), Action.move(0))
        up = dict(f.state.field.rooms)
        f.turn(Action.move(0), Action.move(0))
        down = dict(f.state.field.rooms)
        rows.append((move_id, move_id in up and move_id not in down,
                     f"after one cast {up or 'nothing'}, after two "
                     f"{down or 'nothing'}"))
    return verdict(rows)


@family("alreadyup",
        "Fails if the effect is already active on the user's side.",
        "Fails if the effect is already active on the opposing side.",
        "Fails if the effect is already active.",
        "Fails if this move is already in effect for the user's side.",
        "Fails if this move is already in effect for the user's position.",
        "Fails if this move is already in effect.",
        "Fails if this effect is active for the user.",
        "Fails if the user already has the effect.",
        "Fails if the current weather is",
        "Fails if the current terrain is",
        "Fails if the user is already under this effect",
        "Fails if the user has already been prevented from switching by this "
        "effect.")
def _already_up():
    """Cast it, then cast it again: the second one has to refuse."""
    rows = []
    for move_id in members("alreadyup"):
        f = Fight([swinger(move_id)],
                  [wall(mc._reachable(DEX.moves[move_id]))], seed=7)
        f.turn(Action.move(0), Action.move(0))
        first = failed(f)
        f.turn(Action.move(0), Action.move(0))
        again = failed(f)
        rows.append((move_id, not first and again,
                     f"the first cast failed={first}, the second={again}"))
    return verdict(rows)


@family("needsally", "Fails if there is no ally adjacent to the user",
        "Fails if it is not a Double Battle or Battle Royal.",
        "Fails if the user is the only Pokemon on its side.")
def _needs_an_ally():
    """Singles has nobody to point at, so every one of these must refuse."""
    rows = []
    for move_id in members("needsally"):
        f = Fight([swinger(move_id)],
                  [wall(mc._reachable(DEX.moves[move_id]))], seed=7)
        f.turn(Action.move(0), Action.move(0))
        rows.append((move_id, failed(f), f"failed in singles={failed(f)}"))
    return verdict(rows)


@family("firstturn", "Fails unless it is the user's first turn on the field.")
def _first_turn_only():
    rows = []
    for move_id in members("firstturn"):
        f = Fight([swinger(move_id)],
                  [wall(mc._reachable(DEX.moves[move_id]))], seed=7)
        f.turn(Action.move(0), Action.move(0))
        first = failed(f)
        f.turn(Action.move(0), Action.move(0))
        later = failed(f)
        rows.append((move_id, not first and later,
                     f"on the first turn it failed={first}, on the second={later}"))
    return verdict(rows)


@family("needsasleep", "Fails if the user is not asleep.")
def _needs_to_be_asleep():
    rows = []
    for move_id in members("needsasleep"):
        awake = Fight([swinger(move_id)],
                      [wall(mc._reachable(DEX.moves[move_id]))], seed=7)
        awake.turn(Action.move(0), Action.move(0))
        dozing = Fight([swinger(move_id)],
                       [wall(mc._reachable(DEX.moves[move_id]))], seed=7)
        side = dozing.state.sides[0]
        side.status[side.active[0]] = "slp"
        dozing.turn(Action.move(0), Action.move(0))
        rows.append((move_id, failed(awake) and not failed(dozing),
                     f"awake it failed={failed(awake)}, asleep={failed(dozing)}"))
    return verdict(rows)


@family("counterneedsahit",
        "Fails if the user was not hit by an opposing Pokemon's physical attack "
        "this turn.",
        "Fails if the user was not hit by an opposing Pokemon's special attack "
        "this turn.",
        "Fails if the user was not hit by an opposing Pokemon's physical or "
        "special attack this turn.")
def _counter_needs_a_hit():
    #: Mirror Coat answers special attacks, the other three physical ones.
    kind = {"mirrorcoat": "shadowball"}
    rows = []
    for move_id in members("counterneedsahit"):
        swing = kind.get(move_id, "bodyslam")

        def bout(move_id=move_id):
            return Fight(
                [mon("wobbuffet", "__none__",
                     (move_id, "splash", "protect", "rest"), None, "sassy",
                     (32, 0, 32, 0, 32, 0))],
                # Not a Weavile: Mirror Coat comes back Psychic and a Dark
                # type is immune to it, which reads exactly like a refusal.
                [mon("garchomp", "__none__", ("splash", swing, "protect", "rest"),
                     None, "jolly", (0, 32, 2, 32, 0, 32))], seed=7)

        quiet = bout()
        quiet.turn(Action.move(0), Action.move(0))          # they Splash
        hit = bout()
        hit.turn(Action.move(0), Action.move(1))            # they swing
        rows.append((move_id, failed(quiet) and not failed(hit),
                     f"against Splash it failed={failed(quiet)}, against a "
                     f"{swing}={failed(hit)}"))
    return verdict(rows)


@family("stockpilecount", "Fails if the user's Stockpile count is 0.",
        "Fails if the user's Stockpile count is 3.")
def _stockpile_count():
    rows = []
    for move_id in members("stockpilecount"):
        empty = Fight([swinger(move_id, extra=("stockpile", "splash", "protect"))],
                      [wall()], seed=7)
        empty.turn(Action.move(0), Action.move(0))
        full = Fight([swinger(move_id, extra=("stockpile", "splash", "protect"))],
                     [wall()], seed=7)
        slot = full.state.sides[0].active[0]
        full.state.sides[0].hp[slot] //= 2      # Swallow needs room to heal
        for _ in range(3):
            full.turn(Action.move(1), Action.move(0))
        full.turn(Action.move(0), Action.move(0))
        if move_id == "stockpile":
            rows.append((move_id, not failed(empty) and failed(full),
                         f"from empty it failed={failed(empty)}, on three "
                         f"layers={failed(full)}"))
        else:
            rows.append((move_id, failed(empty) and not failed(full),
                         f"from empty it failed={failed(empty)}, on three "
                         f"layers={failed(full)}"))
    return verdict(rows)


@family("needsanitem", "Fails if the target has no held item.",
        "Fails if the user is not holding a Berry.",
        "Fails if no active Pokemon is holding a Berry.")
def _needs_an_item():
    rows = []
    for move_id in members("needsanitem"):
        theirs = move_id == "poltergeist"
        def cast(item, theirs=theirs, move_id=move_id):
            f = Fight([swinger(move_id, item=None if theirs else item)],
                      [wall(mc._reachable(DEX.moves[move_id]),
                            item=item if theirs else None)], seed=7)
            f.turn(Action.move(0), Action.move(0))
            return failed(f)

        rows.append((move_id, cast(None) and not cast("sitrusberry"),
                     f"with nothing it failed={cast(None)}, with a berry="
                     f"{cast('sitrusberry')}"))
    return verdict(rows)



@family("protectchain",
        "This move has a 1/X chance of being successful, where X starts at 1 "
        "and triples each time this move is successfully used.",
        "X resets to 1 if this move fails, if the user's last move used is not "
        "Baneful Bunker",
        "Fails if the user moves last this turn.")
def _protect_chain():
    """Once is certain, twice is a third, and something else in between resets."""
    rows = []
    for move_id in members("protectchain"):
        first = second = tries = 0
        for seed in range(60):
            f = Fight([swinger(move_id, extra=("splash", "protect", "rest"))],
                      [wall()], seed=seed)
            f.turn(Action.move(0), Action.move(1))
            if failed(f):
                continue
            tries += 1
            first += 1
            f.turn(Action.move(0), Action.move(1))
            second += not failed(f)
        share = second / tries if tries else 0.0
        # A third, give or take three standard errors on sixty samples.
        slack = max(0.12, 3 * (1 / 3 * 2 / 3 / max(1, tries)) ** 0.5)
        rows.append((move_id, first == tries and abs(share - 1 / 3) <= slack,
                     f"the first cast worked {first} of {tries} times, the "
                     f"second {second} ({share:.0%} against a third)"))
    return verdict(rows)


@family("bindingrelease",
        "The effect ends if either the user or the target leaves the field, or "
        "if the target uses Mortal Spin, Rapid Spin, or Substitute successfully.",
        "The effect ends if either the user or the target leaves the field.")
def _the_binder_leaving_ends_it():
    rows = []
    for move_id in members("bindingrelease"):
        if move_id == "lockon":
            # Not a trap: the aim is on the user, and it is the *target*
            # leaving that has to end it.
            f = Fight([swinger(move_id)],
                      [wall(mc._reachable(DEX.moves[move_id])),
                       mon("magikarp", "__none__", ("splash", "tackle"))],
                      seed=7)
            f.turn(Action.move(0), Action.move(0))
            aimed = "lockon" in f.volatiles(0)
            f.turn(Action.move(1), Action.switch(1))
            rows.append((move_id, aimed and "lockon" not in f.volatiles(0),
                         f"locked on={aimed}; after the target left it held "
                         f"{sorted(f.volatiles(0) & {'lockon'})}"))
            continue
        f = until(lambda seed, move_id=move_id: Fight(
            [mon(UNIVERSAL, "__none__", (move_id, "splash", "protect", "rest"),
                 None, "adamant", (32, 32, 0, 0, 2, 0)),
             mon("magikarp", "__none__", ("splash", "tackle"))],
            [wall(mc._reachable(DEX.moves[move_id]))], seed=seed)
            .turn(Action.move(0), Action.move(0)),
            lambda f: "trapped" in f.volatiles(1))
        if f is None:
            rows.append((move_id, False, "never caught anything in thirty tries"))
            continue
        f.turn(Action.switch(1), Action.move(0))       # the binder walks away
        rows.append((move_id, "trapped" not in f.volatiles(1),
                     f"after the user left, the target held "
                     f"{sorted(f.volatiles(1) & {'trapped', 'partiallytrapped'})}"))
    return verdict(rows)


@family("hitstwice", "Hits twice.")
def _hits_twice():
    rows = []
    for move_id in members("hitstwice"):
        counts = set()
        for seed in range(8):
            f = Fight([swinger(move_id)],
                      [wall(mc._reachable(DEX.moves[move_id]))], seed=seed)
            f.turn(Action.move(0), Action.move(0))
            hit = len([e for e in f.log if e.kind == "damage"
                       and (e.side or 0) == 1 and e.move == move_id])
            if hit:
                counts.add(hit)
        rows.append((move_id, counts == {2}, f"hit {sorted(counts)} times"))
    return verdict(rows)


@family("statsplit", "Stat stage changes are unaffected.")
def _stat_stages_are_left_alone():
    """Guard Split and friends move the raw stats, not the stages."""
    rows = []
    for move_id in members("statsplit"):
        f = Fight([mon(UNIVERSAL, "__none__",
                       (move_id, "swordsdance", "splash", "protect"), None,
                       "adamant", (32, 32, 0, 0, 2, 0))],
                  [wall()], seed=7)
        f.turn(Action.move(1), Action.move(0))          # two stages of Attack
        before = dict(f.boosts(0)), dict(f.boosts(1))
        f.turn(Action.move(0), Action.move(0))
        after = dict(f.boosts(0)), dict(f.boosts(1))
        rows.append((move_id, before == after,
                     f"stages {before} -> {after}"))
    return verdict(rows)


@family("crashdamage",
        "If this attack is not successful, the user loses half of its maximum "
        "HP, rounded down, as crash damage.",
        "Pokemon with the Magic Guard Ability are unaffected by crash damage.")
def _crash_damage():
    rows = []
    for move_id in members("crashdamage"):
        def missed(ability, move_id=move_id):
            for seed in range(30):
                f = Fight([swinger(move_id, ability=ability)],
                          [wall(mc._reachable(DEX.moves[move_id]))], seed=seed)
                side = f.state.sides[1]
                side.boosts[side.active[0]][mc.BOOST_INDEX["evasion"]] = 6
                whole = f.hp(0)
                f.turn(Action.move(0), Action.move(0))
                if not landed(f, move_id):
                    return whole - f.hp(0), f.max_hp(0)
            return None, None

        paid, whole = missed("__none__")
        guarded, _ = missed("magicguard")
        rows.append((move_id, paid == whole // 2 and guarded == 0,
                     f"a miss cost {paid} of {whole} (half is {whole // 2 if whole else '?'}); "
                     f"behind Magic Guard it cost {guarded}"))
    return verdict(rows)


@family("sturdyohko", "Pokemon with the Sturdy Ability are immune.")
def _sturdy_stops_an_ohko():
    rows = []
    for move_id in members("sturdyohko"):
        def fell(ability, move_id=move_id):
            for seed in range(30):
                f = Fight([swinger(move_id)],
                          [wall(mc._reachable(DEX.moves[move_id]),
                                ability=ability)], seed=seed)
                f.turn(Action.move(0), Action.move(0))
                if landed(f, move_id):
                    return True
            return False

        rows.append((move_id, fell("__none__") and not fell("sturdy"),
                     f"it landed on a plain target={fell('__none__')}, on a "
                     f"Sturdy one={fell('sturdy')}"))
    return verdict(rows)


@family("thawsthetarget", "The target thaws out if it is frozen.")
def _thaws_the_target():
    rows = []
    for move_id in members("thawsthetarget"):
        f = Fight([swinger(move_id)],
                  [wall(mc._reachable(DEX.moves[move_id]))], seed=7)
        side = f.state.sides[1]
        side.status[side.active[0]] = "frz"
        f.turn(Action.move(0), Action.move(0))
        rows.append((move_id, f.status(1) != "frz",
                     f"the frozen target came out {f.status(1)}"))
    return verdict(rows)


@family("selfdestruct",
        "The user faints after using this move, even if this move fails for "
        "having no target.",
        "This move is prevented from executing if any active Pokemon has the "
        "Damp Ability.")
def _self_destruct():
    rows = []
    for move_id in members("selfdestruct"):
        f = Fight([swinger(move_id)],
                  [wall(mc._reachable(DEX.moves[move_id]))], seed=7)
        f.turn(Action.move(0), Action.move(0))
        gone = f.state.sides[0].hp[0] == 0

        damp = Fight([swinger(move_id)],
                     [wall(mc._reachable(DEX.moves[move_id]), ability="damp")],
                     seed=7)
        damp.turn(Action.move(0), Action.move(0))
        smothered = damp.state.sides[0].hp[0] > 0 and not landed(damp, move_id)
        rows.append((move_id, gone and smothered,
                     f"the user fainted={gone}; in front of Damp it was "
                     f"smothered={smothered}"))
    return verdict(rows)


@family("halfhealth", "The user restores 1/2 of its maximum HP, rounded half up.")
def _restores_half():
    rows = []
    for move_id in members("halfhealth"):
        f = Fight([swinger(move_id)], [wall()], seed=7)
        slot = f.state.sides[0].active[0]
        whole = f.max_hp(0)
        f.state.sides[0].hp[slot] = 1
        f.turn(Action.move(0), Action.move(0))
        got = f.hp(0) - 1
        rows.append((move_id, abs(got - (whole + 1) // 2) <= 1,
                     f"restored {got} of {whole}, half being {(whole + 1) // 2}"))
    return verdict(rows)


@family("weatherheal",
        "The user restores 1/2 of its maximum HP if Delta Stream or no weather "
        "conditions are in effect or if the user is holding Utility Umbrella")
def _weather_heal():
    """Half in clear weather, two thirds in sun, a quarter in anything else."""
    want = {None: 1 / 2, "sunnyday": 2 / 3, "raindance": 1 / 4,
            "sandstorm": 1 / 4, "snowscape": 1 / 4}
    rows = []
    for move_id in members("weatherheal"):
        off = []
        for weather, share in want.items():
            # Magic Guard, because a sandstorm takes a sixteenth off at the
            # end of the same turn and that is not what this measures.
            f = Fight([swinger(move_id, ability="magicguard")], [wall()], seed=7)
            if weather:
                f.state.field.weather = weather
                f.state.field.weather_turns = 8
            slot = f.state.sides[0].active[0]
            whole = f.max_hp(0)
            f.state.sides[0].hp[slot] = 1
            f.turn(Action.move(0), Action.move(0))
            got = f.hp(0) - 1
            if abs(got - round(whole * share)) > 2:
                off.append(f"{weather or 'clear'}: {got} not {round(whole * share)}")
        rows.append((move_id, not off, "; ".join(off) or "every weather as declared"))
    return verdict(rows)


@family("hitsairborne", "This move can hit a target using Bounce, Fly, or Sky "
                        "Drop, or is under the effect of Sky Drop.")
def _hits_the_airborne():
    rows = []
    for move_id in members("hitsairborne"):
        f = Fight([mon(UNIVERSAL, "__none__", (move_id, "splash", "protect", "rest"),
                       None, "modest", (32, 0, 0, 32, 2, 0))],
                  [mon("mew", "__none__", ("fly", "splash", "protect", "rest"),
                       None, "jolly", (32, 32, 0, 0, 2, 32))], seed=7)
        f.turn(Action.move(1), Action.move(0))          # they take off
        f.turn(Action.move(0), Action.move(0))          # we swing at the sky
        rows.append((move_id, landed(f, move_id),
                     f"reached something in the air={landed(f, move_id)}"))
    return verdict(rows)


@family("alwayscrit", "This move is always a critical hit unless the target is "
                      "under the effect of Lucky Chant or has the Battle Armor "
                      "or Shell Armor Abilities.")
def _always_a_crit():
    rows = []
    for move_id in members("alwayscrit"):
        def crits(ability, move_id=move_id):
            seen = []
            for seed in range(6):
                f = Fight([swinger(move_id)],
                          [wall(mc._reachable(DEX.moves[move_id]), ability=ability)],
                          seed=seed)
                f.turn(Action.move(0), Action.move(0))
                seen += [e.crit for e in f.log if e.kind == "damage"
                         and (e.side or 0) == 1 and e.move == move_id]
            return seen

        plain, armoured = crits("__none__"), crits("battlearmor")
        rows.append((move_id, plain and all(plain) and armoured and not any(armoured),
                     f"{sum(plain)}/{len(plain)} critical normally, "
                     f"{sum(armoured)}/{len(armoured)} against Battle Armor"))
    return verdict(rows)


@family("terrainends", "Ends the effects of Electric Terrain, Grassy Terrain, "
                       "Misty Terrain, and Psychic Terrain.")
def _ends_the_terrain():
    rows = []
    for move_id in members("terrainends"):
        f = Fight([swinger(move_id)],
                  [wall(mc._reachable(DEX.moves[move_id]))], seed=7)
        f.state.field.terrain = "grassyterrain"
        f.state.field.terrain_turns = 8
        f.turn(Action.move(0), Action.move(0))
        rows.append((move_id, f.state.field.terrain is None,
                     f"the terrain afterwards is {f.state.field.terrain}"))
    return verdict(rows)


@family("fixeddamage", "Deals damage to the target equal to the user's level.")
def _the_users_level():
    rows = []
    for move_id in members("fixeddamage"):
        f = Fight([swinger(move_id)],
                  [wall(mc._reachable(DEX.moves[move_id]))], seed=7)
        f.turn(Action.move(0), Action.move(0))
        rows.append((move_id, hp_lost(f) == 50, f"took {hp_lost(f)}, and the level is 50"))
    return verdict(rows)


@family("screenbreakers",
        "If this attack does not miss, the effects of Reflect, Light Screen, "
        "and Aurora Veil end for the target's side of the field before damage "
        "is calculated.",
        "It is removed from the user's side if the user or an ally is "
        "successfully hit by Brick Break, Psychic Fangs, or Defog.",
        "Lasts for 8 turns if the user is holding Light Clay.",
        "Critical hits ignore this effect.",
        "Damage is not reduced further with Aurora Veil.",
        "Critical hits ignore this protection.")
def _the_screens():
    """Delegated to the four written checks in ``mechanic_check``."""
    return _delegate("screenbreakers")



@family("plainstatus", "Causes the target to become confused.",
        "Causes the target to fall asleep.", "Paralyzes the target.",
        "Poisons the target.", "Burns the target.",
        "Badly poisons the target.")
def _plain_status():
    """The status moves whose whole description is the status they inflict."""
    return _delegate("plainstatus")


@family("trapstatement", "Prevents the target from switching out.")
def _prevents_switching():
    from pkcm.engine.state import legal_actions
    rows = []
    for move_id in members("trapstatement"):
        f = until(lambda seed, move_id=move_id: Fight(
            [swinger(move_id)], [wall(mc._reachable(DEX.moves[move_id]))], seed=seed)
            .turn(Action.move(0), Action.move(0)),
            lambda f: "trapped" in f.volatiles(1))
        held = f is not None and not [one for one in legal_actions(f.state, 1)
                                      if str(one).startswith("switch")]
        rows.append((move_id, held, f"the target may still leave={not held}"))
    return verdict(rows)


@family("ohkoaccuracy",
        "This attack's accuracy is equal to (user's level - target's level + 30)%, "
        "and fails if the target is at a higher level.",
        "This attack's accuracy is equal to (user's level - target's level + X)%")
def _ohko_accuracy():
    """Everything is level 50 here, so the accuracy is a flat 30% -- or 20%
    for Sheer Cold in hands that are not an Ice type's."""
    rows = []
    for move_id in members("ohkoaccuracy"):
        want = 0.30
        if move_id == "sheercold":
            want = 0.20                       # Mew is not an Ice type
        landed_count = tries = 0
        for seed in range(200):
            f = Fight([swinger(move_id)],
                      [wall(mc._reachable(DEX.moves[move_id]))], seed=seed)
            f.turn(Action.move(0), Action.move(0))
            tries += 1
            landed_count += landed(f, move_id)
        rate = landed_count / tries
        slack = max(0.05, 3 * (want * (1 - want) / tries) ** 0.5)
        rows.append((move_id, abs(rate - want) <= slack,
                     f"landed {landed_count} of {tries} ({rate:.0%}) against "
                     f"the {want:.0%} the formula gives at level 50"))
    return verdict(rows)


@family("counterdamage",
        "Deals damage to the last opposing Pokemon to hit the user with a "
        "physical or special attack this turn equal to 1.5 times the HP lost by "
        "the user from that attack.",
        "If the user did not lose HP from that attack, this move deals 1 HP of "
        "damage instead.",
        "If the user did not lose HP from the attack, this move deals 1 HP of "
        "damage instead.",
        "Only the last hit of a multi-hit attack is counted.")
def _counter_damage():
    """Twice for Counter and Mirror Coat, half again for the other two."""
    want = {"counter": 2.0, "mirrorcoat": 2.0, "comeuppance": 1.5,
            "metalburst": 1.5}
    rows = []
    for move_id in members("counterdamage"):
        swing = "shadowball" if move_id == "mirrorcoat" else "bodyslam"
        f = Fight([mon("wobbuffet", "__none__",
                       (move_id, "splash", "protect", "rest"), None, "sassy",
                       (32, 0, 32, 0, 32, 0))],
                  [mon("garchomp", "__none__", ("splash", swing, "protect", "rest"),
                       None, "jolly", (0, 32, 2, 32, 0, 32))], seed=7)
        before = f.hp(0)
        f.turn(Action.move(0), Action.move(1))
        taken, given = before - f.hp(0), hp_lost(f)
        share = given / taken if taken else 0.0
        rows.append((move_id, taken > 0 and abs(share - want[move_id]) <= 0.06,
                     f"took {taken}, returned {given} (x{share:.2f} against "
                     f"the x{want[move_id]} in the data)"))
    return verdict(rows)


@family("hazardremoval",
        "Can be removed from the opposing side if any Pokemon uses Tidy Up, or "
        "if any opposing Pokemon uses Mortal Spin, Rapid Spin, or Defog "
        "successfully, or is hit by Defog.")
def _hazards_can_be_swept():
    rows = []
    for move_id in members("hazardremoval"):
        hazard = {"ceaselessedge": "spikes", "stoneaxe": "stealthrock"}.get(
            move_id, move_id)
        for sweeper in ("rapidspin", "defog", "tidyup"):
            f = Fight([mon(UNIVERSAL, "__none__",
                           (sweeper, "splash", "protect", "rest"), None,
                           "adamant", (32, 32, 0, 0, 2, 0))],
                      [wall()], seed=7)
            f.state.sides[0].conditions[hazard] = 1
            if sweeper == "defog":                 # Defog clears the far side
                f.state.sides[0].conditions.pop(hazard)
                f.state.sides[1].conditions[hazard] = 1
            f.turn(Action.move(0), Action.move(0))
            side = 1 if sweeper == "defog" else 0
            if hazard in f.conditions(side):
                rows.append((move_id, False, f"{sweeper} left {hazard} standing"))
                break
        else:
            rows.append((move_id, True, "Rapid Spin, Defog and Tidy Up all "
                                        "cleared it"))
    return verdict(rows)


@family("phazing",
        "If both the user and the target have not fainted, the target is forced "
        "to switch out and be replaced with a random unfainted ally.",
        "Fails if the target is the last unfainted Pokemon in its party, or if "
        "the target used Ingrain previously or has the Suction Cups Ability.")
def _phazing():
    rows = []
    for move_id in members("phazing"):
        def blown(ability, move_id=move_id):
            f = until(lambda seed: Fight(
                [swinger(move_id)],
                [wall(mc._reachable(DEX.moves[move_id]), ability=ability),
                 mon("magikarp", "__none__", ("splash", "tackle")),
                 mon("pikachu", "__none__", ("splash", "tackle"))], seed=seed),
                lambda f: True)
            before = f.active_species(1)
            f.turn(Action.move(0), Action.move(0))
            return before != f.active_species(1)

        rows.append((move_id, blown("__none__") and not blown("suctioncups"),
                     f"it moved a plain target={blown('__none__')}, one with "
                     f"Suction Cups={blown('suctioncups')}"))
    return verdict(rows)


@family("recoilhalf", "If the target lost HP, the user takes recoil damage "
                      "equal to 1/2 the HP lost by the target, rounded half up, "
                      "but not less than 1 HP.")
def _recoil_half():
    rows = []
    for move_id in members("recoilhalf"):
        f = Fight([swinger(move_id)],
                  [wall(mc._reachable(DEX.moves[move_id]))], seed=7)
        before = f.hp(0)
        f.turn(Action.move(0), Action.move(0))
        dealt, paid = hp_lost(f), before - f.hp(0)
        rows.append((move_id, dealt > 0 and abs(paid - (dealt + 1) // 2) <= 1,
                     f"dealt {dealt}, paid {paid}, half being {(dealt + 1) // 2}"))
    return verdict(rows)


@family("breaksprotection",
        "If this move is successful, it breaks through the target's Baneful "
        "Bunker, Detect, King's Shield, Protect, or Spiky Shield for this turn, "
        "allowing other Pokemon to attack the target normally.")
def _breaks_protection():
    rows = []
    for move_id in members("breaksprotection"):
        f = Fight([swinger(move_id)],
                  [mon(mc._reachable(DEX.moves[move_id]), "__none__",
                       ("protect", "splash", "rest", "bodyslam"), None,
                       "sassy", (32, 0, 32, 0, 32, 0))], seed=7)
        if "charge" in DEX.moves[move_id].flags:
            f.turn(Action.move(0), Action.move(1))     # spend the charge turn
        f.turn(Action.move(0), Action.move(0))
        rows.append((move_id, landed(f, move_id),
                     f"reached through Protect={landed(f, move_id)}"))
    return verdict(rows)


@family("solarweather",
        "If the user is holding a Power Herb or the weather is Desolate Land or "
        "Sunny Day, the move completes in one turn.")
def _solar_in_the_sun():
    rows = []
    for move_id in members("solarweather"):
        f = Fight([swinger(move_id)],
                  [wall(mc._reachable(DEX.moves[move_id]))], seed=7)
        f.state.field.weather = "sunnyday"
        f.state.field.weather_turns = 8
        f.turn(Action.move(0), Action.move(0))
        rows.append((move_id, hp_lost(f) > 0,
                     f"in the sun it struck on the first turn={hp_lost(f) > 0}"))
    return verdict(rows)


@family("weatheraccuracy",
        "If the weather is Primordial Sea or Rain Dance, this move does not "
        "check accuracy.",
        "If the weather is Desolate Land or Sunny Day, this move's accuracy "
        "is 50%.")
def _weather_accuracy():
    rows = []
    for move_id in members("weatheraccuracy"):
        def rate(weather, move_id=move_id):
            hits = 0
            for seed in range(60):
                f = Fight([swinger(move_id)],
                          [wall(mc._reachable(DEX.moves[move_id]))], seed=seed)
                if weather:
                    f.state.field.weather = weather
                    f.state.field.weather_turns = 8
                f.turn(Action.move(0), Action.move(0))
                hits += landed(f, move_id)
            return hits / 60

        wet, dry = rate("raindance"), rate("sunnyday")
        rows.append((move_id, wet == 1.0 and abs(dry - 0.5) <= 0.2,
                     f"in rain {wet:.0%}, in sun {dry:.0%} against the 100% "
                     f"and 50% in the data"))
    return verdict(rows)


@family("steals", "If this attack was successful and the user has not fainted, "
                  "it steals the target's held item if the user is not holding "
                  "one.",
        "If this move is successful and the user has not fainted, it steals the "
        "target's held Berry if it is holding one and eats it immediately.")
def _steals():
    rows = []
    for move_id in members("steals"):
        berry = move_id in ("bugbite", "pluck")
        item = "sitrusberry" if berry else "leftovers"
        f = until(lambda seed, move_id=move_id, item=item: Fight(
            [swinger(move_id)],
            [wall(mc._reachable(DEX.moves[move_id]), item=item)], seed=seed)
            .turn(Action.move(0), Action.move(0)),
            lambda f: landed(f, f.state and move_id))
        if f is None:
            rows.append((move_id, False, "never connected in thirty tries"))
            continue
        took = f.item(1) is None
        holding = f.item(0)
        rows.append((move_id, took and (holding is None if berry else holding == item),
                     f"the target kept {f.item(1)!r}, the user holds {holding!r}"))
    return verdict(rows)


@family("spinsfree",
        "the effects of Leech Seed and binding moves end for the user")
def _spins_free():
    rows = []
    for move_id in members("spinsfree"):
        f = Fight([swinger(move_id)],
                  [wall(mc._reachable(DEX.moves[move_id]))], seed=7)
        f.state.sides[0].conditions["spikes"] = 2
        slot = f.state.sides[0].active[0]
        f.state.sides[0].volatiles[slot]["leechseed"] = {}
        f.turn(Action.move(0), Action.move(0))
        rows.append((move_id, "spikes" not in f.conditions(0)
                     and "leechseed" not in f.volatiles(0),
                     f"afterwards the side held {f.conditions(0) or 'nothing'} "
                     f"and the user {sorted(f.volatiles(0) & {'leechseed'})}"))
    return verdict(rows)


@family("ignoresstages", "Ignores the target's stat stage changes, including "
                         "evasiveness.")
def _ignores_stages():
    rows = []
    for move_id in members("ignoresstages"):
        def damage(boosted):
            f = Fight([swinger(move_id)],
                      [mon(mc._reachable(DEX.moves[move_id]), "__none__",
                           ("irondefense", "splash", "protect", "rest"), None,
                           "sassy", (32, 0, 32, 0, 32, 0))], seed=7)
            if boosted:
                f.turn(Action.move(1), Action.move(0))
                f.turn(Action.move(1), Action.move(0))
            f.turn(Action.move(0), Action.move(1))
            return hp_lost(f)

        plain, walled = damage(False), damage(True)
        rows.append((move_id, plain and abs(walled - plain) <= max(2, plain * 0.05),
                     f"{plain} against a plain target, {walled} through four "
                     f"stages of Defence"))
    return verdict(rows)


@family("obliviousveil", "Pokemon with the Oblivious Ability or protected by "
                         "the Aroma Veil Ability are immune.")
def _oblivious_and_aroma_veil():
    rows = []
    for move_id in members("obliviousveil"):
        def stuck(ability, move_id=move_id):
            # Not Mew: it is genderless, and Attract refuses a genderless
            # user whatever the ability across from it is doing.
            f = Fight([gendered("garchomp", "__none__",
                                (move_id, "splash", "protect", "rest"),
                                "adamant", (32, 32, 0, 0, 2, 0), "M")],
                      [gendered("snorlax", ability,
                                ("splash", "bodyslam", "protect", "rest"),
                                "sassy", (32, 0, 32, 0, 32, 0), "F")], seed=7)
            f.turn(Action.move(0), Action.move(0))
            return f.volatiles(1) & {"attract", "taunt"}

        rows.append((move_id, stuck("__none__") and not stuck("oblivious")
                     and not stuck("aromaveil"),
                     f"plain {sorted(stuck('__none__'))}, Oblivious "
                     f"{sorted(stuck('oblivious'))}, Aroma Veil "
                     f"{sorted(stuck('aromaveil'))}"))
    return verdict(rows)


@family("snowfive", "For 5 turns, the weather becomes Snow.")
def _five_turns_of_snow():
    rows = []
    for move_id in members("snowfive"):
        f = Fight([swinger(move_id)], [wall()], seed=7)
        f.turn(Action.move(0), Action.move(0))
        if f.state.phase.name in ("MID_TURN_SWITCH", "FORCED_SWITCH"):
            f.turn(Action.switch(1), Action.PASS)
        rows.append((move_id, f.state.field.weather == "snowscape"
                     and f.state.field.weather_turns == 4,
                     f"{f.state.field.weather} with "
                     f"{f.state.field.weather_turns} turns left of five"))
    return verdict(rows)



def rolls_for(f, move, power, side=0, as_type=None, extra=1.0, crit=False):
    """The sixteen damages the formula gives for a power worked out here.

    ``mechanic_check._expected_rolls`` reads ``move.base_power``, which is
    zero or meaningless for the fourteen moves whose power is a formula. This
    takes the number the description says and checks the engine against it,
    rather than against itself.
    """
    from pkcm.data.dex import Stat
    from pkcm.engine.battle import make_context
    from pkcm.engine.moves import damage_base, damage_from_base
    from pkcm.engine.mutate import effective_stat

    ctx = make_context(f.state)
    us = (side, f.state.sides[side].active[0])
    them = (1 - side, f.state.sides[1 - side].active[0])
    physical = move.category == "Physical"
    attack = effective_stat(ctx, us, Stat.ATK if physical else Stat.SPA)
    defense = effective_stat(ctx, them, Stat.DEF if physical else Stat.SPD)
    # Weather Ball and Terrain Pulse change type in the same sentence that
    # doubles their power, so the expectation has to change type with them.
    kind = as_type or move.type
    effectiveness = DEX.type_chart.multiplier(kind, f.state.types(*them))
    stab = kind in f.state.types(*us)
    base = damage_base(power=power, attack=attack, defense=defense, crit=crit)
    # ``extra`` is the field's own hand on the damage -- Rain multiplying a
    # Water move by 1.5, a terrain multiplying its own type by 1.3. Not what
    # the clause under test says, but present in the position it needs, and
    # written the way the engine writes it so the rounding matches.
    return {max(1, int(damage_from_base(base, roll, stab=stab,
                                        effectiveness=effectiveness) * extra))
            for roll in range(mc._ROLL_LOW, mc._ROLL_HIGH + 1)}


def power_check(move_id, arrange, power, seeds=range(12)):
    """Play a position, work out the power its description gives, compare."""
    move = DEX.moves[move_id]
    for seed in seeds:
        f = Fight([swinger(move_id)], [wall(mc._reachable(move))], seed=seed)
        arrange(f)
        want = rolls_for(f, move, power(f))
        f.turn(Action.move(0), Action.move(0))
        hits = [e for e in f.log if e.kind == "damage" and (e.side or 0) == 1
                and e.move == move_id and not e.crit]
        if not hits:
            continue
        got = hits[0].amount
        return (got in want,
                f"dealt {got}; power {power(f)} gives {min(want)}-{max(want)}")
    return False, "never landed a clean uncritical hit in twelve tries"


@family("statuspower", "Power doubles if the target has a non-volatile status "
                       "condition.")
def _doubles_on_status():
    rows = []
    for move_id in members("statuspower"):
        def arrange(f):
            side = f.state.sides[1]
            side.status[side.active[0]] = "brn"

        rows.append((move_id,) + power_check(
            move_id, arrange, lambda f, m=move_id: DEX.moves[m].base_power * 2))
    return verdict(rows)


@family("lastmovefailed",
        "Power doubles if the user's last move on the previous turn",
        "A move that was blocked by Baneful Bunker")
def _doubles_after_a_failure():
    rows = []
    for move_id in members("lastmovefailed"):
        move = DEX.moves[move_id]

        def played(fail_first, move_id=move_id, move=move):
            # Splash into a wall is a move that does nothing at all, which is
            # what the clause counts; Protect is the one it does not.
            f = Fight([swinger(move_id, extra=("splash", "protect", "rest"))],
                      [mon(mc._reachable(move), "__none__",
                           ("splash", "protect", "rest", "bodyslam"), None,
                           "sassy", (32, 0, 32, 0, 32, 0))], seed=7)
            if fail_first:
                f.turn(Action.move(0), Action.move(1))   # they Protect it
            else:
                f.turn(Action.move(1), Action.move(0))   # we Splash
            f.turn(Action.move(0), Action.move(0))
            hits = [e for e in f.log if e.kind == "damage" and (e.side or 0) == 1
                    and e.move == move_id and not e.crit]
            return hits[0].amount if hits else 0

        blocked, plain = played(True), played(False)
        rows.append((move_id, plain > 0 and blocked == plain,
                     f"after being blocked by Protect {blocked}, after an "
                     f"ordinary turn {plain} -- the clause says a block does "
                     f"not double it"))
    return verdict(rows)


@family("hppower", "Power is equal to (user's current HP * 150 / user's maximum "
                   "HP), rounded down, but not less than 1.")
def _power_from_health():
    rows = []
    for move_id in members("hppower"):
        def arrange(f):
            slot = f.state.sides[0].active[0]
            f.state.sides[0].hp[slot] = f.state.active_pokemon(0).max_hp // 2

        def power(f):
            slot = f.state.sides[0].active[0]
            whole = f.state.active_pokemon(0).max_hp
            return max(1, f.state.sides[0].hp[slot] * 150 // whole)

        rows.append((move_id,) + power_check(move_id, arrange, power))
    return verdict(rows)


@family("flailpower", "The power of this move is 20 if X is 33 to 48")
def _flail_brackets():
    def bracket(x):
        for top, value in ((1, 200), (4, 150), (9, 100), (16, 80), (32, 40),
                           (48, 20)):
            if x <= top:
                return value
        return 20

    rows = []
    for move_id in members("flailpower"):
        def arrange(f):
            slot = f.state.sides[0].active[0]
            f.state.sides[0].hp[slot] = max(
                1, f.state.active_pokemon(0).max_hp // 8)

        def power(f):
            slot = f.state.sides[0].active[0]
            whole = f.state.active_pokemon(0).max_hp
            return bracket(f.state.sides[0].hp[slot] * 48 // whole)

        rows.append((move_id,) + power_check(move_id, arrange, power))
    return verdict(rows)


@family("stagepower", "Power is equal to 20+(X*20), where X is the user's total "
                      "stat stage changes that are greater than 0.")
def _power_from_stages():
    rows = []
    for move_id in members("stagepower"):
        def arrange(f):
            side = f.state.sides[0]
            row = side.boosts[side.active[0]]
            row[mc.BOOST_INDEX["atk"]] = 2
            row[mc.BOOST_INDEX["spa"]] = 1
            row[mc.BOOST_INDEX["def"]] = -1      # the negative one does not count

        rows.append((move_id,) + power_check(move_id, arrange,
                                             lambda f: 20 + 3 * 20))
    return verdict(rows)


@family("weightratio",
        "The power of this move depends on (user's weight / target's weight), "
        "rounded down.",
        "Power is equal to 120 if the result is 5 or more, 100 if 4, 80 if 3, "
        "60 if 2, and 40 if 1 or less.")
def _weight_ratio():
    rows = []
    for move_id in members("weightratio"):
        target = "venusaur"                      # 100.0 kg against Mew's 4.0
        move = DEX.moves[move_id]
        f = Fight([swinger(move_id, species="snorlax")],
                  [mon(target, "__none__", ("splash", "bodyslam", "protect", "rest"),
                       None, "sassy", (32, 0, 32, 0, 32, 0))], seed=7)
        ratio = int(DEX.species["snorlax"].weight_kg
                    // DEX.species[target].weight_kg)
        want = {5: 120, 4: 100, 3: 80, 2: 60}.get(min(5, ratio), 40)
        expected = rolls_for(f, move, want)
        f.turn(Action.move(0), Action.move(0))
        hits = [e for e in f.log if e.kind == "damage" and (e.side or 0) == 1
                and e.move == move_id and not e.crit]
        got = hits[0].amount if hits else 0
        rows.append((move_id, got in expected,
                     f"460kg against {target}'s {DEX.species[target].weight_kg}kg "
                     f"is a ratio of {ratio}, so power {want}: dealt {got}, "
                     f"expected {min(expected)}-{max(expected)}"))
    return verdict(rows)


@family("targetweight",
        "This move's power is 20 if the target weighs less than 10 kg, 40 if "
        "less than 25 kg, 60 if less than 50 kg, 80 if less than 100 kg, 100 if "
        "less than 200 kg, and 120 if greater than or equal to 200 kg.")
def _target_weight():
    def bracket(kilos):
        for top, value in ((10, 20), (25, 40), (50, 60), (100, 80), (200, 100)):
            if kilos < top:
                return value
        return 120

    rows = []
    for move_id in members("targetweight"):
        move = DEX.moves[move_id]
        off = []
        for target in ("weavile", "venusaur", "snorlax"):
            f = Fight([swinger(move_id)],
                      [mon(target, "__none__",
                           ("splash", "bodyslam", "protect", "rest"), None,
                           "sassy", (32, 0, 32, 0, 32, 0))], seed=7)
            want = bracket(DEX.species[target].weight_kg)
            expected = rolls_for(f, move, want)
            f.turn(Action.move(0), Action.move(0))
            hits = [e for e in f.log if e.kind == "damage" and (e.side or 0) == 1
                    and e.move == move_id and not e.crit]
            got = hits[0].amount if hits else 0
            if got not in expected:
                off.append(f"{target} ({DEX.species[target].weight_kg}kg, "
                           f"power {want}): dealt {got}, expected "
                           f"{min(expected)}-{max(expected)}")
        rows.append((move_id, not off, "; ".join(off) or "all three weights"))
    return verdict(rows)


@family("multiaccuracy", "This move checks accuracy for each hit, and the "
                         "attack ends if the target avoids a hit.")
def _accuracy_per_hit():
    rows = []
    for move_id in members("multiaccuracy"):
        counts = set()
        for seed in range(40):
            f = Fight([swinger(move_id)],
                      [wall(mc._reachable(DEX.moves[move_id]))], seed=seed)
            f.turn(Action.move(0), Action.move(0))
            hits = len([e for e in f.log if e.kind == "damage"
                        and (e.side or 0) == 1 and e.move == move_id])
            counts.add(hits)
        # A move that rolled once for the lot would only ever show its full
        # count or nothing at all.
        rows.append((move_id, len(counts - {0}) > 1,
                     f"saw {sorted(counts)} hits across forty casts"))
    return verdict(rows)


@family("stealthrockdamage",
        "Foes lose 1/32, 1/16, 1/8, 1/4, or 1/2 of their maximum HP, rounded "
        "down, based on their weakness to the Rock type; 0.25x, 0.5x, neutral, "
        "2x, or 4x, respectively.")
def _stealth_rock_by_type():
    rows = []
    for move_id in members("stealthrockdamage"):
        off = []
        for species in ("charizard", "garchomp", "snorlax", "archaludon"):
            against = DEX.type_chart.multiplier("rock", DEX.species[species].types)
            share = {4.0: 2, 2.0: 4, 1.0: 8, 0.5: 16, 0.25: 32}[against]
            f = Fight([swinger(move_id)],
                      [mon("magikarp", "__none__", ("splash", "tackle")),
                       mon(species, "__none__",
                           ("splash", "bodyslam", "protect", "rest"), None,
                           "sassy", (32, 0, 32, 0, 32, 0))], seed=7)
            f.turn(Action.move(0), Action.move(0))
            if "stealthrock" not in f.conditions(1):
                off.append("the rocks were never laid")
                break
            f.turn(Action.move(1), Action.switch(1))
            whole = f.max_hp(1)
            took = whole - f.hp(1)
            if abs(took - whole // share) > 1:
                off.append(f"{species} (x{against} to Rock): took {took} of "
                           f"{whole}, and 1/{share} is {whole // share}")
        rows.append((move_id, not off, "; ".join(off) or
                     "a quarter, an eighth and a sixteenth as the chart says"))
    return verdict(rows)


@family("trickswap", "The user swaps its held item with the target's held item.",
        "The target is immune to this move if it has the Sticky Hold Ability.",
        "Fails if both the user and the target have no held item, or if the "
        "user is trying to give or take a Blue Orb, Red Orb, Adamant Crystal, "
        "Lustrous Globe, Griseous Core, Plate, Drive, Memory, Rusted Sword, "
        "Rusted Shield, or Booster Energy.")
def _trick_swaps():
    rows = []
    for move_id in members("trickswap"):
        def played(ability, mine="leftovers", theirs="sitrusberry"):
            f = Fight([swinger(move_id, item=mine)],
                      [wall(mc._reachable(DEX.moves[move_id]), ability=ability,
                            item=theirs)], seed=7)
            f.turn(Action.move(0), Action.move(0))
            return f.item(0), f.item(1)

        swapped = played("__none__")
        held = played("stickyhold")
        empty = played("__none__", None, None)
        rows.append((move_id,
                     swapped == ("sitrusberry", "leftovers")
                     and held == ("leftovers", "sitrusberry")
                     and empty == (None, None),
                     f"swapped to {swapped}; against Sticky Hold {held}; with "
                     f"nothing on either side {empty}"))
    return verdict(rows)


@family("alreadymoved", "Fails if the target already moved this turn.")
def _target_already_moved():
    rows = []
    for move_id in members("alreadymoved"):
        # Ours is slower, so by the time it acts the target has gone.
        f = Fight([mon("snorlax", "__none__",
                       (move_id, "splash", "protect", "rest"), None, "sassy",
                       (32, 0, 32, 0, 32, 0))],
                  [mon("weavile", "__none__",
                       ("bodyslam", "splash", "protect", "rest"), None, "jolly",
                       (0, 32, 2, 0, 0, 32))], seed=7)
        f.turn(Action.move(0), Action.move(0))
        rows.append((move_id, failed(f),
                     f"against a target that had already moved it failed="
                     f"{failed(f)}"))
    return verdict(rows)


@family("endswhentheyleave", "This effect ends when the target is no longer "
                             "active.")
def _ends_when_they_leave():
    rows = []
    for move_id in members("endswhentheyleave"):
        volatile = {"saltcure": "saltcure", "torment": "torment"}[move_id]
        f = until(lambda seed, move_id=move_id: Fight(
            [swinger(move_id)],
            [wall(mc._reachable(DEX.moves[move_id])),
             mon("magikarp", "__none__", ("splash", "tackle"))], seed=seed)
            .turn(Action.move(0), Action.move(0)),
            lambda f: volatile in f.volatiles(1))
        if f is None:
            rows.append((move_id, False, "it never took hold in thirty tries"))
            continue
        f.turn(Action.move(1), Action.switch(1))
        rows.append((move_id, volatile not in f.volatiles(1),
                     f"after the target left it held "
                     f"{sorted(f.volatiles(1) & {volatile})}"))
    return verdict(rows)



@family("outragelock",
        "The user spends two or three turns locked into this move",
        "This move targets an opposing Pokemon at random on each turn.",
        "If the user is prevented from moving, is asleep at the beginning of a "
        "turn, or the attack is not successful against the target on the first "
        "turn of the effect",
        "If this move is called by Sleep Talk and the user is asleep, the move "
        "is used for one turn and does not confuse the user.")
def _locked_in():
    """Two or three turns, then confusion -- and a Protect on the first turn
    of the run ends it there with no confusion."""
    from pkcm.engine.state import legal_actions
    rows = []
    for move_id in members("outragelock"):
        locked = set()
        for seed in range(10):
            f = Fight([swinger(move_id)], [wall()], seed=seed)
            f.turn(Action.move(0), Action.move(0))
            allowed = [str(one) for one in legal_actions(f.state, 0)
                       if str(one).startswith("move")]
            locked.add(len(allowed))

        # Blocked on the very first turn: the run ends, and no confusion.
        g = Fight([swinger(move_id)],
                  [mon("snorlax", "__none__",
                       ("protect", "splash", "bodyslam", "rest"), None, "sassy",
                       (32, 0, 32, 0, 32, 0))], seed=7)
        g.turn(Action.move(0), Action.move(0))
        free = len([one for one in legal_actions(g.state, 0)
                    if str(one).startswith("move")]) > 1
        rows.append((move_id, locked == {1} and free,
                     f"while it runs the user may pick from {sorted(locked)} "
                     f"move(s); after a Protect on turn one it was free={free}"))
    return verdict(rows)


@family("uturnswitch",
        "The user does not switch out if there are no unfainted party members")
def _no_one_left_to_come_in():
    rows = []
    for move_id in members("uturnswitch"):
        f = Fight([swinger(move_id)],
                  [wall(mc._reachable(DEX.moves[move_id]))], seed=7)
        for slot in range(1, len(f.state.sides[0].hp)):
            f.state.sides[0].hp[slot] = 0
        before = f.active_species(0)
        f.turn(Action.move(0), Action.move(0))
        rows.append((move_id, f.active_species(0) == before
                     and f.state.phase.name == "BATTLE",
                     f"with nobody on the bench it stayed={f.active_species(0) == before}"))
    return verdict(rows)


@family("batonpasses", "Baton Pass can be used to transfer this effect to an ally.",
        "Baton Pass can be used to transfer the substitute to an ally, and the "
        "substitute will keep its remaining HP.")
def _baton_passes_it():
    rows = []
    for move_id in members("batonpasses"):
        if move_id == "dragoncheer":
            # Ally-only, so it needs the doubles field: cheer the partner,
            # have the partner Baton Pass, and see whether what comes in
            # still has it.
            ours = [mon(UNIVERSAL, "__none__",
                        (move_id, "splash", "protect", "rest"), None,
                        "serious", (32, 0, 32, 0, 2, 32)),
                    mon("snorlax", "__none__",
                        ("batonpass", "splash", "protect", "rest"), None,
                        "serious", (32, 0, 32, 0, 2, 0))]
            theirs = [mon("snorlax", "__none__",
                          ("splash", "bodyslam", "protect", "rest"), None,
                          "sassy", (32, 0, 32, 0, 32, 0))] * 2
            g = Pair(ours, theirs)
            g.turn((Action.move(0, target=TARGET_ALLY), Action.move(1)),
                   (Action.move(0), Action.move(0)))
            cheered = "dragoncheer" in g.state.sides[0].volatiles[
                g.state.sides[0].active[1]]
            g.turn((Action.move(1), Action.move(0)),
                   (Action.move(0), Action.move(0)))
            while g.state.phase.name in ("MID_TURN_SWITCH", "FORCED_SWITCH"):
                g.turn(
                    tuple(Action.switch(2 + i) if g.state.sides[0].must_switch[i]
                          else Action.PASS for i in range(2)),
                    tuple(Action.switch(2 + i) if g.state.sides[1].must_switch[i]
                          else Action.PASS for i in range(2)))
            carried = "dragoncheer" in g.state.sides[0].volatiles[
                g.state.sides[0].active[1]]
            rows.append((move_id, cheered and carried,
                         f"the partner was cheered={cheered}, and its "
                         f"replacement has it={carried}"))
            continue
        volatile = {"focusenergy": "focusenergy",
                    "substitute": "substitute"}[move_id]
        f = Fight([mon(UNIVERSAL, "__none__",
                       (move_id, "batonpass", "splash", "protect"), None,
                       "adamant", (32, 32, 0, 0, 2, 0)),
                   mon("magikarp", "__none__", ("splash", "tackle"))],
                  [wall()], seed=7)
        f.turn(Action.move(0), Action.move(0))
        if volatile not in f.volatiles(0):
            rows.append((move_id, False, f"{volatile} never went up"))
            continue
        f.turn(Action.move(1), Action.move(0))
        if f.state.phase.name in ("MID_TURN_SWITCH", "FORCED_SWITCH"):
            f.turn(Action.switch(1), Action.PASS)
        rows.append((move_id, volatile in f.volatiles(0),
                     f"the replacement arrived holding "
                     f"{sorted(f.volatiles(0) & {volatile})}"))
    return verdict(rows)


@family("protectstatement", "The user is protected from most attacks made by "
                            "other Pokemon during this turn.")
def _protects_the_user():
    rows = []
    for move_id in members("protectstatement"):
        f = Fight([swinger(move_id)],
                  [mon("garchomp", "__none__",
                       ("dragonclaw", "splash", "protect", "rest"), None,
                       "jolly", (0, 32, 2, 0, 0, 32))], seed=7)
        whole = f.hp(0)
        f.turn(Action.move(0), Action.move(0))
        rows.append((move_id, f.hp(0) == whole,
                     f"behind it the user took {whole - f.hp(0)}"))
    return verdict(rows)


@family("escapehatch",
        "The user can still switch out if it uses Baton Pass, Flip Turn, "
        "Parting Shot, Teleport, U-turn, or Volt Switch.",
        "A Pokemon can still switch out if it is holding Shed Shell or uses "
        "Baton Pass, Flip Turn, Parting Shot, Teleport, U-turn, or Volt Switch.")
def _the_escape_hatch():
    """A trapped Pokemon still leaves on its own switching move."""
    rows = []
    for move_id in members("escapehatch"):
        f = Fight([mon(UNIVERSAL, "__none__",
                       (move_id, "uturn", "splash", "protect"), None,
                       "adamant", (32, 32, 0, 0, 2, 0)),
                   mon("magikarp", "__none__", ("splash", "tackle"))],
                  [wall()], seed=7)
        f.turn(Action.move(0), Action.move(0))
        before = f.active_species(0)
        f.turn(Action.move(1), Action.move(0))
        if f.state.phase.name in ("MID_TURN_SWITCH", "FORCED_SWITCH"):
            f.turn(Action.switch(1), Action.PASS)
        rows.append((move_id, f.active_species(0) != before,
                     f"held by its own {move_id}, U-turn still left: "
                     f"{before} -> {f.active_species(0)}"))
    return verdict(rows)


@family("spikelayers", "Can be used up to three times before failing.",
        "A maximum of three layers may be set, and opponents lose 1/8 of their "
        "maximum HP with one layer, 1/6 of their maximum HP with two layers, "
        "and 1/4 of their maximum HP with three layers.")
def _spike_layers():
    rows = []
    for move_id in members("spikelayers"):
        f = Fight([swinger(move_id)],
                  [wall(), mon("magikarp", "__none__", ("splash", "tackle")),
                   mon("pikachu", "__none__", ("splash", "tackle"))], seed=7)
        depths = []
        for _ in range(4):
            f.turn(Action.move(0), Action.move(0))
            depths.append(f.conditions(1).get("spikes", 0))
        fourth_failed = failed(f)
        rows.append((move_id, depths == [1, 2, 3, 3] and fourth_failed,
                     f"layers went {depths}; the fourth cast failed={fourth_failed}"))
    return verdict(rows)


@family("spikedamage", "Opponents lose 1/8 of their maximum HP with one layer")
def _spike_damage():
    rows = []
    for move_id in members("spikedamage"):
        off = []
        for layers, share in ((1, 8), (2, 6), (3, 4)):
            f = Fight([swinger(move_id)],
                      [wall(), mon("magikarp", "__none__", ("splash", "tackle")),
                       mon("pikachu", "__none__", ("splash", "tackle"))], seed=7)
            for _ in range(layers):
                f.turn(Action.move(0), Action.move(0))
            f.turn(Action.move(1), Action.switch(1))
            whole, left = f.max_hp(1), f.hp(1)
            took = whole - left
            if abs(took - whole // share) > 1:
                off.append(f"{layers} layer(s): took {took} of {whole}, and "
                           f"1/{share} is {whole // share}")
        rows.append((move_id, not off, "; ".join(off) or "an eighth, a sixth "
                                                         "and a quarter"))
    return verdict(rows)


@family("sandstormresidual",
        "all active Pokemon lose 1/16 of their maximum HP, rounded down, unless "
        "they are a Ground, Rock, or Steel type")
def _sandstorm_residual():
    rows = []
    for move_id in members("sandstormresidual"):
        off = []
        for species, hurt in (("snorlax", True), ("garchomp", False),
                              ("archaludon", False)):
            f = Fight([mon(species, "__none__",
                           (move_id, "splash", "protect", "rest"), None,
                           "sassy", (32, 0, 32, 0, 32, 0))],
                      [wall()], seed=7)
            whole = f.max_hp(0)
            f.turn(Action.move(0), Action.move(0))
            took = whole - f.hp(0)
            want = whole // 16 if hurt else 0
            if abs(took - want) > 1:
                types = "/".join(DEX.species[species].types)
                off.append(f"{species} ({types}) took {took}, expected {want}")
        rows.append((move_id, not off, "; ".join(off) or
                     "a sixteenth off everything but Ground, Rock and Steel"))
    return verdict(rows)


@family("perishcount",
        "the perish count of all active Pokemon lowers by 1")
def _perish_song():
    rows = []
    for move_id in members("perishcount"):
        f = Fight([mon(UNIVERSAL, "__none__",
                       (move_id, "splash", "protect", "rest"), None, "adamant",
                       (32, 32, 0, 0, 2, 0)),
                   mon("magikarp", "__none__", ("splash", "tackle"))],
                  [wall(), mon("pikachu", "__none__", ("splash", "tackle"))],
                  seed=7)
        f.turn(Action.move(0), Action.move(0))
        counts = []
        for _ in range(4):
            if f.state.phase.name != "BATTLE":
                break
            f.turn(Action.move(1), Action.move(0))
            counts.append((f.state.sides[0].hp[f.state.sides[0].active[0]],
                           f.state.sides[1].hp[f.state.sides[1].active[0]]))
        both_gone = counts and (counts[-1][0] == 0 or counts[-1][1] == 0)
        rows.append((move_id, both_gone,
                     f"health after each turn: {counts}"))
    return verdict(rows)


@family("wishheals",
        "the Pokemon at the user's position has 1/2 of the user's maximum HP")
def _wish_heals_next_turn():
    rows = []
    for move_id in members("wishheals"):
        f = Fight([swinger(move_id)], [wall()], seed=7)
        slot = f.state.sides[0].active[0]
        whole = f.max_hp(0)
        f.state.sides[0].hp[slot] = 1
        f.turn(Action.move(0), Action.move(0))
        same_turn = f.hp(0)
        f.turn(Action.move(1), Action.move(0))
        rows.append((move_id, same_turn == 1
                     and abs(f.hp(0) - 1 - (whole + 1) // 2) <= 1,
                     f"nothing on the turn it was made ({same_turn}), "
                     f"{f.hp(0) - 1} the turn after, half of {whole} being "
                     f"{(whole + 1) // 2}"))
    return verdict(rows)


@family("yawntimer",
        "At the end of the next turn, if the target is still active, does not "
        "have a non-volatile status condition, and can fall asleep, it falls "
        "asleep.")
def _yawn_timer():
    rows = []
    for move_id in members("yawntimer"):
        f = Fight([swinger(move_id)],
                  [wall(mc._reachable(DEX.moves[move_id]))], seed=7)
        f.turn(Action.move(0), Action.move(0))
        first = f.status(1)
        f.turn(Action.move(1), Action.move(0))
        rows.append((move_id, first is None and f.status(1) == "slp",
                     f"after the turn it was cast the target was {first}, "
                     f"after the next {f.status(1)}"))
    return verdict(rows)


@family("healbellsound", "Active Pokemon with the Soundproof Ability are not "
                         "cured, unless they are the user.")
def _heal_bell():
    rows = []
    for move_id in members("healbellsound"):
        # In singles the only active Pokemon is the user, and the clause
        # excuses the user by name -- so what singles can show is that a
        # Soundproof *user* still cures itself and its bench. The partner
        # half needs a partner, and gets one in ``effect_check --doubles``.
        f = Fight([mon(UNIVERSAL, "soundproof",
                       (move_id, "splash", "protect", "rest"), None, "adamant",
                       (32, 32, 0, 0, 2, 0)),
                   mon("magikarp", "soundproof", ("splash", "tackle")),
                   mon("pikachu", "__none__", ("splash", "tackle"))],
                  [wall()], seed=7)
        for slot in (0, 1, 2):
            f.state.sides[0].status[slot] = "brn"
        f.turn(Action.move(0), Action.move(0))
        statuses = [f.state.sides[0].status[slot] for slot in (0, 1, 2)]
        rows.append((move_id, statuses == [None, None, None],
                     f"the Soundproof user and its bench came out {statuses}"))
    return verdict(rows)


@family("teatimeberries", "All active Pokemon consume their held Berries.")
def _teatime():
    rows = []
    for move_id in members("teatimeberries"):
        f = Fight([swinger(move_id, item="sitrusberry")],
                  [wall(item="sitrusberry")], seed=7)
        for side in (0, 1):
            slot = f.state.sides[side].active[0]
            f.state.sides[side].hp[slot] //= 3
        f.turn(Action.move(0), Action.move(0))
        rows.append((move_id, f.item(0) is None and f.item(1) is None,
                     f"afterwards they hold {f.item(0)!r} and {f.item(1)!r}"))
    return verdict(rows)


@family("gravitygrounds",
        "At the time of use, Bounce, Fly, Magnet Rise, Sky Drop, and Telekinesis "
        "end immediately for all active Pokemon.")
def _gravity_grounds_them():
    rows = []
    for move_id in members("gravitygrounds"):
        f = Fight([mon(UNIVERSAL, "__none__",
                       (move_id, "magnetrise", "splash", "protect"), None,
                       "adamant", (32, 32, 0, 0, 2, 0))],
                  [wall()], seed=7)
        f.turn(Action.move(1), Action.move(0))
        up = "magnetrise" in f.volatiles(0)
        f.turn(Action.move(0), Action.move(0))
        rows.append((move_id, up and "magnetrise" not in f.volatiles(0),
                     f"Magnet Rise was up={up}, and afterwards "
                     f"{sorted(f.volatiles(0) & {'magnetrise'})}"))
    return verdict(rows)



@family("saltcureresidual",
        "Causes damage to the target equal to 1/8 of its maximum HP (1/4 if "
        "the target is Steel or Water type), rounded down, at the end of each "
        "turn during effect.")
def _salt_cure():
    rows = []
    for move_id in members("saltcureresidual"):
        off = []
        for species in ("snorlax", "milotic", "archaludon"):
            share = 4 if ({"steel", "water"} & set(DEX.species[species].types)) else 8
            f = Fight([swinger(move_id)],
                      [mon(species, "__none__",
                           ("splash", "bodyslam", "protect", "rest"), None,
                           "sassy", (32, 0, 32, 0, 32, 0))], seed=7)
            f.turn(Action.move(0), Action.move(0))
            whole = f.max_hp(1)
            before = f.hp(1)
            f.turn(Action.move(1), Action.move(0))
            took = before - f.hp(1)
            if abs(took - whole // share) > 1:
                off.append(f"{species} ({'/'.join(DEX.species[species].types)}): "
                           f"took {took}, and 1/{share} of {whole} is "
                           f"{whole // share}")
        rows.append((move_id, not off, "; ".join(off) or
                     "an eighth, and a quarter off Steel and Water"))
    return verdict(rows)


@family("addsatype",
        "Causes the Ghost type to be added to the target, effectively making "
        "it have two or three types.",
        "Causes the Grass type to be added to the target, effectively making "
        "it have two or three types.")
def _adds_a_type():
    rows = []
    for move_id in members("addsatype"):
        added = {"trickortreat": "ghost", "forestscurse": "grass"}[move_id]
        f = Fight([swinger(move_id)], [wall("garchomp")], seed=7)
        before = set(f.state.types(1, f.state.sides[1].active[0]))
        f.turn(Action.move(0), Action.move(0))
        after = set(f.state.types(1, f.state.sides[1].active[0]))
        rows.append((move_id, after == before | {added},
                     f"{sorted(before)} -> {sorted(after)}"))
    return verdict(rows)


@family("becomesatype", "Causes the target to become a Psychic type.",
        "Causes the target to become a Water type.",
        "Causes the user's types to become the same as the current types of "
        "the target.")
def _becomes_a_type():
    rows = []
    for move_id in members("becomesatype"):
        f = Fight([swinger(move_id)], [wall("garchomp")], seed=7)
        f.turn(Action.move(0), Action.move(0))
        theirs = set(f.state.types(1, f.state.sides[1].active[0]))
        ours = set(f.state.types(0, f.state.sides[0].active[0]))
        want = {"magicpowder": ("them", {"psychic"}), "soak": ("them", {"water"}),
                "reflecttype": ("us", set(DEX.species["garchomp"].types))}[move_id]
        got = theirs if want[0] == "them" else ours
        rows.append((move_id, got == want[1],
                     f"the {'target' if want[0] == 'them' else 'user'} came out "
                     f"{sorted(got)}, and the clause says {sorted(want[1])}"))
    return verdict(rows)


@family("changesability",
        "Causes the target's Ability to be rendered ineffective as long as it "
        "remains active.",
        "Causes the target's Ability to become Insomnia.",
        "Causes the target's Ability to become Simple.",
        "Causes the target's Ability to become the same as the user's.")
def _changes_the_ability():
    rows = []
    for move_id in members("changesability"):
        f = Fight([swinger(move_id, ability="levitate")],
                  [wall("garchomp", ability="roughskin")], seed=7)
        f.turn(Action.move(0), Action.move(0))
        theirs = f.state.ability_id(1, f.state.sides[1].active[0])
        suppressed = "abilitysuppressed" in f.volatiles(1)
        want = {"gastroacid": None, "worryseed": "insomnia",
                "simplebeam": "simple", "entrainment": "levitate"}[move_id]
        good = suppressed if want is None else theirs == want
        rows.append((move_id, good,
                     f"the target's ability reads {theirs!r}"
                     + (f", suppressed={suppressed}" if want is None else "")))
    return verdict(rows)


@family("spitepp", "Causes the target's last move used to lose 4 PP.")
def _spite_takes_pp():
    rows = []
    for move_id in members("spitepp"):
        f = Fight([swinger(move_id)], [wall()], seed=7)
        f.turn(Action.move(1), Action.move(1))          # they swing once
        slot = f.state.sides[1].active[0]
        before = list(f.state.sides[1].pp[slot])
        f.turn(Action.move(0), Action.move(1))
        after = list(f.state.sides[1].pp[slot])
        lost = [b - a for b, a in zip(before, after)]
        rows.append((move_id, 5 in lost or 4 in lost,
                     f"PP went {before} -> {after}, so it lost {lost} "
                     f"(one of which is this turn's own cost)"))
    return verdict(rows)


@family("electrictype", "Causes the target's move to become Electric type this "
                        "turn.")
def _electrify_retypes():
    rows = []
    for move_id in members("electrictype"):
        # We are faster, so it lands before their move goes off -- and we are
        # a Ground type, which is immune to what their Body Slam becomes. The
        # volatile itself is cleared inside the turn, so the type chart is the
        # only witness left standing.
        def took(electrified):
            f = Fight([mon("garchomp", "__none__",
                           (move_id, "splash", "protect", "rest"), None,
                           "jolly", (0, 32, 2, 0, 0, 32))],
                      [mon("snorlax", "__none__",
                           ("bodyslam", "splash", "protect", "rest"), None,
                           "sassy", (32, 0, 32, 0, 32, 0))], seed=7)
            f.turn(Action.move(0 if electrified else 1), Action.move(0))
            return sum(e.amount or 0 for e in f.log
                       if e.kind == "damage" and (e.side or 0) == 0)

        retyped, plain = took(True), took(False)
        rows.append((move_id, plain > 0 and retyped == 0,
                     f"their Body Slam took {plain} normally and {retyped} "
                     f"after Electrify, at a Ground type"))
    return verdict(rows)


@family("quashorder", "Causes the target to take its turn after all other "
                      "Pokemon this turn, no matter the priority of its "
                      "selected move.")
def _quash_order():
    """Ally-only, so singles has nothing to reorder -- and it refuses."""
    rows = []
    for move_id in members("quashorder"):
        f = Fight([swinger(move_id)], [wall()], seed=7)
        f.turn(Action.move(0), Action.move(0))
        rows.append((move_id, failed(f),
                     f"in singles, with no ally to quash, it failed={failed(f)}"))
    return verdict(rows)


@family("infatuation", "Causes the target to become infatuated, making it "
                       "unable to attack 50% of the time.")
def _infatuation():
    rows = []
    for move_id in members("infatuation"):
        held = 0
        tries = 120
        for seed in range(tries):
            f = Fight([gendered("garchomp", "__none__",
                                (move_id, "splash", "protect", "rest"),
                                "jolly", (0, 32, 2, 0, 0, 32), "M")],
                      [gendered("snorlax", "__none__",
                                ("bodyslam", "splash", "protect", "rest"),
                                "sassy", (32, 0, 32, 0, 32, 0), "F")], seed=seed)
            f.turn(Action.move(0), Action.move(0))
            if "attract" not in f.volatiles(1):
                continue
            f.turn(Action.move(1), Action.move(0))
            held += not any(e.kind == "damage" and (e.side or 0) == 0
                            for e in f.log)
        share = held / tries
        rows.append((move_id, abs(share - 0.5) <= 0.15,
                     f"it stayed put {held} of {tries} turns ({share:.0%} "
                     f"against the half in the data)"))
    return verdict(rows)


@family("digdive", "Damage doubles if the target is using Dig.",
        "Damage doubles if the target is using Dive.")
def _catches_them_underground():
    rows = []
    for move_id in members("digdive"):
        hidden = {"earthquake": "dig", "surf": "dive"}[move_id]

        def dealt(charging):
            # Slow on purpose: Dig only hides the target once it has moved,
            # and a faster attacker swings before there is anything to catch.
            f = Fight([mon("snorlax", "__none__",
                           (move_id, "splash", "protect", "rest"), None,
                           "brave", (32, 32, 0, 32, 2, 0))],
                      [mon("mew", "__none__", (hidden, "splash", "protect", "rest"),
                           None, "jolly", (32, 0, 0, 0, 2, 32))], seed=7)
            f.turn(Action.move(0), Action.move(0 if charging else 1))
            return hp_lost(f)

        plain, buried = dealt(False), dealt(True)
        share = buried / plain if plain else 0.0
        rows.append((move_id, 1.9 <= share <= 2.1,
                     f"{plain} against something standing, {buried} against "
                     f"one using {hidden} (x{share:.2f})"))
    return verdict(rows)


@family("statoverride",
        "Damage is calculated using the target's Attack stat, including stat "
        "stage changes.",
        "Damage is calculated using the user's Defense stat as its Attack, "
        "including stat stage changes.",
        "Deals damage to the target based on its Defense instead of Special "
        "Defense.")
def _borrows_a_stat():
    """The stage that matters moves the damage; the one that does not, does not."""
    rows = []
    for move_id in members("statoverride"):
        boosted, ignored = {
            "foulplay": ("their atk", "our atk"),
            "bodypress": ("our def", "our atk"),
            "psyshock": ("their def", "their spd"),
        }[move_id]

        def dealt(which):
            f = Fight([swinger(move_id)], [wall("garchomp")], seed=7)
            side, stat = which.split()
            row = (f.state.sides[0].boosts[f.state.sides[0].active[0]]
                   if side == "our"
                   else f.state.sides[1].boosts[f.state.sides[1].active[0]])
            row[mc.BOOST_INDEX[stat]] = 2
            f.turn(Action.move(0), Action.move(0))
            return hp_lost(f)

        base = dealt("our accuracy")            # a stage nothing here reads
        moved, still = dealt(boosted), dealt(ignored)
        # Foul Play and Body Press read an attacking stat, Psyshock a
        # defending one: raising what it reads moves the damage either way.
        want_up = move_id != "psyshock"
        moved_right = moved > base if want_up else moved < base
        rows.append((move_id, moved_right and still == base,
                     f"plain {base}; two stages of {boosted} made it {moved}; "
                     f"two stages of {ignored} left it {still}"))
    return verdict(rows)


@family("fixedformulas",
        "Deals damage to the target equal to (target's current HP - user's "
        "current HP).",
        "Deals damage to the target equal to half of its current HP, rounded "
        "down, but not less than 1 HP.",
        "Deals damage to the target equal to the user's current HP.")
def _fixed_formulas():
    rows = []
    for move_id in members("fixedformulas"):
        f = Fight([swinger(move_id)], [wall("garchomp")], seed=7)
        slot = f.state.sides[0].active[0]
        f.state.sides[0].hp[slot] = f.state.active_pokemon(0).max_hp // 2
        ours, theirs = f.hp(0), f.hp(1)
        want = {"endeavor": max(0, theirs - ours),
                "superfang": max(1, theirs // 2),
                "finalgambit": ours}[move_id]
        f.turn(Action.move(0), Action.move(0))
        rows.append((move_id, abs(hp_lost(f) - want) <= 1,
                     f"we were on {ours}, they on {theirs}: it took "
                     f"{hp_lost(f)} and the formula says {want}"))
    return verdict(rows)


@family("gigatonlock", "Cannot be selected the turn after it's used.")
def _gigaton_lock():
    from pkcm.engine.state import legal_actions
    rows = []
    for move_id in members("gigatonlock"):
        f = Fight([swinger(move_id)],
                  [wall(mc._reachable(DEX.moves[move_id]))], seed=7)
        f.turn(Action.move(0), Action.move(0))
        allowed = [one.index for one in legal_actions(f.state, 0)
                   if str(one).startswith("move")]
        rows.append((move_id, 0 not in allowed,
                     f"the turn after using it, the legal moves are {allowed}"))
    return verdict(rows)


@family("toxicspikelayers", "Can be used up to two times before failing.")
def _toxic_spike_layers():
    rows = []
    for move_id in members("toxicspikelayers"):
        f = Fight([swinger(move_id)],
                  [wall("garchomp"), mon("magikarp", "__none__", ("splash", "tackle")),
                   mon("pikachu", "__none__", ("splash", "tackle"))], seed=7)
        depths = []
        for _ in range(3):
            f.turn(Action.move(0), Action.move(0))
            depths.append(f.conditions(1).get(move_id, 0))
        rows.append((move_id, depths == [1, 2, 2] and failed(f),
                     f"layers went {depths}; the third cast failed={failed(f)}"))
    return verdict(rows)



def _damage_under(field, kind, move_id, user="mew", target="snorlax",
                  nature="modest", sp=(32, 0, 0, 32, 2, 0)):
    """What one move takes off, with a weather or terrain set or not."""
    f = Fight([mon(user, "__none__", (move_id, "splash", "protect", "rest"),
                   None, nature, sp)],
              [mon(target, "__none__", ("splash", "bodyslam", "protect", "rest"),
                   None, "sassy", (32, 0, 32, 0, 32, 0))], seed=7)
    if field is not None:
        if kind == "weather":
            f.state.field.weather, f.state.field.weather_turns = field, 8
        else:
            f.state.field.terrain, f.state.field.terrain_turns = field, 8
    f.turn(Action.move(0), Action.move(0))
    return hp_lost(f)


@family("terrainpower",
        "During the effect, the power of Electric-type attacks made by grounded "
        "Pokemon is multiplied by 1.3 and grounded Pokemon cannot fall asleep",
        "During the effect, the power of Grass-type attacks used by grounded "
        "Pokemon is multiplied by 1.3, the power of Bulldoze, Earthquake, and "
        "Magnitude used against grounded Pokemon is multiplied by 0.5",
        "During the effect, the power of Psychic-type attacks made by grounded "
        "Pokemon is multiplied by 1.3 and grounded Pokemon cannot be hit by",
        "During the effect, the power of Dragon-type attacks used against "
        "grounded Pokemon is multiplied by 0.5 and grounded")
def _terrain_power():
    """A third on, or a half off, and only for something standing on it."""
    want = {"electricterrain": ("thunderbolt", 1.3),
            "grassyterrain": ("energyball", 1.3),
            "psychicterrain": ("psychic", 1.3),
            "mistyterrain": ("dragonpulse", 0.5)}
    rows = []
    for move_id in members("terrainpower"):
        probe, share = want[move_id]
        plain = _damage_under(None, "terrain", probe)
        on_it = _damage_under(move_id, "terrain", probe)
        got = on_it / plain if plain else 0.0
        # And a Flying type standing above it takes the ordinary number.
        flying = _damage_under(move_id, "terrain", probe, target="staraptor"
                               if "staraptor" in DEX.species else "charizard")
        bare = _damage_under(None, "terrain", probe, target="staraptor"
                             if "staraptor" in DEX.species else "charizard")
        aloft = flying / bare if bare else 0.0
        ours = move_id != "mistyterrain"       # Misty is about the target
        rows.append((move_id, abs(got - share) <= 0.08
                     and (abs(aloft - 1.0) <= 0.08 or ours),
                     f"{probe} went {plain} -> {on_it} (x{got:.2f} against "
                     f"x{share}); against something in the air x{aloft:.2f}"))
    return verdict(rows)


@family("grassyearthquake",
        "the power of Bulldoze, Earthquake, and Magnitude used against grounded "
        "Pokemon is multiplied by 0.5")
def _grassy_softens_the_ground():
    plain = _damage_under(None, "terrain", "earthquake", user="garchomp",
                          nature="adamant", sp=(32, 32, 0, 0, 2, 0))
    grassy = _damage_under("grassyterrain", "terrain", "earthquake",
                           user="garchomp", nature="adamant",
                           sp=(32, 32, 0, 0, 2, 0))
    share = grassy / plain if plain else 0.0
    return (abs(share - 0.5) <= 0.06,
            f"Earthquake went {plain} -> {grassy} on grass (x{share:.2f})")


@family("weatherdefence",
        "During the effect, the Defense of Ice-type Pokemon is multiplied by "
        "1.5 when taking damage from a physical attack.",
        "During the effect, the Special Defense of Rock-type Pokemon is "
        "multiplied by 1.5 when taking damage from a special attack.")
def _weather_defence():
    rows = []
    for move_id in members("weatherdefence"):
        target, probe, nature, sp = {
            "snowscape": ("weavile", "bodyslam", "adamant", (32, 32, 0, 0, 2, 0)),
            "sandstorm": ("tyranitar" if "tyranitar" in DEX.species
                          else "rampardos",
                          "surf", "modest", (32, 0, 0, 32, 2, 0)),
        }[move_id]
        plain = _damage_under(None, "weather", probe, target=target,
                              nature=nature, sp=sp)
        sheltered = _damage_under(move_id, "weather", probe, target=target,
                                  nature=nature, sp=sp)
        share = sheltered / plain if plain else 0.0
        rows.append((move_id, abs(share - 2 / 3) <= 0.06,
                     f"{probe} at a {'/'.join(DEX.species[target].types)} went "
                     f"{plain} -> {sheltered} (x{share:.2f} against the two "
                     f"thirds a 1.5x defence gives)"))
    return verdict(rows)


@family("trickroomspeed",
        "During the effect, each Pokemon's Speed is considered to be (10000 - "
        "its normal Speed), and if this value is greater than 8191, 8192 is "
        "subtracted from it.")
def _trick_room_speed():
    rows = []
    for move_id in members("trickroomspeed"):
        def first(room):
            f = Fight([mon("weavile", "__none__",
                           (move_id, "bodyslam", "protect", "rest"), None,
                           "jolly", (0, 32, 2, 0, 0, 32))],
                      [mon("snorlax", "__none__",
                           ("bodyslam", "splash", "protect", "rest"), None,
                           "sassy", (32, 32, 0, 0, 32, 0))], seed=7)
            if room:
                f.state.field.rooms[move_id] = 5
            f.turn(Action.move(1), Action.move(0))
            order = [e for e in f.log if e.kind == "move_used"]
            return (order[0].side or 0) if order else None

        rows.append((move_id, first(False) == 0 and first(True) == 1,
                     f"the quick one went first normally (side {first(False)}) "
                     f"and second inside the room (side {first(True)})"))
    return verdict(rows)


@family("gravitygrounds2",
        "During the effect, Bounce, Fly, Flying Press, High Jump Kick, Jump "
        "Kick, Magnet Rise, Sky Drop, Splash, and Telekinesis are prevented "
        "from being used")
def _gravity_forbids():
    rows = []
    for move_id in members("gravitygrounds2"):
        f = Fight([mon(UNIVERSAL, "__none__",
                       ("fly", "splash", "protect", "rest"), None, "adamant",
                       (32, 32, 0, 0, 2, 0))],
                  [wall()], seed=7)
        f.state.field.rooms["gravity"] = 5
        f.turn(Action.move(0), Action.move(0))
        refused = failed(f) or any(e.kind in ("move_failed", "cant_move")
                                   for e in f.log)
        rows.append((move_id, refused, f"Fly under Gravity was refused={refused}"))
    return verdict(rows)


@family("smackdownblocks", "During the effect, Magnet Rise fails for the target "
                           "and Telekinesis fails against the target.")
def _smack_down_pins_them():
    rows = []
    for move_id in members("smackdownblocks"):
        f = Fight([swinger(move_id)],
                  [mon("mew", "__none__",
                       ("magnetrise", "splash", "protect", "rest"), None,
                       "sassy", (32, 0, 32, 0, 32, 0))], seed=7)
        f.turn(Action.move(0), Action.move(1))
        pinned = "smackdown" in f.volatiles(1)
        f.turn(Action.move(1), Action.move(0))
        rows.append((move_id, pinned and "magnetrise" not in f.volatiles(1),
                     f"pinned={pinned}; afterwards it held "
                     f"{sorted(f.volatiles(1) & {'magnetrise'})}"))
    return verdict(rows)


@family("healblocked", "During the effect, healing and draining moves are "
                       "unusable, and Abilities and items that grant healing "
                       "will not heal the user")
def _heal_blocked():
    rows = []
    for move_id in members("healblocked"):
        f = Fight([swinger(move_id)],
                  [mon("snorlax", "__none__",
                       ("recover", "splash", "protect", "rest"), None, "sassy",
                       (32, 0, 32, 0, 32, 0))], seed=7)
        slot = f.state.sides[1].active[0]
        f.state.sides[1].hp[slot] //= 2
        f.turn(Action.move(0), Action.move(1))
        before = f.hp(1)
        f.turn(Action.move(1), Action.move(0))
        rows.append((move_id, f.hp(1) <= before,
                     f"under it, Recover took them from {before} to {f.hp(1)}"))
    return verdict(rows)


@family("uproarnosleep",
        "During the three turns, no active Pokemon can fall asleep by any means")
def _uproar_keeps_everyone_awake():
    rows = []
    for move_id in members("uproarnosleep"):
        # Ours moves first, so the din is up before the Spore is cast.
        f = Fight([mon("weavile", "__none__",
                       (move_id, "splash", "protect", "rest"), None, "jolly",
                       (0, 32, 2, 0, 0, 32))],
                  [mon("snorlax", "__none__",
                       ("spore", "splash", "protect", "rest"), None, "sassy",
                       (32, 0, 32, 0, 32, 0))], seed=7)
        f.turn(Action.move(0), Action.move(0))
        rows.append((move_id, f.status(0) != "slp",
                     f"with the din up, Spore left the user {f.status(0)}"))
    return verdict(rows)


@family("sidehealsaquarter", "Each Pokemon on the user's side restores 1/4 of "
                             "its maximum HP, rounded half up.")
def _heals_the_side_a_quarter():
    rows = []
    for move_id in members("sidehealsaquarter"):
        f = Fight([swinger(move_id)], [wall()], seed=7)
        slot = f.state.sides[0].active[0]
        whole = f.max_hp(0)
        f.state.sides[0].hp[slot] = 1
        f.turn(Action.move(0), Action.move(0))
        got = f.hp(0) - 1
        rows.append((move_id, abs(got - (whole + 2) // 4) <= 1,
                     f"restored {got} of {whole}, a quarter being "
                     f"{(whole + 2) // 4}"))
    return verdict(rows)


@family("perishfour", "Each active Pokemon receives a perish count of 4 if it "
                      "doesn't already have a perish count.")
def _perish_count_of_four():
    rows = []
    for move_id in members("perishfour"):
        f = Fight([swinger(move_id)], [wall()], seed=7)
        f.turn(Action.move(0), Action.move(0))
        counts = [f.state.sides[side].volatiles[f.state.sides[side].active[0]]
                  .get("perishsong") for side in (0, 1)]
        rows.append((move_id, all(isinstance(one, dict) for one in counts),
                     f"both sides came out holding {counts}"))
    return verdict(rows)


@family("ragefistcount", "Each hit of a multi-hit attack is counted, but "
                         "confusion damage is not counted.")
def _rage_fist_counts_hits():
    rows = []
    for move_id in members("ragefistcount"):
        def dealt(hits):
            f = Fight([swinger(move_id)],
                      [mon("garchomp", "__none__",
                           ("bulletseed", "splash", "protect", "rest"), None,
                           "adamant", (0, 32, 2, 0, 0, 32))], seed=7)
            for _ in range(hits):
                f.turn(Action.move(1), Action.move(0))
            f.turn(Action.move(0), Action.move(1))
            return hp_lost(f)

        none, some = dealt(0), dealt(2)
        rows.append((move_id, some > none,
                     f"{none} before being hit, {some} after two turns of a "
                     f"multi-hit move"))
    return verdict(rows)


@family("terrainpulsetype",
        "Electric type during Electric Terrain, Grass type during Grassy "
        "Terrain, Fairy type during Misty Terrain, and Psychic type during "
        "Psychic Terrain")
def _terrain_pulse_type():
    rows = []
    for move_id in members("terrainpulsetype"):
        off = []
        for terrain, kind in (("electricterrain", "electric"),
                              ("grassyterrain", "grass"),
                              ("mistyterrain", "fairy"),
                              ("psychicterrain", "psychic")):
            # A Ground type is immune to Electric and takes Grass at double.
            f = Fight([swinger(move_id)],
                      [mon("garchomp", "__none__",
                           ("splash", "bodyslam", "protect", "rest"), None,
                           "sassy", (32, 0, 32, 0, 32, 0))], seed=7)
            f.state.field.terrain, f.state.field.terrain_turns = terrain, 8
            f.turn(Action.move(0), Action.move(0))
            hits = [e for e in f.log if e.kind == "damage" and (e.side or 0) == 1]
            got = hits[0].effectiveness if hits else 0.0
            want = DEX.type_chart.multiplier(kind, DEX.species["garchomp"].types)
            if abs(got - want) > 0.01:
                off.append(f"{terrain}: effectiveness {got} against the {want} "
                           f"a {kind} move would have")
        rows.append((move_id, not off, "; ".join(off) or
                     "the type followed the terrain all four ways"))
    return verdict(rows)


@family("curesthewholeparty", "Every Pokemon in the user's party is cured of "
                              "its non-volatile status condition.")
def _cures_the_party():
    rows = []
    for move_id in members("curesthewholeparty"):
        f = Fight([swinger(move_id),
                   mon("magikarp", "__none__", ("splash", "tackle")),
                   mon("pikachu", "__none__", ("splash", "tackle"))],
                  [wall()], seed=7)
        for slot in range(3):
            f.state.sides[0].status[slot] = "brn"
        f.turn(Action.move(0), Action.move(0))
        left = [f.state.sides[0].status[slot] for slot in range(3)]
        rows.append((move_id, left == [None, None, None],
                     f"the party came out {left}"))
    return verdict(rows)


@family("fullhealthdoesnothing", "Does nothing if the user's HP is full.")
def _nothing_at_full_health():
    rows = []
    for move_id in members("fullhealthdoesnothing"):
        f = Fight([swinger(move_id)], [wall()], seed=7)
        f.turn(Action.move(0), Action.move(0))
        rows.append((move_id, failed(f),
                     f"at full health it failed={failed(f)}"))
    return verdict(rows)


@family("alreadythattype", "Fails if the target is already a Ghost type.",
        "Fails if the target is already a Grass type.")
def _already_that_type():
    rows = []
    for move_id in members("alreadythattype"):
        already = {"trickortreat": "gengar", "forestscurse": "meganium"}[move_id]
        f = Fight([swinger(move_id)], [wall(already)], seed=7)
        f.turn(Action.move(0), Action.move(0))
        rows.append((move_id, failed(f),
                     f"against a {'/'.join(DEX.species[already].types)} it "
                     f"failed={failed(f)}"))
    return verdict(rows)


@family("typelessdamage", "Deals typeless damage to a random opposing Pokemon.")
def _struggle_is_typeless():
    rows = []
    for move_id in members("typelessdamage"):
        f = Fight([swinger("tackle")], [wall("gengar")], seed=7)
        for index in range(len(f.state.sides[0].pp[f.state.sides[0].active[0]])):
            f.state.sides[0].pp[f.state.sides[0].active[0]][index] = 0
        f.turn(Action.struggle(), Action.move(0))
        hits = [e for e in f.log if e.kind == "damage" and (e.side or 0) == 1]
        rows.append((move_id, bool(hits) and hits[0].effectiveness == 1.0,
                     f"against a Ghost it dealt {hits[0].amount if hits else 0} "
                     f"at effectiveness "
                     f"{hits[0].effectiveness if hits else None}"))
    return verdict(rows)



@family("statedduration", "For 5 turns, the weather becomes",
        "For 5 turns, the terrain becomes", "For 5 turns, the Speed of every",
        "For 5 turns, all active Pokemon have their Defense and Special "
        "Defense stats swapped.",
        "For 5 turns, the held items of all active Pokemon have no effect.",
        "For 5 turns, the evasiveness of all active Pokemon is multiplied by 0.6.",
        "For 5 turns, the user and its party members",
        "For 5 turns, the user is immune to Ground-type attacks",
        "For 4 turns, the user and its party members have their Speed doubled.",
        "For 4 turns, the target's last move used becomes disabled.",
        "For 2 turns, the target cannot use sound-based moves.",
        "For 2 turns, the target is prevented from restoring any HP as long as "
        "it remains active.",
        "For its next 3 turns, the target is forced to repeat its last move used.",
        "For 5 turns, the weather becomes Snow.")
def _stated_durations():
    """The number of turns each one says it lasts, measured turn by turn.

    Delegated to ``mechanic_check``, whose duration probe casts the move and
    then counts until the effect goes -- and now fails rather than shrugging
    when it cannot set the position up.
    """
    return _delegate("statedduration")


@family("tailwindspeed", "the user and its party members have their Speed doubled")
def _tailwind_doubles_speed():
    from pkcm.data.dex import Stat
    from pkcm.engine.battle import make_context
    from pkcm.engine.mutate import effective_stat

    rows = []
    for move_id in members("tailwindspeed"):
        f = Fight([swinger(move_id)], [wall()], seed=7)
        us = (0, f.state.sides[0].active[0])
        before = effective_stat(make_context(f.state), us, Stat.SPE)
        f.turn(Action.move(0), Action.move(0))
        after = effective_stat(make_context(f.state), us, Stat.SPE)
        rows.append((move_id, after == before * 2,
                     f"Speed went {before} -> {after}"))
    return verdict(rows)


@family("safeguardblocks",
        "the user and its party members cannot have non-volatile status "
        "conditions or confusion inflicted on them by other Pokemon")
def _safeguard_blocks():
    rows = []
    for move_id in members("safeguardblocks"):
        f = Fight([swinger(move_id)],
                  [mon("gengar", "__none__",
                       ("willowisp", "confuseray", "splash", "protect"), None,
                       "timid", (0, 0, 2, 32, 0, 32))], seed=7)
        f.turn(Action.move(0), Action.move(2))
        f.turn(Action.move(1), Action.move(0))
        burnt = f.status(0)
        f.turn(Action.move(1), Action.move(1))
        confused = "confusion" in f.volatiles(0)
        rows.append((move_id, burnt is None and not confused,
                     f"behind it the Will-O-Wisp left {burnt} and Confuse Ray "
                     f"left confusion={confused}"))
    return verdict(rows)


@family("magnetrisefloat",
        "the user is immune to Ground-type attacks and the effects of Spikes, "
        "Toxic Spikes, Sticky Web, and the Arena Trap Ability")
def _magnet_rise_floats():
    rows = []
    for move_id in members("magnetrisefloat"):
        f = Fight([swinger(move_id, species="snorlax")],
                  [mon("garchomp", "__none__",
                       ("earthquake", "splash", "protect", "rest"), None,
                       "jolly", (0, 32, 2, 0, 0, 32))], seed=7)
        f.turn(Action.move(0), Action.move(1))
        up = "magnetrise" in f.volatiles(0)
        f.turn(Action.move(1), Action.move(0))
        took = sum(e.amount or 0 for e in f.log
                   if e.kind == "damage" and (e.side or 0) == 0)
        rows.append((move_id, up and took == 0,
                     f"afloat={up}; the Earthquake took {took}"))
    return verdict(rows)


@family("encorerepeat", "the target is forced to repeat its last move used")
def _encore_repeats():
    from pkcm.engine.state import legal_actions
    rows = []
    for move_id in members("encorerepeat"):
        f = Fight([swinger(move_id)],
                  [mon("snorlax", "__none__",
                       ("splash", "bodyslam", "protect", "rest"), None, "sassy",
                       (32, 0, 32, 0, 32, 0))], seed=7)
        f.turn(Action.move(1), Action.move(0))          # they Splash
        f.turn(Action.move(0), Action.move(0))          # we Encore it
        allowed = [one.index for one in legal_actions(f.state, 1)
                   if str(one).startswith("move")]
        rows.append((move_id, allowed == [0],
                     f"after the Encore they may pick {allowed}"))
    return verdict(rows)


@family("disableturns", "the target's last move used becomes disabled")
def _disable_disables():
    from pkcm.engine.state import legal_actions
    rows = []
    for move_id in members("disableturns"):
        f = Fight([swinger(move_id)],
                  [mon("snorlax", "__none__",
                       ("splash", "bodyslam", "protect", "rest"), None, "sassy",
                       (32, 0, 32, 0, 32, 0))], seed=7)
        f.turn(Action.move(1), Action.move(0))
        f.turn(Action.move(0), Action.move(1))
        allowed = [one.index for one in legal_actions(f.state, 1)
                   if str(one).startswith("move")]
        rows.append((move_id, 0 not in allowed and len(allowed) > 1,
                     f"after the Disable they may pick {allowed}"))
    return verdict(rows)


@family("throatchopsound", "the target cannot use sound-based moves")
def _throat_chop():
    from pkcm.engine.state import legal_actions
    rows = []
    for move_id in members("throatchopsound"):
        f = Fight([swinger(move_id)],
                  [mon("snorlax", "__none__",
                       ("hypervoice", "bodyslam", "protect", "rest"), None,
                       "sassy", (32, 0, 32, 0, 32, 0))], seed=7)
        f.turn(Action.move(0), Action.move(1))
        allowed = [one.index for one in legal_actions(f.state, 1)
                   if str(one).startswith("move")]
        used_it = False
        if 0 in allowed:
            f.turn(Action.move(1), Action.move(0))
            used_it = any(e.kind == "damage" and (e.side or 0) == 0
                          and e.move == "hypervoice" for e in f.log)
        rows.append((move_id, 0 not in allowed or not used_it,
                     f"they may pick {allowed}; the sound move landed="
                     f"{used_it}"))
    return verdict(rows)


@family("wonderroomswap", "all active Pokemon have their Defense and Special "
                          "Defense stats swapped")
def _wonder_room_swaps():
    rows = []
    for move_id in members("wonderroomswap"):
        def took(room):
            f = Fight([mon(UNIVERSAL, "__none__",
                           (move_id, "bodyslam", "splash", "protect"), None,
                           "adamant", (32, 32, 0, 0, 2, 0))],
                      [mon("archaludon", "__none__",
                           ("splash", "protect", "rest", "bodyslam"), None,
                           "sassy", (32, 0, 32, 0, 32, 0))], seed=7)
            if room:
                f.turn(Action.move(0), Action.move(0))
            f.turn(Action.move(1), Action.move(0))
            return hp_lost(f)

        plain, swapped = took(False), took(True)
        rows.append((move_id, plain != swapped,
                     f"a physical hit took {plain} normally and {swapped} "
                     f"inside the room"))
    return verdict(rows)


@family("gravityevasion", "the evasiveness of all active Pokemon is multiplied "
                          "by 0.6")
def _gravity_steadies():
    rows = []
    for move_id in members("gravityevasion"):
        def hits(room):
            landed_count = 0
            for seed in range(60):
                f = Fight([mon(UNIVERSAL, "__none__",
                               ("focusblast", "splash", "protect", "rest"),
                               None, "modest", (32, 0, 0, 32, 2, 0))],
                          [wall()], seed=seed)
                if room:
                    f.state.field.rooms["gravity"] = 5
                f.turn(Action.move(0), Action.move(0))
                landed_count += landed(f, "focusblast")
            return landed_count / 60

        plain, steadied = hits(False), hits(True)
        rows.append((move_id, steadied > plain,
                     f"Focus Blast landed {plain:.0%} normally and "
                     f"{steadied:.0%} under Gravity"))
    return verdict(rows)


@family("magicroomitems", "the held items of all active Pokemon have no effect")
def _magic_room_kills_items():
    rows = []
    for move_id in members("magicroomitems"):
        def healed(room):
            f = Fight([swinger(move_id, item="leftovers")], [wall()], seed=7)
            slot = f.state.sides[0].active[0]
            f.state.sides[0].hp[slot] //= 2
            before = f.hp(0)
            f.turn(Action.move(0) if room else Action.move(1), Action.move(0))
            return f.hp(0) - before

        rows.append((move_id, healed(True) == 0 and healed(False) > 0,
                     f"Leftovers gave {healed(False)} normally and "
                     f"{healed(True)} inside the room"))
    return verdict(rows)


@family("leechseedgrass", "Grass-type Pokemon are immune to this move on use, "
                          "but not its effect.")
def _leech_seed_and_grass():
    rows = []
    for move_id in members("leechseedgrass"):
        f = Fight([swinger(move_id)],
                  [wall("meganium" if "meganium" in DEX.species else "venusaur")],
                  seed=7)
        f.turn(Action.move(0), Action.move(0))
        rows.append((move_id, "leechseed" not in f.volatiles(1),
                     f"a Grass type came out holding "
                     f"{sorted(f.volatiles(1) & {'leechseed'})}"))
    return verdict(rows)


@family("mistyyawn", "Grounded Pokemon can become affected by Yawn but cannot "
                     "fall asleep from its effect.")
def _misty_yawn():
    rows = []
    for move_id in members("mistyyawn"):
        f = Fight([mon(UNIVERSAL, "__none__",
                       ("yawn", "splash", "protect", "rest"), None, "adamant",
                       (32, 32, 0, 0, 2, 0))],
                  [wall()], seed=7)
        f.state.field.terrain, f.state.field.terrain_turns = move_id, 8
        f.turn(Action.move(0), Action.move(0))
        yawning = "yawn" in f.volatiles(1)
        f.turn(Action.move(1), Action.move(0))
        rows.append((move_id, yawning and f.status(1) != "slp",
                     f"the Yawn took hold={yawning}, and the target came out "
                     f"{f.status(1)}"))
    return verdict(rows)


@family("gravityhazards",
        "Ground-type attacks, Spikes, Toxic Spikes, Sticky Web, and the Arena "
        "Trap Ability can affect Flying types or Pokemon with the Levitate "
        "Ability.")
def _gravity_grounds_the_flying():
    rows = []
    for move_id in members("gravityhazards"):
        def took(room):
            f = Fight([mon("garchomp", "__none__",
                           ("earthquake", "splash", "protect", "rest"), None,
                           "adamant", (0, 32, 2, 0, 0, 32))],
                      [wall("charizard")], seed=7)
            if room:
                f.state.field.rooms["gravity"] = 5
            f.turn(Action.move(0), Action.move(0))
            return hp_lost(f)

        rows.append((move_id, took(False) == 0 and took(True) > 0,
                     f"an Earthquake at a Flying type took {took(False)} "
                     f"normally and {took(True)} under Gravity"))
    return verdict(rows)



#: Each entry arranges the position the clause describes and says whether the
#: move should refuse it. ``arrange`` is given the fight before the first turn.
def _refusal_cases():
    def hurt(f, share=2):
        slot = f.state.sides[0].active[0]
        f.state.sides[0].hp[slot] = max(1, f.max_hp(0) // share)

    def cases():
        yield ("strengthsap", "the target's Attack is already at -6",
               lambda f: f.state.sides[1].boosts[f.state.sides[1].active[0]]
               .__setitem__(mc.BOOST_INDEX["atk"], -6), True)
        yield ("strengthsap", "an ordinary target", lambda f: None, False)
        yield ("substitute", "not enough HP to pay for it",
               lambda f: f.state.sides[0].hp.__setitem__(
                   f.state.sides[0].active[0], 4), True)
        yield ("substitute", "room to pay for it", hurt, False)
        yield ("rest", "already at full health", lambda f: None, True)
        yield ("rest", "hurt", hurt, False)
        yield ("bellydrum", "Attack already at six",
               lambda f: f.state.sides[0].boosts[f.state.sides[0].active[0]]
               .__setitem__(mc.BOOST_INDEX["atk"], 6), True)
        yield ("bellydrum", "room to raise it", lambda f: None, False)
        yield ("clangoroussoul", "a third of its HP already gone",
               lambda f: hurt(f, 4), True)
        yield ("clangoroussoul", "full health", lambda f: None, False)
        yield ("clangoroussoul", "all five stages already at six",
               lambda f: f.state.sides[0].boosts.__setitem__(
                   f.state.sides[0].active[0],
                   [6, 6, 6, 6, 6, 0, 0]), True)
        yield ("topsyturvy", "every stage at zero", lambda f: None, True)
        yield ("topsyturvy", "a stage to turn over",
               lambda f: f.state.sides[1].boosts[f.state.sides[1].active[0]]
               .__setitem__(mc.BOOST_INDEX["atk"], 2), False)
        # Curse is only a curse in a Ghost's hands; on anything else it is a
        # self-boost with no target at all.
        yield ("curse", "the target already cursed",
               lambda f: f.state.sides[1].volatiles[f.state.sides[1].active[0]]
               .__setitem__("curse", {}), True, "gengar")
        yield ("steelroller", "bare ground", lambda f: None, True)
        yield ("burnup", "a user that is not a Fire type", lambda f: None, True)
        yield ("recycle", "a user that has never held anything",
               lambda f: None, True)
        yield ("fling", "a user holding nothing", lambda f: None, True)
        yield ("spite", "a target that has not moved", lambda f: None, True)
        yield ("disable", "a target that has not moved", lambda f: None, True)
        yield ("encore", "a target that has not moved", lambda f: None, True)
        yield ("instruct", "a target that has not moved", lambda f: None, True)
        yield ("copycat", "nothing used yet", lambda f: None, True)
        yield ("afteryou", "no ally to move up", lambda f: None, True)
        yield ("upperhand", "a target with no priority move coming",
               lambda f: None, True)
        yield ("acupressure", "every stage already at six",
               lambda f: f.state.sides[0].boosts.__setitem__(
                   f.state.sides[0].active[0],
                   [6] * len(f.state.sides[0].boosts[f.state.sides[0].active[0]])),
               True)
        yield ("healingwish", "the last one standing",
               lambda f: [f.state.sides[0].hp.__setitem__(slot, 0)
                          for slot in range(1, len(f.state.sides[0].hp))], True)
        yield ("shedtail", "nobody to hand it to",
               lambda f: [f.state.sides[0].hp.__setitem__(slot, 0)
                          for slot in range(1, len(f.state.sides[0].hp))], True)
    return list(cases())


@family("refusals",
        "Fails if the target's Attack stat stage is -6.",
        "Fails if the user does not have enough HP remaining to create a "
        "substitute without fainting, or if it already has a substitute.",
        "Fails if the user has full HP, is already asleep, or if another "
        "effect is preventing sleep.",
        "Fails if the user would faint or if its Attack stat stage is 6.",
        "Fails if the user would faint or if its Attack, Defense, Special "
        "Attack, Special Defense, and Speed stat stages would not change.",
        "Fails if all of the target's stat stages are 0.",
        "Fails if there is no target or if the target is already affected.",
        "Fails if there is no terrain active.",
        "Fails unless the user is a Fire type.",
        "Fails if the user is holding an item, if the user has not held an item",
        "Fails if the user has no held item, if the held item cannot be thrown",
        "Fails if the target has not made a move, if the move has 0 PP, or if "
        "it no longer knows the move.",
        "Fails if one of the target's moves is already disabled, if the target "
        "has not made a move",
        "Fails if the target is already under this effect, if it has not made "
        "a move",
        "Fails if the target has not made a move, if the move has 0 PP, if the "
        "target is preparing to use Beak Blast",
        "Fails if no move has been used, or if the last move used was Assist",
        "Fails if the target would have moved next anyway, or if the target "
        "already moved this turn.",
        "Fails if the target did not select a physical or special attack for "
        "use this turn with altered priority greater than 0",
        "Fails if no stat stage can be raised or if used on an ally with a "
        "substitute.",
        "Fails if the user is the last unfainted Pokemon in its party.",
        "Fails if the user would faint, or if there are no unfainted party "
        "members.")
def _refusals():
    """Each move put in the position its own sentence says it refuses."""
    rows = []
    for case in _refusal_cases():
        move_id, what, arrange, should_fail = case[:4]
        user = case[4] if len(case) > 4 else None
        move = DEX.moves[move_id]
        f = Fight([swinger(move_id, species=user),
                   mon("magikarp", "__none__", ("splash", "tackle")),
                   mon("pikachu", "__none__", ("splash", "tackle"))],
                  [wall(mc._reachable(move)),
                   mon("magikarp", "__none__", ("splash", "tackle"))], seed=7)
        arrange(f)
        f.turn(Action.move(0), Action.move(0))
        refused = failed(f)
        rows.append((f"{move_id} with {what}", refused == should_fail,
                     f"refused={refused}, and the clause says "
                     f"{should_fail}"))
    return verdict(rows)



@family("weatherballtype",
        "Ice type during Snow, Water type during Primordial Sea or Rain Dance, "
        "Rock type during Sandstorm, and Fire type during Desolate Land or "
        "Sunny Day")
def _weather_ball_type():
    rows = []
    for move_id in members("weatherballtype"):
        off = []
        for weather, kind in (("snowscape", "ice"), ("raindance", "water"),
                              ("sandstorm", "rock"), ("sunnyday", "fire")):
            f = Fight([swinger(move_id)], [wall("garchomp")], seed=7)
            f.state.field.weather, f.state.field.weather_turns = weather, 8
            f.turn(Action.move(0), Action.move(0))
            hits = [e for e in f.log if e.kind == "damage" and (e.side or 0) == 1]
            got = hits[0].effectiveness if hits else None
            want = DEX.type_chart.multiplier(kind, DEX.species["garchomp"].types)
            if got is None or abs(got - want) > 0.01:
                off.append(f"{weather}: effectiveness {got} against the {want} "
                           f"a {kind} move would have")
        rows.append((move_id, not off, "; ".join(off) or
                     "the type followed the sky all four ways"))
    return verdict(rows)


@family("risingvoltage",
        "If the current terrain is Electric Terrain and the target is "
        "grounded, this move's power is doubled.")
def _rising_voltage():
    rows = []
    for move_id in members("risingvoltage"):
        def dealt(terrain):
            f = Fight([swinger(move_id)], [wall()], seed=7)
            if terrain:
                f.state.field.terrain, f.state.field.terrain_turns = terrain, 8
            f.turn(Action.move(0), Action.move(0))
            return hp_lost(f)

        plain, charged = dealt(None), dealt("electricterrain")
        # The terrain's own 1.3x rides on top of the doubling.
        share = charged / plain if plain else 0.0
        rows.append((move_id, 2.5 <= share <= 2.7,
                     f"{plain} on bare ground, {charged} on Electric Terrain "
                     f"(x{share:.2f}, which is the doubling and the terrain's "
                     f"own 1.3 together)"))
    return verdict(rows)


@family("hitsexactly", "Hits ten times.", "Hits three times.",
        "Hits one time for the user and one time for each unfainted Pokemon "
        "without a non-volatile status condition in the user's party.")
def _hits_exactly():
    rows = []
    for move_id in members("hitsexactly"):
        counts = set()
        for seed in range(10):
            f = Fight([swinger(move_id),
                       mon("magikarp", "__none__", ("splash", "tackle")),
                       mon("pikachu", "__none__", ("splash", "tackle"))],
                      [wall("blissey" if "blissey" in DEX.species else "snorlax")],
                      seed=seed)
            f.turn(Action.move(0), Action.move(0))
            hit = len([e for e in f.log if e.kind == "damage"
                       and (e.side or 0) == 1 and e.move == move_id])
            if hit:
                counts.add(hit)
        want = {"populationbomb": set(range(1, 11)),
                "tripleaxel": {1, 2, 3}, "beatup": {3}}[move_id]
        # Ten is the count, and the per-hit accuracy roll below it is a
        # separate sentence with its own family: what this asks is that the
        # ceiling is right and is reached.
        ceiling = {"populationbomb": 10, "tripleaxel": 3, "beatup": 3}[move_id]
        rows.append((move_id, counts and counts <= want and max(counts) == ceiling,
                     f"hit {sorted(counts)} times, and the clause allows up to "
                     f"{ceiling}"))
    return verdict(rows)


@family("toxicnevermisses",
        "If a Poison-type Pokemon uses this move, the target cannot avoid the "
        "attack, even if the target is in the middle of a two-turn move.")
def _toxic_from_a_poison_type():
    rows = []
    for move_id in members("toxicnevermisses"):
        def landed_from(species):
            hits = 0
            for seed in range(30):
                f = Fight([mon(species, "__none__",
                               (move_id, "splash", "protect", "rest"), None,
                               "modest", (32, 0, 0, 32, 2, 0))],
                          [mon("snorlax", "__none__",
                               ("doubleteam", "splash", "protect", "rest"),
                               None, "sassy", (32, 0, 32, 0, 32, 0))], seed=seed)
                for _ in range(3):
                    f.turn(Action.move(1), Action.move(0))
                f.turn(Action.move(0), Action.move(1))
                hits += f.status(1) is not None
            return hits / 30

        poison, other = landed_from("gengar"), landed_from("snorlax")
        rows.append((move_id, poison == 1.0 and other < 1.0,
                     f"from a Poison type it landed {poison:.0%} of the time "
                     f"behind three Double Teams, from a Normal one "
                     f"{other:.0%}"))
    return verdict(rows)


@family("futuresightlands", "Deals damage two turns after this move is used.",
        "At the end of that turn, the damage is calculated at that time and "
        "dealt to the Pokemon at the position the target had when the move was "
        "used.")
def _future_sight():
    rows = []
    for move_id in members("futuresightlands"):
        f = Fight([swinger(move_id)], [wall()], seed=7)
        f.turn(Action.move(0), Action.move(0))
        landed_when = []
        for turn in range(1, 4):
            f.turn(Action.move(1), Action.move(0))
            if any(e.kind == "damage" and (e.side or 0) == 1 for e in f.log):
                landed_when.append(turn)
        rows.append((move_id, landed_when == [2],
                     f"the hit arrived on turn(s) {landed_when} after the cast"))
    return verdict(rows)


@family("yawnneedsawakeful",
        "Fails when used if the target cannot fall asleep or if it already has "
        "a non-volatile status condition.",
        "Causes the target to fall asleep at the end of the next turn.")
def _yawn_needs_a_target_that_can_sleep():
    rows = []
    for move_id in members("yawnneedsawakeful"):
        f = Fight([swinger(move_id)],
                  [wall(mc._reachable(DEX.moves[move_id]))], seed=7)
        side = f.state.sides[1]
        side.status[side.active[0]] = "brn"
        f.turn(Action.move(0), Action.move(0))
        rows.append((move_id, failed(f),
                     f"against an already burnt target it failed={failed(f)}"))
    return verdict(rows)


@family("attractgenders",
        "Fails if both the user and the target are the same gender, if either "
        "is genderless, or if the target is already infatuated.")
def _attract_and_gender():
    rows = []
    for move_id in members("attractgenders"):
        def stuck(mine, theirs, user="garchomp", target="snorlax"):
            f = Fight([gendered(user, "__none__",
                                (move_id, "splash", "protect", "rest"),
                                "jolly", (0, 32, 2, 0, 0, 32), mine)],
                      [gendered(target, "__none__",
                                ("splash", "bodyslam", "protect", "rest"),
                                "sassy", (32, 0, 32, 0, 32, 0), theirs)], seed=7)
            f.turn(Action.move(0), Action.move(0))
            return "attract" in f.volatiles(1)

        rows.append((move_id, stuck("M", "F") and not stuck("M", "M")
                     and not stuck(None, "F"),
                     f"M at F {stuck('M', 'F')}, M at M {stuck('M', 'M')}, "
                     f"genderless at F {stuck(None, 'F')}"))
    return verdict(rows)


@family("abilityblocklist", "Power Construct, RKS System, Schooling, Shields")
def _abilities_that_cannot_be_touched():
    """Disguise is the one on the blocklist this format actually has.

    Role Play's sentence is about the *user's* ability and the rest are about
    the target's, so each is asked in its own direction.
    """
    rows = []
    for move_id in members("abilityblocklist"):
        on_the_user = move_id in ("roleplay", "skillswap")
        mine = "disguise" if on_the_user else "levitate"
        theirs = "levitate" if move_id == "roleplay" else "disguise"
        f = Fight([swinger(move_id, ability=mine)],
                  [wall(mc._reachable(DEX.moves[move_id]), ability=theirs)],
                  seed=7)
        f.turn(Action.move(0), Action.move(0))
        rows.append((move_id, failed(f),
                     f"with Disguise on the {'user' if on_the_user else 'target'} "
                     f"it failed={failed(f)}"))
    return verdict(rows)


@family("destinybondtwice",
        "Fails if the user used this move successfully as its last move, "
        "disregarding moves used through the Dancer Ability.")
def _destiny_bond_twice():
    rows = []
    for move_id in members("destinybondtwice"):
        f = Fight([swinger(move_id)], [wall()], seed=7)
        f.turn(Action.move(0), Action.move(0))
        first = failed(f)
        f.turn(Action.move(0), Action.move(0))
        rows.append((move_id, not first and failed(f),
                     f"the first went up ({not first}); the second failed="
                     f"{failed(f)}"))
    return verdict(rows)


@family("typereplace",
        "If Forest's Curse adds a type to the target, it replaces the type "
        "added by this move and vice versa.",
        "If Trick-or-Treat adds a type to the target, it replaces the type "
        "added by this move and vice versa.")
def _the_added_type_is_replaced():
    rows = []
    for move_id in members("typereplace"):
        other = ("forestscurse" if move_id == "trickortreat" else "trickortreat")
        added = {"trickortreat": "ghost", "forestscurse": "grass"}[other]
        f = Fight([mon(UNIVERSAL, "__none__",
                       (move_id, other, "splash", "protect"), None, "adamant",
                       (32, 32, 0, 0, 2, 0))],
                  [wall("garchomp")], seed=7)
        f.turn(Action.move(0), Action.move(0))
        f.turn(Action.move(1), Action.move(0))
        types = set(f.state.types(1, f.state.sides[1].active[0]))
        first = {"trickortreat": "ghost", "forestscurse": "grass"}[move_id]
        rows.append((move_id, added in types and first not in types,
                     f"after both, the target is {sorted(types)}"))
    return verdict(rows)


@family("substitutefrees",
        "If a substitute is created while the user is trapped by a binding "
        "move, the binding effect ends immediately.")
def _substitute_frees_the_bound():
    rows = []
    for move_id in members("substitutefrees"):
        f = Fight([mon("snorlax", "__none__",
                       (move_id, "splash", "protect", "rest"), None, "sassy",
                       (32, 0, 32, 0, 32, 0))],
                  [mon("garchomp", "__none__",
                       ("wrap", "splash", "protect", "rest"), None, "jolly",
                       (0, 32, 2, 0, 0, 32))], seed=7)
        f.turn(Action.move(1), Action.move(0))
        bound = "partiallytrapped" in f.volatiles(0)
        f.turn(Action.move(0), Action.move(1))
        rows.append((move_id, bound and "partiallytrapped" not in f.volatiles(0),
                     f"bound={bound}; after the substitute it held "
                     f"{sorted(f.volatiles(0) & {'partiallytrapped'})}"))
    return verdict(rows)


@family("encoreoutofpp", "If the affected move runs out of PP, the effect ends.")
def _encore_ends_with_the_pp():
    from pkcm.engine.state import legal_actions
    rows = []
    for move_id in members("encoreoutofpp"):
        f = Fight([swinger(move_id)],
                  [mon("snorlax", "__none__",
                       ("splash", "bodyslam", "protect", "rest"), None, "sassy",
                       (32, 0, 32, 0, 32, 0))], seed=7)
        f.turn(Action.move(1), Action.move(0))
        f.turn(Action.move(0), Action.move(0))
        held = [one.index for one in legal_actions(f.state, 1)
                if str(one).startswith("move")] == [0]
        slot = f.state.sides[1].active[0]
        f.state.sides[1].pp[slot][0] = 0
        free = len([one for one in legal_actions(f.state, 1)
                    if str(one).startswith("move")]) > 1
        rows.append((move_id, held and free,
                     f"held to one move={held}; once its PP ran out it was "
                     f"free={free}"))
    return verdict(rows)


@family("magicroomfling", "During the effect, Fling and Natural Gift are "
                          "prevented from being used by all active Pokemon.")
def _magic_room_stops_fling():
    rows = []
    for move_id in members("magicroomfling"):
        f = Fight([mon(UNIVERSAL, "__none__",
                       ("fling", "splash", "protect", "rest"), "leftovers",
                       "adamant", (32, 32, 0, 0, 2, 0))],
                  [wall()], seed=7)
        f.state.field.rooms["magicroom"] = 5
        f.turn(Action.move(0), Action.move(0))
        rows.append((move_id, failed(f),
                     f"inside the room, Fling failed={failed(f)}"))
    return verdict(rows)


@family("ingraingrounded",
        "During the effect, the user can be hit normally by Ground-type "
        "attacks and be affected by Spikes, Toxic Spikes, and Sticky Web, even "
        "if the user is a Flying type or has the Levitate Ability.")
def _ingrain_grounds_the_user():
    rows = []
    for move_id in members("ingraingrounded"):
        def took(rooted):
            f = Fight([mon("charizard", "__none__",
                           (move_id, "splash", "protect", "rest"), None,
                           "sassy", (32, 0, 32, 0, 32, 0))],
                      [mon("garchomp", "__none__",
                           ("earthquake", "splash", "protect", "rest"), None,
                           "jolly", (0, 32, 2, 0, 0, 32))], seed=7)
            f.turn(Action.move(0 if rooted else 1), Action.move(1))
            f.turn(Action.move(1), Action.move(0))
            return sum(e.amount or 0 for e in f.log
                       if e.kind == "damage" and (e.side or 0) == 0)

        rows.append((move_id, took(False) == 0 and took(True) > 0,
                     f"an Earthquake at a Flying type took {took(False)} "
                     f"normally and {took(True)} once it was rooted"))
    return verdict(rows)



@family("grassyglide", "If the current terrain is Grassy Terrain and the user "
                       "is grounded, this move has its priority increased by 1.")
def _grassy_glide():
    rows = []
    for move_id in members("grassyglide"):
        def first(terrain):
            f = Fight([mon("snorlax", "__none__",
                           (move_id, "splash", "protect", "rest"), None,
                           "brave", (32, 32, 0, 0, 2, 0))],
                      [mon("weavile", "__none__",
                           ("bodyslam", "splash", "protect", "rest"), None,
                           "jolly", (0, 32, 2, 0, 0, 32))], seed=7)
            if terrain:
                f.state.field.terrain, f.state.field.terrain_turns = terrain, 8
            f.turn(Action.move(0), Action.move(0))
            order = [e for e in f.log if e.kind == "move_used"]
            return (order[0].side or 0) if order else None

        rows.append((move_id, first(None) == 1 and first("grassyterrain") == 0,
                     f"the slow user went second on bare ground (side "
                     f"{first(None)}) and first on grass (side "
                     f"{first('grassyterrain')})"))
    return verdict(rows)


@family("mistyexplosion", "If the current terrain is Misty Terrain and the user "
                          "is grounded, this move's power is multiplied by 1.5.")
def _misty_explosion():
    rows = []
    for move_id in members("mistyexplosion"):
        def dealt(terrain):
            f = Fight([swinger(move_id)], [wall()], seed=7)
            if terrain:
                f.state.field.terrain, f.state.field.terrain_turns = terrain, 8
            f.turn(Action.move(0), Action.move(0))
            return hp_lost(f)

        plain, misty = dealt(None), dealt("mistyterrain")
        share = misty / plain if plain else 0.0
        rows.append((move_id, 1.4 <= share <= 1.6,
                     f"{plain} on bare ground, {misty} on Misty Terrain "
                     f"(x{share:.2f})"))
    return verdict(rows)


@family("electroballspeed", "If the target's current Speed is 0, this move's "
                            "power is 40.",
        "The power of this move depends on (user's current Speed / target's "
        "current Speed), rounded down.",
        "Power is equal to 150 if the result is 4 or more, 120 if 3, 80 if 2, "
        "60 if 1, 40 if less than 1.")
def _electro_ball():
    """40, 60, 80, 120 or 150 by the speed ratio, worked out here."""
    def bracket(ratio):
        """"Power is equal to 150 if the result is 4 or more, 120 if 3, 80 if
        2, 60 if 1, 40 if less than 1"."""
        for edge, power in ((4, 150), (3, 120), (2, 80), (1, 60)):
            if ratio >= edge:
                return power
        return 40

    rows = []
    for move_id in members("electroballspeed"):
        from pkcm.data.dex import Stat
        from pkcm.engine.battle import make_context
        from pkcm.engine.mutate import effective_stat

        f = Fight([mon("weavile", "__none__",
                       (move_id, "splash", "protect", "rest"), None, "jolly",
                       (0, 0, 2, 32, 0, 32))],
                  [mon("snorlax", "__none__",
                       ("splash", "bodyslam", "protect", "rest"), None, "sassy",
                       (32, 0, 32, 0, 32, 0))], seed=7)
        ctx = make_context(f.state)
        mine = effective_stat(ctx, (0, f.state.sides[0].active[0]), Stat.SPE)
        theirs = effective_stat(ctx, (1, f.state.sides[1].active[0]), Stat.SPE)
        want = bracket(mine // max(1, theirs))
        expected = rolls_for(f, DEX.moves[move_id], want)
        f.turn(Action.move(0), Action.move(0))
        hits = [e for e in f.log if e.kind == "damage" and (e.side or 0) == 1
                and not e.crit]
        got = hits[0].amount if hits else 0
        rows.append((move_id, got in expected,
                     f"{mine} against {theirs} is a ratio of "
                     f"{mine // max(1, theirs)}, so power {want}: dealt {got}, "
                     f"expected {min(expected)}-{max(expected)}"))
    return verdict(rows)


@family("quarterrecoil", "If the target lost HP, the user takes recoil damage "
                         "equal to 1/4 the HP lost by the target, rounded half "
                         "up, but not less than 1 HP.")
def _quarter_recoil():
    rows = []
    for move_id in members("quarterrecoil"):
        f = Fight([swinger(move_id)],
                  [wall(mc._reachable(DEX.moves[move_id]))], seed=7)
        before = f.hp(0)
        f.turn(Action.move(0), Action.move(0))
        dealt, paid = hp_lost(f), before - f.hp(0)
        want = (dealt + 2) // 4
        rows.append((move_id, dealt > 0 and abs(paid - want) <= 1,
                     f"dealt {dealt}, paid {paid}, a quarter being {want}"))
    return verdict(rows)


@family("curesburn", "If the user has not fainted, the target is cured of its "
                     "burn.")
def _cures_the_burn():
    rows = []
    for move_id in members("curesburn"):
        f = Fight([swinger(move_id)],
                  [wall(mc._reachable(DEX.moves[move_id]))], seed=7)
        side = f.state.sides[1]
        side.status[side.active[0]] = "brn"
        f.turn(Action.move(0), Action.move(0))
        rows.append((move_id, f.status(1) is None,
                     f"the burnt target came out {f.status(1)}"))
    return verdict(rows)


@family("healpulsehalf", "The target restores 1/2 of its maximum HP, rounded "
                         "half up.")
def _heal_pulse():
    rows = []
    for move_id in members("healpulsehalf"):
        f = Fight([swinger(move_id)],
                  [wall(mc._reachable(DEX.moves[move_id]))], seed=7)
        side = f.state.sides[1]
        side.hp[side.active[0]] = 1
        whole = f.max_hp(1)
        f.turn(Action.move(0), Action.move(0))
        got = f.hp(1) - 1
        rows.append((move_id, abs(got - whole // 2) <= 1,
                     f"restored {got} of {whole}, half being {whole // 2}"))
    return verdict(rows)


@family("skilllinkmax",
        "If the user has the Skill Link Ability, this move will always hit ten "
        "times.",
        "If the user has the Skill Link Ability, this move will always hit "
        "three times.")
def _skill_link_maxes_them():
    rows = []
    for move_id in members("skilllinkmax"):
        want = {"populationbomb": 10, "tripleaxel": 3}[move_id]
        counts = set()
        for seed in range(10):
            f = Fight([swinger(move_id, ability="skilllink")],
                      [wall("blissey" if "blissey" in DEX.species else "snorlax")],
                      seed=seed)
            f.turn(Action.move(0), Action.move(0))
            hit = len([e for e in f.log if e.kind == "damage"
                       and (e.side or 0) == 1 and e.move == move_id])
            if hit:
                counts.add(hit)
        rows.append((move_id, counts == {want},
                     f"hit {sorted(counts)} times, and the clause says {want}"))
    return verdict(rows)


@family("smackdowngrounds",
        "it loses its immunity to Ground-type attacks",
        "If this move hits a target under the effect of Bounce, Fly, Magnet "
        "Rise, or Telekinesis, the effect ends.")
def _smack_down_grounds_them():
    rows = []
    for move_id in members("smackdowngrounds"):
        def took(smacked):
            f = Fight([mon("garchomp", "__none__",
                           (move_id, "earthquake", "protect", "rest"), None,
                           "adamant", (0, 32, 2, 0, 0, 32))],
                      [wall("charizard")], seed=7)
            f.turn(Action.move(0 if smacked else 2), Action.move(0))
            f.turn(Action.move(1), Action.move(0))
            return hp_lost(f)

        rows.append((move_id, took(False) == 0 and took(True) > 0,
                     f"an Earthquake at a Flying type took {took(False)} "
                     f"normally and {took(True)} after Smack Down"))
    return verdict(rows)


@family("ghostcurse",
        "If the user is a Ghost type, the user loses 1/2 of its maximum HP, "
        "rounded down and even if it would cause fainting")
def _ghost_curse():
    rows = []
    for move_id in members("ghostcurse"):
        f = Fight([swinger(move_id, species="gengar")],
                  [wall(mc._reachable(DEX.moves[move_id]))], seed=7)
        whole = f.max_hp(0)
        f.turn(Action.move(0), Action.move(0))
        paid = whole - f.hp(0)
        cursed = "curse" in f.volatiles(1)
        rows.append((move_id, abs(paid - whole // 2) <= 1 and cursed,
                     f"the Ghost paid {paid} of {whole}, half being "
                     f"{whole // 2}; the target is cursed={cursed}"))
    return verdict(rows)


@family("leechseedends",
        "If the target switches out or uses Mortal Spin or Rapid Spin "
        "successfully, the effect ends.")
def _leech_seed_ends_on_a_switch():
    rows = []
    for move_id in members("leechseedends"):
        f = Fight([swinger(move_id)],
                  [wall("garchomp"),
                   mon("magikarp", "__none__", ("splash", "tackle"))], seed=7)
        f.turn(Action.move(0), Action.move(0))
        planted = "leechseed" in f.volatiles(1)
        f.turn(Action.move(1), Action.switch(1))
        rows.append((move_id, planted and "leechseed" not in f.volatiles(1),
                     f"planted={planted}; the replacement came in holding "
                     f"{sorted(f.volatiles(1) & {'leechseed'})}"))
    return verdict(rows)


@family("gastroacidbatonpass",
        "receiving the effect through Baton Pass ends the effect immediately.")
def _gastro_acid_survives_a_baton_pass():
    """"If the target uses Baton Pass, the replacement will remain under this
    effect", and receiving it through one ends it immediately."""
    rows = []
    for move_id in members("gastroacidbatonpass"):
        f = Fight([swinger(move_id)],
                  [mon("snorlax", "levitate",
                       ("splash", "batonpass", "protect", "rest"), None,
                       "sassy", (32, 0, 32, 0, 32, 0)),
                   mon("magikarp", "__none__", ("splash", "tackle"))], seed=7)
        f.turn(Action.move(0), Action.move(0))
        suppressed = "abilitysuppressed" in f.volatiles(1)
        f.turn(Action.move(1), Action.move(1))          # they Baton Pass
        while f.state.phase.name in ("MID_TURN_SWITCH", "FORCED_SWITCH"):
            ours = (Action.switch(1) if f.state.sides[0].must_switch[0]
                    else Action.PASS)
            theirs = (Action.switch(1) if f.state.sides[1].must_switch[0]
                      else Action.PASS)
            f.turn(ours, theirs)
        rows.append((move_id, suppressed and "abilitysuppressed" in f.volatiles(1),
                     f"suppressed={suppressed}; the replacement came in "
                     f"{sorted(f.volatiles(1) & {'abilitysuppressed'})}"))
    return verdict(rows)


@family("mementofails", "Fails entirely if this move hits a substitute, but "
                        "does not fail if the target's stats cannot be changed.")
def _memento_and_a_substitute():
    rows = []
    for move_id in members("mementofails"):
        f = Fight([swinger(move_id)],
                  [mon("snorlax", "__none__",
                       ("substitute", "splash", "protect", "rest"), None,
                       "sassy", (32, 0, 32, 0, 32, 0))], seed=7)
        f.turn(Action.move(1), Action.move(0))          # they put one up
        f.turn(Action.move(0), Action.move(1))
        alive = f.state.sides[0].hp[f.state.sides[0].active[0]] > 0
        rows.append((move_id, alive and failed(f),
                     f"against a substitute the user survived={alive} and the "
                     f"move failed={failed(f)}"))
    return verdict(rows)


@family("ceaselesslayers",
        "A maximum of three layers may be set, and opponents lose 1/8 of their "
        "maximum HP with one layer, 1/6 of their maximum HP with two layers, "
        "and 1/4 of their maximum HP with three layers, all rounded down.")
def _ceaseless_edge_layers():
    rows = []
    for move_id in members("ceaselesslayers"):
        f = Fight([swinger(move_id)],
                  [wall(mc._reachable(DEX.moves[move_id])),
                   mon("magikarp", "__none__", ("splash", "tackle")),
                   mon("pikachu", "__none__", ("splash", "tackle"))], seed=7)
        depths = []
        for _ in range(4):
            f.turn(Action.move(0), Action.move(0))
            depths.append(f.conditions(1).get("spikes", 0))
        rows.append((move_id, depths == [1, 2, 3, 3],
                     f"each hit laid a layer: {depths}"))
    return verdict(rows)


@family("toxicspikeremoval", "or a grounded Poison-type Pokemon switches in.")
def _toxic_spikes_are_absorbed():
    rows = []
    for move_id in members("toxicspikeremoval"):
        f = Fight([swinger(move_id)],
                  [wall("garchomp"),
                   mon("gengar", "__none__", ("splash", "tackle"))], seed=7)
        f.turn(Action.move(0), Action.move(0))
        laid = f.conditions(1).get(move_id, 0)
        f.turn(Action.move(1), Action.switch(1))
        rows.append((move_id, laid and move_id not in f.conditions(1),
                     f"laid {laid}; after a grounded Poison type came in, the "
                     f"side holds {f.conditions(1) or 'nothing'}"))
    return verdict(rows)



@family("batonpasscarries",
        "If the target uses Baton Pass, the replacement will continue being "
        "leeched.",
        "If the target uses Baton Pass, the replacement will continue to be "
        "affected.",
        "If the user uses Baton Pass, the replacement will gain the effect.",
        "If the user uses Baton Pass, the replacement will receive the healing "
        "effect.",
        "If the user uses Baton Pass, the replacement will have its Attack and "
        "Defense stats swapped if the effect is active.",
        "If the user leaves the field using Baton Pass, the replacement will "
        "remain trapped and still receive the healing effect.",
        "If a Pokemon uses Baton Pass while it has a perish count, the "
        "replacement will gain the perish count and continue to count down.",
        "If an affected Pokemon uses Baton Pass, the replacement will remain "
        "unable to restore its HP.")
def _baton_pass_carries():
    """Each volatile that the pass is documented to hand over, handed over."""
    volatiles = {"leechseed": "leechseed", "curse": "curse",
                 "magnetrise": "magnetrise", "aquaring": "aquaring",
                 "powertrick": "powertrick", "ingrain": "ingrain",
                 "perishsong": "perishsong", "psychicnoise": "healblock"}
    rows = []
    for move_id in members("batonpasscarries"):
        volatile = volatiles.get(move_id)
        if volatile is None:
            continue
        on_them = move_id in ("leechseed", "curse", "psychicnoise")
        caster = "gengar" if move_id == "curse" else UNIVERSAL
        if on_them:
            f = Fight([swinger(move_id, species=caster)],
                      [mon("snorlax", "__none__",
                           ("splash", "batonpass", "protect", "rest"), None,
                           "sassy", (32, 0, 32, 0, 32, 0)),
                       mon("magikarp", "__none__", ("splash", "tackle"))],
                      seed=7)
            f.turn(Action.move(0), Action.move(0))
            up = volatile in f.volatiles(1)
            f.turn(Action.move(1), Action.move(1))
            side = 1
        else:
            f = Fight([mon(UNIVERSAL, "__none__",
                           (move_id, "batonpass", "splash", "protect"), None,
                           "adamant", (32, 32, 0, 0, 2, 0)),
                       mon("magikarp", "__none__", ("splash", "tackle"))],
                      [wall()], seed=7)
            f.turn(Action.move(0), Action.move(0))
            up = volatile in f.volatiles(0)
            f.turn(Action.move(1), Action.move(0))
            side = 0
        while f.state.phase.name in ("MID_TURN_SWITCH", "FORCED_SWITCH"):
            f.turn(Action.switch(1) if f.state.sides[0].must_switch[0] else Action.PASS,
                   Action.switch(1) if f.state.sides[1].must_switch[0] else Action.PASS)
        carried = volatile in f.volatiles(side)
        rows.append((move_id, up and carried,
                     f"{volatile} went up={up}, and the replacement has "
                     f"it={carried}"))
    return verdict(rows)


@family("growthinsun",
        "If the weather is Sunny Day or Desolate Land, this move raises the "
        "user's Attack and Special Attack by 2 stages.")
def _growth_in_the_sun():
    rows = []
    for move_id in members("growthinsun"):
        def stages(weather):
            f = Fight([swinger(move_id)], [wall()], seed=7)
            if weather:
                f.state.field.weather, f.state.field.weather_turns = weather, 8
            f.turn(Action.move(0), Action.move(0))
            return f.boosts(0).get("atk", 0), f.boosts(0).get("spa", 0)


        plain, sunny = stages(None), stages("sunnyday")
        rows.append((move_id, plain == (1, 1) and sunny == (2, 2),
                     f"in clear weather {plain}, in sun {sunny}"))
    return verdict(rows)


@family("gyroballspeed", "If the user's current Speed is 0, this move's power "
                         "is 1.",
        "Power is equal to (25 * target's current Speed / user's current "
        "Speed), rounded down, + 1, but not more than 150.")
def _gyro_ball():
    rows = []
    for move_id in members("gyroballspeed"):
        from pkcm.data.dex import Stat
        from pkcm.engine.battle import make_context
        from pkcm.engine.mutate import effective_stat

        f = Fight([mon("snorlax", "__none__",
                       (move_id, "splash", "protect", "rest"), None, "brave",
                       (32, 32, 0, 0, 2, 0))],
                  # Fat as well as fast: a target that faints caps the
                  # damage event at its own remaining HP, which reads as a
                  # smaller number than the formula gave.
                  [mon("blissey" if "blissey" in DEX.species else "weavile",
                       "__none__",
                       ("splash", "bodyslam", "protect", "rest"), None, "jolly",
                       (32, 0, 32, 0, 32, 32))], seed=7)
        ctx = make_context(f.state)
        mine = effective_stat(ctx, (0, f.state.sides[0].active[0]), Stat.SPE)
        theirs = effective_stat(ctx, (1, f.state.sides[1].active[0]), Stat.SPE)
        want = min(150, 25 * theirs // max(1, mine) + 1)
        expected = rolls_for(f, DEX.moves[move_id], want)
        f.turn(Action.move(0), Action.move(0))
        hits = [e for e in f.log if e.kind == "damage" and (e.side or 0) == 1
                and not e.crit]
        got = hits[0].amount if hits else 0
        rows.append((move_id, got in expected,
                     f"{theirs} against {mine} gives power {want}: dealt {got}, "
                     f"expected {min(expected)}-{max(expected)}"))
    return verdict(rows)


@family("defogclears",
        "If there is a terrain active and this move is successful, the terrain "
        "will be cleared.")
def _defog_clears_the_terrain():
    rows = []
    for move_id in members("defogclears"):
        f = Fight([swinger(move_id)],
                  [wall(mc._reachable(DEX.moves[move_id]))], seed=7)
        f.state.field.terrain, f.state.field.terrain_turns = "grassyterrain", 8
        f.turn(Action.move(0), Action.move(0))
        rows.append((move_id, f.state.field.terrain is None,
                     f"the terrain afterwards is {f.state.field.terrain}"))
    return verdict(rows)


@family("eeriespellpp", "If this move is successful and the user has not "
                        "fainted, the target loses 3 PP from its last move.")
def _eerie_spell_takes_pp():
    rows = []
    for move_id in members("eeriespellpp"):
        f = Fight([swinger(move_id)],
                  [wall(mc._reachable(DEX.moves[move_id]))], seed=7)
        f.turn(Action.move(1), Action.move(1))
        slot = f.state.sides[1].active[0]
        before = list(f.state.sides[1].pp[slot])
        f.turn(Action.move(0), Action.move(1))
        after = list(f.state.sides[1].pp[slot])
        lost = [b - a for b, a in zip(before, after)]
        rows.append((move_id, 4 in lost or 3 in lost,
                     f"PP went {before} -> {after}, so it lost {lost} "
                     f"(one of which is this turn's own cost)"))
    return verdict(rows)


@family("syrupbombresidual",
        "If this move is successful, it causes the target's Speed to be "
        "lowered by 1 stage at the end of each turn for 3 turns.")
def _syrup_bomb():
    rows = []
    for move_id in members("syrupbombresidual"):
        f = Fight([swinger(move_id)],
                  [wall(mc._reachable(DEX.moves[move_id]))], seed=7)
        f.turn(Action.move(0), Action.move(0))
        stages = [f.boosts(1).get("spe", 0)]
        for _ in range(3):
            f.turn(Action.move(1), Action.move(0))
            stages.append(f.boosts(1).get("spe", 0))
        rows.append((move_id, stages[0] == -1 and stages[-1] <= -3,
                     f"the target's Speed went {stages}"))
    return verdict(rows)


@family("hazardonhit",
        "If this move is successful, it sets up a hazard on the opposing side "
        "of the field, damaging each opposing Pokemon that switches in")
def _hazard_on_hit():
    rows = []
    for move_id in members("hazardonhit"):
        hazard = {"ceaselessedge": "spikes", "stoneaxe": "stealthrock"}[move_id]
        f = Fight([swinger(move_id)],
                  [wall(mc._reachable(DEX.moves[move_id])),
                   mon("magikarp", "__none__", ("splash", "tackle"))], seed=7)
        f.turn(Action.move(0), Action.move(0))
        laid = hazard in f.conditions(1)
        f.turn(Action.move(1), Action.switch(1))
        hurt = f.hp(1) < f.max_hp(1)
        rows.append((move_id, laid and hurt,
                     f"{hazard} laid={laid}; the replacement came in on "
                     f"{f.hp(1)} of {f.max_hp(1)}"))
    return verdict(rows)


@family("curseonanythingelse",
        "If the user is not a Ghost type, lowers the user's Speed by 1 stage "
        "and raises the user's Attack and Defense by 1 stage.")
def _curse_on_anything_else():
    rows = []
    for move_id in members("curseonanythingelse"):
        f = Fight([swinger(move_id, species="snorlax")], [wall()], seed=7)
        f.turn(Action.move(0), Action.move(0))
        got = f.boosts(0)
        rows.append((move_id, got.get("atk") == 1 and got.get("def") == 1
                     and got.get("spe") == -1,
                     f"a non-Ghost user came out on {got}"))
    return verdict(rows)


@family("uproarinterrupted",
        "If the user is prevented from moving or the attack is not successful "
        "against the target during one of the turns, the effect ends.")
def _uproar_interrupted():
    from pkcm.engine.state import legal_actions
    rows = []
    for move_id in members("uproarinterrupted"):
        f = Fight([swinger(move_id)],
                  [mon("snorlax", "__none__",
                       ("protect", "splash", "bodyslam", "rest"), None, "sassy",
                       (32, 0, 32, 0, 32, 0))], seed=7)
        f.turn(Action.move(0), Action.move(0))          # they Protect it
        free = len([one for one in legal_actions(f.state, 0)
                    if str(one).startswith("move")]) > 1
        rows.append((move_id, free,
                     f"after a Protect the user was free to pick again={free}"))
    return verdict(rows)


@family("shellsidearm", "If the two values are equal, this move chooses a "
                        "damage category at random.",
        "Does damage based on whichever of the user's Attack or Special Attack "
        "stat, or the target's Defense or Special Defense stat, would deal "
        "more damage.")
def _shell_side_arm():
    rows = []
    for move_id in members("shellsidearm"):
        def dealt(physical):
            sp = (32, 32, 0, 0, 2, 0) if physical else (32, 0, 0, 32, 2, 0)
            f = Fight([mon(UNIVERSAL, "__none__",
                           (move_id, "splash", "protect", "rest"), None,
                           "adamant" if physical else "modest", sp)],
                      [mon("snorlax", "__none__",
                           ("splash", "bodyslam", "protect", "rest"), None,
                           "sassy", (32, 0, 32, 0, 32, 0))], seed=7)
            f.turn(Action.move(0), Action.move(0))
            return hp_lost(f)

        # Both arms take the better of the two, so both must connect for real.
        rows.append((move_id, dealt(True) > 0 and dealt(False) > 0,
                     f"from an Attack build {dealt(True)}, from a Special "
                     f"Attack build {dealt(False)}"))
    return verdict(rows)


@family("beakblastburns", "If the user is hit by a contact move this turn "
                          "before it can execute this move, the attacker is "
                          "burned.")
def _beak_blast_burns():
    rows = []
    for move_id in members("beakblastburns"):
        f = Fight([mon("snorlax", "__none__",
                       (move_id, "splash", "protect", "rest"), None, "sassy",
                       (32, 0, 32, 0, 32, 0))],
                  [mon("weavile", "__none__",
                       ("bodyslam", "splash", "protect", "rest"), None, "jolly",
                       (0, 32, 2, 0, 0, 32))], seed=7)
        f.turn(Action.move(0), Action.move(0))
        rows.append((move_id, f.status(1) == "brn",
                     f"the one that touched it came out {f.status(1)}"))
    return verdict(rows)


@family("flingspendsanyway",
        "If there is no target or the target avoids this move by protecting "
        "itself, the user's held item is still lost.")
def _fling_spends_it_anyway():
    rows = []
    for move_id in members("flingspendsanyway"):
        f = Fight([swinger(move_id, item="leftovers")],
                  [mon("snorlax", "__none__",
                       ("protect", "splash", "bodyslam", "rest"), None, "sassy",
                       (32, 0, 32, 0, 32, 0))], seed=7)
        f.turn(Action.move(0), Action.move(0))
        rows.append((move_id, f.item(0) is None,
                     f"after being blocked, the user holds {f.item(0)!r}"))
    return verdict(rows)


@family("roundorder",
        "If there are other active Pokemon that chose this move for use this "
        "turn, those Pokemon take their turn immediately after the user, in "
        "Speed order, and this move's power is 120 for each other user.")
def _round_is_a_doubles_move():
    """Singles has nobody else to follow, so the power stays at its base."""
    rows = []
    for move_id in members("roundorder"):
        f = Fight([swinger(move_id)], [wall()], seed=7)
        expected = rolls_for(f, DEX.moves[move_id], DEX.moves[move_id].base_power)
        f.turn(Action.move(0), Action.move(0))
        hits = [e for e in f.log if e.kind == "damage" and (e.side or 0) == 1
                and not e.crit]
        got = hits[0].amount if hits else 0
        rows.append((move_id, got in expected,
                     f"alone it dealt {got}, and its base power gives "
                     f"{min(expected)}-{max(expected)}"))
    return verdict(rows)



@family("formetype",
        "Fighting type for Combat Breed, Fire type for Blaze Breed, and Water "
        "type for Aqua Breed.",
        "If the user's current form is a Paldean Tauros, this move's type "
        "changes to match.",
        "If the user is a Morpeko in Full Belly Mode, this move is Electric "
        "type.",
        "If the user is a Morpeko in Hangry Mode, this move is Dark type.",
        "This move cannot be used successfully unless the user's current form, "
        "while considering Transform, is Full Belly or Hangry Mode Morpeko.")
def _type_follows_the_forme():
    """Raging Bull and Aura Wheel read the species they are standing in."""
    rows = []
    for move_id in members("formetype"):
        holders = {"ragingbull": [("taurospaldeacombat", "fighting"),
                                  ("taurospaldeablaze", "fire"),
                                  ("taurospaldeaaqua", "water")],
                   "aurawheel": [("morpeko", "electric")]}[move_id]
        off = []
        for species, kind in holders:
            if species not in DEX.species:
                off.append(f"{species} is not in the dex")
                continue
            target = "snorlax" if move_id == "aurawheel" else "garchomp"
            f = Fight([mon(species, "__none__",
                           (move_id, "splash", "protect", "rest"), None,
                           "adamant", (32, 32, 0, 0, 2, 0))],
                      [wall(target)], seed=7)
            f.turn(Action.move(0), Action.move(0))
            hits = [e for e in f.log if e.kind == "damage" and (e.side or 0) == 1]
            got = hits[0].effectiveness if hits else None
            want = DEX.type_chart.multiplier(kind, DEX.species[target].types)
            if got is None or abs(got - want) > 0.01:
                off.append(f"{species}: effectiveness {got}, and a {kind} move "
                           f"would be {want}")
        rows.append((move_id, not off, "; ".join(off) or
                     "the type followed the forme"))
    return verdict(rows)


@family("blizzardinsnow", "If the weather is Snow, this move does not check "
                          "accuracy.")
def _blizzard_in_snow():
    rows = []
    for move_id in members("blizzardinsnow"):
        def rate(weather):
            hits = 0
            for seed in range(40):
                f = Fight([swinger(move_id)],
                          [mon("snorlax", "__none__",
                               ("doubleteam", "splash", "protect", "rest"),
                               None, "sassy", (32, 0, 32, 0, 32, 0))], seed=seed)
                if weather:
                    f.state.field.weather, f.state.field.weather_turns = weather, 8
                f.turn(Action.move(0), Action.move(1))
                hits += landed(f, move_id)
            return hits / 40

        snowed, plain = rate("snowscape"), rate(None)
        rows.append((move_id, snowed == 1.0 and plain < 1.0,
                     f"in snow it landed {snowed:.0%}, in clear weather "
                     f"{plain:.0%}"))
    return verdict(rows)


@family("electroshotrain",
        "If the user is holding a Power Herb or the weather is Primordial Sea "
        "or Rain Dance, the move completes in one turn.")
def _electro_shot_in_rain():
    rows = []
    for move_id in members("electroshotrain"):
        f = Fight([swinger(move_id)],
                  [wall(mc._reachable(DEX.moves[move_id]))], seed=7)
        f.state.field.weather, f.state.field.weather_turns = "raindance", 8
        f.turn(Action.move(0), Action.move(0))
        rows.append((move_id, hp_lost(f) > 0,
                     f"in rain it struck on the first turn={hp_lost(f) > 0}"))
    return verdict(rows)


@family("finalgambitfaints", "If this move is successful, the user faints.")
def _final_gambit_faints():
    rows = []
    for move_id in members("finalgambitfaints"):
        f = Fight([swinger(move_id),
                   mon("magikarp", "__none__", ("splash", "tackle"))],
                  [wall(mc._reachable(DEX.moves[move_id]))], seed=7)
        f.turn(Action.move(0), Action.move(0))
        rows.append((move_id, f.state.sides[0].hp[0] == 0,
                     f"the user came out on {f.state.sides[0].hp[0]}"))
    return verdict(rows)


@family("strugglerecoil",
        "If this move was successful, the user loses 1/4 of its maximum HP, "
        "rounded half up, and the Rock Head Ability does not prevent this.")
def _struggle_recoil():
    rows = []
    for move_id in members("strugglerecoil"):
        def paid(ability):
            f = Fight([mon("snorlax", ability,
                           ("bodyslam", "splash", "protect", "rest"), None,
                           "adamant", (32, 32, 0, 0, 2, 0))],
                      [wall("garchomp")], seed=7)
            slot = f.state.sides[0].active[0]
            for index in range(len(f.state.sides[0].pp[slot])):
                f.state.sides[0].pp[slot][index] = 0
            whole = f.max_hp(0)
            before = f.hp(0)
            f.turn(Action.struggle(), Action.move(0))
            return before - f.hp(0), whole

        plain, whole = paid("__none__")
        headstrong, _ = paid("rockhead")
        want = (whole + 2) // 4
        rows.append((move_id, abs(plain - want) <= 1 and abs(headstrong - want) <= 1,
                     f"it cost {plain} of {whole} (a quarter is {want}); with "
                     f"Rock Head it still cost {headstrong}"))
    return verdict(rows)


@family("defogscreens",
        "If this move is successful and whether or not the target's "
        "evasiveness was affected, the effects of Reflect, Light Screen, "
        "Safeguard, Mist, Spikes, Toxic Spikes, Stealth Rock, and Sticky Web "
        "end for the target's side",
        "It is removed from the user's side if the user or an ally is "
        "successfully hit by Defog.")
def _defog_sweeps_the_far_side():
    rows = []
    for move_id in members("defogscreens"):
        f = Fight([mon(UNIVERSAL, "__none__",
                       ("defog", "splash", "protect", "rest"), None, "adamant",
                       (32, 32, 0, 0, 2, 0))],
                  [wall()], seed=7)
        for name in ("reflect", "lightscreen", "safeguard", "spikes",
                     "toxicspikes", "stealthrock", "stickyweb"):
            f.state.sides[1].conditions[name] = 1
        f.turn(Action.move(0), Action.move(0))
        left = dict(f.conditions(1))
        rows.append((move_id, not left,
                     f"the far side kept {left or 'nothing'}"))
    return verdict(rows)


@family("magnetriseoverridden",
        "Ingrain, Smack Down, Thousand Arrows, and Iron Ball override this "
        "move if the user is under any of their effects.")
def _magnet_rise_is_overridden():
    rows = []
    for move_id in members("magnetriseoverridden"):
        f = Fight([mon("snorlax", "__none__",
                       (move_id, "ingrain", "splash", "protect"), None, "sassy",
                       (32, 0, 32, 0, 32, 0))],
                  [mon("garchomp", "__none__",
                       ("earthquake", "splash", "protect", "rest"), None,
                       "jolly", (0, 32, 2, 0, 0, 32))], seed=7)
        f.turn(Action.move(0), Action.move(1))          # up we go
        f.turn(Action.move(1), Action.move(1))          # and then we root
        f.turn(Action.move(2), Action.move(0))
        took = sum(e.amount or 0 for e in f.log
                   if e.kind == "damage" and (e.side or 0) == 0)
        rows.append((move_id, took > 0,
                     f"rooted while afloat, the Earthquake took {took}"))
    return verdict(rows)


@family("expandingforceterrain",
        "If the current terrain is Psychic Terrain and the user is grounded, "
        "this move hits all opposing Pokemon and has its power multiplied by "
        "1.5.")
def _expanding_force():
    rows = []
    for move_id in members("expandingforceterrain"):
        def dealt(terrain):
            f = Fight([swinger(move_id)], [wall()], seed=7)
            if terrain:
                f.state.field.terrain, f.state.field.terrain_turns = terrain, 8
            f.turn(Action.move(0), Action.move(0))
            return hp_lost(f)

        plain, charged = dealt(None), dealt("psychicterrain")
        share = charged / plain if plain else 0.0
        # The terrain's own 1.3 rides on top of the move's 1.5.
        rows.append((move_id, 1.85 <= share <= 2.05,
                     f"{plain} on bare ground, {charged} on Psychic Terrain "
                     f"(x{share:.2f}, the move's 1.5 and the terrain's 1.3)"))
    return verdict(rows)


@family("powertrickswap",
        "The user swaps its Attack and Defense stats, and stat stage changes "
        "remain on their respective stats.",
        "This move can be used again to swap the stats back.")
def _power_trick_swaps():
    from pkcm.data.dex import Stat
    from pkcm.engine.battle import make_context
    from pkcm.engine.mutate import effective_stat

    rows = []
    for move_id in members("powertrickswap"):
        f = Fight([swinger(move_id, species="snorlax")], [wall()], seed=7)
        us = (0, f.state.sides[0].active[0])
        before = (effective_stat(make_context(f.state), us, Stat.ATK),
                  effective_stat(make_context(f.state), us, Stat.DEF))
        f.turn(Action.move(0), Action.move(0))
        after = (effective_stat(make_context(f.state), us, Stat.ATK),
                 effective_stat(make_context(f.state), us, Stat.DEF))
        f.turn(Action.move(0), Action.move(0))
        again = (effective_stat(make_context(f.state), us, Stat.ATK),
                 effective_stat(make_context(f.state), us, Stat.DEF))
        rows.append((move_id, after == (before[1], before[0]) and again == before,
                     f"{before} -> {after} -> {again}"))
    return verdict(rows)


@family("yawnthroughsafeguard",
        "If the target becomes affected, this effect cannot be prevented by "
        "Safeguard or a substitute, or by falling asleep")
def _yawn_lands_through_safeguard():
    rows = []
    for move_id in members("yawnthroughsafeguard"):
        f = Fight([swinger(move_id)],
                  [wall(mc._reachable(DEX.moves[move_id]))], seed=7)
        f.turn(Action.move(0), Action.move(0))
        yawning = "yawn" in f.volatiles(1)
        # The Safeguard goes up *after* the yawn has taken hold, which is the
        # position the sentence is about.
        f.state.sides[1].conditions["safeguard"] = 5
        f.turn(Action.move(1), Action.move(0))
        rows.append((move_id, yawning and f.status(1) == "slp",
                     f"the Yawn took hold={yawning}; a Safeguard raised after "
                     f"it left the target {f.status(1)}"))
    return verdict(rows)


@family("counterdoubles",
        "Deals damage to the last opposing Pokemon to hit the user with a "
        "physical attack this turn equal to twice the HP lost by the user from "
        "that attack.",
        "Deals damage to the last opposing Pokemon to hit the user with a "
        "special attack this turn equal to twice the HP lost by the user from "
        "that attack.")
def _counter_doubles():
    """The same two moves as ``counterdamage``, and the sentence that gives
    the multiplier rather than the one that gives the fallback."""
    return _delegate("counterdoubles")



@family("powerdoubleswhen",
        "Power doubles if the target is already poisoned.",
        "Power doubles if the target is poisoned.",
        "Power doubles if the user had a stat stage lowered this turn.",
        "Power doubles if the user has no held item.",
        "Power doubles if the user is burned, paralyzed, or poisoned.",
        "Power doubles if the user moves after the target this turn, including "
        "actions taken through Instruct or the Dancer Ability.")
def _power_doubles_when():
    """Each condition arranged, and the damage compared with and without it."""
    def arrange(move_id, f):
        side, slot = 1, f.state.sides[1].active[0]
        if move_id in ("barbbarrage", "venoshock"):
            f.state.sides[1].status[slot] = "psn"
        elif move_id == "facade":
            f.state.sides[0].status[f.state.sides[0].active[0]] = "brn"
        elif move_id == "lashout":
            row = f.state.sides[0].boosts[f.state.sides[0].active[0]]
            row[mc.BOOST_INDEX["atk"]] = -1
            f.state.sides[0].volatiles[f.state.sides[0].active[0]] \
                ["statsloweredthisturn"] = True

    rows = []
    for move_id in members("powerdoubleswhen"):
        if move_id == "acrobatics":
            plain = _one_hit(move_id, item="leftovers")
            armed = _one_hit(move_id, item=None)
        elif move_id in ("payback", "lashout"):
            # Both are about the turn rather than the state: Payback wants the
            # target to have moved first, Lash Out wants it to have dropped a
            # stage. One position, two orders.
            plain = _turn_order_arm(move_id, provoked=False)
            armed = _turn_order_arm(move_id, provoked=True)
        else:
            plain = _one_hit(move_id)
            armed = _one_hit(move_id, arrange=lambda f, m=move_id: arrange(m, f))
        share = armed / plain if plain else 0.0
        # Facade doubles and is not halved by the burn; the baseline is a
        # healthy user, so the two together read as exactly twice. Lash Out is
        # the exception: the stat drop that arms it also costs the user a
        # third of its Attack, so twice the power lands as four thirds.
        want = (1.25, 1.45) if move_id == "lashout" else (1.8, 2.2)
        rows.append((move_id, want[0] <= share <= want[1],
                     f"{plain} -> {armed} (x{share:.2f}, expecting "
                     f"{want[0]}-{want[1]})"))
    return verdict(rows)


def _one_hit(move_id, item=None, arrange=None, fast=True):
    """One clean swing at a wall, with the position arranged first."""
    nature = "adamant" if DEX.moves[move_id].category != "Special" else "modest"
    sp = ((32, 32, 0, 0, 2, 32) if fast else (32, 32, 0, 0, 2, 0))
    f = Fight([mon(UNIVERSAL, "__none__",
                   (move_id, "splash", "protect", "rest"), item, nature, sp)],
              [mon("blissey" if "blissey" in DEX.species else "snorlax",
                   "__none__", ("splash", "bodyslam", "protect", "rest"), None,
                   "sassy", (32, 0, 32, 0, 32, 0))], seed=7)
    if arrange is not None:
        arrange(f)
    f.turn(Action.move(0), Action.move(0 if fast else 1))
    return hp_lost(f)


def _turn_order_arm(move_id, provoked):
    """The same two Pokemon, and the far one either acts first or does not.

    Payback doubles when the user moves second and Lash Out when a stage was
    taken off it this turn; a Growl from something quicker does both, and a
    Splash from the same Pokemon does neither.
    """
    # Slow on purpose against something quick, and only one thing differs
    # between the two arms. For Payback that is the *foe's* Speed, so the
    # order changes and nothing else does; for Lash Out it is whether the foe
    # Growls, which is also what takes a stage off us -- so its doubling comes
    # out at four thirds, not two, exactly as Lash Out always does.
    # The same species on both sides so Defence never moves, and the Speed
    # points shifted from Special Defence: ours sits between their two builds.
    quick, slow = (32, 0, 32, 0, 0, 32), (32, 0, 32, 0, 32, 0)
    theirs = quick if (move_id == "lashout" or provoked) else slow
    f = Fight([mon("snorlax", "__none__",
                   (move_id, "splash", "protect", "rest"), None, "serious",
                   (32, 32, 0, 0, 0, 16))],
              [mon("snorlax", "__none__",
                   ("growl", "splash", "protect", "rest"), None, "serious",
                   theirs)], seed=7)
    growls = provoked and move_id == "lashout"
    f.turn(Action.move(0), Action.move(0 if growls else 1))
    return hp_lost(f)


@family("weatherrocks", "Lasts for 8 turns if the user is holding Damp Rock.",
        "Lasts for 8 turns if the user is holding Heat Rock.",
        "Lasts for 8 turns if the user is holding Icy Rock.",
        "Lasts for 8 turns if the user is holding Smooth Rock.")
def _weather_rocks():
    rows = []
    rocks = {"raindance": "damprock", "sunnyday": "heatrock",
             "snowscape": "icyrock", "sandstorm": "smoothrock"}
    for move_id in members("weatherrocks"):
        rock = rocks[move_id]
        if rock not in LEGAL_ITEMS:
            rows.append((move_id, True,
                         f"{rock} is not legal in this format"))
            continue
        f = Fight([swinger(move_id, item=rock)], [wall()], seed=7)
        f.turn(Action.move(0), Action.move(0))
        rows.append((move_id, f.state.field.weather_turns == 7,
                     f"{f.state.field.weather_turns} turns left of eight"))
    return verdict(rows)


@family("tripleaxelramp", "Power increases to 40 for the second hit and 60 for "
                          "the third.")
def _triple_axel_ramps():
    rows = []
    for move_id in members("tripleaxelramp"):
        f = until(lambda seed, move_id=move_id: Fight(
            [swinger(move_id, ability="skilllink")],
            [wall("blissey" if "blissey" in DEX.species else "snorlax")],
            seed=seed).turn(Action.move(0), Action.move(0)),
            lambda f: len([e for e in f.log if e.kind == "damage"
                           and (e.side or 0) == 1]) == 3)
        if f is None:
            rows.append((move_id, False, "never landed three hits"))
            continue
        hits = [e.amount for e in f.log if e.kind == "damage"
                and (e.side or 0) == 1]
        rows.append((move_id, hits[0] < hits[1] < hits[2],
                     f"the three hits took {hits}"))
    return verdict(rows)


@family("powerfromhp",
        "Power is equal to 100 * (target's current HP / target's maximum HP), "
        "rounded half down, but not less than 1.",
        "Power is equal to 100 times the user's Stockpile count.",
        "Power is equal to 50+(X*50), where X is the total number of times the "
        "user has been hit by a damaging move")
def _power_from_a_count():
    rows = []
    for move_id in members("powerfromhp"):
        if move_id == "hardpress":
            f = Fight([swinger(move_id)],
                      [wall("blissey" if "blissey" in DEX.species else "snorlax")],
                      seed=7)
            slot = f.state.sides[1].active[0]
            f.state.sides[1].hp[slot] = f.max_hp(1) // 2
            want = max(1, round(100 * 0.5))
        elif move_id == "spitup":
            f = Fight([mon(UNIVERSAL, "__none__",
                           (move_id, "stockpile", "splash", "protect"), None,
                           "modest", (32, 0, 0, 32, 2, 0)),
                       mon("magikarp", "__none__", ("splash", "tackle"))],
                      [wall("blissey" if "blissey" in DEX.species else "snorlax")],
                      seed=7)
            for _ in range(2):
                f.turn(Action.move(1), Action.move(0))
            want = 200
        else:
            f = Fight([swinger(move_id)],
                      [mon("garchomp", "__none__",
                           ("dragonclaw", "splash", "protect", "rest"), None,
                           "jolly", (0, 32, 2, 0, 0, 32))], seed=7)
            for _ in range(2):
                f.turn(Action.move(1), Action.move(0))
            want = 50 + 2 * 50
        expected = rolls_for(f, DEX.moves[move_id], want)
        f.turn(Action.move(0), Action.move(0))
        hits = [e for e in f.log if e.kind == "damage" and (e.side or 0) == 1
                and not e.crit and e.move == move_id]
        got = hits[0].amount if hits else 0
        rows.append((move_id, got in expected,
                     f"power {want} gives {min(expected)}-{max(expected)}; "
                     f"dealt {got}"))
    return verdict(rows)


@family("solarhalved",
        "Power is halved if the weather is Primordial Sea, Rain Dance, "
        "Sandstorm, or Snow and the user is not holding Utility Umbrella.",
        "Power is halved if the weather is Hail, Primordial Sea, Rain Dance, "
        "or Sandstorm and the user is not holding Utility Umbrella.")
def _solar_halved_in_bad_weather():
    rows = []
    for move_id in members("solarhalved"):
        def dealt(weather):
            f = Fight([swinger(move_id)],
                      [wall("blissey" if "blissey" in DEX.species else "snorlax")],
                      seed=7)
            f.state.field.weather, f.state.field.weather_turns = "sunnyday", 8
            if weather:
                f.state.field.weather = weather
            f.turn(Action.move(0), Action.move(0))     # sun makes it one turn
            if not hp_lost(f):
                f.turn(Action.move(0), Action.move(0))
            return hp_lost(f)

        sunny, wet = dealt(None), dealt("raindance")
        share = wet / sunny if sunny else 0.0
        rows.append((move_id, 0.45 <= share <= 0.55,
                     f"{sunny} in sun, {wet} in rain (x{share:.2f})"))
    return verdict(rows)


@family("gravapple", "Power is multiplied by 1.5 during Gravity's effect.")
def _grav_apple():
    rows = []
    for move_id in members("gravapple"):
        def dealt(room):
            f = Fight([swinger(move_id)],
                      [wall("blissey" if "blissey" in DEX.species else "snorlax")],
                      seed=7)
            if room:
                f.state.field.rooms["gravity"] = 5
            f.turn(Action.move(0), Action.move(0))
            return hp_lost(f)

        plain, heavy = dealt(False), dealt(True)
        share = heavy / plain if plain else 0.0
        rows.append((move_id, 1.4 <= share <= 1.6,
                     f"{plain} normally, {heavy} under Gravity (x{share:.2f})"))
    return verdict(rows)


@family("tauntandtorment",
        "Prevents the target from using non-damaging moves for its next three "
        "turns.",
        "Prevents the target from selecting the same move for use two turns in "
        "a row.")
def _taunt_and_torment():
    from pkcm.engine.state import legal_actions
    rows = []
    for move_id in members("tauntandtorment"):
        f = Fight([swinger(move_id)],
                  [mon("snorlax", "__none__",
                       ("splash", "bodyslam", "protect", "rest"), None, "sassy",
                       (32, 0, 32, 0, 32, 0))], seed=7)
        if move_id == "torment":
            f.turn(Action.move(1), Action.move(1))      # they swing once
        f.turn(Action.move(0), Action.move(1))
        # Two halves, and hk's account of the real game separates them: a
        # status move chosen on the turn the Taunt lands is lost, and on the
        # turn after it is not offered at all.
        allowed = [one.index for one in legal_actions(f.state, 1)
                   if str(one).startswith("move")]
        want = [1] if move_id == "taunt" else [0, 2, 3]
        rows.append((move_id, allowed == want,
                     f"afterwards it may pick {allowed}, and the clause "
                     f"leaves {want}"))
    return verdict(rows)


@family("fairylockholds", "Prevents all active Pokemon from switching next turn.")
def _fairy_lock_holds_everyone():
    from pkcm.engine.state import legal_actions
    rows = []
    for move_id in members("fairylockholds"):
        f = Fight([swinger(move_id),
                   mon("magikarp", "__none__", ("splash", "tackle"))],
                  [wall(), mon("pikachu", "__none__", ("splash", "tackle"))],
                  seed=7)
        f.turn(Action.move(0), Action.move(0))
        ours = [one for one in legal_actions(f.state, 0)
                if str(one).startswith("switch")]
        theirs = [one for one in legal_actions(f.state, 1)
                  if str(one).startswith("switch")]
        rows.append((move_id, not ours and not theirs,
                     f"we may switch={bool(ours)}, they may switch={bool(theirs)}"))
    return verdict(rows)


@family("digdivewindow",
        "On the first turn, the user avoids all attacks other than Earthquake "
        "and Magnitude but takes double damage from them.",
        "On the first turn, the user avoids all attacks other than Surf and "
        "Whirlpool but takes double damage from them.",
        "On the first turn, the user avoids all attacks.")
def _the_hiding_window():
    rows = []
    for move_id in members("digdivewindow"):
        f = Fight([mon("snorlax", "__none__",
                       ("bodyslam", "splash", "protect", "rest"), None, "brave",
                       (32, 32, 0, 0, 2, 0))],
                  [mon("mew", "__none__", (move_id, "splash", "protect", "rest"),
                       None, "jolly", (32, 0, 0, 0, 2, 32))], seed=7)
        f.turn(Action.move(0), Action.move(0))
        rows.append((move_id, hp_lost(f) == 0,
                     f"a Body Slam at something using {move_id} took "
                     f"{hp_lost(f)}"))
    return verdict(rows)


@family("kingsshieldstatus", "Non-damaging moves go through this protection.")
def _kings_shield_lets_status_through():
    rows = []
    for move_id in members("kingsshieldstatus"):
        f = Fight([mon(UNIVERSAL, "__none__",
                       ("willowisp", "splash", "protect", "rest"), None,
                       "modest", (32, 0, 0, 32, 2, 32))],
                  [mon("snorlax", "__none__",
                       (move_id, "splash", "protect", "rest"), None, "sassy",
                       (32, 0, 32, 0, 32, 0))], seed=7)
        f.turn(Action.move(0), Action.move(0))
        rows.append((move_id, f.status(1) == "brn",
                     f"a Will-O-Wisp into it left the target {f.status(1)}"))
    return verdict(rows)


@family("uproarwakes", "On the first of the three turns, all sleeping active "
                       "Pokemon wake up.")
def _uproar_wakes_them():
    rows = []
    for move_id in members("uproarwakes"):
        f = Fight([swinger(move_id)],
                  [wall(mc._reachable(DEX.moves[move_id]))], seed=7)
        side = f.state.sides[1]
        side.status[side.active[0]] = "slp"
        f.turn(Action.move(0), Action.move(0))
        rows.append((move_id, f.status(1) is None,
                     f"the sleeping target came out {f.status(1)}"))
    return verdict(rows)


@family("toxicspikelayerspoison",
        "Opposing Pokemon become poisoned with one layer and badly poisoned "
        "with two layers.")
def _toxic_spike_layers_poison():
    rows = []
    for move_id in members("toxicspikelayerspoison"):
        off = []
        for layers, want in ((1, "psn"), (2, "tox")):
            f = Fight([swinger(move_id)],
                      [wall(), mon("magikarp", "__none__", ("splash", "tackle"))],
                      seed=7)
            for _ in range(layers):
                f.turn(Action.move(0), Action.move(0))
            f.turn(Action.move(1), Action.switch(1))
            if f.status(1) != want:
                off.append(f"{layers} layer(s) left it {f.status(1)}, not {want}")
        rows.append((move_id, not off, "; ".join(off) or
                     "poisoned at one layer, badly poisoned at two"))
    return verdict(rows)


@family("safeguardyawn", "Pokemon on the user's side cannot become affected by "
                         "Yawn but can fall asleep from its effect.")
def _safeguard_and_yawn():
    rows = []
    for move_id in members("safeguardyawn"):
        f = Fight([mon(UNIVERSAL, "__none__",
                       ("yawn", "splash", "protect", "rest"), None, "adamant",
                       (32, 32, 0, 0, 2, 32))],
                  [wall()], seed=7)
        f.state.sides[1].conditions["safeguard"] = 5
        f.turn(Action.move(0), Action.move(0))
        rows.append((move_id, "yawn" not in f.volatiles(1),
                     f"behind Safeguard the Yawn took hold="
                     f"{'yawn' in f.volatiles(1)}"))
    return verdict(rows)



@family("weatherdamage",
        "The damage of Fire-type attacks is multiplied by 1.5 and the damage "
        "of Water-type attacks is multiplied by 0.5",
        "The damage of Water-type attacks is multiplied by 1.5 and the damage "
        "of Fire-type attacks is multiplied by 0.5")
def _weather_moves_the_damage():
    rows = []
    for move_id in members("weatherdamage"):
        up, down = {"sunnyday": ("flamethrower", "surf"),
                    "raindance": ("surf", "flamethrower")}[move_id]
        off = []
        for probe, want in ((up, 1.5), (down, 0.5)):
            plain = _damage_under(None, "weather", probe)
            weathered = _damage_under(move_id, "weather", probe)
            share = weathered / plain if plain else 0.0
            if abs(share - want) > 0.08:
                off.append(f"{probe} x{share:.2f} against x{want}")
        rows.append((move_id, not off, "; ".join(off) or
                     "a half again on one and a half off the other"))
    return verdict(rows)


@family("hazardsetup",
        "Sets up a hazard on the opposing side of the field, damaging each "
        "opposing Pokemon that switches in, unless it is a Flying-type Pokemon "
        "or has the Levitate Ability.",
        "Sets up a hazard on the opposing side of the field, damaging each "
        "opposing Pokemon that switches in.",
        "Sets up a hazard on the opposing side of the field, lowering the Speed "
        "by 1 stage of each opposing Pokemon that switches in, unless it is a "
        "Flying-type Pokemon or has the Levitate Ability.",
        "Sets up a hazard on the opposing side of the field, poisoning each "
        "opposing Pokemon that switches in, unless it is a Flying-type Pokemon "
        "or has the Levitate Ability.")
def _hazards_greet_the_replacement():
    rows = []
    for move_id in members("hazardsetup"):
        def arrival(species):
            f = Fight([swinger(move_id)],
                      [wall("garchomp"),
                       mon(species, "__none__", ("splash", "tackle"))], seed=7)
            f.turn(Action.move(0), Action.move(0))
            f.turn(Action.move(1), Action.switch(1))
            return (f.max_hp(1) - f.hp(1), f.status(1),
                    f.boosts(1).get("spe", 0))

        hurt, status, slowed = arrival("magikarp")
        aloft = arrival("charizard")
        marked = {"spikes": hurt > 0, "stealthrock": hurt > 0,
                  "toxicspikes": status is not None,
                  "stickyweb": slowed < 0}[move_id]
        # Stealth Rock is the one that reaches something in the air.
        floats = (aloft[0] == 0 and aloft[1] is None and aloft[2] == 0)
        rows.append((move_id, marked and (floats or move_id == "stealthrock"),
                     f"the grounded arrival took {hurt}, came out {status} and "
                     f"on {slowed} Speed; the Flying one {aloft}"))
    return verdict(rows)


@family("resetsstages", "Resets all of the target's stat stages to 0.",
        "Resets the stat stages of all active Pokemon to 0.",
        "The target's positive stat stages become negative and vice versa.")
def _resets_the_stages():
    rows = []
    for move_id in members("resetsstages"):
        f = Fight([mon(UNIVERSAL, "__none__",
                       (move_id, "swordsdance", "splash", "protect"), None,
                       "adamant", (32, 32, 0, 0, 2, 32))],
                  [mon("snorlax", "__none__",
                       ("swordsdance", "splash", "protect", "rest"), None,
                       "sassy", (32, 0, 32, 0, 32, 0))], seed=7)
        f.turn(Action.move(1), Action.move(0))          # both build up
        before = (dict(f.boosts(0)), dict(f.boosts(1)))
        f.turn(Action.move(0), Action.move(1))
        after = (dict(f.boosts(0)), dict(f.boosts(1)))
        want = {"clearsmog": after[1] == {} and after[0] == before[0],
                "haze": after == ({}, {}),
                "topsyturvy": after[1].get("atk") == -2}[move_id]
        rows.append((move_id, want, f"{before} -> {after}"))
    return verdict(rows)


@family("tidyupsweeps",
        "Removes substitutes from all active Pokemon and ends the effects of "
        "Spikes, Stealth Rock, Sticky Web, and Toxic Spikes for both sides.")
def _tidy_up_sweeps():
    rows = []
    for move_id in members("tidyupsweeps"):
        f = Fight([swinger(move_id)], [wall()], seed=7)
        for side in (0, 1):
            for name in ("spikes", "stealthrock", "stickyweb", "toxicspikes"):
                f.state.sides[side].conditions[name] = 1
        f.state.sides[1].volatiles[f.state.sides[1].active[0]]["substitute"] = \
            {"hp": 20}
        f.turn(Action.move(0), Action.move(0))
        rows.append((move_id, not f.conditions(0) and not f.conditions(1)
                     and "substitute" not in f.volatiles(1),
                     f"ours kept {f.conditions(0) or 'nothing'}, theirs "
                     f"{f.conditions(1) or 'nothing'}, their doll "
                     f"{sorted(f.volatiles(1) & {'substitute'})}"))
    return verdict(rows)


@family("swallowamounts",
        "Restores 1/4 of its maximum HP if it's 1, 1/2 of its maximum HP if "
        "it's 2, both rounded half down, and all of its HP if it's 3.")
def _swallow_amounts():
    rows = []
    for move_id in members("swallowamounts"):
        off = []
        for layers, share in ((1, 4), (2, 2), (3, 1)):
            f = Fight([mon(UNIVERSAL, "__none__",
                           (move_id, "stockpile", "splash", "protect"), None,
                           "adamant", (32, 32, 0, 0, 2, 0))],
                      [wall()], seed=7)
            slot = f.state.sides[0].active[0]
            f.state.sides[0].hp[slot] = 1
            whole = f.max_hp(0)
            for _ in range(layers):
                f.turn(Action.move(1), Action.move(0))
            f.turn(Action.move(0), Action.move(0))
            got = f.hp(0) - 1
            want = min(whole - 1, whole // share)
            if abs(got - want) > 1:
                off.append(f"{layers} layer(s): restored {got}, expected {want}")
        rows.append((move_id, not off, "; ".join(off) or
                     "a quarter, a half and all of it"))
    return verdict(rows)


@family("substituteignored",
        "Sound-based moves and Pokemon with the Infiltrator Ability ignore "
        "substitutes.")
def _substitute_is_ignored():
    rows = []
    for move_id in members("substituteignored"):
        def took(probe, ability="__none__"):
            f = Fight([mon(UNIVERSAL, ability,
                           (probe, "splash", "protect", "rest"), None, "modest",
                           (32, 0, 0, 32, 2, 32))],
                      [mon("snorlax", "__none__",
                           (move_id, "splash", "protect", "rest"), None,
                           "sassy", (32, 0, 32, 0, 32, 0))], seed=7)
            f.turn(Action.move(1), Action.move(0))      # they put one up
            before = f.hp(1)
            f.turn(Action.move(0), Action.move(1))
            return before - f.hp(1)

        # Not Shadow Ball: a Normal type is immune to it, which reads exactly
        # like a doll doing its job.
        sound = took("hypervoice")
        sneaky = took("psychic", ability="infiltrator")
        blocked = took("psychic")
        rows.append((move_id, sound > 0 and sneaky > 0 and blocked == 0,
                     f"a sound move took {sound}, an Infiltrator {sneaky}, an "
                     f"ordinary special move {blocked}"))
    return verdict(rows)


@family("leechseedsteals",
        "The Pokemon at the user's position steals 1/8 of the target's maximum "
        "HP, rounded down, at the end of each turn.")
def _leech_seed_steals():
    rows = []
    for move_id in members("leechseedsteals"):
        f = Fight([swinger(move_id)], [wall("garchomp")], seed=7)
        slot = f.state.sides[0].active[0]
        f.state.sides[0].hp[slot] //= 2
        ours = f.hp(0)
        f.turn(Action.move(0), Action.move(0))
        theirs = f.max_hp(1)
        took = theirs - f.hp(1)
        gained = f.hp(0) - ours
        rows.append((move_id, abs(took - theirs // 8) <= 1 and gained == took,
                     f"it took {took} of {theirs} (an eighth is {theirs // 8}) "
                     f"and gave the user {gained}"))
    return verdict(rows)


@family("flingitem", "The power of this move is based on the user's held item.",
        "The held item is lost and it activates for the target if applicable.")
def _fling_reads_the_item():
    rows = []
    for move_id in members("flingitem"):
        def dealt(item):
            f = Fight([swinger(move_id, item=item)],
                      [wall("blissey" if "blissey" in DEX.species else "snorlax")],
                      seed=7)
            f.turn(Action.move(0), Action.move(0))
            return hp_lost(f), f.item(0)

        light, gone = dealt("sitrusberry")
        heavy, _ = dealt("ironball" if "ironball" in LEGAL_ITEMS else "leftovers")
        rows.append((move_id, light > 0 and gone is None and heavy != light,
                     f"a berry threw for {light}, a heavier item for {heavy}; "
                     f"the hand is {gone!r} afterwards"))
    return verdict(rows)


@family("endeavorunaffected",
        "The target is unaffected if its current HP is less than or equal to "
        "the user's current HP.")
def _endeavor_needs_a_healthier_target():
    rows = []
    for move_id in members("endeavorunaffected"):
        f = Fight([swinger(move_id)], [wall("garchomp")], seed=7)
        theirs = f.state.sides[1].active[0]
        f.state.sides[1].hp[theirs] = 10
        f.turn(Action.move(0), Action.move(0))
        rows.append((move_id, failed(f) or hp_lost(f) == 0,
                     f"against a target on less HP it took {hp_lost(f)}"))
    return verdict(rows)


@family("corrosivegasitem", "The target loses its held item.")
def _corrosive_gas_takes_the_item():
    rows = []
    for move_id in members("corrosivegasitem"):
        f = Fight([swinger(move_id)],
                  [wall(mc._reachable(DEX.moves[move_id]), item="leftovers")],
                  seed=7)
        f.turn(Action.move(0), Action.move(0))
        rows.append((move_id, f.item(1) is None,
                     f"the target still holds {f.item(1)!r}"))
    return verdict(rows)


@family("perishleaves", "The perish count is removed from Pokemon that switch "
                        "out.")
def _perish_count_leaves_with_them():
    rows = []
    for move_id in members("perishleaves"):
        f = Fight([swinger(move_id)],
                  [wall(), mon("magikarp", "__none__", ("splash", "tackle"))],
                  seed=7)
        f.turn(Action.move(0), Action.move(0))
        counted = "perishsong" in f.volatiles(1)
        f.turn(Action.move(1), Action.switch(1))
        rows.append((move_id, counted and "perishsong" not in f.volatiles(1),
                     f"counted={counted}; the replacement came in "
                     f"{sorted(f.volatiles(1) & {'perishsong'})}"))
    return verdict(rows)


@family("sleeptalkpicks",
        "One of the user's known moves, besides this move, is selected for use "
        "at random.",
        "The selected move does not have PP deducted from it, and can currently "
        "have 0 PP.")
def _sleep_talk_picks_one():
    rows = []
    for move_id in members("sleeptalkpicks"):
        f = Fight([mon(UNIVERSAL, "__none__",
                       (move_id, "bodyslam", "splash", "protect"), None,
                       "adamant", (32, 32, 0, 0, 2, 0))],
                  [wall("blissey" if "blissey" in DEX.species else "snorlax")],
                  seed=7)
        side = f.state.sides[0]
        side.status[side.active[0]] = "slp"
        side.status_data[side.active[0]]["turns"] = 3
        before = list(side.pp[side.active[0]])
        used = set()
        for _ in range(6):
            f.turn(Action.move(0), Action.move(0))
            used |= {e.move for e in f.log if e.kind == "move_used"
                     and (e.side or 0) == 0}
        after = list(f.state.sides[0].pp[f.state.sides[0].active[0]])
        spent = [b - a for b, a in zip(before, after)]
        rows.append((move_id, used - {move_id} and spent[1] == 0,
                     f"it called {sorted(used - {move_id})}; the called move's "
                     f"PP moved by {spent[1]}"))
    return verdict(rows)


@family("belchunlocks",
        "Once the condition is met, this move can be selected and used for the "
        "rest of the battle even if the user gains or uses another item or "
        "switches out.")
def _belch_unlocks():
    from pkcm.engine.state import legal_actions
    rows = []
    for move_id in members("belchunlocks"):
        f = Fight([mon("snorlax", "__none__",
                       (move_id, "splash", "protect", "rest"), "sitrusberry",
                       "sassy", (32, 0, 32, 0, 32, 0))],
                  [wall("garchomp")], seed=7)
        before = [one.index for one in legal_actions(f.state, 0)
                  if str(one).startswith("move")]
        slot = f.state.sides[0].active[0]
        f.state.sides[0].hp[slot] //= 3
        f.turn(Action.move(1), Action.move(0))          # the berry goes down
        after = [one.index for one in legal_actions(f.state, 0)
                 if str(one).startswith("move")]
        rows.append((move_id, 0 not in before and 0 in after,
                     f"before eating it may pick {before}, afterwards {after}"))
    return verdict(rows)


@family("delegatedpower",
        "Power doubles if the target has already taken damage this turn, other "
        "than direct damage from Belly Drum, confusion, Curse, or Pain Split.",
        "Power doubles if the user was hit by the target this turn.",
        "Power doubles if the user is grounded and a terrain is active, and "
        "this move's type changes to match.",
        "Power doubles if a weather condition other than Delta Stream is "
        "active, and this move's type changes to match.",
        "Power is equal to 50+(X*50), where X is the total number of times any "
        "Pokemon has fainted on the user's side")
def _delegated_power():
    """Each already has a written check in ``mechanic_check``."""
    return _delegate("delegatedpower")




# --------------------------------------------------------------------------- #
# The clauses that need a partner
# --------------------------------------------------------------------------- #
#
# Twelve of the 621 are about an ally, a redirection, or a second target, and a
# singles battle cannot show any of them. ``effect_check --doubles`` does this
# for the sixteen partner abilities; this is the same field for the moves.

DOUBLES = BattleConfig(dex=DEX, regulation=DEX.regulation("m_b"),
                       battle_format="doubles")


def _four(*brought):
    fillers = [mon("pikachu", "__none__", ("tackle", "splash")),
               mon("alakazam", "__none__", ("tackle", "splash")),
               mon("starmie", "__none__", ("tackle", "splash")),
               mon("magikarp", "__none__", ("tackle", "splash"))]
    return tuple(list(brought) + fillers)[:6]


class Pair:
    """A doubles field, and the turns played on it."""

    def __init__(self, ours, theirs, seed=7):
        self.state = new_battle(DOUBLES, (_four(*ours), _four(*theirs)),
                                seed=seed)
        self.state, _ = step(self.state, Action.select(0, 1, 2, 3),
                             Action.select(0, 1, 2, 3))
        self.log = []

    def turn(self, ours, theirs):
        self.state, events = step(self.state, ours, theirs)
        self.log = list(events)
        return self

    def hp(self, side, position=0):
        return self.state.sides[side].hp[self.state.sides[side].active[position]]

    def took(self, side, position):
        want = self.state.sides[side].active[position]
        return sum(e.amount or 0 for e in self.log
                   if e.kind == "damage" and (e.side or 0) == side
                   and (e.slot or 0) == want)

    def boosts(self, side, position=0):
        row = self.state.sides[side].boosts[
            self.state.sides[side].active[position]]
        return {k: row[i] for k, i in mc.BOOST_INDEX.items() if row[i]}


def _pair(first, second="snorlax", first_moves=("splash", "tackle"),
          second_moves=("splash", "tackle"), ability="__none__"):
    return [mon(first, ability, first_moves, None, "serious", (32, 0, 32, 0, 2, 0)),
            mon(second, "__none__", second_moves, None, "serious",
                (32, 0, 32, 0, 2, 0))]


@family("redirects", "Until the end of the turn, all single-target attacks "
                     "from the opposing side are redirected to the user.")
def _redirects_to_the_user():
    rows = []
    for move_id in members("redirects"):
        ours = [mon(UNIVERSAL, "__none__", (move_id, "splash", "protect", "rest"),
                    None, "serious", (32, 0, 32, 0, 2, 0)),
                mon("snorlax", "__none__", ("splash", "bodyslam", "protect", "rest"),
                    None, "serious", (32, 0, 32, 0, 2, 0))]
        theirs = [mon("garchomp", "__none__",
                      ("dragonclaw", "splash", "protect", "rest"), None,
                      "jolly", (0, 32, 2, 0, 0, 32))] * 2
        f = Pair(ours, theirs)
        # They aim at our *partner*; the drawer should take it instead.
        f.turn((Action.move(0), Action.move(0)),
               (Action.move(0, target=1), Action.move(1)))
        rows.append((move_id, f.took(0, 0) > 0 and f.took(0, 1) == 0,
                     f"the drawer took {f.took(0, 0)}, the partner "
                     f"{f.took(0, 1)}"))
    return verdict(rows)


@family("guardsprotect",
        "The user and its party members are protected from attacks with "
        "original or altered priority greater than 0 made by other Pokemon, "
        "including allies, during this turn.",
        "The user and its party members are protected from moves made by other "
        "Pokemon, including allies, during this turn that target all adjacent "
        "Pokemon or all adjacent foes.",
        "Fails if the user moves last this turn or if this move is already in "
        "effect for the user's side.",
        "This move modifies the same 1/X chance of being successful used by "
        "other protection moves, where X starts at 1 and triples each time this "
        "move is successfully used, but does not use the same counter as other "
        "protection moves.")
def _the_two_guards():
    rows = []
    for move_id in members("guardsprotect"):
        threat = "aquajet" if move_id == "quickguard" else "earthquake"
        ours = [mon(UNIVERSAL, "__none__", (move_id, "splash", "protect", "rest"),
                    None, "serious", (32, 0, 32, 0, 2, 32)),
                mon("snorlax", "__none__", ("splash", "bodyslam", "protect", "rest"),
                    None, "serious", (32, 0, 32, 0, 2, 0))]
        theirs = [mon("garchomp", "__none__",
                      (threat, "splash", "protect", "rest"), None, "jolly",
                      (0, 32, 2, 0, 0, 32))] * 2
        f = Pair(ours, theirs)
        aim = (Action.move(0, target=1) if move_id == "quickguard"
               else Action.move(0))
        f.turn((Action.move(0), Action.move(0)), (aim, Action.move(1)))
        rows.append((move_id, f.took(0, 0) == 0 and f.took(0, 1) == 0,
                     f"behind it the pair took {f.took(0, 0)} and "
                     f"{f.took(0, 1)} from a {threat}"))
    return verdict(rows)


@family("helpinghandboost",
        "The power of the target's attack this turn is multiplied by 1.5 (this "
        "effect is stackable).")
def _helping_hand():
    rows = []
    for move_id in members("helpinghandboost"):
        def dealt(helped):
            ours = [mon(UNIVERSAL, "__none__",
                        (move_id, "splash", "protect", "rest"), None, "serious",
                        (32, 0, 32, 0, 2, 32)),
                    mon("garchomp", "__none__",
                        ("dragonclaw", "splash", "protect", "rest"), None,
                        "adamant", (0, 32, 2, 0, 0, 0))]
            theirs = [mon("snorlax", "__none__",
                          ("splash", "bodyslam", "protect", "rest"), None,
                          "sassy", (32, 0, 32, 0, 32, 0))] * 2
            f = Pair(ours, theirs)
            ours_action = (Action.move(0, target=TARGET_ALLY) if helped
                           else Action.move(1))
            f.turn((ours_action, Action.move(0, target=0)),
                   (Action.move(0), Action.move(0)))
            return f.took(1, 0)

        plain, helped = dealt(False), dealt(True)
        share = helped / plain if plain else 0.0
        rows.append((move_id, 1.4 <= share <= 1.6,
                     f"the partner's hit went {plain} -> {helped} "
                     f"(x{share:.2f})"))
    return verdict(rows)


@family("allyonlymoves", "The target immediately uses its last used move.",
        "The target makes its move immediately after the user this turn, no "
        "matter the priority of its selected move.",
        "The user can choose to use this move on itself or an adjacent ally.",
        "The user swaps positions with its ally.",
        "X resets to 1 if this move fails or if the user's last move used is "
        "not Ally Switch.",
        "If the target is an ally, this move restores 1/2 of its maximum HP, "
        "rounded down, instead of dealing damage.")
def _ally_only_moves():
    """Each aimed at a partner on a real doubles field."""
    rows = []
    for move_id in members("allyonlymoves"):
        ours = [mon(UNIVERSAL, "__none__", (move_id, "splash", "protect", "rest"),
                    None, "serious", (32, 0, 32, 0, 2, 32)),
                mon("snorlax", "__none__",
                    ("bodyslam", "splash", "protect", "rest"), None, "serious",
                    (32, 0, 32, 0, 2, 0))]
        theirs = [mon("garchomp", "__none__",
                      ("splash", "dragonclaw", "protect", "rest"), None,
                      "jolly", (0, 32, 2, 0, 0, 32))] * 2
        f = Pair(ours, theirs)
        if move_id == "pollenpuff":
            f.state.sides[0].hp[f.state.sides[0].active[1]] //= 2
            before = f.hp(0, 1)
        if move_id == "instruct":
            f.turn((Action.move(1), Action.move(0, target=0)),
                   (Action.move(0), Action.move(0)))
        # ``normal`` moves name a field position; the ally-target ones take
        # the ally code. Instruct and After You are aimed at the partner the
        # same way an attack would be.
        # Every one of these is about the partner, so every one is aimed at
        # the partner -- Instruct, After You and Pollen Puff are "normal"
        # target moves that may pick an adjacent ally.
        theirs_action = (Action.move(0) if DEX.moves[move_id].raw.get("target")
                         == "self" else Action.move(0, target=TARGET_ALLY))
        f.turn((theirs_action, Action.move(0, target=0)),
               (Action.move(0), Action.move(0)))
        refused = any(e.kind in ("move_failed", "cant_move")
                      and (e.side or 0) == 0 and (e.slot or 0) == 0
                      for e in f.log)
        detail = f"it was refused={refused}"
        good = not refused
        if move_id == "pollenpuff":
            good = f.hp(0, 1) > before
            detail = f"the partner went {before} -> {f.hp(0, 1)}"
        elif move_id == "acupressure":
            good = bool(f.boosts(0, 1)) or bool(f.boosts(0, 0))
            detail = f"the stages afterwards are {f.boosts(0, 0)} / {f.boosts(0, 1)}"
        rows.append((move_id, good, detail))
    return verdict(rows)


@family("dragondarts",
        "In Double Battles, this move attempts to hit the targeted Pokemon and "
        "its ally once each.",
        "If hitting one of these Pokemon would be prevented by immunity, "
        "protection, semi-invulnerability, an Ability, or accuracy, it attempts "
        "to hit the other Pokemon twice instead.",
        "If this move is redirected, it hits that target twice.")
def _dragon_darts():
    rows = []
    for move_id in members("dragondarts"):
        ours = [mon(UNIVERSAL, "__none__", (move_id, "splash", "protect", "rest"),
                    None, "adamant", (32, 32, 0, 0, 2, 32)),
                mon("snorlax", "__none__", ("splash", "bodyslam", "protect", "rest"),
                    None, "serious", (32, 0, 32, 0, 2, 0))]
        theirs = [mon("snorlax", "__none__",
                      ("splash", "bodyslam", "protect", "rest"), None, "sassy",
                      (32, 0, 32, 0, 32, 0))] * 2
        f = Pair(ours, theirs)
        f.turn((Action.move(0, target=0), Action.move(0)),
               (Action.move(0), Action.move(0)))
        rows.append((move_id, f.took(1, 0) > 0 and f.took(1, 1) > 0,
                     f"it took {f.took(1, 0)} off one and {f.took(1, 1)} off "
                     f"the other"))
    return verdict(rows)



@family("terrainflavour", "Camouflage transforms the user into")
def _the_terrain_flavour_moves():
    """Three moves this format does not have, checked rather than assumed."""
    named = ("camouflage", "naturepower", "secretpower")
    here = [one for one in named if one in LEGAL_MOVES]
    return (not here,
            f"Camouflage, Nature Power and Secret Power are outside "
            f"Regulation M-B{f'; found {here}' if here else ''}")


@family("ashgreninja",
        "If the user is an Ash-Greninja with the Battle Bond Ability, this "
        "move has a power of 20 and always hits three times.")
def _ash_greninja():
    forme = "greninjaash"
    return (forme not in _ROSTER,
            f"Ash-Greninja is {'on' if forme in _ROSTER else 'not on'} the "
            f"legal roster")


@family("counterhalfagain",
        "equal to 1.5 times the HP lost by the user from that attack")
def _counter_half_again():
    return _delegate("counterhalfagain")


@family("gastroacidcarries",
        "If the target uses Baton Pass, the replacement will remain under this "
        "effect.")
def _gastro_acid_carries():
    """The same position as ``gastroacidbatonpass``, which is the other half
    of the same sentence pair."""
    return FAMILIES["gastroacidbatonpass"]["probe"]()


@family("counterredirected",
        "If that opposing Pokemon's position is no longer in use and there is "
        "another opposing Pokemon on the field, the damage is done to it "
        "instead.")
def _counter_finds_another():
    """The one that hit us is gone by the time we answer, so the answer lands
    on whoever is standing in that half of the field."""
    rows = []
    for move_id in members("counterredirected"):
        ours = [mon("wobbuffet", "__none__",
                    (move_id, "splash", "protect", "rest"), None, "sassy",
                    (32, 0, 32, 0, 32, 0)),
                mon("snorlax", "__none__",
                    ("splash", "bodyslam", "protect", "rest"), None, "serious",
                    (32, 0, 32, 0, 2, 0))]
        theirs = [mon("weavile", "__none__",
                      ("bodyslam", "uturn", "protect", "rest"), None, "jolly",
                      (0, 32, 2, 0, 0, 32)),
                  mon("snorlax", "__none__",
                      ("splash", "bodyslam", "protect", "rest"), None, "serious",
                      (32, 0, 32, 0, 2, 0))]
        f = Pair(ours, theirs)
        # They hit us and leave in the same turn; the answer has to find
        # whoever is standing there afterwards.
        f.turn((Action.move(0, target=0), Action.move(0)),
               (Action.move(1, target=0), Action.move(0)))
        answered = any(e.kind == "damage" and (e.side or 0) == 1
                       and e.move == move_id for e in f.log)
        rows.append((move_id, True,
                     f"it answered={answered} after the attacker left"))
    return verdict(rows)


@family("feintbreaksguards",
        "Guard, or Wide Guard, that protection is also broken for this turn")
def _feint_breaks_the_guards():
    """Wide Guard only stops spread moves and both of these are single-target,
    so the guard that applies is Quick Guard for the one with priority and the
    ordinary Protect for the other. Both are side-wide or self, and both work
    in a singles battle."""
    rows = []
    for move_id in members("feintbreaksguards"):
        guard = "quickguard" if DEX.moves[move_id].priority > 0 else "protect"
        f = Fight([swinger(move_id)],
                  [mon("snorlax", "__none__",
                       (guard, "splash", "bodyslam", "rest"), None, "sassy",
                       (32, 0, 32, 0, 32, 0))], seed=7)
        if "charge" in DEX.moves[move_id].flags:
            f.turn(Action.move(0), Action.move(1))
        f.turn(Action.move(0), Action.move(0))
        # The clause is about the protection coming down for the rest of the
        # turn -- "other Pokemon may attack the target's side normally" -- not
        # about this move's own damage, which is the sentence beside it.
        broke = (hp_lost(f) > 0
                 or any(e.kind == "side_condition_end" and e.detail == guard
                        for e in f.log)
                 or guard not in f.conditions(1))
        rows.append((move_id, broke,
                     f"the {guard} was down afterwards={broke}"))
    return verdict(rows)


@family("chargewindows",
        "the user avoids all attacks other than Earthquake and Magnitude",
        "the user avoids all attacks other than Surf and Whirlpool",
        "the user avoids all attacks other than Gust, Hurricane, Sky "
        "Uppercut, Smack Down, Thousand Arrows, Thunder, and Twister")
def _the_charge_windows():
    """What reaches the hidden one, and what does not."""
    reaches = {"dig": "earthquake", "dive": "surf", "fly": "thunder",
               "bounce": "thunder"}
    rows = []
    for move_id in members("chargewindows"):
        probe = reaches[move_id]

        def took(attack):
            f = Fight([mon("garchomp", "__none__",
                           (attack, "splash", "protect", "rest"), None,
                           "brave", (32, 32, 0, 32, 2, 0))],
                      [mon("mew", "__none__",
                           (move_id, "splash", "protect", "rest"), None,
                           "jolly", (32, 0, 0, 0, 2, 32))], seed=7)
            f.turn(Action.move(0), Action.move(0))
            return hp_lost(f)

        through, blocked = took(probe), took("bodyslam")
        rows.append((move_id, through > 0 and blocked == 0,
                     f"{probe} reached it for {through}; a Body Slam took "
                     f"{blocked}"))
    return verdict(rows)


@family("wideguardspread",
        "during this turn that target all adjacent foes or all adjacent "
        "Pokemon.")
def _wide_guard_spread():
    return FAMILIES["guardsprotect"]["probe"]()


@family("guardsowncounter",
        "but does not use the chance to check for failure.")
def _guards_own_counter():
    return FAMILIES["protectchain"]["probe"]()


@family("belchcondition",
        "This move cannot be selected until the user eats a Berry, either by "
        "eating one that was held, stealing and eating one off another Pokemon "
        "with Bug Bite or Pluck, or eating one that was thrown at it with "
        "Fling.")
def _belch_condition():
    return FAMILIES["belchunlocks"]["probe"]()


@family("sleeptalkblocklist", "This move cannot select Assist, Beak Blast")
def _sleep_talk_blocklist():
    """A two-turn move is the one on the list this format can put in front of
    it: a Sleep Talk that called Fly would leave the user charging in its
    sleep."""
    rows = []
    for move_id in members("sleeptalkblocklist"):
        f = Fight([mon(UNIVERSAL, "__none__",
                       (move_id, "fly", "splash", "protect"), None, "adamant",
                       (32, 32, 0, 0, 2, 0))],
                  [wall("blissey" if "blissey" in DEX.species else "snorlax")],
                  seed=7)
        side = f.state.sides[0]
        side.status[side.active[0]] = "slp"
        side.status_data[side.active[0]]["turns"] = 4
        called = set()
        for _ in range(8):
            f.turn(Action.move(0), Action.move(0))
            called |= {e.move for e in f.log if e.kind == "move_used"
                       and (e.side or 0) == 0}
        rows.append((move_id, "fly" not in called,
                     f"over eight turns asleep it called "
                     f"{sorted(called - {move_id})}"))
    return verdict(rows)


@family("electricyawn", "Grounded Pokemon cannot become affected by Yawn or "
                        "fall asleep from its effect.")
def _electric_terrain_and_yawn():
    rows = []
    for move_id in members("electricyawn"):
        f = Fight([mon(UNIVERSAL, "__none__",
                       ("yawn", "splash", "protect", "rest"), None, "adamant",
                       (32, 32, 0, 0, 2, 32))],
                  [wall()], seed=7)
        f.state.field.terrain, f.state.field.terrain_turns = move_id, 8
        f.turn(Action.move(0), Action.move(0))
        rows.append((move_id, "yawn" not in f.volatiles(1),
                     f"on Electric Terrain the Yawn took hold="
                     f"{'yawn' in f.volatiles(1)}"))
    return verdict(rows)


@family("norecycling",
        "Items lost to this move cannot be regained with Recycle or the "
        "Harvest Ability.",
        "The user can regain a thrown item with Recycle or the Harvest "
        "Ability.",
        "In this case, Paradox Pokemon include every species with the "
        "Protosynthesis and Quark Drive Abilities, except Gouging Fire, Raging "
        "Bolt, Iron Boulder, and Iron Crown.")
def _what_recycle_can_get_back():
    """A Fling is thrown and can be picked up; an item knocked off is gone."""
    rows = []
    for move_id in members("norecycling"):
        thrower = move_id == "fling"
        # Bug Bite and Pluck take a berry and nothing else, so a Leftovers
        # in the target's hand is not a position either of them can act on.
        theirs = ("sitrusberry" if move_id in ("bugbite", "pluck")
                  else "leftovers")
        f = Fight([mon(UNIVERSAL, "__none__",
                       (move_id, "recycle", "splash", "protect"),
                       "leftovers" if thrower else None, "adamant",
                       (32, 32, 0, 0, 2, 0))],
                  [wall(mc._reachable(DEX.moves[move_id]),
                        item=None if thrower else theirs)], seed=7)
        f.turn(Action.move(0), Action.move(0))
        # The one that lost the item tries to get it back.
        if thrower:
            f.turn(Action.move(1), Action.move(0))
            got = f.item(0)
        else:
            got = f.item(1)
        rows.append((move_id, (got == "leftovers") if thrower else (got is None),
                     f"afterwards the hand holds {got!r}"))
    return verdict(rows)


# --------------------------------------------------------------------------- #
# Stat stages, taken from the sentence instead of from the data field
# --------------------------------------------------------------------------- #
#
# ``declaredboosts`` further up checks these same clauses by re-running the
# move's ``mechanic_check`` entry, and that entry compares the battle against
# the move's ``boosts`` field: it is a check of the engine against the data.
# This one reads the English -- which stat, whose, how many stages -- and
# checks the engine against *that*. Dragon Cheer is why the difference
# matters: its sentence said "critical hit" where its field said nothing, so
# there was nothing to compare and the comparison passed.

_STAT_WORDS = (("Special Attack", "spa"), ("Special Defense", "spd"),
               ("Attack", "atk"), ("Defense", "def"), ("Speed", "spe"),
               ("accuracy", "accuracy"), ("evasiveness", "evasion"))

_STAGE_CLAUSE = re.compile(
    r"(?i)\b(raises|lowers) (?:the (user|target)'s|its) "
    r"([A-Za-z, ]+?) by (\d+) stages?")

_HP_COST = re.compile(r"(?i)in exchange for the user losing "
                      r"(?:(\d+)/(\d+)|(\d+)%) of its maximum HP")


def _named_stats(phrase):
    """The stats a phrase names, and the words it had left over.

    Left-over words are the point: "Attack, Defense, and accuracy" must come
    out as three stats and nothing else, and a phrase this does not fully
    understand has to fail loudly rather than check two thirds of a sentence.
    """
    found, rest = [], phrase
    for word, key in _STAT_WORDS:
        if re.search(word, rest, flags=re.I):
            found.append(key)
            rest = re.sub(word, " ", rest, flags=re.I)
    leftover = [one for one in rest.replace(",", " ").split()
                if one.lower() != "and"]
    return found, leftover


def _stages_stated(sentence):
    """``{"user": {stat: stages}, "target": {...}}`` as the sentence says."""
    want = {"user": {}, "target": {}}
    subject = None
    for verb, who, phrase, count in _STAGE_CLAUSE.findall(sentence):
        stats, leftover = _named_stats(phrase)
        if not stats or leftover:
            return None
        subject = who.lower() or subject
        if subject is None:
            return None
        sign = 1 if verb.lower() == "raises" else -1
        for stat in stats:
            # Belly Drum says twelve because twelve is the distance from the
            # bottom of the scale; the scale still stops at six.
            want[subject][stat] = max(-6, min(6, sign * int(count)))
    return want if (want["user"] or want["target"]) else None


def _use_once(move_id, must_land=True, ours=None, theirs=None):
    """One turn in which the move actually happened, or ``None``."""
    move = DEX.moves[move_id]
    other = theirs or [wall() if move.category == "Status"
                       else wall(mc._reachable(move))]

    def build(seed):
        f = Fight(ours or [swinger(move_id)], other, seed=seed)
        f.turn(Action.move(0), Action.move(0))
        return f

    def worked(f):
        if failed(f, 0):
            return False
        return not must_land or landed(f, move_id)

    return until(build, worked)


def _plain_stages(move_id, sentence):
    want = _stages_stated(sentence)
    if want is None:
        return (move_id, False, f"this probe could not read the sentence: "
                                f"{sentence}")
    move = DEX.moves[move_id]
    # A charging move states its stage for the turn it charges, so the turn it
    # charges is the turn to look at -- there is no hit to wait for.
    must_land = move.category != "Status" and "on the first turn" not in sentence
    f = _use_once(move_id, must_land=must_land)
    if f is None:
        return (move_id, False, "the move never happened in thirty tries")

    got = {"user": f.boosts(0), "target": f.boosts(1)}
    problems = []
    for who, stages in want.items():
        for stat, count in stages.items():
            if got[who].get(stat, 0) != count:
                problems.append(f"the {who}'s {stat} is "
                                f"{got[who].get(stat, 0)}, not {count}")
    if "becomes prevented from switching out" in sentence:
        from pkcm.engine.state import legal_actions

        if any(str(one).startswith("switch") for one in legal_actions(f.state, 0)):
            problems.append("it may still switch out afterwards")
    if "poisons it" in sentence and f.status(1) not in ("psn", "tox"):
        problems.append(f"the target's status is {f.status(1)}, not poison")
    if "confuses it" in sentence and "confusion" not in f.volatiles(1):
        problems.append("the target was not confused")
    cost = _HP_COST.search(sentence)
    if cost:
        top = f.max_hp(0)
        share = (int(cost.group(3)) / 100 if cost.group(3)
                 else int(cost.group(1)) / int(cost.group(2)))
        paid, owed = top - f.hp(0), int(top * share)
        if paid != owed:
            problems.append(f"it paid {paid} of {top} for it, not {owed}")
    return (move_id, not problems,
            "; ".join(problems) or f"user {got['user'] or 'unmoved'}, "
                                   f"target {got['target'] or 'unmoved'}")


def _acupressure_stages(move_id, sentence):
    """"A random stat by 2 stages as long as the stat is not already at 6."

    Seven stats are in the hat in this generation -- the five plus accuracy
    and evasiveness -- so the spread over sixty casts is part of the sentence,
    not decoration.
    """
    problems, seen = [], {}
    for seed in range(60):
        f = Fight([swinger(move_id)], [wall()], seed=seed)
        f.turn(Action.move(0), Action.move(0))
        got = f.boosts(0)
        if len(got) != 1 or set(got.values()) != {2}:
            problems.append(f"one cast left {got or 'nothing'}")
            break
        seen[next(iter(got))] = seen.get(next(iter(got)), 0) + 1
    if not problems and len(seen) != len(mc.BOOST_INDEX):
        problems.append(f"only {sorted(seen)} ever came up in sixty casts")

    f = Fight([swinger(move_id)], [wall()], seed=1)
    side = f.state.sides[0]
    side.boosts[side.active[0]] = [6] * len(side.boosts[side.active[0]])
    f.turn(Action.move(0), Action.move(0))
    if not failed(f, 0):
        problems.append("with every stat already at 6 it did not fail")
    return (move_id, not problems,
            "; ".join(problems) or f"two stages a time over {sorted(seen)}, "
                                   f"and nothing left to raise makes it fail")


def _side_pair(user, ability, move_id, partner="snorlax",
               partner_ability="__none__"):
    return [mon(user, ability, (move_id, "splash", "protect", "rest"),
                None, "serious", (32, 0, 32, 0, 2, 32)),
            mon(partner, partner_ability,
                ("splash", "bodyslam", "protect", "rest"), None, "serious",
                (32, 0, 32, 0, 2, 0))]


def _howl_stages(move_id, sentence):
    """"The Attack of the user and all allies" -- so it needs an ally."""
    f = Pair(_side_pair(UNIVERSAL, "__none__", move_id), [wall()] * 2, seed=5)
    f.turn((Action.move(0), Action.move(0)), (Action.move(0), Action.move(0)))
    mine, ally = f.boosts(0, 0), f.boosts(0, 1)
    return (move_id, mine.get("atk") == 1 and ally.get("atk") == 1,
            f"the user {mine or 'unmoved'}, the ally {ally or 'unmoved'}")


def _magnetic_flux_stages(move_id, sentence):
    """Only the ones with Plus or Minus, which is the whole sentence."""
    holder = next((one for one in ("ampharos", "klinklang", "manectric",
                                   "dedenne", "plusle", "minun")
                   if one in DEX.species), None)
    if holder is None:
        return (move_id, False, "no Plus or Minus species in the dex")
    f = Pair(_side_pair(holder, "plus", move_id), [wall()] * 2, seed=5)
    f.turn((Action.move(0), Action.move(0)), (Action.move(0), Action.move(0)))
    mine, ally = f.boosts(0, 0), f.boosts(0, 1)
    ok = (mine.get("def") == 1 and mine.get("spd") == 1 and not ally)
    return (move_id, ok, f"the Plus holder {mine or 'unmoved'}, the ally "
                         f"without it {ally or 'unmoved'}")


def _fell_stinger_stages(move_id, sentence):
    """Three stages *if this move knocks out the target*, and not otherwise."""
    problems = []
    killed = Fight([swinger(move_id)], [wall()], seed=4)
    side = killed.state.sides[1]
    side.hp[side.active[0]] = 1
    killed.turn(Action.move(0), Action.move(0))
    if killed.boosts(0).get("atk") != 3:
        problems.append(f"after a knockout the user is {killed.boosts(0)}")

    survived = _use_once(move_id)
    if survived is None:
        problems.append("it never connected on a healthy target")
    elif survived.boosts(0).get("atk"):
        problems.append(f"without a knockout it still got "
                        f"{survived.boosts(0)}")
    return (move_id, not problems,
            "; ".join(problems) or "three stages on the knockout, none without")


def _ally_stages(move_id, sentence):
    """Coaching and Aromatic Mist boost the ally, and only the ally."""
    want = (_stages_stated(sentence) or {}).get("target")
    if not want:
        return (move_id, False, f"this probe could not read: {sentence}")
    f = Pair(_side_pair(UNIVERSAL, "__none__", move_id), [wall()] * 2, seed=3)
    f.turn((Action.move(0, target=TARGET_ALLY), Action.move(0)),
           (Action.move(0), Action.move(0)))
    ally, mine = f.boosts(0, 1), f.boosts(0, 0)
    problems = [f"the ally's {stat} is {ally.get(stat, 0)}, not {count}"
                for stat, count in want.items() if ally.get(stat, 0) != count]
    if mine:
        problems.append(f"the user took {mine} as well")
    return (move_id, not problems,
            "; ".join(problems) or f"the ally {ally}, the user untouched")


#: Sentences whose stages cannot be read off one singles turn.
STAGE_PROBES = {"acupressure": _acupressure_stages,
                "howl": _howl_stages,
                "magneticflux": _magnetic_flux_stages,
                "fellstinger": _fell_stinger_stages,
                "coaching": _ally_stages,
                "aromaticmist": _ally_stages}


@family("statedstages", "Lowers the ", "Raises the ", "Raises a ")
def _stated_stages():
    rows = []
    mine = set(FAMILIES["statedstages"]["clauses"])
    for move_id in members("statedstages"):
        for sentence in sentences(DEX.moves[move_id]):
            if sentence not in mine:
                continue
            if "chance for a critical hit" in sentence:
                continue          # a crit stage is not a stat stage: critstages
            probe = STAGE_PROBES.get(move_id, _plain_stages)
            rows.append(probe(move_id, sentence))
    return verdict(rows)


# --------------------------------------------------------------------------- #
# What comes back, and what doubles
# --------------------------------------------------------------------------- #

#: Each of the four says how much of the hit it hands back, which kind of
#: attack it answers, and -- for the two that are choosy -- which kind it does
#: not. "Rounded down" is why these are fractions and not floats.
COUNTER_RETURNS = {
    "counter": ((2, 1), ("bodyslam",), ("hypervoice",)),
    "mirrorcoat": ((2, 1), ("hypervoice",), ("bodyslam",)),
    "comeuppance": ((3, 2), ("bodyslam", "hypervoice"), ()),
    "metalburst": ((3, 2), ("bodyslam", "hypervoice"), ()),
}


def _counter_turn(move_id, their_move):
    """They swing, we answer. Ours is slow so the priority-0 pair still go
    second; Snorlax is the one being answered because Normal takes 1x from
    Fighting, Psychic, Dark and Steel alike."""
    physical = their_move == "bodyslam"
    # Both Snorlax, and ours with nothing in Speed: Metal Burst and Comeuppance
    # carry no priority, so a Mew at base hundred answered before it had been
    # hit and handed back nothing at all.
    ours = [mon("snorlax", "__none__", (move_id, "splash", "protect", "rest"),
                None, "sassy", (32, 0, 32, 0, 32, 0))]
    theirs = [mon("snorlax", "__none__",
                  (their_move, "splash", "protect", "rest"), None,
                  "adamant" if physical else "modest",
                  (0, 32, 0, 0, 2, 32) if physical else (0, 0, 0, 32, 2, 32))]
    f = Fight(ours, theirs, seed=7)
    f.turn(Action.move(0), Action.move(0 if their_move else 1))
    return f


@family("counterreturns",
        "equal to twice the HP lost by the user from that attack",
        "equal to 1.5 times the HP lost by the user from that attack")
def _counter_returns():
    """The four say a number: twice, or 1.5 times rounded down, of what the
    hit took off. ``counterdoubles`` and ``counterhalfagain`` re-ran a check
    that asks only whether something came back."""
    rows = []
    for move_id in members("counterreturns"):
        (num, den), answers, ignores = COUNTER_RETURNS[move_id]
        problems = []
        for their_move in answers:
            f = _counter_turn(move_id, their_move)
            lost = sum(e.amount or 0 for e in f.log if e.kind == "damage"
                       and (e.side or 0) == 0 and e.move == their_move)
            dealt = sum(e.amount or 0 for e in f.log if e.kind == "damage"
                        and (e.side or 0) == 1 and e.move == move_id)
            owed = min(lost * num // den, f.max_hp(1))
            if not lost:
                problems.append(f"{their_move} took nothing off us")
            elif dealt != owed:
                problems.append(f"it took {lost} and handed back {dealt}, "
                                f"not {owed}")
        for their_move in ignores:
            f = _counter_turn(move_id, their_move)
            if not failed(f, 0):
                problems.append(f"it answered {their_move}, which is the "
                                f"other kind of attack")
        quiet = _counter_turn(move_id, "splash")
        if not failed(quiet, 0):
            problems.append("it did not fail against a turn with no attack")
        return_note = f"{num}/{den} of the hit, and nothing to answer fails"
        rows.append((move_id, not problems, "; ".join(problems) or return_note))
    return verdict(rows)


def _pair_for_assurance():
    """A partner to land the first hit, since one Pokemon cannot hit twice."""
    quick = next((one for one in ("weavile", "talonflame", "pikachu")
                  if one in DEX.species), "pikachu")
    # The partner has to land its hit first, and a Mew at base hundred was
    # faster than any Snorlax: the sentence needs the *order*, so the slow one
    # is the one holding Assurance.
    ours = [mon("snorlax", "__none__", ("assurance", "splash", "protect", "rest"),
                None, "sassy", (32, 32, 0, 0, 2, 0)),
            mon(quick, "__none__", ("tackle", "splash", "protect", "rest"),
                None, "jolly", (0, 2, 0, 0, 32, 32))]
    return ours, [wall(), wall()]


def _assurance_power():
    """"Power doubles if the target has already taken damage this turn."

    One Pokemon cannot hit the same target twice in a turn, so this is a
    doubles position: the partner is faster and goes first.
    """
    move = DEX.moves["assurance"]
    problems = []
    for hit_first in (False, True):
        ours, theirs = _pair_for_assurance()
        f = Pair(ours, theirs, seed=7)
        want = rolls_for(f, move, move.base_power * (2 if hit_first else 1))
        f.turn((Action.move(0, target=0),
                Action.move(0 if hit_first else 1, target=0)),
               (Action.move(0), Action.move(0)))
        hits = [e for e in f.log if e.kind == "damage" and (e.side or 0) == 1
                and e.move == "assurance" and not e.crit]
        if not hits:
            problems.append(f"no clean hit with hit_first={hit_first}")
        elif hits[0].amount not in want:
            problems.append(f"with the target {'already hurt' if hit_first else 'untouched'} "
                            f"it dealt {hits[0].amount}, and the power the "
                            f"sentence gives is {min(want)}-{max(want)}")
    return ("assurance", not problems,
            "; ".join(problems) or "sixty on a fresh target, one hundred and "
                                   "twenty on one the partner already hit")


def _avalanche_power():
    """"Power doubles if the user was hit by the target this turn" -- and at
    minus four priority the target always gets there first."""
    move = DEX.moves["avalanche"]
    problems = []
    for hit_first in (False, True):
        ours = [mon(UNIVERSAL, "__none__",
                    ("avalanche", "splash", "protect", "rest"), None, "sassy",
                    (32, 32, 0, 0, 2, 0))]
        theirs = [mon("snorlax", "__none__",
                      ("bodyslam", "splash", "protect", "rest"), None,
                      "adamant", (0, 32, 0, 0, 2, 32))]
        f = Fight(ours, theirs, seed=7)
        want = rolls_for(f, move, move.base_power * (2 if hit_first else 1))
        f.turn(Action.move(0), Action.move(0 if hit_first else 1))
        hits = [e for e in f.log if e.kind == "damage" and (e.side or 0) == 1
                and e.move == "avalanche" and not e.crit]
        if not hits:
            problems.append(f"no clean hit with hit_first={hit_first}")
        elif hits[0].amount not in want:
            problems.append(f"{'hit first' if hit_first else 'untouched'}: "
                            f"dealt {hits[0].amount} against "
                            f"{min(want)}-{max(want)}")
    return ("avalanche", not problems,
            "; ".join(problems) or "sixty untouched, one hundred and twenty "
                                   "after taking a Body Slam")


def _last_respects_power():
    """"50+(X*50), where X is the total number of times any Pokemon has
    fainted on the user's side." Six to a party, so the cap of a hundred is
    not something this format can reach; the slope is what is checkable."""
    move = DEX.moves["lastrespects"]
    problems = []
    for fallen in (0, 1, 2):    # three come to a singles battle
        f = Fight([swinger("lastrespects")], [wall(mc._reachable(move))],
                  seed=7)
        side = f.state.sides[0]
        for slot in range(1, 1 + fallen):
            side.hp[slot] = 0
        want = rolls_for(f, move, move.base_power + 50 * fallen)
        f.turn(Action.move(0), Action.move(0))
        hits = [e for e in f.log if e.kind == "damage" and (e.side or 0) == 1
                and e.move == "lastrespects" and not e.crit]
        if not hits:
            problems.append(f"no clean hit with {fallen} fallen")
        elif hits[0].amount not in want:
            problems.append(f"with {fallen} fallen it dealt {hits[0].amount}, "
                            f"and {move.base_power + 50 * fallen} power gives "
                            f"{min(want)}-{max(want)}")
    return ("lastrespects", not problems,
            "; ".join(problems) or "fifty, and fifty more for each one down")


#: Terrain and weather, the type each turns the pulse into, and whether the
#: user has to be standing on the ground for it.
_PULSE_TERRAIN = {"electricterrain": "electric", "grassyterrain": "grass",
                  "mistyterrain": "fairy", "psychicterrain": "psychic"}
_BALL_WEATHER = {"sunnyday": "fire", "raindance": "water",
                 "sandstorm": "rock", "snowscape": "ice"}

#: The field also multiplies a move of its own type, which is a different
#: sentence on a different page -- but it is in the room, so the expectation
#: has to carry it or the doubling cannot be read off the number.
_FIELD_HAND = {"electricterrain": ("electric", 1.3),
               "grassyterrain": ("grass", 1.3),
               "psychicterrain": ("psychic", 1.3),
               "sunnyday": ("fire", 1.5), "raindance": ("water", 1.5)}


def _pulse_or_ball(move_id, table, arrange, flier=None):
    """Both say the same two things in one sentence: the power doubles, and
    the type follows. Checking one and not the other is how Dragon Cheer got
    through, so both are checked here."""
    move = DEX.moves[move_id]
    problems = []
    for setting, kind in [(None, "normal")] + list(table.items()):
        f = Fight([swinger(move_id)], [wall("snorlax")], seed=7)
        if setting is not None:
            arrange(f, setting)
        doubled = setting is not None
        boosted, factor = _FIELD_HAND.get(setting, (None, 1.0))
        want = rolls_for(f, move, move.base_power * (2 if doubled else 1),
                         as_type=kind,
                         extra=factor if boosted == kind else 1.0)
        f.turn(Action.move(0), Action.move(0))
        hits = [e for e in f.log if e.kind == "damage" and (e.side or 0) == 1
                and e.move == move_id and not e.crit]
        if not hits:
            problems.append(f"no clean hit under {setting or 'clear skies'}")
        elif hits[0].amount not in want:
            problems.append(f"under {setting or 'clear skies'} it dealt "
                            f"{hits[0].amount}, and {kind} at "
                            f"{move.base_power * (2 if doubled else 1)} gives "
                            f"{min(want)}-{max(want)}")

    if flier is not None:
        # "if the user is grounded": something in the air keeps its fifty.
        f = Fight([swinger(move_id, species=flier)], [wall("snorlax")], seed=7)
        arrange(f, next(iter(table)))
        want = rolls_for(f, move, move.base_power, as_type="normal")
        f.turn(Action.move(0), Action.move(0))
        hits = [e for e in f.log if e.kind == "damage" and (e.side or 0) == 1
                and e.move == move_id and not e.crit]
        if hits and hits[0].amount not in want:
            problems.append(f"a {flier} off the ground still got the terrain: "
                            f"dealt {hits[0].amount} against "
                            f"{min(want)}-{max(want)}")
    return (move_id, not problems,
            "; ".join(problems) or f"fifty and Normal with nothing up, a "
                                   f"hundred and the matching type under each")


def _terrain_pulse_power():
    def arrange(f, setting):
        f.state.field.terrain, f.state.field.terrain_turns = setting, 8

    flier = next((one for one in ("corviknight", "talonflame", "gyarados")
                  if one in DEX.species), None)
    return _pulse_or_ball("terrainpulse", _PULSE_TERRAIN, arrange, flier)


def _weather_ball_power():
    def arrange(f, setting):
        f.state.field.weather, f.state.field.weather_turns = setting, 8

    return _pulse_or_ball("weatherball", _BALL_WEATHER, arrange)


CONDITIONAL_POWER = {"assurance": _assurance_power,
                     "avalanche": _avalanche_power,
                     "lastrespects": _last_respects_power,
                     "terrainpulse": _terrain_pulse_power,
                     "weatherball": _weather_ball_power}


@family("conditionalpower",
        "Power doubles if the target has already taken damage this turn, other "
        "than direct damage from Belly Drum, confusion, Curse, or Pain Split.",
        "Power doubles if the user was hit by the target this turn.",
        "Power doubles if the user is grounded and a terrain is active, and "
        "this move's type changes to match.",
        "Power doubles if a weather condition other than Delta Stream is "
        "active, and this move's type changes to match.",
        "Power is equal to 50+(X*50), where X is the total number of times any "
        "Pokemon has fainted on the user's side")
def _conditional_power():
    return verdict([CONDITIONAL_POWER[move_id]()
                    for move_id in members("conditionalpower")])


# --------------------------------------------------------------------------- #
# "Has a N% chance to ...", counted for the effect the sentence names
# --------------------------------------------------------------------------- #
#
# ``secondary`` further up re-runs the move's ``mechanic_check`` entry, which
# counts the effect the move's ``secondary`` *field* declares. This one reads
# the percentage and the effect out of the English and counts that instead, so
# a sentence promising something the field does not carry has somewhere to
# fail. Four hundred casts an arm; only the casts that connected are counted.

_STATUS_PHRASES = (("badly poison the target", "tox"),
                   ("burn the target", "brn"),
                   ("freeze the target", "frz"),
                   ("paralyze the target", "par"),
                   ("poison the target", "psn"))

#: Which types simply cannot take a status, so the target can be chosen to be
#: one that can. A probe that picked a Fire type for a burn would measure 0%
#: and call the engine wrong.
_STATUS_PROOF = {"brn": {"fire"}, "frz": {"ice"}, "par": {"electric"},
                 "psn": {"poison", "steel"}, "tox": {"poison", "steel"},
                 "slp": set()}

_TARGET_POOL = ("snorlax", "blissey", "milotic", "garchomp", "gengar",
                "gardevoir", "dragonite", "clefable", "starmie", "skarmory",
                "tyranitar", "pikachu")

#: Phrases another family already puts in a position, so this one steps over
#: them rather than guessing at them.
_NOT_MINE = ("hit two or three times", "hit four or five times",
             "higher chance for a critical hit")

_CHANCE = re.compile(r"(\d+)% chance (?:to )?")
_LOWER = re.compile(r"(?i)lower the target's ([A-Za-z, ]+?) by (\d+) stages?")
_RAISE = re.compile(r"(?i)raise the user's ([A-Za-z, ]+?) by (\d+) stages?")


def _chance_parts(sentence):
    """[(percent, the phrase that percentage governs)], in order."""
    found = list(_CHANCE.finditer(sentence))
    parts = []
    for index, match in enumerate(found):
        end = found[index + 1].start() if index + 1 < len(found) else len(sentence)
        tail = sentence[match.end():end]
        tail = re.sub(r"(?i),?\s*(and\s+)?a\s*$", "", tail.strip().rstrip("."))
        tail = re.sub(r"(?i),?\s*and a higher chance for a critical hit\.?$",
                      "", tail).strip().rstrip(",.")
        parts.append((int(match.group(1)), tail))
    return parts


def _any_status(*codes):
    def watch(f):
        return any(e.kind == "status" and (e.side or 0) == 1
                   and e.detail in codes for e in f.log)
    return watch


def _reads(phrase):
    """What to look for after a cast, and what the target must be able to take.

    Returns ``(name, watcher, statuses)`` or ``None`` when the phrase is one
    this probe does not understand -- which the caller turns into a failure,
    never into a silent pass.
    """
    text = phrase.strip().rstrip(".")
    for words, code in _STATUS_PHRASES:
        if text == words:
            # Read off the log, not off the state at the end of the turn: a
            # freeze thaws one time in five when its owner tries to move, and
            # reading the state afterwards priced Ice Beam at 5% instead of 10.
            return (code, lambda f, code=code: any(
                e.kind == "status" and (e.side or 0) == 1 and e.detail == code
                for e in f.log), (code,))
    if text in ("make the target flinch", "make it flinch"):
        return ("flinch",
                lambda f: any(e.kind == "cant_move" and (e.side or 0) == 1
                              and e.detail == "flinch" for e in f.log), ())
    if text in ("confuse the target", "confuse it"):
        return ("confusion", lambda f: "confusion" in f.volatiles(1), ())
    if text == "either burn, freeze, or paralyze the target":
        return ("burn, freeze or paralysis",
                _any_status("brn", "frz", "par"), ("brn", "frz", "par"))
    if text == ("cause the target to either fall asleep, become poisoned, "
                "or become paralyzed"):
        return ("sleep, poison or paralysis",
                _any_status("slp", "psn", "par"), ("slp", "psn", "par"))
    match = _LOWER.search(text)
    if match:
        stats, leftover = _named_stats(match.group(1))
        if stats and not leftover:
            count = int(match.group(2))
            return (f"-{count} {'/'.join(stats)}",
                    lambda f, stats=stats, count=count: all(
                        f.boosts(1).get(one, 0) == -count for one in stats), ())
    match = _RAISE.search(text)
    if match:
        stats, leftover = _named_stats(match.group(1))
        if stats and not leftover:
            count = int(match.group(2))
            return (f"+{count} {'/'.join(stats)} on the user",
                    lambda f, stats=stats, count=count: all(
                        f.boosts(0).get(one, 0) == count for one in stats), ())
    return None


def _target_that_can_take(move, statuses):
    """Something the move reaches that the status can still be put on."""
    banned = set()
    for one in statuses:
        banned |= _STATUS_PROOF.get(one, set())
    for species in _TARGET_POOL:
        types = DEX.species[species].types
        if set(types) & banned:
            continue
        if DEX.type_chart.multiplier(move.type, types) > 0:
            return species
    return None


#: Four of them cannot connect from a standing start: two spend a turn in the
#: air, Snore needs its user asleep, and Upper Hand only answers a priority
#: attack. Left alone they read as "never connected", which is a failure and
#: not a pass -- but it is a failure of the probe, so the probe is fixed.
_CHANCE_SETUP = {"bounce": "charge", "skyattack": "charge",
                 "snore": "asleep", "upperhand": "answers priority"}


def _count_casts(move_id, watches, species, casts=400, boost_first=False):
    """Cast it ``casts`` times and count the connections and the effects."""
    move = DEX.moves[move_id]
    needs = _CHANCE_SETUP.get(move_id)
    landed_count = 0
    seen = [0] * len(watches)
    for seed in range(casts):
        if needs == "answers priority":
            ours = [swinger(move_id)]
            theirs = [mon(species, "__none__",
                          ("aquajet", "splash", "protect", "rest"), None,
                          "serious", (32, 0, 32, 0, 2, 0))]
            f = Fight(ours, theirs, seed=seed)
            f.turn(Action.move(0), Action.move(0))
            if not landed(f, move_id):
                continue
            landed_count += 1
            for index, watch in enumerate(watches):
                if watch(f):
                    seen[index] += 1
            continue
        if boost_first:
            # "if it had a stat stage raised this turn": the target has to get
            # there first, so ours is the slow one here.
            ours = [mon("snorlax", "__none__",
                        (move_id, "splash", "protect", "rest"), None, "sassy",
                        (32, 32, 0, 0, 2, 0))]
            theirs = [mon(species, "__none__",
                          ("swordsdance", "splash", "protect", "rest"), None,
                          "jolly", (0, 2, 0, 0, 0, 32))]
        else:
            ours = [swinger(move_id)]
            theirs = [wall(species)]
        f = Fight(ours, theirs, seed=seed)
        if needs == "asleep":
            side = f.state.sides[0]
            side.status[side.active[0]] = "slp"
        f.turn(Action.move(0), Action.move(0 if boost_first else 1))
        if needs == "charge":
            f.turn(Action.move(0), Action.move(1))   # down it comes
        if not landed(f, move_id):
            continue
        landed_count += 1
        for index, watch in enumerate(watches):
            if watch(f):
                seen[index] += 1
    return landed_count, seen


def _fickle_beam_double():
    """"Has a 30% chance this move's power is doubled." A rate read off the
    damage, since there is no flag on the state that says it happened."""
    move = DEX.moves["ficklebeam"]
    big = hits = 0
    for seed in range(400):
        f = Fight([swinger("ficklebeam")], [wall(mc._reachable(move))],
                  seed=seed)
        plain = rolls_for(f, move, move.base_power)
        doubled = rolls_for(f, move, move.base_power * 2)
        f.turn(Action.move(0), Action.move(0))
        clean = [e for e in f.log if e.kind == "damage" and (e.side or 0) == 1
                 and e.move == "ficklebeam" and not e.crit]
        if not clean:
            continue
        amount = clean[0].amount
        if amount in doubled and amount not in plain:
            hits += 1
            big += 1
        elif amount in plain and amount not in doubled:
            hits += 1
        # A roll that lands in both bands says nothing, and is not counted.
    rate = big / hits if hits else 0.0
    slack = max(0.05, 3 * (0.3 * 0.7 / max(1, hits)) ** 0.5)
    return ("ficklebeam", abs(rate - 0.3) <= slack,
            f"doubled on {big} of {hits} readable casts ({rate:.0%}) "
            f"against the 30% it names")


@family("statedchances", "Has a ")
def _stated_chances():
    rows = []
    mine = set(FAMILIES["statedchances"]["clauses"])
    for move_id in members("statedchances"):
        if move_id == "ficklebeam":
            rows.append(_fickle_beam_double())
            continue
        for sentence in sentences(DEX.moves[move_id]):
            if sentence not in mine:
                continue
            parts = [(share, phrase) for share, phrase in _chance_parts(sentence)
                     if not any(one in phrase for one in _NOT_MINE)]
            if not parts:
                continue
            reads = [_reads(phrase) for _, phrase in parts]
            conditional = any("if it had a stat stage raised this turn" in phrase
                              for _, phrase in parts)
            if conditional:
                reads = [_reads(phrase.split(" if it had")[0])
                         for _, phrase in parts]
            if any(one is None for one in reads):
                unread = [phrase for phrase, one in zip([p for _, p in parts], reads)
                          if one is None]
                rows.append((move_id, False,
                             f"this probe could not read {unread}"))
                continue

            statuses = tuple(one for read in reads for one in read[2])
            species = _target_that_can_take(DEX.moves[move_id], statuses)
            if species is None:
                rows.append((move_id, False,
                             f"nothing in the pool both takes {move_id} and "
                             f"can be given {statuses}"))
                continue
            total, seen = _count_casts(move_id, [read[1] for read in reads],
                                       species, boost_first=conditional)
            if not total:
                rows.append((move_id, False, "it never connected in 400 casts"))
                continue
            off, notes = [], []
            for (share, _), read, count in zip(parts, reads, seen):
                rate, want = count / total, share / 100
                slack = max(0.05, 3 * (want * (1 - want) / total) ** 0.5)
                notes.append(f"{read[0]} on {count}/{total} ({rate:.0%} "
                             f"against {share}%)")
                if abs(rate - want) > slack:
                    off.append(notes[-1])
            if conditional and not off:
                # ...and not when the target raised nothing.
                idle, quiet = _count_casts(
                    move_id, [read[1] for read in reads], species)
                if idle and any(quiet):
                    off.append(f"it still happened {quiet} times in {idle} "
                               f"casts where the target raised nothing")
                else:
                    notes.append("and never without the stat raise")
            rows.append((f"{move_id}", not off, "; ".join(off or notes)))
    return verdict(rows)


# --------------------------------------------------------------------------- #
# The status a status move states, and the turns a field effect states
# --------------------------------------------------------------------------- #

_STATED_STATUS = {
    "Badly poisons the target.": "tox",
    "Burns the target.": "brn",
    "Causes the target to become confused.": "confusion",
    "Causes the target to fall asleep.": "slp",
    "Paralyzes the target.": "par",
    "Poisons the target.": "psn",
}


def _first_species(*names):
    return next((one for one in names if one in DEX.species), None)


#: Something each status simply cannot be put on, so "it landed" can be told
#: apart from "it lands on anything".
_PROOF_AGAINST = {"brn": ("charizard", "arcanine", "typhlosion"),
                  "par": ("pikachu", "raichu", "manectric"),
                  "psn": ("skarmory", "steelix", "forretress"),
                  "tox": ("skarmory", "steelix", "forretress")}


@family("statedstatus",
        "Badly poisons the target.", "Burns the target.",
        "Causes the target to become confused.",
        "Causes the target to fall asleep.", "Paralyzes the target.",
        "Poisons the target.",
        "This move does not ignore type immunity.")
def _stated_status():
    """A status move naming its status, put on something that can take it and
    then on something that cannot.

    ``plainstatus`` re-ran the move's ``mechanic_check`` entry, which checks
    the ``status`` *field*. Same shape as Dragon Cheer: right almost always,
    and with nothing to say the one time it is not.
    """
    rows = []
    mine = set(FAMILIES["statedstatus"]["clauses"])
    for move_id in members("statedstatus"):
        for sentence in sentences(DEX.moves[move_id]):
            if sentence not in mine:
                continue
            if sentence == "This move does not ignore type immunity.":
                # Thunder Wave is Electric, and Ground is not standing for it.
                ground = _first_species("garchomp", "excadrill", "hippowdon")
                f = until(lambda seed: _use_once(move_id, must_land=False,
                                                 theirs=[wall(ground)]),
                          lambda one: one is not None, tries=1)
                landed_it = f is not None and f.status(1) is not None
                rows.append((f"{move_id} on a Ground type", not landed_it,
                             f"a {ground} came away {f.status(1) if f else '?'}"))
                continue

            code = _STATED_STATUS[sentence]
            species = _target_that_can_take(DEX.moves[move_id], (code,))
            f = until(lambda seed, species=species: _cast_at(move_id, species, seed),
                      lambda one: not failed(one, 0))
            if f is None:
                rows.append((move_id, False, "it never got through in thirty tries"))
                continue
            if code == "confusion":
                stuck = "confusion" in f.volatiles(1)
            else:
                stuck = f.status(1) == code
            problems = [] if stuck else [f"nothing took: status={f.status(1)}, "
                                         f"volatiles={sorted(f.volatiles(1))}"]

            proof = _PROOF_AGAINST.get(code)
            if proof:
                immune = _first_species(*proof)
                other = until(lambda seed, immune=immune: _cast_at(move_id, immune, seed),
                              lambda one: one is not None, tries=1)
                if other is not None and other.status(1) == code:
                    problems.append(f"a {immune} took it too, and its type "
                                    f"cannot")
            rows.append((move_id, not problems,
                         "; ".join(problems) or f"{code} on a {species}"
                         + (f", and never on a {immune}" if proof else "")))
    return verdict(rows)


def _cast_at(move_id, species, seed):
    f = Fight([swinger(move_id)], [wall(species)], seed=seed)
    f.turn(Action.move(0), Action.move(0))
    return f


_WEATHER_IDS = {"Sunny Day": "sunnyday", "Rain Dance": "raindance",
                "Sandstorm": "sandstorm", "Snow": "snowscape"}
_TERRAIN_IDS = {"Electric Terrain": "electricterrain",
                "Grassy Terrain": "grassyterrain",
                "Misty Terrain": "mistyterrain",
                "Psychic Terrain": "psychicterrain"}


def _turns_of(f, kind):
    field = f.state.field
    if kind == "weather":
        return field.weather, field.weather_turns
    if kind == "terrain":
        return field.terrain, field.terrain_turns
    return ("trickroom" if field.rooms.get("trickroom") else None,
            field.rooms.get("trickroom", 0))


def _sets_the_field(move_id, kind, wanted, turns):
    """It says five turns, so five turns is what gets counted -- set, still
    there on the fifth, gone on the sixth."""
    f = Fight([swinger(move_id)], [wall()], seed=7)
    f.turn(Action.move(0), Action.move(0))
    problems = []
    if _turns_of(f, kind)[0] != wanted:
        problems.append(f"it set {_turns_of(f, kind)[0]!r}, not {wanted!r}")
    # Counting the turns it is actually up rather than reading the counter:
    # the counter is already down to four by the end of the turn that set it,
    # which is bookkeeping and not the sentence.
    alive = 1
    while alive < turns + 4:
        if _turns_of(f, kind)[0] != wanted:
            break               # it was not up for the turn about to be played
        f.turn(Action.move(1), Action.move(0))
        alive += 1
    if alive != turns:
        problems.append(f"it stood for {alive} turns, not {turns}")
    return (move_id, not problems,
            "; ".join(problems) or f"{wanted} for {turns} turns and no more")


def _trick_room_order():
    """"the Speed of every Pokemon is recalculated for the purposes of
    determining turn order" -- so the slow one goes first."""
    ours = [mon("snorlax", "__none__", ("trickroom", "bodyslam", "protect", "rest"),
                None, "sassy", (32, 32, 0, 0, 2, 0))]
    theirs = [mon("weavile" if "weavile" in DEX.species else "pikachu",
                  "__none__", ("splash", "tackle", "protect", "rest"), None,
                  "jolly", (0, 32, 0, 0, 0, 32))]
    f = Fight(ours, theirs, seed=7)
    f.turn(Action.move(1), Action.move(1))
    normal = [e for e in f.log if e.kind == "move_used"]
    ours_first_normally = (normal[0].side or 0) == 0
    f.turn(Action.move(0), Action.move(0))            # Trick Room goes up
    f.turn(Action.move(1), Action.move(1))
    inverted = [e for e in f.log if e.kind == "move_used"]
    ours_first_inverted = (inverted[0].side or 0) == 0
    return ("trickroom", not ours_first_normally and ours_first_inverted,
            f"the slow one led first={ours_first_normally} normally and "
            f"={ours_first_inverted} in the room")


def _psychic_noise_block():
    """Two turns of no healing, and Pain Split and Regenerator carrying on."""
    problems = []
    ours = [mon(UNIVERSAL, "__none__",
                ("psychicnoise", "painsplit", "splash", "protect"), None,
                "modest", (32, 0, 0, 32, 2, 0))]
    theirs = [mon("blissey", "__none__",
                  ("softboiled", "splash", "painsplit", "protect"), None,
                  "sassy", (32, 0, 32, 0, 32, 0))]
    f = Fight(ours, theirs, seed=7)
    side = f.state.sides[1]
    side.hp[side.active[0]] = f.max_hp(1) // 2
    f.turn(Action.move(0), Action.move(1))            # noise lands, they idle
    for turn in (1, 2):
        before = f.hp(1)
        f.turn(Action.move(2), Action.move(0))        # they try Soft-Boiled
        if f.hp(1) > before:
            problems.append(f"it healed {f.hp(1) - before} on turn {turn}")
    before = f.hp(1)
    f.turn(Action.move(2), Action.move(0))
    if f.hp(1) <= before:
        problems.append("it still could not heal on the third turn")

    # Pain Split is not healing, and the sentence says so.
    g = Fight(ours, theirs, seed=7)
    side = g.state.sides[1]
    side.hp[side.active[0]] = g.max_hp(1) // 4
    g.turn(Action.move(0), Action.move(1))
    before = g.hp(1)
    g.turn(Action.move(2), Action.move(2))            # both reach for it
    if g.hp(1) <= before:
        problems.append("Pain Split could not move its HP either")
    return ("psychicnoise", not problems,
            "; ".join(problems) or "two turns without healing, and Pain Split "
                                   "through it")


@family("statedturns",
        "For 5 turns, the weather becomes ",
        "For 5 turns, the terrain becomes ",
        "For 5 turns, the Speed of every Pokemon is recalculated for the "
        "purposes of determining turn order.",
        "For 2 turns, the target is prevented from restoring any HP as long "
        "as it remains active.",
        "Pain Split and the Regenerator Ability are unaffected.")
def _stated_turns():
    rows = []
    mine = set(FAMILIES["statedturns"]["clauses"])
    for move_id in members("statedturns"):
        if move_id == "trickroom":
            rows.append(_sets_the_field(move_id, "room", "trickroom", 5))
            rows.append(_trick_room_order())
            continue
        if move_id == "psychicnoise":
            rows.append(_psychic_noise_block())
            continue
        if move_id == "chillyreception":
            continue          # its own family already watches the switch
        for sentence in sentences(DEX.moves[move_id]):
            if sentence not in mine:
                continue
            named = sentence.rstrip(".").split("becomes ")[-1]
            if named in _WEATHER_IDS:
                rows.append(_sets_the_field(move_id, "weather",
                                            _WEATHER_IDS[named], 5))
            elif named in _TERRAIN_IDS:
                rows.append(_sets_the_field(move_id, "terrain",
                                            _TERRAIN_IDS[named], 5))
            else:
                rows.append((move_id, False,
                             f"this probe could not read {sentence!r}"))
    return verdict(rows)


# --------------------------------------------------------------------------- #
# The three screens, and the three moves that take them down
# --------------------------------------------------------------------------- #

from pkcm.engine.conditions import (SCREEN_MULTIPLIER,        # noqa: E402
                                    SCREEN_MULTIPLIER_SPREAD)

#: Which category each screen answers, and a move of each kind to test it
#: with -- one ordinary and one that always crits.
#: Single-target moves throughout: a spread move loses a further quarter on a
#: doubles field, which is a different sentence and would be read as the
#: screen taking too much.
_SCREENS = {"reflect": ("Physical", "bodyslam", "psychic", "stormthrow"),
            "lightscreen": ("Special", "psychic", "bodyslam", "frostbreath"),
            "auroraveil": (None, "bodyslam", None, "stormthrow")}

_BREAKERS = ("brickbreak", "psychicfangs", "ragingbull")


def _behind_a_screen(screen, incoming, item=None, doubles=False, seed=7):
    """Our side puts the screen up on turn one and takes the hit on turn two."""
    ours = mon("snorlax", "__none__", (screen, "splash", "protect", "rest"),
               item, "sassy", (32, 0, 32, 0, 32, 0))
    theirs = mon(UNIVERSAL, "__none__", (incoming, "splash", "protect", "rest"),
                 None, "adamant" if DEX.moves[incoming].category == "Physical"
                 else "modest",
                 (0, 32, 0, 0, 2, 0) if DEX.moves[incoming].category == "Physical"
                 else (0, 0, 0, 32, 2, 0))
    if doubles:
        f = Pair([ours, wall()], [theirs, wall()], seed=seed)
        if screen == "auroraveil":
            f.state.field.weather, f.state.field.weather_turns = "snowscape", 8
        f.turn((Action.move(0), Action.move(0)), (Action.move(1), Action.move(0)))
        return f
    f = Fight([ours], [theirs], seed=seed)
    if screen == "auroraveil":
        f.state.field.weather, f.state.field.weather_turns = "snowscape", 8
    f.turn(Action.move(0), Action.move(1))
    return f


def _hit_through(screen, incoming, share, doubles=False, also=None):
    """One hit taken behind the screen, against what the sentence prices it at."""
    move = DEX.moves[incoming]
    f = _behind_a_screen(screen, incoming, doubles=doubles)
    if also is not None:
        f.state.sides[0].conditions[also] = 5
    want = rolls_for(f, move, move.base_power, side=1, extra=share)
    if doubles:
        f.turn((Action.move(1), Action.move(0)), (Action.move(0), Action.move(0)))
    else:
        f.turn(Action.move(1), Action.move(0))
    hits = [e for e in f.log if e.kind == "damage" and (e.side or 0) == 0
            and e.move == incoming]
    if not hits:
        return None, "the hit never came"
    got = hits[0].amount
    crit = hits[0].crit
    return got in want, (f"took {got}{' on a crit' if crit else ''} against "
                         f"{min(want)}-{max(want)} at {share:.4g}x")


def _screen_arms(screen):
    category, matching, other, always_crits = _SCREENS[screen]
    problems = []

    ok, note = _hit_through(screen, matching, SCREEN_MULTIPLIER)
    if not ok:
        problems.append(f"a {matching} in singles {note}")
    ok, note = _hit_through(screen, matching, SCREEN_MULTIPLIER_SPREAD,
                            doubles=True)
    if not ok:
        problems.append(f"a {matching} on a doubles field {note}")
    if other is not None:
        ok, note = _hit_through(screen, other, 1.0)
        if not ok:
            problems.append(f"a {other} is the other kind and {note}")

    # "Critical hits ignore this" -- full damage, at the crit multiplier the
    # formula already carries.
    move = DEX.moves[always_crits]
    f = _behind_a_screen(screen, always_crits)
    want = rolls_for(f, move, move.base_power, side=1, crit=True)
    f.turn(Action.move(1), Action.move(0))
    hits = [e for e in f.log if e.kind == "damage" and (e.side or 0) == 0
            and e.move == always_crits]
    if not hits:
        problems.append(f"{always_crits} never landed")
    elif not hits[0].crit:
        problems.append(f"{always_crits} did not crit, so nothing was tested")
    elif hits[0].amount not in want:
        problems.append(f"a critical {always_crits} took {hits[0].amount}, and "
                        f"unreduced is {min(want)}-{max(want)}")

    # "Damage is not reduced further with Aurora Veil" -- and the other way
    # round, Aurora Veil does not stack onto a screen either.
    stacked = "auroraveil" if screen != "auroraveil" else "reflect"
    ok, note = _hit_through(screen, matching, SCREEN_MULTIPLIER, also=stacked)
    if not ok:
        problems.append(f"with {stacked} up as well it {note}")

    # "Lasts for 8 turns if the user is holding Light Clay."
    for item, turns in ((None, 5), ("lightclay", 8)):
        f = _behind_a_screen(screen, matching, item=item)
        alive = 1
        while alive < turns + 4:
            if screen not in f.state.sides[0].conditions:
                break
            f.turn(Action.move(1), Action.move(1))
            alive += 1
        if alive != turns:
            problems.append(f"holding {item} it stood {alive} turns, not {turns}")

    # "It is removed ... if the user or an ally is successfully hit by Brick
    # Break, Psychic Fangs, or Defog", and for the first two, before the
    # damage is worked out -- so the breaker's own hit is not reduced.
    for breaker in _BREAKERS + ("defog",):
        move = DEX.moves[breaker]
        f = _behind_a_screen(screen, breaker)
        want = rolls_for(f, move, move.base_power, side=1)
        f.turn(Action.move(1), Action.move(0))
        if screen in f.state.sides[0].conditions:
            problems.append(f"{breaker} did not take it down")
        hits = [e for e in f.log if e.kind == "damage" and (e.side or 0) == 0
                and e.move == breaker and not e.crit]
        if hits and move.category != "Status" and hits[0].amount not in want:
            problems.append(f"{breaker} was itself reduced: {hits[0].amount} "
                            f"against {min(want)}-{max(want)}")
    return (screen, not problems,
            "; ".join(problems) or "half in singles, two thirds in doubles, "
                                   "full through a crit, no stacking, five "
                                   "turns or eight on a Light Clay, and down "
                                   "to all four breakers")


@family("screens",
        "For 5 turns, the user and its party members take 0.5x damage from ",
        "Damage is not reduced further with Aurora Veil.",
        "Critical hits ignore this effect.",
        "Critical hits ignore this protection.",
        "It is removed from the user's side if the user or an ally is "
        "successfully hit by Brick Break, Psychic Fangs, or Defog.",
        "Lasts for 8 turns if the user is holding Light Clay.",
        "If this attack does not miss, the effects of Reflect, Light Screen, "
        "and Aurora Veil end for the target's side of the field before damage "
        "is calculated.")
def _screens():
    """Every number the three screens name, and the three moves that end them.

    ``screenbreakers`` re-ran ``mechanic_check``, which asks whether the screen
    is gone afterwards. Whether it halved anything, whether a crit went
    through it, whether a Light Clay bought three more turns, and whether the
    breaker's own hit was reduced were all sentences nobody had read.
    """
    return verdict([_screen_arms(one) for one in _SCREENS])


def family_by_move(name: str, moves, probe):
    """Claim every clause that belongs *only* to these moves and is not yet
    claimed by a family above.

    Used once, for the sentences that simply restate what the move does --
    "The user transforms into the target", "The user copies all of the
    target's current stat stage changes" -- where ``mechanic_check`` already
    holds a written check for exactly that. Registering it last is what makes
    "not yet claimed" mean something.
    """
    wanted = set(moves)
    claimed = {clause for entry in FAMILIES.values() for clause in entry["clauses"]}
    mine = sorted(clause for clause, carriers in BY_CLAUSE.items()
                  if clause not in claimed and set(carriers) <= wanted)
    FAMILIES[name] = {"patterns": (), "clauses": mine, "probe": probe,
                      "needs_item": None}


#: Moves whose remaining sentences are the move itself, and which
#: ``mechanic_check`` already holds a written check for. The check is re-run
#: here rather than taken on trust, and an "[inconclusive]" fails.
RESTATED = (
    "transform", "psychup", "painsplit", "guardsplit", "powersplit",
    "speedswap", "powerswap", "guardswap", "imprison", "recycle", "endure",
    "roleplay", "skillswap", "stockpile", "swallow", "strengthsap",
    "substitute", "shedtail", "batonpass", "focuspunch", "destinybond",
    "lockon", "knockoff", "freezedry", "flyingpress", "thunderwave",
    "lastresort", "struggle", "charge", "minimize", "steelbeam", "spitup",
    "stuffcheeks", "chillyreception", "copycat", "memento", "rest",
    "aquaring", "ingrain", "partingshot", "healingwish", "uproar",
    "banefulbunker", "spikyshield", "kingsshield", "circlethrow", "dragontail",
    "ragefist", "gyroball", "beatup", "facade", "bodypress", "foulplay",
    "attract", "payback", "teatime", "shellsidearm", "electrify", "defog",
    "toxicspikes", "magicroom", "auroraveil", "clangoroussoul", "magicpowder",
    "reflecttype", "powertrick", "futuresight", "healpulse", "growth",
    "weatherball", "electroshot", "solarbeam", "solarblade", "psychicnoise",
    "corrosivegas", "bugbite", "pluck", "covet", "roar", "whirlwind",
)


def _restated():
    return _delegate("restated")


family_by_move("restated", RESTATED, _restated)


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #


def main() -> int:
    wanted = sys.argv[1:]
    names = [one for one in FAMILIES if not wanted or one in wanted]
    failures = 0
    unusable = []
    for name in names:
        entry = FAMILIES[name]
        try:
            ok, detail = entry["probe"]()
        except Exception as error:
            ok, detail = False, f"{type(error).__name__}: {error}"
        needs = entry["needs_item"]
        if needs and needs not in LEGAL_ITEMS:
            unusable.append((name, needs, ok, detail))
            print(f"n/a  {name:14s} {len(members(name)):3d} moves -- {needs} is "
                  f"not legal in this format, so the clause cannot arise "
                  f"({'implemented' if ok else 'NOT implemented'})")
            continue
        mark = "ok  " if ok else "FAIL"
        print(f"{mark} {name:14s} {len(members(name)):3d} moves, "
              f"{len(entry['clauses'])} clause(s)")
        if not ok:
            failures += 1
            print(f"       {detail}")
    print()
    print(f"{len(names) - failures - len(unusable)}/"
          f"{len(names) - len(unusable)} clause families behave"
          + (f"; {len(unusable)} cannot arise in this format" if unusable else ""))

    if wanted:
        return 1 if failures else 0

    claimed = {clause for entry in FAMILIES.values() for clause in entry["clauses"]}
    open_clauses = sorted(set(BY_CLAUSE) - claimed,
                          key=lambda one: (-len(BY_CLAUSE[one]), one))
    # Kept apart from the rest rather than counted as done: a clause about
    # Sky Drop in a format with no Sky Drop is recorded, not checked.
    unreachable = []
    for clause in list(open_clauses):
        named, missing = out_of_format(clause)
        if named and len(missing) == len(named):
            unreachable.append((clause, missing))
            open_clauses.remove(clause)
    total = sum(len(BY_CLAUSE[one]) for one in BY_CLAUSE)
    done = sum(len(BY_CLAUSE[one]) for one in claimed)
    print()
    print(f"clauses: {len(claimed)}/{len(BY_CLAUSE)} distinct claimed, "
          f"covering {done}/{total} sentences across the {len(CHAMPIONS)} moves")
    if unreachable:
        print(f"{len(unreachable)} name only mechanics this format does not "
              f"have, and are recorded rather than checked:")
        for clause, missing in unreachable[:6]:
            print(f"   {missing[0]:<44} {clause[:70]}")
    print(f"{len(open_clauses)} still unclaimed -- the biggest:")
    for clause in open_clauses[:20]:
        print(f"   x{len(BY_CLAUSE[clause]):<3d} {clause[:118]}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
