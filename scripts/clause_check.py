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
    total = sum(len(BY_CLAUSE[one]) for one in BY_CLAUSE)
    done = sum(len(BY_CLAUSE[one]) for one in claimed)
    print()
    print(f"clauses: {len(claimed)}/{len(BY_CLAUSE)} distinct claimed, "
          f"covering {done}/{total} sentences across the {len(CHAMPIONS)} moves")
    print(f"{len(open_clauses)} still unclaimed -- the biggest:")
    for clause in open_clauses[:20]:
        print(f"   x{len(BY_CLAUSE[clause]):<3d} {clause[:118]}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
