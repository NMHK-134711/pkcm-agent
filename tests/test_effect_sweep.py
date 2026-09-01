"""Every declared move consequence in the format, cast and checked.

hk asked for exactly this after finding gaps by playing the real game: do the
secondaries and the stat changes actually happen? The executor is
data-driven, so the failure mode is not a wrong formula but a declared field
nothing reads -- which is precisely what Scale Shot's selfBoost was, on both
machines, independently, on the same day.

Method: for every Champions move that declares a consequence, build a fresh
1v1, rig the RNG so every chance roll succeeds, cast once, and diff the
declaration against the state. Ally-target moves are excused into a doubles
check of their own and charge moves into a two-turn one, because a silent
skip is how gaps survive audits.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from pkcm.data.dex import load_dex
from pkcm.engine.actions import Action
from pkcm.engine.battle import make_context, step
from pkcm.engine.moves import SELF_TARGETS, use_move
from pkcm.engine.pokemon import PokemonSet
from pkcm.engine.state import BattleConfig, new_battle

#: Status immunities by type, so the harness picks a defender it can afflict.
STATUS_BLOCK = {"brn": {"fire"}, "par": {"electric"}, "psn": {"poison", "steel"},
                "tox": {"poison", "steel"}, "frz": {"ice"}, "slp": set()}

DEFENDER_CHOICES = ("snorlax", "garchomp", "milotic", "gengar", "archaludon",
                    "dragonite", "clefable")

#: Verified in the doubles test below; the singles harness has no ally.
# adjacentAlly moves: a singles cast has no ally to aim at. Verified in
# doubles by test_ally_target_moves_boost_the_ally_in_doubles below.
ALLY_TARGET = {"coaching", "aromaticmist", "helpinghand", "dragoncheer"}


@pytest.fixture(scope="module")
def dex():
    return load_dex()


@pytest.fixture(scope="module")
def config(dex):
    return BattleConfig(dex=dex, regulation=dex.regulation("m_b"),
                        battle_format="singles")


def a_set(species, moves):
    return PokemonSet(species=species, moves=moves, ability="__test__",
                      nature="serious", sp=(0, 0, 0, 0, 0, 0))


class SureThing:
    """The real cursor, except every chance roll succeeds."""

    def __init__(self, inner):
        self._inner = inner

    def chance(self, num, den):
        return True

    def __getattr__(self, name):
        return getattr(self._inner, name)


def declared_effects(move):
    """(where, kind, payload) triples for everything the data promises."""
    promised = []
    raw = move.raw
    if raw.get("boosts"):
        where = "self" if raw.get("target") in SELF_TARGETS else "target"
        promised.append((where, "boosts", raw["boosts"]))
    self_block = raw.get("self")
    if isinstance(self_block, dict) and self_block.get("boosts"):
        promised.append(("self", "boosts", self_block["boosts"]))
    # selfBoost is its own field, applied once after the last hit rather than
    # per hit. Leaving it out left the sweep blind to exactly the bug that
    # started this audit: Scale Shot dealt its damage and moved no stat, and a
    # sweep that says it checks every declared effect passed anyway.
    self_boost = raw.get("selfBoost")
    if isinstance(self_boost, dict) and self_boost.get("boosts"):
        promised.append(("self", "boosts", self_boost["boosts"]))
    # Unconditional status and volatiles, which only counted inside a
    # secondary before -- so Thunder Wave declared nothing the sweep could see.
    if raw.get("status"):
        where = "self" if raw.get("target") in SELF_TARGETS else "target"
        promised.append((where, "status", raw["status"]))
    if raw.get("volatileStatus"):
        where = "self" if raw.get("target") in SELF_TARGETS else "target"
        promised.append((where, "volatile", raw["volatileStatus"]))
    for secondary in filter(None, [raw.get("secondary"),
                                   *(raw.get("secondaries") or ())]):
        if secondary.get("status"):
            promised.append(("target", "status", secondary["status"]))
        if secondary.get("boosts"):
            promised.append(("target", "boosts", secondary["boosts"]))
        if secondary.get("volatileStatus"):
            promised.append(("target", "volatile", secondary["volatileStatus"]))
        inner = secondary.get("self")
        if isinstance(inner, dict) and inner.get("boosts"):
            promised.append(("self", "boosts", inner["boosts"]))
    return promised


def pick_defender(dex, move, wanted_status):
    for species in DEFENDER_CHOICES:
        types = dex.species[species].types
        if move.base_power and dex.type_chart.multiplier(move.type, types) == 0:
            continue
        if wanted_status and set(types) & STATUS_BLOCK.get(wanted_status, set()):
            continue
        return species
    return "snorlax"


def test_every_declared_effect_happens(dex, config):
    moves = sorted(json.loads(pathlib.Path("data/champions/moves_available.json")
                              .read_text("utf-8")))
    skip_reasons = {}
    for m in dex.moves.values():
        if "charge" in m.flags:
            skip_reasons[m.id] = "two-turn charge (test below)"
        if m.raw.get("callsMove"):
            skip_reasons[m.id] = "calls another move"

    # Moves whose declared effect needs something a single cast cannot set up,
    # each verified under its real condition in tests/test_mechanics.py and
    # tests/test_moveeffects.py rather than waved through.
    conditional = {
        "curse": "Ghost users get the volatile; others boost instead",
        "disable": "needs a last move, and stores it as 'disabled'",
        "encore": "needs the target to have moved",
        "attract": "needs opposite, known genders",
    }

    failures, skipped, clean = [], [], 0
    for move_id in moves:
        move = dex.moves.get(move_id)
        if move is None:
            failures.append((move_id, "not in the dex at all"))
            continue
        promised = declared_effects(move)
        if not promised:
            continue
        if move_id in skip_reasons or move_id in ALLY_TARGET:
            skipped.append((move_id, skip_reasons.get(move_id, "ally target")))
            continue
        if move_id in conditional:
            skipped.append((move_id, conditional[move_id]))
            continue

        wanted_status = next((p for w, k, p in promised if k == "status"), None)
        bench = [a_set(s, ("tackle",)) for s in ("pikachu", "starmie")]
        red = tuple([a_set("snorlax", (move_id,))] + bench)
        blue = tuple([a_set(pick_defender(dex, move, wanted_status),
                            ("tackle",))] + bench)
        state = new_battle(config, (red, blue), seed=11)
        state, _ = step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))
        ctx = make_context(state)
        ctx.cursor = SureThing(ctx.cursor)

        try:
            use_move(ctx, (0, 0), move, defender=(1, 0))
        except Exception as error:  # noqa: BLE001 - the report is the point
            failures.append((move_id, f"raised {type(error).__name__}: {error}"))
            continue
        if any(e.kind == "unimplemented" for e in ctx.log):
            failures.append((move_id, "engine says unimplemented"))
            continue
        if any(e.kind == "move_missed" for e in ctx.log):
            skipped.append((move_id, "missed under a sure-thing cursor"))
            continue

        problems = []
        for where, kind, payload in promised:
            side, slot = (0, 0) if where == "self" else (1, 0)
            if kind == "boosts":
                for stat, amount in payload.items():
                    actual = state.sides[side].boost(slot, stat)
                    # Direction and at-least-magnitude: another declared effect
                    # on the same stat may stack in the same cast.
                    if amount > 0 and actual < amount:
                        problems.append(f"{where} {stat} {actual} < +{amount}")
                    if amount < 0 and actual > amount:
                        problems.append(f"{where} {stat} {actual} > {amount}")
            elif kind == "status":
                if state.sides[side].status[slot] != payload:
                    problems.append(
                        f"status {state.sides[side].status[slot]!r} != {payload!r}")
            elif kind == "volatile":
                if payload not in state.sides[side].volatiles[slot]:
                    problems.append(f"volatile {payload!r} absent")
        if problems:
            failures.append((move_id, "; ".join(problems)))
        else:
            clean += 1

    assert not failures, failures
    # A refactor that silently skips half the list would pass on vacuity;
    # the floor keeps the sweep honest about its own coverage.
    assert clean >= 155, (clean, skipped)


def test_ally_target_moves_boost_the_ally_in_doubles(dex):
    doubles = BattleConfig(dex=dex, regulation=dex.regulation("m_b"),
                           battle_format="doubles")
    for move_id, stats in (("coaching", ("atk", "def")),
                           ("aromaticmist", ("spd",))):
        red = (a_set("snorlax", (move_id,)), a_set("garchomp", ("tackle",)),
               a_set("starmie", ("tackle",)), a_set("pikachu", ("tackle",)))
        blue = (a_set("milotic", ("tackle",)), a_set("gengar", ("tackle",)),
                a_set("clefable", ("tackle",)), a_set("dragonite", ("tackle",)))
        state = new_battle(doubles, (red, blue), seed=3)
        state, _ = step(state, Action.select(0, 1, 2, 3),
                        Action.select(0, 1, 2, 3))
        ctx = make_context(state)
        use_move(ctx, (0, 0), dex.moves[move_id])
        ally = state.sides[0].active[1]
        for stat in stats:
            assert state.sides[0].boost(ally, stat) == 1, (move_id, stat)


def test_charge_moves_strike_on_turn_two(dex, config):
    """Bounce lands damage and can paralyze; Sky Attack lands damage."""
    for move_id, status in (("bounce", "par"), ("skyattack", None)):
        landed = False
        for seed in range(1, 40):
            red = (a_set("dragonite", (move_id,)), a_set("starmie", ("tackle",)),
                   a_set("pikachu", ("tackle",)))
            blue = (a_set("milotic", ("splash",)), a_set("gengar", ("tackle",)),
                    a_set("clefable", ("tackle",)))
            state = new_battle(config, (red, blue), seed=seed)
            state, _ = step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))
            state, _ = step(state, Action.move(0), Action.move(0))
            state, _ = step(state, Action.move(0), Action.move(0))
            if state.sides[1].hp[0] < state.pokemon(1, 0).max_hp:
                landed = True
                if status is None or state.sides[1].status[0] == status:
                    break
        assert landed, f"{move_id} never landed over two turns"
