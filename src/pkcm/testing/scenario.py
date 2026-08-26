"""Turn observations of the real game into permanent tests.

Champions is the only authority on Champions. Pokemon Showdown implements
Scarlet/Violet, and the places where Champions differs -- the SP stat system, no
Terastallization in ranked, Mega Evolution back with brand-new formes and
abilities -- are exactly the places we most need to be right. Diffing against
Showdown would check us against the wrong game.

So the oracle is a person playing the actual game. This module is how what they
see gets into the repository and stays there: a scenario is a JSON file
describing a position, the actions taken, and what was observed to happen.

The important property is that a scenario may be written *before* the mechanic
exists. Each one declares what it ``requires``; anything the engine does not yet
implement is **skipped with its reason stated**, not failed. Observations are
therefore never lost waiting for a milestone -- they sit in the suite and start
enforcing themselves the moment the mechanic lands.

Format (see tests/scenarios/README.md for the annotated version)::

    {
      "name":     "unique-slug",
      "source":   "where the observation came from",
      "requires": ["abilities", "stat-stages"],
      "teams":    [[set, ...], [set, ...]],
      "select":   [[0, 1, 2], [0, 1, 2]],
      "setup":    [{"do": "set_hp", "side": 1, "slot": 0, "value": 1}],
      "turns":    [["move:0", "move:0"]],
      "expect":   [{"check": "winner", "value": 0}]
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pkcm.data.dex import Dex, Stat, load_dex
from pkcm.engine.actions import Action
from pkcm.engine.battle import step
from pkcm.engine.events import Event
from pkcm.engine.pokemon import PokemonSet
from pkcm.engine.state import BattleConfig, BattleState, Phase, new_battle

#: Mechanics the engine implements today. A scenario requiring anything outside
#: this set is skipped. Add a name here in the same commit that implements it --
#: that is what makes the pending observations switch on.
IMPLEMENTED: frozenset[str] = frozenset(
    {
        "accuracy",
        "crit",
        "damage",
        "fainting",
        "move-order",
        "pp",
        "priority",
        "speed",
        "struggle",
        "switching",
        "team-preview",
        "type-chart",
        "sp-stats",
    }
)

SCENARIO_DIR = Path(__file__).resolve().parents[3] / "tests" / "scenarios"


class ScenarioError(AssertionError):
    pass


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    path: Path
    source: str
    requires: tuple[str, ...]
    teams: tuple[tuple[dict, ...], tuple[dict, ...]]
    select: tuple[tuple[int, ...], tuple[int, ...]]
    turns: tuple[tuple[str, str], ...]
    expect: tuple[dict, ...]
    setup: tuple[dict, ...] = ()
    notes: str = ""
    battle_format: str = "singles"

    @property
    def missing_requirements(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.requires) - IMPLEMENTED))

    @property
    def runnable(self) -> bool:
        return not self.missing_requirements


def load_scenarios(directory: Path = SCENARIO_DIR) -> list[Scenario]:
    scenarios = []
    for path in sorted(directory.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        scenarios.append(
            Scenario(
                name=raw["name"],
                path=path,
                source=raw.get("source", "unknown"),
                requires=tuple(raw.get("requires", ())),
                teams=tuple(tuple(side) for side in raw["teams"]),  # type: ignore[arg-type]
                select=tuple(tuple(order) for order in raw["select"]),  # type: ignore[arg-type]
                turns=tuple(tuple(pair) for pair in raw.get("turns", ())),  # type: ignore[arg-type]
                expect=tuple(raw.get("expect", ())),
                setup=tuple(raw.get("setup", ())),
                notes=raw.get("notes", ""),
                battle_format=raw.get("format", "singles"),
            )
        )
    return scenarios


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


def parse_action(text: str) -> Action:
    """``"move:0"``, ``"switch:2"``, ``"struggle"``, ``"pass"``."""
    head, _, tail = text.partition(":")
    if head == "move":
        return Action.move(int(tail))
    if head == "switch":
        return Action.switch(int(tail))
    if head == "struggle":
        return Action.struggle()
    if head == "pass":
        return Action.PASS
    raise ScenarioError(f"unknown action {text!r}")


def _build_set(entry: dict) -> PokemonSet:
    return PokemonSet(
        species=entry["species"],
        ability=entry.get("ability", ""),
        moves=tuple(entry.get("moves", ())),
        nature=entry.get("nature", "serious"),
        sp=tuple(entry.get("sp", (0, 0, 0, 0, 0, 0))),  # type: ignore[arg-type]
        item=entry.get("item"),
    )


def _apply_setup(state: BattleState, operations: tuple[dict, ...]) -> None:
    for operation in operations:
        what = operation["do"]
        if what == "set_hp":
            state.sides[operation["side"]].hp[operation["slot"]] = operation["value"]
        elif what == "set_pp":
            state.sides[operation["side"]].pp[operation["slot"]][operation["move"]] = operation["value"]
        else:
            raise ScenarioError(f"unknown setup operation {what!r}")


@dataclass(slots=True)
class ScenarioRun:
    state: BattleState
    log: list[Event] = field(default_factory=list)


def run(scenario: Scenario, dex: Dex | None = None) -> ScenarioRun:
    dex = dex or load_dex()
    config = BattleConfig(
        dex=dex,
        regulation=dex.regulation("m_b"),
        battle_format=scenario.battle_format,
    )
    teams = tuple(tuple(_build_set(entry) for entry in side) for side in scenario.teams)
    state = new_battle(config, teams, seed=0)  # type: ignore[arg-type]

    state, log = step(
        state,
        Action.select(*scenario.select[0]),
        Action.select(*scenario.select[1]),
    )
    _apply_setup(state, scenario.setup)

    for pair in scenario.turns:
        if state.finished:
            break
        state, turn_log = step(state, parse_action(pair[0]), parse_action(pair[1]))
        log.extend(turn_log)

    return ScenarioRun(state, log)


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def _active_slot(state: BattleState, check: dict) -> int:
    return check.get("slot", state.sides[check["side"]].active)


def verify(scenario: Scenario, result: ScenarioRun) -> list[str]:
    """Every expectation the run failed. Empty list means the scenario passed."""
    failures: list[str] = []
    state = result.state

    for check in scenario.expect:
        kind = check["check"]
        expected = check.get("value")

        if kind == "species":
            slot = _active_slot(state, check)
            actual = state.pokemon(check["side"], slot).species.id
        elif kind == "ability":
            slot = _active_slot(state, check)
            actual = state.pokemon(check["side"], slot).ability
        elif kind == "hp":
            actual = state.sides[check["side"]].hp[_active_slot(state, check)]
        elif kind == "fainted":
            actual = state.sides[check["side"]].is_fainted(_active_slot(state, check))
        elif kind == "active":
            actual = state.sides[check["side"]].active
        elif kind == "stat":
            slot = _active_slot(state, check)
            actual = state.pokemon(check["side"], slot).stats[Stat[check["stat"].upper()]]
        elif kind == "boost":
            side = state.sides[check["side"]]
            boosts = getattr(side, "boosts", None)
            if boosts is None:
                failures.append(f"{scenario.name}: 'boost' needs stat stages, not implemented yet")
                continue
            actual = boosts[_active_slot(state, check)][Stat[check["stat"].upper()]]
        elif kind == "winner":
            actual = state.winner
        elif kind == "phase":
            actual = state.phase.name.lower()
            expected = str(expected).lower()
        elif kind == "event":
            actual = _has_event(result.log, check)
            expected = check.get("value", True)
        elif kind == "first_mover":
            actual = next((e.side for e in result.log if e.kind == "move_used"), None)
        else:
            failures.append(f"{scenario.name}: unknown check {kind!r}")
            continue

        if actual != expected:
            label = check.get("note") or kind
            failures.append(f"{scenario.name}: {label} -- expected {expected!r}, got {actual!r}")

    return failures


def _has_event(log: list[Event], check: dict) -> bool:
    fields = {k: v for k, v in check.items() if k not in ("check", "value", "note")}
    for event in log:
        if all(getattr(event, key, None) == value for key, value in fields.items()):
            return True
    return False


def check_scenario(scenario: Scenario, dex: Dex | None = None) -> list[str]:
    return verify(scenario, run(scenario, dex))
