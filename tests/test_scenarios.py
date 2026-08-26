"""Run every scenario in tests/scenarios/.

A scenario is an observation of how Pokemon Champions actually behaves, written
down so it becomes a permanent test. Scenarios needing mechanics the engine has
not implemented yet are skipped with the reason shown, so an observation can be
recorded long before the code that satisfies it exists.
"""

from __future__ import annotations

import pytest

from pkcm.data.dex import load_dex
from pkcm.testing.scenario import IMPLEMENTED, check_scenario, load_scenarios

SCENARIOS = load_scenarios()


@pytest.fixture(scope="module")
def dex():
    return load_dex()


def test_scenarios_exist():
    assert SCENARIOS, "tests/scenarios/ is empty"


def test_scenario_names_are_unique():
    names = [scenario.name for scenario in SCENARIOS]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_scenario(scenario, dex):
    if not scenario.runnable:
        pytest.skip(
            f"needs {', '.join(scenario.missing_requirements)} "
            f"(recorded from: {scenario.source})"
        )
    failures = check_scenario(scenario, dex)
    assert not failures, "\n".join(failures)


def test_pending_scenarios_are_visible():
    """A pending observation must state a requirement we know we are missing.

    Guards against a typo silently parking a scenario forever: if a requirement
    is misspelled it never appears in IMPLEMENTED and the scenario is skipped
    for the wrong reason.
    """
    known_future = {
        "abilities", "items", "status", "stat-stages", "mega-evolution",
        "transform", "weather", "terrain", "screens", "hazards", "doubles",
        "multi-hit", "two-turn", "variable-power", "secondary-effects",
    }
    for scenario in SCENARIOS:
        unknown = set(scenario.requires) - IMPLEMENTED - known_future
        assert not unknown, (
            f"{scenario.name} requires {sorted(unknown)}, which is neither "
            f"implemented nor a recognized future mechanic -- typo?"
        )
