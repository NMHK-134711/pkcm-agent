"""M0 battle engine: damage, ordering, phases, purity, termination."""

from __future__ import annotations

import pytest

from pkcm.data.dex import Stat, load_dex
from pkcm.engine.actions import Action, ActionKind
from pkcm.engine.battle import IllegalActionError, compute_damage, step
from pkcm.engine.legality import random_team
from pkcm.engine.pokemon import PokemonSet, compile_set
from pkcm.engine.rng import Rng
from pkcm.engine.state import (
    BattleConfig,
    Phase,
    legal_actions,
    new_battle,
)


@pytest.fixture(scope="module")
def dex():
    return load_dex()


@pytest.fixture(scope="module")
def config(dex):
    return BattleConfig(dex=dex, regulation=dex.regulation("m_b"), battle_format="singles")


def a_set(species: str, moves: tuple[str, ...], sp=(0, 0, 0, 0, 0, 0), nature="serious") -> PokemonSet:
    return PokemonSet(species=species, ability="__test__", moves=moves, sp=sp, nature=nature)


def build(config, lead_a, lead_b, bench=("snorlax", "pikachu", "gyarados", "skarmory")):
    """A battle already past team preview, with chosen leads facing each other."""
    filler = [a_set(s, ("tackle",) if s != "skarmory" else ("peck",)) for s in bench]
    team_a = tuple([lead_a] + filler[:2] + filler[:3])[:6]
    team_b = tuple([lead_b] + filler[:2] + filler[:3])[:6]
    # Species clause is not enforced here; these are engine fixtures, not legal teams.
    state = new_battle(config, (team_a, team_b), seed=1)
    return step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))[0]


# --------------------------------------------------------------------------- #
# Damage formula
# --------------------------------------------------------------------------- #


def test_ground_move_cannot_touch_a_flying_type(dex, config):
    chomp = compile_set(dex, a_set("garchomp", ("earthquake",)))
    skarmory = compile_set(dex, a_set("skarmory", ("peck",)))
    result = compute_damage(chomp, skarmory, dex.moves["earthquake"], Rng.from_seed(0).cursor(), dex.type_chart)
    assert result.immune
    assert result.amount == 0


def test_damage_stays_inside_the_analytic_bounds(dex):
    """Every roll must land between the min non-crit and the max crit."""
    chomp = compile_set(dex, a_set("garchomp", ("earthquake",)))
    snorlax = compile_set(dex, a_set("snorlax", ("tackle",)))
    move = dex.moves["earthquake"]

    base = ((2 * 50 // 5 + 2) * move.base_power * chomp.stats[Stat.ATK] // snorlax.stats[Stat.DEF]) // 50 + 2
    lowest = (base * 85 // 100) * 3 // 2          # no crit, min roll, STAB
    highest = (base * 3 // 2 * 100 // 100) * 3 // 2  # crit, max roll, STAB

    cursor = Rng.from_seed(99).cursor()
    seen = [compute_damage(chomp, snorlax, move, cursor, dex.type_chart) for _ in range(4000)]
    amounts = [r.amount for r in seen]

    assert min(amounts) == lowest
    assert max(amounts) == highest
    assert any(r.crit for r in seen) and not all(r.crit for r in seen)
    # 1/24 crit rate; 4000 samples puts the count far from either bound.
    assert 100 < sum(r.crit for r in seen) < 260


def test_stab_and_effectiveness_are_applied(dex):
    """Same attacker, same power, three different type interactions."""
    chomp = compile_set(dex, a_set("garchomp", ("earthquake",)))
    neutral = compile_set(dex, a_set("snorlax", ("tackle",)))  # Normal: 1x, no resist

    cursor = Rng.from_seed(3).cursor()
    stab = [compute_damage(chomp, neutral, dex.moves["earthquake"], cursor, dex.type_chart).amount
            for _ in range(500)]
    cursor = Rng.from_seed(3).cursor()
    no_stab = [compute_damage(chomp, neutral, dex.moves["bodyslam"], cursor, dex.type_chart).amount
               for _ in range(500)]

    # Earthquake is 100 BP with STAB, Body Slam 85 BP without: a wide, stable gap.
    assert sum(stab) > sum(no_stab) * 1.5


# --------------------------------------------------------------------------- #
# Ordering
# --------------------------------------------------------------------------- #


def _first_mover(log) -> int:
    return next(e.side for e in log if e.kind == "move_used")


def test_faster_pokemon_moves_first(dex, config):
    state = build(config, a_set("garchomp", ("earthquake",)), a_set("snorlax", ("bodyslam",)))
    assert state.speed(0) > state.speed(1)
    _, log = step(state, Action.move(0), Action.move(0))
    assert _first_mover(log) == 0


def test_priority_beats_speed(dex, config):
    state = build(config, a_set("garchomp", ("earthquake",)), a_set("pikachu", ("quickattack",)))
    assert state.speed(0) > state.speed(1), "Garchomp is the faster of the two"
    assert dex.moves["quickattack"].priority > dex.moves["earthquake"].priority
    _, log = step(state, Action.move(0), Action.move(0))
    assert _first_mover(log) == 1, "Quick Attack outruns raw Speed"


def test_switches_resolve_before_moves(dex, config):
    state = build(config, a_set("garchomp", ("earthquake",)), a_set("snorlax", ("bodyslam",)))
    _, log = step(state, Action.move(0), Action.switch(1))
    kinds = [e.kind for e in log if e.kind in ("switch_in", "move_used")]
    assert kinds[0] == "switch_in", kinds


# --------------------------------------------------------------------------- #
# Phases and bookkeeping
# --------------------------------------------------------------------------- #


def test_team_preview_brings_three(dex, config):
    team = tuple(a_set(s, ("tackle",)) for s in
                 ("snorlax", "pikachu", "gyarados", "skarmory", "garchomp", "gengar"))
    state = new_battle(config, (team, team), seed=0)
    assert state.phase is Phase.TEAM_PREVIEW
    assert len(legal_actions(state, 0)) == 120  # ordered 3-of-6

    after, log = step(state, Action.select(4, 0, 1), Action.select(2, 3, 5))
    assert after.phase is Phase.BATTLE
    assert after.sides[0].selection == (4, 0, 1)
    assert after.active_pokemon(0).species.name == "Garchomp"
    assert after.active_pokemon(1).species.name == "Gyarados"
    assert len(after.sides[0].hp) == 3
    assert [e.kind for e in log].count("switch_in") == 2


def test_pp_is_spent(dex, config):
    state = build(config, a_set("garchomp", ("earthquake", "dragonclaw")), a_set("snorlax", ("bodyslam",)))
    before = state.sides[0].pp[0][0]
    after, _ = step(state, Action.move(0), Action.move(0))
    assert after.sides[0].pp[0][0] == before - 1
    assert after.sides[0].pp[0][1] == state.sides[0].pp[0][1], "unused move keeps its PP"


def test_struggle_is_the_only_option_with_no_pp(dex, config):
    state = build(config, a_set("garchomp", ("earthquake",)), a_set("snorlax", ("bodyslam",)))
    state.sides[0].pp[0][0] = 0
    actions = legal_actions(state, 0)
    assert Action.struggle() in actions
    assert not any(a.kind is ActionKind.MOVE for a in actions)

    after, log = step(state, Action.struggle(), Action.move(0))
    assert any(e.kind == "recoil" and e.side == 0 for e in log), log


def test_fainting_forces_a_replacement(dex, config):
    state = build(config, a_set("garchomp", ("earthquake",)), a_set("snorlax", ("bodyslam",)))
    state.sides[1].hp[0] = 1
    after, log = step(state, Action.move(0), Action.move(0))
    assert any(e.kind == "faint" and e.side == 1 for e in log)
    assert after.phase is Phase.FORCED_SWITCH
    assert after.sides[1].must_switch and not after.sides[0].must_switch
    assert legal_actions(after, 0) == (Action.PASS,)
    assert all(a.kind is ActionKind.SWITCH for a in legal_actions(after, 1))


def test_last_pokemon_fainting_ends_the_battle(dex, config):
    state = build(config, a_set("garchomp", ("earthquake",)), a_set("snorlax", ("bodyslam",)))
    state.sides[1].hp = [1, 0, 0]
    after, log = step(state, Action.move(0), Action.move(0))
    assert after.phase is Phase.FINISHED
    assert after.winner == 0
    assert any(e.kind == "battle_end" for e in log)


# --------------------------------------------------------------------------- #
# Purity, determinism, termination -- the properties search depends on
# --------------------------------------------------------------------------- #


def test_step_does_not_mutate_its_input(dex, config):
    state = build(config, a_set("garchomp", ("earthquake",)), a_set("snorlax", ("bodyslam",)))
    snapshot = (
        [side.hp.copy() for side in state.sides],
        [[slot.copy() for slot in side.pp] for side in state.sides],
        state.turn,
        state.rng.state,
        state.phase,
    )
    step(state, Action.move(0), Action.move(0))
    assert [side.hp for side in state.sides] == snapshot[0]
    assert [[slot for slot in side.pp] for side in state.sides] == snapshot[1]
    assert (state.turn, state.rng.state, state.phase) == snapshot[2:]


def test_clone_is_independent(dex, config):
    state = build(config, a_set("garchomp", ("earthquake",)), a_set("snorlax", ("bodyslam",)))
    twin = state.clone()
    twin.sides[0].hp[0] = 1
    twin.sides[0].pp[0][0] = 0
    assert state.sides[0].hp[0] != 1
    assert state.sides[0].pp[0][0] != 0
    assert twin.parties is state.parties, "constant data is shared, not copied"


def test_illegal_actions_are_refused(dex, config):
    state = build(config, a_set("garchomp", ("earthquake",)), a_set("snorlax", ("bodyslam",)))
    with pytest.raises(IllegalActionError):
        step(state, Action.move(3), Action.move(0))
    with pytest.raises(IllegalActionError):
        step(state, Action.move(0), Action.select(0, 1, 2))


def play(dex, config, seed: int, turn_limit: int | None = None):
    """A full battle between two random policies. Returns (final state, turns)."""
    build_rng = Rng.from_seed(seed)
    teams = (
        random_team(dex, config.regulation, build_rng.cursor()),
        random_team(dex, config.regulation, Rng.from_seed(seed + 10_000).cursor()),
    )
    if turn_limit is not None:
        config = BattleConfig(dex, config.regulation, config.battle_format, turn_limit)
    state = new_battle(config, teams, seed=seed)

    policy = Rng.from_seed(seed + 777).cursor()
    steps = 0
    while not state.finished:
        actions = tuple(policy.choice(legal_actions(state, player)) for player in (0, 1))
        state, _ = step(state, *actions)
        steps += 1
        assert steps < 2000, "battle failed to terminate"
    return state, steps


@pytest.mark.parametrize("seed", range(30))
def test_random_selfplay_terminates(dex, config, seed):
    state, _ = play(dex, config, seed)
    assert state.phase is Phase.FINISHED
    assert state.winner in (0, 1, None)
    losers = [side.has_lost() for side in state.sides]
    assert state.winner is None or losers[1 - state.winner]


def test_identical_seeds_replay_identically(dex, config):
    first, steps_a = play(dex, config, 5)
    second, steps_b = play(dex, config, 5)
    assert steps_a == steps_b
    assert first.winner == second.winner
    assert [side.hp for side in first.sides] == [side.hp for side in second.sides]
    assert first.rng.state == second.rng.state


def test_turn_limit_produces_a_ruling(dex, config):
    state, _ = play(dex, config, seed=3, turn_limit=4)
    assert state.phase is Phase.FINISHED
    assert state.turn <= 5
    assert state.winner in (0, 1, None)


def test_unsupported_moves_are_named_not_silently_dropped(dex, config):
    """A move M0 cannot run must say which mechanic it needs."""
    state = build(config, a_set("garchomp", ("swordsdance", "earthquake")),
                  a_set("snorlax", ("bodyslam",)))
    _, log = step(state, Action.move(0), Action.move(0))
    skipped = [e for e in log if e.kind == "unimplemented"]
    assert len(skipped) == 1
    assert skipped[0].move == "Swords Dance"
    assert skipped[0].detail == "status move"


def test_variable_power_moves_are_not_treated_as_status(dex, config):
    """Gyro Ball has basePower 0 but is a damaging move, not a status move."""
    from pkcm.engine.scope import move_support

    assert dex.moves["gyroball"].base_power == 0
    assert dex.moves["gyroball"].category == "Physical"
    assert move_support(dex.moves["gyroball"]) == "variable base power"
    assert move_support(dex.moves["swordsdance"]) == "status move"
