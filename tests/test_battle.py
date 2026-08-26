"""M0 battle engine: damage, ordering, phases, purity, termination."""

from __future__ import annotations

import pytest

from pkcm.data.dex import Stat, load_dex
from pkcm.engine.actions import Action, ActionKind
from pkcm.engine.battle import IllegalActionError, make_context, step
from pkcm.engine.moves import compute_damage, rolls_crit
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


ATTACKER = (0, 0)
DEFENDER = (1, 0)


def test_ground_move_cannot_touch_a_flying_type(dex, config):
    state = build(config, a_set("garchomp", ("earthquake",)), a_set("skarmory", ("peck",)))
    ctx = make_context(state)
    damage, effectiveness = compute_damage(
        ctx, ATTACKER, DEFENDER, dex.moves["earthquake"], crit=False
    )
    assert effectiveness == 0.0
    assert damage == 0


def test_damage_stays_inside_the_analytic_bounds(dex, config):
    """Every roll must land between the min non-crit and the max crit."""
    state = build(config, a_set("garchomp", ("earthquake",)), a_set("snorlax", ("bodyslam",)))
    ctx = make_context(state)
    move = dex.moves["earthquake"]

    attack = state.pokemon(*ATTACKER).stats[Stat.ATK]
    defense = state.pokemon(*DEFENDER).stats[Stat.DEF]
    base = ((2 * 50 // 5 + 2) * move.base_power * attack // defense) // 50 + 2
    lowest = (base * 85 // 100) * 3 // 2             # no crit, min roll, STAB
    highest = (base * 3 // 2 * 100 // 100) * 3 // 2  # crit, max roll, STAB

    rolls = [compute_damage(ctx, ATTACKER, DEFENDER, move, crit)[0]
             for _ in range(2000) for crit in (False, True)]
    assert min(rolls) == lowest
    assert max(rolls) == highest


def test_crit_rate_is_one_in_twentyfour(dex, config):
    """Same as the series, crit multiplier included (hk, confirmed)."""
    state = build(config, a_set("garchomp", ("earthquake",)), a_set("snorlax", ("bodyslam",)))
    ctx = make_context(state)
    move = dex.moves["earthquake"]
    crits = sum(rolls_crit(ctx, ATTACKER, DEFENDER, move) for _ in range(4000))
    assert 100 < crits < 260, "4000 samples at 1/24 lands far from either bound"


def test_stab_and_effectiveness_are_applied(dex, config):
    """Same attacker, one STAB move and one not."""
    state = build(config, a_set("garchomp", ("earthquake", "bodyslam")),
                  a_set("snorlax", ("bodyslam",)))  # Normal: 1x, no resist
    ctx = make_context(state)

    stab = sum(compute_damage(ctx, ATTACKER, DEFENDER, dex.moves["earthquake"], False)[0]
               for _ in range(500))
    no_stab = sum(compute_damage(ctx, ATTACKER, DEFENDER, dex.moves["bodyslam"], False)[0]
                  for _ in range(500))

    # Earthquake is 100 BP with STAB, Body Slam 85 BP without: a wide, stable gap.
    assert stab > no_stab * 1.5


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
    assert after.sides[1].owes_switch() and not after.sides[0].owes_switch()
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
    # A win is either a knockout or a timer-out ruling; only the first implies
    # the loser has nothing left standing.
    losers = [side.has_lost() for side in state.sides]
    if state.winner is not None and state.turn < config.turn_limit:
        assert losers[1 - state.winner]


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


def test_unsupported_moves_are_named_not_silently_dropped(dex, config, monkeypatch):
    """Every Champions move is implemented, so this proves the guard, not a gap.

    Nothing is unsupported any more, which would quietly retire this check if it
    depended on finding an example. So one is made: Haze is hidden from the
    executor, and the engine must then say which mechanic is missing rather than
    let the move pass for one that legitimately did nothing.
    """
    from pkcm.engine import moveeffects
    from pkcm.engine.scope import move_support

    assert move_support(dex.moves["haze"]) is None, "implemented, in the normal case"

    hidden = dict(moveeffects.SPECIAL_MOVES)
    hidden.pop("haze")
    monkeypatch.setattr(moveeffects, "SPECIAL_MOVES", hidden)

    assert move_support(dex.moves["haze"]) == "effect not described by the data"

    state = build(config, a_set("gyarados", ("haze", "waterfall")),
                  a_set("snorlax", ("bodyslam",)))
    _, log = step(state, Action.move(0), Action.move(0))
    skipped = [e for e in log if e.kind == "unimplemented"]
    assert len(skipped) == 1
    assert skipped[0].move == "haze", "events carry ids; names are the renderer's job"
    assert skipped[0].detail == "effect not described by the data"


def test_every_move_champions_has_is_implemented(dex):
    """The whole list, not a sample."""
    from pkcm.engine.scope import move_support

    missing = sorted(m.id for m in dex.moves.values()
                     if dex.exists_in_champions(m) and move_support(m))
    assert missing == []


def test_declarative_moves_are_supported_now(dex):
    """Anything the data fully describes runs; nothing else claims to."""
    from pkcm.engine.scope import move_support

    assert move_support(dex.moves["swordsdance"]) is None, "boosts: {atk: 2}"
    assert move_support(dex.moves["thunderbolt"]) is None, "secondary: par"
    assert move_support(dex.moves["gigadrain"]) is None, "drain"
    assert move_support(dex.moves["bulletseed"]) is None, "multihit"
    assert move_support(dex.moves["reflect"]) is None, "sideCondition"
    assert move_support(dex.moves["gyroball"]) is None, "variable power, implemented"

    # These were blanket exclusions once. They are structural rather than
    # declarative, which was a reason to write code for them, not to skip them.
    assert move_support(dex.moves["solarbeam"]) is None, "two-turn"
    assert move_support(dex.moves["uturn"]) is None, "self-switch"
    assert move_support(dex.moves["counter"]) is None, "answers the hit it took"
    assert move_support(dex.moves["roar"]) is None, "forces a switch"
    assert move_support(dex.moves["explosion"]) is None, "self-destructing"
    assert move_support(dex.moves["outrage"]) is None, "locks in"

    # Haze, Trick and Rest were the poster children for "effect not in the
    # data". They are written out by hand now, so nothing on the Champions list
    # is unsupported.
    for move_id in ("haze", "trick", "rest", "taunt", "encore", "defog"):
        assert move_support(dex.moves[move_id]) is None, move_id
