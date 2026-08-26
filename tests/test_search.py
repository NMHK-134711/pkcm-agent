"""The search, and the one property that makes it worth having.

A search that reads the opponent's hidden fields will look extremely strong in
self-play and fall over against anyone real. So the tests here are less about
"does it find good moves" -- that is what ``scripts/arena.py`` measures -- and
more about "is it playing the game everyone else is playing".
"""

from __future__ import annotations

import pytest

from pkcm.data.dex import load_dex
from pkcm.engine.actions import Action, ActionKind
from pkcm.engine.battle import step
from pkcm.engine.legality import random_team
from pkcm.engine.pokemon import PokemonSet
from pkcm.engine.rng import Rng
from pkcm.engine.state import BattleConfig, Phase, legal_actions, new_battle
from pkcm.envs.observation import Observation, determinize
from pkcm.search import MCTS, GreedyPolicy, RandomPolicy, SearchConfig
from pkcm.search.evaluate import heuristic, terminal_value
from pkcm.search.policy import joint_actions, play_out


@pytest.fixture(scope="module")
def dex():
    return load_dex()


def battle(dex, battle_format="singles", seed=5):
    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"),
                          battle_format=battle_format)
    teams = tuple(
        random_team(dex, config.regulation, Rng.from_seed(seed + offset).cursor(),
                    battle_format)
        for offset in (1, 2)
    )
    state = new_battle(config, teams, seed=seed)
    brought = tuple(range(config.brought))
    return step(state, Action.select(*brought), Action.select(*brought))[0]


# --------------------------------------------------------------------------- #
# Determinizing -- what the search is allowed to know
# --------------------------------------------------------------------------- #


def test_determinize_does_not_copy_the_hidden_team(dex):
    """The one that matters. Keeping the real species would give a search that
    plays perfectly against the team in front of it and transfers nothing."""
    state = battle(dex)
    observation = Observation.of(state, 0)
    real_bench = tuple(state.parties[1][state.sides[1].selection[slot]].species.id
                       for slot in (1, 2))

    guesses = set()
    for seed in range(40):
        sampled = determinize(observation, state, Rng.from_seed(seed).cursor())
        guesses.add(tuple(sampled.parties[1][sampled.sides[1].selection[slot]].species.id
                          for slot in (1, 2)))
    assert len(guesses) > 3, "it is copying, not sampling"
    assert real_bench not in guesses or len(guesses) > 5, (
        "landing on the truth is fine; only ever landing on it is not")


def test_determinize_draws_the_bench_from_the_registered_six(dex):
    """Team preview is why that pool is known, and it is the only pool."""
    state = battle(dex)
    observation = Observation.of(state, 0)
    registered = set(observation.registered[1])

    for seed in range(20):
        sampled = determinize(observation, state, Rng.from_seed(seed).cursor())
        for slot in range(len(sampled.sides[1].selection)):
            species = sampled.parties[1][sampled.sides[1].selection[slot]].species.id
            assert species in registered, species


def test_determinize_resamples_the_item_and_the_spread(dex):
    """Both were being copied from the truth, which is the quiet kind of leak."""
    state = battle(dex)
    observation = Observation.of(state, 0)
    real = state.parties[1][state.sides[1].selection[0]]

    items, spreads = set(), set()
    for seed in range(30):
        sampled = determinize(observation, state, Rng.from_seed(seed).cursor())
        lead = sampled.parties[1][sampled.sides[1].selection[0]]
        items.add(lead.item)
        spreads.add(lead.set.sp)
    assert len(items) > 3, "the item is not being resampled"
    assert len(spreads) > 3, "the spread is not being resampled"
    assert items != {real.item}


def test_determinize_keeps_what_was_actually_seen(dex):
    """Sampling is only honest if it stays consistent with the observation."""
    state = battle(dex)
    from pkcm.engine.battle import make_context
    from pkcm.engine.moves import use_move

    ctx = make_context(state)
    their_move = state.moves(1, 0)[0]
    use_move(ctx, (1, 0), their_move, defender=(0, 0))

    observation = Observation.of(state, 0)
    assert their_move.id in observation.foe[0].moves

    for seed in range(10):
        sampled = determinize(observation, state, Rng.from_seed(seed).cursor())
        lead = sampled.parties[1][sampled.sides[1].selection[0]]
        assert lead.species.id == observation.foe[0].species_id, "the seen species"
        assert their_move.id in {move.id for move in lead.moves}, "the seen move"


def test_a_determinized_state_keeps_the_observed_health(dex):
    """Species differ in maximum HP, so an absolute number would be nonsense."""
    state = battle(dex)
    state.sides[1].hp[0] = state.sides[1].hp[0] // 2
    observation = Observation.of(state, 0)

    for seed in range(10):
        sampled = determinize(observation, state, Rng.from_seed(seed).cursor())
        lead = sampled.parties[1][sampled.sides[1].selection[0]]
        fraction = sampled.sides[1].hp[0] / lead.max_hp
        assert abs(fraction - observation.foe[0].hp_fraction) < 0.02
        assert sampled.sides[1].hp[0] <= lead.max_hp


# --------------------------------------------------------------------------- #
# The tree
# --------------------------------------------------------------------------- #


def test_the_search_returns_something_legal(dex):
    state = battle(dex)
    result = MCTS(SearchConfig(iterations=60, determinizations=4)).choose(
        state, 0, Rng.from_seed(3).cursor())
    assert result.action in joint_actions(state, 0)
    assert result.iterations > 0


def test_the_root_distribution_is_a_distribution(dex):
    state = battle(dex)
    result = MCTS(SearchConfig(iterations=120, determinizations=6)).choose(
        state, 0, Rng.from_seed(4).cursor())
    total = sum(share for _, share in result.distribution)
    assert total == pytest.approx(1.0)
    assert all(share >= 0 for _, share in result.distribution)
    assert result.action == max(result.distribution, key=lambda pair: pair[1])[0]


def test_a_forced_decision_costs_no_thinking(dex):
    """One legal option means nothing to search, and searching it is waste."""
    state = battle(dex)
    while state.phase is not Phase.FORCED_SWITCH:
        state.sides[0].hp[state.sides[0].active[0]] = 1
        state, _ = step(state, legal_actions(state, 0, 0)[0], legal_actions(state, 1, 0)[0])
        if state.finished:
            pytest.skip("the battle ended before a forced switch")

    result = MCTS(SearchConfig(iterations=200)).choose(state, 1, Rng.from_seed(1).cursor())
    if len(joint_actions(state, 1)) == 1:
        assert result.iterations == 0


def test_the_search_plays_both_formats(dex):
    for battle_format in ("singles", "doubles"):
        state = battle(dex, battle_format)
        result = MCTS(SearchConfig(iterations=60, determinizations=4,
                                   max_branching=12)).choose(
            state, 0, Rng.from_seed(2).cursor())
        assert len(result.action) == state.config.active_count
        assert result.action in joint_actions(state, 0, 12)


def test_deeper_search_changes_its_mind_sometimes(dex):
    """If the budget never matters, the tree is not doing anything."""
    changed = 0
    for seed in range(8):
        state = battle(dex, seed=seed * 3 + 1)
        shallow = MCTS(SearchConfig(iterations=40, determinizations=4)).choose(
            state, 0, Rng.from_seed(seed).cursor())
        deep = MCTS(SearchConfig(iterations=400, determinizations=10)).choose(
            state, 0, Rng.from_seed(seed).cursor())
        changed += shallow.action != deep.action
    assert changed > 0, "the budget made no difference in eight positions"


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def test_the_heuristic_agrees_with_the_scoreboard(dex):
    state = battle(dex)
    assert heuristic(state, 0) == pytest.approx(0.0, abs=0.05), "an even start"

    state.sides[1].hp[1] = 0
    state.sides[1].hp[2] = 0
    assert heuristic(state, 0) > 0.2, "two up should read clearly ahead"
    assert heuristic(state, 0) == pytest.approx(-heuristic(state, 1))


def test_a_living_pokemon_beats_the_hp_it_is_standing_on(dex):
    """Otherwise the search trades its last Pokemon for chip damage."""
    state = battle(dex)
    one_left = state.clone()
    one_left.sides[0].hp[1] = 0
    one_left.sides[0].hp[2] = 0

    spread_thin = state.clone()
    for slot in range(3):
        spread_thin.sides[0].hp[slot] = max(1, spread_thin.sides[0].hp[slot] // 3)

    assert heuristic(spread_thin, 0) > heuristic(one_left, 0)


def test_terminal_value_matches_what_the_environment_pays(dex):
    state = battle(dex)
    state.winner, state.phase = 0, Phase.FINISHED
    assert terminal_value(state, 0) == 1.0
    assert terminal_value(state, 1) == -1.0
    state.winner = None
    assert terminal_value(state, 0) == 0.0, "a draw is nothing, not half a win"


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #


def test_joint_actions_refuses_to_send_one_pokemon_to_two_places(dex):
    state = battle(dex, "doubles")
    for choice in joint_actions(state, 0):
        switching = [action.index for action in choice
                     if action.kind is ActionKind.SWITCH]
        assert len(switching) == len(set(switching))


def test_every_joint_action_is_accepted_by_the_engine(dex):
    """The search only ever proposes things ``step`` will take."""
    for battle_format in ("singles", "doubles"):
        state = battle(dex, battle_format)
        theirs = joint_actions(state, 1)[0]
        for choice in joint_actions(state, 0)[:12]:
            step(state, choice, theirs)


def test_greedy_beats_random_by_a_lot(dex):
    """The floor. If this is close, the damage calculator is not working."""
    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"),
                          battle_format="singles")
    wins = 0
    played = 0
    for match in range(12):
        teams = tuple(random_team(dex, config.regulation,
                                  Rng.from_seed(match * 2 + offset).cursor())
                      for offset in (1, 2))
        for swap in (False, True):
            greedy = GreedyPolicy.seeded(match)
            chance = RandomPolicy.seeded(match + 999)
            policies = (chance, greedy) if swap else (greedy, chance)
            state = play_out(new_battle(config, teams, seed=match), policies)
            greedy_side = 1 if swap else 0
            if state.winner is not None:
                played += 1
                wins += state.winner == greedy_side
    assert wins / played > 0.65, f"greedy won only {wins}/{played}"


# --------------------------------------------------------------------------- #
# The team pick -- the largest decision in the game, and it was made at random
# --------------------------------------------------------------------------- #


def preview(dex, battle_format="singles", seed=5):
    """A state stopped at team preview, before anything is brought."""
    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"),
                          battle_format=battle_format)
    teams = tuple(
        random_team(dex, config.regulation, Rng.from_seed(seed + offset).cursor(),
                    battle_format)
        for offset in (1, 2)
    )
    return new_battle(config, teams, seed=seed)


@pytest.mark.parametrize("battle_format", ("singles", "doubles"))
def test_every_pokemon_can_be_led_with(dex, battle_format):
    """The regression this exists for.

    ``joint_actions`` truncates to the most promising options, and team picks
    all scored zero, so the sort was a no-op and the cap kept whichever
    ``permutations`` emitted first -- every one of which starts with Pokemon 0
    or 1. The search was not choosing its lead badly. It could not choose it.
    """
    state = preview(dex, battle_format)
    kept = joint_actions(state, 0, 24)
    leads = {choice[0].selection[0] for choice in kept}
    assert len(leads) > 2, f"can only lead with {sorted(leads)}"


def test_a_pick_is_scored_by_the_matchup_not_by_its_index(dex):
    """The ordering has to come from the Pokemon, not from where it sits."""
    from pkcm.search.policy import _pick_promise

    state = preview(dex)
    every = joint_actions(state, 0, None)
    scores = [_pick_promise(state, 0, choice[0].selection) for choice in every]
    assert max(scores) - min(scores) > 0.3, "the prior cannot tell picks apart"


def test_the_pick_prior_reads_only_what_preview_shows(dex):
    """Both registered sixes are public at preview -- that is the whole point of
    the phase -- but nothing below them is, and a prior that peeked at the
    opponent's brought three would be reading the future."""
    from pkcm.search.policy import _pick_promise

    state = preview(dex)
    assert not state.sides[1].selection, (
        "the opponent has not chosen yet, so there is nothing to peek at")
    assert isinstance(_pick_promise(state, 0, (0, 1, 2)), float)


def test_a_prior_does_not_flatten_when_every_option_is_bad(dex):
    """Clamping negatives at a floor made every option identical exactly when
    the choice between them mattered most."""
    from pkcm.search.policy import prior_over

    state = preview(dex)
    options = joint_actions(state, 0, 24)
    weights = prior_over(state, 0, options)
    assert pytest.approx(1.0) == sum(weights)
    assert max(weights) > 1.5 * min(weights), "the prior is flat"


# --------------------------------------------------------------------------- #
# Comparing a Q against an exploration bonus needs them on one scale
# --------------------------------------------------------------------------- #


def test_min_max_puts_the_best_line_at_one_and_the_worst_at_zero():
    from pkcm.search.mcts import MinMax

    bounds = MinMax()
    for value in (-0.2, 0.0, 0.034):
        bounds.add(value)
    assert bounds.scale(-0.2, 0) == pytest.approx(0.0)
    assert bounds.scale(0.034, 0) == pytest.approx(1.0)


def test_min_max_mirrors_the_range_for_the_other_side():
    """Side 1 accumulates the negation, so its means live in the mirrored range
    and scaling them with side 0's formula would push them outside [0, 1]."""
    from pkcm.search.mcts import MinMax

    bounds = MinMax()
    bounds.add(-0.2)
    bounds.add(0.034)
    # Side 1's mean of +0.2 means the value was -0.2, the best thing side 1
    # found, so it scales to 1. The mirror, not a sign flip on the answer.
    assert bounds.scale(0.2, 1) == pytest.approx(1.0)
    assert bounds.scale(-0.034, 1) == pytest.approx(0.0)


def test_min_max_leaves_a_value_alone_until_there_is_a_range():
    """One value seen is not a range, and inventing a span would be worse than
    admitting there is not one yet."""
    from pkcm.search.mcts import MinMax

    bounds = MinMax()
    assert bounds.scale(0.5, 0) == 0.5
    bounds.add(0.3)
    assert bounds.scale(0.3, 0) == 0.3
