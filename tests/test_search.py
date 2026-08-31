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


def test_the_pick_prior_cannot_see_their_moves_or_their_spread(dex):
    """The rule this whole file is arranged around, at the one phase where it
    is easiest to break.

    Preview shows six species. It does not show their movesets and it never
    shows their SP spread. A prior fed those would pick brilliantly in
    self-play and transfer nothing -- so swapping the opponent's sets for
    different ones must not move the score by a hair.
    """
    from dataclasses import replace as _replace

    from pkcm.search.policy import _pick_promise

    state = preview(dex)
    before = [_pick_promise(state, 0, choice[0].selection)
              for choice in joint_actions(state, 0, None)]

    def rewritten(mon):
        # Same species, different everything the opponent is allowed to hide.
        return _replace(mon,
                        moves=tuple(reversed(mon.moves)),
                        stats=tuple(value + 25 for value in mon.stats))

    theirs = tuple(rewritten(mon) for mon in state.parties[1])
    altered = _replace(state, parties=(state.parties[0], theirs))
    after = [_pick_promise(altered, 0, choice[0].selection)
             for choice in joint_actions(altered, 0, None)]

    assert before == after, "the pick prior is reading their set"


def test_the_pick_prior_does_see_their_species(dex):
    """The other half. Species *is* public at preview, and a prior that ignored
    it would be no better than the zero it replaced."""
    from dataclasses import replace as _replace

    from pkcm.search.policy import _pick_promise

    state = preview(dex, seed=5)
    other = preview(dex, seed=41)
    assert ({mon.species.id for mon in state.parties[1]}
            != {mon.species.id for mon in other.parties[1]})

    before = _pick_promise(state, 0, (0, 1, 2))
    after = _pick_promise(_replace(state, parties=(state.parties[0],
                                                   other.parties[1])), 0, (0, 1, 2))
    assert before != after, "the same pick scores the same against any opponent"



# --------------------------------------------------------------------------- #
# Leaf batching -- one forward for many leaves, virtual loss in between
# --------------------------------------------------------------------------- #


class CountingEvaluator:
    """A stand-in network that counts its forwards and answers deterministically.

    The value is a hash of the turn so different states get different numbers,
    and ``calls``/``rows`` say how the search asked: many small forwards or few
    batched ones.
    """

    def __init__(self):
        self.calls = 0
        self.rows = 0
        self.trust = 1.0

    def reset(self):
        pass

    def _one(self, state, player):
        import numpy as np
        width = 512
        value = ((state.turn * 37 + player * 11) % 13 - 6) / 10.0
        return np.full(width, 1.0 / width), value

    def look_many(self, pairs):
        self.calls += 1
        self.rows += len(pairs)
        return [self._one(state, player) for state, player in pairs]

    def prior(self, state, player, options):
        probabilities, _ = self._one(state, player)
        return self.prior_from(probabilities, state, player, options)

    def prior_from(self, probabilities, state, player, options):
        return [1.0 / max(1, len(options))] * len(options)

    def value(self, state, player):
        self.calls += 1
        self.rows += 1
        _, value = self._one(state, player)
        return value

    def value_from(self, value, state, player):
        return float(value)


def test_batched_search_pays_one_forward_per_batch(dex):
    """The point of the whole change: 64 simulations at leaf_batch 16 must ask
    the network a handful of times, not sixty-four."""
    from dataclasses import replace

    state = battle(dex)
    counting = CountingEvaluator()
    config = SearchConfig(iterations=64, determinizations=4, leaf_batch=16)
    result = MCTS(config, evaluator=counting).choose(
        state, 0, Rng.from_seed(3).cursor())

    assert result.iterations == 64
    assert counting.calls <= 8, (
        f"{counting.calls} forwards for 64 simulations -- batching is not batching")
    assert counting.rows >= 64, "every leaf still has to be evaluated"


def test_batched_statistics_are_clean_after_the_batch(dex):
    """Virtual loss must be fully refunded: root counts sum to the simulation
    budget, and no total is left carrying in-flight pessimism."""
    import math as _math

    state = battle(dex)
    config = SearchConfig(iterations=48, determinizations=4, leaf_batch=12)
    search = MCTS(config, evaluator=CountingEvaluator())

    root = None
    original = search._node

    def keep(s, p):
        nonlocal root
        node = original(s, p)
        if root is None:
            root = node
        return node

    search._node = keep
    search.choose(state, 0, Rng.from_seed(5).cursor())

    assert sum(root.counts[0]) == 48
    assert root.visits == 48
    for side in (0, 1):
        for index, count in enumerate(root.counts[side]):
            mean = root.totals[side][index] / count if count else 0.0
            assert _math.isfinite(mean) and -1.5 < mean < 1.5, (
                f"side {side} option {index}: mean {mean} -- a virtual loss "
                "was never refunded")


def test_batch_of_one_is_the_sequential_search(dex):
    """leaf_batch=1 must take the untouched sequential path, bit for bit."""
    state = battle(dex)
    counting = CountingEvaluator()
    config = SearchConfig(iterations=24, determinizations=4, leaf_batch=1)
    result = MCTS(config, evaluator=counting).choose(
        state, 0, Rng.from_seed(7).cursor())
    assert result.iterations == 24
    # The sequential path calls value()/prior() per leaf, never look_many.
    assert counting.rows >= 24


def test_batched_and_sequential_agree_on_a_lopsided_position(dex):
    """Virtual loss changes exploration, not conclusions. On a position where
    one side is nearly dead, both searches must read the same sign."""
    from pkcm.search.policy import RandomPolicy as RP, play_out as po

    state = battle(dex)
    policy = RP(Rng.from_seed(11).cursor())
    state = po(state, (policy, policy), turn_limit=10)
    if state.finished:
        return  # nothing to search

    results = {}
    for batch in (1, 16):
        counting = CountingEvaluator()
        config = SearchConfig(iterations=96, determinizations=6, leaf_batch=batch)
        results[batch] = MCTS(config, evaluator=counting).choose(
            state, 0, Rng.from_seed(13).cursor())
    # Same evaluator, same budget: root values must land in the same region.
    assert abs(results[1].value - results[16].value) < 0.6


def test_it_switches_into_the_immunity(dex):
    """hk's position, and the one a person would name to test this.

    Glimmora is Rock/Poison and takes Earthquake at 4x; Corviknight is
    Flying/Steel and takes it at 0. Garchomp is out holding Earthquake. A
    player switches, and so should this.

    Worth having as a test rather than a one-off because it exercises the whole
    chain at once -- the belief samples a Garchomp set that has Earthquake, the
    leaf evaluation sees what the exchange costs, and the prior has to not bury
    the switch. Any of the three regressing shows up here.
    """
    from pkcm.engine.actions import Action, ActionKind
    from pkcm.engine.battle import step
    from pkcm.engine.pokemon import PokemonSet
    from pkcm.engine.rng import Rng
    from pkcm.engine.state import BattleConfig, new_battle
    from pkcm.search import MCTS, SearchConfig

    def a_set(species, ability, moves, item=None, sp=(0,) * 6, nature="serious"):
        return PokemonSet(species=species, ability=ability, moves=tuple(moves),
                          item=item, nature=nature, sp=sp)

    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"),
                          battle_format="singles")
    filler = [a_set(name, "__none__", ("tackle",))
              for name in ("pikachu", "gengar", "alakazam")]
    ours = [
        a_set("glimmora", "toxicdebris",
              ("powergem", "sludgewave", "energyball", "earthpower"),
              "focussash", (0, 0, 0, 32, 2, 32), "modest"),
        a_set("corviknight", "pressure",
              ("bravebird", "ironhead", "roost", "bulkup"),
              "leftovers", (32, 0, 32, 0, 2, 0), "impish"),
        a_set("dragonite", "multiscale",
              ("dragonclaw", "earthquake", "roost", "dragondance"),
              "sitrusberry", (0, 32, 0, 0, 2, 32), "adamant"),
    ]
    theirs = [
        a_set("garchomp", "roughskin",
              ("earthquake", "dragonclaw", "scaleshot", "firefang"),
              "focussash", (0, 32, 2, 0, 0, 32), "jolly"),
        a_set("snorlax", "thickfat", ("bodyslam", "crunch", "rest", "curse"),
              "leftovers", (32, 32, 2, 0, 0, 0), "adamant"),
        a_set("starmie", "naturalcure",
              ("hydropump", "icebeam", "psychic", "recover"),
              "lifeorb", (0, 0, 0, 32, 2, 32), "timid"),
    ]
    state = new_battle(config, (tuple(ours + filler), tuple(theirs + filler)),
                       seed=11)
    state, _ = step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))

    search = SearchConfig(iterations=800, determinizations=40)
    result = MCTS(search).choose(state, 0, Rng.from_seed(5).cursor())
    assert result.action[0].kind is ActionKind.SWITCH, (
        "stayed in against a 4x Earthquake with the immunity on the bench")
    switching = sum(share for choice, share in result.distribution
                    if choice and choice[0].kind is ActionKind.SWITCH)
    assert switching > 0.5, f"only {switching:.0%} of visits went to a switch"


# --------------------------------------------------------------------------- #
# Mega Evolution, and the forme the estimator is looking at
# --------------------------------------------------------------------------- #


def test_a_stone_holder_is_scored_as_the_forme_it_becomes(dex):
    """Base Starmie is a fast special attacker; Mega Starmie is a physical one
    with Huge Power. A set built for the Mega -- Adamant, four physical moves --
    is incoherent read as the base, which is exactly how it got benched."""
    from pkcm.envs.analysis import fought_as

    sp = (2, 32, 0, 0, 0, 32)
    species, stats, types = fought_as(dex, "starmie", "starminite",
                                      "naturalcure", sp, "adamant")
    assert species == "starmiemega"
    plain = fought_as(dex, "starmie", None, "naturalcure", sp, "adamant")
    assert plain[0] == "starmie"
    assert stats[1] == plain[1][1] * 2 + 56, (
        "Huge Power doubles the Mega's Attack, not the base forme's")
    assert stats[1] > 300 and plain[1][1] < 150

    # Staraptor is the other half of it: the forme changes type, so the same
    # move is same-type after Mega Evolving and not before.
    _, _, base_types = fought_as(dex, "staraptor", None, "reckless",
                                 sp, "adamant")
    _, _, mega_types = fought_as(dex, "staraptor", "staraptite", "reckless",
                                 sp, "adamant")
    assert "fighting" not in base_types and "fighting" in mega_types


def test_a_stoneless_pokemon_is_scored_exactly_as_before(dex):
    """The resolution recomputes stats rather than adjusting them, so it has to
    reproduce ``compile_set`` when there is no stone to resolve."""
    from pkcm.engine.pokemon import PokemonSet, compile_set
    from pkcm.envs.analysis import fought_from_set

    built = compile_set(dex, PokemonSet(
        species="corviknight", ability="pressure",
        moves=("bravebird", "bodypress", "roost", "ironhead"),
        item="leftovers", nature="impish", sp=(32, 0, 32, 0, 2, 0)))
    species, stats, types = fought_from_set(dex, built)
    assert species == "corviknight"
    assert stats == tuple(built.stats)
    assert types == built.species.types


def test_mega_evolving_changes_what_the_prior_thinks_of_a_move(dex):
    """hk's probe, and the one that named the bug.

    Base Staraptor is normal/flying with Reckless, so Close Combat is a
    self-inflicted drop off a non-STAB type. Mega Staraptor is fighting/flying
    with Contrary, so the same move is same-type and the drops are boosts.
    Nothing about the move changes -- only the forme -- and the prior scored
    ``mega+closecombat`` and ``closecombat`` identically to four decimals,
    which left the pair ordered by whichever ``legal_actions`` emitted first.
    """
    from pkcm.engine.actions import Action
    from pkcm.engine.battle import step
    from pkcm.engine.legality import mega_stone_for
    from pkcm.engine.pokemon import PokemonSet
    from pkcm.engine.state import BattleConfig, new_battle
    from pkcm.search.policy import _promise

    def a_set(species, ability, moves, item=None, sp=(0,) * 6, nature="serious"):
        return PokemonSet(species=species, ability=ability, moves=tuple(moves),
                          item=item, nature=nature, sp=sp)

    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"),
                          battle_format="singles")
    stone = mega_stone_for(dex, config.regulation, "staraptor")
    filler = [a_set(name, "__none__", ("tackle",))
              for name in ("pikachu", "gengar", "alakazam")]
    ours = [a_set("staraptor", "reckless",
                  ("closecombat", "bravebird", "uturn", "quickattack"),
                  stone, (2, 32, 0, 0, 0, 32), "adamant")] + filler[:2]
    # Archaludon is steel/dragon: it resists Brave Bird and folds to Fighting,
    # so the two moves are on opposite sides of the question.
    theirs = [a_set("archaludon", "stamina",
                    ("flashcannon", "dracometeor", "bodypress", "thunderwave"),
                    "assaultvest", (30, 0, 0, 32, 4, 0), "modest")] + filler[:2]
    state = new_battle(config, (tuple(ours + filler), tuple(theirs + filler)),
                       seed=3)
    state, _ = step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))

    plain = _promise(state, 0, (Action.move(0),))
    mega = _promise(state, 0, (Action.move(0, mega=True),))
    assert mega > plain, (
        f"mega+closecombat {mega:.4f} did not beat closecombat {plain:.4f}; "
        "the prior is reading the base forme")
    assert mega == pytest.approx(plain * 1.5), "the difference is the STAB"


def test_the_pick_brings_the_pokemon_the_team_is_built_around(dex):
    """The bug this whole change is about.

    ``joint_actions`` keeps the ``max_branching`` most promising of the 120
    orderings, so a Pokemon the prior scores low is not ranked low -- it is
    never expanded, and the search cannot correct what it never sees. Measured
    on the imported parties, Mega Starmie appeared in **0 of 24** candidates
    and both teams built on it finished last.
    """
    from pkcm.engine.legality import mega_stone_for, ranker_parties
    from pkcm.engine.state import BattleConfig, new_battle
    from pkcm.search.policy import joint_actions

    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"),
                          battle_format="singles")
    parties = ranker_parties()
    ours = next(p for p in parties
                if any(m.species == "starmie"
                       and m.item == mega_stone_for(dex, config.regulation, "starmie")
                       for m in p.team))
    slot = next(i for i, m in enumerate(ours.team) if m.species == "starmie")

    # Against the whole field, not one opponent: which opponents it survived
    # against is exactly the thing that used to vary, so a single fixture would
    # pass or fail on the draw.
    missing = []
    for other in parties:
        if other is ours:
            continue
        state = new_battle(config, (ours.team, other.team), seed=0)
        options = joint_actions(state, 0, limit=SearchConfig().max_branching)
        if not any(slot in choice[0].selection for choice in options):
            missing.append(other.title)
    assert not missing, (
        f"the team's own Mega did not survive truncation against "
        f"{len(missing)} of {len(parties) - 1} opponents: {missing[:3]}")


def test_the_preview_grid_and_the_pick_prior_agree_about_a_mega(dex):
    """Two copies of one sum, and the network is trained on the other one.

    Not a regression on the Mega bug -- before the fix both copies read the
    base forme and agreed, wrongly. It is the guard against *half* a fix:
    ``search.policy`` resolving the forme while ``encode_preview`` does not
    would leave a policy imitating one grid while the search runs another, and
    nothing else in the suite would notice.

    ``test_the_preview_grid_is_the_prior_the_pick_is_scored_by`` checks the
    same property on a random team, where holding a stone is a coin flip.
    """
    from pkcm.engine.legality import mega_stone_for, ranker_parties
    from pkcm.engine.state import BattleConfig, new_battle
    from pkcm.envs.encoding import encode_preview
    from pkcm.envs.observation import Observation
    from pkcm.search.policy import _matchup

    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"),
                          battle_format="singles")
    parties = ranker_parties()
    ours = next(p for p in parties
                if any(m.species == "starmie"
                       and m.item == mega_stone_for(dex, config.regulation, "starmie")
                       for m in p.team))
    slot = next(i for i, m in enumerate(ours.team) if m.species == "starmie")
    theirs = next(p for p in parties if p is not ours)

    state = new_battle(config, (ours.team, theirs.team), seed=0)
    observation = Observation.of(state, 0)
    grid = encode_preview(observation, dex)
    for foe_index, foe_id in enumerate(observation.registered[1][:6]):
        assert grid[slot * 6 + foe_index][2] == pytest.approx(
            _matchup(state, state.parties[0][slot], foe_id), abs=1e-5)


def test_a_trap_ends_when_the_trapper_leaves_the_field(dex):
    """Shadow Tag re-applies itself every turn while its Gengar is standing
    there, and nothing ever took the flag off again. An opponent trapped once
    stayed trapped for the rest of the battle -- after the Gengar had switched
    out, and after it had fainted -- which turned one switch-in into a
    permanent lock and made every Mega Gengar team look better than it is."""
    from pkcm.engine.actions import Action, ActionKind
    from pkcm.engine.battle import step
    from pkcm.engine.legality import mega_stone_for
    from pkcm.engine.pokemon import PokemonSet
    from pkcm.engine.state import BattleConfig, legal_actions, new_battle

    def a_set(species, ability, moves, item=None, sp=(0,) * 6, nature="serious"):
        return PokemonSet(species=species, ability=ability, moves=tuple(moves),
                          item=item, nature=nature, sp=sp)

    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"),
                          battle_format="singles")
    filler = [a_set(name, "__none__", ("tackle",))
              for name in ("pikachu", "alakazam", "machamp")]
    ours = [
        a_set("gengar", "cursedbody", ("shadowball", "sludgewave", "protect",
                                       "willowisp"),
              mega_stone_for(dex, config.regulation, "gengar"),
              (0, 0, 2, 32, 0, 32), "timid"),
        a_set("corviknight", "pressure", ("bravebird", "bodypress", "roost",
                                          "ironhead"),
              "leftovers", (32, 0, 32, 0, 2, 0), "impish"),
        a_set("garchomp", "roughskin", ("earthquake", "dragonclaw", "firefang",
                                        "stoneedge"),
              "sitrusberry", (2, 32, 0, 0, 0, 32), "jolly"),
    ]
    theirs = [
        a_set("snorlax", "thickfat", ("bodyslam", "crunch", "rest", "curse"),
              "leftovers", (32, 32, 2, 0, 0, 0), "adamant"),
        a_set("clefable", "magicguard", ("moonblast", "softboiled", "knockoff",
                                         "stealthrock"),
              "rockyhelmet", (32, 0, 30, 4, 0, 0), "bold"),
        a_set("archaludon", "stamina", ("flashcannon", "dracometeor",
                                        "bodypress", "thunderwave"),
              "assaultvest", (30, 0, 0, 32, 4, 0), "modest"),
    ]
    state = new_battle(config, (tuple(ours + filler), tuple(theirs + filler)),
                       seed=4)
    state, _ = step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))

    # Mega Evolve into Shadow Tag, on a Protect so nothing faints on the way.
    state, _ = step(state, Action.move(2, mega=True), Action.move(0))
    assert state.ability_id(0, state.sides[0].active[0]) == "shadowtag"
    theirs_now = state.sides[1].active[0]
    assert state.sides[1].has_volatile(theirs_now, "trapped"), "the hold is on"
    assert not any(action.kind is ActionKind.SWITCH
                   for action in legal_actions(state, 1)), "and it holds"

    # Now walk the Gengar off the field. The hold has to go with it.
    state, _ = step(state, Action.switch(1), Action.move(0))
    assert state.species_id(0, state.sides[0].active[0]) != "gengarmega"
    assert any(action.kind is ActionKind.SWITCH
               for action in legal_actions(state, 1)), (
        "still trapped by a Gengar that is no longer on the field")


def test_a_move_that_traps_is_not_released_by_the_same_rule(dex):
    """Only an ability's hold is tied to the ability standing there. Block and
    Mean Look hold on their own terms, and the release must not reach them."""
    from pkcm.engine.state import _is_trapped

    from pkcm.engine.actions import Action
    from pkcm.engine.battle import step
    from pkcm.engine.pokemon import PokemonSet
    from pkcm.engine.state import BattleConfig, new_battle

    def a_set(species, moves):
        return PokemonSet(species=species, ability="__none__",
                          moves=tuple(moves), item=None, nature="serious",
                          sp=(0,) * 6)

    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"),
                          battle_format="singles")
    six = [a_set(name, ("tackle",))
           for name in ("pikachu", "alakazam", "machamp", "snorlax", "clefable")]
    ours = tuple([a_set("gengar", ("meanlook", "shadowball"))] + six[:5])
    theirs = tuple([a_set("snorlax", ("bodyslam",))] + six[:5])
    state = new_battle(config, (ours, theirs), seed=2)
    state, _ = step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))
    state, _ = step(state, Action.move(0), Action.move(0))

    held = state.sides[1].volatiles[state.sides[1].active[0]].get("trapped")
    assert held is not None and "by" not in held, (
        "a move's hold carries no holder, and so is never released by one "
        "leaving")
    assert _is_trapped(state, 1, state.sides[1].active[0])


def test_a_berry_eaten_in_play_is_still_a_clue(dex):
    """Not only the coach's problem. Any battle where the opponent eats a berry
    left the belief looking at a Pokemon "known to hold nothing" -- and almost
    no ranker set holds nothing, so the pool emptied and every narrowing the
    item could have done was thrown away at the moment it was learned."""
    from pkcm.engine.actions import Action
    from pkcm.engine.battle import step
    from pkcm.engine.pokemon import PokemonSet
    from pkcm.engine.state import BattleConfig, new_battle
    from pkcm.envs.belief import consistent
    from pkcm.envs.observation import Observation

    def a_set(species, ability, moves, item=None, sp=(0,) * 6, nature="serious"):
        return PokemonSet(species=species, ability=ability, moves=tuple(moves),
                          item=item, nature=nature, sp=sp)

    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"),
                          battle_format="singles")
    filler = [a_set(name, "__none__", ("tackle",))
              for name in ("pikachu", "alakazam", "machamp")]
    ours = [a_set("garchomp", "roughskin",
                  ("earthquake", "dragonclaw", "firefang", "stoneedge"),
                  "choicescarf", (2, 32, 0, 0, 0, 32), "jolly")] + filler[:2]
    theirs = [a_set("archaludon", "stamina",
                    ("flashcannon", "dracometeor", "bodypress", "thunderwave"),
                    "sitrusberry", (30, 0, 0, 32, 4, 0), "modest")] + filler[:2]
    state = new_battle(config, (tuple(ours + filler), tuple(theirs + filler)),
                       seed=7)
    state, _ = step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))

    # Stand them just above the berry's threshold so one Earthquake has to take
    # them under it. Playing from full needed a line long enough that the battle
    # ended first, and a test that skips is not a test.
    slot = state.sides[1].active[0]
    state.sides[1].hp[slot] = int(state.pokemon(1, slot).max_hp * 0.55)
    state, _ = step(state, Action.move(2), Action.move(3))   # Fire Fang

    known = Observation.of(state, 0).foe[slot]
    assert known.consumed_item == "sitrusberry", (
        "the berry did not fire; the position is wrong, not the code")
    assert known.item is None, "it was eaten, so it is not held"
    assert known.item_known

    # The candidate has to carry the move we watched as well; the item is the
    # only thing under test here.
    watched = tuple(known.moves) or ("flashcannon",)
    held = a_set("archaludon", "stamina", watched, "sitrusberry")
    assert consistent(held, known), (
        "the set that holds the berry it just ate was ruled out")
    empty = a_set("archaludon", "stamina", watched, None)
    assert not consistent(empty, known), (
        "and a set holding nothing is not what we watched")
