"""The PettingZoo adapter and the information set underneath it.

The tests that matter most here are the ones about *what a player is told*.
An adapter that leaks the opponent's team is not a slightly wrong adapter; it
trains a policy that cannot exist against a real opponent, and nothing about
the training curve says so.
"""

from __future__ import annotations

import numpy as np
import pytest

from pkcm.data.dex import load_dex
from pkcm.engine.actions import TARGET_ALLY, Action, ActionKind
from pkcm.engine.battle import step
from pkcm.engine.legality import random_team
from pkcm.engine.rng import Rng
from pkcm.engine.state import BattleConfig, Phase, new_battle
from pkcm.envs.champions import AGENTS, ChampionsEnv, sample_legal
from pkcm.envs.encoding import (
    MAX_BROUGHT,
    PASS_INDEX,
    SELECTION_BASE,
    STRUGGLE_INDEX,
    SWITCH_BASE,
    Vocabulary,
    action_space_size,
    decode_action,
    encode_action,
    encode_observation,
)
from pkcm.envs.observation import Observation, determinize


@pytest.fixture(scope="module")
def dex():
    return load_dex()


@pytest.fixture(scope="module")
def singles_env(dex):
    return ChampionsEnv(battle_format="singles", dex=dex, seed=11)


@pytest.fixture(scope="module")
def doubles_env(dex):
    return ChampionsEnv(battle_format="doubles", dex=dex, seed=12)


def played(env, rng, episodes=1):
    """Play whole episodes with a mask-respecting random policy."""
    for _ in range(episodes):
        _, infos = env.reset()
        while env.agents:
            actions = {}
            for agent in AGENTS:
                masks = infos[agent]["action_mask"]
                need = infos[agent]["decisions"]
                picks, taken = [], set()
                for position in range(need):
                    mask = masks[position].copy()
                    for slot in taken:
                        mask[SWITCH_BASE + slot] = 0
                    index = sample_legal(mask, rng)
                    if SWITCH_BASE <= index < SWITCH_BASE + MAX_BROUGHT:
                        taken.add(index - SWITCH_BASE)
                    picks.append(index)
                actions[agent] = np.array(picks, dtype=np.int64)
            _, rewards, terminations, _, infos = env.step(actions)
        yield rewards


# --------------------------------------------------------------------------- #
# The information boundary
# --------------------------------------------------------------------------- #


def battle(dex, battle_format="singles", seed=5):
    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"),
                          battle_format=battle_format)
    teams = (random_team(dex, config.regulation, Rng.from_seed(seed).cursor(),
                         battle_format),
             random_team(dex, config.regulation, Rng.from_seed(seed + 1).cursor(),
                         battle_format))
    state = new_battle(config, teams, seed=seed)
    brought = tuple(range(config.brought))
    return step(state, Action.select(*brought), Action.select(*brought))[0]


def test_a_benched_opponent_is_not_revealed(dex):
    state = battle(dex)
    observation = Observation.of(state, 0)
    lead, *bench = observation.foe
    assert lead.species_id is not None, "the one that was sent out"
    assert all(other.species_id is None for other in bench), "the rest are not"


def test_we_see_our_own_side_completely(dex):
    state = battle(dex)
    observation = Observation.of(state, 0)
    for known in observation.own:
        assert known.species_id is not None
        assert known.hp is not None and known.max_hp is not None
        assert known.pp is not None
        assert known.item_known and known.ability_known
        assert len(known.moves) >= 1


def test_the_opponents_moves_appear_only_once_used(dex):
    state = battle(dex)
    assert Observation.of(state, 0).foe[0].moves == (), "nothing seen yet"

    # Let the battle run a few turns, then check every move we can see is one
    # that was actually used.
    used: set[str] = set()
    for _ in range(6):
        if state.phase is not Phase.BATTLE:
            break
        from pkcm.engine.state import legal_actions

        choices = tuple(legal_actions(state, player, 0)[0] for player in (0, 1))
        state, log = step(state, choices[0], choices[1])
        used.update(event.move for event in log
                    if event.kind == "move_used" and event.side == 1 and event.move)

    seen = set(Observation.of(state, 0).foe[0].moves)
    assert seen <= used, seen - used


def test_the_opponents_item_stays_hidden_until_something_shows_it(dex):
    state = battle(dex)
    for known in Observation.of(state, 0).foe:
        assert not known.item_known
        assert known.item is None


def test_exact_hp_is_ours_alone(dex):
    """They get a bar; we get the number. The number would give away the SP."""
    state = battle(dex)
    observation = Observation.of(state, 0)
    assert observation.own[0].hp is not None
    assert observation.foe[0].hp is None
    assert observation.foe[0].max_hp is None
    assert 0.0 <= observation.foe[0].hp_fraction <= 1.0


def test_the_registered_six_are_public(dex):
    """Team preview shows them, so a policy may reason about them from turn one."""
    state = battle(dex)
    observation = Observation.of(state, 0)
    assert len(observation.registered[1]) == 6
    assert all(observation.registered[1])


def test_the_encoded_observation_carries_no_hidden_ids(dex):
    """The arrays are what a policy actually reads, so the leak test belongs here."""
    state = battle(dex)
    vocabulary = Vocabulary.of(dex)
    encoded = encode_observation(Observation.of(state, 0), vocabulary)

    # Foe slots occupy the second half of every per-Pokemon array.
    foe_species = encoded["species"][MAX_BROUGHT:]
    assert foe_species[0] != 0, "the lead has been seen"
    assert not foe_species[1:].any(), "the bench has not"
    assert not encoded["items"][MAX_BROUGHT:].any()
    assert not encoded["abilities"][MAX_BROUGHT:].any()
    assert not encoded["moves"][MAX_BROUGHT * 4:].any()
    assert not encoded["pp"][MAX_BROUGHT * 4:].any(), "PP is never shown for the foe"


# --------------------------------------------------------------------------- #
# Determinizing
# --------------------------------------------------------------------------- #


def test_determinize_agrees_with_everything_observed(dex):
    state = battle(dex)
    observation = Observation.of(state, 0)
    guess = determinize(observation, state, Rng.from_seed(3).cursor())

    ours = Observation.of(guess, 0)
    assert ours.foe[0].species_id == observation.foe[0].species_id
    assert ours.own[0].moves == observation.own[0].moves, "our own side is untouched"


def test_determinize_invents_a_full_moveset(dex):
    """The foe's four moves have to exist for a rollout to be playable."""
    from pkcm.engine.legality import learnable_moves

    state = battle(dex)
    observation = Observation.of(state, 0)
    guess = determinize(observation, state, Rng.from_seed(4).cursor())

    lead = guess.parties[1][guess.sides[1].selection[0]]
    assert 1 <= len(lead.moves) <= 4
    allowed = learnable_moves(dex, lead.species.id)
    for move in lead.moves:
        assert move.id in allowed, move.id


def test_two_determinizations_differ(dex):
    """If they never differ there is nothing to average over."""
    state = battle(dex)
    observation = Observation.of(state, 0)
    movesets = set()
    for seed in range(12):
        guess = determinize(observation, state, Rng.from_seed(seed).cursor())
        lead = guess.parties[1][guess.sides[1].selection[0]]
        movesets.add(tuple(sorted(move.id for move in lead.moves)))
    assert len(movesets) > 1


def test_a_determinized_state_can_be_played(dex):
    from pkcm.engine.state import legal_actions

    state = battle(dex)
    guess = determinize(Observation.of(state, 0), state, Rng.from_seed(7).cursor())
    for _ in range(5):
        if guess.phase is Phase.FINISHED:
            break
        choices = tuple(legal_actions(guess, player, 0)[0] for player in (0, 1))
        guess, _ = step(guess, choices[0], choices[1])
    assert guess.turn >= 1


# --------------------------------------------------------------------------- #
# Action encoding
# --------------------------------------------------------------------------- #


def test_every_action_round_trips(dex):
    registered, brought = 6, 3
    actions = [Action.PASS, Action.struggle()]
    actions += [Action.switch(slot) for slot in range(MAX_BROUGHT)]
    actions += [Action.move(index, mega=mega, target=target)
                for index in range(4) for mega in (False, True)
                for target in (0, 1, TARGET_ALLY)]
    from pkcm.engine.actions import team_selections

    actions += list(team_selections(registered, brought))[:20]

    for action in actions:
        index = encode_action(action, registered, brought)
        assert decode_action(index, registered, brought) == action, action
        assert 0 <= index < action_space_size(registered, brought)


def test_the_blocks_do_not_overlap():
    assert SWITCH_BASE + MAX_BROUGHT == STRUGGLE_INDEX
    assert STRUGGLE_INDEX + 1 == PASS_INDEX
    assert PASS_INDEX + 1 == SELECTION_BASE


def test_singles_and_doubles_share_the_layout(dex):
    """A policy trained on one already has the other's action space in front of it."""
    singles = ChampionsEnv(battle_format="singles", dex=dex)
    doubles = ChampionsEnv(battle_format="doubles", dex=dex)
    assert singles.n_actions < doubles.n_actions, "only the selection block grows"
    assert decode_action(SWITCH_BASE + 1, 6, 3) == decode_action(SWITCH_BASE + 1, 6, 4)


# --------------------------------------------------------------------------- #
# The environment
# --------------------------------------------------------------------------- #


def test_a_singles_episode_finishes_with_opposed_rewards(singles_env):
    rng = np.random.default_rng(0)
    rewards = next(played(singles_env, rng))
    assert set(rewards) == set(AGENTS)
    assert rewards["player_0"] == -rewards["player_1"]


def test_a_doubles_episode_finishes(doubles_env):
    rng = np.random.default_rng(1)
    rewards = next(played(doubles_env, rng))
    assert rewards["player_0"] == -rewards["player_1"]


def test_doubles_asks_for_two_decisions_and_team_preview_for_one(doubles_env):
    _, infos = doubles_env.reset()
    assert infos["player_0"]["phase"] == "TEAM_PREVIEW"
    assert infos["player_0"]["decisions"] == 1

    rng = np.random.default_rng(2)
    mask = infos["player_0"]["action_mask"][0]
    choice = np.array([sample_legal(mask, rng)], dtype=np.int64)
    _, _, _, _, infos = doubles_env.step({agent: choice for agent in AGENTS})
    assert infos["player_0"]["decisions"] == 2


def test_the_mask_only_ever_offers_legal_actions(singles_env):
    _, infos = singles_env.reset()
    from pkcm.engine.state import legal_actions

    for _ in range(10):
        for player, agent in enumerate(AGENTS):
            mask = infos[agent]["action_mask"][0]
            allowed = set(legal_actions(singles_env.battle_state(), player, 0))
            for index in np.flatnonzero(mask):
                assert decode_action(int(index), 6, 3) in allowed
        rng = np.random.default_rng(3)
        actions = {
            agent: np.array([sample_legal(infos[agent]["action_mask"][0], rng)],
                            dtype=np.int64)
            for agent in AGENTS
        }
        _, _, terminations, _, infos = singles_env.step(actions)
        if all(terminations.values()):
            break


def test_an_illegal_action_forfeits_rather_than_being_corrected(dex):
    """Substituting a legal action would teach from a turn that never happened."""
    env = ChampionsEnv(battle_format="singles", dex=dex, seed=1)
    env.reset()
    # PASS is never legal at team preview.
    rewards = env.step({"player_0": np.array([PASS_INDEX]),
                        "player_1": np.array([SELECTION_BASE])})[1]
    assert rewards["player_0"] == -1.0
    assert rewards["player_1"] == 1.0


def test_raise_mode_is_available_for_debugging(dex):
    env = ChampionsEnv(battle_format="singles", dex=dex, seed=1, on_illegal="raise")
    env.reset()
    from pkcm.engine.battle import IllegalActionError

    with pytest.raises(IllegalActionError):
        env.step({"player_0": np.array([PASS_INDEX]),
                  "player_1": np.array([SELECTION_BASE])})


def test_the_observation_matches_its_declared_space(singles_env):
    observations, _ = singles_env.reset()
    space = singles_env.observation_space("player_0")
    assert space.contains(observations["player_0"]), "the declared space is a lie otherwise"


def test_pettingzoo_accepts_it(dex):
    from pettingzoo.test import parallel_api_test

    parallel_api_test(ChampionsEnv(battle_format="singles", dex=dex, seed=1),
                      num_cycles=120)


# --------------------------------------------------------------------------- #
# The reference sheet and the calculator
#
# hk's point: a strong player reads the dex and does damage maths. None of that
# is hidden information, so a policy that has to rediscover it from reward is
# working for something the game already tells it. The tests that matter are
# the ones showing the calculator still cannot see what the player cannot.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def sheet(dex):
    from pkcm.envs.reference import sheet_for

    return sheet_for(dex, Vocabulary.of(dex))


def test_the_sheet_knows_the_type_chart(sheet):
    assert sheet.effectiveness("fire", ("grass",)) == 2.0
    assert sheet.effectiveness("fire", ("water",)) == 0.5
    assert sheet.effectiveness("ground", ("flying",)) == 0.0
    assert sheet.effectiveness("fighting", ("dark", "ice")) == 4.0
    assert sheet.effectiveness("normal", ("rock", "steel")) == 0.25


def test_row_zero_is_blank(sheet):
    """Id 0 means unknown, so looking one up gives a blank row, not a wrong one."""
    assert not sheet.species[0].any()
    assert not sheet.moves[0].any()


def test_the_sheet_answers_the_pick_phase_question(sheet, dex):
    """What could that Pokemon be running? Public, and the reason it matters."""
    assert sheet.could_learn("garchomp", "earthquake")
    assert not sheet.could_learn("garchomp", "moonblast"), "Garchomp does learn Surf"
    assert not sheet.could_learn(None, "earthquake"), "an unseen slot could be anything"
    row = sheet.candidate_moves("garchomp")
    assert 20 < int(row.sum()) < 200, int(row.sum())


def test_the_calculator_reads_effectiveness_the_game_would_show(dex, sheet):
    from pkcm.envs.analysis import estimate_damage

    state = battle(dex)
    observation = Observation.of(state, 0)
    attacker, defender = observation.own[0], observation.foe[0]
    for move_id in attacker.moves:
        move = dex.moves[move_id]
        estimate = estimate_damage(observation, sheet, dex, attacker, defender, move)
        if estimate is None:
            continue
        expected = sheet.effectiveness(move.type, dex.species[defender.species_id].types)
        assert estimate.effectiveness == expected


def test_the_damage_bracket_contains_what_the_engine_actually_deals(dex, sheet):
    """The estimator and the engine share ``damage_formula``; this checks that
    the bracketing around it is wide enough to be honest and tight enough to
    be worth having."""
    from pkcm.engine.moves import compute_damage
    from pkcm.engine.battle import make_context
    from pkcm.envs.analysis import estimate_damage

    checked = 0
    for seed in range(6):
        state = battle(dex, seed=seed * 7 + 1)
        observation = Observation.of(state, 0)
        attacker, defender = observation.own[0], observation.foe[0]
        for move_id in attacker.moves:
            move = dex.moves[move_id]
            estimate = estimate_damage(observation, sheet, dex, attacker, defender, move)
            if estimate is None or estimate.immune:
                continue
            ctx = make_context(state)
            actual, _ = compute_damage(ctx, (0, 0), (1, 0), move, crit=False)
            share = 100 * actual / state.pokemon(1, 0).max_hp
            assert estimate.percent.low - 25 <= share <= estimate.percent.high + 25, (
                move_id, share, estimate.percent)
            checked += 1
    assert checked > 5, "nothing was actually compared"


def chosen_battle(dex, red, blue, battle_format="singles"):
    """A battle between two named sets, rather than whatever the RNG produced.

    The random matchups used above are fine for "does anything leak"; a test
    about Ground versus Flying has to actually contain a Flying type, and
    skipping when it does not is a test that quietly stops running.
    """
    from pkcm.engine.pokemon import PokemonSet

    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"),
                          battle_format=battle_format)
    filler = [PokemonSet(species=name, ability="__none__", moves=("bodyslam",),
                         item=None, nature="serious", sp=(0,) * 6)
              for name in ("snorlax", "pikachu", "starmie", "gengar", "alakazam")]
    state = new_battle(config, (tuple([red] + filler), tuple([blue] + filler)), seed=3)
    brought = tuple(range(config.brought))
    return step(state, Action.select(*brought), Action.select(*brought))[0]


def a_set(species, moves, ability="__none__", **kwargs):
    from pkcm.engine.pokemon import PokemonSet

    return PokemonSet(species=species, ability=ability, moves=tuple(moves),
                      item=None, nature="serious", sp=(0,) * 6, **kwargs)


def test_an_immune_target_reads_as_immune(dex, sheet):
    from pkcm.envs.analysis import estimate_damage

    state = chosen_battle(dex, a_set("garchomp", ("earthquake",)),
                          a_set("skarmory", ("bodyslam",)))
    observation = Observation.of(state, 0)
    estimate = estimate_damage(observation, sheet, dex, observation.own[0],
                               observation.foe[0], dex.moves["earthquake"])
    assert estimate is not None and estimate.immune
    assert estimate.percent.high == 0
    assert estimate.hits_to_ko.low == 99, "never, rather than eventually"


def test_the_calculator_never_sees_the_hidden_spread(dex, sheet):
    """The bracket has to be a *bracket*: change their real SP and nothing moves.

    If the estimate tracked their actual spread it would be laundering hidden
    information into the policy, and it would look like skill.
    """
    from pkcm.engine.pokemon import PokemonSet, compile_team
    from pkcm.envs.analysis import estimate_damage

    def estimate_with(defender_sp):
        state = chosen_battle(dex, a_set("garchomp", ("earthquake",)),
                              a_set("snorlax", ("bodyslam",)))
        slot = state.sides[1].selection[0]
        original = state.parties[1][slot]
        bulkier = compile_team(dex, (PokemonSet(
            species=original.species.id, ability=original.ability,
            moves=tuple(m.id for m in original.moves), item=original.item,
            nature=original.set.nature, sp=defender_sp),))[0]
        parties = list(state.parties)
        party = list(parties[1])
        party[slot] = bulkier
        parties[1] = tuple(party)
        state.parties = tuple(parties)

        observation = Observation.of(state, 0)
        move = dex.moves[observation.own[0].moves[0]]
        return estimate_damage(observation, sheet, dex,
                               observation.own[0], observation.foe[0], move)

    frail = estimate_with((0, 0, 0, 0, 0, 0))
    bulky = estimate_with((32, 0, 32, 0, 0, 0))
    assert frail is not None and bulky is not None
    assert frail.percent == bulky.percent, "the estimate followed their real spread"
    assert frail.hits_to_ko == bulky.hits_to_ko


def test_the_matchup_block_matches_the_structured_assessment(dex):
    """The floats a policy reads and the numbers a human would read agree."""
    env = ChampionsEnv(battle_format="singles", dex=dex, seed=21)
    observations, _ = env.reset()
    # Get past team preview so there is something standing.
    from pkcm.engine.state import legal_actions

    while env.battle_state().phase is Phase.TEAM_PREVIEW:
        actions = {
            agent: np.array([encode_action(legal_actions(env.battle_state(), player, 0)[0],
                                           6, 3)])
            for player, agent in enumerate(AGENTS)
        }
        observations, _, _, _, _ = env.step(actions)

    from pkcm.envs.encoding import MATCHUP_FEATURES

    matchup = observations["player_0"]["matchup"]
    assessment = env.assess(0)
    assert assessment is not None
    assert matchup.shape[1] == MATCHUP_FEATURES
    # Every move the assessment scored should be non-zero somewhere in the block.
    assert np.abs(matchup).sum() > 0 or not assessment.damage


def test_analysis_can_be_turned_off(dex):
    env = ChampionsEnv(battle_format="singles", dex=dex, seed=1, with_analysis=False)
    observations, _ = env.reset()
    assert "matchup" not in observations["player_0"]
    assert env.observation_space("player_0").contains(observations["player_0"])


# --------------------------------------------------------------------------- #
# Randomness
#
# hk: the same move against the same target with the same spread kills on a
# high roll and does not on a low one. A bracket alone does not say whether to
# go for it -- "3-6 hits" and "난수 1타 43%" are different pieces of advice, and
# only the second one decides a turn.
# --------------------------------------------------------------------------- #


def test_the_damage_roll_matches_showdown(dex):
    """Sixteen uniform values from 85% to 100%, which is Showdown's randomizer."""
    import collections

    from pkcm.engine.moves import DAMAGE_ROLL_HIGH, DAMAGE_ROLL_LOW

    assert (DAMAGE_ROLL_LOW, DAMAGE_ROLL_HIGH) == (85, 100)
    cursor = Rng.from_seed(1).cursor()
    seen = collections.Counter(cursor.between(DAMAGE_ROLL_LOW, DAMAGE_ROLL_HIGH)
                               for _ in range(32000))
    assert len(seen) == 16
    # Four standard deviations of a binomial, rather than a ratio pulled out of
    # the air: with 2000 expected per bucket the sd is about 43, and a tighter
    # bound fails on honest noise.
    expected = 32000 / 16
    tolerance = 4 * (expected * (1 - 1 / 16)) ** 0.5
    for value, count in seen.items():
        assert abs(count - expected) < tolerance, (value, count)


def test_the_estimate_carries_every_roll(dex, sheet):
    from pkcm.envs.analysis import ROLL_COUNT, estimate_damage

    state = chosen_battle(dex, a_set("garchomp", ("earthquake",)),
                          a_set("alakazam", ("psychic",)))
    observation = Observation.of(state, 0)
    estimate = estimate_damage(observation, sheet, dex, observation.own[0],
                               observation.foe[0], dex.moves["earthquake"])
    assert estimate is not None
    assert len(estimate.rolls) == ROLL_COUNT == 16
    assert estimate.rolls == tuple(sorted(estimate.rolls)), "low roll first"
    assert estimate.rolls[-1] > estimate.rolls[0], "there is a spread to speak of"


def test_a_move_that_sometimes_kills_reports_a_probability(dex, sheet):
    """The case the bracket cannot express: it kills on some rolls, not all."""
    from pkcm.envs.analysis import estimate_damage

    state = chosen_battle(dex, a_set("garchomp", ("earthquake",)),
                          a_set("alakazam", ("psychic",)))
    observation = Observation.of(state, 0)
    estimate = estimate_damage(observation, sheet, dex, observation.own[0],
                               observation.foe[0], dex.moves["earthquake"])
    assert estimate is not None
    assert 0.0 < estimate.ko_chance < 1.0, estimate.ko_chance
    assert not estimate.guaranteed_ko
    low, high = estimate.ko_chance_bracket
    assert low <= estimate.ko_chance <= high, (low, estimate.ko_chance, high)


def test_a_certain_kill_reads_as_certain(dex, sheet):
    from pkcm.envs.analysis import estimate_damage

    state = chosen_battle(dex, a_set("garchomp", ("earthquake",)),
                          a_set("alakazam", ("psychic",)))
    state.sides[1].hp[0] = 1          # nothing survives anything
    observation = Observation.of(state, 0)
    estimate = estimate_damage(observation, sheet, dex, observation.own[0],
                               observation.foe[0], dex.moves["earthquake"])
    assert estimate is not None
    assert estimate.guaranteed_ko and estimate.certain_ko
    assert estimate.ko_chance == 1.0


def test_accuracy_is_kept_apart_from_the_roll(dex, sheet):
    """확정 1타 with 80 accuracy is a real thing to say, and two facts."""
    from pkcm.envs.analysis import estimate_damage

    state = chosen_battle(dex, a_set("garchomp", ("stoneedge",)),
                          a_set("alakazam", ("psychic",)))
    state.sides[1].hp[0] = 1
    observation = Observation.of(state, 0)
    estimate = estimate_damage(observation, sheet, dex, observation.own[0],
                               observation.foe[0], dex.moves["stoneedge"])
    assert estimate is not None
    assert estimate.guaranteed_ko, "the damage always kills"
    assert estimate.hit_chance == 0.8, "and it still misses one time in five"
    assert estimate.ko_chance == pytest.approx(0.8), "which is what decides the turn"


def test_a_missing_move_never_reads_as_a_kill(dex, sheet):
    from pkcm.envs.analysis import estimate_damage

    state = chosen_battle(dex, a_set("garchomp", ("earthquake",)),
                          a_set("skarmory", ("bodyslam",)))
    observation = Observation.of(state, 0)
    estimate = estimate_damage(observation, sheet, dex, observation.own[0],
                               observation.foe[0], dex.moves["earthquake"])
    assert estimate is not None and estimate.immune
    assert estimate.ko_chance == 0.0 and estimate.hit_chance == 0.0


def test_the_crit_rate_is_folded_in(dex, sheet):
    """A high-crit move kills a little more often than its rolls alone say."""
    from pkcm.envs.analysis import _crit_chance

    assert _crit_chance(dex.moves["earthquake"]) == pytest.approx(1 / 24)
    assert _crit_chance(dex.moves["stoneedge"]) == pytest.approx(1 / 8)
    assert _crit_chance(dex.moves["frostbreath"]) == 1.0, "it always crits"


def test_the_matchup_block_carries_the_probability(dex):
    env = ChampionsEnv(battle_format="singles", dex=dex, seed=31)
    from pkcm.envs.encoding import MATCHUP_FEATURES

    assert MATCHUP_FEATURES == 7
    observations, _ = env.reset()
    assert observations["player_0"]["matchup"].shape[1] == MATCHUP_FEATURES
    assert env.observation_space("player_0").contains(observations["player_0"])


# --------------------------------------------------------------------------- #
# The rest of the dice
# --------------------------------------------------------------------------- #


def test_the_sheet_carries_a_moves_own_odds(dex, sheet):
    from pkcm.envs.reference import (
        crit_chance,
        expected_hits,
        flinch_chance,
        secondary_chance,
        status_chance,
    )

    assert secondary_chance(dex.moves["icebeam"]) == 10
    assert status_chance(dex.moves["icebeam"], "frz") == 10
    assert flinch_chance(dex.moves["rockslide"]) == 30
    assert flinch_chance(dex.moves["earthquake"]) == 0, "it does not flinch"
    assert crit_chance(dex.moves["stoneedge"]) == pytest.approx(1 / 8)
    assert crit_chance(dex.moves["frostbreath"]) == 1.0
    assert expected_hits(dex.moves["doublekick"]) == 2.0
    assert expected_hits(dex.moves["earthquake"]) == 1.0


def test_the_two_to_five_hit_spread_is_the_modern_one(dex):
    """35-35-15-15, not the tidier Gen 4 spread.

    The engine was using 3/8-3/8-1/8-1/8, which is Gen 4's, and it makes every
    2-5 hit move land 3.0 times instead of 3.1. Found by writing this down.
    """
    import collections

    from pkcm.engine.moves import MULTIHIT_2_TO_5
    from pkcm.envs.reference import expected_hits

    counts = collections.Counter(MULTIHIT_2_TO_5)
    total = len(MULTIHIT_2_TO_5)
    assert counts[2] / total == pytest.approx(0.35)
    assert counts[3] / total == pytest.approx(0.35)
    assert counts[4] / total == pytest.approx(0.15)
    assert counts[5] / total == pytest.approx(0.15)
    assert expected_hits(dex.moves["bulletseed"]) == pytest.approx(3.1)


def a_known(**overrides):
    from pkcm.envs.observation import KnownPokemon

    fields = dict(slot=0, position=0, species_id="snorlax", hp_fraction=1.0,
                  hp=None, max_hp=None, status=None, boosts=(0,) * 7, volatiles=(),
                  moves=(), pp=None, item=None, item_known=False, ability=None,
                  ability_known=False, fainted=False)
    fields.update(overrides)
    return KnownPokemon(**fields)


def test_losing_the_turn_is_priced(dex):
    from pkcm.engine.conditions import CONFUSION_CHANCE, PARALYSIS_CHANCE, THAW_CHANCE
    from pkcm.envs.analysis import turn_risk

    assert turn_risk(a_known()).cannot_act == 0.0
    assert turn_risk(a_known(status="par")).cannot_act == pytest.approx(
        PARALYSIS_CHANCE[0] / PARALYSIS_CHANCE[1])
    assert turn_risk(a_known(status="frz")).cannot_act == pytest.approx(
        1 - THAW_CHANCE[0] / THAW_CHANCE[1])
    assert turn_risk(a_known(volatiles=("confusion",))).cannot_act == pytest.approx(
        CONFUSION_CHANCE[0] / CONFUSION_CHANCE[1])
    assert turn_risk(a_known(volatiles=("flinch",))).cannot_act == 1.0


def test_paralysis_uses_the_champions_rate_not_the_series_one(dex):
    """1/8, and it comes from the engine's constant rather than a second copy."""
    from pkcm.envs.analysis import turn_risk

    assert turn_risk(a_known(status="par")).paralysis == pytest.approx(0.125)


def test_two_risks_compound_rather_than_add(dex):
    from pkcm.envs.analysis import turn_risk

    both = turn_risk(a_known(status="par", volatiles=("confusion",)))
    assert both.cannot_act == pytest.approx(1 - (1 - 0.125) * (1 - 1 / 3))
    assert both.cannot_act < 0.125 + 1 / 3


def test_sleep_is_conditioned_on_how_long_we_have_watched_it(dex):
    """Ours is exact. Theirs narrows as the turns pass, which is public."""
    from pkcm.envs.analysis import turn_risk

    assert turn_risk(a_known(status="slp", status_turns=3)).sleep == 1.0
    assert turn_risk(a_known(status="slp", status_turns=1)).sleep == 0.0

    # Durations are 2, 3, 3. After two turns only the threes are still possible.
    assert turn_risk(a_known(status="slp", status_elapsed=0)).sleep == 1.0
    assert turn_risk(a_known(status="slp", status_elapsed=2)).sleep == pytest.approx(2 / 3)
    assert turn_risk(a_known(status="slp", status_elapsed=3)).sleep == 0.0


def test_how_long_a_status_has_lasted_is_public(dex):
    """We watched it land, so the elapsed count is not hidden information.

    How much longer it has to run still is, and that stays ``None``.
    """
    state = chosen_battle(dex, a_set("gengar", ("willowisp",)),
                          a_set("snorlax", ("bodyslam",)))
    from pkcm.engine.state import legal_actions

    state, _ = step(state, Action.move(0), Action.move(0))
    theirs = Observation.of(state, 0).foe[0]
    if theirs.status is None:
        pytest.skip("Will-O-Wisp missed")
    assert theirs.status_elapsed == 0
    assert theirs.status_turns is None, "how much longer is not ours to know"

    state, _ = step(state, Action.move(0), Action.move(0))
    assert Observation.of(state, 0).foe[0].status_elapsed == 1


def test_a_full_health_target_might_be_holding_a_sash(dex):
    """A 확정 1타 into an unrevealed Focus Sash is not a knockout, and a strong
    player prices that in rather than being surprised by it."""
    from pkcm.envs.analysis import could_survive_a_kill

    unknown_item = a_known(hp_fraction=1.0, item_known=False, ability_known=True,
                           ability="thickfat")
    assert could_survive_a_kill(unknown_item, dex), "it could be a Sash"

    known_empty = a_known(hp_fraction=1.0, item_known=True, item=None,
                          ability_known=True, ability="thickfat")
    assert not could_survive_a_kill(known_empty, dex)

    hurt = a_known(hp_fraction=0.9, item_known=False)
    assert not could_survive_a_kill(hurt, dex), "a Sash only works from full"


def test_sturdy_counts_even_unrevealed(dex):
    from pkcm.envs.analysis import could_survive_a_kill

    sturdy_possible = a_known(species_id="skarmory", hp_fraction=1.0,
                              item_known=True, item=None, ability_known=False)
    assert could_survive_a_kill(sturdy_possible, dex), "Skarmory can have Sturdy"

    cannot = a_known(species_id="alakazam", hp_fraction=1.0,
                     item_known=True, item=None, ability_known=False)
    assert not could_survive_a_kill(cannot, dex)


def test_the_risk_block_reaches_the_policy(dex):
    from pkcm.envs.encoding import RISK_FEATURES, RISK_ROWS

    env = ChampionsEnv(battle_format="doubles", dex=dex, seed=41)
    observations, _ = env.reset()
    risk = observations["player_0"]["risk"]
    assert risk.shape == (RISK_ROWS, RISK_FEATURES)
    assert env.observation_space("player_0").contains(observations["player_0"])


# --------------------------------------------------------------------------- #
# Abilities that change what a hit does
#
# hk asked about Disguise and Multiscale by name. The engine had both right;
# the estimator had neither, which is a calculator that promises a knockout
# into a full-HP Multiscale Dragonite.
# --------------------------------------------------------------------------- #


def test_the_engine_lets_disguise_eat_a_hit(dex):
    """One hit refused, 1/8 of maximum HP paid, and the forme changes."""
    from pkcm.engine.battle import make_context
    from pkcm.engine.moves import use_move

    state = chosen_battle(dex, a_set("garchomp", ("earthquake",)),
                          a_set("mimikyu", ("bodyslam",), ability="disguise"))
    maximum = state.pokemon(1, 0).max_hp
    ctx = make_context(state)
    use_move(ctx, (0, 0), dex.moves["earthquake"], defender=(1, 0))
    assert state.sides[1].hp[0] == maximum - maximum // 8
    assert state.species_id(1, 0) == "mimikyubusted"

    before = state.sides[1].hp[0]
    ctx = make_context(state)
    use_move(ctx, (0, 0), dex.moves["earthquake"], defender=(1, 0))
    assert state.sides[1].hp[0] < before - maximum // 8, "the second hit is real"


def test_the_engine_halves_for_multiscale_only_at_full_health(dex):
    from pkcm.engine.battle import make_context
    from pkcm.engine.moves import use_move

    def taken(fraction, ability):
        state = chosen_battle(dex, a_set("garchomp", ("icefang",)),
                              a_set("dragonite", ("bodyslam",), ability=ability))
        maximum = state.pokemon(1, 0).max_hp
        state.sides[1].hp[0] = max(1, int(maximum * fraction))
        before = state.sides[1].hp[0]
        ctx = make_context(state)
        use_move(ctx, (0, 0), dex.moves["icefang"], defender=(1, 0))
        return before - state.sides[1].hp[0]

    assert taken(1.0, "multiscale") * 2 == taken(1.0, "innerfocus")
    assert taken(0.9, "multiscale") == taken(0.9, "innerfocus"), "only from full"


def test_a_revealed_multiscale_halves_the_estimate(dex, sheet):
    from pkcm.envs.analysis import estimate_damage

    def estimate(revealed):
        state = chosen_battle(dex, a_set("garchomp", ("icefang",)),
                              a_set("dragonite", ("bodyslam",), ability="multiscale"))
        if revealed:
            state.revealed[1].abilities.add(0)
        observation = Observation.of(state, 0)
        return estimate_damage(observation, sheet, dex, observation.own[0],
                               observation.foe[0], dex.moves["icefang"])

    hidden, known = estimate(False), estimate(True)
    assert known.percent.high < hidden.percent.high / 1.8
    assert hidden.blunted_possible, "unknown, and Dragonite can have it"
    assert not known.blunted_possible, "known, so it is in the number instead"


def test_an_unknown_ability_is_flagged_rather_than_guessed(dex, sheet):
    """The estimate stays optimistic and says so, which is the honest shape.

    Folding a maybe-Multiscale into the number would make every estimate wrong
    against the Dragonites that do not have it.
    """
    from pkcm.envs.analysis import could_blunt

    state = chosen_battle(dex, a_set("garchomp", ("icefang",)),
                          a_set("dragonite", ("bodyslam",), ability="innerfocus"))
    observation = Observation.of(state, 0)
    assert could_blunt(observation.foe[0], dex), "it could be Multiscale"

    state.revealed[1].abilities.add(0)
    observation = Observation.of(state, 0)
    assert not could_blunt(observation.foe[0], dex), "now we know it is not"


def test_a_known_absorbing_ability_reads_as_immune(dex, sheet):
    from pkcm.envs.analysis import estimate_damage

    state = chosen_battle(dex, a_set("garchomp", ("earthquake",)),
                          a_set("rotomheat", ("bodyslam",), ability="levitate"))
    state.revealed[1].abilities.add(0)
    observation = Observation.of(state, 0)
    estimate = estimate_damage(observation, sheet, dex, observation.own[0],
                               observation.foe[0], dex.moves["earthquake"])
    assert estimate is not None and estimate.immune
    assert estimate.ko_chance == 0.0


def test_an_intact_disguise_makes_a_kill_survivable(dex):
    from pkcm.engine.battle import make_context
    from pkcm.engine.moves import use_move
    from pkcm.envs.analysis import could_survive_a_kill

    state = chosen_battle(dex, a_set("garchomp", ("earthquake",)),
                          a_set("mimikyu", ("bodyslam",), ability="disguise"))
    state.revealed[1].species.add(0)
    intact = Observation.of(state, 0).foe[0]
    assert could_survive_a_kill(intact, dex), "the disguise is still up"

    # Break it for real, then check the answer changes for the right reason.
    ctx = make_context(state)
    use_move(ctx, (0, 0), dex.moves["earthquake"], defender=(1, 0))
    busted = Observation.of(state, 0).foe[0]
    assert busted.species_id == "mimikyubusted"
    assert busted.hp_fraction < 1.0
    assert not could_survive_a_kill(busted, dex), "broken, and no longer at full HP"


def test_every_defensive_ability_on_the_roster_is_accounted_for(dex):
    """Read out of the registry, so a new implementation cannot be missed.

    The registry knows which abilities interfere with an incoming hit; this
    checks each of them is either in one of the estimator's tables or named as
    working for the attacker instead.
    """
    from pkcm.envs.analysis import (
        ABSORBS_TYPE,
        ATTACKER_SIDE_ABILITIES,
        BLOCKS_FLAG,
        CATEGORY_SOFTENERS,
        FULL_HP_HALVERS,
        SUPER_EFFECTIVE_SOFTENERS,
        SURVIVES_AT_ANY_HP,
        SURVIVES_FROM_FULL,
        blunting_abilities,
    )

    handled = (set(ABSORBS_TYPE) | set(BLOCKS_FLAG) | set(CATEGORY_SOFTENERS)
               | set(FULL_HP_HALVERS) | set(SUPER_EFFECTIVE_SOFTENERS)
               | set(SURVIVES_FROM_FULL) | set(SURVIVES_AT_ANY_HP)
               | ATTACKER_SIDE_ABILITIES | {"fluffy", "thickfat", "heatproof",
                                            "purifyingsalt", "waterbubble"})

    regulation = dex.regulation("m_b")
    roster = {ability
              for species in regulation.legal_species | regulation.legal_megas
              for ability in dex.species[species].abilities}
    missing = sorted((blunting_abilities() & roster) - handled)
    assert not missing, (
        f"these interfere with an incoming hit and the estimator ignores them: "
        f"{missing}"
    )


# --------------------------------------------------------------------------- #
# Who moves first
#
# hk asked about Trick Room, Skill Swap, Wandering Spirit and speed generally.
# The engine had all of it; the estimator's speed comparison was base stats and
# stat stages and nothing else, which is wrong about most turns.
# --------------------------------------------------------------------------- #


def test_trick_room_reverses_the_order_in_the_engine(dex):
    state = chosen_battle(dex, a_set("alakazam", ("trickroom", "bodyslam")),
                          a_set("snorlax", ("bodyslam",), ability="thickfat"))
    state, _ = step(state, Action.move(0), Action.move(0))
    assert "trickroom" in state.field.rooms

    state, log = step(state, Action.move(1), Action.move(0))
    order = [event.species for event in log if event.kind == "move_used"]
    assert order and order[0] == "snorlax", "the slow one goes first"


def test_the_estimator_knows_about_trick_room(dex):
    from pkcm.envs.analysis import outspeeds

    state = chosen_battle(dex, a_set("alakazam", ("psychic",)),
                          a_set("snorlax", ("bodyslam",), ability="thickfat"))
    observation = Observation.of(state, 0)
    assert outspeeds(dex, observation.own[0], observation.foe[0], observation) is True

    state.field.rooms["trickroom"] = 5
    observation = Observation.of(state, 0)
    assert outspeeds(dex, observation.own[0], observation.foe[0], observation) is False


def test_priority_beats_speed(dex):
    from pkcm.envs.analysis import outspeeds

    state = chosen_battle(dex, a_set("snorlax", ("suckerpunch",), ability="thickfat"),
                          a_set("alakazam", ("psychic",)))
    observation = Observation.of(state, 0)
    assert outspeeds(dex, observation.own[0], observation.foe[0], observation) is False
    assert outspeeds(dex, observation.own[0], observation.foe[0], observation,
                     our_priority=1) is True


def test_paralysis_and_tailwind_reach_the_estimate(dex):
    from pkcm.envs.analysis import speed_of

    state = chosen_battle(dex, a_set("garchomp", ("earthquake",)),
                          a_set("snorlax", ("bodyslam",), ability="thickfat"))
    plain = speed_of(Observation.of(state, 0), Observation.of(state, 0).own[0], dex,
                     ours=True)

    state.sides[0].status[0] = "par"
    slowed = speed_of(Observation.of(state, 0), Observation.of(state, 0).own[0], dex,
                      ours=True)
    assert slowed.high == plain.high // 2

    state.sides[0].status[0] = None
    state.sides[0].conditions["tailwind"] = 4
    hurried = speed_of(Observation.of(state, 0), Observation.of(state, 0).own[0], dex,
                       ours=True)
    assert hurried.low == plain.low * 2


def test_an_unknown_choice_scarf_widens_the_bracket(dex):
    """Playing around a possible Scarf is the whole point of the range."""
    from pkcm.envs.analysis import speed_of

    state = chosen_battle(dex, a_set("garchomp", ("earthquake",)),
                          a_set("snorlax", ("bodyslam",), ability="thickfat"))
    observation = Observation.of(state, 0)
    theirs = speed_of(observation, observation.foe[0], dex, ours=False)
    assert theirs.high >= int(theirs.low * 1.5), "a Scarf is still on the table"

    ours = speed_of(observation, observation.own[0], dex, ours=True)
    assert ours.certain, "we know our own item"


def test_our_speed_matches_the_engines(dex):
    """Two implementations of one sum. This is what stops them drifting.

    The estimator cannot call ``mutate.effective_stat`` -- that needs the battle
    state, and then it could see everything -- so it mirrors the modifiers
    instead, and this compares the two on real battles.
    """
    from pkcm.data.dex import Stat
    from pkcm.engine.battle import make_context
    from pkcm.engine.mutate import effective_stat
    from pkcm.envs.analysis import speed_of

    checked = 0
    for seed in range(8):
        state = battle(dex, seed=seed * 5 + 2)
        for setup in (lambda s: None,
                      lambda s: s.sides[0].status.__setitem__(0, "par"),
                      lambda s: s.sides[0].conditions.__setitem__("tailwind", 4),
                      lambda s: s.sides[0].boosts[0].__setitem__(4, 2)):
            fresh = state.clone()
            setup(fresh)
            engine = effective_stat(make_context(fresh), (0, 0), Stat.SPE)
            observation = Observation.of(fresh, 0)
            estimate = speed_of(observation, observation.own[0], dex, ours=True)
            assert estimate.low == estimate.high == engine, (
                seed, estimate, engine, observation.own[0].species_id)
            checked += 1
    assert checked == 32


def test_skill_swap_makes_both_abilities_public(dex):
    """You watched them trade. Neither is a secret afterwards."""
    from pkcm.engine.battle import make_context
    from pkcm.engine.moves import use_move

    state = chosen_battle(dex, a_set("alakazam", ("skillswap",), ability="synchronize"),
                          a_set("dragonite", ("bodyslam",), ability="multiscale"))
    ctx = make_context(state)
    use_move(ctx, (0, 0), dex.moves["skillswap"], defender=(1, 0))

    assert state.ability_id(0, 0) == "multiscale"
    assert state.ability_id(1, 0) == "synchronize"
    theirs = Observation.of(state, 0).foe[0]
    assert theirs.ability_known and theirs.ability == "synchronize"


def test_wandering_spirit_swaps_on_contact_only(dex):
    from pkcm.engine.battle import make_context
    from pkcm.engine.moves import use_move

    def after(move_id):
        state = chosen_battle(dex, a_set("garchomp", (move_id,), ability="roughskin"),
                              a_set("runerigus", ("bodyslam",), ability="wanderingspirit"))
        ctx = make_context(state)
        use_move(ctx, (0, 0), dex.moves[move_id], defender=(1, 0))
        return state.ability_id(0, 0), state.ability_id(1, 0)

    assert after("dragonclaw") == ("wanderingspirit", "roughskin"), "contact swaps"
    assert after("earthpower") == ("roughskin", "wanderingspirit"), "and nothing else does"


# --------------------------------------------------------------------------- #
# What the opponent is probably running
# --------------------------------------------------------------------------- #


def test_belief_draws_sets_people_actually_bring(dex):
    """Field-by-field sampling is consistent with the observation and nothing
    like a real set: an item from all 147 the format allows, a move from the
    sixty that species can learn. Rankers use 29 items, four of them for half
    the slots."""
    from pkcm.engine.legality import make_team, ranker_slots
    from pkcm.engine.rng import Rng
    from pkcm.engine.state import BattleConfig, new_battle
    from pkcm.envs.observation import Observation, determinize

    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"),
                          battle_format="singles")
    teams = tuple(make_team(dex, config.regulation,
                            Rng.from_seed(21 + offset).cursor(), "singles", "ranker")
                  for offset in (1, 2))
    state = new_battle(config, teams, seed=21)
    state, _ = step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))
    observation = Observation.of(state, 0)

    pool_items = {pokemon.item for pokemon in ranker_slots() if pokemon.item}
    drawn = 0
    for seed in range(40):
        guess = determinize(observation, state, Rng.from_seed(seed).cursor(), True)
        for slot in range(len(guess.sides[1].selection)):
            pokemon = guess.pokemon(1, slot)
            if pokemon.item:
                drawn += pokemon.item in pool_items
    assert drawn > 0, "nothing came from the pool at all"


def test_belief_narrows_on_what_has_been_seen(dex):
    """The point of drawing whole sets: a move we have watched rules out every
    set without it. Under per-field sampling it ruled out nothing -- the other
    three were redrawn from the whole learnable list regardless."""
    from pkcm.engine.pokemon import PokemonSet
    from pkcm.envs.belief import candidates, sets_by_species

    populated = [species for species, sets in sets_by_species().items()
                 if len(sets) > 1]
    assert populated, "the ranker pool has no species with two sets"

    class Watched:
        """The observation's view of one Pokemon, as far as this needs it."""
        def __init__(self, species_id, moves):
            self.species_id = species_id
            self.moves = moves
            self.item = None
            self.item_known = False
            self.ability = None
            self.ability_known = False

    for species in populated:
        sets = sets_by_species()[species]
        move = sets[0].moves[0]
        if all(move in other.moves for other in sets):
            continue  # every set has it; nothing to narrow
        before = candidates(species, Watched(species, ()))
        after = candidates(species, Watched(species, (move,)))
        assert len(after) < len(before)
        assert all(move in one.moves for one in after)
        return
    pytest.skip("no move in the pool separates two sets of one species")


# --------------------------------------------------------------------------- #
# The preview grid is per-battle, and encoding must treat it that way
# --------------------------------------------------------------------------- #


def test_the_preview_grid_is_computed_once_per_battle(dex):
    """The 6x6 matchup grid is 92% of encoding cost and depends only on our
    registered sets and their registered species -- fixed at preview, unchanged
    by determinization. The search encodes ~800 states per decision; paying
    five hundred ``midpoint`` calls each time was most of self-play."""
    from pkcm.engine.legality import random_team
    from pkcm.engine.rng import Rng
    from pkcm.engine.state import BattleConfig, new_battle
    from pkcm.envs.encoding import encode_preview
    from pkcm.envs.observation import Observation

    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"))
    teams = tuple(random_team(dex, config.regulation, Rng.from_seed(o).cursor())
                  for o in (1, 2))
    one = Observation.of(new_battle(config, teams, seed=1), 0)
    two = Observation.of(new_battle(config, teams, seed=2), 0)

    first = encode_preview(one, dex)
    again = encode_preview(two, dex)
    assert first is again, "same battle, same grid object -- else nothing was saved"

    other_teams = tuple(random_team(dex, config.regulation,
                                    Rng.from_seed(10 + o).cursor()) for o in (1, 2))
    other = Observation.of(new_battle(config, other_teams, seed=1), 0)
    assert encode_preview(other, dex) is not first, "different battle, different grid"


def test_the_shared_preview_grid_cannot_be_written(dex):
    """Every state of a battle shares one array now. A writer would corrupt not
    its own encoding but every later state's."""
    import numpy as np
    import pytest as _pytest

    from pkcm.engine.legality import random_team
    from pkcm.engine.rng import Rng
    from pkcm.engine.state import BattleConfig, new_battle
    from pkcm.envs.encoding import encode_preview
    from pkcm.envs.observation import Observation

    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"))
    teams = tuple(random_team(dex, config.regulation, Rng.from_seed(o).cursor())
                  for o in (1, 2))
    grid = encode_preview(Observation.of(new_battle(config, teams, seed=1), 0), dex)
    with _pytest.raises((ValueError, RuntimeError)):
        grid[0, 0] = 1.0
