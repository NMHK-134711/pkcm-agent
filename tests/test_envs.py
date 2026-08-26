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
