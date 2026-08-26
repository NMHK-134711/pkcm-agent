"""Self-play data, the network, and the loop that joins them.

Nothing here asserts the agent gets stronger -- that is what
``scripts/arena.py`` is for, and it takes minutes rather than seconds. These
check the pipeline carries what it claims to: that a sample is what the two
heads need, that the network consumes it, that training moves in the right
direction, and above all that none of it sees anything the player cannot.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pkcm.data.dex import load_dex  # noqa: E402
from pkcm.envs.encoding import (  # noqa: E402
    SCALAR_SIZE,
    SWITCH_BASE,
    Vocabulary,
    action_space_size,
    decode_action,
)
from pkcm.envs.reference import sheet_for  # noqa: E402
from pkcm.search import SearchConfig  # noqa: E402
from pkcm.train.net import NetConfig, build, collate  # noqa: E402
from pkcm.train.samples import SelfPlayConfig, play_one  # noqa: E402
from pkcm.train.trainer import TrainConfig, fit, save  # noqa: E402


@pytest.fixture(scope="module")
def dex():
    return load_dex()


@pytest.fixture(scope="module")
def pieces(dex):
    vocabulary = Vocabulary.of(dex)
    return vocabulary, sheet_for(dex, vocabulary), action_space_size(6, 3)


@pytest.fixture(scope="module")
def samples(dex):
    """One real self-play battle. Slow, so it is shared by everything here."""
    config = SelfPlayConfig(search=SearchConfig(iterations=40, determinizations=4))
    return play_one(dex, config, seed=3)


# --------------------------------------------------------------------------- #
# What a sample is
# --------------------------------------------------------------------------- #


def test_a_battle_produces_samples_for_both_sides(samples):
    assert samples, "the battle produced nothing"
    assert {sample.player for sample in samples} == {0, 1}, (
        "both sides see different things and reach different conclusions, so a "
        "battle is two games' worth of data")


def test_the_policy_target_is_a_distribution(samples):
    for sample in samples:
        total = sample.policy.sum()
        assert total == pytest.approx(1.0, abs=1e-5), total
        assert (sample.policy >= 0).all()


def test_the_policy_target_only_covers_legal_actions(samples, dex):
    """The mask already says what is illegal; the head should not relearn it."""
    for sample in samples:
        for index in np.flatnonzero(sample.policy):
            action = decode_action(int(index), 6, 3)
            assert action is not None


def test_the_value_target_is_the_outcome(samples):
    values = {sample.value for sample in samples}
    assert values <= {-1.0, 0.0, 1.0}, values
    # Both sides cannot have won.
    by_player = {player: {s.value for s in samples if s.player == player}
                 for player in (0, 1)}
    if by_player[0] == {1.0}:
        assert by_player[1] == {-1.0}


def test_samples_carry_no_hidden_information(samples, dex):
    """The one that matters. A network trained on the truth learns to read the
    opponent's team, and is then worthless against a real one."""
    from pkcm.envs.encoding import MAX_BROUGHT

    early = min(samples, key=lambda sample: sample.turn)
    observation = early.observation
    # The foe half of every per-Pokemon array. At least one bench slot must
    # still be blank on the first recorded turn.
    foe_species = observation["species"][MAX_BROUGHT:]
    assert (foe_species == 0).any(), "the whole opposing team was visible"
    assert not observation["pp"][MAX_BROUGHT * 4:].any(), "their PP is never shown"


# --------------------------------------------------------------------------- #
# The network
# --------------------------------------------------------------------------- #


def test_the_network_consumes_what_self_play_produces(pieces, samples):
    vocabulary, sheet, action_space = pieces
    net = build(vocabulary, sheet, action_space, SCALAR_SIZE,
                NetConfig(hidden=64, blocks=1))
    batch = collate([sample.observation for sample in samples[:8]], "cpu")
    logits, value = net(batch)
    assert logits.shape == (min(8, len(samples)), action_space)
    assert value.shape == (min(8, len(samples)),)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()


def test_the_value_head_is_bounded(pieces, samples):
    """It predicts an outcome in [-1, 1], so it should not be able to say 4."""
    vocabulary, sheet, action_space = pieces
    net = build(vocabulary, sheet, action_space, SCALAR_SIZE,
                NetConfig(hidden=64, blocks=1))
    _, value = net(collate([sample.observation for sample in samples[:8]], "cpu"))
    assert (value.abs() <= 1.0).all()


def test_the_dex_tables_are_frozen(pieces):
    """They are facts, fed in rather than learned. A gradient on them would be
    the network editing the dex to suit itself."""
    vocabulary, sheet, action_space = pieces
    net = build(vocabulary, sheet, action_space, SCALAR_SIZE,
                NetConfig(hidden=64, blocks=1))
    trainable = {name for name, parameter in net.named_parameters()
                 if parameter.requires_grad}
    assert not any("facts" in name for name in trainable), trainable
    assert net.species_facts.requires_grad is False


def test_unknown_ids_embed_to_nothing(pieces):
    """Index 0 means "unknown or none", so it has to be the padding index --
    otherwise a hidden opponent item is a *specific* learned vector."""
    vocabulary, sheet, action_space = pieces
    net = build(vocabulary, sheet, action_space, SCALAR_SIZE,
                NetConfig(hidden=64, blocks=1))
    for table in (net.species, net.moves, net.items, net.abilities, net.status):
        assert table.padding_idx == 0
        assert float(table.weight[0].abs().sum()) == 0.0
    assert float(net.species_facts[0].abs().sum()) == 0.0


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def test_training_reduces_the_loss_on_what_it_saw(pieces, samples):
    """Not evidence of strength -- evidence the gradient reaches the heads."""
    vocabulary, sheet, action_space = pieces
    net = build(vocabulary, sheet, action_space, SCALAR_SIZE,
                NetConfig(hidden=64, blocks=1))
    device = torch.device("cpu")
    settings = TrainConfig(epochs=1, batch_size=32)

    first = fit(net, samples, device, settings)
    for _ in range(6):
        last = fit(net, samples, device, settings)
    assert last["policy_loss"] < first["policy_loss"]
    assert last["value_mae"] < first["value_mae"]


def test_a_checkpoint_round_trips(pieces, samples, tmp_path):
    vocabulary, sheet, action_space = pieces
    config = NetConfig(hidden=64, blocks=1)
    net = build(vocabulary, sheet, action_space, SCALAR_SIZE, config)
    fit(net, samples, torch.device("cpu"), TrainConfig(epochs=1, batch_size=32))

    path = tmp_path / "net.pt"
    save(net, path, {"iteration": 0})
    before, _ = net.evaluate([samples[0].observation], "cpu")

    from pkcm.train.trainer import load_into

    reloaded = build(vocabulary, sheet, action_space, SCALAR_SIZE, config)
    payload = load_into(reloaded, path, torch.device("cpu"))
    after, _ = reloaded.evaluate([samples[0].observation], "cpu")
    assert payload["iteration"] == 0
    assert np.allclose(before, after)


# --------------------------------------------------------------------------- #
# Putting it back into the search
# --------------------------------------------------------------------------- #


def test_the_evaluator_gives_the_search_what_it_asks_for(dex, pieces, samples):
    from pkcm.engine.actions import Action
    from pkcm.engine.battle import step
    from pkcm.engine.legality import random_team
    from pkcm.engine.rng import Rng
    from pkcm.engine.state import BattleConfig, new_battle
    from pkcm.search.policy import joint_actions
    from pkcm.train.evaluator import Evaluator

    vocabulary, sheet, action_space = pieces
    net = build(vocabulary, sheet, action_space, SCALAR_SIZE,
                NetConfig(hidden=64, blocks=1))
    evaluator = Evaluator(net=net, dex=dex, device="cpu",
                          _vocabulary=vocabulary, _sheet=sheet)

    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"),
                          battle_format="singles")
    teams = tuple(random_team(dex, config.regulation,
                              Rng.from_seed(offset).cursor()) for offset in (1, 2))
    state = new_battle(config, teams, seed=1)
    state, _ = step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))

    options = joint_actions(state, 0)
    prior = evaluator.prior(state, 0, options)
    assert len(prior) == len(options)
    assert sum(prior) == pytest.approx(1.0)
    assert all(p >= 0 for p in prior)
    assert -1.0 <= evaluator.value(state, 0) <= 1.0


def test_the_search_accepts_an_evaluator_and_still_plays_legally(dex, pieces):
    from pkcm.engine.actions import Action
    from pkcm.engine.battle import step
    from pkcm.engine.legality import random_team
    from pkcm.engine.rng import Rng
    from pkcm.engine.state import BattleConfig, new_battle
    from pkcm.search import MCTS
    from pkcm.search.policy import joint_actions
    from pkcm.train.evaluator import Evaluator

    vocabulary, sheet, action_space = pieces
    net = build(vocabulary, sheet, action_space, SCALAR_SIZE,
                NetConfig(hidden=64, blocks=1))
    evaluator = Evaluator(net=net, dex=dex, device="cpu",
                          _vocabulary=vocabulary, _sheet=sheet)

    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"),
                          battle_format="singles")
    teams = tuple(random_team(dex, config.regulation,
                              Rng.from_seed(offset).cursor()) for offset in (1, 2))
    state = new_battle(config, teams, seed=2)
    state, _ = step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))

    result = MCTS(SearchConfig(iterations=30, determinizations=3),
                  evaluator=evaluator).choose(state, 0, Rng.from_seed(4).cursor())
    assert result.action in joint_actions(state, 0)


def test_trust_zero_is_the_handcrafted_prior(dex, pieces):
    """The first iteration has no trained network, so it must be able to say so."""
    from pkcm.engine.actions import Action
    from pkcm.engine.battle import step
    from pkcm.engine.legality import random_team
    from pkcm.engine.rng import Rng
    from pkcm.engine.state import BattleConfig, new_battle
    from pkcm.search.policy import joint_actions, prior_over
    from pkcm.train.evaluator import Evaluator

    vocabulary, sheet, action_space = pieces
    net = build(vocabulary, sheet, action_space, SCALAR_SIZE,
                NetConfig(hidden=64, blocks=1))
    evaluator = Evaluator(net=net, dex=dex, device="cpu", trust=0.0,
                          _vocabulary=vocabulary, _sheet=sheet)

    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"),
                          battle_format="singles")
    teams = tuple(random_team(dex, config.regulation,
                              Rng.from_seed(offset).cursor()) for offset in (1, 2))
    state = new_battle(config, teams, seed=3)
    state, _ = step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))

    options = joint_actions(state, 0)
    assert evaluator.prior(state, 0, options) == pytest.approx(
        prior_over(state, 0, options))
