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


# --------------------------------------------------------------------------- #
# Reporting a win rate honestly
# --------------------------------------------------------------------------- #


def test_the_interval_does_not_collapse_at_the_extremes():
    """Wald says six losses out of six is "0.0% +/- 0.0%", which reads as
    certainty and means the opposite."""
    from pkcm.train.interval import wilson

    _, low, high = wilson(0, 6)
    assert low == 0.0 and 0.3 < high < 0.5, (low, high)

    _, low, high = wilson(6, 6)
    assert high == 1.0 and 0.5 < low < 0.7


def test_the_interval_narrows_with_evidence():
    from pkcm.train.interval import wilson

    _, small_low, small_high = wilson(30, 50)
    _, large_low, large_high = wilson(300, 500)
    assert (large_high - large_low) < (small_high - small_low) / 2


def test_separability_matches_the_recorded_measurements():
    """The numbers this project actually produced, and what they meant."""
    from pkcm.train.interval import separable

    assert not separable(26, 49), "53.1% over 49 battles said nothing"
    assert not separable(69, 119), "58.0% said nothing either"
    assert separable(79, 119), "66.4% did"
    assert not separable(55, 120), "45.8% is a coin flip, not a loss"


def test_an_empty_sample_is_ignorance_not_a_tie():
    from pkcm.train.interval import wilson

    rate, low, high = wilson(0, 0)
    assert (rate, low, high) == (0.5, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Telling learning apart from memorising
# --------------------------------------------------------------------------- #


def test_the_split_keeps_whole_battles_together():
    """Splitting samples at random puts turn 11 in training and turn 12 in
    validation, and those are nearly the same position with the same label."""
    from pkcm.train.samples import Sample
    from pkcm.train.trainer import split_by_battle

    made = [Sample(observation={}, policy=np.zeros(4), value=1.0,
                   player=index % 2, turn=index, battle=index // 5)
            for index in range(50)]
    training, validation = split_by_battle(made, 0.2, seed=1)
    assert training and validation
    assert not ({made[i].battle for i in training}
                & {made[i].battle for i in validation})
    assert len(training) + len(validation) == len(made)


def test_a_tiny_sample_is_not_split():
    """Two battles cannot be divided into a train and a validation half that
    both mean anything, so it says so rather than pretending."""
    from pkcm.train.samples import Sample
    from pkcm.train.trainer import split_by_battle

    made = [Sample(observation={}, policy=np.zeros(4), value=1.0,
                   player=0, turn=index, battle=index // 5)
            for index in range(10)]
    training, validation = split_by_battle(made, 0.2)
    assert validation == [] and len(training) == len(made)


def test_fit_reports_a_held_out_score(pieces, samples):
    """The first run here scored 0.032 in training and 0.771 on battles it had
    not seen. A constant zero scores 1.0. The curve looked excellent throughout.
    """
    from dataclasses import replace

    vocabulary, sheet, action_space = pieces
    net = build(vocabulary, sheet, action_space, SCALAR_SIZE,
                NetConfig(hidden=64, blocks=1))
    # One battle's worth of samples all share a battle id, so give them several.
    spread = [replace(sample, battle=index % 6)
              for index, sample in enumerate(samples)]
    result = fit(net, spread, torch.device("cpu"),
                 TrainConfig(epochs=1, batch_size=16, validation_fraction=0.25))
    assert "val_value_mae" in result and "val_policy_loss" in result
    assert result["val_battles"] >= 1


def test_the_value_head_is_scored_against_a_baseline(pieces, samples):
    """Predicting nothing at all scores 1.0 on a target of plus or minus one."""
    values = np.array([sample.value for sample in samples])
    assert np.abs(0.0 - values).mean() == pytest.approx(1.0, abs=0.2), (
        "if this is not about 1.0 the baseline claim in the docs is wrong")


def test_trust_governs_the_leaf_value_and_not_only_the_prior(pieces, samples):
    """``trust`` promised to hold back a network the loop does not believe yet,
    and only did it for the prior. The leaf value is the half that matters:
    a saturated value head does not fail to rank the lines, it ranks them
    backwards with conviction."""
    from pkcm.data.dex import load_dex
    from pkcm.engine.legality import random_team
    from pkcm.engine.rng import Rng
    from pkcm.engine.state import BattleConfig, new_battle
    from pkcm.search.evaluate import heuristic
    from pkcm.train.evaluator import Evaluator

    vocabulary, sheet, action_space = pieces
    net = build(vocabulary, sheet, action_space, SCALAR_SIZE,
                NetConfig(hidden=64, blocks=1))
    dex = load_dex()
    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"))
    teams = tuple(random_team(dex, config.regulation, Rng.from_seed(o).cursor())
                  for o in (1, 2))
    state = new_battle(config, teams, seed=3)

    ignored = Evaluator(net=net, dex=dex, trust=0.0)
    believed = Evaluator(net=net, dex=dex, trust=1.0)
    assert ignored.value(state, 0) == pytest.approx(heuristic(state, 0))
    if believed.value(state, 0) != pytest.approx(heuristic(state, 0)):
        half = Evaluator(net=net, dex=dex, trust=0.5)
        assert (min(ignored.value(state, 0), believed.value(state, 0))
                <= half.value(state, 0)
                <= max(ignored.value(state, 0), believed.value(state, 0)))


def test_blending_the_value_target_still_scores_against_the_truth(pieces, samples):
    """A target the network helped write cannot also be the exam.

    ``search_value_weight`` mixes the search's root value into what the value
    head is fitted to. If validation scored that same blend, a network that
    collapsed into predicting its own output would post an excellent number
    while having learned nothing about who wins.
    """
    from dataclasses import replace

    vocabulary, sheet, action_space = pieces
    spread = [replace(sample, battle=index % 6, search_value=0.0)
              for index, sample in enumerate(samples)]

    scores = {}
    for weight in (0.0, 1.0):
        net = build(vocabulary, sheet, action_space, SCALAR_SIZE,
                    NetConfig(hidden=64, blocks=1))
        result = fit(net, spread, torch.device("cpu"),
                     TrainConfig(epochs=1, batch_size=16, validation_fraction=0.25,
                                 search_value_weight=weight))
        scores[weight] = result

    # Fitted to a constant zero, the training error collapses; the held-out
    # score against the real +-1 outcome cannot follow it down.
    assert scores[1.0]["value_mae"] < scores[0.0]["value_mae"]
    assert scores[1.0]["val_value_mae"] > 0.5, (
        "validation is scoring the blended target, not the outcome")


# --------------------------------------------------------------------------- #
# The loop's own arena
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def checkpoint(pieces, samples, tmp_path_factory):
    """A saved network for the arena to drive. Untrained is fine -- these ask
    whether the measurement is reproducible, not whether it wins."""
    vocabulary, sheet, action_space = pieces
    net = build(vocabulary, sheet, action_space, SCALAR_SIZE,
                NetConfig(hidden=32, blocks=1))
    path = tmp_path_factory.mktemp("arena") / "net.pt"
    save(net, path, {"iteration": 0})
    return path


def test_a_match_plays_the_same_game_every_time(dex, checkpoint):
    """Every seed in ``play_match`` comes from the match number, so a pool may
    hand matches out in any order and to any worker."""
    from pkcm.train.matchup import MatchConfig, play_match

    config = MatchConfig(checkpoint=str(checkpoint),
                         search=SearchConfig(iterations=12, determinizations=2))
    first = play_match(dex, config, 0)
    assert play_match(dex, config, 0) == first
    assert first.wins + first.losses + first.draws == 2, (
        "a match is both seatings of one pair of teams")


def test_the_pool_returns_what_one_process_would(checkpoint):
    """The parallel arena is only worth having if it is the same measurement.

    Self-play may be sloppy about this -- more data is more data -- but the
    arena is the one number in the loop that knows anything, and a win rate
    that depends on how many cores were free is not one.
    """
    from pkcm.train.matchup import MatchConfig, stream

    config = MatchConfig(checkpoint=str(checkpoint),
                         search=SearchConfig(iterations=12, determinizations=2))
    pooled = list(stream(config, 2, workers=2))
    serial = list(stream(config, 2, workers=1))
    if len(pooled) != 2 or len(serial) != 2:
        # A worker died and its match was dropped after every retry. That is the
        # tolerance working, not a mismatch -- and comparing a short run against
        # a full one would fail for the wrong reason.
        pytest.skip("a worker crashed hard enough to drop a match")
    assert sorted(pooled, key=repr) == sorted(serial, key=repr)


def test_the_root_value_is_recorded_per_sample():
    """It has to come from the search, not be back-filled from the outcome --
    the whole point is that it does not know who won."""
    from pkcm.train.samples import Sample

    made = Sample(observation={}, policy=np.zeros(4), value=1.0, player=0,
                  turn=3, battle=7, search_value=-0.25)
    assert made.search_value == -0.25
    assert made.value == 1.0


def test_a_worker_that_dies_does_not_hang_the_pool(tmp_path):
    """``Pool.imap_unordered`` waits forever for a task whose worker segfaulted.

    That is not hypothetical: two arena workers died with 0xc0000005 ten
    minutes into an evaluation on 2026-08-27 and the run sat idle for
    seventy-eight minutes, no error and no CPU, until someone looked at it.
    """
    import _crashy

    from pkcm.train.parallel import map_unordered

    marker = str(tmp_path / "died-once")
    got = list(map_unordered(_crashy.crash_first_time, range(6),
                             initializer=_crashy.remember, initargs=(marker,),
                             workers=3, attempts=3, what="item"))
    assert sorted(got) == [0, 2, 4, 6, 8, 10], (
        "the task the dead worker was holding has to be re-run, not waited on")


# --------------------------------------------------------------------------- #
# Imitating the handcrafted prior
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def imitated(dex):
    """One cheap battle's worth of imitation targets. No search runs."""
    from pkcm.train.imitate import ImitateConfig, play_one as imitate_one

    return imitate_one(dex, ImitateConfig(), seed=11)


def test_imitation_targets_are_the_handcrafted_prior(dex, imitated):
    """The target has to be ``prior_over`` over the options the *search*
    enumerates. Built from the full option list it would be a different
    function, and the network would be fine-tuned on something it never saw."""
    from pkcm.engine.rng import Rng
    from pkcm.engine.legality import random_team
    from pkcm.engine.state import BattleConfig, new_battle
    from pkcm.envs.encoding import action_space_size
    from pkcm.search.policy import joint_actions, prior_over
    from pkcm.train.imitate import ImitateConfig
    from pkcm.train.samples import policy_target

    config = ImitateConfig()
    battle_config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"),
                                 battle_format="singles")
    teams = tuple(random_team(dex, battle_config.regulation,
                              Rng.from_seed(11 * 2 + offset).cursor(), "singles")
                  for offset in (1, 2))
    state = new_battle(battle_config, teams, seed=11)

    options = joint_actions(state, 0, config.max_branching)
    expected = policy_target(
        zip(options, prior_over(state, 0, options)),
        action_space_size(battle_config.registered, battle_config.brought),
        battle_config)
    first = next(s for s in imitated if s.player == 0)
    assert np.allclose(first.policy, expected)


def test_imitation_targets_are_distributions(imitated):
    assert imitated, "the battle produced nothing"
    for sample in imitated:
        assert sample.policy.sum() == pytest.approx(1.0, abs=1e-5)
        assert (sample.policy >= 0).all()
        assert -1.0 <= sample.value <= 1.0


def test_imitation_samples_carry_no_hidden_information(imitated):
    """Same rule as self-play, and it is easier to break here: the targets are
    computed from the full ``BattleState`` and only the observation is kept."""
    from pkcm.envs.encoding import MAX_BROUGHT

    early = min(imitated, key=lambda sample: sample.turn)
    assert (early.observation["species"][MAX_BROUGHT:] == 0).any(), (
        "the whole opposing team was visible")
    assert not early.observation["pp"][MAX_BROUGHT * 4:].any(), (
        "their PP is never shown")


def test_the_value_baseline_is_quoted_against_the_right_scale(imitated):
    """Self-play targets are +-1 so a constant zero scores exactly 1.0. These
    are material, so the same printed error means something else entirely and
    the baseline has to travel with it."""
    from pkcm.train.imitate import baseline_mae

    floor = baseline_mae(imitated)
    assert 0.0 <= floor < 1.0
    assert floor == pytest.approx(
        float(np.mean([abs(s.value) for s in imitated])))


def test_a_pretrained_network_can_be_loaded_into_a_fresh_one(pieces, imitated,
                                                             tmp_path):
    """What ``train_loop.py --init`` does. A shape mismatch here would only
    show up an hour into a run."""
    from pkcm.train.trainer import load_into

    vocabulary, sheet, action_space = pieces
    config = NetConfig(hidden=64, blocks=1)
    net = build(vocabulary, sheet, action_space, SCALAR_SIZE, config)
    fit(net, imitated, torch.device("cpu"), TrainConfig(epochs=1, batch_size=32))
    path = tmp_path / "net.pt"
    save(net, path, {"iteration": -1, "pretrained": "handcrafted-prior"})

    fresh = build(vocabulary, sheet, action_space, SCALAR_SIZE, config)
    payload = load_into(fresh, path, torch.device("cpu"))
    assert payload["pretrained"] == "handcrafted-prior"
    before, _ = net.evaluate([imitated[0].observation], "cpu")
    after, _ = fresh.evaluate([imitated[0].observation], "cpu")
    assert np.allclose(before, after)


def test_the_registered_six_carry_their_facts(dex, pieces):
    """At team preview nothing has been brought, so ``registered`` is the only
    non-zero input -- and the handcrafted pick prior is computed from types and
    base stats. Shown opaque ids alone, the policy agreed with it 5.0% of the
    time on a 24-way choice, which is exactly chance."""
    from pkcm.engine.legality import random_team
    from pkcm.engine.rng import Rng
    from pkcm.engine.state import BattleConfig, new_battle
    from pkcm.envs.encoding import encode_observation
    from pkcm.envs.observation import Observation

    vocabulary, sheet, action_space = pieces
    net = build(vocabulary, sheet, action_space, SCALAR_SIZE,
                NetConfig(hidden=64, blocks=1))
    # Blank the learned table, so anything the network still tells apart came
    # from the facts rather than from having memorised an id.
    with torch.no_grad():
        net.species.weight.zero_()

    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"),
                          battle_format="singles")
    encoded = []
    for seed in (3, 4):
        teams = tuple(random_team(dex, config.regulation,
                                  Rng.from_seed(seed * 2 + o).cursor(), "singles")
                      for o in (1, 2))
        state = new_battle(config, teams, seed=seed)
        assert not state.sides[0].selection, "this has to be the preview position"
        encoded.append(encode_observation(Observation.of(state, 0), vocabulary,
                                          sheet, dex))
    first, _ = net.evaluate([encoded[0]], "cpu")
    second, _ = net.evaluate([encoded[1]], "cpu")
    assert not np.allclose(first, second), (
        "two different registered sixes look identical at team preview -- the "
        "network cannot be picking on anything but chance")


def test_two_different_sets_do_not_encode_the_same_at_team_preview(dex, pieces):
    """The pick turns on what our six actually are, not on their names.

    Before our own sets were encoded, a physical Garchomp carrying Earthquake
    and a special one carrying Water Gun produced byte-identical preview
    observations, while the handcrafted pick prior told them apart. The target
    varied and the input did not: the policy could not beat chance at picking,
    and measured over three pre-training runs it never did -- 2.5%, 5.0%, 6.7%
    top-1 agreement on a 24-way choice, against 4.2% for guessing.
    """
    from pkcm.engine.pokemon import PokemonSet
    from pkcm.engine.state import BattleConfig, new_battle
    from pkcm.envs.encoding import encode_observation
    from pkcm.envs.observation import Observation

    vocabulary, sheet, _ = pieces

    def one(species, moves, sp):
        return PokemonSet(species=species, ability="__none__", moves=tuple(moves),
                          item=None, nature="serious", sp=sp)

    filler = [one(name, ("bodyslam",), (0,) * 6)
              for name in ("pikachu", "starmie", "gengar", "alakazam", "machamp")]
    foes = tuple([one("snorlax", ("bodyslam",), (0,) * 6)] + filler)
    physical = tuple([one("garchomp", ("earthquake", "dragonclaw", "firefang", "crunch"),
                          (0, 252, 0, 0, 0, 252))] + filler)
    special = tuple([one("garchomp", ("watergun", "confusion", "absorb", "ember"),
                         (252, 0, 252, 0, 0, 0))] + filler)

    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"),
                          battle_format="singles")
    encoded = []
    for team in (physical, special):
        state = new_battle(config, (team, foes), seed=1)
        assert not state.sides[0].selection, "this has to be the preview position"
        encoded.append(encode_observation(Observation.of(state, 0), vocabulary,
                                          sheet, dex))

    assert any(not np.array_equal(encoded[0][key], encoded[1][key])
               for key in encoded[0]), (
        "two completely different sets encode identically at team preview")


def test_the_opponents_sets_are_still_hidden_at_team_preview(dex, pieces):
    """Ours in full is legitimate; theirs is not. Preview shows their species
    and nothing else, and this is the easiest place in the project to leak."""
    from pkcm.engine.pokemon import PokemonSet
    from pkcm.engine.state import BattleConfig, new_battle
    from pkcm.envs.encoding import encode_observation
    from pkcm.envs.observation import Observation

    vocabulary, sheet, _ = pieces

    def one(species, moves):
        return PokemonSet(species=species, ability="__none__", moves=tuple(moves),
                          item=None, nature="serious", sp=(0,) * 6)

    filler = [one(name, ("bodyslam",))
              for name in ("pikachu", "starmie", "gengar", "alakazam", "machamp")]
    ours = tuple([one("snorlax", ("bodyslam",))] + filler)
    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"),
                          battle_format="singles")

    encoded = []
    for foe_moves in (("earthquake", "crunch"), ("watergun", "absorb")):
        theirs = tuple([one("garchomp", foe_moves)] + filler)
        state = new_battle(config, (ours, theirs), seed=1)
        encoded.append(encode_observation(Observation.of(state, 0), vocabulary,
                                          sheet, dex))

    for key in encoded[0]:
        assert np.array_equal(encoded[0][key], encoded[1][key]), (
            f"'{key}' changed when only the opponent's moves changed -- their "
            f"set is hidden at team preview")


def test_the_preview_grid_is_the_prior_the_pick_is_scored_by(dex, pieces):
    """The pick prior is a sum over our-six-by-their-six matchups. A flat MLP
    over a concatenation memorises those combinations instead of composing
    them -- a network four times the size fitted the training picks better and
    generalised worse. So the grid is handed over rather than rediscovered, and
    it has to be the *same* arithmetic the search scores picks with."""
    from pkcm.engine.legality import random_team
    from pkcm.engine.rng import Rng
    from pkcm.engine.state import BattleConfig, new_battle
    from pkcm.envs.analysis import matchup as pair_matchup
    from pkcm.envs.encoding import encode_preview
    from pkcm.envs.observation import Observation

    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"),
                          battle_format="singles")
    teams = tuple(random_team(dex, config.regulation,
                              Rng.from_seed(31 + offset).cursor(), "singles")
                  for offset in (1, 2))
    state = new_battle(config, teams, seed=31)
    observation = Observation.of(state, 0)
    grid = encode_preview(observation, dex)

    ours = state.parties[0][2]
    foe_id = observation.registered[1][4]
    expected = pair_matchup(dex, ours.moves, ours.stats, ours.species.types, foe_id)
    assert grid[2 * 6 + 4][2] == pytest.approx(expected, abs=1e-5)


def test_the_preview_grid_never_reads_their_set(dex, pieces):
    """Their half of every row comes from base-stat midpoints. Their moves are
    hidden at preview and a grid that leaked them would make the pick look
    brilliant in self-play and transfer nothing."""
    from pkcm.engine.pokemon import PokemonSet
    from pkcm.engine.state import BattleConfig, new_battle
    from pkcm.envs.encoding import encode_preview
    from pkcm.envs.observation import Observation

    def one(species, moves):
        return PokemonSet(species=species, ability="__none__", moves=tuple(moves),
                          item=None, nature="serious", sp=(0,) * 6)

    filler = [one(name, ("bodyslam",))
              for name in ("pikachu", "starmie", "gengar", "alakazam", "machamp")]
    ours = tuple([one("snorlax", ("bodyslam", "earthquake"))] + filler)
    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"),
                          battle_format="singles")

    grids = []
    for foe_moves in (("earthquake", "crunch"), ("watergun", "absorb")):
        theirs = tuple([one("garchomp", foe_moves)] + filler)
        state = new_battle(config, (ours, theirs), seed=1)
        grids.append(encode_preview(Observation.of(state, 0), dex))
    assert np.array_equal(grids[0], grids[1]), (
        "the grid moved when only the opponent's moves changed")


def test_the_evaluator_cache_survives_an_address_being_reused(dex, pieces):
    """``id()`` is only unique among objects that are alive at the same time.

    The cache was keyed by ``id(state)`` and did not hold the state, so a
    determinization dropped mid-search freed its address for the next one, and
    the next one was answered with its predecessor's evaluation. Nothing failed;
    the tree just got a wrong number sometimes, and which times depended on
    allocation order. It showed up as a pooled run and a serial run disagreeing
    about one battle in three from the same seed.

    Reusing an address on purpose is not something a test can arrange, so this
    plants the entry that a reuse would have left behind and checks it is not
    believed.
    """
    from pkcm.engine.legality import random_team
    from pkcm.engine.rng import Rng
    from pkcm.engine.state import BattleConfig, new_battle
    from pkcm.train.evaluator import Evaluator

    vocabulary, sheet, action_space = pieces
    net = build(vocabulary, sheet, action_space, SCALAR_SIZE,
                NetConfig(hidden=32, blocks=1))
    evaluator = Evaluator(net=net, dex=dex, device="cpu",
                          _vocabulary=vocabulary, _sheet=sheet)
    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"),
                          battle_format="singles")

    def a_state(seed: int):
        teams = tuple(random_team(dex, config.regulation,
                                  Rng.from_seed(seed * 2 + offset).cursor(),
                                  "singles") for offset in (1, 2))
        return new_battle(config, teams, seed=seed)

    state = a_state(11)
    honest = evaluator.value(state, 0)

    # What a reused address leaves behind: this state's key, another state's
    # answer. The old cache returned the planted number.
    stranger = a_state(12)
    key = (id(state), 0, state.turn, state.phase)
    evaluator._cache[key] = (stranger, (np.zeros(action_space, dtype=np.float32), -0.75))

    assert evaluator.value(state, 0) == pytest.approx(honest), (
        "the cache answered for a state it was not computed from")


def test_a_run_directory_always_means_the_same_wandb_run(tmp_path):
    """Power cuts are routine on the machine this runs on, so a run gets
    interrupted and resumed repeatedly. The id is derived from the output
    directory -- which is what ``--resume`` keys on -- so the chart resumes with
    the loop instead of the dashboard filling with fragments of one experiment.
    """
    from pkcm.train.logging import _id_for

    first = _id_for(tmp_path / "fifth")
    assert first == _id_for(tmp_path / "fifth")
    assert first == _id_for(tmp_path / "." / "fifth"), "resolve, not string match"
    assert first != _id_for(tmp_path / "sixth")
    assert first.isalnum() and len(first) == 16



# --------------------------------------------------------------------------- #
# Rehearsal -- teaching the policy head without arguing with the value head
# --------------------------------------------------------------------------- #


def test_a_zero_weighted_row_does_not_move_the_value_head(pieces, samples):
    """The assumption the whole rehearsal design rests on.

    An imitation sample's value target is the heuristic, on roughly a twelfth
    of the scale of the win/loss the loop fits. Mixed in at full weight it
    would ask the value head to satisfy two different questions and it would
    answer with the average. ``value_weight=0`` has to mean *silent*, not
    *quiet*.
    """
    from dataclasses import replace

    vocabulary, sheet, action_space = pieces
    honest = [replace(s, battle=i % 6) for i, s in enumerate(samples)]
    # The same rows again, with an absurd value target and no vote.
    muted = [replace(s, battle=i % 6, value=-1.0 if s.value > 0 else 1.0,
                     value_weight=0.0)
             for i, s in enumerate(samples)]

    scores = {}
    for label, rows in (("clean", honest), ("with muted lies", honest + muted)):
        torch.manual_seed(0)
        net = build(vocabulary, sheet, action_space, SCALAR_SIZE,
                    NetConfig(hidden=64, blocks=1))
        result = fit(net, rows, torch.device("cpu"),
                     TrainConfig(epochs=1, batch_size=32, validation_fraction=0.0))
        scores[label] = result["value_mae"]

    # Rows carrying inverted targets at zero weight must not drag the value
    # error around. Some movement is inevitable -- the shared trunk still gets
    # policy gradients from them -- but not the wholesale corruption that
    # counting them would cause.
    assert abs(scores["with muted lies"] - scores["clean"]) < 0.35, scores


def test_value_error_is_averaged_over_the_rows_that_voted(pieces, samples):
    """Not over the batch. A batch that is mostly rehearsal would otherwise
    report a value error diluted by rows that abstained, which would read as
    the value head improving when nothing had happened to it."""
    from dataclasses import replace

    vocabulary, sheet, action_space = pieces
    voting = [replace(s, battle=i % 6) for i, s in enumerate(samples)]
    padded = voting + [replace(s, battle=i % 6, value_weight=0.0)
                       for i, s in enumerate(samples)]

    torch.manual_seed(0)
    net = build(vocabulary, sheet, action_space, SCALAR_SIZE,
                NetConfig(hidden=64, blocks=1))
    one = fit(net, voting, torch.device("cpu"),
              TrainConfig(epochs=1, batch_size=1024, validation_fraction=0.0))

    torch.manual_seed(0)
    net = build(vocabulary, sheet, action_space, SCALAR_SIZE,
                NetConfig(hidden=64, blocks=1))
    two = fit(net, padded, torch.device("cpu"),
              TrainConfig(epochs=1, batch_size=1024, validation_fraction=0.0))

    # Doubling the rows with abstainers must not halve the reported error.
    assert two["value_mae"] > one["value_mae"] * 0.6, (one, two)
