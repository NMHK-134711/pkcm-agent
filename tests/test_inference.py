"""The inference server answers with exactly what the in-process net would.

The server exists to change where the forward runs, not what it computes, and
the seam is one method wide -- so the tests hold the stand-in to the real
network's behaviour: same shapes, same rows-to-requests accounting, agreement
within float tolerance, and interleaved callers getting their own answers back.

Everything here runs the server on the CPU. The GPU changes the last bits of
the output (a matmul reduces in a different order), which is documented and
accepted -- but it makes exact comparison meaningless, so correctness is
pinned where it can be pinned.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pkcm.data.dex import load_dex  # noqa: E402
from pkcm.envs.encoding import (SCALAR_SIZE, Vocabulary,  # noqa: E402
                                action_space_size, encode_observation)
from pkcm.envs.observation import Observation  # noqa: E402
from pkcm.envs.reference import sheet_for  # noqa: E402
from pkcm.engine.legality import make_team  # noqa: E402
from pkcm.engine.rng import Rng  # noqa: E402
from pkcm.engine.state import BattleConfig, new_battle  # noqa: E402
from pkcm.train.inference import (InferencePool, RemoteNet,  # noqa: E402
                                  ServerConfig)


@pytest.fixture(scope="module")
def world():
    dex = load_dex()
    vocabulary = Vocabulary.of(dex)
    sheet = sheet_for(dex, vocabulary)
    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"),
                          battle_format="singles")
    teams = tuple(make_team(dex, config.regulation, Rng.from_seed(s).cursor(),
                            "singles", "parties") for s in (1, 2))
    state = new_battle(config, teams, seed=1)
    encoded = [encode_observation(Observation.of(state, player),
                                  vocabulary, sheet, dex) for player in (0, 1)]
    return dex, vocabulary, sheet, encoded


@pytest.fixture(scope="module")
def server_config(tmp_path_factory, world):
    """A real checkpoint for the server to load, saved from a fresh net."""
    from pkcm.train.net import NetConfig, build
    from pkcm.train.trainer import save

    dex, vocabulary, sheet, _ = world
    action_space = action_space_size(6, 3)
    net = build(vocabulary, sheet, action_space, SCALAR_SIZE,
                NetConfig(hidden=64, blocks=1))
    path = tmp_path_factory.mktemp("inference") / "net.pt"
    save(net, path)
    return ServerConfig(checkpoint=str(path), action_space=action_space,
                        scalar_size=SCALAR_SIZE, device="cpu"), net


def test_server_matches_the_in_process_network(world, server_config):
    """Same observations, same answers -- the seam changes nothing."""
    _, _, _, encoded = world
    config, net = server_config
    torch.set_num_threads(1)
    local_probabilities, local_values = net.evaluate(encoded, "cpu")

    with InferencePool(config, workers=1) as pool:
        remote = pool.net_for(0)
        probabilities, values = remote.evaluate(encoded)

    assert probabilities.shape == local_probabilities.shape
    assert values.shape == local_values.shape
    np.testing.assert_allclose(probabilities, local_probabilities,
                               rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(values, local_values, rtol=1e-4, atol=1e-6)


def test_interleaved_workers_get_their_own_rows(world, server_config):
    """Two callers, different batch sizes, answers routed and sized right.

    The server merges whatever is queued into one forward and splits the
    result by request. The split is the part that can silently go wrong, so
    the two requests are made distinguishable by size.
    """
    _, _, _, encoded = world
    config, _ = server_config
    with InferencePool(config, workers=2) as pool:
        first, second = pool.net_for(0), pool.net_for(1)
        a_probabilities, a_values = first.evaluate(encoded)      # two rows
        b_probabilities, b_values = second.evaluate(encoded[:1])  # one row
        assert len(a_probabilities) == len(a_values) == 2
        assert len(b_probabilities) == len(b_values) == 1
        # The same observation must get the same answer through either request.
        np.testing.assert_allclose(a_probabilities[0], b_probabilities[0],
                                   rtol=1e-4, atol=1e-6)


def test_empty_batch_returns_empty(world, server_config):
    config, _ = server_config
    with InferencePool(config, workers=1) as pool:
        probabilities, values = pool.net_for(0).evaluate([])
    assert len(probabilities) == 0 and len(values) == 0


def _child_evaluates(remote, encoded):
    """Runs in a spawned child: the stand-in has to work over there."""
    probabilities, values = remote.evaluate(encoded)
    assert len(probabilities) == len(encoded) and len(values) == len(encoded)


def test_the_stand_in_survives_spawn(world, server_config):
    """A queue crosses to a child only through multiprocessing's own reducer,
    and that reducer only runs during an actual spawn -- so the only honest
    test is a real child process calling ``evaluate`` and coming back clean.
    This is exactly the road a self-play worker would take."""
    import multiprocessing as mp

    _, _, _, encoded = world
    config, _ = server_config
    with InferencePool(config, workers=1) as pool:
        remote = pool.net_for(0)
        child = mp.get_context("spawn").Process(
            target=_child_evaluates, args=(remote, encoded[:1]))
        child.start()
        child.join(120)
        assert child.exitcode == 0
