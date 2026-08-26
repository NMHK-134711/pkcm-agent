"""Self-play across every core, because generating games is the bottleneck.

One self-play battle costs about twelve seconds -- almost all of it engine
steps inside the search -- so a single process produces roughly three hundred
battles an hour. That is not enough data to train anything, and no amount of
GPU fixes it: the GPU is idle while the CPU plays Pokemon.

Sixteen cores make it sixteen times better, and the shape is embarrassingly
parallel: battles do not talk to each other.

Windows spawns rather than forks, so every worker starts from nothing. That
costs about a third of a second of dex loading each, paid once per worker
rather than once per battle -- which is why the pool is created with an
initialiser and workers are handed seeds, not states. A ``BattleState`` would
have to be pickled, and it holds the dex.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import replace
from typing import Iterator

from pkcm.data.dex import Dex, load_dex
from pkcm.train.samples import Sample, SelfPlayConfig, play_one

#: Per-worker state, built once by the initialiser. Module-level because that
#: is the only thing a spawned worker and its task function reliably share.
_DEX: Dex | None = None
_CONFIG: SelfPlayConfig | None = None


def _start_worker(config: SelfPlayConfig) -> None:
    global _DEX, _CONFIG
    _DEX = load_dex()
    _CONFIG = config
    # One process per core already saturates it; letting a maths library open
    # its own threads inside each worker oversubscribes and slows everything.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def _play(seed: int) -> list[Sample]:
    assert _DEX is not None and _CONFIG is not None, "worker was not initialised"
    return play_one(_DEX, _CONFIG, seed)


def default_workers() -> int:
    """Leave a core for the machine, and for whatever else is running on it."""
    return max(1, (os.cpu_count() or 2) - 1)


def generate(config: SelfPlayConfig, battles: int, seed: int = 0,
             workers: int | None = None) -> Iterator[list[Sample]]:
    """Play ``battles`` self-play games, yielding each one's samples as it lands.

    Yields per battle rather than returning a list so a caller can write to a
    replay buffer, print progress, or stop early without waiting for the rest.
    """
    count = workers if workers is not None else default_workers()
    seeds = [seed + index for index in range(battles)]

    if count <= 1:
        dex = load_dex()
        for one in seeds:
            yield play_one(dex, config, one)
        return

    context = mp.get_context("spawn")
    with context.Pool(count, initializer=_start_worker, initargs=(config,)) as pool:
        # Unordered: a battle that ends quickly should not wait behind a long
        # one, and nothing downstream cares which order they arrive in.
        yield from pool.imap_unordered(_play, seeds, chunksize=1)


def quick(config: SelfPlayConfig, battles: int, **kwargs) -> list[Sample]:
    """Everything at once, for callers that want the list rather than a stream."""
    return [sample for batch in generate(config, battles, **kwargs) for sample in batch]
