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
from typing import Any, Iterator

from pkcm.data.dex import Dex, load_dex
from pkcm.train.samples import Sample, SelfPlayConfig, play_one

#: Per-worker state, built once by the initialiser. Module-level because that
#: is the only thing a spawned worker and its task function reliably share.
_DEX: Dex | None = None
_CONFIG: SelfPlayConfig | None = None


def _start_worker(config: SelfPlayConfig, remote=None) -> None:
    global _DEX, _CONFIG
    _DEX = load_dex()
    _CONFIG = config
    # One process per core already saturates it; letting a maths library open
    # its own threads inside each worker oversubscribes and slows everything.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    if remote is not None:
        from pkcm.train.inference import RemoteNet
        from pkcm.train.samples import use_remote_net

        # Every worker gets the same initargs, so ownership of a reply queue
        # cannot be assigned by argument. Each worker claims an index off a
        # queue the parent pre-filled instead -- one pop, one queue, no two
        # workers ever waiting on the same channel.
        requests, replies, claims = remote
        mine = claims.get(timeout=30)
        use_remote_net(RemoteNet(requests, replies[mine], mine))


def _play(seed: int) -> list[Sample]:
    assert _DEX is not None and _CONFIG is not None, "worker was not initialised"
    return play_one(_DEX, _CONFIG, seed)


def default_workers() -> int:
    """Leave a core for the machine, and for whatever else is running on it."""
    return max(1, (os.cpu_count() or 2) - 1)


def map_unordered(task, items, *, initializer, initargs, workers: int,
                  attempts: int = 3, what: str = "task") -> Iterator[Any]:
    """Map ``task`` over ``items``, surviving a worker that dies mid-task.

    ``Pool.imap_unordered`` does not. When a worker segfaults -- and one did,
    twice, ten minutes into an arena on 2026-08-27 (``0xc0000005`` in
    python313.dll) -- the pool quietly starts a replacement, but the task the
    dead worker was holding is never rescheduled and never returned. The
    parent waits for it forever: no error, no CPU, no output. That run sat
    idle for seventy-eight minutes before anyone looked.

    ``ProcessPoolExecutor`` raises ``BrokenProcessPool`` instead, which is the
    difference between a hang and a retry. Items are re-run in a fresh pool,
    and whatever still will not survive is **dropped loudly** -- a silent cap
    reads as "we measured everything" when we did not.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool

    context = mp.get_context("spawn")
    remaining = list(items)
    for attempt in range(attempts):
        if not remaining:
            return
        done: set = set()
        pool = ProcessPoolExecutor(max_workers=min(workers, len(remaining)),
                                   mp_context=context,
                                   initializer=initializer, initargs=initargs)
        try:
            futures = {pool.submit(task, item): item for item in remaining}
            for future in as_completed(futures):
                result = future.result()
                done.add(futures[future])
                yield result
        except BrokenProcessPool:
            pass
        finally:
            # A broken pool has already lost its workers; waiting on them can
            # hang for the same reason we are here.
            pool.shutdown(wait=False, cancel_futures=True)

        remaining = [item for item in remaining if item not in done]
        if remaining and attempt + 1 < attempts:
            print(f"  ({len(remaining)} {what}(s) lost to a worker crash "
                  f"-- retrying in a fresh pool)")
    if remaining:
        print(f"  (!! {len(remaining)} {what}(s) dropped after {attempts} attempts)")


def generate(config: SelfPlayConfig, battles: int, seed: int = 0,
             workers: int | None = None,
             gpu_server: bool = False) -> Iterator[list[Sample]]:
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

    if not gpu_server:
        # Unordered: a battle that ends quickly should not wait behind a long
        # one, and nothing downstream cares which order they arrive in.
        yield from map_unordered(_play, seeds, initializer=_start_worker,
                                 initargs=(config,), workers=count,
                                 what="battle")
        return

    # One GPU process serves every worker's forward passes. Reply-queue
    # ownership is claimed by each worker off a pre-filled queue, because
    # ``map_unordered`` hands every worker identical initargs and two workers
    # waiting on one channel would swap answers. Retry pools claim fresh
    # queues from the same over-provisioned set; the server itself holds no
    # per-battle state.
    from pkcm.envs.encoding import SCALAR_SIZE, action_space_size
    from pkcm.engine.state import BattleConfig
    from pkcm.train.inference import InferencePool, ServerConfig

    dex = load_dex()
    battle_config = BattleConfig(dex=dex,
                                 regulation=dex.regulation(config.regulation),
                                 battle_format=config.battle_format)
    server = ServerConfig(
        checkpoint=config.checkpoint,
        action_space=action_space_size(battle_config.registered,
                                       battle_config.brought),
        scalar_size=SCALAR_SIZE)
    # Enough reply queues for every pool the retry loop might ever build.
    import multiprocessing as mp

    spawn = mp.get_context("spawn")
    with InferencePool(server, workers=count * 4) as pool:
        claims = spawn.Queue()
        for index in range(pool.workers):
            claims.put(index)
        yield from map_unordered(
            _play, seeds, initializer=_start_worker,
            initargs=(config, (pool.requests, pool.replies, claims)),
            workers=count, what="battle")


def quick(config: SelfPlayConfig, battles: int, **kwargs) -> list[Sample]:
    """Everything at once, for callers that want the list rather than a stream."""
    return [sample for batch in generate(config, battles, **kwargs) for sample in batch]
