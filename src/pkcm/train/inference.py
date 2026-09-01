"""One GPU process serving the network to every self-play worker.

The workers run the network on the CPU, one process per physical core, and each
MCTS batch is ``leaf_batch`` leaves wide. Measured on this machine, at batch 16
a forward costs 2.87ms on one CPU thread and 1.62ms on the GPU including the
copy across; at batch 160 it is 15.40ms against 2.50ms. The GPU time barely
moves between those -- 0.9ms of compute either way -- because a network this
small is latency-bound, not compute-bound. Batching is nearly free once someone
is collecting the batches.

That is what this is: a server process holding the network on the GPU, and a
stand-in for ``ChampionsNet`` that the workers hold instead. The stand-in
satisfies exactly one method, ``evaluate``, which is the only thing
``Evaluator`` asks of a network -- so nothing in the search, the evaluator or
the self-play loop has to know this exists.

**What this does not buy.** Ten workers already run ten forwards at once, so
the throughput gain over the current arrangement is small: ten batches of 16 on
ten cores is about 56,000 positions a second, and one GPU taking 160 at a time
is about 64,000. The reason to do it is the other one -- the GPU's time is flat
in the batch size, so a *bigger* search becomes affordable at the same wall
clock, and search budget is the one lever whose payment this project has
measured repeatedly (+9.7, +7.6, +6.5 points per doubling).

**Numerics.** A GPU matmul reduces in a different order than a CPU one. The
last bits of the output differ, an argmax flips every so often, and the battle
goes somewhere else. Runs served this way are not comparable game-for-game with
runs that were not; they have to be compared by win rate, over enough games.

No artificial delay is added to grow batches. The server takes whatever is
queued when it wakes and runs that, so the batch grows exactly when the workers
are ahead of it and shrinks when they are not. A fixed wait would trade the
workers' latency for a batch size they may not need.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

#: How many requests the server will merge into one forward. Past this the GPU
#: stops being free -- measured, 512 costs 4.18ms against 2.50ms at 160 -- and
#: the workers at the back of the queue wait for leaves they did not ask about.
MAX_BATCH = 256

#: How long a worker will wait for its answer before deciding the server is
#: gone. Long enough to cover a GPU hiccup, short enough that a dead server is
#: an error rather than a hang.
REPLY_TIMEOUT = 120.0


@dataclass(frozen=True)
class ServerConfig:
    """What the server needs to rebuild the network it will serve."""

    checkpoint: str
    action_space: int
    scalar_size: int
    device: str = "cuda"
    max_batch: int = MAX_BATCH


class RemoteNet:
    """A stand-in for ``ChampionsNet`` that asks the server instead.

    Holds only the two queues, so it survives being pickled into a worker at
    spawn time -- a real network would not, and a CUDA context certainly would
    not.
    """

    def __init__(self, requests: Any, replies: Any, worker: int) -> None:
        self._requests = requests
        self._replies = replies
        self._worker = worker

    def evaluate(self, observations: list[dict[str, np.ndarray]],
                 device: Any = "cpu") -> tuple[np.ndarray, np.ndarray]:
        """The one method ``Evaluator`` calls. ``device`` is ignored: the
        server decides where the network lives."""
        if not observations:
            empty = np.zeros((0, 0), dtype=np.float32)
            return empty, np.zeros((0,), dtype=np.float32)
        self._requests.put((self._worker, observations))
        try:
            probabilities, values = self._replies.get(timeout=REPLY_TIMEOUT)
        except queue.Empty as error:  # pragma: no cover - a dead server
            raise RuntimeError(
                "the inference server did not answer in "
                f"{REPLY_TIMEOUT:.0f}s") from error
        return probabilities, values


def serve(config: ServerConfig, requests: Any, replies: list[Any],
          stop: Any) -> None:
    """Answer forward-pass requests until ``stop`` is set.

    Runs in its own process. Everything torch is imported here so the parent
    never builds a CUDA context it would have to fork around.
    """
    import torch

    from pkcm.data.dex import load_dex
    from pkcm.envs.encoding import Vocabulary
    from pkcm.envs.reference import sheet_for
    from pkcm.train.net import build, collate
    from pkcm.train.trainer import load_into

    dex = load_dex()
    vocabulary = Vocabulary.of(dex)
    sheet = sheet_for(dex, vocabulary)
    payload = torch.load(config.checkpoint, map_location="cpu", weights_only=False)
    net = build(vocabulary, sheet, config.action_space, config.scalar_size,
                payload.get("config"))
    device = torch.device(config.device)
    load_into(net, config.checkpoint, device)
    net = net.to(device).eval()

    served = batches = 0
    while not stop.is_set():
        try:
            first = requests.get(timeout=0.05)
        except queue.Empty:
            continue

        # Whatever else is already waiting joins this batch. Nothing is waited
        # for: the queue's depth is the only thing that decides the size.
        gathered = [first]
        rows = len(first[1])
        while rows < config.max_batch:
            try:
                more = requests.get_nowait()
            except queue.Empty:
                break
            gathered.append(more)
            rows += len(more[1])

        flat = [one for _, observations in gathered for one in observations]
        with torch.no_grad():
            batch = collate(flat, device)
            logits, value = net(batch)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            values = value.cpu().numpy()

        at = 0
        for worker, observations in gathered:
            width = len(observations)
            replies[worker].put((probabilities[at:at + width],
                                 values[at:at + width]))
            at += width
        served += len(flat)
        batches += 1

    # A last word for whoever is measuring: the mean batch is the whole story
    # about whether the workers are keeping the GPU fed.
    if batches:
        print(f"  inference server: {served} positions in {batches} batches, "
              f"mean {served / batches:.1f}", flush=True)


class InferencePool:
    """The server process and the queues, started and stopped as a unit."""

    def __init__(self, config: ServerConfig, workers: int) -> None:
        self.config = config
        self.workers = workers
        context = mp.get_context("spawn")
        self._requests = context.Queue()
        self._replies = [context.Queue() for _ in range(workers)]
        self._stop = context.Event()
        self._process = context.Process(
            target=serve, args=(config, self._requests, self._replies, self._stop),
            daemon=True)

    def start(self) -> "InferencePool":
        self._process.start()
        return self

    def net_for(self, worker: int) -> RemoteNet:
        """The stand-in one worker should hold. Pickles cleanly into spawn."""
        return RemoteNet(self._requests, self._replies[worker], worker)

    @property
    def requests(self):
        return self._requests

    @property
    def replies(self):
        return self._replies

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._process.pid is None:
            return                      # never started; nothing to join
        self._process.join(timeout)
        if self._process.is_alive():  # pragma: no cover - a wedged server
            self._process.terminate()

    def __enter__(self) -> "InferencePool":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()


def probe(config: ServerConfig, batch: list[dict[str, np.ndarray]],
          calls: int = 200, workers: int = 1) -> dict[str, float]:
    """Round-trip cost through the server, which is what a worker actually pays.

    The benchmark that decided to build this measured the forward alone. This
    measures the forward plus the queues, from the caller's side, which is the
    only number that can be compared against running the network in-process.
    """
    with InferencePool(config, workers) as pool:
        net = pool.net_for(0)
        for _ in range(5):
            net.evaluate(batch)
        started = time.perf_counter()
        for _ in range(calls):
            net.evaluate(batch)
        spent = time.perf_counter() - started
    return {"calls": calls, "batch": len(batch),
            "per_call_ms": spent / calls * 1000,
            "per_position_us": spent / calls / max(1, len(batch)) * 1e6}
