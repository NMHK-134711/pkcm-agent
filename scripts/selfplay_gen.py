"""Generate self-play training data across every core.

The bottleneck in the whole pipeline. One battle is about twelve seconds of
search, so this is where the wall clock goes and where parallelism pays.

Usage:
    python scripts/selfplay_gen.py --battles 32                # measure throughput
    python scripts/selfplay_gen.py --battles 500 --out data/selfplay/run1.npz
    python scripts/selfplay_gen.py --battles 8 --workers 1     # single process
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pkcm.search import SearchConfig  # noqa: E402
from pkcm.train.parallel import default_workers, generate  # noqa: E402
from pkcm.train.samples import SelfPlayConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--battles", type=int, default=32)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--format", default="singles", choices=("singles", "doubles"))
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--determinizations", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None,
                        help="write the samples to an .npz instead of discarding them")
    args = parser.parse_args()

    config = SelfPlayConfig(
        battle_format=args.format,
        search=SearchConfig(iterations=args.iterations,
                            determinizations=args.determinizations),
    )
    workers = args.workers if args.workers is not None else default_workers()
    print(f"{args.battles} battles, {workers} workers, {args.format}, "
          f"{args.iterations} iterations")

    samples, battles = [], 0
    start = time.perf_counter()
    for batch in generate(config, args.battles, seed=args.seed, workers=workers):
        samples.extend(batch)
        battles += 1
        elapsed = time.perf_counter() - start
        print(f"\r  {battles}/{args.battles} battles, {len(samples)} samples, "
              f"{battles / elapsed:.2f} battles/s", end="", flush=True)
    elapsed = time.perf_counter() - start
    print()

    print(f"  {battles} battles in {elapsed:.1f}s")
    print(f"  {battles / elapsed:.2f} battles/s   {len(samples) / elapsed:.1f} samples/s")
    print(f"  {len(samples) / max(1, battles):.1f} samples per battle")
    print(f"  projected: {3600 * battles / elapsed:.0f} battles/hour")

    if args.out and samples:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            f"obs_{key}": np.stack([s.observation[key] for s in samples])
            for key in samples[0].observation
        }
        payload["policy"] = np.stack([s.policy for s in samples])
        payload["value"] = np.array([s.value for s in samples], dtype=np.float32)
        payload["player"] = np.array([s.player for s in samples], dtype=np.int8)
        payload["turn"] = np.array([s.turn for s in samples], dtype=np.int16)
        np.savez_compressed(args.out, **payload)
        size = args.out.stat().st_size / 1e6
        print(f"  wrote {args.out} ({size:.1f} MB, {len(samples)} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
