"""What a training run has done so far, without a dashboard.

``history.json`` is written after every iteration whether or not wandb is
reachable, so this works on a machine that was never logged in -- and on a run
that is still going.

    python scripts/watch_run.py runs/third
    python scripts/watch_run.py runs/third runs/second     # side by side
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def rows(directory: Path) -> list[dict]:
    path = directory / "history.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def show(directory: Path) -> None:
    history = rows(directory)
    print(f"\n{directory}  --  {len(history)} iteration(s)")
    if not history:
        print("  nothing yet")
        return
    print(f"  {'iter':>4} {'trust':>6} {'battles':>8} {'held-out mae':>13} "
          f"{'held-out policy':>16} {'arena vs search':>24}")
    for row in history:
        arena = ""
        if "arena/win_rate_vs_search" in row:
            arena = (f"{100*row['arena/win_rate_vs_search']:5.1f}% "
                     f"[{100*row['arena/ci_low']:.1f}, {100*row['arena/ci_high']:.1f}]")
        print(f"  {row['iteration']:>4} {row.get('trust', 0):>6.2f} "
              f"{row.get('fresh_samples', 0):>8} {row.get('val/value_mae', 0):>13.3f} "
              f"{row.get('val/policy_loss', 0):>16.3f} {arena:>24}")
    # The arena is the only line here that knows anything; the losses fall
    # because the network is fitting whatever it was handed.
    last = [r for r in history if "arena/win_rate_vs_search" in r]
    if last:
        print(f"  latest arena: {100*last[-1]['arena/win_rate_vs_search']:.1f}%"
              f"  (50% would mean the network matched the handcrafted search)")


if __name__ == "__main__":
    targets = sys.argv[1:] or ["runs/third"]
    for target in targets:
        show(Path(target))
