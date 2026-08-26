"""Play the PettingZoo environment with a mask-respecting random policy.

The point of the demo is the *mask*. A policy that ignores it forfeits (see
``ChampionsEnv``), so this shows the minimum a policy has to do: read
``infos[agent]["action_mask"]``, and in doubles refuse to send the same Pokemon
to both positions -- the one rule that spans field positions and therefore
cannot live in a per-position mask.

Usage:
    python scripts/env_demo.py                       # one singles episode
    python scripts/env_demo.py --format doubles
    python scripts/env_demo.py --episodes 200 --quiet # throughput and win split
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pkcm.envs.champions import AGENTS, ChampionsEnv, sample_legal  # noqa: E402
from pkcm.envs.encoding import MAX_BROUGHT, SWITCH_BASE  # noqa: E402


def choose(infos: dict, agent: str, rng: np.random.Generator) -> np.ndarray:
    masks = infos[agent]["action_mask"]
    picks: list[int] = []
    taken: set[int] = set()
    for position in range(infos[agent]["decisions"]):
        mask = masks[position].copy()
        for slot in taken:
            mask[SWITCH_BASE + slot] = 0
        index = sample_legal(mask, rng)
        if SWITCH_BASE <= index < SWITCH_BASE + MAX_BROUGHT:
            taken.add(index - SWITCH_BASE)
        picks.append(index)
    return np.array(picks, dtype=np.int64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", default="singles", choices=("singles", "doubles"))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true", help="no battle text")
    args = parser.parse_args()

    env = ChampionsEnv(battle_format=args.format, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    print(f"action space   {env.action_space('player_0')}")
    print(f"observation    {sorted(env.observation_space('player_0').spaces)}")

    outcomes: Counter[str] = Counter()
    steps = 0
    start = time.perf_counter()
    for _ in range(args.episodes):
        _, infos = env.reset()
        while env.agents:
            actions = {agent: choose(infos, agent, rng) for agent in AGENTS}
            _, rewards, _, _, infos = env.step(actions)
            steps += 1
            if not args.quiet:
                text = env.render()
                if text:
                    print(text)
        winner = env.battle_state().winner
        outcomes["draw" if winner is None else AGENTS[winner]] += 1
    elapsed = time.perf_counter() - start

    print()
    print(f"{args.episodes} episodes, {steps} env steps in {elapsed:.2f}s "
          f"({steps / elapsed:.0f} steps/s)")
    for name, count in outcomes.most_common():
        print(f"  {name:10} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
