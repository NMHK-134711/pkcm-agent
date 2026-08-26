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


def show_assessment(env) -> None:
    """Print what the calculator can work out, the way a player would read it.

    The same numbers the ``matchup`` block feeds a network -- effectiveness,
    the damage bracket, whether it is a guaranteed knockout, who moves first.
    All of it from the observation, so none of it is anything the player cannot
    see for themselves.
    """
    env.reset()
    while env.battle_state().phase.name == "TEAM_PREVIEW":
        from pkcm.engine.state import legal_actions
        from pkcm.envs.encoding import encode_action

        picks = {}
        for player, agent in enumerate(AGENTS):
            action = legal_actions(env.battle_state(), player, 0)[0]
            picks[agent] = np.array([encode_action(action, env.config.registered,
                                                   env.config.brought)])
        env.step(picks)

    for position in range(env.positions):
        assessment = env.assess(0, position)
        if assessment is None:
            continue
        observation = env.observation_of(0)
        mine = next(k for k in observation.own if k.position == position)
        print()
        print(f"position {position}: {mine.species_id}")
        for slot, estimate in assessment.damage:
            target = next(k for k in observation.foe if k.slot == slot)
            note = " KO" if estimate.guaranteed_ko else ""
            print(f"  {estimate.move_id:16} -> {target.species_id:14} "
                  f"x{estimate.effectiveness:<5} {str(estimate.percent):>7}%  "
                  f"{str(estimate.hits_to_ko):>5} hits{note}")
        for slot, faster in assessment.outspeeds:
            target = next(k for k in observation.foe if k.slot == slot)
            answer = {True: "we move first", False: "they move first"}.get(
                faster, "depends on their spread")
            print(f"  speed vs {target.species_id:14} {answer}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", default="singles", choices=("singles", "doubles"))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true", help="no battle text")
    parser.add_argument("--explain", action="store_true",
                        help="print the damage calculator's view of turn one")
    args = parser.parse_args()

    env = ChampionsEnv(battle_format=args.format, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    print(f"action space   {env.action_space('player_0')}")
    print(f"observation    {sorted(env.observation_space('player_0').spaces)}")

    if args.explain:
        show_assessment(env)
        return 0

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
