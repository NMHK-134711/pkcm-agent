"""Play a Champions singles battle between two random policies and print it.

Also reports engine throughput, which is the number that decides how much search
the hybrid policy can afford later (docs/DESIGN.md §1a).

Usage:
    python scripts/selfplay_demo.py                 # watch one battle
    python scripts/selfplay_demo.py --seed 7
    python scripts/selfplay_demo.py --bench 2000    # throughput only
"""

from __future__ import annotations

import argparse
import sys
import time

from pkcm.data.dex import load_dex
from pkcm.engine.battle import step
from pkcm.engine.legality import random_team
from pkcm.engine.pokemon import compile_team
from pkcm.engine.rng import Rng
from pkcm.engine.state import BattleConfig, legal_actions, new_battle
from pkcm.render.names import Names
from pkcm.render.text import Renderer

TRAINERS = ("레드", "블루")


def make_battle(config, seed: int):
    teams = (
        random_team(config.dex, config.regulation, Rng.from_seed(seed).cursor()),
        random_team(config.dex, config.regulation, Rng.from_seed(~seed & 0xFFFF).cursor()),
    )
    return teams, new_battle(config, teams, seed=seed)


def play(config, seed: int, log_lines: list[str] | None,
         renderer: Renderer | None = None) -> tuple[int | None, int]:
    _, state = make_battle(config, seed)
    policy = Rng.from_seed(seed ^ 0x5EED).cursor()
    decisions = 0
    while not state.finished:
        actions = tuple(policy.choice(legal_actions(state, player)) for player in (0, 1))
        state, log = step(state, *actions)
        decisions += 1
        if log_lines is not None and renderer is not None:
            log_lines.append(renderer.render_log(log))
    return state.winner, state.turn


def show_teams(config, teams, names: Names) -> None:
    for player, team in enumerate(teams):
        print()
        print(f"{TRAINERS[player]}의 팀")
        for pokemon in compile_team(config.dex, team):
            moves = ", ".join(names.move(move.id) for move in pokemon.moves)
            item = names.item(pokemon.item) if pokemon.item else "-"
            print(
                f"  {names.species(pokemon.species.id):<16} @ {item:<12}"
                f"  HP {pokemon.max_hp:<4} 스피드 {pokemon.stats[5]:<4} | {moves}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bench", type=int, default=0, help="play N battles, print throughput only")
    parser.add_argument("--lang", default="ko", choices=("ko", "en"))
    args = parser.parse_args()

    dex = load_dex()
    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"), battle_format="singles")

    if args.bench:
        # Team generation is measured separately: it is a team-builder cost, not
        # an engine cost, and folding it in understates engine throughput badly.
        start = time.perf_counter()
        matches = [make_battle(config, args.seed + offset)[0] for offset in range(args.bench)]
        build_elapsed = time.perf_counter() - start

        start = time.perf_counter()
        turns = 0
        for offset, teams in enumerate(matches):
            state = new_battle(config, teams, seed=args.seed + offset)
            policy = Rng.from_seed((args.seed + offset) ^ 0x5EED).cursor()
            while not state.finished:
                actions = tuple(policy.choice(legal_actions(state, p)) for p in (0, 1))
                state, _ = step(state, *actions)
            turns += state.turn
        elapsed = time.perf_counter() - start

        print(f"team generation : {args.bench} teams in {build_elapsed:.2f}s "
              f"({args.bench / build_elapsed:.0f}/s)")
        print(f"battle engine   : {args.bench} battles in {elapsed:.2f}s")
        print(f"  {args.bench / elapsed:8.0f} battles/s")
        print(f"  {turns / elapsed:8.0f} turns/s")
        print(f"  {turns / args.bench:8.1f} turns/battle")
        return 0

    teams, _ = make_battle(config, args.seed)
    renderer = Renderer(args.lang, dex, TRAINERS)
    show_teams(config, teams, renderer.names)

    lines: list[str] = []
    winner, turns = play(config, args.seed, lines, renderer)
    print("\n" + "=" * 60)
    print("\n".join(lines))
    print("=" * 60)
    result = "무승부" if winner is None else f"{TRAINERS[winner]} 승리"
    print(f"{result} — {turns}턴.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
