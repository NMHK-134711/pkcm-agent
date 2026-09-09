"""Evolve a party around one Pokemon, then judge the finalists properly.

    python scripts/evolve_party.py --core garchomp \
        --parties data/champions/parties_field.json \
        --population 24 --generations 8 --budget 3000 --workers 19

Two agents, on purpose. The search runs on the cheap one, because a generation
is thousands of battles; the finalists are re-scored with a rollout, because
the cheap one is measurably blind to exactly the parties worth finding -- ten
points on screens, nine on setup, nothing on plain offence. Search with the
one you can afford, decide with the one that is right.

``--judge 0`` skips the second stage when only the search is wanted.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:                       # pragma: no cover - piping
        pass

from pkcm.data.dex import load_dex                              # noqa: E402
from pkcm.search import SearchConfig                            # noqa: E402
from pkcm.train.parallel import default_workers                 # noqa: E402
from pkcm.train.party_evolve import (EvolveConfig, as_party,    # noqa: E402
                                     evolve, race)
from pkcm.train.party_floor import FloorConfig, score           # noqa: E402


def describe(dex, team) -> str:
    return " / ".join(dex.species[one.species].name for one in team)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", required=True,
                        help="the species every candidate carries, by id")
    parser.add_argument("--parties",
                        default=str(ROOT / "data/champions/parties_field.json"),
                        help="the field to be measured against, and the warm "
                             "start the population is seeded from")
    parser.add_argument("--regulation", default="m_c",
                        help="which roster the candidates must be legal in")
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--budget", type=int, default=3000,
                        help="battles a generation may spend, shared out by "
                             "successive halving over the population")
    parser.add_argument("--search-iterations", type=int, default=200,
                        help="the search that grades candidates. Low on "
                             "purpose: a generation is thousands of battles "
                             "and the finalists are re-judged anyway")
    parser.add_argument("--seed-games", type=int, default=4)
    parser.add_argument("--opponents", type=int, default=40,
                        help="how much of the field a candidate faces while "
                             "the search is running. The seed pass is every "
                             "opponent, so against 243 parties it is 486 "
                             "battles a candidate before any racing -- which "
                             "no generation can pay. The finalists face the "
                             "whole field")
    parser.add_argument("--judge", type=int, default=4,
                        help="how many finalists to re-score with a rollout "
                             "agent. 0 skips the second stage")
    parser.add_argument("--judge-iterations", type=int, default=400)
    parser.add_argument("--judge-budget", type=int, default=400)
    parser.add_argument("--judge-opponents", type=int, default=80,
                        help="the rollout agent costs about fifteen times a "
                             "material leaf, so the whole field is not "
                             "affordable here either. Every finalist faces the "
                             "same eighty, which is what makes them "
                             "comparable; it is not an absolute floor")
    parser.add_argument("--rollout-turns", type=int, default=20)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=900_000)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    dex = load_dex()
    regulation = dex.regulation(args.regulation)
    if args.core not in dex.species:
        parser.error(f"no species called {args.core!r}")
    workers = args.workers if args.workers is not None else default_workers()

    searching = SearchConfig(iterations=args.search_iterations,
                             determinizations=max(4, args.search_iterations // 20))
    config = EvolveConfig(
        core=args.core, parties=args.parties, population=args.population,
        generations=args.generations, budget=args.budget, seed=args.seed,
        floor=FloorConfig(parties=args.parties, search=searching,
                          regulation=args.regulation,
                          seed_games=args.seed_games,
                          opponents=args.opponents))

    print(f"core {dex.species[args.core].name} | population {args.population}"
          f" | {args.generations} generations | {args.budget} battles each"
          f" | {args.search_iterations} sims | {workers} workers", flush=True)

    started = time.perf_counter()

    def report(generation, survivors):
        best = survivors[0]
        elapsed = time.perf_counter() - started
        print(f"  gen {generation + 1}: {len(survivors)} left, "
              f"best floor {best.fitness:.3f} [{best.floor.low:.3f}, "
              f"{best.floor.high:.3f}] from {best.origin}  "
              f"({elapsed / 60:.0f}m)", flush=True)
        print(f"        {describe(dex, best.team)}", flush=True)

    final: list = []
    for survivors in evolve(dex, regulation, config, workers=workers,
                            on_generation=report):
        final = survivors

    finalists = sorted(final, key=lambda one: -one.fitness)[:max(1, args.judge)]
    print("\nsearch finished. finalists by the cheap agent:", flush=True)
    for rank, one in enumerate(finalists, 1):
        print(f"  {rank}. floor {one.fitness:.3f}  {describe(dex, one.team)}",
              flush=True)

    judged = []
    if args.judge:
        print(f"\nre-judging {len(finalists)} with a rollout agent "
              f"({args.judge_iterations} sims, {args.rollout_turns}-turn "
              f"greedy rollout). This is the number to read.", flush=True)
        # A wider field than the search used, and the same one for every
        # finalist. Not the whole field: see --judge-opponents.
        judging = FloorConfig(
            parties=args.parties, seed_games=args.seed_games,
            regulation=args.regulation,
            opponents=args.judge_opponents,
            search=SearchConfig(iterations=args.judge_iterations,
                                determinizations=max(4, args.judge_iterations // 20),
                                rollout_turns=args.rollout_turns,
                                rollout_policy="greedy"))
        for rank, one in enumerate(finalists, 1):
            floor = score(one.team, judging, args.judge_budget, workers=workers)
            judged.append((floor, one))
            print(f"  {rank}. floor {floor.cvar:.3f} [{floor.low:.3f}, "
                  f"{floor.high:.3f}] over {floor.games} games  "
                  f"{describe(dex, one.team)}", flush=True)
        judged.sort(key=lambda pair: -pair[0].cvar)
        print(f"\nbest under the agent that can see: "
              f"{describe(dex, judged[0][1].team)}", flush=True)

    if args.out:
        payload = {
            "core": args.core,
            "settings": {"population": args.population,
                         "generations": args.generations,
                         "budget": args.budget,
                         "search_iterations": args.search_iterations,
                         "judge_iterations": args.judge_iterations if args.judge else None},
            "finalists": [
                {"floor_cheap": one.fitness, "origin": one.origin,
                 **as_party(one.team, f"evolved on {args.core}")}
                for one in finalists],
            "judged": [
                {"floor": floor.cvar, "low": floor.low, "high": floor.high,
                 "games": floor.games,
                 **as_party(one.team, f"evolved on {args.core}")}
                for floor, one in judged],
        }
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False,
                                             indent=1) + "\n",
                                  encoding="utf-8")
        print(f"\nwritten to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
