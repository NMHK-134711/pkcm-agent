"""Give the team preview more thinking, rather than more options to think about.

``preview_branching`` asked whether the cap at 24 of 120 orderings throws the
answer away. The visit-mass measurement said the kept 24 are the wrong 24;
the games said removing the cap costs five points anyway. Both are true:
spreading a fixed budget over 120 orderings leaves 6.7 visits each, and
thirty-three visits on a badly chosen shortlist beats that.

So the lever is not width. It is depth on the shortlist we keep -- the preview
is one node in the whole game, and it is the largest single decision in it.
This gives that node ``--multiplier`` times the simulations and changes
nothing else.

    python scripts/spikes/preview_budget.py --matches 2000 --multiplier 4

Written as its own runner rather than through ``MatchConfig`` on purpose:
``matchup`` builds the MCTS itself and there is nowhere to hand it a
per-phase budget, and editing ``SearchConfig`` while an experiment is in
flight would have a freshly spawned worker importing different code from its
siblings. The team draw copies ``matchup``'s exactly, seeds included, so this
is comparable game for game with the branching runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pkcm.data.dex import Dex, load_dex  # noqa: E402
from pkcm.engine.legality import make_team  # noqa: E402
from pkcm.engine.rng import Rng  # noqa: E402
from pkcm.engine.state import BattleConfig, Phase, new_battle  # noqa: E402
from pkcm.search import MCTS, SearchConfig  # noqa: E402
from pkcm.search.policy import SearchPolicy, play_out  # noqa: E402
from pkcm.train.interval import wilson  # noqa: E402
from pkcm.train.parallel import default_workers, map_unordered  # noqa: E402

#: matchup.MatchConfig's default, so the same match number draws the same pair.
TEAM_SEED = 90000


class PreviewBudget(MCTS):
    """An MCTS that spends a different number of simulations at team preview."""

    def __init__(self, config: SearchConfig, preview: SearchConfig,
                 evaluator=None) -> None:
        super().__init__(config, evaluator)
        self.preview = preview

    def choose(self, state, player, cursor=None):
        if state.phase is not Phase.TEAM_PREVIEW:
            return super().choose(state, player, cursor)
        base = self.config
        self.config = self.preview
        try:
            return super().choose(state, player, cursor)
        finally:
            self.config = base


def teams_for(dex: Dex, battle_config: BattleConfig, kind: str, match: int):
    return tuple(
        make_team(dex, battle_config.regulation,
                  Rng.from_seed(TEAM_SEED + match * 2 + offset).cursor(),
                  battle_config.battle_format, kind)
        for offset in (1, 2)
    )


def play(dex: Dex, settings: dict, match: int) -> tuple[int, int, int]:
    """One pair of teams both ways round. Returns the boosted side's record."""
    battle_config = BattleConfig(dex=dex,
                                 regulation=dex.regulation(settings["regulation"]),
                                 battle_format=settings["format"])
    teams = teams_for(dex, battle_config, settings["teams"], match)
    plain = SearchConfig(iterations=settings["iterations"])
    boosted = replace(plain, iterations=settings["iterations"] * settings["multiplier"])

    wins = losses = draws = 0
    for swap in (False, True):
        ours = SearchPolicy(PreviewBudget(plain, boosted), Rng.from_seed(match).cursor())
        theirs = SearchPolicy(MCTS(plain), Rng.from_seed(match + 7777).cursor())
        policies = (theirs, ours) if swap else (ours, theirs)
        state = play_out(new_battle(battle_config, teams, seed=match), policies)

        seat = 1 if swap else 0
        if state.winner is None:
            draws += 1
        elif state.winner == seat:
            wins += 1
        else:
            losses += 1
    return wins, losses, draws


# -- worker state, the same shape as matchup's --------------------------------- #

_DEX: Dex | None = None
_SETTINGS: dict | None = None


def _start_worker(settings: dict) -> None:
    global _DEX, _SETTINGS
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    _DEX = load_dex()
    _SETTINGS = settings


def _play(match: int) -> tuple[int, int, int]:
    return play(_DEX, _SETTINGS, match)


def main() -> int:
    for out in (sys.stdout, sys.stderr):
        try:
            out.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matches", type=int, default=2000)
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--multiplier", type=int, default=4,
                        help="simulations at the preview node, as a multiple "
                             "of the rest of the game's")
    parser.add_argument("--teams", default="ranker")
    parser.add_argument("--format", default="singles")
    parser.add_argument("--regulation", default="m_b")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    settings = {"iterations": args.iterations, "multiplier": args.multiplier,
                "teams": args.teams, "format": args.format,
                "regulation": args.regulation}
    workers = args.workers if args.workers is not None else default_workers()
    out = Path(args.out) if args.out else \
        ROOT / f"runs/preview_budget_{args.iterations}x{args.multiplier}.json"

    print(f"preview at {args.iterations * args.multiplier} simulations vs the "
          f"usual {args.iterations}, everything else identical")
    print(f"{args.matches} pairs = {2 * args.matches} games on {workers} workers, "
          f"teams from {args.teams}")

    wins = losses = draws = 0
    started = time.time()
    if workers <= 1:
        dex = load_dex()
        results = (play(dex, settings, match) for match in range(args.matches))
    else:
        results = map_unordered(_play, range(args.matches),
                                initializer=_start_worker, initargs=(settings,),
                                workers=workers, what="match")
    for done, (w, l, d) in enumerate(results, 1):
        wins, losses, draws = wins + w, losses + l, draws + d
        if done % 25 == 0 or done == args.matches:
            rate, low, high = wilson(wins, wins + losses)
            spent = time.time() - started
            left = spent / done * (args.matches - done)
            print(f"  {done:4}/{args.matches}  {wins}-{losses}-{draws}  "
                  f"{rate:.1%} [{low:.1%}, {high:.1%}]  "
                  f"{spent / 60:.0f}m spent, {left / 60:.0f}m left", flush=True)

    rate, low, high = wilson(wins, wins + losses)
    separable = low > 0.5 or high < 0.5
    summary = {
        "iterations": args.iterations, "multiplier": args.multiplier,
        "preview_iterations": args.iterations * args.multiplier,
        "teams": args.teams, "matches": args.matches,
        "wins": wins, "losses": losses, "draws": draws,
        "rate": round(rate, 4), "low": round(low, 4), "high": round(high, 4),
        "separable": separable,
        "minutes": round((time.time() - started) / 60, 1),
    }
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print()
    print(f"boosted preview: {wins}-{losses}-{draws}  {rate:.1%} [{low:.1%}, {high:.1%}]")
    print("separable from even" if separable else
          "NOT separable from even -- the interval covers 50%")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
