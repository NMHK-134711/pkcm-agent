"""One checkpoint against the handcrafted search, at a sample size that decides.

``pretrain.py`` and ``train_loop.py`` run forty matches because they run them
every couple of iterations. Forty matches is +-10 points, which cannot tell 30%
from 35% -- and six pre-training runs have now landed inside that band while the
imitation loss moved by a factor of two. When a decision rests on the number,
pay for the number.

    python scripts/judge.py runs/imitate6/net.pt --matches 200
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pkcm.search import SearchConfig  # noqa: E402
from pkcm.train.interval import wilson  # noqa: E402
from pkcm.train.matchup import MatchConfig, Record  # noqa: E402
from pkcm.train.matchup import stream as play  # noqa: E402
from pkcm.train.parallel import default_workers  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", default=None,
                        help="a saved network for side A. Omit to put "
                             "one search configuration against another")
    parser.add_argument("--matches", type=int, default=200,
                        help="each is two battles, the same teams both seats")
    parser.add_argument("--search-iterations", type=int, default=800)
    parser.add_argument("--trust", type=float, default=1.0)
    parser.add_argument("--trust-prior", type=float, default=None,
                        help="override --trust for the policy head alone")
    parser.add_argument("--trust-value", type=float, default=None,
                        help="override --trust for the value head alone. "
                             "--trust-prior 1 --trust-value 0 asks what the "
                             "network's prior is worth over the handcrafted "
                             "leaf value, and the other way round")
    parser.add_argument("--switch-matchup", type=float, default=None,
                        help="side A's switch ranking weight; 0 is the old flat "
                             "score for every switch")
    parser.add_argument("--switch-matchup-b", type=float, default=None,
                        help="side B's, to put the two directly against "
                             "each other")
    parser.add_argument("--leaf-batch", type=int, default=None,
                        help="side A's leaves per network forward")
    parser.add_argument("--leaf-batch-b", type=int, default=None,
                        help="side B's")
    parser.add_argument("--checkpoint-b", action="store_true",
                        help="side B gets the same network as side A, so the "
                             "match prices the search change on the search "
                             "that will run it")
    parser.add_argument("--belief", action="store_true",
                        help="side A draws the opponent's hidden fields from "
                             "the ranker pool rather than uniformly")
    parser.add_argument("--belief-b", action="store_true",
                        help="side B does too")
    parser.add_argument("--teams", default="random",
                        choices=("random", "ranker"),
                        help="which distribution teams come from. ``ranker`` "
                             "recombines the imported pkmnchamps parties; "
                             "37.5%% of random Pokemon carry no same-type "
                             "attack at all, against 4.9%% of those")
    parser.add_argument("--format", default="singles", choices=("singles", "doubles"))
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    workers = args.workers if args.workers is not None else default_workers()
    def search_for(weight, belief, leaf_batch):
        extra = {} if weight is None else {"switch_matchup": weight}
        if leaf_batch is not None:
            extra["leaf_batch"] = leaf_batch
        return SearchConfig(iterations=args.search_iterations,
                            determinizations=max(4, args.search_iterations // 20),
                            belief=belief, **extra)

    config = MatchConfig(
        checkpoint=args.checkpoint, battle_format=args.format, trust=args.trust,
        teams=args.teams, checkpoint_b=args.checkpoint_b,
        trust_prior=args.trust_prior, trust_value=args.trust_value,
        search=search_for(args.switch_matchup, args.belief, args.leaf_batch),
        search_b=search_for(args.switch_matchup_b, args.belief_b,
                            args.leaf_batch_b))

    def label(search, netted):
        return (f"search(switch={search.switch_matchup}, "
                f"belief={search.belief}, batch={search.leaf_batch}"
                f"{', net' if netted else ''})")

    side_a = label(config.search, args.checkpoint is not None)
    side_b = label(config.search_b, args.checkpoint_b)
    print(f"{side_a} vs {side_b} | {args.matches} matches | {args.teams} teams "
          f"| {workers} workers", flush=True)
    started = beat = time.perf_counter()
    total = Record()
    for done, one in enumerate(play(config, args.matches, workers), 1):
        total += one
        now = time.perf_counter()
        if now - beat >= 30.0 or done == args.matches:
            rate = done / max(now - started, 1e-9)
            print(f"  {done}/{args.matches}  {total.wins}-{total.losses}"
                  f"  ~{(args.matches - done) / rate:.0f}s left", flush=True)
            beat = now

    rate, low, high = wilson(total.wins, total.decided)
    print(f"\n  {total.wins}-{total.losses} ({total.draws} drawn) over "
          f"{total.decided} decided games")
    print(f"  {rate:.1%} [{low:.1%}, {high:.1%}]")
    rival = 'side B' if args.checkpoint_b else 'the handcrafted search'
    if low > 0.5:
        print(f"  stronger than {rival}.")
    elif high < 0.5:
        print(f"  weaker than {rival}, and separably so.")
    else:
        print(f"  not separable from {rival}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
