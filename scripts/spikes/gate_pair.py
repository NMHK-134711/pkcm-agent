"""Gate-condition match between two saved networks: A vs B, same teams both sides.

Exactly train_loop's gate (MatchConfig checkpoint / checkpoint_b, 800 sims,
belief on, trust 1.0), so a known-good old net can be put against the incumbent
on the current engine. Usage:
  python gate_pair.py <ckpt_a> <ckpt_b> <teams> [matches=200] [workers=19]
"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))

def main():
    from pkcm.search.mcts import SearchConfig
    from pkcm.train.interval import wilson
    from pkcm.train.matchup import MatchConfig, Record, stream
    a, b, teams = sys.argv[1], sys.argv[2], sys.argv[3]
    matches = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 19
    cfg = MatchConfig(checkpoint=a, checkpoint_b=b, battle_format="singles", teams=teams,
                      search=SearchConfig(iterations=800, determinizations=40, leaf_batch=16), trust=1.0)
    print(f"A={a}\nB={b}\nteams={teams} both sides | {matches} matches | {workers} workers", flush=True)
    tally = Record(); t0 = time.perf_counter()
    for rec in stream(cfg, matches, workers):
        tally += rec
    _, lo, hi = wilson(tally.wins, tally.decided)
    print(f"\nA vs B: {tally.wins}-{tally.losses} ({tally.draws} drawn) over {tally.decided} decided games")
    print(f"A win rate {tally.wins / max(1, tally.decided):.1%} [{lo:.1%}, {hi:.1%}]  ({time.perf_counter()-t0:.0f}s)")

if __name__ == "__main__":
    main()
