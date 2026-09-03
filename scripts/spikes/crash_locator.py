"""Find which gate matches kill a worker on this machine, then name them.

Replays pilot43_v2's final gate (candidate net.pt vs incumbent best.pt, party 43
vs the field, 800 sims, 200 matches) through the same pool path the loop uses,
with one addition: each worker writes 'start <match>' before playing and
'done <match>' after. A match that started and never finished is the one that
took its worker down. Run: python crash_locator.py [workers]
"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
LOGDIR = Path(__file__).resolve().parent / "crashlog"

def _play_logged(match):
    from pkcm.train import matchup
    f = LOGDIR / f"w{os.getpid()}.txt"
    with f.open("a") as h: h.write(f"start {match}\n"); h.flush()
    rec = matchup._play(match)
    with f.open("a") as h: h.write(f"done {match}\n"); h.flush()
    return rec

def main():
    from pkcm.search.mcts import SearchConfig
    from pkcm.train.matchup import MatchConfig, Record, _start_worker
    from pkcm.train.parallel import map_unordered
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    cfg = MatchConfig(checkpoint=str(ROOT / "runs/pilot43_v2/net.pt"),
                      checkpoint_b=str(ROOT / "runs/pilot43_v2/best.pt"),
                      battle_format="singles", teams="parties:43", foe_teams="parties",
                      search=SearchConfig(iterations=800, determinizations=40, leaf_batch=16),
                      trust=1.0)
    tally = Record(); n = 0
    for rec in map_unordered(_play_logged, range(200), initializer=_start_worker,
                             initargs=(cfg,), workers=workers, what="match"):
        tally += rec; n += 1
    started, done = set(), set()
    for f in LOGDIR.glob("w*.txt"):
        for line in f.read_text().splitlines():
            kind, m = line.split(); (started if kind == "start" else done).add(int(m))
    killers = sorted(started - done)
    print(f"\ncompleted {n}/200 matches | started-but-never-finished: {killers}")
    print(f"record {tally.wins}-{tally.losses} ({tally.draws} drawn)")

if __name__ == "__main__":
    main()
