"""Replay one gate match single-process, so a crash shows itself.

Run as:  python -X faulthandler -X dev replay_match.py <match> ; echo exit=$?
Exit 3221225477 (0xC0000005) = access violation, 3221225725 (0xC00000FD) = stack
overflow, a Traceback = Python caught it first, exit 0 = it does not reproduce alone.
"""
import faulthandler, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
faulthandler.enable(all_threads=True)

def main():
    import torch; torch.set_num_threads(1)
    from pkcm.data.dex import load_dex
    from pkcm.search.mcts import SearchConfig
    from pkcm.train.matchup import MatchConfig, play_match
    match = int(sys.argv[1])
    cfg = MatchConfig(checkpoint=str(ROOT / "runs/pilot43_v2/net.pt"),
                      checkpoint_b=str(ROOT / "runs/pilot43_v2/best.pt"),
                      battle_format="singles", teams="parties:43", foe_teams="parties",
                      search=SearchConfig(iterations=800, determinizations=40, leaf_batch=16),
                      trust=1.0)
    dex = load_dex(); t0 = time.perf_counter()
    print(f"replaying match {match} ...", flush=True)
    rec = play_match(dex, cfg, match)
    print(f"match {match} finished: {rec.wins}-{rec.losses} ({rec.draws} drawn) in {time.perf_counter()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
