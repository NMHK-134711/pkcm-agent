"""Measurement 0b: the search's self-consistency on the fixed engine (Notion s2.6 re-run).

Same position, two independent search seeds, same budget. TV and CE between the
two visit distributions bound how much of a policy target is dice. s2.6 (old
engine, 800 sims): H(A) 1.107, CE(A,B) 1.239, gap 0.132, TV 0.136, top-1 80.4%.

Usage: python noise_floor.py [battles] [workers]   -- run on an idle machine.
"""
import json, math, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
_DEX = None; _CFG = None; _SEARCH = None

def _init(cfg):
    global _DEX, _CFG, _SEARCH
    os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
    from pkcm.data.dex import load_dex
    from pkcm.search import MCTS
    from pkcm.train.samples import _evaluator
    _DEX, _CFG = load_dex(), cfg
    _SEARCH = MCTS(cfg.search, evaluator=_evaluator(_DEX, cfg))

def _H(p): return -sum(x * math.log(x) for x in p if x > 0)
def _CE(p, q): return -sum(a * math.log(max(b, 1e-9)) for a, b in zip(p, q) if a > 0)

def _battle(seed):
    from pkcm.engine.battle import step
    from pkcm.engine.legality import make_team
    from pkcm.engine.rng import Rng
    from pkcm.engine.state import BattleConfig, Phase, new_battle
    from pkcm.search.policy import joint_actions
    dex, cfg, search = _DEX, _CFG, _SEARCH
    bc = BattleConfig(dex=dex, regulation=dex.regulation(cfg.regulation), battle_format=cfg.battle_format)
    teams = tuple(make_team(dex, bc.regulation, Rng.from_seed(seed * 2 + off).cursor(), cfg.battle_format,
                            cfg.teams if off == 1 else (cfg.foe_teams or cfg.teams)) for off in (1, 2))
    state = new_battle(bc, teams, seed=seed)
    play_cursor = Rng.from_seed(seed ^ 0x5EED).cursor()
    rows = []
    while not state.finished and state.turn <= cfg.max_turns:
        chosen = []
        for player in (0, 1):
            options = joint_actions(state, player, cfg.search.max_branching,
                                    cfg.search.switch_matchup, cfg.search.switch_promise)
            a = search.choose(state, player, play_cursor)      # the move actually played
            chosen.append(a.action)
            if not options or len(options) < 2 or a.iterations == 0:
                continue
            b = search.choose(state, player, Rng.from_seed((seed * 9973 + state.turn * 31 + player) ^ 0xB0B).cursor())
            prior = search._prior(state, player, options)
            da, db = dict(a.distribution), dict(b.distribution)
            pa = [da.get(o, 0.0) for o in options]; pb = [db.get(o, 0.0) for o in options]
            za, zb, zp = sum(pa) or 1, sum(pb) or 1, sum(prior) or 1
            pa = [x / za for x in pa]; pb = [x / zb for x in pb]; pp = [x / zp for x in prior]
            ta = max(range(len(pa)), key=pa.__getitem__); tb = max(range(len(pb)), key=pb.__getitem__)
            rows.append(dict(
                phase=("preview" if state.phase is Phase.TEAM_PREVIEW else "switch" if state.phase is not Phase.BATTLE
                       else "t1-6" if state.turn <= 6 else "t7-12" if state.turn <= 12 else "t13+"),
                n=len(options), tv_ab=0.5 * sum(abs(x - y) for x, y in zip(pa, pb)),
                ce_ab=_CE(pa, pb), h_a=_H(pa), agree_ab=float(ta == tb),
                tv_a_prior=0.5 * sum(abs(x - y) for x, y in zip(pa, pp))))
        state, _ = step(state, chosen[0], chosen[1])
    return rows

def main():
    from pkcm.search.mcts import SearchConfig
    from pkcm.train.samples import SelfPlayConfig
    from pkcm.train.parallel import map_unordered
    battles = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 19
    imitate = str(ROOT / "runs/imitate8/net.pt"); four = "parties:39,43,14,42"
    configs = [("imitate8 | 4parties vs field | noise .25 (loop)", imitate, four, "parties", 0.25),
               ("imitate8 | 4parties vs field | noise 0",          imitate, four, "parties", 0.0),
               ("imitate8 | ranker vs ranker | noise .25 (s2.6 row)", imitate, "ranker", None, 0.25)]
    out = {}
    for name, ckpt, teams, foe, noise in configs:
        cfg = SelfPlayConfig(teams=teams, foe_teams=foe, checkpoint=ckpt, trust=1.0,
                             search=SearchConfig(iterations=800, determinizations=40, leaf_batch=16,
                                                 root_noise=noise, sample_turns=12))
        t0 = time.perf_counter(); rows = []
        for got in map_unordered(_battle, [800000 + i for i in range(battles)],
                                 initializer=_init, initargs=(cfg,), workers=workers, what="battle"):
            rows.extend(got)
        dt = time.perf_counter() - t0
        def agg(sub):
            k = len(sub) or 1
            return dict(n=len(sub), tv_ab=sum(r["tv_ab"] for r in sub) / k, ce_ab=sum(r["ce_ab"] for r in sub) / k,
                        h_a=sum(r["h_a"] for r in sub) / k, agree_ab=sum(r["agree_ab"] for r in sub) / k,
                        tv_a_prior=sum(r["tv_a_prior"] for r in sub) / k)
        res = dict(seconds=round(dt), overall=agg(rows),
                   phases={ph: agg([r for r in rows if r["phase"] == ph]) for ph in ("preview", "switch", "t1-6", "t7-12", "t13+")})
        out[name] = res; o = res["overall"]
        print(f"\n== {name}  ({battles} battles, {o['n']} decisions, {dt:.0f}s)", flush=True)
        print(f"   H(A) {o['h_a']:.3f} | CE(A,B) {o['ce_ab']:.3f} | gap {o['ce_ab']-o['h_a']:.3f} | TV(A,B) {o['tv_ab']:.3f} | top-1 agree {o['agree_ab']:.1%} | TV(A,prior) {o['tv_a_prior']:.3f}", flush=True)
        for ph, a in res["phases"].items():
            if a["n"]: print(f"   {ph:8} n={a['n']:4}  TV(A,B) {a['tv_ab']:.3f}  gap {a['ce_ab']-a['h_a']:.3f}  agree {a['agree_ab']:.1%}  TV(A,prior) {a['tv_a_prior']:.3f}", flush=True)
    (ROOT / "runs/noise_floor_fixed.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\ns2.6 reference (old engine, 800): H 1.107 | CE 1.239 | gap 0.132 | TV 0.136 | top-1 80.4%")

if __name__ == "__main__":
    main()
