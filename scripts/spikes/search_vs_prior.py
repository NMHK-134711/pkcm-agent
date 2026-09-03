"""Measurement 0 (Notion s17.4): does the search still beat its prior on the fixed engine?

Re-runs Notion s2.2 on the current engine, along the exact code path the loop
uses (play_one's loop, _evaluator, joint_actions, MCTS._prior). Per decision it
records the clean root prior and the root visit distribution and compares them.

s2.2 (old engine, ranker teams):  top-1 agree 48.4%, TV 0.362, prior mass on the
search's top action 0.302, visit share on it 0.586, Dirichlet-only TV 0.084.

Usage: python search_vs_prior.py [battles] [workers]
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

def _entropy(p):
    return -sum(x * math.log(x) for x in p if x > 0)

def _battle(seed):
    from pkcm.engine.actions import Action
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
    cursor = Rng.from_seed(seed ^ 0x5EED).cursor()
    rows = []
    while not state.finished and state.turn <= cfg.max_turns:
        chosen = []
        for player in (0, 1):
            options = joint_actions(state, player, cfg.search.max_branching,
                                    cfg.search.switch_matchup, cfg.search.switch_promise)
            result = search.choose(state, player, cursor)
            chosen.append(result.action)
            if not options or len(options) < 2 or result.iterations == 0:
                continue
            prior = search._prior(state, player, options)
            visits = dict(result.distribution)
            q = [visits.get(o, 0.0) for o in options]; z = sum(q) or 1.0; q = [x / z for x in q]
            p = list(prior); zp = sum(p) or 1.0; p = [x / zp for x in p]
            s_top = max(range(len(q)), key=q.__getitem__); p_top = max(range(len(p)), key=p.__getitem__)
            rows.append(dict(
                phase=("preview" if state.phase is Phase.TEAM_PREVIEW else
                       "switch" if state.phase is not Phase.BATTLE else
                       "t1-6" if state.turn <= 6 else "t7-12" if state.turn <= 12 else "t13+"),
                n=len(options), tv=0.5 * sum(abs(a - b) for a, b in zip(p, q)),
                agree=float(s_top == p_top), p_on_s=p[s_top], q_on_s=q[s_top],
                hp=_entropy(p), hq=_entropy(q)))
        state, _ = step(state, chosen[0], chosen[1])
    return rows

def main():
    from pkcm.search.mcts import SearchConfig
    from pkcm.train.samples import SelfPlayConfig
    from pkcm.train.parallel import map_unordered
    battles = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 19
    imitate = str(ROOT / "runs/imitate8/net.pt")
    four = sys.argv[3] if len(sys.argv) > 3 else "parties:39,43,14,42"
    out_name = sys.argv[4] if len(sys.argv) > 4 else "search_vs_prior_fixed"
    configs = [
        ("imitate8 | set both sides | noise .25", imitate, four, None, 0.25),
        ("handcrafted | set both sides | noise .25 (i0)", None, four, None, 0.25),
        
        
    ]
    out = {}
    for name, ckpt, teams, foe, noise in configs:
        cfg = SelfPlayConfig(teams=teams, foe_teams=foe, checkpoint=ckpt, trust=1.0,
                             search=SearchConfig(iterations=800, determinizations=40, leaf_batch=16,
                                                 root_noise=noise, sample_turns=12))
        t0 = time.perf_counter(); rows = []
        for got in map_unordered(_battle, [700000 + i for i in range(battles)],
                                 initializer=_init, initargs=(cfg,), workers=workers, what="battle"):
            rows.extend(got)
        dt = time.perf_counter() - t0
        def agg(sub):
            k = len(sub) or 1
            return dict(n=len(sub), tv=sum(r["tv"] for r in sub) / k, agree=sum(r["agree"] for r in sub) / k,
                        p_on_s=sum(r["p_on_s"] for r in sub) / k, q_on_s=sum(r["q_on_s"] for r in sub) / k,
                        hp=sum(r["hp"] for r in sub) / k, hq=sum(r["hq"] for r in sub) / k,
                        opts=sum(r["n"] for r in sub) / k)
        res = dict(seconds=round(dt), overall=agg(rows),
                   phases={ph: agg([r for r in rows if r["phase"] == ph]) for ph in ("preview", "switch", "t1-6", "t7-12", "t13+")})
        out[name] = res
        o = res["overall"]
        print(f"\n== {name}  ({battles} battles, {o['n']} decisions, {dt:.0f}s)", flush=True)
        print(f"   TV {o['tv']:.3f} | top-1 agree {o['agree']:.1%} | prior on search-top {o['p_on_s']:.3f} | "
              f"visits on search-top {o['q_on_s']:.3f} | H prior {o['hp']:.3f} -> visits {o['hq']:.3f} | opts {o['opts']:.1f}")
        for ph, a in res["phases"].items():
            print(f"   {ph:8} n={a['n']:4}  TV {a['tv']:.3f}  agree {a['agree']:.1%}  H {a['hp']:.3f}->{a['hq']:.3f}  opts {a['opts']:.1f}", flush=True)
    (ROOT / f"runs/{out_name}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\ns2.2 reference (old engine, ranker): TV 0.362 | agree 48.4% | prior on search-top 0.302 | visits on search-top 0.586")

if __name__ == "__main__":
    main()
