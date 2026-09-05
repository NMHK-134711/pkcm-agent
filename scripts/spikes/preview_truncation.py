"""Does keeping only 24 of the 120 team-preview orderings throw the answer away?

The first attempt asked where the wide search's single favourite sat in
``_pick_promise``'s ranking. That was too weak: over 120 options the winner
can hold 4% of the visits, and an argmax at 4% is not an opinion.

So ask it of the whole distribution instead. Search all 120, then measure how
much of the search's visit mass landed on the 24 that ``joint_actions`` would
have kept. Twenty-four of a hundred and twenty is 20%, so 20% is what a
ranking that knows nothing scores. Well above it means the truncation keeps
what matters. At or below it means the search's budget is being spent on a
list chosen by something no better than chance.

Also reported: the same for the top 48, and Spearman between promise rank and
visit share, which says whether the ranking is ordered or merely lucky at the
top.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pkcm.data.dex import load_dex
from pkcm.engine.legality import ranker_parties
from pkcm.engine.rng import Rng
from pkcm.engine.state import BattleConfig, new_battle
from pkcm.search import SearchConfig
from pkcm.search.mcts import MCTS
from pkcm.search.policy import joint_actions


def spearman(xs, ys):
    """Rank correlation, with no scipy in the venv."""
    def ranked(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            mean = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = mean
            i = j + 1
        return out

    n = len(xs)
    if n < 3:
        return None
    rx, ry = ranked(xs), ranked(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return round(num / den, 4) if den else None


def main() -> int:
    for out in (sys.stdout, sys.stderr):
        try:
            out.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=25600)
    parser.add_argument("--out", default=r"C:\pkcm-agent\runs\preview_truncation.json")
    args = parser.parse_args()
    out = Path(args.out) if args.out else ROOT / "runs/preview_truncation.json"

    dex = load_dex()
    parties = ranker_parties(None)
    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"),
                          battle_format="singles")

    rng = random.Random(11)
    pairs = set()
    for a in (39, 43, 14, 42):
        for b in (39, 43, 14, 42, 10, 37):
            if a != b:
                pairs.add((a, b))
    while len(pairs) < args.pairs:
        a, b = rng.sample(range(len(parties)), 2)
        pairs.add((a, b))
    pairs = sorted(pairs)[:args.pairs]

    wide = MCTS(SearchConfig(iterations=args.iterations, max_branching=120),
                evaluator=None)
    rows = []
    started = time.time()
    for index, (a, b) in enumerate(pairs, 1):
        state = new_battle(config, (parties[a].team, parties[b].team), seed=1)
        ranked = joint_actions(state, 0, 120, wide.config.switch_matchup,
                               wide.config.switch_promise)
        rank_of = {tuple(choice): rank for rank, choice in enumerate(ranked, 1)}

        result = wide.choose(state, 0, Rng.from_seed(100).cursor())
        shares = {tuple(choice): share for choice, share in result.distribution}
        kept = sum(share for choice, share in shares.items()
                   if rank_of.get(choice, 999) <= 24)
        kept48 = sum(share for choice, share in shares.items()
                     if rank_of.get(choice, 999) <= 48)
        ranks = [rank_of.get(choice, 999) for choice in shares]
        rho = spearman([-r for r in ranks], list(shares.values()))
        best = max(shares.items(), key=lambda pair: pair[1])
        rows.append({
            "a": a, "b": b, "options": len(ranked),
            "mass_in_top24": round(kept, 4),
            "mass_in_top48": round(kept48, 4),
            "spearman": rho,
            "top_share": round(best[1], 4),
            "top_rank": rank_of.get(best[0]),
        })
        spent = time.time() - started
        print(f"[{index:3}/{len(pairs)}] {a:2} vs {b:2}  visit mass in the kept "
              f"24: {kept:5.1%}  (top48 {kept48:5.1%})  spearman {rho}  "
              f"argmax rank {rank_of.get(best[0]):3} at {best[1]:.1%}  "
              f"{spent / index:.0f}s/pos", flush=True)

    mass = [row["mass_in_top24"] for row in rows]
    mass48 = [row["mass_in_top48"] for row in rows]
    rhos = [row["spearman"] for row in rows if row["spearman"] is not None]
    summary = {
        "iterations": args.iterations,
        "positions": len(rows),
        "baseline_top24": round(24 / 120, 4),
        "mean_mass_in_top24": round(sum(mass) / len(mass), 4),
        "median_mass_in_top24": round(sorted(mass)[len(mass) // 2], 4),
        "positions_below_baseline": sum(1 for m in mass if m <= 24 / 120),
        "mean_mass_in_top48": round(sum(mass48) / len(mass48), 4),
        "baseline_top48": round(48 / 120, 4),
        "mean_spearman": round(sum(rhos) / len(rhos), 4) if rhos else None,
        "rows": rows,
    }
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print()
    print(f"visit mass landing in the kept 24: mean {summary['mean_mass_in_top24']:.1%}, "
          f"median {summary['median_mass_in_top24']:.1%}, against a 20.0% baseline")
    print(f"top 48: mean {summary['mean_mass_in_top48']:.1%} against 40.0%")
    print(f"positions at or below the baseline: "
          f"{summary['positions_below_baseline']}/{len(rows)}")
    print(f"mean spearman(promise rank, visit share): {summary['mean_spearman']}")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
