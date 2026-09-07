"""Read the field round robin as floors rather than means.

The field cycles, so a mean win rate mostly reports which opponents were in
it. What the party generator is being pointed at is the floor -- the mean of
the worst quarter of matchups -- so that is what this ranks by.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")

from pkcm.train.party_floor import Matchup, summarise  # noqa: E402


TAIL = 0.25


def load(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in
                path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return json.loads(path.read_text(encoding="utf-8"))["fixtures"]


def records(fixtures):
    pair = defaultdict(lambda: [0, 0, 0])
    for f in fixtures:
        row = pair[(f["a"], f["b"])]
        row[0] += f["a_wins"]
        row[1] += f["b_wins"]
        row[2] += f.get("draws", 0)
    seen = sorted({p for k in pair for p in k})
    out = {}
    for party in seen:
        ms = []
        for other in seen:
            if other == party:
                continue
            if (party, other) in pair:
                w, l, d = pair[(party, other)]
            elif (other, party) in pair:
                l, w, d = pair[(other, party)]
            else:
                continue
            ms.append(Matchup(other, w, l, d))
        out[party] = ms
    return out


def split_half(path: Path, by_party) -> None:
    """Is the tail real, or is it whichever matchups went unlucky?

    The worst quarter is chosen from 252 noisy estimates and then scored on
    the same games that chose it, which biases it down however many games
    there are. The size of that bias is measurable: choose the tail on one
    half of the games and score those same opponents on the other half, which
    had no say in picking them. The .jsonl keeps ``repeat``, so the halves are
    there; the result JSON has already pooled them.
    """
    progress = path.with_suffix(".jsonl")
    if not progress.exists():
        print("\n(no .jsonl beside the result, so no split-half check)")
        return
    rows = [json.loads(l) for l in
            progress.read_text(encoding="utf-8").splitlines() if l.strip()]
    a, b = (records([r for r in rows if r["repeat"] == which])
            for which in (0, 1))
    picked = honest = 0.0
    counted = 0
    for party in a:
        if party not in b:
            continue
        ordered = sorted(a[party], key=lambda m: m.rate)
        take = max(1, round(len(ordered) * TAIL))
        tail = {m.opponent for m in ordered[:take]}
        chose = [m for m in ordered[:take] if m.decided]
        score = [m for m in b[party] if m.opponent in tail and m.decided]
        if not chose or not score:
            continue
        picked += sum(m.wins for m in chose) / sum(m.decided for m in chose)
        honest += sum(m.wins for m in score) / sum(m.decided for m in score)
        counted += 1
    if not counted:
        return
    picked, honest = picked / counted, honest / counted
    print(f"\nsplit-half, over {counted} parties:")
    print(f"  tail chosen and scored on the same games : {picked:.3f}")
    print(f"  same opponents, scored on the other half : {honest:.3f}")
    print(f"  winner's-curse bias                      : {honest - picked:+.3f}")
    if honest - picked > 0.1:
        print("  -> the tail is mostly noise at this sample size. Read the "
              "mean ranking, and measure\n     floors with party_floor's "
              "racing instead.")


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/field_rr_800.json")
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    field = json.loads(
        Path("data/champions/parties_field.json").read_text(encoding="utf-8"))
    fixtures = load(path)
    by_party = records(fixtures)

    floors = {p: summarise(ms, TAIL) for p, ms in by_party.items()}
    complete = [p for p, ms in by_party.items() if len(ms) == len(field) - 1]
    print(f"{path.name}: {len(fixtures)} fixtures, "
          f"{sum(f['a_wins'] + f['b_wins'] + f.get('draws', 0) for f in fixtures)} games")
    print(f"{len(complete)}/{len(field)} parties have played the whole field\n")

    ranked = sorted(floors, key=lambda p: -floors[p].cvar)
    by_mean = sorted(floors, key=lambda p: -floors[p].mean)
    mean_rank = {p: i for i, p in enumerate(by_mean, 1)}

    print(f"{'#':>3} {'idx':>4} {'floor':>7} {'interval':>15} "
          f"{'mean':>7} {'min':>6} {'평균순위':>6}  party")
    for i, p in enumerate(ranked[:top], 1):
        f = floors[p]
        title = field[p]["title"][:52]
        star = "" if len(by_party[p]) == len(field) - 1 else " *"
        print(f"{i:>3} {p:>4} {f.cvar:>7.3f} "
              f"[{f.low:.3f}, {f.high:.3f}] {f.mean:>7.3f} {f.minimum:>6.3f} "
              f"{mean_rank[p]:>6}  {title}{star}")

    print("\nworst quarter of the top three:")
    for p in ranked[:3]:
        print(f"  [{p}] {field[p]['title'][:46]}")
        for m in floors[p].worst(5):
            print(f"      vs [{m.opponent:>3}] {m.rate:.2f} "
                  f"({m.wins}-{m.losses})  {field[m.opponent]['title'][:40]}")

    split_half(path, by_party)

    # Reported, not claimed. The field really is intransitive -- it was
    # measured at eighty games a pair -- but four games cannot establish an
    # edge, so these are a pointer at what to re-measure, not evidence.
    print("\ncycles among the top eight (a>b>c>a, each edge over 50%)."
          "\nAt four games an edge these are NOT separable -- a pointer at "
          "what to re-measure:")
    head = ranked[:8]
    rate, games = {}, {}
    for p in head:
        for m in by_party[p]:
            if m.opponent in head and m.decided:
                rate[(p, m.opponent)] = m.rate
                games[(p, m.opponent)] = m.decided
    found = 0
    for a in head:
        for b in head:
            for c in head:
                if len({a, b, c}) < 3:
                    continue
                if a != min(a, b, c):
                    continue
                if (rate.get((a, b), 0) > .5 and rate.get((b, c), 0) > .5
                        and rate.get((c, a), 0) > .5):
                    print(f"  {a} > {b} > {c} > {a}   "
                          f"({rate[(a,b)]:.2f}, {rate[(b,c)]:.2f}, "
                          f"{rate[(c,a)]:.2f})  {games[(a,b)]} games an edge")
                    found += 1
    if not found:
        print("  none -- the top eight sort")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
