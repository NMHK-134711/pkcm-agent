"""Pre-train the network on the handcrafted prior, before AlphaZero touches it.

Two runs of the loop (``runs/second`` at prior_weight 1.5, ``runs/third`` at
0.35) both sat between 16% and 23% against the handcrafted search and neither
climbed out. The deficit is there at iteration 1, before self-play has had a
chance to do anything, because a freshly initialised network is a much worse
prior than ``policy.prior_over`` and the loop hands it the tree anyway.

This closes that gap by supervision rather than by hoping self-play pays it off.

    python scripts/pretrain.py --battles 4000 --out runs/imitate
    python scripts/train_loop.py --init runs/imitate/net.pt --out runs/fourth ...

**The number that decides this is the arena, not the loss.** A network that
has copied the prior should be indistinguishable from it -- 50%, not
separable. Anything clearly below and the copy is lossy, and the loop would
start by paying for the difference all over again.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pkcm.data.dex import load_dex  # noqa: E402
from pkcm.envs.encoding import SCALAR_SIZE, Vocabulary, action_space_size  # noqa: E402
from pkcm.envs.reference import sheet_for  # noqa: E402
from pkcm.search import SearchConfig  # noqa: E402
from pkcm.train.imitate import (  # noqa: E402
    ImitateConfig,
    baseline_mae,
    baseline_policy_loss,
    generate,
)
from pkcm.train.interval import wilson  # noqa: E402
from pkcm.train.net import NetConfig, build, pick_device  # noqa: E402
from pkcm.train.parallel import default_workers  # noqa: E402
from pkcm.train.trainer import TrainConfig, fit, save  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--battles", type=int, default=4000,
                        help="cheap games to draw positions from. No search "
                             "runs here, so these are seconds each, not minutes")
    parser.add_argument("--preview-battles", type=int, default=100000,
                        help="extra games drawn for their pick only, not played "
                             "out. A battle gives ~26 battle turns and exactly "
                             "2 picks, so a corpus balanced by battles is "
                             "starved at the game's largest decision")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--teams", default="random",
                        choices=("random", "ranker"),
                        help="which distribution teams come from. ``ranker`` "
                             "recombines the imported pkmnchamps parties; "
                             "37.5%% of random Pokemon carry no same-type "
                             "attack at all, against 4.9%% of those")
    parser.add_argument("--epsilon", type=float, default=0.25,
                        help="how often to move at random rather than greedily, "
                             "so the positions are not a thin greedy corridor")
    parser.add_argument("--format", default="singles", choices=("singles", "doubles"))
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=500000,
                        help="held away from the self-play seeds")
    parser.add_argument("--out", type=Path, default=Path("runs/imitate"))
    parser.add_argument("--evaluate-battles", type=int, default=40,
                        help="arena matches against the handcrafted search. "
                             "0 skips it -- and then nothing here is measured")
    parser.add_argument("--search-iterations", type=int, default=800,
                        help="for the arena only; match the loop's setting")
    args = parser.parse_args()

    dex = load_dex()
    vocabulary = Vocabulary.of(dex)
    sheet = sheet_for(dex, vocabulary)
    registered, brought = dex.regulation("m_b").bring_select(args.format)
    action_space = action_space_size(registered, brought)

    device = pick_device()
    net = build(vocabulary, sheet, action_space, SCALAR_SIZE,
                NetConfig(hidden=args.hidden, blocks=args.blocks)).to(device)
    workers = args.workers if args.workers is not None else default_workers()
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "net.pt"
    print(f"device {device} | {workers} workers | {args.format} | "
          f"{sum(p.numel() for p in net.parameters()) / 1e6:.2f}M parameters")

    config = ImitateConfig(battle_format=args.format, epsilon=args.epsilon,
                           teams=args.teams)
    started = beat = time.perf_counter()
    samples: list = []
    for done, batch in enumerate(generate(config, args.battles, seed=args.seed,
                                          workers=workers), 1):
        samples.extend(batch)
        now = time.perf_counter()
        if now - beat >= 30.0 or done == args.battles:
            rate = done / max(now - started, 1e-9)
            print(f"        positions {done}/{args.battles}  {rate:.1f} battles/s "
                  f"  {len(samples)} samples  ~{(args.battles - done) / rate:.0f}s left",
                  flush=True)
            beat = now
    played = len(samples)
    if args.preview_battles:
        picks = ImitateConfig(battle_format=args.format, epsilon=args.epsilon,
                              preview_only=True, teams=args.teams)
        for done, batch in enumerate(generate(
                picks, args.preview_battles,
                seed=args.seed + 10 * args.battles, workers=workers), 1):
            samples.extend(batch)
            now = time.perf_counter()
            if now - beat >= 30.0 or done == args.preview_battles:
                rate = done / max(now - started, 1e-9)
                print(f"        picks {done}/{args.preview_battles}  "
                      f"{len(samples)} samples total", flush=True)
                beat = now
    drawn = time.perf_counter() - started
    print(f"  {played} positions from {args.battles} battles, "
          f"{len(samples) - played} picks from {args.preview_battles} more, "
          f"in {drawn:.0f}s")

    floor = baseline_mae(samples)
    entropy = baseline_policy_loss(samples)
    losses = fit(net, samples, device, TrainConfig(epochs=args.epochs))
    save(net, checkpoint, {"iteration": -1, "samples": len(samples),
                           "pretrained": "handcrafted-prior"})

    print(f"  train     policy {losses['policy_loss']:.3f}  "
          f"mae {losses['value_mae']:.3f}")
    print(f"  held out  policy {losses.get('val_policy_loss', 0):.3f}  "
          f"mae {losses.get('val_value_mae', 0):.3f}")
    # Neither number reads on the self-play scale. The policy floor is the
    # targets' own entropy -- reaching it is perfect imitation, and no amount of
    # training goes below it. The value floor is what a constant zero scores;
    # the self-play "1.0 = learned nothing" does not carry over to material.
    print(f"            (floors: policy {entropy:.3f} = perfect imitation, "
          f"mae {floor:.3f} = predicting a constant zero)")
    learned = losses.get("val_policy_loss", 0) - entropy
    print(f"            policy is {learned:+.3f} nats off perfect imitation")
    print(f"  wrote {checkpoint}")

    record = {"battles": args.battles, "preview_battles": args.preview_battles,
              "samples": len(samples), "played_samples": played,
              "seconds": round(drawn, 1), "value_baseline": round(floor, 4),
              "policy_entropy_floor": round(entropy, 4),
              **{key: round(value, 4) for key, value in losses.items()}}

    if args.evaluate_battles:
        from pkcm.train.matchup import MatchConfig, Record
        from pkcm.train.matchup import stream as play

        print(f"  arena vs the handcrafted search "
              f"({args.evaluate_battles} matches, both seats)...", flush=True)
        match = MatchConfig(
            checkpoint=str(checkpoint), battle_format=args.format,
            teams=args.teams,
            search=SearchConfig(
                iterations=args.search_iterations,
                determinizations=max(4, args.search_iterations // 20)),
            trust=1.0)
        total = Record()
        for one in play(match, args.evaluate_battles, workers):
            total += one
        rate, low, high = wilson(total.wins, total.decided)
        separable = low > 0.5 or high < 0.5
        print(f"  vs handcrafted search: {rate:.1%} [{low:.1%}, {high:.1%}]"
              f"{'' if separable else '   (not separable -- which is the goal)'}")
        record.update(arena_win_rate=round(rate, 4), arena_ci_low=round(low, 4),
                      arena_ci_high=round(high, 4),
                      arena_separable=float(separable))
        if not separable:
            print("  the copy is worth what it copied. Start the loop from it.")
        else:
            print("  the copy is NOT worth what it copied -- more battles or "
                  "epochs before the loop is worth running.")

    (args.out / "summary.json").write_text(json.dumps(record, indent=2),
                                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
