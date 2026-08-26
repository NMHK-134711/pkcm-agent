"""The AlphaZero loop: play, learn, replace, repeat.

    1. the search plays itself, recording (observation, visit share, outcome)
    2. the network learns to predict both
    3. the next round's search uses it as its prior and its leaf value

And then the part that is not optional: measure against a fixed opponent.
Everything in this loop produces numbers that go up whether or not anything
improved. Training loss falls because the network is fitting. The root value
rises because the network is confident. Only ``scripts/arena.py`` knows whether
it plays better, and the ablation earlier in this project is the reason to
insist on it -- two configurations looked like slow progress at 53% and 57% and
were later shown to be no progress at all.

Usage:
    python scripts/train_loop.py --iterations 3 --battles 40
    python scripts/train_loop.py --iterations 10 --battles 200 --out runs/first
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
from pkcm.train.interval import wilson  # noqa: E402
from pkcm.train.logging import RunLog  # noqa: E402
from pkcm.train.net import NetConfig, build, pick_device  # noqa: E402
from pkcm.train.parallel import default_workers, generate  # noqa: E402
from pkcm.train.samples import SelfPlayConfig  # noqa: E402
from pkcm.train.trainer import TrainConfig, fit, save  # noqa: E402


def trust_for(iteration: int) -> float:
    """How far the search should believe the network this round.

    Zero on the first pass: there is no trained network yet, and a randomly
    initialised one given authority over the prior makes the search worse than
    no search. After that it climbs, because the handcrafted prior is a floor
    to get off, not a target.
    """
    if iteration == 0:
        return 0.0
    return min(1.0, 0.4 + 0.3 * iteration)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--battles", type=int, default=40, help="per iteration")
    parser.add_argument("--search-iterations", type=int, default=200)
    parser.add_argument("--format", default="singles", choices=("singles", "doubles"))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--buffer", type=int, default=40000,
                        help="samples kept across iterations")
    parser.add_argument("--out", type=Path, default=Path("runs/latest"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", default="pkcm-agent", help="wandb project")
    parser.add_argument("--name", default=None, help="wandb run name")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--evaluate-every", type=int, default=2,
                        help="battles against the handcrafted search, every N "
                             "iterations. 0 disables it -- and then nothing in "
                             "this loop measures anything")
    parser.add_argument("--evaluate-battles", type=int, default=24)
    args = parser.parse_args()

    dex = load_dex()
    vocabulary = Vocabulary.of(dex)
    sheet = sheet_for(dex, vocabulary)
    registered, brought = dex.regulation("m_b").bring_select(args.format)
    action_space = action_space_size(registered, brought)

    device = pick_device()
    net = build(vocabulary, sheet, action_space, SCALAR_SIZE,
                NetConfig(hidden=args.hidden, blocks=args.blocks)).to(device)
    settings = TrainConfig(epochs=args.epochs)
    optimiser = torch.optim.AdamW(net.parameters(), lr=settings.learning_rate,
                                  weight_decay=settings.weight_decay)

    workers = args.workers if args.workers is not None else default_workers()
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "net.pt"
    parameters = sum(p.numel() for p in net.parameters())
    print(f"device {device} | {workers} workers | {args.format} | "
          f"{parameters / 1e6:.2f}M parameters")

    log = RunLog(
        directory=args.out,
        project=args.project,
        name=args.name,
        use_wandb=not args.no_wandb,
        config={
            "format": args.format,
            "battles_per_iteration": args.battles,
            "search_iterations": args.search_iterations,
            "epochs": args.epochs,
            "hidden": args.hidden,
            "blocks": args.blocks,
            "buffer": args.buffer,
            "workers": workers,
            "parameters": parameters,
            "device": str(device),
            "seed": args.seed,
        },
    )
    if log.url:
        print(f"  wandb: {log.url}")

    buffer: list = []
    history: list[dict] = []
    for iteration in range(args.iterations):
        trust = trust_for(iteration)
        selfplay = SelfPlayConfig(
            battle_format=args.format,
            search=SearchConfig(
                iterations=args.search_iterations,
                determinizations=max(4, args.search_iterations // 20)),
            checkpoint=None if iteration == 0 else str(checkpoint),
            trust=trust,
        )

        started = time.perf_counter()
        fresh: list = []
        for batch in generate(selfplay, args.battles,
                              seed=args.seed + iteration * 10000, workers=workers):
            fresh.extend(batch)
        played = time.perf_counter() - started

        buffer.extend(fresh)
        buffer = buffer[-args.buffer:]

        started = time.perf_counter()
        losses = fit(net, buffer, device, settings, optimizer=optimiser)
        learned = time.perf_counter() - started
        save(net, checkpoint, {"iteration": iteration, "samples": len(buffer)})

        row = {
            "iteration": iteration,
            "trust": round(trust, 2),
            "fresh_samples": len(fresh),
            "buffer": len(buffer),
            "battles_per_second": round(args.battles / max(played, 1e-9), 3),
            "selfplay_seconds": round(played, 1),
            "train_seconds": round(learned, 1),
            # Training losses are diagnostics -- they fall because the network
            # is fitting what it was handed. The val/ rows are on battles it
            # has never seen, and a val_value_mae near 1.0 means it has learned
            # nothing, because predicting a constant zero scores exactly that.
            **{(f"val/{key[4:]}" if key.startswith("val_") else f"loss/{key}"):
               round(value, 4) for key, value in losses.items()},
        }

        if args.evaluate_every and (iteration + 1) % args.evaluate_every == 0:
            rate, low, high = measure(args, dex, checkpoint, iteration)
            beats = low > 0.5
            row["arena/win_rate_vs_search"] = round(rate, 4)
            row["arena/ci_low"] = round(low, 4)
            row["arena/ci_high"] = round(high, 4)
            row["arena/separable"] = float(beats or high < 0.5)
            print(f"        vs handcrafted search: {rate:.1%} "
                  f"[{low:.1%}, {high:.1%}]"
                  f"{'' if beats or high < 0.5 else '   (not separable)'}")

        history.append(row)
        log.log(row, step=iteration)

        print(f"  [{iteration}] trust {trust:.1f}  {len(fresh):5} new  "
              f"buffer {len(buffer):6}  play {played:6.1f}s  train {learned:5.1f}s")
        print(f"        train  policy {losses['policy_loss']:.3f}  "
              f"mae {losses['value_mae']:.3f}")
        if "val_value_mae" in losses:
            gap = losses["val_value_mae"] - losses["value_mae"]
            print(f"        held out  policy {losses['val_policy_loss']:.3f}  "
                  f"mae {losses['val_value_mae']:.3f}  "
                  f"(gap {gap:+.3f}; 1.0 = learned nothing)")

    log.artifact(checkpoint)
    last = history[-1] if history else {}
    log.summary({
        "iterations": args.iterations,
        "final_buffer": last.get("buffer", 0),
        "final_policy_loss": last.get("loss/policy_loss"),
        "final_value_mae": last.get("loss/value_mae"),
        "final_win_rate_vs_search": last.get("arena/win_rate_vs_search"),
    })
    log.finish()

    print(f"\nwrote {checkpoint}")
    print("the losses above are diagnostics. This is the measurement:")
    print(f"  python scripts/arena.py --a net --b search --battles 60 "
          f"--checkpoint {checkpoint}")
    return 0


def measure(args, dex, checkpoint: Path,
            iteration: int) -> tuple[float, float, float]:
    """Put the network's search against the handcrafted one.

    Not against greedy: greedy is the floor, and beating it says only that the
    search works. The question training has to answer is whether the network is
    better than the power-times-effectiveness score it replaced.
    """
    from pkcm.engine.legality import random_team
    from pkcm.engine.rng import Rng
    from pkcm.engine.state import BattleConfig, new_battle
    from pkcm.search import MCTS, SearchConfig
    from pkcm.search.policy import SearchPolicy, play_out
    from pkcm.train.evaluator import from_checkpoint

    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"),
                          battle_format=args.format)
    registered, brought = config.regulation.bring_select(args.format)
    evaluator = from_checkpoint(checkpoint, dex, action_space_size(registered, brought),
                                SCALAR_SIZE, device="cpu", trust=1.0)
    search = SearchConfig(iterations=args.search_iterations,
                          determinizations=max(4, args.search_iterations // 20))

    wins = losses = 0
    for match in range(args.evaluate_battles):
        teams = tuple(random_team(dex, config.regulation,
                                  Rng.from_seed(90000 + match * 2 + offset).cursor(),
                                  args.format) for offset in (1, 2))
        # Both seatings, so a win rate cannot come from the draw.
        for swap in (False, True):
            netted = SearchPolicy(MCTS(search, evaluator=evaluator),
                                  Rng.from_seed(match).cursor())
            plain = SearchPolicy(MCTS(search), Rng.from_seed(match + 7777).cursor())
            policies = (plain, netted) if swap else (netted, plain)
            state = play_out(new_battle(config, teams, seed=match), policies)
            net_side = 1 if swap else 0
            if state.winner is None:
                continue
            wins += state.winner == net_side
            losses += state.winner != net_side

    rate, low, high = wilson(wins, wins + losses)
    return rate, low, high


if __name__ == "__main__":
    raise SystemExit(main())
