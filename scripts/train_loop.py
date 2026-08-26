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

        history.append({
            "iteration": iteration,
            "trust": round(trust, 2),
            "fresh_samples": len(fresh),
            "buffer": len(buffer),
            "selfplay_seconds": round(played, 1),
            "train_seconds": round(learned, 1),
            **{key: round(value, 4) for key, value in losses.items()},
        })
        (args.out / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8")

        print(f"  [{iteration}] trust {trust:.1f}  {len(fresh):5} new  "
              f"buffer {len(buffer):6}  play {played:6.1f}s  train {learned:5.1f}s  "
              f"policy {losses['policy_loss']:.3f}  "
              f"value {losses['value_loss']:.3f}  mae {losses['value_mae']:.3f}")

    print(f"\nwrote {checkpoint}")
    print("now measure it -- the losses above are not evidence of anything:")
    print(f"  python scripts/arena.py --a search --b greedy --battles 60 "
          f"--checkpoint {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
