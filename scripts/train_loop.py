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
import os
import sys
from dataclasses import replace
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
from pkcm.train.imitate import ImitateConfig  # noqa: E402
from pkcm.train.imitate import generate as imitation  # noqa: E402
from pkcm.train.samples import SelfPlayConfig  # noqa: E402
from pkcm.train.trainer import TrainConfig, fit, save  # noqa: E402


def trust_for(iteration: int, value_mae: float | None = None) -> float:
    """How far the search should believe the network this round.

    Zero on the first pass: there is no trained network yet, and a randomly
    initialised one given authority makes the search worse than no search.
    After that it climbs, because the handcrafted prior is a floor to get off,
    not a target.

    **Capped by what the network is measured to be worth.** The ramp alone
    handed it complete authority by the second iteration; at that point its
    held-out value error was 0.51 on a scale where predicting a constant zero
    scores 1.0, and the search it drove lost to the handcrafted one 16.2%
    [9.8, 25.8]. A value head that is wrong by half the range does not get to
    decide half the tree on a schedule -- it earns its say by scoring better.

    Held-out error rather than training error, because the training number for
    that same network was 0.12 and it meant nothing.
    """
    if iteration == 0:
        return 0.0
    ramp = min(1.0, 0.4 + 0.3 * iteration)
    if value_mae is None:
        return ramp
    return max(0.0, min(ramp, 1.0 - value_mae))


#: How often self-play and the arena say they are still alive.
#:
#: One line per iteration is seven minutes of silence, and **a run that has hung
#: prints exactly what a run that is merely slow prints**. One did, on
#: 2026-08-27: two arena workers died and the loop waited seventy-eight minutes
#: with no output, no error and no CPU. A heartbeat is the difference between
#: noticing that in a minute and noticing it in an hour.
HEARTBEAT_SECONDS = 30.0


def _heartbeat(done: int, total: int, what: str, started: float,
               last: float) -> float:
    """Print progress at most every ``HEARTBEAT_SECONDS``. Returns the new clock."""
    now = time.perf_counter()
    if now - last < HEARTBEAT_SECONDS and done < total:
        return last
    rate = done / max(now - started, 1e-9)
    remaining = (total - done) / rate if rate > 0 else 0.0
    print(f"        {what} {done}/{total}  {rate:.2f}/s  ~{remaining:.0f}s left",
          flush=True)
    return now



def save_state(path: Path, net, optimiser, buffer: list, history: list,
               iteration: int, earned: float | None) -> None:
    """Everything the next iteration needs, written so a power cut cannot eat it.

    ``torch.save`` straight to the real path is a long window in which the file
    is neither the old state nor the new one, and this runs on a machine whose
    power goes off when its owner leaves the room. Written beside it and moved
    into place instead: ``os.replace`` is atomic, so the file is always one
    whole checkpoint or the other.

    The replay buffer is the expensive part -- 40,000 samples is about 124 MB --
    and it is also the part that cannot be rebuilt. Weights survive in
    ``net.pt``; the buffer is seven iterations of self-play, and starting over
    without it is most of the run.
    """
    scratch = path.with_suffix(".tmp")
    torch.save({
        "iteration": iteration,
        "earned": earned,
        "history": history,
        "buffer": buffer,
        "net": net.state_dict(),
        "optimiser": optimiser.state_dict(),
    }, scratch)
    os.replace(scratch, path)


def load_state(path: Path, net, optimiser, device):
    """Pick a run back up, or say why it cannot be picked up."""
    payload = torch.load(path, map_location=device, weights_only=False)
    net.load_state_dict(payload["net"])
    optimiser.load_state_dict(payload["optimiser"])
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--battles", type=int, default=40, help="per iteration")
    parser.add_argument("--search-iterations", type=int, default=800,
                        help="simulations per decision when generating "
                             "self-play data, where time trades directly "
                             "against games")
    parser.add_argument("--eval-search-iterations", type=int, default=None,
                        help="simulations per decision in the in-loop arena. "
                             "Defaults to --search-iterations. Set it higher "
                             "to measure at the budget the agent would "
                             "actually play at: singles gains 6.5 to 7.6 "
                             "points per doubling and has not saturated, so "
                             "the cheap budget self-play needs understates "
                             "what the network is worth")
    parser.add_argument("--format", default="singles", choices=("singles", "doubles"))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--teams", default="random",
                        choices=("random", "ranker"),
                        help="which distribution teams come from. ``ranker`` "
                             "recombines the imported pkmnchamps parties; "
                             "37.5%% of random Pokemon carry no same-type "
                             "attack at all, against 4.9%% of those")
    parser.add_argument("--trust-prior", type=float, default=None,
                        help="how far self-play and the in-loop arena believe "
                             "the policy head. Without it the prior is "
                             "throttled by the value head's held-out error, "
                             "which is the right guard when the value head is "
                             "in play and nonsense when --trust-value is 0")
    parser.add_argument("--trust-value", type=float, default=None,
                        help="how far self-play AND the in-loop arena believe "
                             "the value head, apart from the policy head. 0 "
                             "keeps the handcrafted leaf value: measured, a "
                             "self-played network's policy head with the "
                             "heuristic scored 55.0%% where the same network's "
                             "value head scored 39.9%%")
    parser.add_argument("--rehearse", type=int, default=0,
                        help="imitation battles to regenerate and mix into "
                             "each training pass. Guards against forgetting "
                             "the pre-trained prior: pre-training fitted 2.1M "
                             "samples and the replay buffer holds 40,000, so "
                             "self-play can walk the network off what it was "
                             "started from. 0 is off")
    parser.add_argument("--rehearse-value", action="store_true",
                        help="let rehearsal train the value head too. Off by "
                             "default: the imitation value target is the "
                             "heuristic, on a twelfth of the scale of the "
                             "win/loss the loop fits, and mixing them asks the "
                             "head to satisfy two answers at once")
    parser.add_argument("--leaf-batch", type=int, default=16,
                        help="leaves per network forward in self-play search. "
                             "16 is 2.35x the sequential search and measured "
                             "51.0%% [46.1, 55.9] against it over 396 games. "
                             "1 restores the sequential path")
    parser.add_argument("--bootstrap-weight", type=float, default=0.0,
                        help="how much of the value target is the n-step "
                             "bootstrap rather than who won. 1.0 is MuZero's "
                             "target, 0.0 is AlphaZero's. The value head "
                             "fitted to outcomes measured 39.9%% against the "
                             "handcrafted search while the policy head from "
                             "the same network measured 55.0%%")
    parser.add_argument("--n-step", type=int, default=5,
                        help="how far forward the bootstrap looks")
    parser.add_argument("--search-value-weight", type=float, default=0.0,
                        help="how much of the value target is the search's "
                             "root value rather than who won. Off by default "
                             "until an arena run says it helps")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--buffer", type=int, default=40000,
                        help="samples kept across iterations")
    parser.add_argument("--out", type=Path, default=Path("runs/latest"))
    parser.add_argument("--resume", action="store_true",
                        help="carry on from <out>/state.pt if it is there. The "
                             "buffer and the optimiser moments are in it too, "
                             "so the run continues rather than restarting with "
                             "the same weights")
    parser.add_argument("--init", type=Path, default=None,
                        help="start from a saved network instead of a random "
                             "one -- e.g. scripts/pretrain.py's output. A random "
                             "network is a far worse prior than the handcrafted "
                             "one, and the loop spends every iteration it has "
                             "climbing back to par rather than past it")
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
    settings = TrainConfig(epochs=args.epochs,
                           search_value_weight=args.search_value_weight,
                           bootstrap_weight=args.bootstrap_weight)
    optimiser = torch.optim.AdamW(net.parameters(), lr=settings.learning_rate,
                                  weight_decay=settings.weight_decay)

    if args.init is not None:
        from pkcm.train.trainer import load_into

        payload = load_into(net, args.init, device)
        print(f"  started from {args.init} "
              f"({payload.get('pretrained', 'checkpoint')})")

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
            "eval_search_iterations": (args.eval_search_iterations
                                       or args.search_iterations),
            "epochs": args.epochs,
            "hidden": args.hidden,
            "blocks": args.blocks,
            "buffer": args.buffer,
            "workers": workers,
            "parameters": parameters,
            "device": str(device),
            "seed": args.seed,
            "init": str(args.init) if args.init else None,
        },
    )
    if log.url:
        print(f"  wandb: {log.url}")

    buffer: list = []
    history: list[dict] = []
    #: Last round's held-out value error, which is what this round's trust is
    #: capped by. ``None`` before there is one -- the ramp alone decides then.
    earned: float | None = None
    state_path = args.out / "state.pt"
    first = 0

    if args.resume and state_path.exists():
        payload = load_state(state_path, net, optimiser, device)
        buffer = payload["buffer"]
        history = payload["history"]
        earned = payload["earned"]
        first = payload["iteration"] + 1
        print(f"  resumed at iteration {first} "
              f"({len(buffer)} samples in the buffer, {len(history)} recorded)")
        # The rows before the cut are already on the chart: the wandb run is
        # keyed to the output directory, so this reconnects to it rather than
        # starting a second one. Re-sending them would only argue with it about
        # steps it has already recorded.
    elif args.resume:
        print(f"  (nothing to resume at {state_path} -- starting fresh)")

    for iteration in range(first, args.iterations):
        # A pre-trained network is not the random one the ramp was written for:
        # it has already been measured against the handcrafted search, so it
        # starts trusted and stays capped by its held-out error like any other.
        trust = (trust_for(iteration, earned) if args.init is None
                 else max(0.0, min(1.0, 1.0 - (earned if earned is not None else 0.0))))
        selfplay = SelfPlayConfig(
            battle_format=args.format,
            search=SearchConfig(
                iterations=args.search_iterations,
                determinizations=max(4, args.search_iterations // 20),
                leaf_batch=args.leaf_batch),
            checkpoint=None if iteration == 0 else str(checkpoint),
            n_step=args.n_step,
            trust=trust,
            trust_prior=args.trust_prior,
            trust_value=args.trust_value,
            teams=args.teams,
        )

        started = time.perf_counter()
        fresh: list = []
        beat = started
        for done, batch in enumerate(generate(
                selfplay, args.battles,
                seed=args.seed + iteration * 10000, workers=workers), 1):
            fresh.extend(batch)
            beat = _heartbeat(done, args.battles, "self-play", started, beat)
        played = time.perf_counter() - started

        buffer.extend(fresh)
        buffer = buffer[-args.buffer:]

        # Rehearsal. Regenerated rather than stored: imitation play is greedy
        # with no search behind it, so a few hundred battles cost seconds
        # against self-play's hour, and 2.1M encoded observations would be
        # tens of gigabytes on disk.
        rehearsal: list = []
        if args.rehearse:
            started = time.perf_counter()
            beat = started
            recall = ImitateConfig(battle_format=args.format, teams=args.teams)
            for done, batch in enumerate(imitation(
                    recall, args.rehearse,
                    seed=args.seed + 500000 + iteration * 10000,
                    workers=workers), 1):
                rehearsal.extend(batch)
                beat = _heartbeat(done, args.rehearse, "rehearsal", started, beat)
            if not args.rehearse_value:
                # The imitation value target is the heuristic and the loop's is
                # who won -- a twelfth of the scale. Asking the head to fit both
                # gets an average of two different questions, so rehearsal
                # speaks only to the policy head unless told otherwise.
                rehearsal = [replace(sample, value_weight=0.0)
                             for sample in rehearsal]
            recalled = time.perf_counter() - started

        started = time.perf_counter()
        losses = fit(net, buffer + rehearsal, device, settings, optimizer=optimiser)
        learned = time.perf_counter() - started
        save(net, checkpoint, {"iteration": iteration, "samples": len(buffer)})
        earned = losses.get("val_value_mae")

        row = {
            "iteration": iteration,
            "trust": round(trust, 2),
            "fresh_samples": len(fresh),
            "buffer": len(buffer),
            "rehearsal_samples": len(rehearsal),
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
            rate, low, high = measure(args, checkpoint, workers)
            beats = low > 0.5
            row["arena/win_rate_vs_search"] = round(rate, 4)
            row["arena/ci_low"] = round(low, 4)
            row["arena/ci_high"] = round(high, 4)
            row["arena/separable"] = float(beats or high < 0.5)
            print(f"        vs handcrafted search: {rate:.1%} "
                  f"[{low:.1%}, {high:.1%}]"
                  f"{'' if beats or high < 0.5 else '   (not separable)'}")

        history.append(row)
        save_state(state_path, net, optimiser, buffer, history, iteration, earned)
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


def measure(args, checkpoint: Path, workers: int) -> tuple[float, float, float]:
    """Put the network's search against the handcrafted one.

    Not against greedy: greedy is the floor, and beating it says only that the
    search works. The question training has to answer is whether the network is
    better than the power-times-effectiveness score it replaced.

    Across the same pool self-play uses. Played one at a time this was the
    slowest thing in the loop by a factor of four -- forty battles of search
    against search on one core, while the other eighteen sat out.
    """
    from pkcm.train.matchup import MatchConfig, Record
    from pkcm.train.matchup import stream as play

    # Both sides get it, so the comparison is still like for like -- it just
    # happens at a different operating point.
    budget = (args.eval_search_iterations if args.eval_search_iterations
              else args.search_iterations)
    config = MatchConfig(
        checkpoint=str(checkpoint), battle_format=args.format,
        teams=args.teams,
        search=SearchConfig(iterations=budget,
                            determinizations=max(4, budget // 20),
                            leaf_batch=args.leaf_batch),
        trust=1.0, trust_prior=args.trust_prior, trust_value=args.trust_value)

    started = beat = time.perf_counter()
    record = Record()
    for done, one in enumerate(play(config, args.evaluate_battles, workers), 1):
        record += one
        beat = _heartbeat(done, args.evaluate_battles, "arena", started, beat)
    return wilson(record.wins, record.decided)


if __name__ == "__main__":
    raise SystemExit(main())
