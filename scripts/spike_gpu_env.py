"""Can the engine's inner loop be batched onto the GPU? Measure before building.

The reason to want this is not the current loop's speed. It is the search
budget: every simulation costs engine steps, so the budget is bounded by how
fast states can be stepped, and a batched environment would raise that bound
rather than shave it.

``docs`` §13 profiled a self-play second and found the engine at 48.7%, the
network forward at ~6%. Amdahl on that says making the engine *free* is worth
1.95x -- which is the wrong number for this question, because a batched
environment does not make one battle's engine free, it steps ten thousand
battles at once. What that is worth is a constant, and this measures it.

**What this spike implements is the easy half on purpose.** The damage formula,
type effectiveness by table lookup, STAB, crits, the roll, and applying the
result. No abilities, no items, no volatiles, no switch logic, no turn order --
the parts that are branchy, which is to say the parts a GPU is bad at and which
are most of ``effects.py``'s 16%. So whatever number comes out is an **upper
bound** on a real vectorised engine, and the decision rule follows from that:

* a big number does not prove the project works, it only fails to rule it out
* a small number rules it out, because the real thing can only be slower

Correctness first. The batched path is checked against the engine's own
``damage_formula`` on every case before anything is timed, because a fast
number from a wrong formula is worse than no number.

    python scripts/spike_gpu_env.py
    python scripts/spike_gpu_env.py --batch 65536 --steps 200
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from pkcm.data.dex import load_dex  # noqa: E402
from pkcm.engine.moves import (  # noqa: E402
    CRIT_MULTIPLIER_DEN,
    CRIT_MULTIPLIER_NUM,
    DAMAGE_ROLL_HIGH,
    DAMAGE_ROLL_LOW,
    STAB_DEN,
    STAB_NUM,
    damage_formula,
)
from pkcm.engine.stats import LEVEL  # noqa: E402


def type_table(dex) -> tuple[torch.Tensor, dict[str, int]]:
    """The type chart as a matrix, which is what the lookup becomes on a GPU.

    An 18x18 gather replaces a dict-of-dicts walk per hit -- one of the few
    places where the engine's work is genuinely table-shaped rather than
    branchy, and so one of the few that vectorises without a rewrite of the
    rules.
    """
    # The chart's own keys, not the species' -- a few species carry a legacy
    # type ("bird") that the chart has never had a row for.
    names = sorted(dex.type_chart._matrix)
    index = {name: i for i, name in enumerate(names)}
    table = torch.ones(len(names), len(names), dtype=torch.float32)
    for attacking in names:
        for defending in names:
            table[index[attacking], index[defending]] = \
                dex.type_chart.multiplier(attacking, (defending,))
    return table, index


def batched_damage(power, attack, defense, roll, crit, stab, effectiveness,
                   level: int = LEVEL) -> torch.Tensor:
    """``moves.damage_formula`` for a whole batch at once.

    Every floor division in the original is an integer floor here too. The
    order is the order: the formula truncates at each step and a reordering
    changes the answer by a point, which over a battle is the difference
    between a two-hit knockout and a three.
    """
    damage = ((2 * level // 5 + 2) * power * attack // defense) // 50 + 2
    damage = torch.where(crit, damage * CRIT_MULTIPLIER_NUM // CRIT_MULTIPLIER_DEN,
                         damage)
    damage = damage * roll // 100
    damage = torch.where(stab, damage * STAB_NUM // STAB_DEN, damage)
    # int(x * effectiveness) truncates toward zero, which for a non-negative
    # damage is a floor.
    return (damage.to(torch.float64) * effectiveness).to(torch.int64)


def check_against_the_engine(device: str, cases: int = 4096) -> None:
    """Every batched result equals the engine's, or the timing means nothing."""
    generator = torch.Generator().manual_seed(7)
    power = torch.randint(10, 251, (cases,), generator=generator)
    attack = torch.randint(50, 501, (cases,), generator=generator)
    defense = torch.randint(50, 501, (cases,), generator=generator)
    roll = torch.randint(DAMAGE_ROLL_LOW, DAMAGE_ROLL_HIGH + 1, (cases,),
                         generator=generator)
    crit = torch.randint(0, 2, (cases,), generator=generator).bool()
    stab = torch.randint(0, 2, (cases,), generator=generator).bool()
    choices = torch.tensor([0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
    effectiveness = choices[torch.randint(0, len(choices), (cases,),
                                          generator=generator)]

    mine = batched_damage(power.to(device), attack.to(device), defense.to(device),
                          roll.to(device), crit.to(device), stab.to(device),
                          effectiveness.to(device)).cpu()

    wrong = 0
    for i in range(cases):
        theirs = damage_formula(
            power=int(power[i]), attack=int(attack[i]), defense=int(defense[i]),
            roll=int(roll[i]), crit=bool(crit[i]), stab=bool(stab[i]),
            effectiveness=float(effectiveness[i]))
        if int(mine[i]) != theirs:
            if wrong < 5:
                print(f"    MISMATCH power={int(power[i])} atk={int(attack[i])} "
                      f"def={int(defense[i])} roll={int(roll[i])} "
                      f"crit={bool(crit[i])} stab={bool(stab[i])} "
                      f"eff={float(effectiveness[i])}: "
                      f"batched {int(mine[i])} != engine {theirs}")
            wrong += 1
    if wrong:
        raise SystemExit(f"{wrong}/{cases} disagree with the engine -- "
                         f"the timing below would be meaningless")
    print(f"  correctness: {cases}/{cases} match the engine exactly")


def time_batched(device: str, batch: int, steps: int, table: torch.Tensor) -> float:
    """Steps per second, where a step is one hit resolved for every battle."""
    generator = torch.Generator(device=device).manual_seed(11)
    types = table.to(device)
    n_types = types.shape[0]

    power = torch.randint(10, 251, (batch,), device=device)
    attack = torch.randint(50, 501, (batch,), device=device)
    defense = torch.randint(50, 501, (batch,), device=device)
    move_type = torch.randint(0, n_types, (batch,), device=device)
    foe_type = torch.randint(0, n_types, (batch,), device=device)
    stab = torch.randint(0, 2, (batch,), device=device).bool()
    hp = torch.full((batch,), 300, device=device, dtype=torch.int64)

    if device == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(steps):
        roll = torch.randint(DAMAGE_ROLL_LOW, DAMAGE_ROLL_HIGH + 1, (batch,),
                             device=device, generator=generator)
        crit = torch.randint(0, 24, (batch,), device=device,
                             generator=generator) == 0
        effectiveness = types[move_type, foe_type]
        damage = batched_damage(power, attack, defense, roll, crit, stab,
                                effectiveness)
        # Applying it is part of a step: clamp at zero and respawn the dead so
        # the batch stays full, which is what a real vectorised env does.
        hp = torch.clamp(hp - damage, min=0)
        dead = hp <= 0
        hp = torch.where(dead, torch.full_like(hp, 300), hp)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return batch * steps / elapsed


def time_engine(cases: int, dex) -> float:
    """The same hit through the engine's own formula, one at a time.

    ``damage_formula`` alone, not ``step`` -- the batched path implements this
    and nothing else, so this is the honest comparison. A whole ``step`` does
    far more, which is the point made in the summary.
    """
    import random

    rng = random.Random(3)
    args = [(rng.randint(10, 250), rng.randint(50, 500), rng.randint(50, 500),
             rng.randint(DAMAGE_ROLL_LOW, DAMAGE_ROLL_HIGH),
             rng.random() < 1 / 24, rng.random() < 0.5,
             rng.choice([0.0, 0.25, 0.5, 1.0, 2.0, 4.0]))
            for _ in range(cases)]
    started = time.perf_counter()
    for power, attack, defense, roll, crit, stab, eff in args:
        damage_formula(power=power, attack=attack, defense=defense, roll=roll,
                       crit=crit, stab=stab, effectiveness=eff)
    return cases / (time.perf_counter() - started)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):  # pragma: no cover
            pass

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=int, default=16384)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--engine-cases", type=int, default=200_000)
    args = parser.parse_args()

    print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()}"
          + (f" | {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
    dex = load_dex()
    table, _ = type_table(dex)
    print(f"type chart: {table.shape[0]}x{table.shape[0]} matrix\n")

    print("correctness, before any timing:")
    check_against_the_engine("cpu")
    if torch.cuda.is_available():
        check_against_the_engine("cuda")
    print()

    engine = time_engine(args.engine_cases, dex)
    print(f"engine, one hit at a time      {engine:>12,.0f} hits/s")

    for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        for batch in (1024, args.batch):
            rate = time_batched(device, batch, args.steps, table)
            print(f"batched {device:4} batch {batch:<7}       {rate:>12,.0f} hits/s"
                  f"   {rate / engine:>7.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
