"""A vectorised battle step, to find out what a GPU engine could possibly be worth.

This is NOT the engine. It is the arithmetic core with every branch removed:
damage, type effectiveness, STAB, speed order, fainting, auto-replacement.
No abilities, no items, no status, no volatiles, no PP, no switching choices,
no team preview -- the 293 code paths the real engine dispatches are exactly
what has been left out.

That is the point. With zero divergence this is the ceiling a fully vectorised
engine could approach and never reach, so if the ceiling is not far above the
Python engine there is nothing to chase. Every step advances all B battles.

The numbers below are throughput only; the values are synthetic (stats and
powers drawn in realistic ranges) because throughput does not depend on them.
"""
from __future__ import annotations

import time

import torch

LEVEL = 50
TYPES = 18
BROUGHT = 3
MOVES = 4


def make_batch(size: int, device, generator) -> dict:
    """A batch of plausible teams, as tensors."""
    kw = {"device": device, "generator": generator}
    stats = torch.randint(80, 220, (size, 2, BROUGHT, 5), **kw)
    hp = torch.randint(140, 210, (size, 2, BROUGHT), **kw)
    return {
        "hp": hp.clone(),
        "stats": stats,                                    # atk def spa spd spe
        "types": torch.randint(0, TYPES, (size, 2, BROUGHT, 2), **kw),
        "power": torch.randint(40, 120, (size, 2, BROUGHT, MOVES), **kw),
        "move_type": torch.randint(0, TYPES, (size, 2, BROUGHT, MOVES), **kw),
        "move_cat": torch.randint(0, 2, (size, 2, BROUGHT, MOVES), **kw),
        "active": torch.zeros((size, 2), dtype=torch.long, device=device),
        "alive": torch.ones((size, 2, BROUGHT), dtype=torch.bool, device=device),
    }


def gather_active(field: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """field[b, s, active[b, s], ...] -> [B, 2, ...]"""
    index = active.view(*active.shape, *([1] * (field.dim() - 2)))
    index = index.expand(-1, -1, 1, *field.shape[3:])
    return field.gather(2, index).squeeze(2)


def damage(state, attacker_side, choice, chart, generator):
    """One side's active hits the other's. Returns damage [B]."""
    size = state["hp"].shape[0]
    rows = torch.arange(size, device=state["hp"].device)
    defender_side = 1 - attacker_side

    stats = gather_active(state["stats"], state["active"])       # [B,2,5]
    types = gather_active(state["types"], state["active"])       # [B,2,2]
    power = gather_active(state["power"], state["active"])       # [B,2,MOVES]
    move_type = gather_active(state["move_type"], state["active"])
    move_cat = gather_active(state["move_cat"], state["active"])

    picked = choice.unsqueeze(1)
    base = power[rows, attacker_side].gather(1, picked).squeeze(1)
    kind = move_cat[rows, attacker_side].gather(1, picked).squeeze(1)
    element = move_type[rows, attacker_side].gather(1, picked).squeeze(1)

    # Physical reads atk/def, special reads spa/spd. A branch on CPU; a select
    # here, and both halves are computed for every battle -- which is exactly
    # the masking tax a real vectorised engine pays on all 293 paths.
    attack = torch.where(kind == 0, stats[rows, attacker_side, 0],
                         stats[rows, attacker_side, 2])
    defense = torch.where(kind == 0, stats[rows, defender_side, 1],
                          stats[rows, defender_side, 3])

    effectiveness = (chart[element, types[rows, defender_side, 0]]
                     * chart[element, types[rows, defender_side, 1]])
    stab = ((element == types[rows, attacker_side, 0])
            | (element == types[rows, attacker_side, 1]))

    roll = torch.randint(85, 101, (size,), device=base.device, generator=generator)
    crit = torch.randint(0, 24, (size,), device=base.device, generator=generator) == 0

    # The same floor-at-every-step order as moves.damage_formula.
    value = torch.div((2 * LEVEL // 5 + 2) * base * attack, defense, rounding_mode="floor")
    value = torch.div(value, 50, rounding_mode="floor") + 2
    value = torch.where(crit, torch.div(value * 3, 2, rounding_mode="floor"), value)
    value = torch.div(value * roll, 100, rounding_mode="floor")
    value = torch.where(stab, torch.div(value * 3, 2, rounding_mode="floor"), value)
    return (value * effectiveness).long()


def step(state, choices, chart, generator) -> None:
    """Advance every battle one turn, in place."""
    stats = gather_active(state["stats"], state["active"])
    first = stats[:, 0, 4] >= stats[:, 1, 4]          # speed order, ties to side 0

    for order in (0, 1):
        for side in (0, 1):
            moves_now = first if side == 0 else ~first
            moves_now = moves_now if order == 0 else ~moves_now
            hurt = damage(state, side, choices[:, side], chart, generator)
            rows = torch.arange(state["hp"].shape[0], device=hurt.device)
            target = state["active"][:, 1 - side]
            current = state["hp"][rows, 1 - side, target]
            # Only battles where this side moves now, and where both are standing.
            acts = moves_now & (current > 0) & (
                state["hp"][rows, side, state["active"][:, side]] > 0)
            state["hp"][rows, 1 - side, target] = torch.where(
                acts, (current - hurt).clamp(min=0), current)

    state["alive"] = state["hp"] > 0
    # Replace anything that fainted with the lowest-numbered survivor.
    for side in (0, 1):
        rows = torch.arange(state["hp"].shape[0], device=state["hp"].device)
        down = state["hp"][rows, side, state["active"][:, side]] <= 0
        replacement = state["alive"][:, side].float().argmax(dim=1)
        state["active"][:, side] = torch.where(down, replacement,
                                               state["active"][:, side])


def benchmark(size: int, device: str, steps: int = 60) -> float:
    generator = torch.Generator(device=device).manual_seed(7)
    chart = torch.tensor([0.5, 1.0, 2.0, 0.0], device=device)[
        torch.randint(0, 4, (TYPES, TYPES), device=device, generator=generator)]
    state = make_batch(size, device, generator)
    choices = torch.randint(0, MOVES, (size, 2), device=device, generator=generator)

    for _ in range(5):
        step(state, choices, chart, generator)
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(steps):
        step(state, choices, chart, generator)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / steps


if __name__ == "__main__":
    torch.set_num_threads(1)
    print(f"{'batch':>8} {'device':>6} {'per step':>12} {'us / battle-step':>18} "
          f"{'vs python engine':>18}")
    PYTHON_ENGINE_US = 163.0
    for device in ("cpu", "cuda"):
        for size in (64, 256, 1024, 4096, 16384, 65536):
            if device == "cpu" and size > 4096:
                continue
            seconds = benchmark(size, device)
            each = seconds / size * 1e6
            print(f"{size:>8} {device:>6} {seconds*1e3:>10.2f}ms {each:>18.3f} "
                  f"{PYTHON_ENGINE_US/each:>17.0f}x")
