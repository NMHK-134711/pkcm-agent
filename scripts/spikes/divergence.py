"""Does divergence saturate? That is the whole argument for a GPU engine.

A batch needs every code path any of its battles wants. If the number of
distinct paths keeps growing with the batch, vectorising buys nothing. If it
saturates -- there are only so many abilities in the game -- then a big enough
batch amortises the masking and the GPU wins on physics.

One pass over 256 battles, recording the paths each takes at each step, then
the union recomputed for every batch size from the same recording.
"""
import random, statistics, time
from pkcm.data.dex import load_dex
from pkcm.engine import effects
from pkcm.engine.battle import step
from pkcm.engine.legality import random_team
from pkcm.engine.rng import Rng
from pkcm.engine.state import BattleConfig, new_battle
from pkcm.search.policy import RandomPolicy

TOTAL, STEPS = 256, 40

fired: set = set()
original = effects._ordered
def watched(held, event, ref):
    out = original(held, event, ref)
    for _, effect, key in out:
        fired.add((effect.id, key))
    return out
effects._ordered = watched

dex = load_dex()
config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"), battle_format="singles")

battles, policies = [], []
for index in range(TOTAL):
    teams = tuple(random_team(dex, config.regulation,
                              Rng.from_seed(5000 + index * 2 + o).cursor(), "singles")
                  for o in (1, 2))
    battles.append(new_battle(config, teams, seed=index))
    policies.append((RandomPolicy.seeded(index), RandomPolicy.seeded(index + 999)))

# paths[step][battle] -> frozenset of code paths that battle took
paths: list[list[frozenset]] = []
engine_seconds = 0.0
engine_steps = 0
for tick in range(STEPS):
    row = []
    for index, state in enumerate(battles):
        if state.finished:
            row.append(frozenset())
            continue
        actions = tuple(policies[index][p].act(state, p) for p in (0, 1))
        fired.clear()
        start = time.perf_counter()
        battles[index], _ = step(state, actions[0], actions[1])
        engine_seconds += time.perf_counter() - start
        engine_steps += 1
        row.append(frozenset(fired))
    paths.append(row)

print(f"one engine step costs {1e6*engine_seconds/engine_steps:.0f} us on this CPU "
      f"({engine_steps} steps timed, machine busy with training)\n")

everything = set().union(*(p for row in paths for p in row))
print(f"distinct code paths seen anywhere: {len(everything)}\n")

print(f"{'batch':>7} {'paths/batch':>12} {'lane use':>10} {'divergence ceiling':>20}")
random.seed(0)
for size in (1, 8, 32, 64, 128, 256):
    unions, singles = [], []
    for row in paths:
        live = [p for p in row if p]
        if len(live) < size:
            continue
        picked = random.sample(live, size)
        unions.append(len(set().union(*picked)))
        singles.append(statistics.mean(len(p) for p in picked))
    if not unions:
        continue
    one, many = statistics.mean(singles), statistics.mean(unions)
    print(f"{size:>7} {many:>12.1f} {100*one/many:>9.1f}% {size*one/many:>19.1f}x")
