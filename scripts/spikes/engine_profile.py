"""What the 163 us of a real engine step is actually spent on."""
import cProfile, pstats, io
from pkcm.data.dex import load_dex
from pkcm.engine.battle import step
from pkcm.engine.legality import random_team
from pkcm.engine.rng import Rng
from pkcm.engine.state import BattleConfig, new_battle
from pkcm.search.policy import RandomPolicy

dex = load_dex()
config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"), battle_format="singles")

def run():
    for index in range(40):
        teams = tuple(random_team(dex, config.regulation,
                                  Rng.from_seed(700 + index * 2 + o).cursor(), "singles")
                      for o in (1, 2))
        state = new_battle(config, teams, seed=index)
        policies = (RandomPolicy.seeded(index), RandomPolicy.seeded(index + 3))
        for _ in range(40):
            if state.finished:
                break
            state, _ = step(state, policies[0].act(state, 0), policies[1].act(state, 1))

profiler = cProfile.Profile()
profiler.enable(); run(); profiler.disable()

buffer = io.StringIO()
stats = pstats.Stats(profiler, stream=buffer).sort_stats("tottime")
stats.print_stats(18)
for line in buffer.getvalue().splitlines():
    if "pkcm" in line or "ncalls" in line or "function calls" in line:
        print(line[:150])
