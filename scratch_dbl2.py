import collections, random, traceback
from pkcm.data.dex import load_dex
from pkcm.engine.battle import step
from pkcm.engine.legality import random_team
from pkcm.engine.rng import Rng
from pkcm.engine.state import BattleConfig, Phase, legal_actions, new_battle

dex = load_dex()
config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"), battle_format="doubles")

kinds = collections.Counter()
endings = collections.Counter()
spread_seen = 0
failures = []

for seed in range(120):
    rng = random.Random(seed)
    teams = (random_team(dex, config.regulation, Rng.from_seed(seed * 2 + 1).cursor(), "doubles"),
             random_team(dex, config.regulation, Rng.from_seed(seed * 2 + 2).cursor(), "doubles"))
    state = new_battle(config, teams, seed=seed)
    try:
        for _ in range(600):
            if state.phase is Phase.FINISHED:
                break
            choices = []
            for player in (0, 1):
                positions = 1 if state.phase is Phase.TEAM_PREVIEW else config.active_count
                picks, taken = [], set()
                for position in range(positions):
                    options = [a for a in legal_actions(state, player, position)
                               if not (a.kind.name == "SWITCH" and a.index in taken)]
                    pick = rng.choice(options)
                    if pick.kind.name == "SWITCH":
                        taken.add(pick.index)
                    picks.append(pick)
                choices.append(tuple(picks))
            state, log = step(state, choices[0], choices[1])
            for e in log:
                kinds[e.kind] += 1
        endings[state.phase.name] += 1
        if state.phase is Phase.FINISHED:
            endings["winner=%s" % state.winner] += 1
    except Exception:
        failures.append((seed, traceback.format_exc().strip().splitlines()[-1]))

print("battles:", 120, "failures:", len(failures))
for seed, why in failures[:6]:
    print("  seed", seed, why)
print("endings:", dict(endings))
print("unimplemented events:", kinds.get("unimplemented", 0))
print("move_failed:", kinds.get("move_failed", 0), " immune:", kinds.get("immune", 0))
