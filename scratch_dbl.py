import random

from pkcm.engine.rng import Rng
from pkcm.data.dex import load_dex
from pkcm.engine.actions import Action
from pkcm.engine.battle import step
from pkcm.engine.legality import random_team
from pkcm.engine.state import BattleConfig, Phase, legal_actions, new_battle

dex = load_dex()
config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"), battle_format="doubles")
print("registered/brought/positions:", config.registered, config.brought, config.active_count)

rng = random.Random(0)
teams = (random_team(dex, config.regulation, Rng.from_seed(11).cursor(), "doubles"),
         random_team(dex, config.regulation, Rng.from_seed(22).cursor(), "doubles"))
state = new_battle(config, teams, seed=1)

for turn in range(300):
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

print("phase:", state.phase.name, "turn:", state.turn, "winner:", state.winner)
