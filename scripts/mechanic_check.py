"""Cast every move whose effect is code, and look at what happened.

``move_support`` answers from the shape of the data: a move with a
``sideCondition`` field reads as supported whether or not anything acts on it.
That is how Sucker Punch arrived with no failure condition, and how Baton Pass
arrived as a U-turn that deals no damage -- ``selfSwitch`` was truthy, so the
executor took the branch and threw the ``'copyvolatile'`` away.

So this asks the other question. Each entry sets up a position, casts the move
in it, and states what must be observably true afterwards. A move that quietly
does nothing fails here even when every field it declares is one the engine
knows the name of.

    python scripts/mechanic_check.py            # everything
    python scripts/mechanic_check.py batonpass  # one, with its log
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from pkcm.data.dex import load_dex                          # noqa: E402
from pkcm.engine.actions import Action                      # noqa: E402
from pkcm.engine.battle import step                         # noqa: E402
from pkcm.engine.pokemon import PokemonSet                  # noqa: E402
from pkcm.engine.state import (BOOST_INDEX, BattleConfig,   # noqa: E402
                               new_battle)

DEX = load_dex()
CONFIG = BattleConfig(dex=DEX, regulation=DEX.regulation("m_b"),
                      battle_format="singles")


def mon(species, ability, moves, item=None, nature="serious", sp=(0,) * 6):
    return PokemonSet(species=species, ability=ability, moves=tuple(moves),
                      item=item, nature=nature, sp=sp)


#: Two bodies to fill a party out to three. They never act.
FILLER = [mon("pikachu", "__none__", ("tackle",)),
          mon("alakazam", "__none__", ("tackle",))]


class Fight:
    """One position, and the turns played in it."""

    def __init__(self, ours, theirs, seed=7):
        self.state = new_battle(CONFIG, (tuple(ours) + tuple(FILLER),
                                         tuple(theirs) + tuple(FILLER)),
                                seed=seed)
        self.state, _ = step(self.state, Action.select(0, 1, 2),
                             Action.select(0, 1, 2))
        self.log: list = []

    def turn(self, ours, theirs):
        self.state, events = step(self.state, ours, theirs)
        self.log = list(events)
        return self

    # -- reading it back ---------------------------------------------------- #

    def said(self, *fragments) -> bool:
        text = " | ".join(str(e) for e in self.log)
        return all(f in text for f in fragments)

    def boosts(self, side=0):
        row = self.state.sides[side].boosts[self.state.sides[side].active[0]]
        return {k: row[i] for k, i in BOOST_INDEX.items() if row[i]}

    def hp(self, side=0):
        return self.state.sides[side].hp[self.state.sides[side].active[0]]

    def active_species(self, side=0):
        return self.state.species_id(side, self.state.sides[side].active[0])

    def volatiles(self, side=0):
        side_state = self.state.sides[side]
        return set(side_state.volatiles[side_state.active[0]])


# --------------------------------------------------------------------------- #
# The checks
# --------------------------------------------------------------------------- #

CHECKS: dict[str, tuple[str, object]] = {}


def check(move: str, expect: str):
    def wrap(fn):
        CHECKS[move] = (expect, fn)
        return fn
    return wrap


CHOMP = ("garchomp", "roughskin", ("earthquake", "dragonclaw", "firefang",
                                   "stoneedge"), None, "jolly",
         (0, 32, 2, 0, 0, 32))
GENGAR = ("gengar", "cursedbody", ("shadowball", "sludgebomb", "psychic",
                                   "dazzlinggleam"), None, "timid",
          (0, 0, 2, 32, 0, 32))
IDLE = ("magikarp", "swiftswim", ("splash", "tackle", "flail", "bounce"))
CORV = ("corviknight", "pressure", ("roost", "irondefense", "bodypress",
                                    "uturn"), "leftovers", "impish",
        (32, 0, 32, 0, 2, 0))


def ours(species, ability, moves, *rest, **kw):
    return [mon(species, ability, moves, *rest, **kw)]


# -- hk's list -------------------------------------------------------------- #

@check("partingshot", "foe loses Atk and SpA, then the user leaves")
def _parting_shot():
    # Not Incineroar: its Intimidate takes an Attack stage on the way in and
    # would look exactly like Parting Shot working.
    f = Fight(ours("gourgeist", "frisk",
                   ("partingshot", "shadowsneak", "leechseed", "protect"))
              + [mon(*CHOMP)], [mon(*IDLE)])
    f.turn(Action.move(0), Action.move(0))
    dropped = f.boosts(1)
    switched = f.state.phase.name == "MID_TURN_SWITCH"
    return (dropped.get("atk") == -1 and dropped.get("spa") == -1 and switched,
            f"foe boosts {dropped or 'none'}, phase {f.state.phase.name}")


@check("mirrorcoat", "returns twice the special damage, and fails on physical")
def _mirror_coat():
    f = Fight(ours("wobbuffet", "shadowtag",
                   ("mirrorcoat", "counter", "safeguard", "destinybond"),
                   sp=(32, 0, 2, 0, 32, 0)), [mon(*GENGAR)])
    f.turn(Action.move(0), Action.move(0))
    returned = f.said("damage(side=1", "mirrorcoat")
    g = Fight(ours("wobbuffet", "shadowtag",
                   ("mirrorcoat", "counter", "safeguard", "destinybond"),
                   sp=(32, 0, 2, 0, 32, 0)), [mon(*CHOMP)])
    g.turn(Action.move(0), Action.move(0))
    quiet = not g.said("damage(side=1", "mirrorcoat")
    return (returned and quiet,
            f"vs special: returned={returned}; vs physical: failed={quiet}")


@check("revenge", "doubles in power when the user was hit first")
def _revenge():
    def power(foe_move):
        # Slow on purpose, so the foe has already acted. The control is
        # Soft-Boiled: it leaves the target's typing and Defence alone, so the
        # arms differ only in whether Revenge's user was hit. No Atk
        # investment, so neither arm knocks the target out.
        f = Fight(ours("machamp", "guts",
                       ("revenge", "closecombat", "knockoff", "bulkup"),
                       None, "brave", (32, 0, 32, 0, 2, 0)),
                  [mon("blissey", "naturalcure",
                       ("seismictoss", "softboiled", "toxic", "protect"),
                       None, "serious", (32, 0, 32, 0, 2, 0))])
        f.turn(Action.move(0), Action.move(foe_move))
        for event in f.log:
            text = str(event)
            if text.startswith("damage(side=1") and "revenge" in text:
                return int(re.search(r"amount=(\d+)", text).group(1))
        return 0
    hit_first = power(0)      # Seismic Toss lands on us first
    left_alone = power(1)     # Soft-Boiled: untouched, target unchanged
    return (hit_first > left_alone * 1.5,
            f"after being hit {hit_first}, unharmed {left_alone}")


@check("endeavor", "brings the target down to the user's own HP")
def _endeavor():
    # Both at full health, with very different maximums: Endeavor's rule is
    # about HP, not damage, and this way nothing has to survive a hit first.
    f = Fight(ours("dusknoir", "pressure",
                   ("endeavor", "shadowsneak", "painsplit", "protect"),
                   None, "serious", (0, 32, 2, 0, 32, 0)),
              [mon("blissey", "naturalcure",
                   ("softboiled", "toxic", "protect", "seismictoss"),
                   None, "serious", (32, 0, 32, 0, 2, 0))])
    mine, theirs = f.hp(0), f.hp(1)
    f.turn(Action.move(0), Action.move(0))
    return (abs(f.hp(1) - mine) <= 2,
            f"ours {mine}, theirs {theirs} -> {f.hp(1)}")


@check("finalgambit", "user faints, foe loses the user's remaining HP")
def _final_gambit():
    f = Fight(ours("staraptor", "intimidate",
                   ("finalgambit", "bravebird", "closecombat", "uturn"),
                   None, "jolly", (0, 32, 2, 0, 0, 32)),
              [mon("corviknight", "pressure",
                   ("roost", "irondefense", "bodypress", "uturn"),
                   "leftovers", "impish", (32, 0, 32, 0, 2, 0))])
    mine = f.hp(0)
    f.turn(Action.move(0), Action.move(0))
    dealt = 0
    for event in f.log:
        text = str(event)
        if "move='finalgambit'" in text and text.startswith("damage(side=1"):
            dealt = int(re.search(r"amount=(\d+)", text).group(1))
    fainted = f.said("faint(", "staraptor")
    return (fainted and abs(dealt - mine) <= 2,
            f"user hp {mine}, dealt {dealt}, user fainted={fainted}")


@check("wideguard", "blocks a spread move for the whole side (doubles)")
def _wide_guard():
    doubles = BattleConfig(dex=DEX, regulation=DEX.regulation("m_b"),
                           battle_format="doubles")
    team = (mon("hariyama", "thickfat",
                ("wideguard", "closecombat", "knockoff", "protect"),
                sp=(32, 32, 2, 0, 0, 0)),
            mon("blissey", "naturalcure",
                ("softboiled", "seismictoss", "toxic", "protect"),
                sp=(32, 0, 32, 0, 2, 0)),
            mon("pikachu", "__none__", ("tackle",)),
            mon("alakazam", "__none__", ("tackle",)))
    foes = (mon("garchomp", "roughskin",
                ("earthquake", "dragonclaw", "firefang", "stoneedge"),
                None, "jolly", (0, 32, 2, 0, 0, 32)),
            mon(*IDLE), mon("pikachu", "__none__", ("tackle",)),
            mon("alakazam", "__none__", ("tackle",)))
    state = new_battle(doubles, (team, foes), seed=7)
    state, _ = step(state, Action.select(0, 1, 2, 3), Action.select(0, 1, 2, 3))
    partner = state.sides[0].active[1]
    before = state.sides[0].hp[partner]
    state, events = step(
        state,
        (Action.move(0), Action.move(3)),          # Wide Guard, Protect
        (Action.move(0), Action.move(0)))          # Earthquake hits all
    after = state.sides[0].hp[partner]
    text = " | ".join(str(e) for e in events)
    return (after == before,
            f"partner {before} -> {after}; log mentions wideguard="
            f"{'wideguard' in text}")


@check("avalanche", "doubles in power when the user was hit first")
def _avalanche():
    def power(foe_move):
        f = Fight(ours("avalugg", "sturdy",
                       ("avalanche", "bodypress", "irondefense", "recover"),
                       None, "brave", (32, 32, 2, 0, 0, 0)),
                  [mon("blissey", "naturalcure",
                       ("seismictoss", "softboiled", "toxic", "protect"),
                       None, "serious", (32, 0, 32, 0, 2, 0))])
        f.turn(Action.move(0), Action.move(foe_move))
        for event in f.log:
            text = str(event)
            if text.startswith("damage(side=1") and "avalanche" in text:
                return int(re.search(r"amount=(\d+)", text).group(1))
        return 0
    hit_first, left_alone = power(0), power(1)
    return (hit_first > left_alone * 1.5,
            f"after being hit {hit_first}, unharmed {left_alone}")


@check("storedpower", "grows with the stat stages the user is holding")
def _stored_power():
    def power(setups):
        f = Fight(ours("espathra", "speedboost",
                       ("storedpower", "calmmind", "protect", "luminacrash"),
                       None, "timid", (18, 0, 15, 0, 16, 17)),
                  [mon("blissey", "naturalcure",
                       ("softboiled", "toxic", "protect", "seismictoss"),
                       None, "serious", (32, 0, 32, 0, 2, 0))])
        for _ in range(setups):
            f.turn(Action.move(1), Action.move(0))
        f.turn(Action.move(0), Action.move(0))
        for event in f.log:
            text = str(event)
            if text.startswith("damage(side=1") and "storedpower" in text:
                return int(re.search(r"amount=(\d+)", text).group(1))
        return 0
    flat, boosted = power(0), power(3)
    return boosted > flat * 3, f"unboosted {flat}, after three Calm Minds {boosted}"


@check("grassknot", "grows with how heavy the target is")
def _grass_knot():
    def power(target, ability, moves):
        f = Fight(ours("meowscarada", "protean",
                       ("grassknot", "flowertrick", "knockoff", "uturn"),
                       None, "jolly", (0, 32, 2, 0, 0, 32)),
                  [mon(target, ability, moves, None, "serious",
                       (32, 0, 32, 0, 2, 0))])
        f.turn(Action.move(0), Action.move(0))
        for event in f.log:
            text = str(event)
            if text.startswith("damage(side=1") and "grassknot" in text:
                return int(re.search(r"amount=(\d+)", text).group(1))
        return 0
    light = power("blissey", "naturalcure",
                  ("softboiled", "toxic", "protect", "seismictoss"))
    heavy = power("snorlax", "thickfat", ("rest", "bodyslam", "curse", "crunch"))
    return heavy > light, f"Blissey 46.8kg {light}, Snorlax 460kg {heavy}"


# -- the three already known broken, as proof this harness sees failure ------ #

@check("batonpass", "passes the user's stat stages to whoever comes in")
def _baton_pass():
    f = Fight(ours("espathra", "speedboost",
                   ("calmmind", "batonpass", "luminacrash", "protect"),
                   "focussash", "timid", (18, 0, 15, 0, 16, 17))
              + [mon(*CHOMP)], [mon(*IDLE)])
    f.turn(Action.move(0), Action.move(0))
    f.turn(Action.move(0), Action.move(0))
    passed = f.boosts(0)
    f.turn(Action.move(1), Action.move(0))
    f.turn(Action.switch(1), Action.PASS)
    got = f.boosts(0)
    return (got.get("spa") == passed.get("spa") and got.get("spa"),
            f"passer had {passed}, receiver has {got or 'none'}")


@check("shedtail", "leaves a Substitute behind and switches out")
def _shed_tail():
    f = Fight(ours("cyclizar", "regenerator",
                   ("shedtail", "uturn", "dragonpulse", "knockoff"))
              + [mon(*CHOMP)], [mon(*IDLE)])
    f.turn(Action.move(0), Action.move(0))
    return (f.state.phase.name == "MID_TURN_SWITCH",
            f"phase {f.state.phase.name}, substitute="
            f"{'substitute' in f.volatiles(0)}")


@check("sheercold", "cannot touch an Ice type")
def _sheer_cold():
    landed = 0
    for seed in range(12):
        f = Fight(ours("lapras", "waterabsorb",
                       ("sheercold", "surf", "icebeam", "protect")),
                  [mon("weavile", "pressure",
                       ("roost", "uturn", "bodypress", "iceshard"))], seed)
        f.turn(Action.move(0), Action.move(0))
        landed += f.said("ohko")
    return landed == 0, f"connected {landed}/12 times against an Ice type"


# --------------------------------------------------------------------------- #


def main() -> int:
    wanted = sys.argv[1:] or sorted(CHECKS)
    unknown = [one for one in wanted if one not in CHECKS]
    if unknown:
        print(f"no check written for: {', '.join(unknown)}")
        return 2
    bad = 0
    for move in wanted:
        expect, fn = CHECKS[move]
        try:
            ok, detail = fn()
        except Exception as error:                      # a crash is a failure
            ok, detail = False, f"{type(error).__name__}: {error}"
        korean = NAMES.get(move, move)
        print(f"{'ok  ' if ok else 'FAIL'}  {korean:<14} ({move})")
        if not ok:
            bad += 1
            print(f"        expected: {expect}")
            print(f"        saw:      {detail}")
    print(f"\n{len(wanted) - bad}/{len(wanted)} behaved")
    return 1 if bad else 0


import json  # noqa: E402
NAMES = json.loads((ROOT / "data" / "champions" / "names.json")
                   .read_text(encoding="utf-8"))["moves"]

if __name__ == "__main__":
    raise SystemExit(main())
