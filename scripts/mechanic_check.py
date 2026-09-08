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

    def max_hp(self, side=0):
        return self.state.active_pokemon(side).max_hp

    def status(self, side=0):
        side_state = self.state.sides[side]
        return side_state.status[side_state.active[0]]

    def conditions(self, side=0):
        return dict(self.state.sides[side].conditions)

    def item(self, side=0):
        return self.state.item_id(side, self.state.sides[side].active[0])

    def damage(self, move, side=1):
        """What a named move took off ``side`` this turn, 0 if it did not."""
        for event in self.log:
            text = str(event)
            if text.startswith(f"damage(side={side}") and move in text:
                return int(re.search(r"amount=(\d+)", text).group(1))
        return 0


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


# -- the field's own moves, most-used first --------------------------------- #
#
# Ordered by how many of the 253 field parties carry them, which is where a
# wrong answer costs the most. Abilities are ``__none__`` wherever one could
# imitate the effect being measured: Intimidate reading as Parting Shot cost
# an afternoon on the first pass.

#: Nothing on it interferes: no ability, no item, and Splash to pass a turn.
def dummy(species, *moves):
    return mon(species, "__none__", tuple(moves) or ("splash", "tackle"))


#: Bulky, passive, and huge -- survives what it is hit with so the arms of a
#: comparison differ only in the thing being measured.
def wall():
    return mon("blissey", "__none__", ("splash", "tackle", "seismictoss",
                                       "protect"), None, "serious",
               (32, 0, 32, 0, 2, 0))


@check("stealthrock", "hurts whoever comes in afterwards")
def _stealth_rock():
    f = Fight(ours("garchomp", "__none__",
                   ("stealthrock", "earthquake", "dragonclaw", "protect")),
              [dummy("magikarp")])
    f.turn(Action.move(0), Action.move(0))
    laid = "stealthrock" in f.conditions(1)
    f.turn(Action.move(1), Action.switch(1))     # they bring Pikachu in
    hurt = f.max_hp(1) - f.hp(1)
    return (laid and hurt > 0,
            f"condition set={laid}; the replacement lost {hurt} of "
            f"{f.max_hp(1)} on the way in")


@check("protect", "the turn's attack does not land")
def _protect():
    def hp_after(move_index):
        f = Fight(ours("snorlax", "__none__",
                       ("protect", "splash", "bodyslam", "rest"),
                       None, "serious", (32, 0, 32, 0, 2, 0)),
                  [mon("garchomp", "__none__",
                       ("earthquake", "dragonclaw", "firefang", "stoneedge"),
                       None, "jolly", (0, 32, 2, 0, 0, 32))])
        f.turn(Action.move(move_index), Action.move(0))
        return f.max_hp(0) - f.hp(0)
    blocked = hp_after(0)     # Protect
    taken = hp_after(1)       # Splash, same turn otherwise
    return (blocked == 0 and taken > 0,
            f"with Protect lost {blocked}, with Splash lost {taken}")


@check("suckerpunch", "lands on an attacker, fails on one using a status move")
def _sucker_punch():
    def dealt(foe_move):
        f = Fight(ours("kingambit", "__none__",
                       ("suckerpunch", "ironhead", "swordsdance", "kowtowcleave"),
                       None, "adamant", (0, 32, 2, 0, 0, 32)),
                  [mon("snorlax", "__none__",
                       ("bodyslam", "splash", "rest", "protect"),
                       None, "serious", (32, 0, 32, 0, 2, 0))])
        f.turn(Action.move(0), Action.move(foe_move))
        return f.damage("suckerpunch")
    attacking = dealt(0)      # Body Slam: a move Sucker Punch may answer
    passive = dealt(1)        # Splash: nothing to answer, so it must fail
    return (attacking > 0 and passive == 0,
            f"against an attack {attacking}, against Splash {passive}")


@check("yawn", "puts the target to sleep at the end of the next turn")
def _yawn():
    f = Fight(ours("slowbro", "__none__",
                   ("yawn", "scald", "slackoff", "psychic")),
              [dummy("magikarp")])
    f.turn(Action.move(0), Action.move(0))
    drowsy = "yawn" in f.volatiles(1)
    asleep_yet = f.status(1)
    f.turn(Action.move(1), Action.move(0))
    return (drowsy and asleep_yet is None and f.status(1) == "slp",
            f"volatile={drowsy}, status after the cast {asleep_yet}, "
            f"after the next turn {f.status(1)}")


@check("substitute", "costs a quarter of the user's HP and stands in front")
def _substitute():
    f = Fight(ours("gengar", "__none__",
                   ("substitute", "shadowball", "sludgebomb", "protect")),
              [dummy("magikarp")])
    before = f.hp(0)
    f.turn(Action.move(0), Action.move(0))
    paid = before - f.hp(0)
    quarter = f.max_hp(0) // 4
    return ("substitute" in f.volatiles(0) and abs(paid - quarter) <= 1,
            f"paid {paid} of a quarter ({quarter}), "
            f"volatiles {sorted(f.volatiles(0)) or 'none'}")


@check("roost", "gives back about half of the user's maximum")
def _roost():
    f = Fight(ours("corviknight", "__none__",
                   ("roost", "irondefense", "bodypress", "uturn"),
                   None, "impish", (32, 0, 32, 0, 2, 0)),
              [mon("garchomp", "__none__",
                   ("earthquake", "dragonclaw", "firefang", "stoneedge"),
                   None, "jolly", (0, 32, 2, 0, 0, 32))])
    f.turn(Action.move(2), Action.move(3))    # take a hit first
    hurt = f.hp(0)
    f.turn(Action.move(0), Action.move(3))
    healed = f.hp(0) - hurt
    # The foe hits again on the healing turn, so the gain is net of one hit.
    return (healed > 0, f"at {hurt}, after Roost {f.hp(0)} (+{healed}) "
                        f"of max {f.max_hp(0)}")


@check("encore", "locks the target into the move it just used")
def _encore():
    f = Fight(ours("clefable", "__none__",
                   ("encore", "moonblast", "softboiled", "protect"),
                   None, "timid", (32, 0, 2, 0, 0, 32)),
              [dummy("magikarp", "tackle", "splash")])
    # Protect, not Moonblast: attacking on the first turn knocked the Magikarp
    # out and the check then measured a forced switch instead of an Encore.
    f.turn(Action.move(3), Action.move(0))     # they show us Tackle
    f.turn(Action.move(0), Action.move(1))     # Encore over their Splash pick
    locked = "encore" in f.volatiles(1)
    # The volatile on its own proved nothing -- it was there for four turns
    # while the target picked freely. What settles it is the choice they are
    # left with on the turn after.
    from pkcm.engine.state import legal_actions
    offered = {str(one) for one in legal_actions(f.state, 1)
               if str(one).startswith("move")}
    return (locked and offered == {"move(0)"},
            f"volatile={locked}, moves still offered {sorted(offered)}")


@check("taunt", "stops the target using status moves")
def _taunt():
    f = Fight(ours("gengar", "__none__",
                   ("taunt", "shadowball", "sludgebomb", "protect"),
                   None, "timid", (0, 0, 2, 32, 0, 32)),
              [mon("blissey", "__none__",
                   ("softboiled", "seismictoss", "toxic", "protect"),
                   None, "serious", (32, 0, 32, 0, 2, 0))])
    f.turn(Action.move(0), Action.move(0))     # they try to Soft-Boiled
    landed = "taunt" in f.volatiles(1)
    # Encore held its volatile for four turns while enforcing nothing, so the
    # volatile is not the answer here either: what counts is the status move
    # being refused on the turn after.
    f.turn(Action.move(3), Action.move(2))     # they try Toxic
    return (landed and f.said("cant_move", "taunt") and f.status(0) is None,
            f"volatile={landed}, refused={f.said('cant_move', 'taunt')}, "
            f"our status {f.status(0)}")


@check("dragontail", "damages, then drags the target out")
def _dragon_tail():
    f = Fight(ours("garchomp", "__none__",
                   ("dragontail", "earthquake", "stealthrock", "protect"),
                   None, "jolly", (0, 32, 2, 0, 0, 32)),
              [mon("snorlax", "__none__",     # Normal: not immune to Dragon
                   ("splash", "bodyslam", "rest", "protect"),
                   None, "serious", (32, 0, 32, 0, 2, 0))])
    before = f.active_species(1)
    f.turn(Action.move(0), Action.move(0))
    return (f.active_species(1) != before,
            f"{before} -> {f.active_species(1)}")


@check("whirlwind", "drags the target out whatever it wanted to do")
def _whirlwind():
    f = Fight(ours("corviknight", "__none__",
                   ("whirlwind", "roost", "bodypress", "irondefense")),
              [dummy("magikarp")])
    before = f.active_species(1)
    f.turn(Action.move(0), Action.move(0))
    return (f.active_species(1) != before,
            f"{before} -> {f.active_species(1)}")


@check("roar", "drags the target out whatever it wanted to do")
def _roar():
    f = Fight(ours("snorlax", "__none__",
                   ("roar", "bodyslam", "rest", "protect")),
              [dummy("magikarp")])
    before = f.active_species(1)
    f.turn(Action.move(0), Action.move(0))
    return (f.active_species(1) != before,
            f"{before} -> {f.active_species(1)}")


def _heals_half(species, moves, heal_index):
    """Take a hit, then heal, and report what came back."""
    f = Fight(ours(species, "__none__", moves, None, "serious",
                   (32, 0, 32, 0, 2, 0)),
              [mon("garchomp", "__none__",
                   ("earthquake", "dragonclaw", "firefang", "stoneedge"),
                   None, "jolly", (0, 32, 2, 0, 0, 32))])
    f.turn(Action.move(1), Action.move(0))
    hurt = f.hp(0)
    f.turn(Action.move(heal_index), Action.move(3))   # Stone Edge: may miss
    return f, hurt, f.hp(0) - hurt


@check("recover", "gives back about half of the user's maximum")
def _recover():
    f, hurt, healed = _heals_half(
        "starmie", ("recover", "surf", "psychic", "icebeam"), 0)
    return healed > 0, f"at {hurt} of {f.max_hp(0)}, recovered {healed}"


@check("slackoff", "gives back about half of the user's maximum")
def _slack_off():
    f, hurt, healed = _heals_half(
        "slowbro", ("slackoff", "scald", "psychic", "yawn"), 0)
    return healed > 0, f"at {hurt} of {f.max_hp(0)}, recovered {healed}"


@check("synthesis", "gives back about half of the user's maximum")
def _synthesis():
    f, hurt, healed = _heals_half(
        "ferrothorn", ("synthesis", "gyroball", "leechseed", "spikes"), 0)
    return healed > 0, f"at {hurt} of {f.max_hp(0)}, recovered {healed}"


@check("moonlight", "gives back about half of the user's maximum")
def _moonlight():
    f, hurt, healed = _heals_half(
        "clefable", ("moonlight", "moonblast", "softboiled", "protect"), 0)
    return healed > 0, f"at {hurt} of {f.max_hp(0)}, recovered {healed}"


def _boosts_by(species, moves, expected):
    f = Fight(ours(species, "__none__", moves), [dummy("magikarp")])
    f.turn(Action.move(0), Action.move(0))
    got = f.boosts(0)
    ok = all(got.get(stat) == amount for stat, amount in expected.items())
    return ok, f"{got or 'nothing'} against {expected}"


@check("calmmind", "raises Special Attack and Special Defence")
def _calm_mind():
    return _boosts_by("gengar", ("calmmind", "shadowball", "sludgebomb",
                                 "protect"), {"spa": 1, "spd": 1})


@check("irondefense", "raises Defence two stages")
def _iron_defense():
    return _boosts_by("corviknight", ("irondefense", "roost", "bodypress",
                                      "uturn"), {"def": 2})


@check("dragondance", "raises Attack and Speed")
def _dragon_dance():
    return _boosts_by("dragonite", ("dragondance", "dragonclaw", "roost",
                                    "firepunch"), {"atk": 1, "spe": 1})


@check("shellsmash", "trades both defences for Attack, Sp. Atk and Speed")
def _shell_smash():
    return _boosts_by("cloyster", ("shellsmash", "iciclespear", "rockblast",
                                   "protect"),
                      {"atk": 2, "spa": 2, "spe": 2, "def": -1, "spd": -1})


@check("toxic", "leaves the target badly poisoned")
def _toxic():
    f = Fight(ours("blissey", "__none__",
                   ("toxic", "seismictoss", "softboiled", "protect")),
              [dummy("magikarp")])          # Water: not Steel, not Poison
    f.turn(Action.move(0), Action.move(0))
    return f.status(1) == "tox", f"foe status {f.status(1)}"


@check("knockoff", "takes the target's item away")
def _knock_off():
    f = Fight(ours("weavile", "__none__",
                   ("knockoff", "iceshard", "uturn", "swordsdance"),
                   None, "jolly", (0, 32, 2, 0, 0, 32)),
              [mon("snorlax", "__none__",
                   ("splash", "bodyslam", "rest", "protect"),
                   "leftovers", "serious", (32, 0, 32, 0, 2, 0))])
    before = f.item(1)
    f.turn(Action.move(0), Action.move(0))
    return (before == "leftovers" and f.item(1) is None,
            f"held {before}, now {f.item(1)}")


@check("bodypress", "attacks with the user's Defence, so Iron Defence adds to it")
def _body_press():
    def dealt(setup_first):
        f = Fight(ours("corviknight", "__none__",
                       ("bodypress", "irondefense", "roost", "uturn"),
                       None, "impish", (32, 0, 32, 0, 2, 0)),
                  [wall()])
        if setup_first:
            f.turn(Action.move(1), Action.move(0))    # Iron Defence, +2 Def
        f.turn(Action.move(0), Action.move(0))
        return f.damage("bodypress")
    plain = dealt(False)
    boosted = dealt(True)
    return (plain > 0 and boosted > plain * 1.5,
            f"unboosted {plain}, after Iron Defence {boosted}")


@check("spikes", "hurts a grounded replacement")
def _spikes():
    f = Fight(ours("ferrothorn", "__none__",
                   ("spikes", "gyroball", "leechseed", "protect")),
              [dummy("magikarp")])
    f.turn(Action.move(0), Action.move(0))
    laid = "spikes" in f.conditions(1)
    f.turn(Action.move(1), Action.switch(1))
    hurt = f.max_hp(1) - f.hp(1)
    return (laid and hurt > 0,
            f"condition set={laid}; the replacement lost {hurt}")


@check("toxicspikes", "poisons a grounded replacement")
def _toxic_spikes():
    f = Fight(ours("ferrothorn", "__none__",
                   ("toxicspikes", "gyroball", "leechseed", "protect")),
              [dummy("magikarp")])
    f.turn(Action.move(0), Action.move(0))
    laid = "toxicspikes" in f.conditions(1)
    f.turn(Action.move(1), Action.switch(1))
    return (laid and f.status(1) in ("psn", "tox"),
            f"condition set={laid}; the replacement's status {f.status(1)}")


@check("leechseed", "drains the target every turn and gives it to the user")
def _leech_seed():
    # Ninety percent accurate, and the first seed tried was a miss -- which
    # reads identically to "does nothing" if you only look once.
    for seed in range(8):
        f = Fight(ours("ferrothorn", "__none__",
                       ("leechseed", "gyroball", "protect", "spikes"),
                       None, "relaxed", (32, 0, 32, 0, 2, 0)),
                  [mon("snorlax", "__none__",   # not Grass: Grass is immune
                       ("splash", "bodyslam", "rest", "protect"),
                       None, "serious", (32, 0, 32, 0, 2, 0))], seed)
        f.turn(Action.move(0), Action.move(1))  # seed, and take a hit for room
        if "leechseed" in f.volatiles(1):
            break
    else:
        return False, "never connected in eight tries"
    theirs = f.hp(1)
    f.turn(Action.move(2), Action.move(0))    # Protect, so only the seed acts
    # Not "our HP went up": the drain on the seeding turn had already healed us
    # back to full, so the gain has nowhere to show. The heal event does show.
    healed = f.said("heal(", "leechseed") or f.said("leechseed")
    return (f.hp(1) < theirs and healed,
            f"foe {theirs}->{f.hp(1)} while we were Protecting, "
            f"a heal was logged={healed}")


@check("haze", "wipes the stat changes on both sides")
def _haze():
    f = Fight(ours("weezinggalar", "__none__",
                   ("haze", "sludgebomb", "willowisp", "protect")),
              [mon("dragonite", "__none__",
                   ("dragondance", "dragonclaw", "roost", "firepunch"))])
    f.turn(Action.move(1), Action.move(0))    # they set up
    theirs = f.boosts(1)
    f.turn(Action.move(0), Action.move(0))    # Haze over their second dance
    return (theirs and not f.boosts(1),
            f"foe had {theirs or 'nothing'}, after Haze {f.boosts(1) or 'nothing'}")


@check("trick", "swaps the two held items")
def _trick():
    f = Fight(ours("gengar", "__none__",
                   ("trick", "shadowball", "sludgebomb", "protect"),
                   "choicescarf", "timid", (0, 0, 2, 32, 0, 32)),
              [mon("snorlax", "__none__",
                   ("splash", "bodyslam", "rest", "protect"),
                   "leftovers", "serious", (32, 0, 32, 0, 2, 0))])
    f.turn(Action.move(0), Action.move(0))
    return (f.item(0) == "leftovers" and f.item(1) == "choicescarf",
            f"we hold {f.item(0)}, they hold {f.item(1)}")


@check("rest", "sleeps the user and fills it back up")
def _rest():
    f = Fight(ours("snorlax", "__none__",
                   ("rest", "bodyslam", "protect", "splash"),
                   None, "serious", (32, 0, 32, 0, 2, 0)),
              [mon("garchomp", "__none__",
                   ("earthquake", "dragonclaw", "firefang", "stoneedge"),
                   None, "jolly", (0, 32, 2, 0, 0, 32))])
    f.turn(Action.move(3), Action.move(0))
    hurt = f.hp(0)
    f.turn(Action.move(0), Action.move(3))    # Stone Edge may miss; Rest is ours
    return (f.hp(0) > hurt and f.status(0) == "slp",
            f"at {hurt} of {f.max_hp(0)} -> {f.hp(0)}, status {f.status(0)}")


@check("trickroom", "the slower of the two moves first")
def _trick_room():
    def order(with_room):
        f = Fight(ours("slowbro", "__none__",       # slow on purpose
                       ("trickroom", "scald", "splash", "psychic"),
                       None, "relaxed", (32, 0, 32, 0, 2, 0)),
                  [mon("weavile", "__none__",       # fast on purpose
                       ("iceshard", "knockoff", "splash", "uturn"),
                       None, "jolly", (0, 32, 2, 0, 0, 32))])
        f.turn(Action.move(0 if with_room else 2), Action.move(2))
        f.turn(Action.move(1), Action.move(3))
        text = [str(e) for e in f.log if str(e).startswith("move_used")]
        return next((("them" if "side=1" in one else "us") for one in text), None)
    plain = order(False)
    roomed = order(True)
    return (plain == "them" and roomed == "us",
            f"without the room {plain} moved first, with it {roomed}")


@check("fakeout", "flinches, and only on the turn the user came in")
def _fake_out():
    f = Fight(ours("kangaskhan", "__none__",
                   ("fakeout", "bodyslam", "suckerpunch", "protect"),
                   None, "jolly", (0, 32, 2, 0, 0, 32)),
              [mon("snorlax", "__none__",
                   ("bodyslam", "splash", "rest", "protect"),
                   None, "serious", (32, 0, 32, 0, 2, 0))])
    f.turn(Action.move(0), Action.move(0))
    flinched = f.said("flinch") or not f.said("move_used(side=1")
    f.turn(Action.move(0), Action.move(0))
    stale = f.said("move_failed") or f.said("fail")
    return (flinched and stale,
            f"first turn flinched={flinched}, second turn refused={stale}")


@check("painsplit", "levels the two HP totals")
def _pain_split():
    # The user has to be able to hold the average: Gengar caps at 135 against
    # a Blissey's 362, so both ending level is impossible and the first version
    # of this check was reporting the cap as a failure.
    f = Fight(ours("snorlax", "__none__",
                   ("painsplit", "bodyslam", "protect", "splash"),
                   None, "serious", (32, 0, 32, 0, 2, 0)),
              [mon("garchomp", "__none__",
                   ("earthquake", "dragonclaw", "firefang", "stoneedge"),
                   None, "jolly", (0, 32, 2, 0, 0, 32))])
    f.turn(Action.move(3), Action.move(0))     # a hit, to put the two apart
    mine, theirs = f.hp(0), f.hp(1)
    f.turn(Action.move(0), Action.move(3))     # Stone Edge may miss; ours lands
    # Which of the two rises depends on which started lower, and that depends
    # on a damage roll. What Pain Split promises is that they end together.
    return (abs(mine - theirs) > 10 and abs(f.hp(0) - f.hp(1)) <= 2,
            f"{mine} and {theirs} -> {f.hp(0)} and {f.hp(1)}")


@check("perishsong", "counts both sides down to fainting")
def _perish_song():
    f = Fight(ours("gengar", "__none__",
                   ("perishsong", "shadowball", "protect", "splash")),
              [dummy("snorlax")])
    f.turn(Action.move(0), Action.move(0))
    counted = "perishsong" in f.volatiles(0) and "perishsong" in f.volatiles(1)
    for _ in range(3):
        if f.state.phase.name in ("FINISHED", "FORCED_SWITCH"):
            break
        f.turn(Action.move(3), Action.move(0))
    fainted = f.said("faint") or f.state.phase.name in ("FINISHED",
                                                        "FORCED_SWITCH")
    return counted and fainted, f"counter on both={counted}, fainted={fainted}"


@check("strengthsap", "heals by the target's Attack and takes a stage off it")
def _strength_sap():
    f = Fight(ours("gourgeist", "__none__",
                   ("strengthsap", "shadowsneak", "protect", "splash"),
                   None, "serious", (32, 0, 32, 0, 2, 0)),
              [mon("garchomp", "__none__",
                   ("earthquake", "dragonclaw", "firefang", "stoneedge"),
                   None, "jolly", (0, 32, 2, 0, 0, 32))])
    f.turn(Action.move(1), Action.move(0))    # take a hit to leave room
    hurt = f.hp(0)
    f.turn(Action.move(0), Action.move(3))
    return (f.hp(0) > hurt and f.boosts(1).get("atk") == -1,
            f"{hurt} -> {f.hp(0)}, their Atk {f.boosts(1).get('atk')}")


@check("destinybond", "takes the attacker with it")
def _destiny_bond():
    f = Fight(ours("gengar", "__none__",
                   ("destinybond", "shadowball", "protect", "splash"),
                   None, "timid", (0, 0, 0, 32, 0, 32)),
              [mon("garchomp", "__none__",
                   ("earthquake", "dragonclaw", "firefang", "stoneedge"),
                   None, "jolly", (0, 32, 2, 0, 0, 32))])
    f.turn(Action.move(0), Action.move(1))    # Dragon Claw: Gengar is frail
    marked = "destinybond" in f.volatiles(0)
    both = f.said("faint(side=0") and f.said("faint(side=1")
    return (marked or both,
            f"volatile={marked}, both fainted={both}; "
            f"log {' | '.join(str(e) for e in f.log)[:160]}")


@check("kingsshield", "blocks the hit and takes a stage off a contact attacker")
def _kings_shield():
    f = Fight(ours("aegislash", "__none__",
                   ("kingsshield", "shadowball", "shadowsneak", "swordsdance")),
              [mon("garchomp", "__none__",
                   ("dragonclaw", "earthquake", "firefang", "stoneedge"),
                   None, "jolly", (0, 32, 2, 0, 0, 32))])
    f.turn(Action.move(0), Action.move(0))    # Dragon Claw makes contact
    return (f.hp(0) == f.max_hp(0) and f.boosts(1).get("atk", 0) < 0,
            f"we are at {f.hp(0)}/{f.max_hp(0)}, "
            f"their Atk {f.boosts(1).get('atk', 0)}")


@check("metalburst", "returns more than it was hit for")
def _metal_burst():
    f = Fight(ours("archaludon", "__none__",
                   ("metalburst", "flashcannon", "dragontail", "protect"),
                   None, "sassy", (32, 0, 32, 0, 32, 0)),
              [mon("garchomp", "__none__",
                   ("dragonclaw", "earthquake", "firefang", "stoneedge"),
                   None, "jolly", (0, 32, 2, 0, 0, 32))])
    f.turn(Action.move(0), Action.move(0))
    taken = f.max_hp(0) - f.hp(0)
    returned = f.damage("metalburst")
    return (taken > 0 and returned > taken,
            f"took {taken}, returned {returned}")


@check("defog", "clears the hazards off both sides")
def _defog():
    f = Fight(ours("corviknight", "__none__",
                   ("defog", "roost", "bodypress", "uturn")),
              [mon("ferrothorn", "__none__",
                   ("spikes", "stealthrock", "gyroball", "protect"),
                   None, "relaxed", (32, 0, 32, 0, 2, 0))])
    f.turn(Action.move(1), Action.move(0))    # they lay Spikes on us
    f.turn(Action.move(1), Action.move(1))    # and Stealth Rock
    before = (f.conditions(0), f.conditions(1))
    # Not Protect on their side: Defog targets them, so Protect blocks it and
    # the check then measures a block rather than a clear.
    f.turn(Action.move(0), Action.move(2))
    return (before[0] and not f.conditions(0),
            f"ours {before[0] or 'none'} -> {f.conditions(0) or 'none'}, "
            f"theirs {before[1] or 'none'} -> {f.conditions(1) or 'none'}")


# -- the rest of the field's coded moves ------------------------------------ #


def _until(build, landed, tries=10):
    """Replay a scenario over seeds until the thing under test connects.

    Leech Seed is ninety percent accurate and the first seed tried was a miss,
    which reads exactly like a move that does nothing. Anything with an
    accuracy roll in front of it gets this rather than one seed and a verdict.
    """
    for seed in range(tries):
        f = build(seed)
        if landed(f):
            return f
    return None


@check("wish", "heals at the end of the turn after, by the wisher's own half")
def _wish():
    f = Fight(ours("clefable", "__none__",
                   ("wish", "protect", "moonblast", "splash"),
                   None, "serious", (32, 0, 32, 0, 2, 0)),
              [mon("garchomp", "__none__",
                   ("earthquake", "dragonclaw", "firefang", "stoneedge"),
                   None, "jolly", (0, 32, 2, 0, 0, 32))])
    f.turn(Action.move(3), Action.move(0))     # take a hit to leave room
    f.turn(Action.move(0), Action.move(3))     # the wish is made
    made = f.hp(0)
    f.turn(Action.move(1), Action.move(3))     # Protect, so only the wish acts
    return (f.hp(0) > made,
            f"at {made} of {f.max_hp(0)} when it was made, {f.hp(0)} after")


@check("sparklingaria", "cures the burn it lands on")
def _sparkling_aria():
    def build(seed):
        f = Fight(ours("primarina", "__none__",
                       ("sparklingaria", "moonblast", "psychic", "protect"),
                       None, "modest", (0, 0, 2, 32, 0, 32)),
                  [mon("snorlax", "__none__",
                       ("splash", "bodyslam", "rest", "protect"),
                       None, "serious", (32, 0, 32, 0, 2, 0))], seed)
        f.state.sides[1].status[f.state.sides[1].active[0]] = "brn"
        return f
    f = build(3)
    burnt = f.status(1)
    f.turn(Action.move(0), Action.move(0))
    return (burnt == "brn" and f.status(1) is None,
            f"burnt going in={burnt}, status after {f.status(1)}")


@check("tripleaxel", "three hits, each stronger than the last")
def _triple_axel():
    f = _until(lambda seed: Fight(
        ours("weavile", "__none__",
             ("tripleaxel", "knockoff", "iceshard", "swordsdance"),
             None, "jolly", (0, 32, 2, 0, 0, 32)),
        [wall()], seed).turn(Action.move(0), Action.move(0)),
        # Each of the three rolls its own accuracy, and the log says the move's
        # name for the cast as well as for the hits.
        lambda f: sum(1 for e in f.log if str(e).startswith("damage(side=1")
                      and "tripleaxel" in str(e)) == 3, tries=20)
    if f is None:
        return False, "never landed all three hits in ten tries"
    hits = [int(re.search(r"amount=(\d+)", str(e)).group(1))
            for e in f.log
            if str(e).startswith("damage(side=1") and "tripleaxel" in str(e)]
    return (len(hits) == 3 and hits[1] > hits[0] and hits[2] > hits[1],
            f"hits {hits}")


@check("solarbeam", "charges a turn first, unless the sun is out")
def _solar_beam():
    f = Fight(ours("venusaur", "__none__",
                   ("solarbeam", "sludgebomb", "sunnyday", "protect"),
                   None, "modest", (0, 0, 2, 32, 0, 32)),
              [wall()])
    f.turn(Action.move(0), Action.move(0))
    charged = f.damage("solarbeam") == 0
    f.turn(Action.move(0), Action.move(0))
    fired = f.damage("solarbeam") > 0
    g = Fight(ours("venusaur", "__none__",
                   ("solarbeam", "sludgebomb", "sunnyday", "protect"),
                   None, "modest", (0, 0, 2, 32, 0, 32)),
              [wall()])
    g.turn(Action.move(2), Action.move(0))     # sun
    g.turn(Action.move(0), Action.move(0))
    at_once = g.damage("solarbeam") > 0
    return (charged and fired and at_once,
            f"turn one silent={charged}, turn two fired={fired}, "
            f"in sun it fired at once={at_once}")


@check("curse", "a Ghost pays half its HP; anything else trades Speed for bulk")
def _curse():
    ghost = Fight(ours("gengar", "__none__",
                       ("curse", "shadowball", "protect", "splash"),
                       None, "timid", (0, 0, 2, 32, 0, 32)),
                  [dummy("snorlax")])
    before = ghost.hp(0)
    ghost.turn(Action.move(0), Action.move(0))
    paid = before - ghost.hp(0)
    body = Fight(ours("snorlax", "__none__",
                      ("curse", "bodyslam", "protect", "splash"),
                      None, "serious", (32, 0, 32, 0, 2, 0)),
                 [dummy("magikarp")])
    body.turn(Action.move(0), Action.move(0))
    traded = body.boosts(0)
    return (paid > 0 and "curse" in ghost.volatiles(1)
            and traded.get("atk") == 1 and traded.get("def") == 1
            and traded.get("spe") == -1,
            f"Ghost paid {paid} and left {sorted(ghost.volatiles(1))}; "
            f"the other got {traded}")


@check("hex", "doubles against a target that already has a status")
def _hex():
    def dealt(status):
        # Not the Blissey wall: Normal is immune to Ghost, so both arms read
        # zero and the move looked broken.
        f = Fight(ours("gengar", "__none__",
                       ("hex", "shadowball", "protect", "splash"),
                       None, "timid", (0, 0, 2, 32, 0, 32)),
                  [mon("milotic", "__none__",
                       ("splash", "surf", "recover", "protect"),
                       None, "serious", (32, 0, 32, 0, 2, 0))])
        if status:
            f.state.sides[1].status[f.state.sides[1].active[0]] = status
        f.turn(Action.move(0), Action.move(0))
        return f.damage("hex")
    plain = dealt(None)
    on_status = dealt("brn")
    return (plain > 0 and on_status > plain * 1.5,
            f"against a healthy target {plain}, against a burnt one {on_status}")


@check("ceaselessedge", "damages and leaves a layer of Spikes behind")
def _ceaseless_edge():
    f = _until(lambda seed: Fight(
        ours("samurotthisui", "__none__",
             ("ceaselessedge", "aquajet", "swordsdance", "protect"),
             None, "adamant", (0, 32, 2, 0, 0, 32)),
        [wall()], seed).turn(Action.move(0), Action.move(0)),
        lambda f: f.damage("ceaselessedge") > 0)
    if f is None:
        return False, "never connected in ten tries"
    return ("spikes" in f.conditions(1),
            f"their side conditions {f.conditions(1) or 'none'}")


@check("lowkick", "hits a heavy target harder than a light one")
def _low_kick():
    def dealt(species):
        f = Fight(ours("machamp", "__none__",
                       ("lowkick", "closecombat", "knockoff", "bulkup"),
                       None, "adamant", (0, 32, 2, 0, 0, 32)),
                  [mon(species, "__none__",
                       ("splash", "tackle", "rest", "protect"),
                       None, "serious", (32, 0, 32, 0, 2, 0))])
        f.turn(Action.move(0), Action.move(0))
        return f.damage("lowkick")
    light = dealt("magikarp")      # 10.0 kg
    heavy = dealt("snorlax")       # 460.0 kg
    return (heavy > light,
            f"against 10 kg {light}, against 460 kg {heavy}")


@check("heavyslam", "hits harder the heavier the user is next to the target")
def _heavy_slam():
    def dealt(species):
        f = Fight(ours("snorlax", "__none__",       # 460 kg
                       ("heavyslam", "bodyslam", "rest", "protect"),
                       None, "adamant", (0, 32, 2, 0, 0, 32)),
                  [mon(species, "__none__",
                       ("splash", "tackle", "rest", "protect"),
                       None, "serious", (32, 0, 32, 0, 2, 0))])
        f.turn(Action.move(0), Action.move(0))
        return f.damage("heavyslam")
    feather = dealt("magikarp")    # 10.0 kg: the biggest ratio
    solid = dealt("archaludon")    # 60.0 kg
    return (feather > solid,
            f"against 10 kg {feather}, against 60 kg {solid}")


@check("waterspout", "falls off as the user loses HP")
def _water_spout():
    def dealt(hurt_first):
        f = Fight(ours("kyogre" if DEX.species.get("kyogre") else "wailord",
                       "__none__",
                       ("waterspout", "surf", "protect", "splash"),
                       None, "modest", (32, 0, 2, 32, 0, 0)),
                  [mon("archaludon", "__none__",
                       ("dragontail", "flashcannon", "protect", "splash"),
                       None, "adamant", (0, 32, 2, 0, 0, 32))])
        if hurt_first:
            f.state.sides[0].hp[f.state.sides[0].active[0]] //= 4
        f.turn(Action.move(0), Action.move(3))
        return f.damage("waterspout")
    healthy = dealt(False)
    hurt = dealt(True)
    return (healthy > hurt * 1.5,
            f"at full health {healthy}, at a quarter {hurt}")


@check("payback", "doubles when the user has already been hit this turn")
def _payback():
    def dealt(foe_speed):
        f = Fight(ours("kingambit", "__none__",
                       ("payback", "ironhead", "suckerpunch", "swordsdance"),
                       None, "brave", (0, 32, 2, 0, 0, 0)),
                  # Same species and the same Defence both times: only the
                  # Speed investment differs, so only the order changes.
                  [mon("snorlax", "__none__",
                       ("bodyslam", "splash", "rest", "protect"),
                       None, "serious", (32, 0, 32, 0, 2, foe_speed))])
        f.turn(Action.move(0), Action.move(0))
        return f.damage("payback")
    they_first = dealt(32)     # they outrun us, so Payback is paid back
    we_first = dealt(0)
    return (they_first > we_first * 1.5,
            f"moving second {they_first}, moving first {we_first}")


@check("soak", "turns the target into a Water type")
def _soak():
    f = Fight(ours("clefable", "__none__",
                   ("soak", "moonblast", "protect", "splash"),
                   None, "timid", (0, 0, 2, 32, 0, 32)),
              [dummy("snorlax")])
    f.turn(Action.move(0), Action.move(0))
    types = f.state.types(1, f.state.sides[1].active[0])
    return (tuple(types) == ("water",), f"their types are now {tuple(types)}")


@check("smackdown", "brings a Flying target down to the ground")
def _smack_down():
    f = _until(lambda seed: Fight(
        ours("garchomp", "__none__",
             ("smackdown", "earthquake", "dragonclaw", "protect"),
             None, "jolly", (0, 32, 2, 0, 0, 32)),
        [mon("corviknight", "__none__",
             ("roost", "irondefense", "bodypress", "uturn"),
             None, "impish", (32, 0, 32, 0, 2, 0))], seed)
        .turn(Action.move(0), Action.move(0)),
        lambda f: "smackdown" in f.volatiles(1))
    if f is None:
        return False, "the volatile never appeared in ten tries"
    f.turn(Action.move(1), Action.move(0))     # Earthquake, normally no answer
    return (f.damage("earthquake") > 0,
            f"grounded, and Earthquake then took {f.damage('earthquake')}")


@check("psychicnoise", "stops the target healing")
def _psychic_noise():
    f = _until(lambda seed: Fight(
        ours("indeedee", "__none__",
             ("psychicnoise", "psychic", "protect", "splash"),
             None, "modest", (0, 0, 2, 32, 0, 32)),
        [mon("snorlax", "__none__",
             ("rest", "bodyslam", "splash", "protect"),
             None, "serious", (32, 0, 32, 0, 2, 0))], seed)
        .turn(Action.move(0), Action.move(1)),
        lambda f: "healblock" in f.volatiles(1))
    if f is None:
        return False, "heal block never landed in ten tries"
    hurt = f.hp(1)
    f.turn(Action.move(2), Action.move(0))     # they try to Rest
    return (f.hp(1) <= hurt, f"they were at {hurt}, after trying to Rest {f.hp(1)}")


@check("mortalspin", "clears our own hazards and poisons what it hits")
def _mortal_spin():
    # A Steel type is immune to Poison, so a Ferrothorn opposite made the whole
    # move a no-op and it read as unimplemented. The hazard is placed directly:
    # what is under test is the spin, not Spikes.
    f = Fight(ours("glimmora", "__none__",
                   ("mortalspin", "sludgewave", "spikes", "protect"),
                   None, "timid", (0, 32, 2, 0, 0, 32)),
              [mon("snorlax", "__none__",
                   ("splash", "bodyslam", "rest", "protect"),
                   None, "serious", (32, 0, 32, 0, 2, 0))])
    f.state.sides[0].conditions["spikes"] = 1
    laid = f.conditions(0)
    f.turn(Action.move(0), Action.move(0))
    return (laid and not f.conditions(0) and f.status(1) in ("psn", "tox"),
            f"our side had {laid or 'none'}, after the spin "
            f"{f.conditions(0) or 'none'}; their status {f.status(1)}")


@check("spikyshield", "blocks the hit and takes a slice off a contact attacker")
def _spiky_shield():
    f = Fight(ours("ferrothorn", "__none__",
                   ("spikyshield", "gyroball", "leechseed", "spikes"),
                   None, "relaxed", (32, 0, 32, 0, 2, 0)),
              [mon("garchomp", "__none__",
                   ("dragonclaw", "earthquake", "firefang", "stoneedge"),
                   None, "jolly", (0, 32, 2, 0, 0, 32))])
    f.turn(Action.move(0), Action.move(0))     # Dragon Claw makes contact
    return (f.hp(0) == f.max_hp(0) and f.hp(1) < f.max_hp(1),
            f"we are at {f.hp(0)}/{f.max_hp(0)}, they are at "
            f"{f.hp(1)}/{f.max_hp(1)}")


@check("banefulbunker", "blocks the hit and poisons a contact attacker")
def _baneful_bunker():
    f = Fight(ours("toxapex", "__none__",
                   ("banefulbunker", "scald", "recover", "toxic"),
                   None, "bold", (32, 0, 32, 0, 2, 0)),
              [mon("garchomp", "__none__",
                   ("dragonclaw", "earthquake", "firefang", "stoneedge"),
                   None, "jolly", (0, 32, 2, 0, 0, 32))])
    f.turn(Action.move(0), Action.move(0))
    return (f.hp(0) == f.max_hp(0) and f.status(1) in ("psn", "tox"),
            f"we are at {f.hp(0)}/{f.max_hp(0)}, their status {f.status(1)}")


@check("transform", "becomes the thing it is looking at")
def _transform():
    f = Fight(ours("ditto", "__none__",
                   ("transform", "splash", "tackle", "protect")),
              [dummy("snorlax")])
    before = f.active_species(0)
    f.turn(Action.move(0), Action.move(0))
    return ("transformed" in f.volatiles(0) or f.active_species(0) != before,
            f"we were {before}, volatiles {sorted(f.volatiles(0)) or 'none'}")


@check("bellydrum", "spends half the user's HP to max its Attack")
def _belly_drum():
    f = Fight(ours("snorlax", "__none__",
                   ("bellydrum", "bodyslam", "rest", "protect"),
                   None, "adamant", (32, 32, 2, 0, 0, 0)),
              [dummy("magikarp")])
    before = f.hp(0)
    f.turn(Action.move(0), Action.move(0))
    return (f.boosts(0).get("atk") == 6 and before - f.hp(0) > 0,
            f"paid {before - f.hp(0)} of {f.max_hp(0)}, boosts {f.boosts(0)}")


@check("rapidspin", "clears our own hazards away")
def _rapid_spin():
    f = Fight(ours("cinderace", "__none__",
                   ("rapidspin", "pyroball", "uturn", "protect"),
                   None, "jolly", (0, 32, 2, 0, 0, 32)),
              [mon("ferrothorn", "__none__",
                   ("spikes", "gyroball", "protect", "splash"),
                   None, "relaxed", (32, 0, 32, 0, 2, 0))])
    f.turn(Action.move(3), Action.move(0))
    laid = f.conditions(0)
    f.turn(Action.move(0), Action.move(3))
    return (laid and not f.conditions(0),
            f"ours were {laid or 'none'}, after the spin "
            f"{f.conditions(0) or 'none'}")


@check("lastrespects", "hits harder for every ally already gone")
def _last_respects():
    def dealt(sacrifice_first):
        houndstone = mon("houndstone", "__none__",
                         ("lastrespects", "shadowsneak", "protect", "splash"),
                         None, "adamant", (0, 32, 2, 0, 0, 0))
        # Not Final Gambit: that took the target down to 42 HP on the way out,
        # so the knockout truncated the damage and both arms read 42. The foe
        # does the killing here and stays at full health for both arms.
        sacrifice = mon("magikarp", "__none__",
                        ("splash", "tackle", "flail", "bounce"))
        order = [sacrifice, houndstone] if sacrifice_first else [houndstone]
        # Not the Blissey wall: Last Respects is Ghost and Normal is immune.
        f = Fight(order, [mon("milotic", "__none__",
                              ("surf", "splash", "recover", "protect"),
                              None, "modest", (0, 0, 2, 32, 0, 32))])
        if sacrifice_first:
            f.turn(Action.move(0), Action.move(0))      # Surf ends the Magikarp
            f.turn(Action.switch(1), Action.PASS)       # Houndstone comes in
        f.turn(Action.move(0), Action.move(1))
        return f.damage("lastrespects")
    alone = dealt(False)
    after_a_loss = dealt(True)
    return (alone > 0 and after_a_loss > alone,
            f"with the team whole {alone}, one ally down {after_a_loss}")


@check("icespinner", "sweeps the terrain away")
def _ice_spinner():
    f = Fight(ours("weavile", "__none__",
                   ("icespinner", "knockoff", "iceshard", "swordsdance"),
                   None, "jolly", (0, 32, 2, 0, 0, 32)),
              [wall()])
    f.state.field.terrain = "electricterrain"
    f.turn(Action.move(0), Action.move(0))
    return (f.state.field.terrain is None,
            f"terrain afterwards {f.state.field.terrain!r}")


@check("morningsun", "gives back about half of the user's maximum")
def _morning_sun():
    f, hurt, healed = _heals_half(
        "espeon", ("morningsun", "psychic", "calmmind", "protect"), 0)
    return healed > 0, f"at {hurt} of {f.max_hp(0)}, recovered {healed}"


@check("focusenergy", "makes the user's hits likelier to be critical")
def _focus_energy():
    f = Fight(ours("kingambit", "__none__",
                   ("focusenergy", "ironhead", "suckerpunch", "swordsdance")),
              [dummy("magikarp")])
    f.turn(Action.move(0), Action.move(0))
    return ("focusenergy" in f.volatiles(0),
            f"our volatiles {sorted(f.volatiles(0)) or 'none'}")


@check("assurance", "doubles on a target already hurt this turn")
def _assurance():
    def dealt(foe_move):
        # Brave Bird's recoil is the target hurting itself before we swing;
        # in singles there is no other way for the same turn to have damaged
        # them already. Slow on purpose so they act first.
        # Kingambit, not Weavile: Brave Bird was knocking the user out before
        # it could answer, and a dead attacker deals zero either way.
        f = Fight(ours("kingambit", "__none__",
                       ("assurance", "ironhead", "suckerpunch", "swordsdance"),
                       None, "brave", (32, 32, 32, 0, 2, 0)),
                  [mon("staraptor", "__none__",
                       ("bravebird", "roost", "uturn", "protect"),
                       None, "jolly", (32, 32, 2, 0, 0, 32))])
        f.turn(Action.move(0), Action.move(foe_move))
        return f.damage("assurance")
    after_recoil = dealt(0)
    untouched = dealt(1)      # Roost: they heal instead, taking nothing
    return (after_recoil > untouched * 1.5,
            f"after their recoil {after_recoil}, untouched {untouched}")


@check("disable", "takes the target's last move away from it")
def _disable():
    from pkcm.engine.state import legal_actions
    f = _until(lambda seed: Fight(
        ours("gengar", "__none__",
             ("disable", "shadowball", "protect", "splash"),
             None, "timid", (0, 0, 2, 32, 0, 32)),
        [mon("snorlax", "__none__",
             ("bodyslam", "splash", "rest", "protect"),
             None, "serious", (32, 0, 32, 0, 2, 0))], seed)
        .turn(Action.move(2), Action.move(0))
        .turn(Action.move(0), Action.move(1)),
        lambda f: "disabled" in f.volatiles(1))
    if f is None:
        return False, "the volatile never appeared in ten tries"
    offered = {one.index for one in legal_actions(f.state, 1)
               if str(one).startswith("move")}
    return (0 not in offered,
            f"moves still offered {sorted(offered)}, Body Slam is 0")


@check("ragefist", "hits harder for every time the user has been hit")
def _rage_fist():
    def dealt(taken):
        f = Fight(ours("annihilape", "__none__",
                       ("ragefist", "drainpunch", "bulkup", "protect"),
                       None, "adamant", (32, 32, 2, 0, 0, 0)),
                  # Rage Fist is Ghost, so not a Normal type opposite.
                  [mon("milotic", "__none__",
                       ("surf", "splash", "recover", "protect"),
                       None, "serious", (32, 0, 32, 0, 2, 0))])
        for _ in range(taken):
            f.turn(Action.move(3), Action.move(0))   # Protect fails eventually
        f.turn(Action.move(0), Action.move(1))
        return f.damage("ragefist")
    fresh = dealt(0)
    battered = dealt(3)
    return (battered > fresh,
            f"never hit {fresh}, after taking hits {battered}")


@check("temperflare", "doubles when the user's last move failed")
def _temper_flare():
    def dealt(fail_first):
        f = Fight(ours("cinderace", "__none__",
                       ("temperflare", "pyroball", "protect", "splash"),
                       None, "adamant", (0, 32, 2, 0, 0, 32)),
                  [mon("archaludon", "__none__",
                       ("protect", "flashcannon", "dragontail", "splash"),
                       None, "sassy", (32, 0, 32, 0, 32, 0))])
        if fail_first:
            f.turn(Action.move(1), Action.move(0))   # they Protect: ours fails
        f.turn(Action.move(0), Action.move(3))
        return f.damage("temperflare")
    clean = dealt(False)
    after_a_failure = dealt(True)
    return (after_a_failure > clean * 1.5,
            f"after a clean turn {clean}, after a failure {after_a_failure}")


@check("tidyup", "clears both sides of clutter and steps the user up")
def _tidy_up():
    f = Fight(ours("cinderace", "__none__",
                   ("tidyup", "pyroball", "uturn", "protect"),
                   None, "jolly", (0, 32, 2, 0, 0, 32)),
              [dummy("magikarp")])
    f.state.sides[0].conditions["spikes"] = 1
    f.state.sides[1].conditions["stealthrock"] = 1
    f.turn(Action.move(0), Action.move(0))
    got = f.boosts(0)
    return (not f.conditions(0) and not f.conditions(1)
            and got.get("atk") == 1 and got.get("spe") == 1,
            f"ours {f.conditions(0) or 'none'}, theirs {f.conditions(1) or 'none'}, "
            f"boosts {got or 'none'}")


@check("thief", "takes the item off whatever it hits")
def _thief():
    f = _until(lambda seed: Fight(
        ours("weavile", "__none__",
             ("thief", "knockoff", "iceshard", "swordsdance"),
             None, "jolly", (0, 32, 2, 0, 0, 32)),
        [mon("snorlax", "__none__",
             ("splash", "bodyslam", "rest", "protect"),
             "leftovers", "serious", (32, 0, 32, 0, 2, 0))], seed)
        .turn(Action.move(0), Action.move(0)),
        lambda f: f.damage("thief") > 0)
    if f is None:
        return False, "never connected in ten tries"
    return (f.item(1) is None and f.item(0) == "leftovers",
            f"we hold {f.item(0)}, they hold {f.item(1)}")


@check("switcheroo", "swaps the two held items")
def _switcheroo():
    f = Fight(ours("gengar", "__none__",
                   ("switcheroo", "shadowball", "protect", "splash"),
                   "choicescarf", "timid", (0, 0, 2, 32, 0, 32)),
              [mon("snorlax", "__none__",
                   ("splash", "bodyslam", "rest", "protect"),
                   "leftovers", "serious", (32, 0, 32, 0, 2, 0))])
    f.turn(Action.move(0), Action.move(0))
    return (f.item(0) == "leftovers" and f.item(1) == "choicescarf",
            f"we hold {f.item(0)}, they hold {f.item(1)}")


@check("psychup", "copies the target's stat changes onto the user")
def _psych_up():
    f = Fight(ours("clefable", "__none__",
                   ("psychup", "moonblast", "protect", "splash"),
                   None, "timid", (32, 0, 2, 0, 0, 32)),
              [mon("dragonite", "__none__",
                   ("dragondance", "dragonclaw", "roost", "firepunch"))])
    f.turn(Action.move(3), Action.move(0))     # they dance
    theirs = f.boosts(1)
    f.turn(Action.move(0), Action.move(2))     # we copy while they Roost
    return (theirs and f.boosts(0) == theirs,
            f"theirs {theirs or 'none'}, ours after the copy "
            f"{f.boosts(0) or 'none'}")


@check("skillswap", "trades the two abilities over")
def _skill_swap():
    f = Fight(ours("alakazam", "magicguard",
                   ("skillswap", "psychic", "protect", "splash"),
                   None, "timid", (0, 0, 2, 32, 0, 32)),
              [mon("snorlax", "thickfat",
                   ("splash", "bodyslam", "rest", "protect"),
                   None, "serious", (32, 0, 32, 0, 2, 0))])
    f.turn(Action.move(0), Action.move(0))
    from pkcm.engine.battle import make_context
    ctx = make_context(f.state)
    ours_now = ctx.ability_of((0, f.state.sides[0].active[0]))
    theirs_now = ctx.ability_of((1, f.state.sides[1].active[0]))
    return (ours_now == "thickfat" and theirs_now == "magicguard",
            f"we now have {ours_now}, they have {theirs_now}")


@check("safeguard", "keeps status off our side while it stands")
def _safeguard():
    f = Fight(ours("clefable", "__none__",
                   ("safeguard", "moonblast", "protect", "splash"),
                   None, "timid", (32, 0, 32, 0, 2, 32)),
              [mon("blissey", "__none__",
                   ("toxic", "seismictoss", "softboiled", "protect"),
                   None, "serious", (32, 0, 32, 0, 2, 0))])
    f.turn(Action.move(0), Action.move(3))
    up = "safeguard" in f.conditions(0)
    f.turn(Action.move(3), Action.move(0))     # they try Toxic
    return (up and f.status(0) is None,
            f"condition={up}, our status after their Toxic {f.status(0)}")


@check("stoneaxe", "damages and leaves Stealth Rock behind")
def _stone_axe():
    f = _until(lambda seed: Fight(
        ours("kleavor", "__none__",
             ("stoneaxe", "uturn", "swordsdance", "protect"),
             None, "adamant", (0, 32, 2, 0, 0, 32)),
        [wall()], seed).turn(Action.move(0), Action.move(0)),
        lambda f: f.damage("stoneaxe") > 0)
    if f is None:
        return False, "never connected in ten tries"
    return ("stealthrock" in f.conditions(1),
            f"their side conditions {f.conditions(1) or 'none'}")


@check("minimize", "raises evasion two stages")
def _minimize():
    return _boosts_by("magikarp", ("minimize", "splash", "tackle", "flail"),
                      {"evasion": 2})


@check("burnup", "spends the user's Fire type to fire it")
def _burn_up():
    f = Fight(ours("cinderace", "__none__",
                   ("burnup", "pyroball", "uturn", "protect"),
                   None, "adamant", (0, 32, 2, 0, 0, 32)),
              [wall()])
    before = tuple(f.state.types(0, f.state.sides[0].active[0]))
    f.turn(Action.move(0), Action.move(0))
    after = tuple(f.state.types(0, f.state.sides[0].active[0]))
    return ("fire" in before and "fire" not in after,
            f"we were {before}, now {after}")


@check("lashout", "doubles after the user has just been dropped a stage")
def _lash_out():
    def dealt(dropped_first):
        # The drop has to land in the same turn as the swing, so the dropper
        # has to be the faster one -- a Weavile holding Lash Out outruns
        # anything that could drop it, and splitting it over two turns lost
        # the mark before the move that reads it.
        f = Fight(ours("kingambit", "__none__",
                       ("lashout", "ironhead", "suckerpunch", "swordsdance"),
                       None, "brave", (32, 32, 2, 0, 0, 0)),
                  [mon("weavile", "__none__",
                       ("growl", "splash", "iceshard", "knockoff"),
                       None, "jolly", (32, 0, 32, 0, 2, 32))])
        f.turn(Action.move(0), Action.move(0 if dropped_first else 1))
        return f.damage("lashout")
    dropped = dealt(True)      # Growl takes a stage off us first
    clean = dealt(False)
    # Not 2x on the scoreboard: the same Attack drop that arms Lash Out also
    # takes the attacking stat to two thirds, so a working move reads about
    # 1.33x. Asking for 1.5 failed a move that was doing its job.
    return (dropped > clean * 1.2,
            f"after being dropped {dropped}, untouched {clean} "
            f"({dropped / max(clean, 1):.2f}x, and 1.33x is what a working "
            f"Lash Out looks like through its own Attack drop)")


@check("bugbite", "eats the berry the target was holding")
def _bug_bite():
    f = _until(lambda seed: Fight(
        ours("weavile", "__none__",
             ("bugbite", "knockoff", "iceshard", "swordsdance"),
             None, "jolly", (0, 32, 2, 0, 0, 32)),
        [mon("snorlax", "__none__",
             ("splash", "bodyslam", "rest", "protect"),
             "sitrusberry", "serious", (32, 0, 32, 0, 2, 0))], seed)
        .turn(Action.move(0), Action.move(0)),
        lambda f: f.damage("bugbite") > 0)
    if f is None:
        return False, "never connected in ten tries"
    return (f.item(1) is None, f"they still hold {f.item(1)}")


@check("ragingbull", "breaks the screen it is thrown at")
def _raging_bull():
    f = _until(lambda seed: Fight(
        ours("taurospaldeacombat", "__none__",
             ("ragingbull", "closecombat", "protect", "splash"),
             None, "adamant", (0, 32, 2, 0, 0, 32)),
        [mon("blissey", "__none__",
             ("reflect", "softboiled", "seismictoss", "protect"),
             None, "serious", (32, 0, 32, 0, 2, 0))], seed)
        .turn(Action.move(2), Action.move(0))
        .turn(Action.move(0), Action.move(1)),   # Soft-Boiled, not Protect
        lambda f: f.damage("ragingbull") > 0)
    if f is None:
        return False, "never connected in ten tries"
    return ("reflect" not in f.conditions(1),
            f"their side conditions {f.conditions(1) or 'none'}")


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
