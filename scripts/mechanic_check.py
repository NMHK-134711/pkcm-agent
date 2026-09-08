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
    def crits(psyched):
        seen = 0
        for seed in range(24):
            f = Fight(ours("kingambit", "__none__",
                           ("focusenergy", "ironhead", "suckerpunch",
                            "swordsdance"), None, "adamant",
                           (0, 0, 2, 0, 0, 0)),
                      [wall()], seed)
            if psyched:
                f.turn(Action.move(0), Action.move(0))
            f.turn(Action.move(1), Action.move(0))
            seen += f.said("crit=True")
        return seen
    # Two stages is a coin flip in this generation against one in twenty-four,
    # and 24 casts separate those comfortably. The volatile on its own says
    # nothing about whether anything reads it.
    keyed_up = crits(True)
    calm = crits(False)
    return (keyed_up > calm + 4,
            f"critical hits in 24: keyed up {keyed_up}, calm {calm}")


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


# -- moves no field party carries ------------------------------------------- #
#
# Nothing here shows up in the 253, so none of it can be reading wrong in a
# measurement today. They are checked because the party optimizer picks from
# the whole legal move pool: Minimize sat behind a clause nobody had lifted,
# and the moment it was lifted two things downstream of an evasion stage had
# never run once. Anything a mutation can reach can be reached tomorrow.


def _power_ratio(move, ours_kw, theirs_a, theirs_b):
    """Two arms of a damage comparison that differ only in the opponent."""
    def dealt(theirs):
        f = Fight([mon(**ours_kw)], [mon(**theirs)])
        f.turn(Action.move(0), Action.move(0))
        return f.damage(move)
    return dealt(theirs_a), dealt(theirs_b)


@check("acrobatics", "doubles when the user is holding nothing")
def _acrobatics():
    def dealt(item):
        f = Fight(ours("staraptor", "__none__",
                       ("acrobatics", "bravebird", "uturn", "protect"),
                       item, "jolly", (0, 32, 2, 0, 0, 32)),
                  [wall()])
        f.turn(Action.move(0), Action.move(0))
        return f.damage("acrobatics")
    empty = dealt(None)
    holding = dealt("leftovers")
    return (empty > holding * 1.5,
            f"empty-handed {empty}, holding Leftovers {holding}")


@check("aquaring", "gives a little back every turn")
def _aqua_ring():
    f = Fight(ours("milotic", "__none__",
                   ("aquaring", "surf", "recover", "protect"),
                   None, "serious", (32, 0, 32, 0, 2, 0)),
              [mon("garchomp", "__none__",
                   ("earthquake", "dragonclaw", "firefang", "stoneedge"),
                   None, "jolly", (0, 32, 2, 0, 0, 32))])
    f.turn(Action.move(1), Action.move(0))     # take a hit to leave room
    f.turn(Action.move(0), Action.move(3))     # the ring goes up
    up = "aquaring" in f.volatiles(0)
    hurt = f.hp(0)
    f.turn(Action.move(3), Action.move(3))     # Protect, so only the ring acts
    return (up and f.hp(0) > hurt,
            f"volatile={up}, {hurt} -> {f.hp(0)} of {f.max_hp(0)}")


@check("attract", "an opposite-gender target sometimes cannot act")
def _attract():
    f = _until(lambda seed: Fight(
        [PokemonSet(species="clefable", ability="__none__", gender="F",
                    moves=("attract", "moonblast", "protect", "splash"),
                    item=None, nature="timid", sp=(32, 0, 32, 0, 2, 32))],
        # Both genders stated: a set that does not say leaves the engine with
        # "unknown", and Attract refuses that rather than guessing -- which
        # reads exactly like the move doing nothing.
        [PokemonSet(species="snorlax", ability="__none__", gender="M",
                    moves=("bodyslam", "splash", "rest", "protect"),
                    item=None, nature="serious", sp=(32, 0, 32, 0, 2, 0))], seed)
        .turn(Action.move(0), Action.move(1)),
        lambda f: "attract" in f.volatiles(1))
    if f is None:
        return False, "the volatile never appeared -- genders may be fixed"
    refused = 0
    for _ in range(8):
        if f.state.phase.name != "BATTLE":
            break
        f.turn(Action.move(2), Action.move(0))
        refused += f.said("cant_move")
    return (refused > 0, f"infatuated, and refused to act {refused} times in 8")


@check("block", "the target cannot leave")
def _block():
    from pkcm.engine.state import legal_actions
    f = Fight(ours("dusknoir", "__none__",
                   ("block", "shadowsneak", "protect", "splash")),
              [dummy("magikarp")])
    f.turn(Action.move(0), Action.move(0))
    switches = [one for one in legal_actions(f.state, 1)
                if str(one).startswith("switch")]
    return (not switches,
            f"they are still offered {len(switches)} switches; "
            f"volatiles {sorted(f.volatiles(1))}")


@check("meanlook", "the target cannot leave")
def _mean_look():
    from pkcm.engine.state import legal_actions
    f = Fight(ours("gengar", "__none__",
                   ("meanlook", "shadowball", "protect", "splash")),
              [dummy("magikarp")])
    f.turn(Action.move(0), Action.move(0))
    switches = [one for one in legal_actions(f.state, 1)
                if str(one).startswith("switch")]
    return (not switches, f"they are still offered {len(switches)} switches")


@check("copycat", "uses whatever was used last")
def _copycat():
    f = Fight(ours("clefable", "__none__",
                   ("copycat", "moonblast", "protect", "splash"),
                   None, "serious", (32, 0, 32, 0, 2, 0)),
              [mon("snorlax", "__none__",
                   ("swordsdance", "bodyslam", "rest", "protect"),
                   None, "serious", (32, 0, 32, 0, 2, 32))])
    f.turn(Action.move(0), Action.move(0))     # they dance, we copy it
    return (f.boosts(0).get("atk") == 2,
            f"our boosts {f.boosts(0) or 'none'} after copying Swords Dance")


@check("covet", "takes the item off whatever it hits")
def _covet():
    f = _until(lambda seed: Fight(
        ours("kangaskhan", "__none__",
             ("covet", "bodyslam", "suckerpunch", "protect"),
             None, "jolly", (0, 32, 2, 0, 0, 32)),
        [mon("snorlax", "__none__",
             ("splash", "bodyslam", "rest", "protect"),
             "leftovers", "serious", (32, 0, 32, 0, 2, 0))], seed)
        .turn(Action.move(0), Action.move(0)),
        lambda f: f.damage("covet") > 0)
    if f is None:
        return False, "never connected in ten tries"
    return (f.item(1) is None and f.item(0) == "leftovers",
            f"we hold {f.item(0)}, they hold {f.item(1)}")


@check("pluck", "eats the berry the target was holding")
def _pluck():
    f = _until(lambda seed: Fight(
        ours("staraptor", "__none__",
             ("pluck", "bravebird", "uturn", "protect"),
             None, "jolly", (0, 32, 2, 0, 0, 32)),
        [mon("snorlax", "__none__",
             ("splash", "bodyslam", "rest", "protect"),
             "sitrusberry", "serious", (32, 0, 32, 0, 2, 0))], seed)
        .turn(Action.move(0), Action.move(0)),
        lambda f: f.damage("pluck") > 0)
    if f is None:
        return False, "never connected in ten tries"
    return (f.item(1) is None, f"they still hold {f.item(1)}")


@check("electroball", "hits harder the more Speed the user has over the target")
def _electro_ball():
    fast, slow = _power_ratio(
        "electroball",
        dict(species="pikachu", ability="__none__",
             moves=("electroball", "thunderbolt", "protect", "splash"),
             nature="timid", sp=(0, 0, 2, 32, 0, 32)),
        dict(species="snorlax", ability="__none__",
             moves=("splash", "bodyslam", "rest", "protect"),
             nature="serious", sp=(32, 0, 32, 0, 2, 0)),
        dict(species="snorlax", ability="__none__",
             moves=("splash", "bodyslam", "rest", "protect"),
             nature="serious", sp=(32, 0, 32, 0, 2, 32)))
    return (fast > slow,
            f"against a slow target {fast}, against a faster one {slow}")


@check("gyroball", "hits harder the slower the user is")
def _gyro_ball():
    def dealt(our_speed):
        f = Fight(ours("ferrothorn", "__none__",
                       ("gyroball", "leechseed", "protect", "spikes"),
                       None, "brave", (0, 32, 2, 0, 0, our_speed)),
                  [mon("snorlax", "__none__",
                       ("splash", "bodyslam", "rest", "protect"),
                       None, "jolly", (32, 0, 32, 0, 2, 32))])
        f.turn(Action.move(0), Action.move(0))
        return f.damage("gyroball")
    crawling = dealt(0)
    quicker = dealt(32)
    return (crawling > quicker,
            f"at rock bottom Speed {crawling}, with Speed invested {quicker}")


@check("endure", "survives on one hit point")
def _endure():
    f = Fight(ours("magikarp", "__none__",
                   ("endure", "splash", "tackle", "flail")),
              [mon("garchomp", "__none__",
                   ("earthquake", "dragonclaw", "firefang", "stoneedge"),
                   None, "jolly", (0, 32, 2, 0, 0, 32))])
    f.turn(Action.move(0), Action.move(0))
    return (f.hp(0) == 1 and f.state.phase.name == "BATTLE",
            f"we are on {f.hp(0)} hit points, phase {f.state.phase.name}")


@check("flail", "hits hardest when the user is nearly gone")
def _flail():
    def dealt(fraction):
        f = Fight(ours("magikarp", "__none__",
                       ("flail", "splash", "tackle", "bounce"),
                       None, "adamant", (0, 32, 2, 0, 0, 32)),
                  [wall()])
        slot = f.state.sides[0].active[0]
        f.state.sides[0].hp[slot] = max(1, f.state.sides[0].hp[slot] // fraction)
        f.turn(Action.move(0), Action.move(0))
        return f.damage("flail")
    healthy = dealt(1)
    dying = dealt(20)
    return (dying > healthy,
            f"at full health {healthy}, at a twentieth {dying}")


@check("eruption", "falls off as the user loses HP")
def _eruption():
    def dealt(hurt_first):
        f = Fight(ours("typhlosion", "__none__",
                       ("eruption", "flamethrower", "protect", "splash"),
                       None, "modest", (32, 0, 2, 32, 0, 0)),
                  [mon("archaludon", "__none__",
                       ("splash", "flashcannon", "dragontail", "protect"),
                       None, "sassy", (32, 0, 32, 0, 32, 0))])
        if hurt_first:
            slot = f.state.sides[0].active[0]
            f.state.sides[0].hp[slot] //= 4
        f.turn(Action.move(0), Action.move(0))
        return f.damage("eruption")
    healthy = dealt(False)
    hurt = dealt(True)
    return (healthy > hurt * 1.5,
            f"at full health {healthy}, at a quarter {hurt}")


@check("heatcrash", "hits harder the heavier the user is next to the target")
def _heat_crash():
    def dealt(species):
        f = Fight(ours("snorlax", "__none__",       # 460 kg
                       ("heatcrash", "bodyslam", "rest", "protect"),
                       None, "adamant", (0, 32, 2, 0, 0, 0)),
                  [mon(species, "__none__",
                       ("splash", "tackle", "rest", "protect"),
                       None, "serious", (32, 0, 32, 0, 2, 0))])
        f.turn(Action.move(0), Action.move(0))
        return f.damage("heatcrash")
    # Two pure Water types, so the type chart says the same thing to both and
    # the only difference left is the weight ratio. Against a 60 kg Archaludon
    # the ratio was still over five, which is the cap -- so both arms were at
    # 120 power and what the numbers showed was Fire against Steel.
    feather = dealt("magikarp")     # 10 kg: ratio 46, the cap
    whale = dealt("wailord")        # 398 kg: ratio 1.16, the floor
    return (feather > whale,
            f"against 10 kg {feather}, against 398 kg {whale}")


@check("gastroacid", "switches the target's ability off")
def _gastro_acid():
    from pkcm.engine.battle import make_context
    f = Fight(ours("gengar", "__none__",
                   ("gastroacid", "shadowball", "protect", "splash")),
              [mon("snorlax", "thickfat",
                   ("splash", "bodyslam", "rest", "protect"),
                   None, "serious", (32, 0, 32, 0, 2, 0))])
    f.turn(Action.move(0), Action.move(0))
    ability = make_context(f.state).ability_of((1, f.state.sides[1].active[0]))
    return (ability is None or ability == "",
            f"their ability reads {ability!r}, "
            f"volatiles {sorted(f.volatiles(1))}")


@check("gravity", "pulls a Flying type down where Ground can reach it")
def _gravity():
    f = Fight(ours("clefable", "__none__",
                   ("gravity", "moonblast", "protect", "splash"),
                   None, "timid", (32, 0, 32, 0, 2, 32))
              + [mon("garchomp", "__none__",
                     ("earthquake", "dragonclaw", "protect", "splash"),
                     None, "jolly", (0, 32, 2, 0, 0, 32))],
              [mon("corviknight", "__none__",
                   ("roost", "irondefense", "bodypress", "uturn"),
                   None, "impish", (32, 0, 32, 0, 2, 0))])
    f.turn(Action.move(0), Action.move(0))
    up = "gravity" in f.state.field.rooms
    f.turn(Action.switch(1), Action.move(0))
    f.turn(Action.move(0), Action.move(0))
    return (up and f.damage("earthquake") > 0,
            f"gravity up={up}, Earthquake then took {f.damage('earthquake')}")


@check("magnetrise", "floats out of Ground's reach")
def _magnet_rise():
    f = Fight(ours("archaludon", "__none__",
                   ("magnetrise", "flashcannon", "protect", "splash"),
                   None, "sassy", (32, 0, 32, 0, 32, 0)),
              [mon("garchomp", "__none__",
                   ("earthquake", "dragonclaw", "firefang", "stoneedge"),
                   None, "jolly", (0, 32, 2, 0, 0, 32))])
    f.turn(Action.move(0), Action.move(3))
    up = "magnetrise" in f.volatiles(0)
    f.turn(Action.move(3), Action.move(0))     # they try Earthquake
    return (up and f.said("immune"),
            f"volatile={up}, Earthquake was refused={f.said('immune')}")


@check("healbell", "clears the status off the whole party")
def _heal_bell():
    f = Fight(ours("clefable", "__none__",
                   ("healbell", "moonblast", "protect", "splash"),
                   None, "timid", (32, 0, 32, 0, 2, 32)),
              [dummy("magikarp")])
    f.state.sides[0].status[f.state.sides[0].active[0]] = "brn"
    f.state.sides[0].status[1] = "par"
    f.turn(Action.move(0), Action.move(0))
    return (f.status(0) is None and f.state.sides[0].status[1] is None,
            f"active {f.status(0)}, the one on the bench "
            f"{f.state.sides[0].status[1]}")


@check("healpulse", "heals whoever it is aimed at")
def _heal_pulse():
    f = Fight(ours("clefable", "__none__",
                   ("healpulse", "moonblast", "protect", "splash"),
                   None, "timid", (32, 0, 32, 0, 2, 32)),
              [mon("snorlax", "__none__",
                   ("splash", "bodyslam", "rest", "protect"),
                   None, "serious", (32, 0, 32, 0, 2, 0))])
    slot = f.state.sides[1].active[0]
    f.state.sides[1].hp[slot] //= 3
    hurt = f.hp(1)
    f.turn(Action.move(0), Action.move(0))
    return (f.hp(1) > hurt, f"they were on {hurt}, now {f.hp(1)}")


@check("imprison", "seals the moves both sides carry")
def _imprison():
    from pkcm.engine.state import legal_actions
    f = Fight(ours("gengar", "__none__",
                   ("imprison", "shadowball", "protect", "splash")),
              [mon("snorlax", "__none__",
                   ("shadowball", "bodyslam", "rest", "protect"),
                   None, "serious", (32, 0, 32, 0, 2, 0))])
    f.turn(Action.move(0), Action.move(1))
    offered = {one.index for one in legal_actions(f.state, 1)
               if str(one).startswith("move")}
    return (0 not in offered,
            f"Shadow Ball is their move 0, and they are offered "
            f"{sorted(offered)}")


@check("lockon", "the next move cannot miss")
def _lock_on():
    def landed(locked_on):
        hits = 0
        for seed in range(12):
            f = Fight(ours("magikarp", "__none__",
                           ("lockon", "sheercold", "splash", "tackle")),
                      [dummy("snorlax")], seed)
            if locked_on:
                f.turn(Action.move(0), Action.move(0))
            f.turn(Action.move(1), Action.move(0))
            hits += f.said("ohko") or f.said("faint(side=1")
        return hits
    # Sheer Cold is thirty percent; locked on it cannot miss. Reading the
    # volatile proved only that something wrote a word into the state.
    aimed = landed(True)
    blind = landed(False)
    return (aimed == 12 and blind < 12,
            f"locked on it connected {aimed}/12, unaided {blind}/12")


@check("magicroom", "held items stop working")
def _magic_room():
    def healed(room_up):
        f = Fight([PokemonSet(species="snorlax", ability="__none__",
                              gender=None,
                              moves=("magicroom", "splash", "protect", "rest"),
                              item="leftovers", nature="serious",
                              sp=(32, 0, 32, 0, 2, 0))],
                  [dummy("magikarp")])
        slot = f.state.sides[0].active[0]
        f.state.sides[0].hp[slot] //= 2
        before = f.hp(0)
        f.turn(Action.move(0 if room_up else 1), Action.move(0))
        return f.hp(0) - before
    # Leftovers is the cheapest item to watch: it ticks at the end of every
    # turn, so a room that suppresses items shows up as the tick not happening.
    with_room = healed(True)
    without = healed(False)
    return (without > 0 and with_room <= 0,
            f"Leftovers gave back {without} normally and {with_room} "
            f"inside Magic Room")


@check("noretreat", "raises everything and pins the user down")
def _no_retreat():
    from pkcm.engine.state import legal_actions
    f = Fight(ours("falinks", "__none__",
                   ("noretreat", "closecombat", "protect", "splash"),
                   None, "adamant", (32, 32, 2, 0, 0, 0)),
              [dummy("magikarp")])
    f.turn(Action.move(0), Action.move(0))
    got = f.boosts(0)
    switches = [one for one in legal_actions(f.state, 0)
                if str(one).startswith("switch")]
    return (all(got.get(stat) == 1 for stat in
                ("atk", "def", "spa", "spd", "spe")) and not switches,
            f"boosts {got or 'none'}, switches still offered {len(switches)}")


@check("powertrick", "swaps the user's Attack and Defence")
def _power_trick():
    f = Fight(ours("shuckle", "__none__",
                   ("powertrick", "protect", "splash", "tackle"),
                   None, "serious", (32, 0, 32, 0, 32, 0)),
              [dummy("magikarp")])
    before = (_stat(f, 0, "atk"), _stat(f, 0, "def"))
    f.turn(Action.move(0), Action.move(0))
    after = (_stat(f, 0, "atk"), _stat(f, 0, "def"))
    # Shuckle's two are far enough apart that a swap cannot be mistaken for
    # noise, and the volatile alone said only that a word had been written.
    return (before[0] != before[1] and after == (before[1], before[0]),
            f"Attack and Defence were {before}, now {after}")


@check("powertrip", "grows with the user's own stat stages")
def _power_trip():
    def dealt(boosted):
        f = Fight(ours("kingambit", "__none__",
                       ("powertrip", "swordsdance", "protect", "splash"),
                       None, "adamant", (0, 32, 2, 0, 0, 0)),
                  [wall()])
        if boosted:
            f.turn(Action.move(1), Action.move(0))
            f.turn(Action.move(1), Action.move(0))
        f.turn(Action.move(0), Action.move(0))
        return f.damage("powertrip")
    flat = dealt(False)
    stacked = dealt(True)
    return (stacked > flat * 1.5,
            f"with no stages {flat}, after two Swords Dances {stacked}")


def _fails_cleanly(species, moves):
    """A move that needs an ally has nothing to do in singles.

    ``ALLY_ONLY`` lists five that the coverage report already excuses; these
    are the ones that are not on it, so the check is that they refuse rather
    than doing something to the wrong Pokemon.
    """
    f = Fight(ours(species, "__none__", moves), [dummy("magikarp")])
    f.turn(Action.move(0), Action.move(0))
    said = " | ".join(str(e) for e in f.log)
    quiet = ("move_failed" in said or "no target" in said
             or not f.boosts(1) and not f.boosts(0))
    return quiet, f"log: {said[:170]}"


@check("acupressure", "picks one of the user's stats and raises it two stages")
def _acupressure():
    f = Fight(ours("shuckle", "__none__",
                   ("acupressure", "protect", "splash", "tackle")),
              [dummy("magikarp")])
    f.turn(Action.move(0), Action.move(0))
    got = f.boosts(0)
    return (any(value == 2 for value in got.values()),
            f"our boosts {got or 'none'}")


@check("corrosivegas", "melts the target's item away")
def _corrosive_gas():
    f = _until(lambda seed: Fight(
        ours("glimmora", "__none__",
             ("corrosivegas", "sludgewave", "protect", "splash"),
             None, "timid", (0, 0, 2, 32, 0, 32)),
        [mon("snorlax", "__none__",
             ("splash", "bodyslam", "rest", "protect"),
             "leftovers", "serious", (32, 0, 32, 0, 2, 0))], seed)
        .turn(Action.move(0), Action.move(0)),
        lambda f: f.item(1) is None)
    return (f is not None,
            "the item survived ten casts" if f is None else "item removed")


@check("electrify", "the target's move turns Electric")
def _electrify():
    f = Fight(ours("garchomp", "__none__",     # Ground: immune to Electric
                   ("electrify", "earthquake", "protect", "splash"),
                   None, "timid", (32, 0, 32, 0, 2, 32)),
              [mon("snorlax", "__none__",
                   ("bodyslam", "splash", "rest", "protect"),
                   None, "serious", (32, 0, 32, 0, 2, 0))])
    f.turn(Action.move(0), Action.move(0))
    # The marker existing proves nothing -- Encore held one for four turns
    # while enforcing none of it. A Normal move turned Electric cannot touch a
    # Ground type, so the Garchomp behind us is what answers the question.
    return (f.said("immune"),
            f"their Body Slam was refused as Electric={f.said('immune')}; "
            f"volatiles {sorted(f.volatiles(1))}")


@check("entrainment", "hands the user's ability to the target")
def _entrainment():
    from pkcm.engine.battle import make_context
    f = Fight(ours("clefable", "magicguard",
                   ("entrainment", "moonblast", "protect", "splash"),
                   None, "timid", (32, 0, 32, 0, 2, 32)),
              [mon("snorlax", "thickfat",
                   ("splash", "bodyslam", "rest", "protect"),
                   None, "serious", (32, 0, 32, 0, 2, 0))])
    f.turn(Action.move(0), Action.move(0))
    theirs = make_context(f.state).ability_of((1, f.state.sides[1].active[0]))
    return (theirs == "magicguard", f"their ability now reads {theirs!r}")


@check("expandingforce", "hits harder on Psychic Terrain")
def _expanding_force():
    def dealt(terrain):
        f = Fight(ours("indeedee", "__none__",
                       ("expandingforce", "psychic", "protect", "splash"),
                       None, "modest", (0, 0, 2, 32, 0, 32)),
                  [mon("snorlax", "__none__",
                       ("splash", "bodyslam", "rest", "protect"),
                       None, "serious", (32, 0, 32, 0, 2, 0))])
        f.state.field.terrain = terrain
        f.turn(Action.move(0), Action.move(0))
        return f.damage("expandingforce")
    bare = dealt(None)
    on_terrain = dealt("psychicterrain")
    return (on_terrain > bare,
            f"on bare ground {bare}, on Psychic Terrain {on_terrain}")


@check("fairylock", "nobody leaves the turn after")
def _fairy_lock():
    from pkcm.engine.state import legal_actions
    f = Fight(ours("clefable", "__none__",
                   ("fairylock", "moonblast", "protect", "splash"),
                   None, "timid", (32, 0, 32, 0, 2, 32)),
              [dummy("magikarp")])
    f.turn(Action.move(0), Action.move(0))
    switches = [one for one in legal_actions(f.state, 1)
                if str(one).startswith("switch")]
    return (not switches, f"they are still offered {len(switches)} switches")


@check("fellstinger", "the user leaps three stages when it lands the knockout")
def _fell_stinger():
    f = Fight(ours("weavile", "__none__",
                   ("fellstinger", "knockoff", "iceshard", "swordsdance"),
                   None, "adamant", (0, 32, 2, 0, 0, 32)),
              [mon("magikarp", "__none__", ("splash", "tackle", "flail",
                                            "bounce"))])
    # Fifty base power does not finish a healthy target, and no knockout means
    # no boost -- which is the move working, not failing.
    slot = f.state.sides[1].active[0]
    f.state.sides[1].hp[slot] = 1
    f.turn(Action.move(0), Action.move(0))
    return (f.boosts(0).get("atk") == 3,
            f"our boosts {f.boosts(0) or 'none'}, "
            f"they fainted={f.said('faint')}")


@check("forestscurse", "adds Grass to whatever the target already was")
def _forests_curse():
    f = Fight(ours("venusaur", "__none__",
                   ("forestscurse", "sludgebomb", "protect", "splash"),
                   None, "modest", (0, 0, 2, 32, 0, 32)),
              [dummy("snorlax")])
    f.turn(Action.move(0), Action.move(0))
    types = tuple(f.state.types(1, f.state.sides[1].active[0]))
    return ("grass" in types, f"their types are now {types}")


@check("trickortreat", "adds Ghost to whatever the target already was")
def _trick_or_treat():
    f = Fight(ours("gourgeist", "__none__",
                   ("trickortreat", "shadowsneak", "protect", "splash")),
              [dummy("snorlax")])
    f.turn(Action.move(0), Action.move(0))
    types = tuple(f.state.types(1, f.state.sides[1].active[0]))
    return ("ghost" in types, f"their types are now {types}")


@check("magicpowder", "turns the target Psychic")
def _magic_powder():
    f = Fight(ours("indeedee", "__none__",
                   ("magicpowder", "psychic", "protect", "splash"),
                   None, "modest", (0, 0, 2, 32, 0, 32)),
              [dummy("snorlax")])
    f.turn(Action.move(0), Action.move(0))
    types = tuple(f.state.types(1, f.state.sides[1].active[0]))
    return (types == ("psychic",), f"their types are now {types}")


@check("reflecttype", "the user takes the target's typing")
def _reflect_type():
    f = Fight(ours("ditto", "__none__",
                   ("reflecttype", "splash", "tackle", "protect")),
              [dummy("corviknight")])
    f.turn(Action.move(0), Action.move(0))
    ours_now = tuple(f.state.types(0, f.state.sides[0].active[0]))
    theirs = tuple(f.state.types(1, f.state.sides[1].active[0]))
    return (ours_now == theirs, f"we are {ours_now}, they are {theirs}")


@check("guardsplit", "levels the two defences")
def _guard_split():
    f = Fight(ours("shuckle", "__none__",       # famously all defence
                   ("guardsplit", "protect", "splash", "tackle"),
                   None, "serious", (32, 0, 32, 0, 32, 0)),
              [mon("magikarp", "__none__", ("splash", "tackle", "flail",
                                            "bounce"))])
    before = (_stat(f, 0, "def"), _stat(f, 1, "def"))
    f.turn(Action.move(0), Action.move(0))
    after = (_stat(f, 0, "def"), _stat(f, 1, "def"))
    # "It did not say it failed" was the old verdict, which a move that does
    # nothing also passes. Shuckle's Defence against a Magikarp's is the widest
    # gap in the format; splitting it has to close it.
    return (abs(before[0] - before[1]) > 40 and abs(after[0] - after[1]) <= 1,
            f"Defence was {before}, now {after}")


@check("powersplit", "levels the two attacking stats")
def _power_split():
    f = Fight(ours("shuckle", "__none__",
                   ("powersplit", "protect", "splash", "tackle"),
                   None, "serious", (32, 0, 32, 0, 32, 0)),
              [mon("garchomp", "__none__",
                   ("splash", "dragonclaw", "protect", "earthquake"),
                   None, "adamant", (0, 32, 2, 0, 0, 0))])
    before = (_stat(f, 0, "atk"), _stat(f, 1, "atk"))
    f.turn(Action.move(0), Action.move(0))
    after = (_stat(f, 0, "atk"), _stat(f, 1, "atk"))
    return (abs(before[0] - before[1]) > 40 and abs(after[0] - after[1]) <= 1,
            f"Attack was {before}, now {after}")


@check("guardswap", "trades the two defensive stat stages over")
def _guard_swap():
    f = Fight(ours("gengar", "__none__",
                   ("guardswap", "shadowball", "protect", "splash"),
                   None, "timid", (32, 0, 32, 0, 2, 32)),
              [mon("corviknight", "__none__",
                   ("irondefense", "roost", "bodypress", "uturn"),
                   None, "impish", (32, 0, 32, 0, 2, 0))])
    f.turn(Action.move(3), Action.move(0))     # they raise Defence
    theirs = f.boosts(1)
    f.turn(Action.move(0), Action.move(1))
    return (theirs.get("def") == 2 and f.boosts(0).get("def") == 2
            and not f.boosts(1).get("def"),
            f"they had {theirs}, now we have {f.boosts(0)} and they "
            f"have {f.boosts(1) or 'nothing'}")


@check("powerswap", "trades the two offensive stat stages over")
def _power_swap():
    f = Fight(ours("gengar", "__none__",
                   ("powerswap", "shadowball", "protect", "splash"),
                   None, "timid", (32, 0, 32, 0, 2, 32)),
              [mon("dragonite", "__none__",
                   ("dragondance", "dragonclaw", "roost", "firepunch"))])
    f.turn(Action.move(3), Action.move(0))     # they dance: Atk and Spe
    theirs = f.boosts(1)
    f.turn(Action.move(0), Action.move(2))
    return (theirs.get("atk") == 1 and f.boosts(0).get("atk") == 1
            and not f.boosts(1).get("atk"),
            f"they had {theirs}, now we have {f.boosts(0)} and they "
            f"have {f.boosts(1) or 'nothing'}")


@check("speedswap", "trades the two Speed stat stages over")
def _speed_swap():
    f = Fight(ours("gengar", "__none__",
                   ("speedswap", "shadowball", "protect", "splash"),
                   None, "timid", (32, 0, 32, 0, 2, 32)),
              [mon("dragonite", "__none__",
                   ("dragondance", "dragonclaw", "roost", "firepunch"))])
    f.turn(Action.move(3), Action.move(0))
    theirs = f.boosts(1)
    f.turn(Action.move(0), Action.move(2))
    return (theirs.get("spe") == 1 and f.boosts(0).get("spe") == 1,
            f"they had {theirs}, now we have {f.boosts(0)}")


@check("hardpress", "falls off as the target loses HP")
def _hard_press():
    def dealt(hurt_first):
        f = Fight(ours("archaludon", "__none__",
                       ("hardpress", "flashcannon", "protect", "splash"),
                       None, "adamant", (0, 32, 2, 0, 0, 32)),
                  [wall()])
        if hurt_first:
            slot = f.state.sides[1].active[0]
            f.state.sides[1].hp[slot] //= 5
        f.turn(Action.move(0), Action.move(0))
        return f.damage("hardpress")
    healthy = dealt(False)
    hurt = dealt(True)
    return (healthy > hurt * 1.5,
            f"against a healthy target {healthy}, against a hurt one {hurt}")


@check("infernalparade", "doubles against a target that already has a status")
def _infernal_parade():
    def dealt(status):
        f = Fight(ours("gengar", "__none__",
                       ("infernalparade", "shadowball", "protect", "splash"),
                       None, "timid", (0, 0, 2, 32, 0, 32)),
                  [mon("milotic", "__none__",
                       ("splash", "surf", "recover", "protect"),
                       None, "serious", (32, 0, 32, 0, 2, 0))])
        if status:
            f.state.sides[1].status[f.state.sides[1].active[0]] = status
        f.turn(Action.move(0), Action.move(0))
        return f.damage("infernalparade")
    plain = dealt(None)
    on_status = dealt("par")
    return (plain > 0 and on_status > plain * 1.5,
            f"against a healthy target {plain}, against a paralysed one "
            f"{on_status}")


@check("mistyexplosion", "the user goes up with it")
def _misty_explosion():
    f = Fight(ours("glimmora", "__none__",
                   ("mistyexplosion", "sludgewave", "protect", "splash"),
                   None, "modest", (0, 0, 2, 32, 0, 32)),
              [wall()])
    f.turn(Action.move(0), Action.move(0))
    return (f.hp(0) == 0 or f.state.phase.name in ("FORCED_SWITCH", "FINISHED"),
            f"we are on {f.hp(0)}, phase {f.state.phase.name}")


@check("quickguard", "turns a priority move away")
def _quick_guard():
    f = Fight(ours("falinks", "__none__",
                   ("quickguard", "closecombat", "protect", "splash"),
                   None, "adamant", (32, 32, 32, 0, 2, 0)),
              [mon("weavile", "__none__",
                   ("iceshard", "knockoff", "splash", "uturn"),
                   None, "jolly", (0, 32, 2, 0, 0, 32))])
    f.turn(Action.move(0), Action.move(0))
    return (f.hp(0) == f.max_hp(0),
            f"we are on {f.hp(0)}/{f.max_hp(0)} after their Ice Shard")


@check("recycle", "gets the spent item back")
def _recycle():
    f = Fight(ours("snorlax", "__none__",
                   ("recycle", "bodyslam", "splash", "protect"),
                   "sitrusberry", "serious", (32, 0, 32, 0, 2, 0)),
              [mon("garchomp", "__none__",
                   ("earthquake", "dragonclaw", "firefang", "stoneedge"),
                   None, "jolly", (0, 32, 2, 0, 0, 32))])
    for _ in range(4):
        if f.item(0) is None:
            break
        f.turn(Action.move(2), Action.move(0))     # take hits until it is eaten
    eaten = f.item(0) is None
    f.turn(Action.move(0), Action.move(3))
    # Not "we are holding it now": we are under half health, which is what ate
    # it in the first place, so a working Recycle hands it back and the berry
    # goes straight down again. The event is the thing to read.
    return (eaten and f.said("item_restored"),
            f"eaten={eaten}, restored={f.said('item_restored')}, "
            f"holding {f.item(0)}")


@check("helpinghand", "has no ally to help in singles")
def _helping_hand():
    return _fails_cleanly("clefable", ("helpinghand", "moonblast", "protect",
                                       "splash"))


@check("instruct", "has no ally to instruct in singles")
def _instruct():
    return _fails_cleanly("indeedee", ("instruct", "psychic", "protect",
                                       "splash"))


@check("magneticflux", "has no ally to flux in singles")
def _magnetic_flux():
    return _fails_cleanly("archaludon", ("magneticflux", "flashcannon",
                                         "protect", "splash"))


@check("dragoncheer", "has no ally to cheer in singles")
def _dragon_cheer():
    return _fails_cleanly("dragonite", ("dragoncheer", "dragonclaw", "roost",
                                        "firepunch"))


def _stat(f, side, stat):
    """The stat the engine would actually use, stages and all."""
    from pkcm.data.dex import Stat
    from pkcm.engine.battle import make_context
    from pkcm.engine.mutate import effective_stat
    ref = (side, f.state.sides[side].active[0])
    return effective_stat(make_context(f.state), ref, getattr(Stat, stat.upper()))


def _ability_after(f, side):
    from pkcm.engine.battle import make_context
    return make_context(f.state).ability_of((side, f.state.sides[side].active[0]))


@check("reversal", "hits hardest when the user is nearly gone")
def _reversal():
    def dealt(fraction):
        f = Fight(ours("machamp", "__none__",
                       ("reversal", "closecombat", "knockoff", "bulkup"),
                       None, "adamant", (32, 32, 2, 0, 0, 32)),
                  [wall()])
        slot = f.state.sides[0].active[0]
        f.state.sides[0].hp[slot] = max(1, f.state.sides[0].hp[slot] // fraction)
        f.turn(Action.move(0), Action.move(0))
        return f.damage("reversal")
    healthy = dealt(1)
    dying = dealt(20)
    return (dying > healthy,
            f"at full health {healthy}, at a twentieth {dying}")


@check("risingvoltage", "doubles on Electric Terrain against a grounded target")
def _rising_voltage():
    def dealt(terrain):
        f = Fight(ours("pikachu", "__none__",
                       ("risingvoltage", "thunderbolt", "protect", "splash"),
                       None, "modest", (0, 0, 2, 32, 0, 32)),
                  [mon("snorlax", "__none__",   # grounded, and not immune
                       ("splash", "bodyslam", "rest", "protect"),
                       None, "serious", (32, 0, 32, 0, 2, 0))])
        f.state.field.terrain = terrain
        f.turn(Action.move(0), Action.move(0))
        return f.damage("risingvoltage")
    bare = dealt(None)
    charged = dealt("electricterrain")
    return (charged > bare * 1.5,
            f"on bare ground {bare}, on Electric Terrain {charged}")


@check("roleplay", "the user takes the target's ability")
def _role_play():
    f = Fight(ours("alakazam", "magicguard",
                   ("roleplay", "psychic", "protect", "splash"),
                   None, "timid", (0, 0, 2, 32, 0, 32)),
              [mon("snorlax", "thickfat",
                   ("splash", "bodyslam", "rest", "protect"),
                   None, "serious", (32, 0, 32, 0, 2, 0))])
    f.turn(Action.move(0), Action.move(0))
    return (_ability_after(f, 0) == "thickfat",
            f"our ability now reads {_ability_after(f, 0)!r}")


@check("simplebeam", "the target's ability becomes Simple")
def _simple_beam():
    f = Fight(ours("clefable", "__none__",
                   ("simplebeam", "moonblast", "protect", "splash"),
                   None, "timid", (32, 0, 32, 0, 2, 32)),
              [mon("snorlax", "thickfat",
                   ("splash", "bodyslam", "rest", "protect"),
                   None, "serious", (32, 0, 32, 0, 2, 0))])
    f.turn(Action.move(0), Action.move(0))
    return (_ability_after(f, 1) == "simple",
            f"their ability now reads {_ability_after(f, 1)!r}")


@check("worryseed", "the target's ability becomes Insomnia")
def _worry_seed():
    f = Fight(ours("venusaur", "__none__",
                   ("worryseed", "sludgebomb", "protect", "splash"),
                   None, "modest", (0, 0, 2, 32, 0, 32)),
              [mon("snorlax", "thickfat",
                   ("splash", "bodyslam", "rest", "protect"),
                   None, "serious", (32, 0, 32, 0, 2, 0))])
    f.turn(Action.move(0), Action.move(0))
    return (_ability_after(f, 1) == "insomnia",
            f"their ability now reads {_ability_after(f, 1)!r}")


@check("saltcure", "keeps taking HP off after it lands")
def _salt_cure():
    f = _until(lambda seed: Fight(
        ours("garganacl", "__none__",
             ("saltcure", "rockslide", "recover", "protect"),
             None, "adamant", (32, 32, 2, 0, 0, 0)),
        [mon("snorlax", "__none__",
             ("splash", "bodyslam", "rest", "protect"),
             None, "serious", (32, 0, 32, 0, 2, 0))], seed)
        .turn(Action.move(0), Action.move(0)),
        lambda f: "saltcure" in f.volatiles(1))
    if f is None:
        return False, "the volatile never appeared in ten tries"
    theirs = f.hp(1)
    f.turn(Action.move(3), Action.move(0))     # Protect, so only the cure acts
    return (f.hp(1) < theirs, f"they were on {theirs}, now {f.hp(1)}")


@check("sleeptalk", "picks one of the other moves while the user sleeps")
def _sleep_talk():
    f = Fight(ours("snorlax", "__none__",
                   ("sleeptalk", "bodyslam", "rest", "protect"),
                   None, "adamant", (32, 32, 32, 0, 2, 0)),
              [wall()])
    f.state.sides[0].status[f.state.sides[0].active[0]] = "slp"
    f.state.sides[0].status_data[f.state.sides[0].active[0]]["turns"] = 3
    f.turn(Action.move(0), Action.move(0))
    said = " | ".join(str(e) for e in f.log)
    return ("bodyslam" in said or "rest" in said or "protect" in said,
            f"log {said[:200]}")


@check("solarblade", "charges a turn first, unless the sun is out")
def _solar_blade():
    f = Fight(ours("venusaur", "__none__",
                   ("solarblade", "sludgebomb", "sunnyday", "protect"),
                   None, "adamant", (0, 32, 2, 0, 0, 32)),
              [wall()])
    f.turn(Action.move(0), Action.move(0))
    charged = f.damage("solarblade") == 0
    f.turn(Action.move(0), Action.move(0))
    fired = f.damage("solarblade") > 0
    return (charged and fired,
            f"turn one silent={charged}, turn two fired={fired}")


@check("spite", "takes PP off the move the target last used")
def _spite():
    f = Fight(ours("gengar", "__none__",
                   ("spite", "shadowball", "protect", "splash"),
                   None, "timid", (32, 0, 32, 0, 2, 32)),
              [mon("snorlax", "__none__",
                   ("bodyslam", "splash", "rest", "protect"),
                   None, "serious", (32, 0, 32, 0, 2, 0))])
    f.turn(Action.move(3), Action.move(0))     # they show us Body Slam
    before = f.state.sides[1].pp[f.state.sides[1].active[0]][0]
    f.turn(Action.move(0), Action.move(1))
    after = f.state.sides[1].pp[f.state.sides[1].active[0]][0]
    return (after < before, f"their Body Slam had {before} PP, now {after}")


@check("steelroller", "sweeps the terrain, and refuses when there is none")
def _steel_roller():
    f = Fight(ours("archaludon", "__none__",
                   ("steelroller", "flashcannon", "protect", "splash"),
                   None, "adamant", (0, 32, 2, 0, 0, 32)),
              [wall()])
    f.turn(Action.move(0), Action.move(0))
    refused = f.said("move_failed")
    g = Fight(ours("archaludon", "__none__",
                   ("steelroller", "flashcannon", "protect", "splash"),
                   None, "adamant", (0, 32, 2, 0, 0, 32)),
              [wall()])
    g.state.field.terrain = "electricterrain"
    g.turn(Action.move(0), Action.move(0))
    return (refused and g.state.field.terrain is None,
            f"on bare ground it refused={refused}; on terrain the terrain "
            f"afterwards is {g.state.field.terrain!r}")


@check("stockpile", "stacks layers and raises both defences")
def _stockpile():
    f = Fight(ours("snorlax", "__none__",
                   ("stockpile", "swallow", "splash", "protect"),
                   None, "serious", (32, 0, 32, 0, 2, 0)),
              [dummy("magikarp")])
    f.turn(Action.move(0), Action.move(0))
    f.turn(Action.move(0), Action.move(0))
    got = f.boosts(0)
    return ("stockpile" in f.volatiles(0) and got.get("def") == 2
            and got.get("spd") == 2,
            f"volatiles {sorted(f.volatiles(0))}, boosts {got or 'none'}")


@check("swallow", "spends the layers to heal")
def _swallow():
    f = Fight(ours("snorlax", "__none__",
                   ("stockpile", "swallow", "splash", "protect"),
                   None, "serious", (32, 0, 32, 0, 2, 0)),
              # Splash on their fourth slot: leaving Stone Edge there meant the
              # foe took more off on the healing turn than Swallow put back,
              # and the check was reading the difference of the two.
              [mon("garchomp", "__none__",
                   ("earthquake", "dragonclaw", "firefang", "splash"),
                   None, "jolly", (0, 32, 2, 0, 0, 32))])
    f.turn(Action.move(2), Action.move(0))     # take a hit to leave room
    f.turn(Action.move(0), Action.move(3))     # one layer
    hurt = f.hp(0)
    f.turn(Action.move(1), Action.move(3))
    return (f.hp(0) > hurt and "stockpile" not in f.volatiles(0),
            f"{hurt} -> {f.hp(0)}, layers left "
            f"{'stockpile' in f.volatiles(0)}")


@check("stompingtantrum", "doubles when the user's last move failed")
def _stomping_tantrum():
    def dealt(fail_first):
        f = Fight(ours("garchomp", "__none__",
                       ("stompingtantrum", "dragonclaw", "protect", "splash"),
                       None, "adamant", (0, 32, 2, 0, 0, 32)),
                  [mon("snorlax", "__none__",
                       ("protect", "splash", "bodyslam", "rest"),
                       None, "serious", (32, 0, 32, 0, 2, 0))])
        if fail_first:
            f.turn(Action.move(1), Action.move(0))   # they Protect: ours fails
        f.turn(Action.move(0), Action.move(1))
        return f.damage("stompingtantrum")
    clean = dealt(False)
    after_a_failure = dealt(True)
    return (after_a_failure > clean * 1.5,
            f"after a clean turn {clean}, after a failure {after_a_failure}")


@check("stuffcheeks", "eats the held berry and puts the Defence up")
def _stuff_cheeks():
    f = Fight(ours("snorlax", "__none__",
                   ("stuffcheeks", "bodyslam", "rest", "protect"),
                   "sitrusberry", "serious", (32, 0, 32, 0, 2, 0)),
              [dummy("magikarp")])
    f.turn(Action.move(0), Action.move(0))
    return (f.item(0) is None and f.boosts(0).get("def") == 2,
            f"holding {f.item(0)}, boosts {f.boosts(0) or 'none'}")


@check("teatime", "everyone holding a berry eats it")
def _teatime():
    f = Fight(ours("clefable", "__none__",
                   ("teatime", "moonblast", "protect", "splash"),
                   "sitrusberry", "timid", (32, 0, 32, 0, 2, 32)),
              [mon("snorlax", "__none__",
                   ("splash", "bodyslam", "rest", "protect"),
                   "sitrusberry", "serious", (32, 0, 32, 0, 2, 0))])
    f.turn(Action.move(0), Action.move(0))
    return (f.item(0) is None and f.item(1) is None,
            f"we hold {f.item(0)}, they hold {f.item(1)}")


@check("syrupbomb", "keeps taking the target's Speed down")
def _syrup_bomb():
    f = _until(lambda seed: Fight(
        ours("dipplin", "__none__",
             ("syrupbomb", "dragonpulse", "recover", "protect"),
             None, "modest", (32, 0, 32, 32, 0, 0)),
        [mon("snorlax", "__none__",
             ("splash", "bodyslam", "rest", "protect"),
             None, "serious", (32, 0, 32, 0, 2, 0))], seed)
        .turn(Action.move(0), Action.move(0)),
        lambda f: "syrupbomb" in f.volatiles(1))
    if f is None:
        return False, "the volatile never appeared in ten tries"
    f.turn(Action.move(3), Action.move(0))
    return (f.boosts(1).get("spe", 0) < 0,
            f"their boosts {f.boosts(1) or 'none'}")


@check("terrainpulse", "changes type and doubles on terrain")
def _terrain_pulse():
    def dealt(terrain):
        f = Fight(ours("indeedee", "__none__",
                       ("terrainpulse", "psychic", "protect", "splash"),
                       None, "modest", (0, 0, 2, 32, 0, 32)),
                  [mon("snorlax", "__none__",
                       ("splash", "bodyslam", "rest", "protect"),
                       None, "serious", (32, 0, 32, 0, 2, 0))])
        f.state.field.terrain = terrain
        f.turn(Action.move(0), Action.move(0))
        return f.damage("terrainpulse")
    bare = dealt(None)
    on_terrain = dealt("psychicterrain")
    return (on_terrain > bare,
            f"on bare ground {bare}, on Psychic Terrain {on_terrain}")


@check("topsyturvy", "turns the target's stat changes upside down")
def _topsy_turvy():
    f = Fight(ours("gengar", "__none__",
                   ("topsyturvy", "shadowball", "protect", "splash"),
                   None, "timid", (32, 0, 32, 0, 2, 32)),
              [mon("dragonite", "__none__",
                   ("dragondance", "dragonclaw", "roost", "firepunch"))])
    f.turn(Action.move(3), Action.move(0))     # they dance
    theirs = f.boosts(1)
    f.turn(Action.move(0), Action.move(2))
    return (theirs.get("atk") == 1 and f.boosts(1).get("atk") == -1,
            f"they had {theirs}, now {f.boosts(1) or 'nothing'}")


@check("torment", "the target cannot use the same move twice running")
def _torment():
    from pkcm.engine.state import legal_actions
    f = Fight(ours("gengar", "__none__",
                   ("torment", "shadowball", "protect", "splash"),
                   None, "timid", (32, 0, 32, 0, 2, 32)),
              [mon("snorlax", "__none__",
                   ("bodyslam", "splash", "rest", "protect"),
                   None, "serious", (32, 0, 32, 0, 2, 0))])
    f.turn(Action.move(0), Action.move(0))     # tormented, having used Body Slam
    landed = "torment" in f.volatiles(1)
    f.turn(Action.move(2), Action.move(0))     # they try Body Slam again
    return (landed and f.said("cant_move", "torment"),
            f"volatile={landed}, refused={f.said('cant_move', 'torment')}")


@check("uproar", "keeps going by itself and keeps everyone awake")
def _uproar():
    from pkcm.engine.state import legal_actions
    f = Fight(ours("snorlax", "__none__",
                   ("uproar", "bodyslam", "rest", "protect"),
                   None, "adamant", (32, 32, 32, 0, 2, 0)),
              [wall()])
    f.turn(Action.move(0), Action.move(0))
    going = "uproar" in f.volatiles(0)
    offered = {one.index for one in legal_actions(f.state, 0)
               if str(one).startswith("move")}
    return (going and offered == {0},
            f"volatile={going}, and we are offered {sorted(offered)}")


@check("wonderroom", "the two defences trade places")
def _wonder_room():
    f = Fight(ours("clefable", "__none__",
                   ("wonderroom", "moonblast", "protect", "splash"),
                   None, "timid", (32, 0, 32, 0, 2, 32)),
              [dummy("magikarp")])
    f.turn(Action.move(0), Action.move(0))
    return ("wonderroom" in f.state.field.rooms,
            f"the rooms up are {sorted(f.state.field.rooms) or 'none'}")


# -- the flag-driven moves, generated ---------------------------------------- #
#
# The count that mattered turned out not to be 497 or 136. Of the format's 497
# standard moves, 220 declare their effect in data and ``test_effect_sweep``
# casts every one of those; 136 have effects that are code and are checked by
# hand above; 75 are plain damage and have nothing to check past the formula.
# That leaves 66 whose behaviour hangs off a *flag* -- drain, recoil, multihit,
# charge, OHKO, a switch, a crit ratio -- and the sweep does not cast them,
# because they declare no effect for it to diff against. Sucker Punch and Fake
# Out were exactly that shape.
#
# These are generated rather than written out, one per move, because the
# question is the same each time and a family with sixty members is a family
# where a hand-written scenario would go stale silently. Each still plays a
# real battle and each asserts the consequence, not the flag.

#: Bodies to aim at, in order of preference; the first the move can touch wins.
_TARGETS = ("snorlax", "milotic", "garchomp", "archaludon", "dragonite",
            "clefable", "blissey")


def _reachable(move):
    for species in _TARGETS:
        if DEX.type_chart.multiplier(move.type, DEX.species[species].types) > 0:
            return species
    return "snorlax"


def _swinger(move, moves):
    """One attacker, invested on whichever side the move swings from."""
    sp = (32, 32, 0, 0, 2, 0) if move.category == "Physical" else (32, 0, 0, 32, 2, 0)
    nature = "adamant" if move.category == "Physical" else "modest"
    return mon("mew" if "mew" in DEX.species else "snorlax", "__none__",
               moves, None, nature, sp)


#: Positions the number probes cannot reach from a standing start. Every
#: entry below was an "[inconclusive]" the report used to count as a pass:
#: Burn Up on something that is not a Fire type simply fails, Snore needs the
#: user asleep, Sucker Punch needs the target swinging, and a 90% move that
#: never missed in twenty tries never tested its crash damage.
_USER_SPECIES = {"burnup": "charizard"}

#: What the target holds, for the moves that need it to do something specific.
_TARGET_MOVES = {"upperhand": ("splash", "aquajet", "protect", "rest")}

#: Poltergeist refuses a target with empty hands, so the probe has to fill them.
_TARGET_ITEM = {"poltergeist": "leftovers"}

#: Which of the target's moves it uses while the probe casts.
_TARGET_MOVE = {"suckerpunch": 1, "upperhand": 1}

#: Moves the user spends first: Last Resort refuses until the rest are gone,
#: and a charge move spends the first turn charging.
_LEAD_MOVES = {"lastresort": (1, 2, 3)}


def _evasive(f):
    """Six stages of evasion, so an accurate move can actually miss."""
    side = f.state.sides[1]
    side.boosts[side.active[0]][BOOST_INDEX["evasion"]] = 6


def _asleep(f):
    side = f.state.sides[0]
    side.status[side.active[0]] = "slp"


def _on_terrain(f):
    f.state.field.terrain = "electricterrain"
    f.state.field.terrain_turns = 8


_SETUP = {
    "axekick": _evasive, "highjumpkick": _evasive, "supercellslam": _evasive,
    "snore": _asleep, "steelroller": _on_terrain,
}


def _bout(move, extra=("splash", "tackle", "protect"), target=None, seed=7):
    them = mon(target or _reachable(move), "__none__",
               _TARGET_MOVES.get(move.id, ("splash", "tackle", "protect", "rest")),
               _TARGET_ITEM.get(move.id), "serious", (32, 0, 32, 0, 2, 0))
    user = _swinger(move, (move.id,) + extra)
    species = _USER_SPECIES.get(move.id)
    if species is not None:
        user = mon(species, user.ability, user.moves, user.item, user.nature,
                   user.sp)
    f = Fight([user], [them], seed)
    _set_up_field(f, move)
    setup = _SETUP.get(move.id)
    if setup is not None:
        setup(f)
    return f


def _staged(move, seed=7, extra=("splash", "tackle", "protect")):
    """A fight with the move cast once, from whatever position it needs."""
    f = _bout(move, extra=extra, seed=seed)
    theirs = _TARGET_MOVE.get(move.id, 0)
    for index in _LEAD_MOVES.get(move.id, ()):
        f.turn(Action.move(index), Action.move(theirs))
    if "charge" in move.flags:
        f.turn(Action.move(0), Action.move(theirs))   # the turn it spends
    f.turn(Action.move(0), Action.move(theirs))
    return f


def _hits(f, move_id, side=1):
    return [e for e in f.log
            if str(e).startswith(f"damage(side={side}") and move_id in str(e)]


def _flag_families(move):
    """Every flag family this move belongs to, with how to see each one."""
    raw, found = move.raw, []

    if raw.get("drain"):
        def drain(m=move):
            f = _bout(m)
            slot = f.state.sides[0].active[0]
            f.state.sides[0].hp[slot] //= 2
            before = f.hp(0)
            f.turn(Action.move(0), Action.move(0))
            return (f.hp(0) > before if _hits(f, m.id) else None,
                    f"landed, and the user went {before} -> {f.hp(0)}")
        found.append(("drains back into the user", drain))

    if raw.get("recoil") or raw.get("mindBlownRecoil"):
        def recoil(m=move):
            f = _bout(m)
            before = f.hp(0)
            f.turn(Action.move(0), Action.move(0))
            return (f.hp(0) < before if _hits(f, m.id) else None,
                    f"landed, and the user went {before} -> {f.hp(0)}")
        found.append(("costs the user HP", recoil))

    if raw.get("multihit"):
        def multihit(m=move):
            f = _bout(m)
            f.turn(Action.move(0), Action.move(0))
            hits = _hits(f, m.id)
            return (len(hits) >= 2 if hits else None,
                    f"{len(hits)} separate hits landed")
        found.append(("lands more than once", multihit))

    if "charge" in move.flags:
        def charge(m=move):
            f = _bout(m)
            f.turn(Action.move(0), Action.move(0))
            first = bool(_hits(f, m.id))
            f.turn(Action.move(0), Action.move(0))
            second = bool(_hits(f, m.id))
            return (not first and second,
                    f"damage on the charging turn={first}, on the next={second}")
        found.append(("spends a turn charging, then strikes", charge))

    if raw.get("ohko"):
        def ohko(m=move):
            partial = 0
            for seed in range(12):
                f = _bout(m, seed=seed)
                f.turn(Action.move(0), Action.move(0))
                if _hits(f, m.id) and f.hp(1) > 0:
                    partial += 1
            return (partial == 0,
                    f"connected without finishing the target {partial}/12 times")
        found.append(("either takes the target out or does nothing", ohko))

    if raw.get("selfSwitch"):
        def self_switch(m=move):
            f = _bout(m)
            f.turn(Action.move(0), Action.move(0))
            return (f.state.phase.name == "MID_TURN_SWITCH",
                    f"phase after using it: {f.state.phase.name}")
        found.append(("the user leaves the field", self_switch))

    if raw.get("forceSwitch"):
        def force_switch(m=move):
            f = _bout(m)
            before = f.active_species(1)
            f.turn(Action.move(0), Action.move(0))
            return (f.active_species(1) != before,
                    f"{before} -> {f.active_species(1)}")
        found.append(("drags the target out", force_switch))

    if raw.get("willCrit"):
        def will_crit(m=move):
            landed = crits = 0
            for seed in range(10):
                f = _bout(m, seed=seed)
                f.turn(Action.move(0), Action.move(0))
                for hit in _hits(f, m.id):
                    landed += 1
                    crits += "crit=True" in str(hit)
            return (landed > 0 and crits == landed,
                    f"{crits} of {landed} landed hits were critical")
        found.append(("every hit is critical", will_crit))

    if raw.get("critRatio"):
        def crit_ratio(m=move):
            # Sampling 1/8 against 1/24 needs hundreds of games to separate,
            # and sixty gave four against two -- true, and not evidence.
            # Focus Energy adds two stages, so a move that really is at stage
            # two lands on stage four, which is *every hit*; a move whose ratio
            # is being ignored sits at stage three, which is half of them. Two
            # arms of twelve settle it.
            landed = crits = 0
            for seed in range(12):
                f = Fight([_swinger(m, (m.id, "focusenergy", "tackle",
                                        "protect"))],
                          [mon(_reachable(m), "__none__",
                               ("splash", "tackle", "protect", "rest"),
                               None, "serious", (32, 0, 32, 0, 2, 0))], seed)
                f.turn(Action.move(1), Action.move(0))     # Focus Energy
                f.turn(Action.move(0), Action.move(0))
                if "charge" in m.flags:
                    # Sky Attack spends the first of those turns in the air,
                    # so there is nothing to count until the one after.
                    f.turn(Action.move(0), Action.move(0))
                for hit in _hits(f, m.id):
                    landed += 1
                    crits += "crit=True" in str(hit)
            return (landed > 0 and crits == landed,
                    f"under Focus Energy {crits} of {landed} landed hits were "
                    f"critical, and a stage-two move should be all of them")
        found.append(("crits more often than an ordinary move", crit_ratio))

    if raw.get("breaksProtect"):
        def breaks_protect(m=move):
            f = _bout(m)
            if "charge" in m.flags:
                # Phantom Force spends a turn underground first, so the
                # Protect has to be on the turn it comes back up.
                f.turn(Action.move(0), Action.move(0))
            f.turn(Action.move(0), Action.move(2))     # they Protect
            return (bool(_hits(f, m.id)),
                    f"log {' | '.join(str(e) for e in f.log)[:160]}")
        found.append(("goes through Protect", breaks_protect))

    if raw.get("thawsTarget"):
        def thaws(m=move):
            f = _bout(m)
            f.state.sides[1].status[f.state.sides[1].active[0]] = "frz"
            f.turn(Action.move(0), Action.move(0))
            return (f.status(1) != "frz" if _hits(f, m.id) else None,
                    f"landed, and their status is now {f.status(1)}")
        found.append(("thaws whatever it hits", thaws))

    if raw.get("hasCrashDamage"):
        def crash(m=move):
            for seed in range(20):
                f = _bout(m, seed=seed)
                before = f.hp(0)
                f.turn(Action.move(0), Action.move(0))
                if f.said("missed") and m.id in str(f.log):
                    return (f.hp(0) < before,
                            f"missed, and the user went {before} -> {f.hp(0)}")
            return None, "never missed in twenty tries, so the crash never came up"
        found.append(("costs the user HP when it misses", crash))

    return found


def _register_flag_checks():
    from pkcm.engine.moveeffects import SPECIAL_MOVES
    from pkcm.engine.moves import VARIABLE_POWER

    for move in DEX.moves.values():
        if not DEX.exists_in_champions(move) or move.id in CHECKS:
            continue
        if move.id in SPECIAL_MOVES or move.id in VARIABLE_POWER:
            continue
        families = _flag_families(move)
        if not families:
            continue

        def run(families=families):
            notes, verdict = [], True
            for what, probe in families:
                ok, detail = probe()
                # ``None`` is "the scenario never got to ask" -- a move that
                # missed every seed, say. Reported, not counted as a pass.
                notes.append(f"{what}: {detail}")
                if ok is False:
                    verdict = False
                elif ok is None:
                    notes[-1] += "  [inconclusive]"
            return verdict, "; ".join(notes)

        CHECKS[move.id] = (" and ".join(what for what, _ in families), run)


_register_flag_checks()


# -- everything else: the numbers, not only the behaviour -------------------- #
#
# The buckets above answer "does it do the thing". They do not answer "does it
# do it as often as the data says, for as long as the data says, and for as
# much". Forcing every chance roll to succeed, which is what the effect sweep
# does, proves the branch exists and says nothing about the number in front of
# it. So every standard move gets its numbers checked too:
#
#   power/type/category   the damage is one of the sixteen rolls the formula
#                         gives for the declared power, type and category
#   secondary chance      the observed rate over 200 casts sits inside a
#                         binomial interval around the declared chance
#   duration              the condition ends on the turn the data says

_ROLL_LOW, _ROLL_HIGH = 85, 100
VARIABLE_POWER_IDS: set = set()


def _expected_rolls(f, move, side=0):
    """The sixteen damages the formula gives, computed outside the engine."""
    from pkcm.data.dex import Stat
    from pkcm.engine.battle import make_context
    from pkcm.engine.moves import damage_base, damage_from_base
    from pkcm.engine.mutate import effective_stat

    ctx = make_context(f.state)
    us = (side, f.state.sides[side].active[0])
    them = (1 - side, f.state.sides[1 - side].active[0])
    physical = move.category == "Physical"
    raw = move.raw
    # Body Press swings with Defence, Foul Play with the target's Attack,
    # Psyshock lands on Defence though it is Special. Computing the plain pair
    # made all three read as wrong numbers from a right engine.
    offensive = raw.get("overrideOffensiveStat")
    striker = them if raw.get("overrideOffensivePokemon") == "target" else us
    attack = effective_stat(
        ctx, striker,
        getattr(Stat, offensive.upper()) if offensive
        else (Stat.ATK if physical else Stat.SPA))
    defensive = raw.get("overrideDefensiveStat")
    defense = effective_stat(
        ctx, them,
        getattr(Stat, defensive.upper()) if defensive
        else (Stat.DEF if physical else Stat.SPD))
    effectiveness = DEX.type_chart.multiplier(move.type, f.state.types(*them))
    stab = move.type in f.state.types(*us)
    # A move that always crits is always the crit number, and comparing it
    # with the ordinary one is how Flower Trick, Frost Breath and Storm Throw
    # looked wrong from a right engine.
    base = damage_base(power=move.base_power, attack=attack, defense=defense,
                       crit=bool(raw.get("willCrit")))
    rolls = {max(1, damage_from_base(base, roll, stab=stab,
                                     effectiveness=effectiveness))
             for roll in range(_ROLL_LOW, _ROLL_HIGH + 1)}
    return rolls, effectiveness


def _damage_probe(move):
    def probe(m=move):
        for seed in range(24):
            f = _bout(m, seed=seed)
            expected, effectiveness = _expected_rolls(f, m)
            if effectiveness == 0:
                continue
            f = _staged(m, seed=seed)
            hits = _hits(f, m.id)
            # A move that always crits has no uncritical hit to wait for, and
            # waiting for one is how Flower Trick, Frost Breath and Storm
            # Throw went unmeasured.
            crit_ok = bool(m.raw.get("willCrit"))
            if not hits or ("crit=True" in str(hits[0]) and not crit_ok):
                continue
            got = int(re.search(r"amount=(\d+)", str(hits[0])).group(1))
            return (got in expected,
                    f"dealt {got}; {m.base_power} power at x{effectiveness} "
                    f"gives {min(expected)}-{max(expected)}")
        return None, "never landed a clean hit in twenty-four tries"
    return ("deals what its declared power, type and category say", probe)


def _fired(f, block, move=None):
    """Whether the *declared* secondary happened -- not whether anything did.

    Comparing whole states counted ``lastmove`` and ``movesused``, which
    change every turn, so every secondary read as firing a hundred percent of
    the time.
    """
    from pkcm.engine.moves import RANDOM_SECONDARY_STATUS

    if block.get("status"):
        return f.status(1) == block["status"]
    # Dire Claw and Tri Attack export a bare chance and pick the status in
    # code, so "what the data declares" is a set rather than one name.
    if move is not None and move.id in RANDOM_SECONDARY_STATUS:
        return f.status(1) in RANDOM_SECONDARY_STATUS[move.id]
    if block.get("volatileStatus") == "flinch":
        # Applied and spent inside the turn: by the time this reads the state
        # the volatile has been cleared, so every flinch read as never firing.
        return f.said("flinch")
    if block.get("volatileStatus"):
        return block["volatileStatus"] in f.volatiles(1)
    if block.get("boosts"):
        got = f.boosts(1)
        return any(got.get(stat) for stat in block["boosts"])
    inner = block.get("self")
    if isinstance(inner, dict) and inner.get("boosts"):
        got = f.boosts(0)
        return any(got.get(stat) for stat in inner["boosts"])
    return None


def _chance_probe(move, chance, what, block):
    def probe(m=move, chance=chance, what=what, block=block):
        landed = fired = 0
        for seed in range(200):
            f = _staged(m, seed=seed)
            if m.category != "Status" and not _hits(f, m.id):
                continue
            saw = _fired(f, block, m)
            if saw is None:
                return None, "the secondary declares nothing this can watch"
            landed += 1
            fired += saw
        if landed < 40:
            return None, f"only landed {landed} times in 200, too few to rate"
        rate = fired / landed
        want = chance / 100
        # Three standard errors, floored at five points so a 10% secondary is
        # not failed for coming in at seven.
        slack = max(0.05, 3 * (want * (1 - want) / landed) ** 0.5)
        return (abs(rate - want) <= slack,
                f"{what} fired {fired} of {landed} landings ({rate:.0%}) "
                f"against the {chance}% in the data")
    return (f"fires its {chance}% {what} at about that rate", probe)


#: What the field must already look like for a move to work at all. Aurora
#: Veil fails outside snow, and a probe that cannot cast the move at all comes
#: back "inconclusive" -- which counts as a pass.
_NEEDS_FIELD = {"auroraveil": ("snowscape", 12)}


def _set_up_field(f, move):
    weather = _NEEDS_FIELD.get(move.id)
    if weather is not None:
        f.state.field.weather, f.state.field.weather_turns = weather


def _duration_probe(move, name, kind, turns):
    def probe(m=move, name=name, kind=kind, turns=turns):
        f = _bout(m)
        _set_up_field(f, m)
        f.turn(Action.move(0), Action.move(0))

        def alive():
            if kind == "side":
                return name in f.conditions(0) or name in f.conditions(1)
            if kind == "room":
                return name in f.state.field.rooms
            if kind == "weather":
                return f.state.field.weather == name
            return f.state.field.terrain == name

        if not alive():
            if turns == 1:
                return (True, f"{name} lasts the turn it is cast and is gone "
                              f"by the end of it, which is the one turn the "
                              f"data gives it")
            return None, f"{name} was not up after the cast"
        # The turn it was cast on counts: a five-turn screen is up for the
        # turn it goes up and the four after, and the tick at the end of the
        # fifth is what takes it down. Counting only the turns it survived to
        # the end of read one short for every condition in the format.
        lasted = 1
        for _ in range(turns + 3):
            if f.state.phase.name == "MID_TURN_SWITCH":
                # Chilly Reception sets the weather and leaves; the side owes a
                # replacement before anything else can be submitted.
                f.turn(Action.switch(1), Action.PASS)
            from pkcm.engine.state import legal_actions
            # Anything but slot zero, which is the move under test: taking the
            # first legal move re-cast Trick Room every turn and it never
            # looked like ending.
            spare = next((one.index for one in legal_actions(f.state, 0)
                          if str(one).startswith("move") and one.index != 0),
                         None)
            if spare is None:
                spare = next((one.index for one in legal_actions(f.state, 0)
                              if str(one).startswith("move")), 0)
            f.turn(Action.move(spare), Action.move(0))
            lasted += 1
            if not alive():
                break
        return (lasted == turns,
                f"{name} stood {lasted} turns against the {turns} in the data")
    return (f"its {name} lasts the {turns} turns the data gives it", probe)


#: Durations the move itself does not carry, taken from the conditions file.
_DURATIONS = {"reflect": 5, "lightscreen": 5, "auroraveil": 5, "tailwind": 4,
              "safeguard": 5, "mist": 5, "luckychant": 5, "sunnyday": 5,
              "raindance": 5, "sandstorm": 5, "snowscape": 5,
              "electricterrain": 5, "grassyterrain": 5, "mistyterrain": 5,
              "psychicterrain": 5, "trickroom": 5, "magicroom": 5,
              "wonderroom": 5, "gravity": 5}


def _number_families(move):
    raw, found = move.raw, []

    if (move.category != "Status" and move.base_power
            and not raw.get("multihit") and "charge" not in move.flags
            and not raw.get("ohko") and move.id not in VARIABLE_POWER_IDS):
        found.append(_damage_probe(move))

    blocks = [raw.get("secondary")] + list(raw.get("secondaries") or ())
    for block in blocks:
        if isinstance(block, dict) and 0 < block.get("chance", 100) < 100:
            found.append(_chance_probe(move, block["chance"], "secondary",
                                       block))
            break

    for key, kind in (("sideCondition", "side"), ("weather", "weather"),
                      ("terrain", "terrain"), ("pseudoWeather", "room")):
        name = raw.get(key)
        if not name:
            continue
        name = re.sub(r"[^a-z0-9]", "", str(name).lower())
        turns = (_DURATIONS.get(name) if kind in ("weather", "terrain")
                 else (raw.get("condition") or {}).get("duration")
                 or _DURATIONS.get(name))
        if turns:
            found.append(_duration_probe(move, name, kind, turns))
    return found


def _register_number_checks():
    from pkcm.engine.moves import VARIABLE_POWER
    VARIABLE_POWER_IDS.update(VARIABLE_POWER)

    for move in DEX.moves.values():
        if not DEX.exists_in_champions(move):
            continue
        families = _number_families(move)
        if not families:
            continue
        was = CHECKS.get(move.id)

        def run(families=families, was=was):
            notes, verdict = [], True
            if was is not None:
                ok, detail = was[1]()
                notes.append(f"behaviour: {detail}")
                verdict = ok is not False
            for what, probe in families:
                ok, detail = probe()
                notes.append(f"{what}: {detail}")
                if ok is False:
                    verdict = False
                elif ok is None:
                    notes[-1] += "  [inconclusive]"
            return verdict, "; ".join(notes)

        expect = " and ".join(([was[0]] if was else [])
                              + [what for what, _ in families])
        CHECKS[move.id] = (expect, run)


_register_number_checks()


# -- and the rest: every standard move gets one ------------------------------ #
#
# What is left after the three buckets above is the moves whose whole effect is
# a declaration the executor applies -- Bulk Up's two stages, Hypnosis's sleep,
# Charge's volatile. ``tests/test_effect_sweep.py`` already casts these, but it
# lives in a different file and answers a different question, and "all of them"
# should mean one command. So they are cast here too, against the declaration.

SELF_TARGETS_LOCAL = ("self", "adjacentAllyOrSelf", "allySide", "allyTeam",
                      "allies")
#: Aimed at a partner, so in singles there is nobody to aim at. The effect
#: sweep excuses the same set into a doubles check of its own.
ALLY_TARGETS_LOCAL = ("adjacentAlly",)


def _declared_probe(move):
    raw = move.raw

    def probe(m=move, raw=raw):
        for seed in range(10):
            f = _bout(m, seed=seed)
            f.turn(Action.move(0), Action.move(0))
            if m.category != "Status" and not _hits(f, m.id):
                continue
            if f.said("missed"):
                continue     # Sing is 55%, and a miss is not a broken move
            mine, theirs = 0, 1
            side = mine if raw.get("target") in SELF_TARGETS_LOCAL else theirs
            problems = []
            if raw.get("boosts"):
                got = f.boosts(side)
                for stat, amount in raw["boosts"].items():
                    if amount > 0 and got.get(stat, 0) < amount:
                        problems.append(f"{stat} {got.get(stat, 0)} < +{amount}")
                    if amount < 0 and got.get(stat, 0) > amount:
                        problems.append(f"{stat} {got.get(stat, 0)} > {amount}")
            block = raw.get("self")
            if isinstance(block, dict) and block.get("boosts"):
                got = f.boosts(mine)
                for stat, amount in block["boosts"].items():
                    if amount > 0 and got.get(stat, 0) < amount:
                        problems.append(f"self {stat} {got.get(stat, 0)}")
            if raw.get("status") and f.status(theirs) != raw["status"]:
                problems.append(f"status {f.status(theirs)} != {raw['status']}")
            if raw.get("volatileStatus"):
                name = raw["volatileStatus"]
                if name not in f.volatiles(side) and not f.said(name):
                    problems.append(f"volatile {name} absent")
            return (not problems,
                    "; ".join(problems) or "everything it declares happened")
        return None, "never connected in ten tries"

    return ("does what its data declares", probe)


def _register_declared_checks():
    for move in DEX.moves.values():
        if not DEX.exists_in_champions(move) or move.id in CHECKS:
            continue
        raw = move.raw
        if raw.get("target") in ALLY_TARGETS_LOCAL:
            continue
        if not (raw.get("boosts") or raw.get("status")
                or raw.get("volatileStatus")
                or (isinstance(raw.get("self"), dict)
                    and raw["self"].get("boosts"))):
            continue
        what, probe = _declared_probe(move)

        def run(probe=probe):
            ok, detail = probe()
            return (ok is not False, detail + ("  [inconclusive]"
                                               if ok is None else ""))

        CHECKS[move.id] = (what, run)


_register_declared_checks()


# -- the last eleven --------------------------------------------------------- #
#
# Fixed damage, counters and a hazard: no declared effect, no flag and no entry
# in SPECIAL_MOVES, because their arithmetic lives in ``tactics`` or in the
# damage path itself. Five more are ally-only and correctly refuse in singles.


def _veil_bout(weather="snowscape"):
    """Alolan Ninetales behind its own veil, and something swinging at it."""
    f = Fight(ours("ninetalesalola", "__none__",
                   ("auroraveil", "splash", "reflect", "protect"),
                   None, "serious", (32, 0, 32, 0, 2, 0)),
              [mon("garchomp", "__none__",
                   ("bodyslam", "shadowball", "splash", "focusenergy"),
                   None, "jolly", (0, 32, 2, 32, 0, 32))])
    if weather:
        f.state.field.weather = weather
        f.state.field.weather_turns = 12
    return f


@check("auroraveil", "goes up in snow and nowhere else, halves both kinds, "
                     "does not compound with Reflect, and a critical hit "
                     "goes straight through it")
def _aurora_veil():
    outside = _veil_bout(None)
    outside.turn(Action.move(0), Action.move(2))
    refused = "auroraveil" not in outside.conditions(0)

    def took(setup, foe_move):
        f = _veil_bout()
        for index in setup:
            f.turn(Action.move(index), Action.move(2))
        before = f.hp(0)
        f.turn(Action.move(1), Action.move(foe_move))
        return before - f.hp(0)

    # The control spends the same number of turns doing nothing, so the two
    # arms differ in the veil and in nothing else.
    physical = took([0], 0) / max(1, took([1], 0))
    special = took([0], 1) / max(1, took([1], 1))
    stacked, alone = took([0, 2], 0), took([0, 1], 0)

    def under_a_crit(veil):
        f = _veil_bout()
        f.turn(Action.move(1), Action.move(3))     # two Focus Energies is
        f.turn(Action.move(1), Action.move(3))     # stage four: every hit crits
        f.turn(Action.move(0) if veil else Action.move(1), Action.move(2))
        before = f.hp(0)
        f.turn(Action.move(1), Action.move(0))
        return before - f.hp(0)

    crit = under_a_crit(True) / max(1, under_a_crit(False))
    return (refused and 0.45 <= physical <= 0.55 and 0.45 <= special <= 0.55
            and stacked == alone and 0.95 <= crit <= 1.05,
            f"outside snow it refused={refused}; physical x{physical:.3f}, "
            f"special x{special:.3f}; with Reflect as well {stacked} against "
            f"{alone} alone; under a critical hit x{crit:.3f}")


def _breaks_screens(move_id):
    """The screen is down afterwards, and the breaker was not softened by it."""
    def probe():
        def swing(screen):
            f = Fight(ours("garchomp", "__none__",
                           (move_id, "splash", "protect", "rest"), None,
                           "adamant", (0, 32, 2, 0, 0, 32)),
                      [mon("snorlax", "__none__",
                           ("splash", "bodyslam", "protect", "rest"), None,
                           "sassy", (32, 0, 32, 0, 32, 0))])
            if screen:
                f.state.sides[1].conditions[screen] = 5
            before = f.hp(1)
            f.turn(Action.move(0), Action.move(0))
            return before - f.hp(1), f.conditions(1)

        behind, left = swing("reflect")
        clear, _ = swing(None)
        veiled, after_veil = swing("auroraveil")
        return (behind == clear and veiled == clear
                and "reflect" not in left and "auroraveil" not in after_veil,
                f"through Reflect {behind}, through Aurora Veil {veiled}, "
                f"with nothing up {clear}; the side kept {left or 'nothing'}")
    return probe


for _breaker in ("brickbreak", "psychicfangs", "ragingbull"):
    if _breaker in DEX.moves:
        CHECKS[_breaker] = ("takes the screens down before its own damage is "
                            "worked out", _breaks_screens(_breaker))


@check("seismictoss", "takes the user's level, whatever the target is")
def _seismic_toss():
    f = _bout(DEX.moves["seismictoss"])
    f.turn(Action.move(0), Action.move(0))
    hits = _hits(f, "seismictoss")
    got = int(re.search(r"amount=(\d+)", str(hits[0])).group(1)) if hits else 0
    return got == 50, f"dealt {got}, and the level is 50"


@check("nightshade", "takes the user's level, whatever the target is")
def _night_shade():
    f = _bout(DEX.moves["nightshade"])
    f.turn(Action.move(0), Action.move(0))
    hits = _hits(f, "nightshade")
    got = int(re.search(r"amount=(\d+)", str(hits[0])).group(1)) if hits else 0
    return got == 50, f"dealt {got}, and the level is 50"


@check("superfang", "takes half of what the target has left")
def _super_fang():
    f = _bout(DEX.moves["superfang"])
    before = f.hp(1)
    f.turn(Action.move(0), Action.move(0))
    hits = _hits(f, "superfang")
    got = int(re.search(r"amount=(\d+)", str(hits[0])).group(1)) if hits else 0
    return (abs(got - before // 2) <= 1,
            f"they were on {before} and it took {got}")


@check("counter", "returns twice the physical damage, and fails on special")
def _counter():
    def dealt(their_move):
        f = Fight(ours("wobbuffet", "__none__",
                       ("counter", "mirrorcoat", "splash", "protect"),
                       None, "sassy", (32, 0, 32, 0, 32, 0)),
                  [mon("garchomp", "__none__",
                       ("dragonclaw", "dragonpulse", "splash", "protect"),
                       None, "jolly", (0, 32, 2, 32, 0, 32))])
        f.turn(Action.move(0), Action.move(their_move))
        hits = _hits(f, "counter")
        return int(re.search(r"amount=(\d+)", str(hits[0])).group(1)) if hits else 0
    physical = dealt(0)     # Dragon Claw
    special = dealt(1)      # Dragon Pulse: Counter has no answer to it
    return (physical > 0 and special == 0,
            f"against a physical hit {physical}, against a special one {special}")


@check("comeuppance", "returns half again what last hit the user")
def _comeuppance():
    f = Fight(ours("kingambit", "__none__",
                   ("comeuppance", "ironhead", "splash", "protect"),
                   None, "sassy", (32, 0, 32, 0, 32, 0)),
              [mon("garchomp", "__none__",
                   ("dragonclaw", "splash", "protect", "earthquake"),
                   None, "jolly", (0, 32, 2, 0, 0, 32))])
    before = f.hp(0)
    f.turn(Action.move(0), Action.move(0))
    taken = before - f.hp(0)
    hits = _hits(f, "comeuppance")
    got = int(re.search(r"amount=(\d+)", str(hits[0])).group(1)) if hits else 0
    return (taken > 0 and got > taken,
            f"took {taken}, returned {got}")


@check("fling", "throws the held item away and needs one to throw")
def _fling():
    f = _until(lambda seed: Fight(
        [PokemonSet(species="weavile", ability="__none__", gender=None,
                    moves=("fling", "knockoff", "iceshard", "swordsdance"),
                    item="lifeorb", nature="jolly", sp=(0, 32, 2, 0, 0, 32))],
        [mon("snorlax", "__none__", ("splash", "bodyslam", "rest", "protect"),
             None, "serious", (32, 0, 32, 0, 2, 0))], seed)
        .turn(Action.move(0), Action.move(0)),
        lambda f: bool(_hits(f, "fling")))
    if f is None:
        return False, "never connected in ten tries"
    thrown = f.item(0) is None
    g = Fight(ours("weavile", "__none__",
                   ("fling", "knockoff", "iceshard", "swordsdance"),
                   None, "jolly", (0, 32, 2, 0, 0, 32)),
              [dummy("snorlax")])
    g.turn(Action.move(0), Action.move(0))
    empty_handed = not _hits(g, "fling")
    return (thrown and empty_handed,
            f"holding nothing afterwards={thrown}; with no item it did "
            f"nothing={empty_handed}")


@check("beatup", "hits once for every healthy body on the user's side")
def _beat_up():
    f = _bout(DEX.moves["beatup"])
    f.turn(Action.move(0), Action.move(0))
    return (len(_hits(f, "beatup")) >= 2,
            f"{len(_hits(f, 'beatup'))} separate hits, and we brought three")


@check("spitup", "spends the stockpiled layers, and fails without them")
def _spit_up():
    f = Fight(ours("snorlax", "__none__",
                   ("spitup", "stockpile", "splash", "protect"),
                   None, "modest", (32, 0, 32, 32, 2, 0)),
              [wall()])
    f.turn(Action.move(0), Action.move(0))
    empty = not _hits(f, "spitup")
    g = Fight(ours("snorlax", "__none__",
                   ("spitup", "stockpile", "splash", "protect"),
                   None, "modest", (32, 0, 32, 32, 2, 0)),
              [wall()])
    g.turn(Action.move(1), Action.move(0))
    g.turn(Action.move(1), Action.move(0))
    g.turn(Action.move(0), Action.move(0))
    fired = bool(_hits(g, "spitup"))
    return (empty and fired,
            f"with nothing stored it did nothing={empty}; "
            f"after two layers it fired={fired}")


@check("stickyweb", "slows down whoever comes in")
def _sticky_web():
    f = Fight(ours("ferrothorn", "__none__",
                   ("stickyweb", "gyroball", "protect", "spikes"),
                   None, "relaxed", (32, 0, 32, 0, 2, 0)),
              [dummy("magikarp")])
    f.turn(Action.move(0), Action.move(0))
    laid = "stickyweb" in f.conditions(1)
    f.turn(Action.move(1), Action.switch(1))
    return (laid and f.boosts(1).get("spe", 0) < 0,
            f"web laid={laid}; the replacement's boosts {f.boosts(1) or 'none'}")


@check("healingwish", "the user goes, and whoever follows arrives whole")
def _healing_wish():
    f = Fight([mon("clefable", "__none__",
                   ("healingwish", "moonblast", "protect", "splash"),
                   None, "timid", (32, 0, 32, 0, 2, 32)),
               mon("snorlax", "__none__",
                   ("splash", "bodyslam", "rest", "protect"),
                   None, "serious", (32, 0, 32, 0, 2, 0))],
              [dummy("magikarp")])
    f.state.sides[0].hp[1] //= 3          # the one waiting is hurt
    hurt = f.state.sides[0].hp[1]
    f.turn(Action.move(0), Action.move(0))
    if f.state.phase.name in ("FORCED_SWITCH", "MID_TURN_SWITCH"):
        f.turn(Action.switch(1), Action.PASS)
    return (f.hp(0) > hurt,
            f"the one who came in was on {hurt}, now {f.hp(0)}")


@check("lifedew", "heals the user when there is nobody else to heal")
def _life_dew():
    f = Fight(ours("milotic", "__none__",
                   ("lifedew", "surf", "protect", "splash"),
                   None, "serious", (32, 0, 32, 0, 2, 0)),
              [mon("garchomp", "__none__",
                   ("earthquake", "dragonclaw", "firefang", "splash"),
                   None, "jolly", (0, 32, 2, 0, 0, 32))])
    f.turn(Action.move(1), Action.move(0))
    hurt = f.hp(0)
    f.turn(Action.move(0), Action.move(3))
    return (f.hp(0) > hurt, f"{hurt} -> {f.hp(0)} of {f.max_hp(0)}")


for _ally in ("afteryou", "allyswitch", "aromaticmist", "coaching", "quash"):
    if _ally not in CHECKS and _ally in DEX.moves:
        def _ally_probe(move_id=_ally):
            move = DEX.moves[move_id]
            species = ("clefable" if move.category == "Status" else "weavile")
            return _fails_cleanly(species, (move_id, "splash", "tackle",
                                            "protect"))
        CHECKS[_ally] = ("has no ally to use it on in singles", _ally_probe)


# --------------------------------------------------------------------------- #


def main() -> int:
    wanted = sys.argv[1:] or sorted(CHECKS)
    unknown = [one for one in wanted if one not in CHECKS]
    if unknown:
        print(f"no check written for: {', '.join(unknown)}")
        return 2
    bad, unmeasured = 0, []
    for move in wanted:
        expect, fn = CHECKS[move]
        try:
            ok, detail = fn()
        except Exception as error:                      # a crash is a failure
            ok, detail = False, f"{type(error).__name__}: {error}"
        # A probe that could not set its position up says "[inconclusive]"
        # and used to come out as "ok". Aurora Veil went that way the
        # moment it started needing snow: the move stopped being castable,
        # the probe stopped measuring, and the report still read 498/498.
        blind = ok and "[inconclusive]" in str(detail)
        korean = NAMES.get(move, move)
        mark = "??  " if blind else ("ok  " if ok else "FAIL")
        print(f"{mark}  {korean:<14} ({move})")
        if blind:
            unmeasured.append(move)
            print(f"        unmeasured: {detail}")
        elif not ok:
            bad += 1
            print(f"        expected: {expect}")
            print(f"        saw:      {detail}")
    measured = len(wanted) - bad - len(unmeasured)
    print()
    print(f"{measured}/{len(wanted)} behaved"
          + (f"; {len(unmeasured)} could not be measured: "
             f"{', '.join(unmeasured)}" if unmeasured else ""))
    return 1 if bad or unmeasured else 0


import json  # noqa: E402
NAMES = json.loads((ROOT / "data" / "champions" / "names.json")
                   .read_text(encoding="utf-8"))["moves"]

if __name__ == "__main__":
    raise SystemExit(main())
