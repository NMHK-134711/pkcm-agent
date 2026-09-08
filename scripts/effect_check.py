"""Abilities and items, held in a real battle to see whether they do anything.

``mechanic_check`` did this for all 497 moves and found eleven that were
registered, named, and silent. Abilities and items have never had it: 224 of
the format's abilities and 73 of its items carry a registered effect, and the
only thing standing behind them is 78 hand-written tests. The two bugs that
cost the most this month were both of this shape -- King's Shield hung on an
event a successful block guarantees will not arrive, Encore wrote an index
nobody read -- and abilities are where the same mistake is easiest to make,
because an ability that never fires looks exactly like one that never came up.

The first pass is a differ, not a verdict. Each ability is put on a Pokemon,
the same battle is played with it and with nothing, and the two traces are
compared. An ability whose battle is byte-identical to the one without it did
nothing *in that position* -- which is a question, not an accusation: half of
them need rain, or a status, or a low bar of HP. The report is the triage list
for the scenarios that follow.

    python scripts/effect_check.py                 # abilities and items
    python scripts/effect_check.py --abilities     # one side only
    python scripts/effect_check.py --stones        # the 74 mega stones
    python scripts/effect_check.py --doubles       # the 16 partner abilities
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pkcm.engine.abilities  # noqa: E402,F401  (registers the effects)
import pkcm.engine.items      # noqa: E402,F401
from pkcm.engine.abilities import (HEALER_CHANCE,          # noqa: E402
                                   HOSPITALITY_FRACTION)
from pkcm.data.dex import load_dex                      # noqa: E402
from pkcm.engine.actions import Action                  # noqa: E402
from pkcm.engine.battle import step                     # noqa: E402
from pkcm.engine.effects import registered              # noqa: E402
from pkcm.engine.pokemon import PokemonSet              # noqa: E402
from pkcm.engine.state import (BOOST_INDEX, BattleConfig,   # noqa: E402
                               legal_actions, new_battle)

DEX = load_dex()
CONFIG = BattleConfig(dex=DEX, regulation=DEX.regulation("m_b"),
                      battle_format="singles")
NAMES = json.loads((ROOT / "data" / "champions" / "names.json")
                   .read_text(encoding="utf-8"))


def mon(species, ability, moves, item=None, nature="serious", sp=(0,) * 6,
        gender=None):
    return PokemonSet(species=species, ability=ability, moves=tuple(moves),
                      item=item, nature=nature, sp=sp, gender=gender)


FILLER = [mon("pikachu", "__none__", ("tackle",)),
          mon("alakazam", "__none__", ("tackle",))]


def trace(ours, theirs, turns, seed=7):
    """Play a scripted battle and return everything that could have differed."""
    state = new_battle(CONFIG, (tuple(ours) + tuple(FILLER),
                                tuple(theirs) + tuple(FILLER)), seed=seed)
    state, _ = step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))
    lines = []
    for ours_action, theirs_action in turns:
        if state.phase.name != "BATTLE":
            break
        state, events = step(state, ours_action, theirs_action)
        lines.extend(str(one) for one in events)
    side = state.sides[0]
    other = state.sides[1]
    lines.append(f"hp={side.hp} {other.hp} status={side.status} {other.status}")
    lines.append(f"boosts={side.boosts} {other.boosts}")
    lines.append(f"conditions={side.conditions} {other.conditions}")
    lines.append(f"field={state.field.weather} {state.field.terrain} "
                 f"{state.field.rooms}")
    return "\n".join(lines)


#: An ability fires when its own condition arrives, so one position answers
#: for one ability and calls the other two hundred silent. These are the
#: conditions, one battle each: a contact hit, a special hit, each weather, a
#: type that somebody absorbs, a stat drop, a status, low health, a berry, and
#: an entry. Between them almost everything gets its moment.
_HOLDER = ("snorlax", "serious", (32, 0, 32, 0, 2, 0))

#: Nothing a singles battle can show, because the effect is about a partner.
#: Not silence: there is no second Pokemon on the field for them to act on.
DOUBLES_ONLY = frozenset({
    "battery", "powerspot", "friendguard", "healer", "symbiosis",
    "powerofalchemy", "receiver", "costar", "curiousmedicine", "hospitality",
    "plus", "minus", "sweetveil", "telepathy", "propellertail", "stalwart",
    # Shields Grass types on its own side, and the two holders in this format
    # are Fairies: in singles there is nobody on the side it could shield.
    "flowerveil",
})


def _foe(moves, nature="jolly", sp=(0, 32, 2, 32, 0, 32), species="garchomp"):
    return mon(species, "__none__", moves, None, nature, sp)


_ROSTER = sorted(DEX.regulation("m_b").legal_species
                 | DEX.regulation("m_b").legal_megas)


def _natural_holder(ability, item=None):
    """The first legal species that actually has it, or None."""
    if ability is None:
        return None
    for species in _ROSTER:
        if ability in DEX.species[species].abilities:
            return mon(species, ability,
                       ("splash", "bodyslam", "rest", "protect"), item,
                       "serious", (32, 0, 32, 0, 2, 0))
    return None


def _positions(ability, item=None):
    """(name, ours, theirs, script, setup) for each condition worth trying."""
    def holder(moves=("splash", "bodyslam", "rest", "protect")):
        return mon(_HOLDER[0], ability or "__none__", moves, item,
                   _HOLDER[1], _HOLDER[2])

    quiet = [(Action.move(0), Action.move(3)),
             (Action.move(0), Action.move(3))]
    trade = [(Action.move(1), Action.move(0)),
             (Action.move(1), Action.move(0)),
             (Action.move(0), Action.move(3))]

    def weather(name):
        def setup(state):
            state.field.weather = name
            state.field.weather_turns = 8
        return setup

    def hurt(state):
        slot = state.sides[0].active[0]
        state.sides[0].hp[slot] //= 4

    def status(name):
        def setup(state):
            state.sides[0].status[state.sides[0].active[0]] = name
        return setup

    typed = _foe(("firefang", "surf", "thunderbolt", "earthquake"))
    plain_foe = _foe(("dragonclaw", "dragonpulse", "willowisp", "splash"))
    dropper = _foe(("growl", "hypervoice", "bulletseed", "splash"))

    out = [
        ("a contact hit and a swing back", trade, plain_foe, None),
        ("a special hit", [(Action.move(0), Action.move(1))] + quiet,
         plain_foe, None),
        ("a status move at us", [(Action.move(0), Action.move(2))] + quiet,
         plain_foe, None),
        ("fire at us", [(Action.move(0), Action.move(0))] + quiet, typed, None),
        ("water at us", [(Action.move(0), Action.move(1))] + quiet, typed, None),
        ("electricity at us", [(Action.move(0), Action.move(2))] + quiet,
         typed, None),
        ("ground at us", [(Action.move(0), Action.move(3))] + quiet, typed,
         None),
        ("a stat drop at us", [(Action.move(0), Action.move(0))] + quiet,
         dropper, None),
        ("a sound move at us", [(Action.move(0), Action.move(1))] + quiet,
         dropper, None),
        ("a bullet move at us", [(Action.move(0), Action.move(2))] + quiet,
         dropper, None),
    ]
    for name in ("sunnyday", "raindance", "sandstorm", "snowscape"):
        out.append((f"{name} overhead", trade, plain_foe, weather(name)))
    for name in ("psn", "par", "brn", "slp"):
        out.append((f"already {name}", trade, plain_foe, status(name)))
    positions = [(what, [holder()], [foe], script, setup)
                 for what, script, foe, setup in out]

    # The ones above are all about what happens *to* the holder. These are
    # about what it does: Blaze wants us swinging a Fire move on a quarter of
    # our health, Compound Eyes wants a move that can miss, Frisk wants
    # something to find, Arena Trap wants somebody trying to leave.
    attacker = mon(_HOLDER[0], ability or "__none__",
                   ("flamethrower", "surf", "focusblast", "thunderwave"),
                   item, "modest", (0, 0, 2, 32, 0, 32))
    swing = [(Action.move(0), Action.move(3)),
             (Action.move(1), Action.move(3)),
             (Action.move(2), Action.move(3)),
             (Action.move(3), Action.move(3))]
    positions += [
        ("swinging four types of our own", [attacker], [plain_foe], swing,
         None),
        ("swinging them on a quarter of our health", [attacker], [plain_foe],
         swing, hurt),
        ("a super-effective hit at us", [holder()],
         [_foe(("closecombat", "dragonclaw", "splash", "willowisp"))],
         [(Action.move(0), Action.move(0))] + quiet, None),
        ("a foe holding something", [holder()],
         [mon("garchomp", "__none__",
              ("dragonclaw", "splash", "protect", "willowisp"), "leftovers",
              "jolly", (0, 32, 2, 0, 0, 32))],
         [(Action.move(0), Action.move(1))] + quiet, None),
        ("a foe trying to leave", [holder()], [plain_foe],
         [(Action.move(0), Action.switch(1)),
          (Action.move(0), Action.move(3))], None),
    ]
    # -- the conditions the first battery could not make --------------------- #
    #
    # Grouped by the hook each ability registers, because that is what says
    # what it is waiting for. Eleven want a switch-in, six want a status that
    # arrives *as a move* rather than being written into the state, five want
    # a priority move, nine want the holder swinging a particular kind of move.

    def bench_first(moves=("splash", "bodyslam", "rest", "protect"),
                    item_held=None):
        """Lead with a filler so the holder's entry is a switch-in."""
        return [mon("magikarp", "__none__", ("splash", "tackle")),
                mon(_HOLDER[0], ability or "__none__", moves,
                    item_held or item, _HOLDER[1], _HOLDER[2])]

    swap_in = [(Action.switch(1), Action.move(3)),
               (Action.move(0), Action.move(0)),
               (Action.move(0), Action.move(3))]
    swap_out = [(Action.move(0), Action.move(3)),
                (Action.switch(1), Action.move(3)),
                (Action.move(0), Action.move(3))]

    statuser = _foe(("toxic", "thunderwave", "hypnosis", "willowisp"),
                    "jolly", (0, 0, 2, 0, 0, 32), "gengar")
    quick = _foe(("aquajet", "iceshard", "fakeout", "splash"), "jolly",
                 (0, 32, 2, 0, 0, 32), "weavile")
    bother = _foe(("confuseray", "screech", "flash", "swordsdance"), "timid",
                  (0, 0, 2, 32, 0, 32), "gengar")

    #: The holder swinging one kind of move after another: punch, pulse,
    #: sound, bite, slicing, multihit, a weak one, a recoil one, and the four
    #: types the pinch abilities care about.
    #: Garchomp was the wrong thing to swing at: it is immune to Electric,
    #: which read as Transistor doing nothing, and it fell before the fourth
    #: swing, which read as Strong Jaw doing nothing. This one is neutral to
    #: everything below and fat enough to still be there at the end.
    punchbag = mon(_HOLDER[0], "__none__",
                   ("splash", "bodyslam", "rest", "protect"), None,
                   "sassy", (32, 0, 32, 0, 32, 0))

    #: Fast and light, for the two swings where the punchbag's bulk *was* the
    #: measurement: priority only shows against something quicker, and a
    #: weight ratio only shows against something that weighs differently.
    nimble = _foe(("bodyslam", "iceshard", "splash", "protect"), "jolly",
                  (0, 32, 2, 0, 0, 32), "weavile")

    def swinging(moves, nature="modest", sp=(0, 0, 2, 32, 0, 32), setup=None,
                 foe=None):
        return ([mon(_HOLDER[0], ability or "__none__", moves, item,
                     nature, sp)], [foe or punchbag],
                [(Action.move(i), Action.move(0)) for i in range(4)], setup)

    extra = [
        ("coming in as a switch", bench_first(), [plain_foe], swap_in, None),
        ("leaving the field", bench_first(), [plain_foe], swap_out, None),
        ("ice at us, over and over", [holder()],
         [_foe(("icebeam", "splash", "protect", "willowisp"), "timid",
               (0, 0, 2, 32, 0, 32), "gengar")],
         [(Action.move(0), Action.move(0))] * 4, None),
        ("a priority move at us", [holder()], [quick],
         [(Action.move(0), Action.move(i)) for i in range(3)], None),
        ("confusion and a Defence drop at us", [holder()], [bother],
         [(Action.move(0), Action.move(i)) for i in range(4)], None),
        ("a foe setting up in front of us", [holder()], [bother],
         [(Action.move(0), Action.move(3)), (Action.move(1), Action.move(3)),
          (Action.move(1), Action.move(3))], None),
    ]
    #: Toxic first and Thunder Wave second meant the second one found the
    #: target already poisoned, so Insomnia and Magma Armor never saw a roll.
    for _index, _label in enumerate(("toxic", "a thunder wave", "hypnosis",
                                     "a will-o-wisp")):
        extra.append((f"{_label} at us, three times over", [holder()],
                      [statuser],
                      [(Action.move(0), Action.move(_index))] * 3, None))

    extra.append(("something reaching for what we are holding",
                  [mon(_HOLDER[0], ability or "__none__",
                       ("splash", "bodyslam", "rest", "protect"),
                       item or "leftovers", _HOLDER[1], _HOLDER[2])],
                  [_foe(("knockoff", "crunch", "splash", "protect"), "jolly",
                        (0, 32, 2, 0, 0, 32), "weavile")],
                  [(Action.move(0), Action.move(0))] * 2, None))

    def hurt_them(state):
        slot = state.sides[1].active[0]
        state.sides[1].hp[slot] = state.sides[1].hp[slot] * 2 // 5

    #: Pickpocket wants to be *hit* by something with an item -- every other
    #: item position had us doing the hitting -- and Pickup wants somebody
    #: else to spend one where it can see.
    extra.append(("something holding an item hitting us", [holder()],
                  [mon("garchomp", "__none__",
                       ("dragonclaw", "splash", "protect", "willowisp"),
                       "leftovers", "jolly", (0, 32, 2, 0, 0, 32))],
                  [(Action.move(0), Action.move(0))] * 2, None))
    extra.append(("something spending a berry in front of us",
                  [mon(_HOLDER[0], ability or "__none__",
                       ("bodyslam", "splash", "rest", "protect"), item,
                       "adamant", (0, 32, 2, 0, 0, 32))],
                  [mon("garchomp", "__none__",
                       ("splash", "dragonclaw", "protect", "willowisp"),
                       "sitrusberry", "jolly", (0, 32, 2, 0, 0, 32))],
                  [(Action.move(0), Action.move(0))] * 3, hurt_them))

    #: Sun and a status at once, and sun and a special move at once. Leaf
    #: Guard refuses a status only while the sun is out, and Solar Power only
    #: pays and only earns then -- neither could be seen by a position that
    #: had the weather but no status, or the status but no weather.
    def sunny(state):
        state.field.weather = "sunnyday"
        state.field.weather_turns = 8

    extra.append(("a will-o-wisp at us with the sun out", [holder()],
                  [statuser], [(Action.move(0), Action.move(3))] * 3, sunny))
    extra.append(("swinging specially with the sun out",
                  [mon(_HOLDER[0], ability or "__none__",
                       ("dragonpulse", "splash", "rest", "protect"), item,
                       "modest", (0, 0, 2, 32, 0, 32))],
                  [punchbag], [(Action.move(0), Action.move(0))] * 3, sunny))

    #: Heavy Slam reads a ratio, and Snorlax against Snorlax is 1:1 whether
    #: the weight is doubled, halved or left alone. Venusaur weighs 100kg to
    #: Snorlax's 460, which puts the three answers in three different buckets.
    extra.append(("a heavy hit where the weights are close",
                  [mon(_HOLDER[0], ability or "__none__",
                       ("heavyslam", "splash", "rest", "protect"), item,
                       "adamant", (0, 32, 2, 0, 0, 32))],
                  [mon("venusaur", "__none__",
                       ("splash", "bodyslam", "rest", "protect"), None,
                       "sassy", (32, 0, 32, 0, 32, 0))],
                  [(Action.move(0), Action.move(0))] * 2, None))

    #: Snorlax is a Normal type, so the Ghost move in the position below never
    #: touched it: anything that softens Ghost damage read as silent.
    extra.append(("a Ghost move at something it can actually hit",
                  [mon("milotic", ability or "__none__",
                       ("splash", "bodyslam", "rest", "protect"), item,
                       "sassy", (32, 0, 32, 0, 32, 0))],
                  [mon("gengar", "__none__",
                       ("shadowball", "willowisp", "splash", "protect"), None,
                       "timid", (0, 0, 2, 32, 0, 32))],
                  [(Action.move(0), Action.move(0))] * 2, None))

    #: ``bench_first`` leads with a filler, so its "leaving" scripts were
    #: switching the holder *in*. Regenerator and Natural Cure want the other
    #: direction, and never got it.
    def leads(moves=("splash", "bodyslam", "rest", "protect")):
        return [mon(_HOLDER[0], ability or "__none__", moves, item,
                    _HOLDER[1], _HOLDER[2]),
                mon("magikarp", "__none__", ("splash", "tackle"))]

    leave = [(Action.move(0), Action.move(0)),
             (Action.switch(1), Action.move(3)),
             (Action.move(0), Action.move(3))]
    extra.append(("hurt, and then leaving", leads(), [plain_foe], leave, None))
    extra.append(("burnt, and then leaving", leads(), [plain_foe], leave,
                  status("brn")))

    #: Disguise is Mimikyu's, Zero to Hero is Palafin's: a Snorlax wearing
    #: them cannot show what they do, because what they do is change forme.
    natural = _natural_holder(ability, item)
    if natural is not None:
        extra.append(("on a Pokemon that really has it",
                      [natural, mon("magikarp", "__none__",
                                    ("splash", "tackle"))],
                      [plain_foe],
                      [(Action.move(0), Action.move(0)),
                       (Action.switch(1), Action.move(3)),
                       (Action.switch(0), Action.move(3)),
                       (Action.move(1), Action.move(0))], None))

    extra.append(("punches, pulses, sound and a bite",) +
                 swinging(("firepunch", "dragonpulse", "hypervoice", "bite")))
    extra.append(("a slice, a swarm of hits, a weak move and a recoil one",) +
                 swinging(("nightslash", "bulletseed", "tackle", "doubleedge"),
                          "adamant", (0, 32, 2, 0, 0, 32)))
    extra.append(("dragon, grass, bug and electric",) +
                 swinging(("dragonpulse", "energyball", "bugbuzz",
                           "thunderbolt")))
    extra.append(("the same four on a quarter of our health",) +
                 swinging(("dragonpulse", "energyball", "bugbuzz",
                           "thunderbolt"), setup=hurt))
    extra.append(("a flying move and a heal against something quicker",) +
                 swinging(("bravebird", "recover", "splash", "protect"),
                          "adamant", (0, 32, 2, 0, 0, 32), foe=nimble))
    extra.append(("weight against weight",) +
                 swinging(("grassknot", "heavyslam", "lowkick", "tackle"),
                          "adamant", (0, 32, 2, 0, 0, 32), foe=nimble))
    extra.append(("normal and fighting at a Ghost",) +
                 ([mon(_HOLDER[0], ability or "__none__",
                       ("bodyslam", "closecombat", "shadowball", "splash"),
                       item, "adamant", (0, 32, 2, 0, 0, 32))],
                  [mon("gengar", "__none__",
                       ("splash", "shadowball", "protect", "willowisp"),
                       None, "timid", (0, 0, 2, 32, 0, 32))],
                  [(Action.move(i), Action.move(0)) for i in range(3)], None))
    extra.append(("a dark move at us",) +
                 ([holder()],
                  [_foe(("knockoff", "crunch", "splash", "protect"), "jolly",
                        (0, 32, 2, 0, 0, 32), "weavile")],
                  [(Action.move(0), Action.move(0)),
                   (Action.move(0), Action.move(1))], None))

    def poisoned_foe(state):
        state.sides[1].status[state.sides[1].active[0]] = "psn"

    extra.append(("swinging at something already poisoned",) +
                 ([mon(_HOLDER[0], ability or "__none__",
                       ("bodyslam", "splash", "rest", "protect"), item,
                       "adamant", (0, 32, 2, 0, 0, 32))],
                  [plain_foe],
                  [(Action.move(0), Action.move(3))] * 2, poisoned_foe))

    def dying_foe(state):
        state.sides[1].hp[state.sides[1].active[0]] = 1

    extra.append(("finishing something off",) +
                 ([mon(_HOLDER[0], ability or "__none__",
                       ("bodyslam", "splash", "rest", "protect"), item,
                       "adamant", (0, 32, 2, 0, 0, 32))],
                  [plain_foe],
                  [(Action.move(0), Action.move(3))] * 2, dying_foe))

    def lethal(state):
        """Sturdy and Focus Sash both want a hit that would end it outright."""
        slot = state.sides[0].active[0]
        state.sides[0].hp[slot] = state.active_pokemon(0).max_hp

    extra.append(("a hit that would end us from full health",) +
                 ([mon("shedinja" if "shedinja" in DEX.species else _HOLDER[0],
                       ability or "__none__",
                       ("splash", "tackle", "protect", "rest"), item,
                       "serious", (0,) * 6)],
                  [_foe(("earthquake", "dragonclaw", "splash", "willowisp"))],
                  [(Action.move(0), Action.move(0))], lethal))

    def screens(state):
        state.sides[1].conditions["reflect"] = 5
        state.sides[1].conditions["lightscreen"] = 5

    extra.append(("screens up on their side",) +
                 ([mon(_HOLDER[0], ability or "__none__",
                       ("bodyslam", "dragonpulse", "splash", "protect"), item,
                       "adamant", (0, 32, 2, 32, 0, 0))],
                  [plain_foe],
                  [(Action.move(0), Action.move(3)),
                   (Action.move(1), Action.move(3))], screens))

    def terrain(name):
        def setup(state):
            state.field.terrain = name
            state.field.terrain_turns = 8
        return setup

    for name in ("electricterrain", "grassyterrain", "mistyterrain",
                 "psychicterrain"):
        extra.append((f"{name} underfoot", [holder()], [plain_foe], trade,
                      terrain(name)))

    def ally_down(state):
        """Supreme Overlord counts the ones already gone."""
        state.sides[0].hp[1] = 0
        state.sides[0].hp[2] = 0

    extra.append(("two of ours already down",) +
                 ([mon(_HOLDER[0], ability or "__none__",
                       ("bodyslam", "splash", "rest", "protect"), item,
                       "adamant", (0, 32, 2, 0, 0, 32))],
                  [plain_foe],
                  [(Action.move(0), Action.move(3))] * 2, ally_down))

    positions += [(what, ours, theirs, script, setup)
                  for what, ours, theirs, script, setup in extra]
    # -- the last conditions, each one named after what was waiting for it --- #

    def with_status(name, weather_name=None):
        def setup(state):
            state.sides[0].status[state.sides[0].active[0]] = name
            if weather_name:
                state.field.weather = weather_name
                state.field.weather_turns = 8
        return setup

    def gendered(who, what):
        return mon(who, ability or "__none__",
                   ("bodyslam", "splash", "rest", "protect"), item,
                   "adamant", (0, 32, 2, 0, 0, 32), gender=what)

    last = [
        # Damp: nothing explodes while it is out.
        ("something exploding at us", [holder()],
         [_foe(("explosion", "dragonclaw", "splash", "protect"))],
         [(Action.move(0), Action.move(0))], None),
        # Magician and Pickpocket: an item to take, and a hit that takes it.
        ("hitting a foe that is holding something",
         [mon(_HOLDER[0], ability or "__none__",
              ("bodyslam", "splash", "rest", "protect"), item,
              "adamant", (0, 32, 2, 0, 0, 32))],
         [mon("garchomp", "__none__",
              ("splash", "dragonclaw", "protect", "willowisp"), "leftovers",
              "jolly", (0, 32, 2, 0, 0, 32))],
         [(Action.move(0), Action.move(0))] * 2, None),
        # Suction Cups: the drag it refuses.
        ("being blown out", [holder()],
         [_foe(("whirlwind", "dragontail", "splash", "protect"))],
         [(Action.move(0), Action.move(0)),
          (Action.move(0), Action.move(1))], None),
        # Hydration, Leaf Guard, Quick Feet: a status *and* the weather.
        ("burnt in the rain", [holder()], [plain_foe], trade,
         with_status("brn", "raindance")),
        ("burnt in the sun", [holder()], [plain_foe], trade,
         with_status("brn", "sunnyday")),
        ("asleep in the rain", [holder()], [plain_foe], trade,
         with_status("slp", "raindance")),
        # Natural Cure, Regenerator: carrying something off the field.
        ("leaving while burnt", bench_first(), [plain_foe], swap_out,
         with_status("brn")),
        ("leaving hurt", bench_first(), [plain_foe],
         [(Action.move(0), Action.move(0)), (Action.switch(1), Action.move(3)),
          (Action.move(0), Action.move(3))], None),
        # Long Reach, Unseen Fist: whether our move counts as contact.
        ("touching something that punishes touch",
         [mon(_HOLDER[0], ability or "__none__",
              ("bodyslam", "splash", "rest", "protect"), item,
              "adamant", (0, 32, 2, 0, 0, 32))],
         [mon("garchomp", "roughskin",
              ("splash", "dragonclaw", "protect", "willowisp"), "rockyhelmet",
              "jolly", (0, 32, 2, 0, 0, 32))],
         [(Action.move(0), Action.move(0))] * 2, None),
        ("swinging at something behind Protect",
         [mon(_HOLDER[0], ability or "__none__",
              ("bodyslam", "splash", "rest", "protect"), item,
              "adamant", (0, 32, 2, 0, 0, 32))],
         [_foe(("protect", "dragonclaw", "splash", "willowisp"))],
         [(Action.move(0), Action.move(0))] * 2, None),
        # Corrosion, Magnet Pull: a Steel type to poison and to hold.
        ("poisoning a Steel type",
         [mon(_HOLDER[0], ability or "__none__",
              ("toxic", "bodyslam", "rest", "protect"), item,
              "serious", (32, 0, 32, 0, 2, 0))],
         [mon("archaludon", "__none__",
              ("splash", "flashcannon", "protect", "dragontail"), None,
              "sassy", (32, 0, 32, 0, 32, 0))],
         [(Action.move(0), Action.move(0))] * 2, None),
        ("a Steel type trying to leave", [holder()],
         [mon("archaludon", "__none__",
              ("splash", "flashcannon", "protect", "dragontail"), None,
              "sassy", (32, 0, 32, 0, 32, 0))],
         [(Action.move(0), Action.switch(1)),
          (Action.move(0), Action.move(0))], None),
        # Rivalry: the sets carry no gender unless one is stated.
        ("swinging at the same gender", [gendered(_HOLDER[0], "M")],
         [mon("garchomp", "__none__",
              ("splash", "dragonclaw", "protect", "willowisp"), None, "jolly",
              (0, 32, 2, 0, 0, 32), gender="M")],
         [(Action.move(0), Action.move(0))] * 2, None),
        ("swinging at the other gender", [gendered(_HOLDER[0], "M")],
         [mon("garchomp", "__none__",
              ("splash", "dragonclaw", "protect", "willowisp"), None, "jolly",
              (0, 32, 2, 0, 0, 32), gender="F")],
         [(Action.move(0), Action.move(0))] * 2, None),
        # Unburden, Quick Feet, Stall: Speed only shows when it changes who
        # goes first, so the two are close enough for it to matter.
        ("a berry eaten in a close race",
         [mon("weavile", ability or "__none__",
              ("bodyslam", "splash", "rest", "protect"),
              item or "sitrusberry", "jolly", (0, 32, 2, 0, 0, 16))],
         [mon("weavile", "__none__",
              ("earthquake", "bodyslam", "splash", "protect"), None, "jolly",
              (0, 32, 2, 0, 0, 16))],
         [(Action.move(0), Action.move(0))] * 3, None),
        ("paralysed in a close race",
         [mon("weavile", ability or "__none__",
              ("bodyslam", "splash", "rest", "protect"), item, "jolly",
              (0, 32, 2, 0, 0, 16))],
         [mon("weavile", "__none__",
              ("bodyslam", "splash", "protect", "rest"), None, "jolly",
              (0, 32, 2, 0, 0, 16))],
         [(Action.move(0), Action.move(0))] * 3, with_status("par")),
        # Inner Focus, Steadfast, Oblivious: a flinch and an infatuation.
        ("flinched by something faster",
         [mon(_HOLDER[0], ability or "__none__",
              ("bodyslam", "splash", "rest", "protect"), item, "brave",
              (32, 32, 2, 0, 0, 0), gender="M")],
         [mon("weavile", "__none__",
              ("fakeout", "attract", "iceshard", "taunt"), None, "jolly",
              (0, 32, 2, 0, 0, 32), gender="F")],
         [(Action.move(0), Action.move(i)) for i in range(4)], None),
        # Purifying Salt: Ghost damage, and a status that should not stick.
        ("a Ghost move and a burn at us", [holder()],
         [mon("gengar", "__none__",
              ("shadowball", "willowisp", "splash", "protect"), None, "timid",
              (0, 0, 2, 32, 0, 32))],
         [(Action.move(0), Action.move(0)), (Action.move(0), Action.move(1)),
          (Action.move(0), Action.move(2))], None),
    ]
    positions += last
    return positions


def _berry_position(ability, item):
    """Eating a berry: Cheek Pouch, Ripen, Harvest, Gluttony, Cud Chew."""
    ours = mon(_HOLDER[0], ability or "__none__",
               ("splash", "bodyslam", "rest", "protect"),
               item or "sitrusberry", _HOLDER[1], _HOLDER[2])
    return ("holding a berry through a big hit",
            [ours], [_foe(("earthquake", "dragonclaw", "splash", "willowisp"))],
            [(Action.move(0), Action.move(0))] * 4, None)


def trace_with(ours, theirs, script, seed, setup):
    state = new_battle(CONFIG, (tuple(ours) + tuple(FILLER),
                                tuple(theirs) + tuple(FILLER)), seed=seed)
    state, _ = step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))
    if setup is not None:
        setup(state)
    lines = []
    for ours_action, theirs_action in script:
        if state.phase.name != "BATTLE":
            break
        # An action that has *become* illegal is the whole point of Arena Trap
        # and Magnet Pull, and submitting one raised, which the differ caught
        # and swallowed -- so every trapping ability read as silent. Refusing
        # is now a line in the trace like any other.
        chosen = []
        for side_index, action in ((0, ours_action), (1, theirs_action)):
            allowed = legal_actions(state, side_index)
            if action not in allowed:
                lines.append(f"refused(side={side_index}, {action})")
                action = allowed[0]
            chosen.append(action)
        state, events = step(state, *chosen)
        lines.extend(str(one) for one in events)
    side, other = state.sides[0], state.sides[1]
    lines.append(f"hp={side.hp} {other.hp} status={side.status} {other.status}")
    lines.append(f"boosts={side.boosts} {other.boosts}")
    lines.append(f"conditions={side.conditions} {other.conditions}")
    lines.append(f"field={state.field.weather} {state.field.terrain} "
                 f"{state.field.rooms}")
    return "\n".join(lines)


def _differs(ability=None, item=None, seeds=(7, 11)):
    """Whether any of the positions plays out differently with it than without.

    Returns the position that showed the difference, or ``None``.
    """
    positions = _positions(ability, item) + [_berry_position(ability, item)]
    for what, ours, theirs, script, setup in positions:
        # The control differs in exactly one field: the thing under test.
        # It used to differ in two, and to leave the item alone -- so every
        # ability passed on the gendered position whether or not it did
        # anything, and no item was ever actually compared against nothing.
        plain = [mon(one.species, "__none__" if ability else one.ability,
                     one.moves, None if item else one.item,
                     one.nature, one.sp, one.gender) for one in ours]
        for seed in seeds:
            try:
                with_it = trace_with(ours, theirs, script, seed, setup)
                without = trace_with(plain, theirs, script, seed, setup)
            except Exception:
                continue
            if with_it != without:
                return what
    return None


# -- doubles ----------------------------------------------------------------- #
#
# Sixteen abilities are about a partner, and a singles battle has none: Battery
# boosts an ally's special move, Friend Guard softens what an ally takes,
# Symbiosis hands an ally its item, Receiver inherits a fallen ally's ability.
# The differ above cannot reach them and they are not silent, so they get a
# field with two Pokemon on each side and a written check each -- the same
# standard the 497 moves were held to, with the numbers where there are
# numbers: 1.3x, 0.75x, 1.5x, a quarter of the maximum, three turns in ten.
#
# The format brings six and sends four, so every script submits two actions a
# side and names the field position it is aimed at.

from pkcm.engine.mutate import max_hp as _max_hp  # noqa: E402

DOUBLES_CONFIG = BattleConfig(dex=DEX, regulation=DEX.regulation("m_b"),
                              battle_format="doubles")

_FILLERS = [mon("pikachu", "__none__", ("tackle", "splash")),
            mon("alakazam", "__none__", ("tackle", "splash")),
            mon("starmie", "__none__", ("tackle", "splash")),
            mon("magikarp", "__none__", ("tackle", "splash"))]


def _team(*brought):
    """The four that go out, padded so team preview has something to send."""
    return tuple(list(brought) + _FILLERS)[:6]


class Duo:
    """A doubles field, and the turns played on it."""

    def __init__(self, ours, theirs, seed=7, setup=None):
        self.state = new_battle(DOUBLES_CONFIG, (_team(*ours), _team(*theirs)),
                                seed=seed)
        self.state, _ = step(self.state, Action.select(0, 1, 2, 3),
                             Action.select(0, 1, 2, 3))
        if setup is not None:
            setup(self)
        self.log: list = []

    def turn(self, ours, theirs):
        self.state, events = step(self.state, ours, theirs)
        self.log = list(events)
        return self

    # -- reading it back ---------------------------------------------------- #

    def slot(self, side, pos):
        return self.state.sides[side].active[pos]

    def hp(self, side, pos=0):
        return self.state.sides[side].hp[self.slot(side, pos)]

    def max_hp(self, side, pos=0):
        return _max_hp(self.state, (side, self.slot(side, pos)))

    def set_hp(self, side, pos, value):
        self.state.sides[side].hp[self.slot(side, pos)] = value

    def status(self, side, pos=0):
        return self.state.sides[side].status[self.slot(side, pos)]

    def set_status(self, side, pos, value):
        self.state.sides[side].status[self.slot(side, pos)] = value

    def boosts(self, side, pos=0):
        row = self.state.sides[side].boosts[self.slot(side, pos)]
        return {k: row[i] for k, i in BOOST_INDEX.items() if row[i]}

    def item(self, side, pos=0):
        return self.state.item_id(side, self.slot(side, pos))

    def ability(self, side, pos=0):
        return self.state.ability_id(side, self.slot(side, pos))

    def volatiles(self, side, pos=0):
        return set(self.state.sides[side].volatiles[self.slot(side, pos)])

    def damage(self, side, pos, move=None):
        """What landed on one field position this turn, 0 if nothing did."""
        want = self.slot(side, pos)
        total = 0
        for event in self.log:
            if event.kind != "damage" or (event.side or 0) != side:
                continue
            if (event.slot or 0) != want:
                continue
            if move is not None and event.move != move:
                continue
            total += event.amount or 0
        return total

    def said(self, *fragments) -> bool:
        text = " | ".join(str(one) for one in self.log)
        return all(f in text for f in fragments)


#: Two that stand there and take it, so nothing but the ability moves.
def _wall(species="snorlax"):
    return mon(species, "__none__", ("splash", "bodyslam", "protect", "rest"),
               None, "sassy", (32, 0, 32, 0, 32, 0))


def _walls():
    return [_wall(), _wall("blissey" if "blissey" in DEX.species else "snorlax")]


SPLASH_BOTH = (Action.move(0), Action.move(0))

DOUBLES_CHECKS: dict[str, tuple[str, object]] = {}


def doubles_check(ability: str, expect: str):
    def wrap(fn):
        DOUBLES_CHECKS[ability] = (expect, fn)
        return fn
    return wrap


def _ratio(with_it, without):
    return with_it / without if without else 0.0


def _ally_power_pair(ability, ally_move):
    """The partner swinging, with the ability beside it and without."""
    def played(which):
        holder = mon(_HOLDER[0], which, ("splash", "bodyslam", "rest", "protect"),
                     None, _HOLDER[1], _HOLDER[2])
        ally = mon("milotic", "__none__",
                   ("icebeam", "bodyslam", "splash", "protect"), None,
                   "quiet", (32, 32, 32, 32, 0, 0))
        f = Duo([holder, ally], _walls())
        f.turn((Action.move(0), Action.move(ally_move, target=0)), SPLASH_BOTH)
        return f.damage(1, 0)
    return played(ability), played("__none__")


@doubles_check("battery", "half again a third onto an ally's special move, "
                          "and nothing onto a physical one")
def _battery():
    special = _ratio(*_ally_power_pair("battery", 0))
    physical = _ratio(*_ally_power_pair("battery", 1))
    return (1.25 <= special <= 1.35 and 0.99 <= physical <= 1.01,
            f"the ally's special move x{special:.3f}, its physical one "
            f"x{physical:.3f}")


@doubles_check("powerspot", "a third onto whatever the ally swings")
def _power_spot():
    special = _ratio(*_ally_power_pair("powerspot", 0))
    physical = _ratio(*_ally_power_pair("powerspot", 1))
    return (1.25 <= special <= 1.35 and 1.25 <= physical <= 1.35,
            f"special x{special:.3f}, physical x{physical:.3f}")


@doubles_check("friendguard", "a quarter off what the ally takes, and none "
                              "off the holder's own")
def _friend_guard_check():
    def played(which):
        holder = mon(_HOLDER[0], which, ("splash", "bodyslam", "rest", "protect"),
                     None, _HOLDER[1], _HOLDER[2])
        ally = mon(_HOLDER[0], "__none__",
                   ("splash", "bodyslam", "rest", "protect"), None,
                   _HOLDER[1], _HOLDER[2])
        foe = mon("garchomp", "__none__",
                  ("dragonclaw", "splash", "protect", "willowisp"), None,
                  "jolly", (0, 32, 2, 0, 0, 32))
        f = Duo([holder, ally], [foe, foe])
        f.turn(SPLASH_BOTH, (Action.move(0, target=1), Action.move(0, target=0)))
        return f.damage(0, 1), f.damage(0, 0)
    (ally_with, self_with), (ally_without, self_without) = \
        played("friendguard"), played("__none__")
    ally = _ratio(ally_with, ally_without)
    mine = _ratio(self_with, self_without)
    return (0.72 <= ally <= 0.78 and 0.99 <= mine <= 1.01,
            f"the ally took x{ally:.3f}, the holder itself x{mine:.3f}")


@doubles_check("healer", "three turns in ten, the partner's status goes away")
def _healer_check():
    def burnt(state):
        state.set_status(0, 1, "brn")

    cured = 0
    tries = 600
    for seed in range(tries):
        f = Duo([mon(_HOLDER[0], "healer", ("splash", "bodyslam", "rest",
                                            "protect"), None, *_HOLDER[1:]),
                 mon(_HOLDER[0], "__none__", ("splash", "bodyslam", "rest",
                                              "protect"), None, *_HOLDER[1:])],
                _walls(), seed=seed, setup=burnt)
        f.turn(SPLASH_BOTH, SPLASH_BOTH)
        cured += f.status(0, 1) is None
    rate = cured / tries
    want = HEALER_CHANCE[0] / HEALER_CHANCE[1]
    spread = 3 * (want * (1 - want) / tries) ** 0.5
    return (abs(rate - want) <= spread,
            f"{cured} of {tries} cured, {rate:.3f} against {want:.3f}")


def _plus_minus_pair(mine, theirs):
    """The holder's own Special Attack, beside a partner and beside nobody."""
    def played(ally_ability):
        holder = mon("milotic", mine, ("icebeam", "bodyslam", "splash",
                                       "protect"), None, "quiet",
                     (32, 32, 32, 32, 0, 0))
        ally = mon(_HOLDER[0], ally_ability,
                   ("splash", "bodyslam", "rest", "protect"), None, *_HOLDER[1:])
        f = Duo([holder, ally], _walls())
        f.turn((Action.move(0, target=0), Action.move(0)), SPLASH_BOTH)
        return f.damage(1, 0)
    return played(theirs), played("__none__")


@doubles_check("plus", "half again the Special Attack, but only beside the "
                       "other half")
def _plus():
    paired = _ratio(*_plus_minus_pair("plus", "minus"))
    alone = _ratio(*_plus_minus_pair("plus", "__none__"))
    return (1.45 <= paired <= 1.55 and 0.99 <= alone <= 1.01,
            f"beside Minus x{paired:.3f}, beside nobody x{alone:.3f}")


@doubles_check("minus", "half again the Special Attack, but only beside the "
                        "other half")
def _minus():
    paired = _ratio(*_plus_minus_pair("minus", "plus"))
    alone = _ratio(*_plus_minus_pair("minus", "__none__"))
    return (1.45 <= paired <= 1.55 and 0.99 <= alone <= 1.01,
            f"beside Plus x{paired:.3f}, beside nobody x{alone:.3f}")


@doubles_check("sweetveil", "neither of them sleeps, and Yawn does not take")
def _sweet_veil():
    def played(which, move_index):
        holder = mon(_HOLDER[0], which, ("splash", "bodyslam", "rest",
                                         "protect"), None, *_HOLDER[1:])
        ally = mon(_HOLDER[0], "__none__", ("splash", "bodyslam", "rest",
                                            "protect"), None, *_HOLDER[1:])
        foe = mon("gengar", "__none__", ("spore", "yawn", "splash", "protect"),
                  None, "timid", (0, 0, 2, 32, 0, 32))
        f = Duo([holder, ally], [foe, _wall()])
        f.turn(SPLASH_BOTH,
               (Action.move(move_index, target=1), Action.move(0)))
        return f.status(0, 1), f.volatiles(0, 1)
    guarded_sleep, guarded_yawn = played("sweetveil", 0)[0], played("sweetveil", 1)[1]
    plain_sleep, plain_yawn = played("__none__", 0)[0], played("__none__", 1)[1]
    return (guarded_sleep is None and plain_sleep == "slp"
            and "yawn" not in guarded_yawn and "yawn" in plain_yawn,
            f"Spore at the ally left it {guarded_sleep} with Sweet Veil and "
            f"{plain_sleep} without; Yawn stuck={'yawn' in guarded_yawn} "
            f"with it, {'yawn' in plain_yawn} without")


@doubles_check("flowerveil", "a Grass ally takes no status and no stat drop")
def _flower_veil():
    def played(which):
        holder = mon("florges", which, ("splash", "bodyslam", "rest", "protect"),
                     None, *_HOLDER[1:])
        ally = mon("meganium" if "meganium" in DEX.species else "venusaur",
                   "__none__", ("splash", "bodyslam", "rest", "protect"),
                   None, *_HOLDER[1:])
        foe = mon("gengar", "__none__",
                  ("willowisp", "growl", "splash", "protect"), None, "timid",
                  (0, 0, 2, 32, 0, 32))
        f = Duo([holder, ally], [foe, _wall()])
        f.turn(SPLASH_BOTH, (Action.move(0, target=1), Action.move(0)))
        f.turn(SPLASH_BOTH, (Action.move(1), Action.move(0)))
        return f.status(0, 1), f.boosts(0, 1)
    guarded = played("flowerveil")
    plain = played("__none__")
    return (guarded == (None, {}) and plain[0] == "brn"
            and plain[1].get("atk", 0) < 0,
            f"the Grass ally came through on {guarded}; with no Flower Veil "
            f"beside it, {plain}")


@doubles_check("telepathy", "an ally's spread move goes around the holder")
def _telepathy():
    def played(which):
        holder = mon(_HOLDER[0], which, ("splash", "bodyslam", "rest",
                                         "protect"), None, *_HOLDER[1:])
        ally = mon("milotic", "__none__", ("surf", "bodyslam", "splash",
                                           "protect"), None, "quiet",
                   (32, 32, 32, 32, 0, 0))
        f = Duo([holder, ally], _walls())
        f.turn((Action.move(0), Action.move(0)), SPLASH_BOTH)
        return f.damage(0, 0)
    guarded, plain = played("telepathy"), played("__none__")
    return (guarded == 0 and plain > 0,
            f"Surf took {guarded} off the holder with Telepathy and {plain} "
            f"without")


@doubles_check("symbiosis", "the item goes across as soon as the ally spends "
                            "its own")
def _symbiosis_check():
    def played(which):
        holder = mon(_HOLDER[0], which, ("splash", "bodyslam", "rest",
                                         "protect"), "sitrusberry", *_HOLDER[1:])
        ally = mon(_HOLDER[0], "__none__", ("splash", "bodyslam", "rest",
                                            "protect"), "sitrusberry",
                   *_HOLDER[1:])

        def hurt(state):
            state.set_hp(0, 1, state.max_hp(0, 1) * 2 // 5)

        f = Duo([holder, ally], _walls(), setup=hurt)
        f.turn(SPLASH_BOTH, SPLASH_BOTH)
        return f.item(0, 0), f.item(0, 1)
    passed = played("symbiosis")
    kept = played("__none__")
    return (passed == (None, "sitrusberry") and kept == ("sitrusberry", None),
            f"with Symbiosis the two held {passed}, without it {kept}")


def _inherit_pair(which):
    """The partner goes down holding an ability worth taking."""
    holder = mon(_HOLDER[0], which, ("splash", "bodyslam", "rest", "protect"),
                 None, *_HOLDER[1:])
    ally = mon(_HOLDER[0], "intimidate", ("splash", "bodyslam", "rest",
                                          "protect"), None, *_HOLDER[1:])
    foe = mon("garchomp", "__none__", ("dragonclaw", "splash", "protect",
                                       "willowisp"), None, "jolly",
              (0, 32, 2, 0, 0, 32))

    def dying(state):
        state.set_hp(0, 1, 1)

    f = Duo([holder, ally], [foe, foe], setup=dying)
    f.turn(SPLASH_BOTH, (Action.move(0, target=1), Action.move(0, target=1)))
    return f.ability(0, 0), f.hp(0, 1)


@doubles_check("receiver", "takes up the fallen partner's ability")
def _receiver():
    got, ally_hp = _inherit_pair("receiver")
    plain, _ = _inherit_pair("__none__")
    return (ally_hp == 0 and got == "intimidate" and plain == "__none__",
            f"the ally went down ({ally_hp} left) and the holder became "
            f"{got!r}; without Receiver it stayed {plain!r}")


@doubles_check("powerofalchemy", "takes up the fallen partner's ability")
def _power_of_alchemy():
    got, ally_hp = _inherit_pair("powerofalchemy")
    plain, _ = _inherit_pair("__none__")
    return (ally_hp == 0 and got == "intimidate" and plain == "__none__",
            f"the ally went down ({ally_hp} left) and the holder became "
            f"{got!r}; without Power of Alchemy it stayed {plain!r}")


def _switch_in_pair(which, ally_moves, script, read):
    """Bench the holder, play the turn, then bring it in beside the ally."""
    def played(ability):
        ours = [mon("magikarp", "__none__", ("splash", "tackle")),
                mon(_HOLDER[0], "__none__", ally_moves, None, *_HOLDER[1:]),
                mon(_HOLDER[0], ability, ("splash", "bodyslam", "rest",
                                          "protect"), None, *_HOLDER[1:])]
        f = Duo(ours, _walls())
        for ours_actions in script:
            f.turn(ours_actions, SPLASH_BOTH)
        return read(f)
    return played(which), played("__none__")


@doubles_check("hospitality", "a quarter of the partner's health, on arrival")
def _hospitality_check():
    def read(f):
        return f.hp(0, 1), f.max_hp(0, 1)

    def hurt_then_swap(ability):
        ours = [mon("magikarp", "__none__", ("splash", "tackle")),
                mon(_HOLDER[0], "__none__", ("splash", "bodyslam", "rest",
                                             "protect"), None, *_HOLDER[1:]),
                mon(_HOLDER[0], ability, ("splash", "bodyslam", "rest",
                                          "protect"), None, *_HOLDER[1:])]

        def hurt(state):
            state.set_hp(0, 1, state.max_hp(0, 1) // 2)

        f = Duo(ours, _walls(), setup=hurt)
        before = f.hp(0, 1)
        f.turn((Action.switch(2), Action.move(0)), SPLASH_BOTH)
        return before, f.hp(0, 1), f.max_hp(0, 1)
    before, after, whole = hurt_then_swap("hospitality")
    _, plain, _ = hurt_then_swap("__none__")
    healed = after - before
    return (abs(healed - whole // HOSPITALITY_FRACTION) <= 1 and plain == before,
            f"the partner went {before} -> {after} of {whole}, a quarter being "
            f"{whole // HOSPITALITY_FRACTION}; with no ability it stayed {plain}")


@doubles_check("curiousmedicine", "wipes the partner's stages, the good ones "
                                  "too")
def _curious_medicine_check():
    got, plain = _switch_in_pair(
        "curiousmedicine",
        ("swordsdance", "bodyslam", "rest", "protect"),
        [(Action.move(0), Action.move(0)), (Action.switch(2), Action.move(1))],
        lambda f: f.boosts(0, 1))
    return (got == {} and plain.get("atk") == 2,
            f"after Swords Dance the partner held {plain} normally and {got} "
            f"with Curious Medicine beside it")


@doubles_check("costar", "arrives holding whatever the partner built up")
def _costar_check():
    got, plain = _switch_in_pair(
        "costar",
        ("swordsdance", "bodyslam", "rest", "protect"),
        [(Action.move(0), Action.move(0)), (Action.switch(2), Action.move(1))],
        lambda f: (f.boosts(0, 0), f.boosts(0, 1)))
    return (got[0] == got[1] == {"atk": 2} and plain[0] == {},
            f"the partner was on {got[1]} and the arrival copied {got[0]}; "
            f"without Costar it arrived on {plain[0]}")


def _redirection_pair(which):
    """Follow Me on the far left, and our move aimed at the far right."""
    def played(ability):
        holder = mon("weavile", ability, ("iceshard", "bodyslam", "splash",
                                          "protect"), None, "jolly",
                     (0, 32, 2, 0, 0, 32))
        ally = mon(_HOLDER[0], "__none__", ("splash", "bodyslam", "rest",
                                            "protect"), None, *_HOLDER[1:])
        puller = mon(_HOLDER[0], "__none__", ("followme", "splash", "rest",
                                              "protect"), None, *_HOLDER[1:])
        f = Duo([holder, ally], [puller, _wall()])
        f.turn((Action.move(0, target=1), Action.move(0)),
               (Action.move(0), Action.move(0)))
        return f.damage(1, 0), f.damage(1, 1)
    return played(which), played("__none__")


def _ignores_redirection(ability, label):
    (pulled_left, pulled_right), (plain_left, plain_right) = \
        _redirection_pair(ability)
    return (pulled_right > 0 and pulled_left == 0
            and plain_left > 0 and plain_right == 0,
            f"with {label} the hit landed right ({pulled_right}) rather than "
            f"on the puller ({pulled_left}); without it the puller took "
            f"{plain_left} and the target {plain_right}")


@doubles_check("propellertail", "Follow Me does not pull the holder's move")
def _propeller_tail():
    return _ignores_redirection("propellertail", "Propeller Tail")


@doubles_check("stalwart", "Follow Me does not pull the holder's move")
def _stalwart():
    return _ignores_redirection("stalwart", "Stalwart")


def run_doubles(verbose=False):
    """Every partner ability, on a field that has a partner."""
    missing = sorted(DOUBLES_ONLY - set(DOUBLES_CHECKS))
    failures = []
    for ability, (expect, probe) in sorted(DOUBLES_CHECKS.items()):
        try:
            ok, detail = probe()
        except Exception as error:            # a broken scenario is a failure
            ok, detail = False, f"{type(error).__name__}: {error}"
        if not ok:
            failures.append((ability, expect, detail))
        if verbose or not ok:
            mark = "ok  " if ok else "FAIL"
            print(f"  {mark} {ability:16s} {expect}")
            print(f"       {detail}")
    print(f"\ndoubles: {len(DOUBLES_CHECKS) - len(failures)}/"
          f"{len(DOUBLES_CHECKS)} partner abilities behave as described")
    if missing:
        print(f"  no doubles check written for: {', '.join(missing)}")
    return not failures and not missing


def _mega_stones():
    return sorted({s.required_item for s in DEX.species.values()
                   if getattr(s, "is_mega", False)
                   and getattr(s, "required_item", None)})


def _stone_holders(stone):
    """Everything that might hold this stone, and the forme it becomes.

    ``base_species`` is not always the forme that can use the stone:
    Floettite's mega records its base as ``floette``, and the Pokemon that
    actually Mega Evolves with it is Floette-Eternal. So every forme sharing
    that base is a candidate, and one of them working is the stone working.
    """
    mega = next((one for one in DEX.species.values()
                 if getattr(one, "is_mega", False)
                 and getattr(one, "required_item", None) == stone), None)
    if mega is None:
        return [], None
    family = [one.id for one in DEX.species.values()
              if not getattr(one, "is_mega", False)
              and getattr(one, "base_species", one.id) == mega.base_species]
    return family, mega.id


def check_stones():
    """Every stone does one job: let its own species Mega Evolve, and nobody
    else's. Seventy-four of them, so they are checked as a family."""
    bad, skipped = [], []
    for stone in _mega_stones():
        family, mega = _stone_holders(stone)
        if not family or not any(DEX.exists_in_champions(DEX.species[one])
                                 for one in family):
            skipped.append((stone, "nothing that can hold it is in the format"))
            continue
        worked = None
        for base in family:
            ours = [mon(base, DEX.species[base].abilities[0],
                        ("splash", "tackle", "protect", "rest"), stone)]
            theirs = [mon("magikarp", "__none__", ("splash", "tackle"))]
            state = new_battle(CONFIG, (tuple(ours) + tuple(FILLER),
                                        tuple(theirs) + tuple(FILLER)), seed=7)
            state, _ = step(state, Action.select(0, 1, 2),
                            Action.select(0, 1, 2))
            if not state.can_mega_evolve(0, state.sides[0].active[0]):
                continue
            state, _ = step(state, Action.move(0, mega=True), Action.move(0))
            became = state.species_id(0, state.sides[0].active[0])
            worked = (became == mega, base, became)
            if worked[0]:
                break
        if worked is None:
            bad.append((stone, f"none of {family} could Mega Evolve holding it"))
        elif not worked[0]:
            bad.append((stone, f"{worked[1]} became {worked[2]}, not {mega}"))
    return bad, skipped


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--abilities", action="store_true")
    parser.add_argument("--items", action="store_true")
    parser.add_argument("--stones", action="store_true")
    parser.add_argument("--doubles", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    everything = not (args.abilities or args.items or args.stones
                      or args.doubles)

    legal_items = {one["id"] for one in json.loads(
        (ROOT / "data" / "champions" / "items_m_b.json")
        .read_text(encoding="utf-8"))["items"]}
    stones = set(_mega_stones())

    if args.stones or everything:
        bad, skipped = check_stones()
        total = len(_mega_stones()) - len(skipped)
        print(f"mega stones: {total - len(bad)}/{total} let their own species "
              f"Mega Evolve  ({len(skipped)} not in the format, skipped)")
        for stone, why in bad:
            print(f"   FAIL {stone}: {why}")
        print()

    if args.abilities or everything:
        reachable = {a for species in DEX.species.values()
                     if DEX.exists_in_champions(species)
                     for a in species.abilities}
        have = set(registered("ability"))
        silent, seen = [], sorted(reachable & have)
        for ability in seen:
            if _differs(ability=ability) is not None:
                continue
            # Anything that turns on a roll -- Compound Eyes on an accuracy
            # check, Insomnia on a Hypnosis that has to land -- can go two
            # seeds without the roll ever going the other way. Only the ones
            # that came back silent pay for the wider search.
            if _differs(ability=ability, seeds=range(14)) is None:
                silent.append(ability)
        from pkcm.engine.abilities import INERT

        doubles = sorted(one for one in silent if one in DOUBLES_ONLY)
        silent = [one for one in silent if one not in DOUBLES_ONLY]
        # "Silent" said two things at once: nobody wrote it, and somebody
        # wrote it as nothing and said why. Only the first is a finding.
        inert = [one for one in silent if one in INERT]
        silent = [one for one in silent if one not in INERT]
        print(f"abilities: {len(seen)} registered and reachable, "
              f"{len(seen) - len(silent) - len(doubles) - len(inert)} changed "
              f"the battle they were in, {len(doubles)} need an ally, "
              f"{len(inert)} are inert on purpose, {len(silent)} still silent")
        for ability in inert:
            korean = NAMES.get("abilities", {}).get(ability, ability)
            print(f"   inert   {korean:<14} {ability}")
        for ability in silent:
            korean = NAMES.get("abilities", {}).get(ability, ability)
            print(f"   silent  {korean:<14} {ability}")
        for ability in doubles:
            korean = NAMES.get("abilities", {}).get(ability, ability)
            print(f"   doubles {korean:<14} {ability}")
        # "Reachable" is anything on a legal species; the roster is narrower.
        # Reporting only the first number made 49 look like 49 holes.
        roster = DEX.regulation("m_b")
        on_roster = {one
                     for species_id in roster.legal_species | roster.legal_megas
                     for one in DEX.species[species_id].abilities}
        pending = sorted(on_roster - have)
        print(f"\n   {len(reachable - have)} more are reachable with no "
              f"registered effect; {len(pending)} of those are on the legal "
              f"roster: {pending}")
        print()

    if args.doubles or everything:
        run_doubles(verbose=args.verbose)
        print()

    if args.items or everything:
        rest = sorted(legal_items - stones)
        have = set(registered("item"))
        silent = [one for one in rest
                  if one in have and _differs(item=one) is None]
        print(f"items: {len(rest)} legal non-stone items, "
              f"{len([o for o in rest if o in have])} with a registered effect, "
              f"{len(rest) - len(silent)} changed the battle they were in")
        for item in silent:
            korean = NAMES.get("items", {}).get(item, item)
            print(f"   silent  {korean:<14} {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
