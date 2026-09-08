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
from pkcm.data.dex import load_dex                      # noqa: E402
from pkcm.engine.actions import Action                  # noqa: E402
from pkcm.engine.battle import step                     # noqa: E402
from pkcm.engine.effects import registered              # noqa: E402
from pkcm.engine.pokemon import PokemonSet              # noqa: E402
from pkcm.engine.state import BattleConfig, new_battle  # noqa: E402

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


def _foe(moves, nature="jolly", sp=(0, 32, 2, 32, 0, 32), species="garchomp"):
    return mon(species, "__none__", moves, None, nature, sp)


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
        state, events = step(state, ours_action, theirs_action)
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
        plain = [mon(one.species, "__none__", one.moves, one.item,
                     one.nature, one.sp) for one in ours]
        for seed in seeds:
            try:
                with_it = trace_with(ours, theirs, script, seed, setup)
                without = trace_with(plain, theirs, script, seed, setup)
            except Exception:
                continue
            if with_it != without:
                return what
    return None


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
    args = parser.parse_args()
    everything = not (args.abilities or args.items or args.stones)

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
            if _differs(ability=ability) is None:
                silent.append(ability)
        print(f"abilities: {len(seen)} registered and reachable, "
              f"{len(seen) - len(silent)} changed the battle they were in")
        for ability in silent:
            korean = NAMES.get("abilities", {}).get(ability, ability)
            print(f"   silent  {korean:<14} {ability}")
        print(f"\n   and {len(reachable - have)} reachable abilities have no "
              f"registered effect at all")
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
