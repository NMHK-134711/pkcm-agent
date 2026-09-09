"""Held items, from the roster Champions actually has.

The list comes from ``data/champions/items_m_b.json`` -- op.gg's scrape of the
live game's item screen. It matters that it is short: Champions kept Choice
Scarf but cut Choice Band and Choice Specs, and cut Assault Vest, Eviolite,
Rocky Helmet, Air Balloon, Flame Orb, Toxic Orb, Heavy-Duty Boots and Weakness
Policy outright. Showdown's Champions mod agrees on every one of those, which is
about as much corroboration as two independent sources can give.

So 147 items: 72 battle items and 75 Mega Stones, and a third of the battle
items are berries. Behaviour is ported from ``data/reference/items.ts``.

Consumption is permanent: a used item is gone for the battle, not just until the
holder switches, so it writes through ``set_override(..., permanent=True)``.
"""

from __future__ import annotations

import json
from pathlib import Path

from pkcm.data.dex import Stat
from pkcm.engine import mutate
from pkcm.engine.abilities import announce
from pkcm.engine.effects import Context, Ref, register
from pkcm.engine.events import Event
from pkcm.engine.abilities import CONTACT
from pkcm.engine.moves import (X0_25, X0_5, X0_9, X1_1, X1_2, X1_3,
                               X1_5, X2, chain_modify)
from pkcm.engine.mutate import boost, consume_item, fraction_of_max, heal

ROSTER_PATH = Path(__file__).resolve().parents[3] / "data" / "champions" / "items_m_b.json"


def champions_items() -> set[str]:
    """Every item id Champions has, stones included.

    Three lists, because they have three different provenances: hk's scrape of
    the live game, the Mega Stones inferred from which formes are legal, and
    what a later update added on top.
    """
    data = json.loads(ROSTER_PATH.read_text(encoding="utf-8"))
    return ({entry["id"] for entry in data["items"]}
            | set(data.get("inferred_mega_stones", ()))
            | set(data.get("added_in_m_c", ())))


def used(ctx: Context, ref: Ref, item: str) -> None:
    ctx.emit(Event("item", side=ref[0], slot=ref[1], detail=item))
    # "This move cannot be selected until the user eats a Berry" -- and once
    # it has, it stays selectable for the rest of the battle, switches
    # included, which is why this lives in ``status_data``.
    if item.endswith("berry"):
        ctx.state.sides[ref[0]].status_data[ref[1]]["ateberry"] = True


# --------------------------------------------------------------------------- #
# Type-boosting items: 1.2x for one type
# --------------------------------------------------------------------------- #

TYPE_BOOSTERS = {
    "blackbelt": "fighting",
    "blackglasses": "dark",
    "charcoal": "fire",
    "dragonfang": "dragon",
    "fairyfeather": "fairy",
    "hardstone": "rock",
    "magnet": "electric",
    "metalcoat": "steel",
    "miracleseed": "grass",
    "mysticwater": "water",
    "nevermeltice": "ice",
    "poisonbarb": "poison",
    "sharpbeak": "flying",
    "silkscarf": "normal",
    "silverpowder": "bug",
    "softsand": "ground",
    "spelltag": "ghost",
    "twistedspoon": "psychic",
}


def _type_booster(move_type: str):
    def handler(ctx, ref, value, attacker, defender, move, **_):
        if ref == attacker and move.type == move_type:
            return chain_modify(value, X1_2)
        return None

    return handler


for _item, _type in TYPE_BOOSTERS.items():
    register("item", _item, name=_item.title(), modify_base_power=_type_booster(_type))


# --------------------------------------------------------------------------- #
# Flat power items
# --------------------------------------------------------------------------- #


def _category_booster(category: str, modifier: int):
    def handler(ctx, ref, value, attacker, defender, move, **_):
        if ref == attacker and move.category == category:
            return chain_modify(value, modifier)
        return None

    return handler


register("item", "muscleband", name="Muscle Band",
         modify_base_power=_category_booster("Physical", X1_1))
register("item", "wiseglasses", name="Wise Glasses",
         modify_base_power=_category_booster("Special", X1_1))


def _expert_belt(ctx, ref, value, attacker, defender, move, **_):
    from pkcm.engine.moves import type_effectiveness

    if ref != attacker or type_effectiveness(ctx, attacker, defender, move) <= 1.0:
        return None
    return chain_modify(value, X1_2)


register("item", "expertbelt", name="Expert Belt", modify_damage=_expert_belt)


def _life_orb_damage(ctx, ref, value, attacker, defender, move, **_):
    return chain_modify(value, X1_3) if ref == attacker else None


def _life_orb_recoil(ctx, ref, attacker, defender, move, damage, **_):
    if damage > 0:
        used(ctx, ref, "lifeorb")
        mutate.apply_damage(ctx, ref, fraction_of_max(ctx.state, ref, 10),
                            "recoil", detail="lifeorb")


register("item", "lifeorb", name="Life Orb",
         modify_damage=_life_orb_damage, dealt_damage=_life_orb_recoil)


def _metronome_power(ctx, ref, value, attacker, defender, move, **_):
    """Consecutive uses of one move ramp from 1.2x to 2x."""
    if ref != attacker:
        return None
    data = mutate.volatile(ctx.state, attacker, "metronome")
    if data is None or data.get("move") != move.id:
        return None
    steps = min(5, data.get("count", 0))
    if steps <= 0:
        return None
    return chain_modify(value, 4096 + steps * 819)  # +0.2 per repeat, capped at 2x


def _metronome_track(ctx, ref, move, move_index, **_):
    volatiles = ctx.state.sides[ref[0]].volatiles[ref[1]]
    data = volatiles.get("metronome")
    if data is not None and data.get("move") == move.id:
        data["count"] = data.get("count", 0) + 1
    else:
        volatiles["metronome"] = {"move": move.id, "count": 0}


register("item", "metronome", name="Metronome",
         modify_base_power=_metronome_power, commit_move=_metronome_track)
register("volatile", "metronome", name="Metronome count")
register("volatile", "lastmove", name="Last move")


def _light_ball(ctx, ref, value, **kwargs):
    if kwargs.get("stat") not in (Stat.ATK, Stat.SPA):
        return None
    if ctx.state.config.dex.species[ctx.state.species_id(*ref)].base_species != "pikachu":
        return None
    return chain_modify(value, X2)


register("item", "lightball", name="Light Ball", modify_stat=_light_ball)


def _big_root(ctx, ref, value, move, **_):
    """Draining moves give back 30% more -- Leech Seed and Strength Sap too."""
    return chain_modify(value, X1_3)


register("item", "bigroot", name="Big Root", modify_drain=_big_root)


# --------------------------------------------------------------------------- #
# Choice Scarf -- the only Choice item Champions kept
# --------------------------------------------------------------------------- #


def _choice_speed(ctx, ref, value, **kwargs):
    return chain_modify(value, X1_5) if kwargs.get("stat") is Stat.SPE else None


def _choice_lock(ctx, ref, move, move_index, **_):
    volatiles = ctx.state.sides[ref[0]].volatiles[ref[1]]
    if "choicelock" not in volatiles:
        volatiles["choicelock"] = {"move": move_index}


register("item", "choicescarf", name="Choice Scarf",
         modify_stat=_choice_speed, commit_move=_choice_lock)
register("volatile", "choicelock", name="Choice lock")


# --------------------------------------------------------------------------- #
# Survival and recovery
# --------------------------------------------------------------------------- #


def _leftovers(ctx, ref, **_):
    heal(ctx, ref, fraction_of_max(ctx.state, ref, 16), reason="leftovers")


register("item", "leftovers", name="Leftovers", residual=_leftovers)


def _shell_bell(ctx, ref, attacker, defender, move, damage, **_):
    restored = heal(ctx, ref, max(1, damage // 8), reason="shellbell")
    if restored:
        used(ctx, ref, "shellbell")


register("item", "shellbell", name="Shell Bell", dealt_damage=_shell_bell)


def _focus_sash(ctx, ref, value, attacker, defender, move, **_):
    """Survives one lethal hit, from full health, then is gone."""
    if ref != defender:
        return None
    current = mutate.current_hp(ctx.state, defender)
    if current != mutate.max_hp(ctx.state, defender) or value < current:
        return None
    used(ctx, defender, "focussash")
    consume_item(ctx, defender, "focussash")
    return current - 1


register("item", "focussash", name="Focus Sash", priority=30, modify_damage=_focus_sash)


def _focus_band(ctx, ref, value, attacker, defender, move, **_):
    """A 10% chance at the same thing, and it is not used up."""
    if ref != defender:
        return None
    current = mutate.current_hp(ctx.state, defender)
    if value < current or not ctx.cursor.chance(1, 10):
        return None
    used(ctx, defender, "focusband")
    return current - 1


register("item", "focusband", name="Focus Band", priority=30, modify_damage=_focus_band)


# --------------------------------------------------------------------------- #
# Accuracy, criticals and turn order
# --------------------------------------------------------------------------- #


register("item", "widelens", name="Wide Lens",
         modify_accuracy=lambda ctx, ref, value, attacker, defender, move, **_:
             chain_modify(int(value), X1_1) if ref == attacker else None)
register("item", "brightpowder", name="Bright Powder",
         modify_accuracy=lambda ctx, ref, value, attacker, defender, move, **_:
             chain_modify(int(value), X0_9) if ref == defender else None)
register("item", "scopelens", name="Scope Lens",
         modify_crit_ratio=lambda ctx, ref, value, attacker, defender, move, **_:
             value + 1 if ref == attacker else None)


def _zoom_lens(ctx, ref, value, attacker, defender, move, **_):
    """+20% accuracy, but only when the holder is moving *after* its target.

    Showdown asks ``!this.queue.willMove(target)`` -- has the target already
    gone? We keep the same question on the context: ``ctx.acted`` is the set of
    Pokemon that have taken their action this turn, which exists because
    Analytic needed exactly this and nothing else can reconstruct it.
    """
    if ref != attacker or defender not in ctx.acted:
        return None
    return chain_modify(int(value), X1_2)


# Missing until the pokechams dex turned it up: the op.gg item scrape we built
# the item list from does not have it, so it was never on the list to implement.
register("item", "zoomlens", name="Zoom Lens", modify_accuracy=_zoom_lens)


def _kings_rock(ctx, ref, attacker, defender, move, damage, **_):
    if damage > 0 and move.category != "Status" and ctx.cursor.chance(1, 10):
        used(ctx, ref, "kingsrock")
        mutate.add_volatile(ctx, defender, "flinch", source=ref)


register("item", "kingsrock", name="King's Rock", dealt_damage=_kings_rock)


def _quick_claw(ctx, ref, value, move, **_):
    if ctx.cursor.chance(1, 5):
        used(ctx, ref, "quickclaw")
        return value + 1
    return None


#: To the front of its own bracket, not out of it: Quick Claw has never let
#: anything beat an Extreme Speed.
register("item", "quickclaw", name="Quick Claw",
         modify_fractional_priority=_quick_claw)


def _iron_ball(ctx, ref, value, **kwargs):
    return chain_modify(value, X0_5) if kwargs.get("stat") is Stat.SPE else None


#: Iron Ball also grounds its holder; ``conditions.is_grounded`` reads the item.
register("item", "ironball", name="Iron Ball", modify_boosted_stat=_iron_ball)

#: Shed Shell lets its holder switch out of anything. Read directly in
#: ``state.legal_actions``, the same way Showdown reads it in ``isTrapped``,
#: because that question is asked without a battle context to hand.
register("item", "shedshell", name="Shed Shell")


# --------------------------------------------------------------------------- #
# Field extenders
# --------------------------------------------------------------------------- #

FIELD_EXTENDERS = {
    "damprock": ("raindance", "weather"),
    "heatrock": ("sunnyday", "weather"),
    "icyrock": ("snowscape", "weather"),
    "smoothrock": ("sandstorm", "weather"),
    "terrainextender": (None, "terrain"),
}


def _extender(target: str | None, kind: str):
    def handler(ctx, ref, value, field, kind_arg=None, **kwargs):
        if kwargs.get("kind") != kind:
            return None
        if target is not None and field != target:
            return None
        return value + 3

    return handler


for _item, (_target, _kind) in FIELD_EXTENDERS.items():
    register("item", _item, name=_item.title(), modify_field_duration=_extender(_target, _kind))


register("item", "lightclay", name="Light Clay",
         modify_field_duration=lambda ctx, ref, value, field, **kwargs:
             value + 3 if kwargs.get("kind") == "side"
             and field in ("reflect", "lightscreen", "auroraveil") else None)


# --------------------------------------------------------------------------- #
# Herbs
# --------------------------------------------------------------------------- #

MENTAL_HERB_CURES = ("attract", "taunt", "encore", "torment", "disable", "healblock")


def _mental_herb(ctx, ref, **_):
    volatiles = ctx.state.sides[ref[0]].volatiles[ref[1]]
    afflicted = [name for name in MENTAL_HERB_CURES if name in volatiles]
    if not afflicted:
        return
    used(ctx, ref, "mentalherb")
    for name in afflicted:
        mutate.remove_volatile(ctx, ref, name)
    consume_item(ctx, ref, "mentalherb")


register("item", "mentalherb", name="Mental Herb", update=_mental_herb)


def _white_herb(ctx, ref, **_):
    side = ctx.state.sides[ref[0]]
    stages = side.boosts[ref[1]]
    if all(stage >= 0 for stage in stages):
        return
    used(ctx, ref, "whiteherb")
    side.boosts[ref[1]] = [max(0, stage) for stage in stages]
    ctx.emit(Event("boost_restored", side=ref[0], slot=ref[1], detail="whiteherb"))
    consume_item(ctx, ref, "whiteherb")


register("item", "whiteherb", name="White Herb", update=_white_herb)


# --------------------------------------------------------------------------- #
# Berries
# --------------------------------------------------------------------------- #

#: Halve one super-effective hit of this type, then be eaten.
RESIST_BERRIES = {
    "babiriberry": "steel",
    "chartiberry": "rock",
    "chilanberry": "normal",
    "chopleberry": "fighting",
    "cobaberry": "flying",
    "colburberry": "dark",
    "habanberry": "dragon",
    "kasibberry": "ghost",
    "kebiaberry": "poison",
    "occaberry": "fire",
    "passhoberry": "water",
    "payapaberry": "psychic",
    "rindoberry": "grass",
    "roseliberry": "fairy",
    "shucaberry": "ground",
    "tangaberry": "bug",
    "wacanberry": "electric",
    "yacheberry": "ice",
}


#: What a berry *does*, apart from the condition that makes it fire.
#:
#: Cud Chew eats one again a turn later and Harvest grows it back, and neither
#: of them re-asks the question that made it fire the first time -- Champions'
#: own dex says 되새김질 is "같은 나무열매를 한 번 더 먹는다", which is an eat
#: and not a check. So the two halves have to be separable.
#:
#: Resist berries are absent on purpose: their effect is a damage modifier on
#: the hit that triggered them, so there is nothing to re-apply at end of turn.
BERRY_EFFECTS: dict[str, Any] = {}


def berry_effect(berry: str):
    """Register the effect half, and hand it back for the trigger half to use."""
    def keep(effect):
        BERRY_EFFECTS[berry] = effect
        return effect

    return keep


def ripened(ctx: Context, ref: Ref) -> bool:
    """Whether this one's berries come out twice as strong.

    Champions on Ripen: 나무열매의 효과가 2배가 된다. In this format the
    berries that have a size are the two that heal and the eighteen that
    halve a super-effective hit; the status and PP berries have nothing to
    double. It was the last ability on the roster with no implementation.
    """
    return ctx.ability_of(ref) == "ripen"


def eat_berry(ctx: Context, ref: Ref, berry: str) -> bool:
    """Apply a berry's effect without it being held. Cud Chew's whole job."""
    effect = BERRY_EFFECTS.get(berry)
    if effect is None:
        return False
    used(ctx, ref, berry)
    effect(ctx, ref)
    return True


def _resist_berry(berry_type: str, berry: str):
    def handler(ctx, ref, value, attacker, defender, move, **_):
        from pkcm.engine.moves import type_effectiveness

        if ref != defender or move.type != berry_type:
            return None
        # Chilan Berry is the odd one: it works on any Normal hit, not just a
        # super-effective one, because nothing is weak to Normal.
        if berry != "chilanberry" and type_effectiveness(ctx, attacker, defender, move) <= 1.0:
            return None
        used(ctx, defender, berry)
        consume_item(ctx, defender, berry)
        # Ripen halves it twice over, which is the quarter the ability's own
        # constant is there for.
        return chain_modify(value, X0_25 if ripened(ctx, defender) else X0_5)

    return handler


for _berry, _type in RESIST_BERRIES.items():
    register("item", _berry, name=_berry.title(), modify_damage=_resist_berry(_type, _berry))


STATUS_BERRIES = {
    "cheriberry": "par",
    "chestoberry": "slp",
    "pechaberry": "psn",
    "rawstberry": "brn",
    "aspearberry": "frz",
}


def _status_berry(status: str, berry: str):
    @berry_effect(berry)
    def effect(ctx, ref):
        mutate.cure_status(ctx, ref)

    def handler(ctx, ref, **_):
        current = ctx.state.sides[ref[0]].status[ref[1]]
        if current != status and not (status == "psn" and current == "tox"):
            return
        used(ctx, ref, berry)
        effect(ctx, ref)
        consume_item(ctx, ref, berry)

    return handler


for _berry, _status in STATUS_BERRIES.items():
    register("item", _berry, name=_berry.title(), update=_status_berry(_status, _berry))


@berry_effect("lumberry")
def _lum_effect(ctx, ref):
    mutate.cure_status(ctx, ref)
    if ctx.state.sides[ref[0]].has_volatile(ref[1], "confusion"):
        mutate.remove_volatile(ctx, ref, "confusion")


def _lum_berry(ctx, ref, **_):
    side = ctx.state.sides[ref[0]]
    if side.status[ref[1]] is None and not side.has_volatile(ref[1], "confusion"):
        return
    used(ctx, ref, "lumberry")
    _lum_effect(ctx, ref)
    consume_item(ctx, ref, "lumberry")


register("item", "lumberry", name="Lum Berry", update=_lum_berry)


@berry_effect("persimberry")
def _persim_effect(ctx, ref):
    mutate.remove_volatile(ctx, ref, "confusion")


def _persim_berry(ctx, ref, **_):
    if not ctx.state.sides[ref[0]].has_volatile(ref[1], "confusion"):
        return
    used(ctx, ref, "persimberry")
    _persim_effect(ctx, ref)
    consume_item(ctx, ref, "persimberry")


register("item", "persimberry", name="Persim Berry", update=_persim_berry)


def _healing_berry(berry: str, amount, threshold: int = 2):
    @berry_effect(berry)
    def effect(ctx, ref):
        given = amount(mutate.max_hp(ctx.state, ref))
        heal(ctx, ref, given * 2 if ripened(ctx, ref) else given, reason=berry)

    def handler(ctx, ref, **_):
        total = mutate.max_hp(ctx.state, ref)
        if mutate.current_hp(ctx.state, ref) * threshold > total:
            return
        used(ctx, ref, berry)
        effect(ctx, ref)
        consume_item(ctx, ref, berry)

    return handler


register("item", "oranberry", name="Oran Berry",
         update=_healing_berry("oranberry", lambda total: 10))
register("item", "sitrusberry", name="Sitrus Berry",
         update=_healing_berry("sitrusberry", lambda total: max(1, total // 4)))


@berry_effect("leppaberry")
def _leppa_effect(ctx, ref):
    side = ctx.state.sides[ref[0]]
    for index, remaining in enumerate(side.pp[ref[1]]):
        if remaining == 0:
            side.pp[ref[1]][index] = 10
            return


def _leppa_berry(ctx, ref, **_):
    side = ctx.state.sides[ref[0]]
    if 0 not in side.pp[ref[1]]:
        return
    used(ctx, ref, "leppaberry")
    _leppa_effect(ctx, ref)
    consume_item(ctx, ref, "leppaberry")


register("item", "leppaberry", name="Leppa Berry", update=_leppa_berry)


def register_mega_stones() -> None:
    """Stones do nothing until Mega Evolution exists, but they must be known.

    Registering them keeps the coverage report from calling 63 of the roster's
    135 items missing when what is actually missing is the mechanic.
    """
    from pkcm.data.dex import load_dex

    for item in load_dex().items.values():
        if item.mega_stone:
            register("item", item.id, name=item.name)


register_mega_stones()


# --------------------------------------------------------------------------- #
# The 2026-09-09 update's items
# --------------------------------------------------------------------------- #

#: "If held by a Farfetch'd or Sirfetch'd" -- and their formes, which is why
#: this is the base species rather than the id.
LEEK_HOLDERS = frozenset({"farfetchd", "farfetchdgalar", "sirfetchd"})


def _leek(ctx, ref, value, attacker, defender, move, **_):
    if ref != attacker:
        return None
    from pkcm.engine.legality import base_species_of

    species = ctx.state.species_id(*ref)
    if base_species_of(ctx.state.config.dex, species) not in LEEK_HOLDERS:
        return None
    return value + 2


register("item", "leek", name="Leek", modify_crit_ratio=_leek)


def _rocky_helmet(ctx, ref, attacker, defender, move, damage, **_):
    """A sixth of the attacker's maximum, for touching the holder."""
    if ref != defender or damage <= 0 or CONTACT not in move.flags:
        return
    used(ctx, ref, "rockyhelmet")
    mutate.apply_damage(ctx, attacker,
                        fraction_of_max(ctx.state, attacker, 6),
                        "item", detail="rockyhelmet")


register("item", "rockyhelmet", name="Rocky Helmet", after_damage=_rocky_helmet)


def _air_balloon(ctx, ref, attacker, defender, move, damage, **_):
    """It pops on any hit; the floating is read straight off the item.

    ``conditions.is_grounded`` already asks whether the holder is carrying one,
    so the Ground immunity needed nothing here. What it did need is losing the
    balloon, which nothing was doing -- so a holder floated for the whole
    battle however many times it was hit.
    """
    if ref != defender or damage <= 0:
        return
    used(ctx, ref, "airballoon")
    consume_item(ctx, ref, "airballoon")


register("item", "airballoon", name="Air Balloon", after_damage=_air_balloon)


def _red_card(ctx, ref, attacker, defender, move, damage, **_):
    """Survive a hit and the attacker is dragged out, not asked to leave."""
    from pkcm.engine import tactics

    if ref != defender or damage <= 0:
        return
    if ctx.state.sides[ref[0]].is_fainted(ref[1]):
        return
    used(ctx, ref, "redcard")
    consume_item(ctx, ref, "redcard")
    tactics.force_switch(ctx, attacker)


register("item", "redcard", name="Red Card", after_damage=_red_card)


def _eject_button(ctx, ref, attacker, defender, move, damage, **_):
    """The holder leaves, and its player picks the replacement.

    Marking the position rather than dragging: Eject Button is a choice, which
    is the difference between it and the Red Card above.
    """
    if ref != defender or damage <= 0:
        return
    side = ctx.state.sides[ref[0]]
    if side.is_fainted(ref[1]):
        return
    if not [slot for slot in side.living_slots() if slot not in side.active]:
        return
    position = side.position_of(ref[1])
    if position is None:
        return
    used(ctx, ref, "ejectbutton")
    consume_item(ctx, ref, "ejectbutton")
    side.must_switch[position] = True
    ctx.emit(Event("self_switch", side=ref[0], slot=ref[1]))


register("item", "ejectbutton", name="Eject Button", after_damage=_eject_button)


def _normal_gem(ctx, ref, value, attacker, defender, move, **_):
    """One Normal attack at 1.3, then gone.

    Spent in ``modify_base_power`` rather than after the hit, because the
    boost and the spending are the same event: a Gem that survived its own
    move would boost every Normal attack for the rest of the battle.
    """
    if ref != attacker or move.type != "normal" or move.category == "Status":
        return None
    used(ctx, ref, "normalgem")
    consume_item(ctx, ref, "normalgem")
    return chain_modify(int(value), X1_3)


register("item", "normalgem", name="Normal Gem", modify_base_power=_normal_gem)


def _seed(terrain: str, stat: str, item_id: str):
    """The four terrain seeds: one stage, once, while that terrain is up.

    On ``switch_in`` for a holder arriving onto the terrain, and on ``update``
    for a holder standing on it when the terrain arrives -- the same
    checkpoint the berries watch. Both, because either order happens.
    """
    def handler(ctx, ref, **_):
        if ctx.state.field.terrain != terrain:
            return
        if ctx.state.item_id(*ref) != item_id:
            return
        used(ctx, ref, item_id)
        consume_item(ctx, ref, item_id)
        boost(ctx, ref, {stat: 1}, source=ref)

    return handler


for _seed_item, _terrain, _stat in (
        ("electricseed", "electricterrain", "def"),
        ("grassyseed", "grassyterrain", "def"),
        ("psychicseed", "psychicterrain", "spd"),
        ("mistyseed", "mistyterrain", "spd")):
    register("item", _seed_item, name=_seed_item.title(),
             switch_in=_seed(_terrain, _stat, _seed_item),
             update=_seed(_terrain, _stat, _seed_item))


#: Engine-side, like Shed Shell above. What Binding Band changes is the
#: fraction a binding move takes each turn, and that is decided in
#: ``tactics._trapping_residual``, which reads the item off whoever did the
#: binding -- so there is nothing for a handler here to do.
register("item", "bindingband", name="Binding Band")
