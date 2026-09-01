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
from pkcm.engine.moves import X0_5, X0_9, X1_1, X1_2, X1_3, X1_5, X2, chain_modify
from pkcm.engine.mutate import boost, consume_item, fraction_of_max, heal

ROSTER_PATH = Path(__file__).resolve().parents[3] / "data" / "champions" / "items_m_b.json"


def champions_items() -> set[str]:
    """Every item id Champions has, stones included."""
    data = json.loads(ROSTER_PATH.read_text(encoding="utf-8"))
    return {entry["id"] for entry in data["items"]} | set(data["inferred_mega_stones"])


def used(ctx: Context, ref: Ref, item: str) -> None:
    ctx.emit(Event("item", side=ref[0], slot=ref[1], detail=item))


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


register("item", "quickclaw", name="Quick Claw", modify_priority=_quick_claw)


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
        return chain_modify(value, X0_5)

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
        heal(ctx, ref, amount(mutate.max_hp(ctx.state, ref)), reason=berry)

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
