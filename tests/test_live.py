"""Advising on a game we are not simulating.

The mirror's opponent is a placeholder, and the whole design rests on nobody
ever reading it. These tests are mostly about that: what the person at the
keyboard says goes into the state, and what they have not said stays out of it.
"""

from __future__ import annotations

import pytest

from pkcm.data.dex import load_dex
from pkcm.engine.actions import Action
from pkcm.engine.pokemon import PokemonSet
from pkcm.engine.state import Phase
from pkcm.envs.observation import Observation
from pkcm.live import Mirror, MirrorError


@pytest.fixture(scope="module")
def dex():
    return load_dex()


@pytest.fixture(scope="module")
def our_team():
    def one(species, ability, moves, item, nature, sp):
        return PokemonSet(species=species, ability=ability, moves=tuple(moves),
                          item=item, nature=nature, sp=sp)

    return (
        one("garchomp", "roughskin",
            ("earthquake", "dragonclaw", "stealthrock", "firefang"),
            "choicescarf", "jolly", (2, 32, 0, 0, 0, 32)),
        one("corviknight", "pressure",
            ("bravebird", "bodypress", "roost", "ironhead"),
            "leftovers", "impish", (32, 0, 32, 0, 2, 0)),
        one("primarina", "torrent",
            ("moonblast", "surf", "psychic", "calmmind"),
            "sitrusberry", "modest", (4, 0, 0, 32, 30, 0)),
        one("aegislash", "stancechange",
            ("shadowball", "shadowsneak", "kingsshield", "flashcannon"),
            "focussash", "naughty", (6, 30, 0, 30, 0, 0)),
        one("volcarona", "flamebody",
            ("fierydance", "quiverdance", "willowisp", "morningsun"),
            "lumberry", "modest", (4, 0, 0, 30, 0, 32)),
        one("hippowdon", "sandstream",
            ("earthquake", "slackoff", "stealthrock", "icefang"),
            "rockyhelmet", "impish", (32, 2, 32, 0, 0, 0)),
    )


#: Six a person might read off a real team preview.
THEIR_SIX = ("gengar", "kangaskhan", "starmie", "clefable", "meowscarada",
             "archaludon")


def a_mirror(dex, our_team, seed=1):
    return Mirror.begin(dex, dex.regulation("m_b"), our_team, THEIR_SIX,
                        seed=seed)


def open_battle(mirror, lead="gengar", ours=(0, 1, 2)):
    mirror.choose_ours(ours)
    mirror.their_lead(lead)
    mirror.open()
    return mirror


def test_it_starts_at_preview_knowing_only_their_six(dex, our_team):
    mirror = a_mirror(dex, our_team)
    assert mirror.phase is Phase.TEAM_PREVIEW
    observation = Observation.of(mirror.state, 0)
    assert observation.registered[1] == THEIR_SIX
    assert not mirror.state.revealed[1].species, (
        "nothing of theirs is revealed before the battle opens")


def test_the_search_cannot_read_the_placeholder(dex, our_team):
    """The one property that makes this worth having.

    The placeholder exists so the engine has something to step. If any of it
    reached the observation, the advice would be built on a set we invented and
    would look excellent right up until it met a real opponent.
    """
    mirror = open_battle(a_mirror(dex, our_team))
    observation = Observation.of(mirror.state, 0)
    for known in observation.foe:
        if known.species_id is None:
            continue                      # never been out; nothing to leak
        assert not known.moves, "a move leaked before it was used"
        assert known.item is None and not known.item_known
        assert known.stats is None, "their spread is not ours to know"
    benched = [k for k in observation.foe if k.species_id is None]
    assert benched, "the two they have not shown should still be hidden"


def test_a_move_we_had_not_guessed_is_taught_and_played(dex, our_team):
    """Their placeholder's four moves are a guess. When the guess is wrong the
    engine cannot step the turn at all, so the watched move has to go on."""
    mirror = open_battle(a_mirror(dex, our_team))
    before = [m.id for m in mirror.state.active_pokemon(1).moves]
    assert "painsplit" not in before, "pick a move the placeholder lacks"

    theirs = mirror.report_move("painsplit")
    assert [m.id for m in mirror.state.active_pokemon(1).moves][theirs.index] \
        == "painsplit"
    # Stealth Rock rather than Earthquake: a Scarf Garchomp knocks this Gengar
    # out before it moves, and a move that never went off is never revealed.
    mirror.advance(Action.move(2), theirs)
    assert "painsplit" in mirror.state.revealed[1].moves_of(
        mirror.state.sides[1].active[0])


def test_a_watched_move_is_never_overwritten_by_a_later_guess(dex, our_team):
    """Watched moves are facts. Teaching a fifth has to replace a guess."""
    mirror = open_battle(a_mirror(dex, our_team))
    slot = mirror.state.sides[1].active[0]
    for move in ("painsplit", "shadowball", "sludgebomb"):
        mirror.report_move(move)
        mirror.state.revealed[1].saw_move(slot, move)
    kept = [m.id for m in mirror.state.active_pokemon(1).moves]
    assert {"painsplit", "shadowball", "sludgebomb"} <= set(kept)


def test_a_pokemon_they_bring_that_we_did_not_guess_is_revealed(dex, our_team):
    """Preview says which six they registered, never which three they bring.

    Two of their three start as arbitrary picks, and the first time one is
    contradicted by a Pokemon walking out, the placeholder is swapped.
    """
    mirror = open_battle(a_mirror(dex, our_team), lead="gengar")
    selection = mirror.state.sides[1].selection
    absent = next(species for index, species in enumerate(THEIR_SIX)
                  if index not in selection)

    theirs = mirror.report_switch(absent)
    assert mirror.state.pokemon(1, theirs.index).species.id == absent
    mirror.advance(Action.move(0), theirs)
    assert mirror.state.species_id(1, mirror.state.sides[1].active[0]) == absent


def test_a_switch_to_one_we_did_guess_keeps_its_slot(dex, our_team):
    mirror = open_battle(a_mirror(dex, our_team), lead="gengar")
    already = mirror.state.sides[1].selection[1]
    theirs = mirror.report_switch(THEIR_SIX[already])
    assert theirs.index == 1, "an already-selected Pokemon keeps its slot"


def test_the_screen_overrides_our_arithmetic(dex, our_team):
    """Our damage roll is not the game's, so the bars are the truth."""
    mirror = open_battle(a_mirror(dex, our_team))
    mirror.observe(1, hp_fraction=0.35)
    side = mirror.state.sides[1]
    slot = side.active[0]
    maximum = mirror.state.pokemon(1, slot).max_hp
    assert side.hp[slot] == pytest.approx(round(maximum * 0.35), abs=1)

    mirror.observe(0, hp_fraction=0.5, status="brn")
    ours = mirror.state.sides[0]
    assert ours.status[ours.active[0]] == "brn"
    assert mirror.state.revealed[0].status_since[ours.active[0]] is not None


def test_a_sliver_of_health_is_not_a_faint(dex, our_team):
    """The bar shows 1% at a hundredth of a point. Rounding that to zero would
    have the mirror call a battle over that is still going."""
    mirror = open_battle(a_mirror(dex, our_team))
    mirror.observe(1, hp_fraction=0.004)
    slot = mirror.state.sides[1].active[0]
    assert mirror.state.sides[1].hp[slot] == 1
    mirror.observe(1, hp_fraction=0.0)
    assert mirror.state.sides[1].hp[slot] == 0


def test_it_refuses_a_report_it_cannot_reconcile(dex, our_team):
    mirror = open_battle(a_mirror(dex, our_team))
    with pytest.raises(MirrorError, match="registered"):
        mirror.report_switch("pikachu")
    with pytest.raises(MirrorError, match="no such move"):
        mirror.report_move("notamove")
    with pytest.raises(MirrorError, match="team preview shows"):
        Mirror.begin(dex, dex.regulation("m_b"), our_team, THEIR_SIX[:4])


def test_it_advises_a_legal_action(dex, our_team):
    from pkcm.search import MCTS, SearchConfig

    mirror = open_battle(a_mirror(dex, our_team))
    result = mirror.advise(MCTS(SearchConfig(iterations=60, determinizations=4)))
    assert result.action[0] in mirror.our_options()


def test_their_mega_evolution_can_be_reported(dex, our_team):
    """determinize never hands an unrevealed Pokemon a Mega Stone -- one that
    never fires would have the search planning around a Mega Evolution that
    cannot happen. That is the right default, and it is exactly why a Mega
    that *does* fire has to be sayable: the placeholder is holding something
    else and the engine will not run the turn."""
    mirror = open_battle(a_mirror(dex, our_team), lead="gengar")
    slot = mirror.state.sides[1].active[0]
    # The pool sometimes draws a Gengar already holding its stone, which would
    # let this pass without the fix. Take it away so the case under test is the
    # one that matters.
    mirror._rewrite(mirror.state.sides[1].selection[slot], item="leftovers")
    assert mirror.state.pokemon(1, slot).item == "leftovers"

    theirs = mirror.report_move("shadowball", mega=True)
    assert theirs.mega
    assert mirror.state.pokemon(1, slot).item == "gengarite"
    mirror.advance(Action.move(2), theirs)

    assert mirror.state.mega_used[1]
    assert mirror.state.species_id(1, mirror.state.sides[1].active[0]) \
        == "gengarmega"
    # And it is a fact from here on: the observation reports the Mega, so
    # every later determinization builds on it.
    seen = Observation.of(mirror.state, 0)
    assert any(known.species_id == "gengarmega" for known in seen.foe)


def test_a_second_mega_evolution_is_refused(dex, our_team):
    mirror = open_battle(a_mirror(dex, our_team), lead="gengar")
    mirror.advance(Action.move(2), mirror.report_move("shadowball", mega=True))
    with pytest.raises(MirrorError, match="이미"):
        mirror.report_move("sludgebomb", mega=True)


def test_a_species_with_no_mega_is_refused(dex, our_team):
    mirror = open_battle(a_mirror(dex, our_team), lead="archaludon")
    with pytest.raises(MirrorError, match="메가진화가 없습니다"):
        mirror.report_move("flashcannon", mega=True)


def test_a_correction_after_a_switch_lands_on_who_came_in(dex, our_team):
    """hk hit this: switch, and the Pokemon whose HP you are typing is the one
    that arrived, not the one that left. The write has to follow the field."""
    mirror = open_battle(a_mirror(dex, our_team), lead="gengar")
    before = mirror.state.sides[0].active[0]
    mirror.advance(Action.switch(1), mirror.report_move("shadowball"))
    after = mirror.state.sides[0].active[0]
    assert after != before, "the switch happened"

    left_at = mirror.state.sides[0].hp[before]
    mirror.observe(0, hp_fraction=0.8)
    arrived = mirror.state.pokemon(0, after)
    assert mirror.state.sides[0].hp[after] == pytest.approx(
        round(arrived.max_hp * 0.8), abs=1), "the correction missed the arrival"
    assert mirror.state.sides[0].hp[before] == left_at, (
        "and it must not have touched the one that left")


def test_a_correction_follows_their_switch_too(dex, our_team):
    mirror = open_battle(a_mirror(dex, our_team), lead="gengar")
    absent = next(species for index, species in enumerate(THEIR_SIX)
                  if index not in mirror.state.sides[1].selection)
    mirror.advance(Action.move(2), mirror.report_switch(absent))
    slot = mirror.state.sides[1].active[0]
    assert mirror.state.species_id(1, slot) == absent

    mirror.observe(1, hp_fraction=0.5)
    maximum = mirror.state.pokemon(1, slot).max_hp
    assert mirror.state.sides[1].hp[slot] == pytest.approx(
        round(maximum * 0.5), abs=1)


def test_a_revealed_ability_narrows_what_they_might_be(dex, our_team):
    """Worth more than the display suggests: belief.consistent filters the
    ranker pool on ability_known, so one of these can cut the set of Pokemon
    the search thinks it is facing to a handful."""
    from pkcm.envs.belief import candidates

    mirror = open_battle(a_mirror(dex, our_team), lead="clefable")
    slot = mirror.state.sides[1].active[0]
    known = Observation.of(mirror.state, 0).foe[slot]
    before = len(candidates("clefable", known))

    mirror.report_ability("magicguard")
    known = Observation.of(mirror.state, 0).foe[slot]
    assert known.ability_known and known.ability == "magicguard"
    after = len(candidates("clefable", known))
    assert after <= before, "knowing the ability cannot widen the pool"
    assert all(one.ability == "magicguard"
               for one in candidates("clefable", known))


def test_a_revealed_item_narrows_it_too(dex, our_team):
    from pkcm.envs.belief import candidates

    mirror = open_battle(a_mirror(dex, our_team), lead="meowscarada")
    slot = mirror.state.sides[1].active[0]
    mirror.report_item("choicescarf")
    known = Observation.of(mirror.state, 0).foe[slot]
    assert known.item_known and known.item == "choicescarf"
    assert all(one.item == "choicescarf"
               for one in candidates("meowscarada", known))


def test_a_berry_they_ate_is_gone_and_still_identifies_the_set(dex, our_team):
    """hk: a berry that has been eaten is gone, and Harvest giving berries back
    is the whole reason that distinction is mechanical rather than cosmetic.

    Recording it as still held -- which this did -- let the search feed them the
    same Sitrus Berry every time it dropped them below half. Recording it as
    "holds nothing" instead would throw away the identification, because almost
    no ranker set holds nothing. The observation carries both: what it holds now
    and what it used up.
    """
    from pkcm.envs.belief import candidates

    # Archaludon, because four of the six in the ranker pool hold a Sitrus
    # Berry -- so there is something for the filter to keep.
    mirror = open_battle(a_mirror(dex, our_team), lead="archaludon")
    slot = mirror.state.sides[1].active[0]
    mirror.report_item("sitrusberry", consumed=True)

    assert mirror.state.item_id(1, slot) is None, "they ate it; it is gone"
    known = Observation.of(mirror.state, 0).foe[slot]
    assert known.item is None and known.consumed_item == "sitrusberry"
    assert known.item_known
    found = candidates("archaludon", known)
    assert found and all(one.item == "sitrusberry" for one in found), (
        "the berry still says which set this is")


def test_an_ability_that_species_cannot_have_is_refused(dex, our_team):
    mirror = open_battle(a_mirror(dex, our_team), lead="clefable")
    with pytest.raises(MirrorError, match="가질 수 없습니다"):
        mirror.report_ability("intimidate")
    with pytest.raises(MirrorError, match="그런 특성이 없습니다"):
        mirror.report_ability("notanability")
    with pytest.raises(MirrorError, match="챔피언스에 없는 도구"):
        mirror.report_item("notanitem")


def test_a_switch_in_ability_is_applied_before_the_turn_is_stepped(dex, our_team):
    """The abilities that announce themselves mostly do it by *doing*
    something. Reporting one after the switch had already been played would
    leave its effect off the board, and the search would plan on a position
    that never happened."""
    six = ("hippowdon", "gengar", "kangaskhan", "starmie", "clefable",
           "archaludon")
    mirror = Mirror.begin(dex, dex.regulation("m_b"), our_team, six, seed=3)
    mirror.choose_ours((0, 1, 2))
    mirror.their_lead("gengar")
    mirror.open()
    assert mirror.state.field.weather is None

    # Sand Stream announces itself by changing the weather, which is about as
    # visible as an ability gets.
    landing = mirror.report_switch("hippowdon")
    mirror.report_ability("sandstream", landing.index)
    mirror.advance(Action.move(2), landing)

    assert mirror.state.field.weather == "sandstorm", (
        "the ability did not reach the field")
    assert landing.index in mirror.state.revealed[1].abilities


def test_our_own_item_can_be_reported_spent(dex, our_team):
    """Our set is not a guess, so there is nothing to reveal -- the only thing
    the screen can say that the mirror does not know is *when* it went.

    That matters because our damage rolls are not the game's: the engine eats
    a Sitrus Berry a turn early or late, and after that it is planning for a
    Pokemon holding something it no longer has. A Focus Sash the search still
    believes in is a turn spent on a line that does not exist.
    """
    mirror = open_battle(a_mirror(dex, our_team), ours=(3, 1, 2))
    slot = mirror.state.sides[0].active[0]
    assert mirror.our_item() == "focussash", "Aegislash is leading with it"

    mirror.report_our_item(consumed=True)
    assert mirror.our_item() is None
    assert mirror.state.item_id(0, slot) is None
    # And it is spent rather than never-held, the same shape a berry leaves.
    known = Observation.of(mirror.state, 0).own[slot]
    assert known.consumed_item == "focussash"


def test_our_item_can_change_hands_rather_than_vanish(dex, our_team):
    """Trick and Switcheroo do not use an item up, they swap it."""
    mirror = open_battle(a_mirror(dex, our_team), ours=(3, 1, 2))
    mirror.report_our_item(consumed=False, item_id="choicescarf")
    assert mirror.our_item() == "choicescarf"

    with pytest.raises(MirrorError, match="챔피언스에 없는 도구"):
        mirror.report_our_item(consumed=False, item_id="notanitem")


def test_reporting_our_item_lands_on_whoever_is_out(dex, our_team):
    """Same rule as the HP correction: it follows the field, not the sheet."""
    mirror = open_battle(a_mirror(dex, our_team), ours=(3, 1, 2))
    mirror.advance(Action.switch(1), mirror.report_move("shadowball"))
    after = mirror.state.sides[0].active[0]
    assert mirror.our_item() == "leftovers", "Corviknight came in"

    mirror.report_our_item(consumed=True)
    assert mirror.state.item_id(0, after) is None
    assert mirror.state.item_id(0, 0) == "focussash", (
        "the one that left still has its own")
