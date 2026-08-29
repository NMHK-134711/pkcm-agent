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
