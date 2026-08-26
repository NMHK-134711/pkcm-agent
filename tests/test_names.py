"""Display names and Korean particles.

Ids never change; only what a person reads does. These check that the split
holds and that the Korean actually reads like Korean.
"""

from __future__ import annotations

import pytest

from pkcm.data.dex import load_dex
from pkcm.engine.actions import Action
from pkcm.engine.battle import step
from pkcm.engine.events import Event
from pkcm.engine.items import champions_items
from pkcm.engine.pokemon import PokemonSet
from pkcm.engine.state import BattleConfig, new_battle
from pkcm.render.names import Names
from pkcm.render.text import Renderer, has_final_consonant, josa


@pytest.fixture(scope="module")
def dex():
    return load_dex()


@pytest.fixture(scope="module")
def names(dex):
    return Names("ko", dex)


# --------------------------------------------------------------------------- #
# The tables
# --------------------------------------------------------------------------- #


def test_species_names_including_formes(names):
    assert names.species("gengar") == "팬텀"
    assert names.species("gengarmega") == "메가팬텀"
    assert names.species("raichualola") == "알로라 라이츄"
    assert names.species("arcaninehisui") == "히스이 윈디"
    assert names.species("slowbrogalar") == "가라르 야도란"


def test_move_and_item_and_ability_names(names):
    assert names.move("thunderbolt") == "10만볼트"
    assert names.move("earthquake") == "지진"
    assert names.item("leftovers") == "먹다남은음식"
    assert names.item("choicescarf") == "구애스카프"
    assert names.item("gengarite") == "팬텀나이트"
    assert names.ability("levitate") == "부유"
    assert names.ability("intimidate") == "위협"


def test_champions_original_stones_come_from_the_scrape(names):
    """PokeAPI has never heard of these; hk's op.gg list has their real names."""
    assert names.item("meganiumite") == "메가니움나이트"
    assert names.item("greninjite") == "개굴닌자나이트"
    assert names.item("alakazite") == "후디나이트", "the game drops the final consonant"


def test_everything_champions_uses_has_a_korean_name(dex, names):
    regulation = dex.regulation("m_b")
    for species_id in regulation.legal_species | regulation.legal_megas:
        assert names.species(species_id) != species_id, species_id
    for move in dex.moves.values():
        if dex.exists_in_champions(move):
            assert names.move(move.id) != move.id, move.id
    for item_id in champions_items():
        assert names.item(item_id) != item_id, item_id


def test_unknown_ids_fall_back_rather_than_vanish(dex):
    """Two Champions-original abilities have no published Korean name."""
    korean = Names("ko", dex)
    assert korean.ability("eelevate") == "Eelevate", "English, not a raw id"
    assert korean.ability("nonsense_id") == "nonsense_id"


def test_english_renderer_is_still_available(dex):
    english = Names("en", dex)
    assert english.species("gengarmega") == "Gengar-Mega"
    assert english.move("thunderbolt") == "Thunderbolt"


# --------------------------------------------------------------------------- #
# Particles
# --------------------------------------------------------------------------- #


def test_final_consonant_detection():
    assert has_final_consonant("핫삼") is True
    assert has_final_consonant("팬텀") is True
    assert has_final_consonant("피카츄") is False
    assert has_final_consonant("Gengar") is None, "not Hangul"


@pytest.mark.parametrize("word,particle,expected", [
    ("핫삼", "을", "핫삼을"),
    ("피카츄", "을", "피카츄를"),
    ("특수공격", "이", "특수공격이"),
    ("스피드", "이", "스피드가"),
    ("팬텀", "은", "팬텀은"),
    ("루차불", "으로", "루차불로"),
    ("메가팬텀", "으로", "메가팬텀으로"),
])
def test_particles_agree_with_the_word(word, particle, expected):
    assert josa(word, particle) == expected


# --------------------------------------------------------------------------- #
# The engine stays language-free
# --------------------------------------------------------------------------- #


def test_events_carry_ids_not_names(dex):
    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"))
    team = tuple(
        PokemonSet(species=s, ability="__none__", moves=("bodyslam",))
        for s in ("gengar", "snorlax", "pikachu", "starmie", "alakazam", "skarmory")
    )
    state = new_battle(config, (team, team), seed=1)
    state, log = step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))

    entries = [e for e in log if e.kind == "switch_in"]
    assert entries and all(e.species == "gengar" for e in entries), "the id, not 'Gengar'"

    state, log = step(state, Action.move(0), Action.move(0))
    used = [e for e in log if e.kind == "move_used"]
    assert used and all(e.move == "bodyslam" for e in used)


def test_the_same_log_renders_in_either_language(dex):
    log = [
        Event("move_used", side=0, slot=0, species="gengar", move="thunderbolt"),
        Event("faint", side=1, slot=0, species="snorlax"),
    ]
    korean = Renderer("ko", dex, ("레드", "블루")).render_log(log)
    english = Renderer("en", dex, ("Red", "Blue")).render_log(log)

    assert "팬텀의 10만볼트!" in korean
    assert "잠만보는 쓰러졌다!" in korean
    assert "Gengar used Thunderbolt!" in english
    assert "Snorlax fainted!" in english


def test_screens_report_turns_and_hazards_report_layers(dex):
    renderer = Renderer("ko", dex, ("레드", "블루"))
    screen = renderer.render(Event("side_condition", side=0, detail="lightscreen", amount=5))
    hazard = renderer.render(Event("side_condition", side=0, detail="spikes", amount=2))
    assert "5턴" in screen[0]
    assert "2겹" in hazard[0]
