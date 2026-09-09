"""Integrity checks on the data layer.

These guard the seam between upstream data and our curated legality layer.
Showdown mutates its data files in place as the metagame moves, so a silent
rename upstream must fail loudly here rather than at battle time.
"""

from __future__ import annotations

import pytest

from pkcm.data.dex import TYPES, Stat, load_dex, to_id


@pytest.fixture(scope="module")
def dex():
    return load_dex()


@pytest.fixture(scope="module")
def regulation(dex):
    return dex.regulation("m_b")


def test_to_id_matches_showdown_conventions():
    assert to_id("Raichu-Alola") == "raichualola"
    assert to_id("Charizard-Mega X") == "charizardmegax"
    assert to_id("Farfetch'd") == "farfetchd"


def test_type_chart_spot_checks(dex):
    chart = dex.type_chart
    assert chart.multiplier("fire", ("grass",)) == 2.0
    assert chart.multiplier("water", ("fire",)) == 2.0
    assert chart.multiplier("electric", ("ground",)) == 0.0
    assert chart.multiplier("fighting", ("ghost",)) == 0.0
    assert chart.multiplier("ice", ("dragon", "flying")) == 4.0
    assert chart.multiplier("grass", ("fire", "flying")) == 0.25


def test_type_chart_is_total(dex):
    for attacking in TYPES:
        for defending in TYPES:
            assert dex.type_chart.multiplier(attacking, (defending,)) in (0.0, 0.5, 1.0, 2.0)


def test_species_parsing(dex):
    venusaur = dex.species["venusaur"]
    assert venusaur.types == ("grass", "poison")
    assert venusaur.base_stats == (80, 82, 83, 100, 100, 80)
    assert venusaur.base_stats[Stat.SPE] == 80
    assert venusaur.abilities == ("overgrow", "chlorophyll")
    assert not venusaur.is_mega

    mega = dex.species["venusaurmega"]
    assert mega.is_mega
    assert mega.base_species == "venusaur"
    assert mega.required_item == "venusaurite"


def test_champions_original_mega_is_present(dex):
    """Champions introduced Megas that predate no other game; Showdown carries them."""
    mega = dex.species["meganiummega"]
    assert mega.base_species == "meganium"
    assert mega.abilities == ("megasol",)
    assert "megasol" in dex.abilities


def test_move_parsing(dex):
    bolt = dex.moves["thunderbolt"]
    assert (bolt.type, bolt.category, bolt.base_power, bolt.accuracy) == (
        "electric", "Special", 90, 100,
    )
    assert "protect" in bolt.flags

    assert dex.moves["swift"].accuracy is None, "never-miss moves carry accuracy=None"
    assert dex.moves["protect"].priority == 4
    assert dex.moves["earthquake"].target == "allAdjacent"
    assert dex.moves["swordsdance"].is_status


def test_every_legal_entry_resolves(regulation, dex):
    unknown = sorted(
        i for i in regulation.legal_species | regulation.legal_megas if i not in dex.species
    )
    assert unknown == []
    assert len(regulation.legal_species) == 235
    assert len(regulation.legal_megas) == 76


def test_legal_species_reference_known_abilities(regulation, dex):
    missing = {
        ability
        for species_id in regulation.legal_species | regulation.legal_megas
        for ability in dex.species[species_id].abilities
        if ability not in dex.abilities
    }
    assert missing == set()


def test_legal_species_types_are_known(regulation, dex):
    for species_id in regulation.legal_species | regulation.legal_megas:
        species = dex.species[species_id]
        assert 1 <= len(species.types) <= 2
        assert set(species.types) <= set(TYPES)


def test_every_legal_mega_has_a_legal_stone(regulation):
    reachable = {m for megas in regulation.legal_mega_stones.values() for m in megas}
    assert regulation.legal_megas <= reachable


def test_mega_stones_point_at_real_species(dex):
    for item_id, mega_ids in dex.mega_stones.items():
        for mega_id in mega_ids:
            assert mega_id in dex.species, f"{item_id} points at unknown species {mega_id}"


def test_mega_evolution_lookup(dex):
    assert dex.mega_evolution("venusaur", "venusaurite") == "venusaurmega"
    assert dex.mega_evolution("charizard", "charizarditex") == "charizardmegax"
    assert dex.mega_evolution("venusaur", "leftovers") is None
    assert dex.mega_evolution("blastoise", "venusaurite") is None
    assert dex.mega_evolution("venusaur", None) is None


def test_regulation_rules(regulation):
    assert regulation.name == "M-B"
    assert regulation.team_size == 6
    assert regulation.level == 50
    assert regulation.bring_select("singles") == (6, 3)
    assert regulation.bring_select("doubles") == (6, 4)
    assert regulation.rules["confirmed"]["mega_evolutions_per_battle"] == 1
    assert regulation.rules["confirmed"]["terastallization"] is False


def test_learnsets_load_lazily(dex):
    assert "venusaur" in dex.learnsets


def test_every_battle_relevant_species_field_comes_from_the_rom():
    """Showdown must not be the source of a fact about Champions.

    Types, base stats, abilities and weight are all read out of the game's own
    personal table by scripts/build_species_from_rom.py. What the base data
    still supplies is everything a battle never reads -- forme names, egg
    groups, evolutions -- plus the spelling of a type index and an ability
    number, which is a dictionary rather than a claim.

    Anything the ROM has an entry for must match it exactly. This is the guard
    that stops the next silent fallback.
    """
    import json
    from pathlib import Path

    from pkcm.data.dex import load_dex, to_id

    dex = load_dex()
    table = json.loads(
        Path("data/champions/species.json").read_text(encoding="utf-8"))["species"]
    assert len(table) > 300, "the ROM table did not load"

    for key, entry in table.items():
        species = dex.species[key]
        assert list(species.types) == [t.lower() for t in entry["types"]], key
        assert list(species.base_stats) == [
            entry["baseStats"][k] for k in ("hp", "atk", "def", "spa", "spd", "spe")], key
        assert list(species.abilities) == [
            to_id(entry["abilities"][slot]) for slot in ("0", "1", "H", "S")
            if slot in entry["abilities"]], key
        assert species.weight_kg == entry["weightkg"], key


def test_the_rom_species_table_is_not_stale():
    """It is generated; regenerate rather than hand-edit."""
    import subprocess
    import sys

    done = subprocess.run(
        [sys.executable, "scripts/build_species_from_rom.py", "--check"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert done.returncode == 0, done.stdout + done.stderr


def test_the_rom_disagrees_with_showdown_where_we_know_it_should():
    """The six the base data gets wrong, named so a change is visible.

    Five are the Megas Champions invented, where Showdown has no ability to
    carry and falls back to the base species'. The sixth is Greninja, which
    Showdown gives Battle Bond and the game does not.

    If this list ever shrinks silently, something stopped applying.
    """
    import json
    from pathlib import Path

    from pkcm.data.dex import load_dex

    dex = load_dex()
    expected = {
        "golisopodmega": ("toughclaws",),
        "lucariomegaz": ("auraguard",),
        "absolmegaz": ("sharpness",),
        "garchompmegaz": ("levitate",),
        "baxcaliburmega": ("thermalexchange",),
        "greninja": ("torrent", "protean"),
    }
    base = json.loads(Path("data/raw/pokedex.json").read_text(encoding="utf-8"))
    for key, abilities in expected.items():
        assert dex.species[key].abilities == abilities, key
        theirs = tuple(base[key].get("abilities", {}).values())
        assert tuple(a.lower().replace(" ", "") for a in theirs) != abilities, (
            f"{key}: the base data now agrees, so this test is measuring "
            "nothing. Check why before deleting the row.")


def test_an_ability_champions_added_can_be_looked_up():
    """Aura Guard is in no upstream file, so it is added rather than edited.

    Without the entry it was registrable on Mega Lucario Z but absent from
    dex.abilities, which is the kind of half-presence that reads as working
    until something asks for its name. The ROM gives it number 319, past the
    end of the mainline numbering, which is how a Champions-original ability
    announces itself.
    """
    from pkcm.data.dex import load_dex

    dex = load_dex()
    assert "auraguard" in dex.abilities
    assert dex.abilities["auraguard"].name == "Aura Guard"
    assert dex.exists_in_champions(dex.abilities["auraguard"])


def test_the_hidden_ability_slot_survives_the_rom_table():
    """The regression the slot names caused, kept as a test.

    The personal table has three ability columns and the base data names its
    slots "0", "1" and "H". Writing "0", "1", "2" instead drops every hidden
    ability in the game, which reads as a data-format detail and lands as 86
    of 243 field parties turning illegal.
    """
    from pkcm.data.dex import load_dex

    dex = load_dex()
    assert dex.species["sableye"].abilities == ("keeneye", "stall", "prankster")
    assert dex.species["clefable"].abilities == ("cutecharm", "magicguard",
                                                 "unaware")
    assert dex.species["toxapex"].abilities == ("merciless", "limber",
                                                "regenerator")


def test_every_imported_party_is_still_legal():
    """The end-to-end consequence, which is what actually caught the bug.

    A species table that drops abilities does not fail loudly; it makes real
    teams illegal. These parties are people's ladder teams and every one of
    them was legal when it was imported.
    """
    from pkcm.data.dex import load_dex
    from pkcm.engine.legality import ranker_parties, team_errors

    dex = load_dex()
    regulation = dex.regulation("m_c")
    for path in ("data/champions/parties_field.json",
                 "data/champions/parties_hand.json"):
        parties = ranker_parties(path)
        assert parties, path
        for party in parties:
            assert not team_errors(dex, regulation, party.team, "singles"),                 f"{path}: {party.title[:40]}"
