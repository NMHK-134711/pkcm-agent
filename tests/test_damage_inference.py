"""The damage ledger and the pricer that inverts it.

The one invariant everything rests on: **the true set always survives its own
hits.** The recorder only keeps hits the analytic formula can price, and the
pricer prices candidates with that same formula -- so if the attacker's real
set is ever eliminated by a number it really rolled, either the recorder let a
dirty hit through or the pricer's arithmetic disagrees with the engine's.
Both are bugs here, not in the pool.

The test plays real battles on real pool sets and checks the invariant on
every recorded hit, which is how the Body Press family was found to need
guarding in the first place: moves that read a stat their category does not
say price wrong everywhere, and only an end-to-end sweep notices.
"""

from __future__ import annotations

import pytest

from pkcm.data.dex import load_dex
from pkcm.engine.battle import step
from pkcm.engine.legality import make_team
from pkcm.engine.rng import Rng
from pkcm.engine.state import BattleConfig, Phase, legal_actions, new_battle
from pkcm.envs.belief import survives_hits
from pkcm.envs.observation import Observation


@pytest.fixture(scope="module")
def dex():
    return load_dex()


@pytest.fixture(scope="module")
def config(dex):
    return BattleConfig(dex=dex, regulation=dex.regulation("m_b"),
                        battle_format="singles")


def _random_playout(dex, config, seed, turns=50):
    teams = tuple(make_team(dex, config.regulation,
                            Rng.from_seed(seed * 2 + o).cursor(),
                            "singles", "parties") for o in (1, 2))
    state = new_battle(config, teams, seed=seed)
    cursor = Rng.from_seed(seed ^ 0xF00D).cursor()
    played = 0
    while not state.finished and played < turns:
        chosen = []
        for player in (0, 1):
            if state.phase is Phase.TEAM_PREVIEW:
                legal = legal_actions(state, player, 0)
                chosen.append((legal[cursor.between(0, len(legal) - 1)],))
                continue
            options = []
            for position in range(len(state.sides[player].active)):
                legal = legal_actions(state, player, position)
                options.append(legal[cursor.between(0, len(legal) - 1)]
                               if legal else None)
            chosen.append(tuple(o for o in options if o is not None))
        state, _ = step(state, chosen[0], chosen[1])
        played += 1
    return state, teams


def test_the_true_set_survives_every_hit_it_landed(dex, config):
    battles = hits = 0
    for seed in range(1, 25):
        state, teams = _random_playout(dex, config, seed)
        battles += 1
        for observer in (0, 1):
            observation = Observation.of(state, observer)
            foe = 1 - observer
            for known in observation.foe:
                if not known.hits_on_us:
                    continue
                hits += len(known.hits_on_us)
                party_index = state.sides[foe].selection[known.slot]
                # The set as the opponent registered it -- compiled teams keep
                # the original PokemonSet on the BattlePokemon.
                truth = state.parties[foe][party_index].set
                assert survives_hits(truth, known), (
                    f"seed {seed}: the real {truth.species} was eliminated by "
                    f"its own hits {known.hits_on_us}")
    # A sweep that recorded nothing would pass vacuously and prove nothing.
    assert hits >= 30, f"only {hits} clean hits over {battles} battles"


def test_a_wrong_attack_investment_is_eliminated(dex, config):
    """The positive case: the filter must actually cut, not just not-harm.

    Two candidates a full Attack investment apart throw visibly different
    numbers off a 100-power move; a hit from the strong one must strike the
    weak build off, and (the invariant again) never the reverse.
    """
    from pkcm.engine.legality import PokemonSet
    from pkcm.envs.belief import _hit_rolls

    strong = PokemonSet(species="garchomp", ability="roughskin",
                        moves=("earthquake",), nature="adamant",
                        sp=(0, 32, 0, 0, 0, 0), item=None)
    weak = PokemonSet(species="garchomp", ability="roughskin",
                      moves=("earthquake",), nature="modest",
                      sp=(0, 0, 0, 0, 0, 32), item=None)
    defender_stats = (200, 100, 120, 100, 120, 100)
    defender_types = ("steel",)

    strong_rolls = _hit_rolls(strong, "garchomp", "earthquake",
                              defender_stats, defender_types)
    weak_rolls = _hit_rolls(weak, "garchomp", "earthquake",
                            defender_stats, defender_types)
    assert strong_rolls and weak_rolls
    assert max(weak_rolls) < min(strong_rolls), (weak_rolls, strong_rolls)


def test_hits_are_ours_alone(dex, config):
    """A player is handed only the hits their own Pokemon took -- the exact
    integers. The opponent's HP is a fraction to us and our reads of it never
    enter their ledger view or ours."""
    for seed in range(1, 12):
        state, _ = _random_playout(dex, config, seed)
        if not state.observed_hits:
            continue
        for observer in (0, 1):
            observation = Observation.of(state, observer)
            expected = sum(1 for entry in state.observed_hits
                           if entry[0] == observer)
            handed = sum(len(known.hits_on_us) for known in observation.foe)
            assert handed == expected
        return
    pytest.skip("no clean hits landed in these seeds")


def test_the_skin_family_is_priced_not_silenced(dex, config):
    """hk's objection, kept as a test: party 43's own Primarina runs Liquid
    Voice, and Mega Gardevoir's whole doubles game is a skinned Hyper Voice.
    Silencing modify_move abilities threw those observations away. Priced
    instead, the number identifies the *ability*: the same Hyper Voice into
    the same wall lands in disjoint ranges."""
    from pkcm.engine.legality import PokemonSet
    from pkcm.envs.belief import _hit_rolls

    defender_stats = (200, 100, 120, 100, 120, 100)

    liquid = PokemonSet(species="primarina", ability="liquidvoice",
                        moves=("hypervoice",), nature="modest",
                        sp=(0, 0, 0, 32, 0, 0), item=None)
    rolls = _hit_rolls(liquid, "primarina", "hypervoice",
                       defender_stats, ("fire", "ground"))
    assert rolls, "Liquid Voice must be priceable"
    # Water into fire/ground is 2x with STAB; a Normal-typed reading of the
    # same move is neutral and unboosted, under a third of this.
    assert min(rolls) > 300

    pixilate = PokemonSet(species="sylveon", ability="pixilate",
                          moves=("hypervoice",), nature="modest",
                          sp=(0, 0, 0, 32, 0, 0), item=None)
    plain = PokemonSet(species="sylveon", ability="cutecharm",
                       moves=("hypervoice",), nature="modest",
                       sp=(0, 0, 0, 32, 0, 0), item=None)
    into_dragon = ("dragon", "ground")
    fairy_rolls = _hit_rolls(pixilate, "sylveon", "hypervoice",
                             defender_stats, into_dragon)
    normal_rolls = _hit_rolls(plain, "sylveon", "hypervoice",
                              defender_stats, into_dragon)
    assert fairy_rolls and normal_rolls
    assert min(fairy_rolls) > max(normal_rolls) * 2, (
        "one observed integer should separate Pixilate from a plain ability")
