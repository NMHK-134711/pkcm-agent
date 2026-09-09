"""Scoring a party by its worst matchups instead of its average.

The field does not sort, so a mean win rate mostly reports which opponents
were in it. These check the two decisions that make the floor usable: the
tail is a quarter rather than a single minimum, and the games go where the
answer is.
"""

from __future__ import annotations

import random

from pkcm.train.party_floor import Matchup, Selection, summarise
from pkcm.train.party_floor import _needs_games as needs_games


def field(rates, games=8, seed=1):
    """One record per opponent, drawn at the rate given."""
    rng = random.Random(seed)
    out = []
    for index, rate in enumerate(rates):
        wins = sum(1 for _ in range(games) if rng.random() < rate)
        out.append(Matchup(index, wins, games - wins))
    return out


def test_the_floor_is_steadier_than_the_minimum():
    """Why the minimum is not used.

    Forty-four even matchups and one genuinely bad one. The sample minimum
    lands on whichever even matchup went worst -- it reports the luck, not
    the party. The quarter's mean is dragged down too, but far less, and
    unlike the minimum it recovers as games accumulate (below).
    """
    rows = field([0.2] + [0.5] * 44, games=8, seed=7)
    floor = summarise(rows, tail=0.25)

    assert floor.minimum <= 0.25, "some even matchup came out looking terrible"
    assert floor.cvar > floor.minimum + 0.15
    assert 0.45 < floor.mean < 0.55, "the average is not fooled either way"


def test_the_tail_is_biased_low_until_it_has_the_games():
    """A property to know rather than to hide.

    The tail is chosen by the same games that score it, so at eight games a
    matchup it selects whichever came out unluckiest and reports their rate --
    the order statistic's own winner's curse, pointing down. It lifts towards
    the truth as the games behind it grow, which is what the race spends its
    budget on, and it is why two candidates should not be compared at very
    different game counts.
    """
    rates = [0.5] * 45
    thin = summarise(field(rates, games=8, seed=4), tail=0.25)
    thick = summarise(field(rates, games=96, seed=4), tail=0.25)

    assert thin.cvar < thick.cvar - 0.1, (thin.cvar, thick.cvar)
    assert thick.cvar > 0.4, "and it is heading for the true 0.5"


def test_a_party_with_no_answer_scores_below_one_that_is_merely_average():
    """The property the whole measure exists for: a high mean does not save a
    party that loses badly to a slice of the field."""
    spiky = summarise(field([0.05] * 8 + [0.85] * 37, seed=3), tail=0.25)
    flat = summarise(field([0.52] * 45, seed=3), tail=0.25)

    assert spiky.mean > flat.mean, "the spiky one looks better on average"
    assert spiky.cvar < flat.cvar, "and worse where it counts"


def test_the_games_go_to_the_tail_and_the_boundary():
    """A matchup that is comfortably won does not change the answer, so it
    does not get played again."""
    rows = field([0.05] * 6 + [0.5] * 6 + [0.95] * 33, games=40, seed=5)
    wanted = set(needs_games(rows, tail=0.25))

    assert {row.opponent for row in rows[:6]} <= wanted, "the bad ones are in"
    assert not (wanted & {row.opponent for row in rows[12:]}), (
        "the comfortable ones are not")


def test_an_unplayed_field_asks_for_everyone():
    rows = [Matchup(index) for index in range(10)]
    assert set(needs_games(rows, tail=0.25)) == set(range(10))


def test_the_tail_narrows_as_its_games_accumulate():
    thin = summarise(field([0.3] * 45, games=8, seed=11), tail=0.25)
    thick = summarise(field([0.3] * 45, games=64, seed=11), tail=0.25)

    assert thick.high - thick.low < thin.high - thin.low
    assert thick.games > thin.games


def test_a_dead_slot_is_invisible_to_the_floor():
    """Why the selection is measured separately.

    Two parties with the same matchup record. One brought all six across the
    field; the other never brought its sixth at all and was playing three
    against two in every game. The floor is identical -- it is computed from
    win rates and knows nothing about who was on the mat -- so the difference
    has to come from somewhere else.
    """
    rows = field([0.3] * 4 + [0.55] * 16, games=8, seed=3)
    flexible = Selection(brought=(48, 44, 40, 38, 36, 34), battles=80,
                         combos=((( 0, 1, 2), 20), ((0, 1, 3), 18)))
    lopsided = Selection(brought=(80, 80, 80, 0, 0, 0), battles=80,
                         combos=(((0, 1, 2), 80),))

    assert summarise(rows, 0.25, flexible).cvar ==         summarise(rows, 0.25, lopsided).cvar
    assert flexible.live() == 6
    assert lopsided.live() == 3
    assert lopsided.rigidity == 1.0, "the same three every single game"
    assert flexible.rigidity < 0.3


def test_the_selection_record_is_empty_rather_than_wrong_when_unmeasured():
    """field_report.py calls summarise with no selection and must not break."""
    rows = field([0.5] * 8, games=8, seed=4)
    floor = summarise(rows, 0.25)

    assert floor.selection.battles == 0
    assert floor.selection.rates == ()
    assert floor.selection.live() == 0
    assert floor.selection.rigidity == 0.0


def test_every_battle_brings_exactly_three():
    """The counts come off a real battle, so the invariant is worth asserting.

    Three of six is the regulation, and a bug here would show up as a party
    that looks like it brought four -- which would quietly break every rate
    the Selection reports.
    """
    from pkcm.data.dex import load_dex
    from pkcm.engine.legality import ranker_parties
    from pkcm.search import SearchConfig
    from pkcm.train.party_floor import FloorConfig, play_pair

    dex = load_dex()
    parties = ranker_parties("data/champions/parties_field.json")
    config = FloorConfig(parties="data/champions/parties_field.json",
                         regulation="m_c",
                         search=SearchConfig(iterations=8, determinizations=2))
    _, _, _, brought = play_pair(dex, config, parties[0].team,
                                 parties[1].team, repeat=0)

    assert len(brought) == 2, "both seatings are played"
    for one in brought:
        assert len(one) == 3
        assert len(set(one)) == 3, "the same Pokemon cannot be brought twice"
        assert all(0 <= index < 6 for index in one)
