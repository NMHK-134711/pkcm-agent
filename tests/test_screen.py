"""Reading the game's screen.

Most of this cannot be tested without hk's screen in front of it, so what is
tested here is the arithmetic that does not need one: a bar's filled fraction,
a region's placement inside a window, and a profile surviving a round trip.
The parts that do need a screen are skipped rather than mocked -- a mocked
screen capture tests the mock.
"""

from __future__ import annotations

import pytest

from pkcm.live.screen import Profile, Region, ScreenError, bar_fraction

PIL = pytest.importorskip("PIL", reason="capture needs Pillow")


def a_bar(filled: float, width: int = 200, height: int = 12,
          colour=(110, 231, 168), empty=(42, 48, 64)):
    """A health bar drawn the way one is drawn."""
    from PIL import Image

    image = Image.new("RGB", (width, height), empty)
    cut = round(width * filled)
    for x in range(cut):
        for y in range(height):
            image.putpixel((x, y), colour)
    return image


@pytest.mark.parametrize("filled", [0.0, 0.13, 0.5, 0.72, 1.0])
def test_a_bar_is_measured_not_read(filled):
    """OCR on a small number is a coin flip and HP is the number the leaf
    evaluation cares about most. The bar is a rectangle, and that is arithmetic."""
    got = bar_fraction(a_bar(filled))
    assert got == pytest.approx(filled, abs=0.01)


def test_the_bar_colour_may_change_without_the_reading_changing():
    """It goes green to amber to red on the way down, and the question being
    asked is "is this pixel part of the bar", never "which bar colour is this"."""
    for colour in ((110, 231, 168), (246, 196, 83), (240, 138, 138)):
        assert bar_fraction(a_bar(0.4, colour=colour)) == pytest.approx(0.4, abs=0.01)


def test_an_empty_bar_reads_as_empty():
    """The reading that has to be right, and was not.

    The first version took the leftmost pixel as the filled colour, which is
    true of every bar except the one that matters: at 0% the leftmost pixel is
    track, the whole row matched it, and a fainted Pokemon read as 100%.
    """
    from PIL import Image

    assert bar_fraction(Image.new("RGB", (200, 12), (42, 48, 64))) == 0.0


@pytest.mark.parametrize("track", [(42, 48, 64), (60, 60, 60), (30, 30, 30),
                                   (90, 90, 90), (120, 120, 120)])
@pytest.mark.parametrize("colour", [(110, 231, 168), (246, 196, 83),
                                    (240, 138, 138), (220, 60, 60),
                                    (255, 203, 0)])
def test_it_reads_the_same_on_any_plausible_pair_of_colours(track, colour):
    """Amber and red came back 0% while green did not, which is worse than
    being wrong everywhere: it only showed up once the Pokemon was in trouble.
    ``(high - low) * 255`` overflows int16 for exactly the saturated colours
    this is meant to detect.
    """
    for filled in (0.0, 0.13, 0.5, 0.72, 1.0):
        assert bar_fraction(a_bar(filled, colour=colour, empty=track)) \
            == pytest.approx(filled, abs=0.02)


def test_the_two_colours_can_be_pinned_when_a_layout_needs_it():
    """The default judges each pixel on saturation, which is right for a
    coloured bar on a grey track. A layout that breaks it can name the colours."""
    grey_on_grey = a_bar(0.4, colour=(150, 150, 150), empty=(40, 40, 40))
    assert bar_fraction(grey_on_grey, filled=(150, 150, 150)) \
        == pytest.approx(0.4, abs=0.02)
    assert bar_fraction(grey_on_grey, empty=(40, 40, 40)) \
        == pytest.approx(0.4, abs=0.02)


def test_a_region_is_a_fraction_of_the_window_not_a_pixel_box():
    """Drawn once at one window size and still pointing at the same thing after
    the emulator is resized, which is the first thing that happens to it."""
    region = Region(left=0.25, top=0.5, right=0.75, bottom=0.6)
    assert region.pixels((0, 0, 400, 200)) == (100, 100, 300, 120)
    # Same window, twice the size, and the same feature.
    assert region.pixels((0, 0, 800, 400)) == (200, 200, 600, 240)
    # Moved, not resized: the box moves with it.
    assert region.pixels((100, 50, 500, 250)) == (200, 150, 400, 170)


def test_a_profile_round_trips(tmp_path):
    profile = Profile(window="BlueStacks",
                      regions={"their_hp": Region(0.1, 0.2, 0.3, 0.22),
                               "their_name": Region(0.1, 0.1, 0.4, 0.16)})
    path = profile.save(tmp_path / "screen_profile.json")
    back = Profile.load(path)
    assert back.window == "BlueStacks"
    assert back.regions["their_hp"] == profile.regions["their_hp"]
    assert back.regions["their_name"] == profile.regions["their_name"]


def test_a_missing_profile_says_what_to_run(tmp_path):
    with pytest.raises(ScreenError, match="calibrate"):
        Profile.load(tmp_path / "not_here.json")


def test_a_window_that_is_not_open_lists_what_is():
    """The failure a person actually hits is "the emulator is not running", and
    the useful reply is the list of windows that are."""
    from pkcm.live.screen import anchor

    try:
        anchor("no such window exists 12345")
    except ScreenError as error:
        assert "no visible window" in str(error)
    except Exception as error:  # pragma: no cover - no desktop in CI
        pytest.skip(f"cannot enumerate windows here: {error}")
    else:
        pytest.fail("a window that is not open should not resolve")


# --------------------------------------------------------------------------- #
# Making sense of what came back
# --------------------------------------------------------------------------- #

from pkcm.live.screen import (  # noqa: E402
    our_hp_percent,
    parse_hp_fraction,
    parse_percent,
    scan_log,
)

VOCABULARY = {
    "moves": {"칼춤": "swordsdance", "지진": "earthquake",
              "드라이브": "shorter", "플레어드라이브": "flaredrive",
              "아쿠아브레이크": "aquabreak"},
    "species": {"킬가르도": "aegislash", "한카리아스": "garchomp"},
}


@pytest.mark.parametrize("text, want", [
    ("72%", 72), (" 100 % ", 100), ("HP 72", 72), ("", None),
    ("abc", None), ("150%", None), ("-", None),
])
def test_a_percentage_is_pulled_out_of_whatever_else_ocr_saw(text, want):
    assert parse_percent(text) == want


@pytest.mark.parametrize("text, want", [
    ("152/187", (152, 187)),
    ("152 | 187", (152, 187)),      # OCR reads the slash as a bar
    ("152 / 187", (152, 187)),
    ("200/187", None),             # more than full is a misread
    ("187", None), ("", None), (None, None),
])
def test_our_hp_is_a_pair_however_the_slash_came_out(text, want):
    assert parse_hp_fraction(text) == want


def test_our_own_hp_checks_itself_against_a_maximum_we_know():
    """The one reading that can be verified rather than trusted. Our maximum HP
    is not a guess -- the engine computed it from the set -- so a denominator
    matching none of the Pokemon we brought means OCR misread, and a wrong HP
    is worth more to catch than a missing one."""
    assert our_hp_percent("152/187", [187, 205]) == (81, "152/187")
    assert our_hp_percent("152/188", [187, 205]) is None, (
        "one digit out is the common OCR slip, and near is not evidence")
    assert our_hp_percent("nonsense", [187]) is None


def test_the_log_is_scanned_for_names_not_parsed_as_a_sentence():
    """Champions' phrasing is not something this may assume, so it looks for
    the names themselves and lets the caller decide what they mean."""
    found = scan_log("상대의 킬가르도는 칼춤을 썼다!", VOCABULARY)
    assert [name for name, _ in found["moves"]] == ["swordsdance"]
    assert [name for name, _ in found["species"]] == ["aegislash"]


def test_a_name_survives_ocr_getting_a_syllable_wrong():
    """Windows OCR read 랭크배틀 as 랭크dH틀 on this machine, so exact matching
    would drop real moves. The score comes back with the hit, because the form
    it fills is checked by a person and "probably" is worth showing."""
    found = scan_log("상대의 킬가르dH는 칼춤을 썼다!", VOCABULARY)
    names = dict(found["species"])
    assert "aegislash" in names and names["aegislash"] < 1.0


def test_a_shorter_name_inside_a_longer_one_is_not_a_second_hit():
    found = scan_log("한카리아스의 플레어드라이브!", VOCABULARY)
    assert [name for name, _ in found["moves"]] == ["flaredrive"], (
        "플레어드라이브 in the line is not also a 드라이브")


def test_a_line_that_names_nothing_finds_nothing():
    """The threshold has to be tight enough that noise does not become a move.
    A wrong move goes into the mirror as a fact."""
    for line in ("아무 말도 없는 줄", "Play & win", "63.594VP", ""):
        found = scan_log(line, VOCABULARY)
        assert not found["moves"] and not found["species"], line
