"""Reading the game's own screen, so a turn timer is survivable.

Typing what happened is the slowest part of the coach and the turn clock does
not wait. The screen already says all of it, so this reads it: capture the
window the game is in, measure the HP bars, and put the text through OCR.

Three rules the design turns on.

**It proposes, it never commits.** Every reading lands in the form for a person
to glance at and correct. A misread that goes straight into the mirror is worse
than typing, because the advice built on it looks exactly as confident as
advice built on the truth.

**HP is measured, not read.** OCR on a number is a coin flip at small sizes and
HP is the number the leaf evaluation cares about most. The bar is a coloured
rectangle and its filled fraction is arithmetic -- no model, no language, and
wrong only if the calibration is wrong.

**Nothing here knows what Champions looks like.** Every position comes from a
profile the person draws once, because a layout guessed from memory is a
layout that silently drifts when the window moves or the emulator rescales.
``anchor`` re-finds the window each capture so moving it costs nothing.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROFILE_PATH = ROOT / "data" / "screen_profile.json"
#: Written next to this module so it ships with it; it is a PowerShell bridge
#: to Windows' own OCR, which is present on Windows 10 and later and already
#: speaks Korean. No model to download and no service to run.
OCR_SCRIPT = Path(__file__).resolve().parent / "ocr.ps1"


class ScreenError(Exception):
    """The screen cannot be read the way the profile says it can."""


@dataclass(frozen=True, slots=True)
class Region:
    """A rectangle inside the game window, in fractions of it.

    Fractions rather than pixels so that a profile drawn at one window size
    still points at the same thing after the window is resized -- which is the
    first thing that happens to an emulator window.
    """

    left: float
    top: float
    right: float
    bottom: float

    def pixels(self, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x, y, x2, y2 = box
        width, height = x2 - x, y2 - y
        return (round(x + self.left * width), round(y + self.top * height),
                round(x + self.right * width), round(y + self.bottom * height))


@dataclass(frozen=True, slots=True)
class Profile:
    """Where everything is, and which window to look in."""

    #: Substring of the window title. The emulator's window, usually.
    window: str = ""
    regions: dict[str, Region] = field(default_factory=dict)

    @staticmethod
    def load(path: str | Path | None = None) -> "Profile":
        target = Path(path) if path else PROFILE_PATH
        if not target.exists():
            raise ScreenError(
                f"{target} is not there -- run scripts/calibrate.py once")
        raw = json.loads(target.read_text(encoding="utf-8"))
        return Profile(window=raw.get("window", ""),
                       regions={name: Region(**box)
                                for name, box in raw.get("regions", {}).items()})

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else PROFILE_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({
            "window": self.window,
            "regions": {name: {"left": box.left, "top": box.top,
                               "right": box.right, "bottom": box.bottom}
                        for name, box in self.regions.items()},
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        return target


# --------------------------------------------------------------------------- #
# Finding the window
# --------------------------------------------------------------------------- #

_ENUMERATE = r'''
Add-Type @"
using System; using System.Runtime.InteropServices; using System.Text;
public class WinEnum {
  [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc f, IntPtr l);
  // CharSet.Unicode is not optional: DllImport marshals a StringBuilder as ANSI
  // by default, so the UTF-16 that GetWindowTextW writes is read back a byte at
  // a time and every title truncates at its first NUL -- "BlueStacks" arrives
  // as "B".
  [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern int GetWindowTextW(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  public delegate bool EnumProc(IntPtr h, IntPtr l);
  public struct RECT { public int Left, Top, Right, Bottom; }
}
"@
$found = New-Object System.Collections.ArrayList
$callback = [WinEnum+EnumProc]{
  param($handle, $lparam)
  if ([WinEnum]::IsWindowVisible($handle)) {
    $text = New-Object System.Text.StringBuilder 512
    [void][WinEnum]::GetWindowTextW($handle, $text, 512)
    $title = $text.ToString()
    if ($title.Length -gt 0) {
      $rect = New-Object WinEnum+RECT
      [void][WinEnum]::GetWindowRect($handle, [ref]$rect)
      [void]$found.Add([pscustomobject]@{ title = $title; left = $rect.Left
        top = $rect.Top; right = $rect.Right; bottom = $rect.Bottom })
    }
  }
  return $true
}
[void][WinEnum]::EnumWindows($callback, [IntPtr]::Zero)
# Out-File rather than stdout: Windows PowerShell 5.1 encodes the pipe in the
# console codepage, which mangles every Korean window title on the way out and
# truncates the ASCII ones at their first NUL, because GetWindowTextW hands
# back UTF-16. Setting $OutputEncoding inside the script is too late.
$found | ConvertTo-Json -Compress | Out-File -Encoding utf8 "__OUT__"
'''


def windows() -> list[dict]:
    """Every visible titled window, with its rectangle."""
    with tempfile.TemporaryDirectory() as folder:
        out = Path(folder) / "windows.json"
        done = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-Command", _ENUMERATE.replace("__OUT__", str(out))],
            capture_output=True, timeout=30)
        if not out.exists():
            raise ScreenError(
                "could not enumerate windows: "
                + done.stderr.decode("utf-8", errors="replace")[:200])
        found = json.loads(out.read_text(encoding="utf-8-sig"))
    return found if isinstance(found, list) else [found]


def anchor(title: str) -> tuple[int, int, int, int]:
    """The rectangle of the window whose title contains ``title``.

    Looked up per capture rather than stored, so moving or resizing the window
    between turns costs nothing. An empty title means the whole screen.
    """
    if not title:
        from PIL import ImageGrab

        return (0, 0) + ImageGrab.grab().size
    matches = [one for one in windows() if title.lower() in one["title"].lower()]
    if not matches:
        raise ScreenError(
            f"no visible window with {title!r} in its title. Open: "
            f"{[one['title'][:30] for one in windows()][:6]}")
    # The biggest, because emulators keep small helper windows with the same name.
    best = max(matches, key=lambda one: (one["right"] - one["left"])
               * (one["bottom"] - one["top"]))
    return (best["left"], best["top"], best["right"], best["bottom"])


def capture(box: tuple[int, int, int, int]):
    """One grab of a screen rectangle, as a PIL image."""
    from PIL import ImageGrab

    left, top, right, bottom = box
    if right <= left or bottom <= top:
        raise ScreenError(f"empty capture box {box}")
    return ImageGrab.grab(bbox=box, all_screens=True)


# --------------------------------------------------------------------------- #
# Reading it
# --------------------------------------------------------------------------- #

#: A pixel counts as bar rather than track when it is this colourful and this
#: bright. Health bars are saturated green, amber or red; the track behind them
#: is grey or dark. Judging a pixel on its own terms is what makes an *empty*
#: bar readable -- the first version took the leftmost pixel as the filled
#: colour, which is true of every bar except the one that matters, and read a
#: fainted Pokemon at 100%.
BAR_SATURATION = 90
BAR_VALUE = 70


def bar_fraction(image, filled=None, empty=None, tolerance: int = 60) -> float:
    """How much of a health bar is filled, in 0..1.

    Measured rather than read. The bar's middle row is scanned and the run of
    bar-coloured pixels from the left is its length -- which survives the colour
    going green to amber to red on the way down, because the question asked of
    each pixel is "are you bar or are you track", not "which bar colour are you".

    ``filled`` or ``empty`` pin the two colours for a layout the default cannot
    tell apart. Calibration can store them; most bars do not need it.
    """
    import numpy as np

    pixels = np.asarray(image.convert("RGB"), dtype=np.int16)
    if pixels.size == 0:
        return 0.0
    row = pixels[pixels.shape[0] // 2]
    if filled is not None:
        is_bar = np.abs(row - np.array(filled, dtype=np.int16)).sum(axis=1) <= tolerance
    elif empty is not None:
        is_bar = np.abs(row - np.array(empty, dtype=np.int16)).sum(axis=1) > tolerance
    else:
        # int32, because (high - low) * 255 overflows int16 for exactly the
        # saturated colours this is here to detect: amber and red both wrap and
        # come back as "not a bar", so a Pokemon in the red read as 0%.
        high = row.max(axis=1).astype(np.int32)
        low = row.min(axis=1).astype(np.int32)
        # Saturation and value, the two HSV terms that separate a coloured bar
        # from the grey it sits on, without needing to know which colour it is.
        saturation = np.where(high > 0, (high - low) * 255 // np.maximum(high, 1), 0)
        is_bar = (saturation >= BAR_SATURATION) & (high >= BAR_VALUE)
    # The run from the left rather than the total, so a track drawn in a similar
    # colour further along cannot be counted as health.
    run = int(np.argmin(is_bar)) if not is_bar.all() else len(is_bar)
    return run / len(row)


def read_text(image, language: str = "ko") -> str:
    """Windows' own OCR, which is already installed and already speaks Korean.

    Tesseract would need installing and a language pack, and the cloud OCRs
    would send the screen somewhere. This is a PowerShell bridge to an API that
    is part of the operating system.
    """
    if not OCR_SCRIPT.exists():
        raise ScreenError(f"{OCR_SCRIPT} is missing")
    with tempfile.TemporaryDirectory() as folder:
        shot = Path(folder) / "region.png"
        image.save(shot)
        out = Path(folder) / "text.txt"
        command = (f"$OutputEncoding=[Console]::OutputEncoding=[Text.Encoding]::UTF8; "
                   f"& '{OCR_SCRIPT}' -Path '{shot}' -Lang '{language}' "
                   f"| Out-File -Encoding utf8 '{out}'")
        done = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-Command", command], capture_output=True, timeout=60)
        if not out.exists():
            raise ScreenError(
                "OCR failed: "
                + done.stderr.decode("utf-8", errors="replace")[:300])
        return out.read_text(encoding="utf-8-sig").strip()


def read(profile: Profile, names=None) -> dict[str, object]:
    """Every calibrated region at once, as text or as a bar fraction.

    A region whose name ends in ``_hp`` is measured; everything else is read.
    Failures are returned rather than raised -- one unreadable region should
    not cost the person the other five.
    """
    box = anchor(profile.window)
    shot = capture(box)
    origin = (box[0], box[1])
    out: dict[str, object] = {}
    for name, region in profile.regions.items():
        if names is not None and name not in names:
            continue
        left, top, right, bottom = region.pixels(box)
        crop = shot.crop((left - origin[0], top - origin[1],
                          right - origin[0], bottom - origin[1]))
        try:
            out[name] = (round(bar_fraction(crop) * 100) if name.endswith("_hp")
                         else read_text(crop))
        except Exception as error:  # one bad region is not six bad regions
            out[name] = None
            out.setdefault("_errors", {})[name] = str(error)[:200]
    return out
