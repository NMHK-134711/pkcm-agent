"""Draw once where the game keeps things, so the coach can read them.

    python scripts/calibrate.py            # opens http://127.0.0.1:8770

Nothing in this project knows what Champions looks like, and nothing should:
a layout written from memory is a layout that drifts the first time the window
moves or the emulator rescales. So the positions come from here -- a capture of
the game window with boxes dragged over it -- and they are stored as fractions
of the window, which is what makes them survive a resize.

The result is ``data/screen_profile.json``. Run it again whenever the layout
changes; it takes about a minute.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pkcm.live.screen import (  # noqa: E402
    BAR_SUFFIX,
    REGIONS,
    Profile,
    Region,
    ScreenError,
    anchor,
    bar_fraction,
    capture,
    read_text,
    windows,
)

PAGE = ROOT / "scripts" / "calibrate.html"


class Calibrator:
    def __init__(self, args) -> None:
        self.lock = threading.Lock()
        self.window = args.window
        self.profile = self._existing()
        #: The capture the page is drawing on. Samples are cropped out of this
        #: rather than grabbed afresh -- see ``sample``.
        self.shot_image = None

    def _existing(self) -> Profile:
        try:
            return Profile.load()
        except ScreenError:
            return Profile(window=self.window)

    def shot(self, delay: float = 0.0) -> dict:
        """The game window as a PNG the page can draw boxes on.

        ``delay`` is there because capture reads the *screen*: whatever is in
        front of the game at those coordinates is what gets grabbed, and the
        browser doing the calibrating is a window like any other. A few seconds
        is enough to click the game and let it come forward.
        """
        if delay > 0:
            time.sleep(min(delay, 15.0))
        box = anchor(self.window)
        image = capture(box)
        self.shot_image = image
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return {
            "image": "data:image/png;base64,"
                     + base64.b64encode(buffer.getvalue()).decode("ascii"),
            "width": image.width, "height": image.height,
            "window": self.window,
            "regions": {name: {"left": box.left, "top": box.top,
                               "right": box.right, "bottom": box.bottom}
                        for name, box in self.profile.regions.items()},
            "wanted": REGIONS,
            "bar_suffix": BAR_SUFFIX,
        }

    def sample(self, name: str, region: Region) -> dict:
        """Read one box, so a bad rectangle is obvious immediately.

        Drawing a box and finding out three turns into a real game that it was
        two pixels short is the failure this is here to prevent.

        **Cropped out of the capture the page is showing, never grabbed
        afresh.** A second grab reads the screen as it is *now*, which is with
        the browser in front of the game -- so every sample came back reading
        whatever was covering it. One capture, many samples, and what is
        measured is what is on the picture being drawn on.
        """
        if self.shot_image is None:
            raise ScreenError("아직 캡처가 없습니다 -- 다시 캡처를 누르세요")
        width, height = self.shot_image.size
        crop = self.shot_image.crop((
            round(region.left * width), round(region.top * height),
            round(region.right * width), round(region.bottom * height)))
        if crop.width < 2 or crop.height < 2:
            raise ScreenError(f"영역이 너무 작습니다 ({crop.width}x{crop.height})")
        if name.endswith(BAR_SUFFIX):
            return {"kind": "bar", "value": round(bar_fraction(crop) * 100),
                    "size": f"{crop.width}x{crop.height}"}
        return {"kind": "text", "value": read_text(crop),
                "size": f"{crop.width}x{crop.height}"}

    def save(self, window: str, regions: dict) -> str:
        self.window = window
        self.profile = Profile(window=window, regions=regions)
        return str(self.profile.save())


class Handler(BaseHTTPRequestHandler):
    tool: Calibrator = None  # type: ignore[assignment]

    def log_message(self, *args) -> None:  # quiet
        pass

    def _json(self, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        route = urlparse(self.path)
        query = parse_qs(route.query)
        try:
            if route.path in ("/", "/index.html"):
                body = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            with self.tool.lock:
                if route.path == "/shot":
                    if query.get("window"):
                        self.tool.window = query["window"][0]
                    self._json(self.tool.shot(
                        float(query.get("delay", ["0"])[0])))
                elif route.path == "/windows":
                    self._json({"windows": [one["title"] for one in windows()]})
                elif route.path == "/sample":
                    name = query["name"][0]
                    region = Region(*[float(query[key][0])
                                      for key in ("left", "top", "right", "bottom")])
                    self._json(self.tool.sample(name, region))
                elif route.path == "/save":
                    regions = {
                        name: Region(**{key: float(value)
                                        for key, value in box.items()})
                        for name, box in json.loads(query["regions"][0]).items()
                    }
                    where = self.tool.save(query["window"][0], regions)
                    self._json({"saved": where})
                else:
                    self.send_error(404)
        except Exception as error:  # the page shows it; the server keeps running
            self._json({"error": f"{type(error).__name__}: {error}"})


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):  # pragma: no cover
            pass

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--window", default="BlueStacks",
                        help="part of the game window's title. Empty means the "
                             "whole screen")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    Handler.tool = Calibrator(args)
    print("보정할 영역:")
    for name, why in REGIONS.items():
        print(f"  {name:16} {why}")
    address = f"http://127.0.0.1:{args.port}"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"\n열림: {address}   (Ctrl+C로 종료)")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(address)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
