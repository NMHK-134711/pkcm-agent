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

    def _existing(self) -> Profile:
        try:
            return Profile.load()
        except ScreenError:
            return Profile(window=self.window)

    def shot(self) -> dict:
        """The game window as a PNG the page can draw boxes on."""
        box = anchor(self.window)
        image = capture(box)
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
        """Read one box right now, so a bad rectangle is obvious immediately.

        Drawing a box and finding out three turns into a real game that it was
        two pixels short is the failure this is here to prevent.
        """
        box = anchor(self.window)
        left, top, right, bottom = region.pixels(box)
        crop = capture((left, top, right, bottom))
        if name.endswith(BAR_SUFFIX):
            return {"kind": "bar", "value": round(bar_fraction(crop) * 100)}
        return {"kind": "text", "value": read_text(crop)}

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
                    self._json(self.tool.shot())
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
