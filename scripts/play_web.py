"""Play the agent in a browser, with sprites.

``scripts/play.py`` says everything this does and says it in a terminal, which
is hard to read a field position out of: weather, hazards, six Pokemon and
their boosts arrive as lines of text and a person has to assemble the picture.
Sprites and a layout do that assembling.

The engine is Python, so the page cannot hold the game. This serves it: one
process, one battle, a handful of JSON endpoints, and a browser drawing what
they return. Nothing is installed and nothing is a dependency -- it is
``http.server`` and a page.

**The page is built from ``Observation.of(state, you)`` and never from the
state.** That is the same rule the terminal version holds to, and it matters
more here because a UI is exactly where "just show it, it is only a display"
gets said. The opponent's unrevealed Pokemon are silhouettes because the
observation has ``species_id is None`` for them, not because the page chose to
hide something it had.

    python scripts/fetch_sprites.py     # once
    python scripts/play_web.py          # opens http://127.0.0.1:8760

Options mirror play.py: --checkpoint, --search-iterations, --seed, --format.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pkcm.data.dex import load_dex  # noqa: E402
from pkcm.engine.actions import TARGET_ALLY, TARGET_SELF, Action, ActionKind  # noqa: E402
from pkcm.engine.battle import step  # noqa: E402
from pkcm.engine.legality import make_team  # noqa: E402
from pkcm.engine.rng import Rng  # noqa: E402
from pkcm.engine.state import BattleConfig, Phase, legal_actions, new_battle  # noqa: E402
from pkcm.envs.observation import Observation  # noqa: E402
from pkcm.render.names import Names  # noqa: E402
from pkcm.render.text import Renderer  # noqa: E402
from pkcm.search import MCTS, SearchConfig  # noqa: E402

SPRITE_DIR = ROOT / "data" / "raw" / "sprites"
PAGE = ROOT / "scripts" / "play_web.html"

STATUS_KO = {"brn": "화상", "par": "마비", "psn": "독", "tox": "맹독",
             "slp": "잠듦", "frz": "얼음"}
BOOST_KO = ("공격", "방어", "특공", "특방", "스피드", "명중", "회피")
SIDE_KO = {"spikes": "압정뿌리기", "toxicspikes": "독압정", "stealthrock": "스텔스록",
           "stickyweb": "끈적끈적네트", "reflect": "리플렉터", "lightscreen": "빛의장막",
           "auroraveil": "오로라베일", "tailwind": "순풍", "safeguard": "신비의부적",
           "mist": "하얀안개", "luckychant": "행운의주문"}
ROOM_KO = {"trickroom": "트릭룸", "magicroom": "매직룸", "wonderroom": "원더룸",
           "gravity": "중력"}
WEATHER_KO = {"sunnyday": "쨍쨍햇살", "raindance": "비", "sandstorm": "모래바람",
              "snowscape": "눈", "hail": "싸라기눈", "desolateland": "큰햇살",
              "primordialsea": "큰비", "deltastream": "델타스트림"}
TERRAIN_KO = {"electricterrain": "일렉트릭필드", "grassyterrain": "그래스필드",
              "mistyterrain": "미스트필드", "psychicterrain": "사이코필드"}
#: Hazards stack, screens count down. Two different numbers wearing one label.
LAYERED = frozenset({"spikes", "toxicspikes", "stealthrock", "stickyweb"})

YOU, AGENT = 0, 1


class Battle:
    """One game, and the only mutable thing here."""

    def __init__(self, args) -> None:
        self.lock = threading.Lock()
        self.dex = load_dex()
        self.names = Names("ko", self.dex)
        self.renderer = Renderer("ko", self.dex, ("당신", "에이전트"))
        self.config = BattleConfig(dex=self.dex,
                                   regulation=self.dex.regulation("m_b"),
                                   battle_format=args.format)
        self.args = args
        self.sprites = self._sprite_map()
        search = SearchConfig(
            iterations=args.search_iterations,
            determinizations=max(4, args.search_iterations // 20))
        self.evaluator = None
        if args.checkpoint:
            from pkcm.envs.encoding import SCALAR_SIZE, action_space_size
            from pkcm.train.evaluator import from_checkpoint

            self.evaluator = from_checkpoint(
                args.checkpoint, self.dex,
                action_space_size(self.config.registered, self.config.brought),
                SCALAR_SIZE, device="cpu", trust=args.trust)
        self.search = search
        self.log: list[str] = []
        self.start(args.seed)

    def _sprite_map(self) -> dict[str, int]:
        manifest = SPRITE_DIR / "MANIFEST.json"
        if not manifest.exists():
            print("  (no sprites -- run scripts/fetch_sprites.py; "
                  "the page will fall back to names)")
            return {}
        return json.loads(manifest.read_text(encoding="utf-8"))["species"]

    # -- the game -------------------------------------------------------- #

    def start(self, seed: int) -> None:
        self.seed = seed
        teams = tuple(
            make_team(self.dex, self.config.regulation,
                      Rng.from_seed(seed * 2 + offset).cursor(),
                      self.args.format, self.args.teams)
            for offset in (1, 2))
        self.state = new_battle(self.config, teams, seed=seed)
        self.agent = MCTS(self.search, evaluator=self.evaluator)
        self.cursor = Rng.from_seed(seed ^ 0xA9E27).cursor()
        self.log = []

    def submit(self, indices: list[int]) -> None:
        """Take the human's choice, let the agent answer, advance one step."""
        mine = self._chosen(indices)
        theirs = self.agent.choose(self.state, AGENT, self.cursor).action
        self.state, events = step(self.state, mine, theirs)
        text = self.renderer.render_log(events)
        if text.strip():
            self.log.extend(text.splitlines())

    def _chosen(self, indices: list[int]) -> tuple[Action, ...]:
        if self.state.phase is Phase.TEAM_PREVIEW:
            return (Action.select(*indices),)
        picks: list[Action] = []
        taken: set[int] = set()
        for position, index in enumerate(indices):
            options = self._options(position, taken)
            if not options:
                picks.append(Action.PASS)
                continue
            action = options[min(max(index, 0), len(options) - 1)]
            if action.kind is ActionKind.SWITCH:
                taken.add(action.index)
            picks.append(action)
        return tuple(picks)

    def _options(self, position: int, taken: set[int]) -> list[Action]:
        """The one rule a per-position mask cannot carry: no Pokemon twice."""
        return [action for action in legal_actions(self.state, YOU, position)
                if not (action.kind is ActionKind.SWITCH and action.index in taken)]

    # -- what the page is allowed to know --------------------------------- #

    def view(self) -> dict:
        observation = Observation.of(self.state, YOU)
        return {
            "phase": self.state.phase.name,
            "turn": observation.turn,
            "finished": self.state.finished,
            "winner": self.state.winner,
            "field": self._field(observation),
            "you": [self._seen(known, mine=True) for known in observation.own],
            "foe": [self._seen(known, mine=False) for known in observation.foe],
            "yours": self._side(observation.own_conditions),
            "theirs": self._side(observation.foe_conditions),
            "registered": [
                [self._named(species) for species in observation.registered[0]],
                [self._named(species) for species in observation.registered[1]],
            ],
            "choices": self._choices(),
            "log": self.log[-40:],
            "seed": self.seed,
        }

    def _named(self, species_id: str | None) -> dict:
        return {"id": species_id,
                "name": self.names.species(species_id) if species_id else "???",
                "sprite": self.sprites.get(species_id) if species_id else None}

    def _seen(self, known, mine: bool) -> dict:
        entry = self._named(known.species_id)
        entry.update({
            "slot": known.slot,
            "active": known.position is not None,
            "fainted": known.fainted,
            "hp": round(known.hp_fraction * 100, 1),
            "status": STATUS_KO.get(known.status, known.status),
            "boosts": [{"name": BOOST_KO[index], "value": value}
                       for index, value in enumerate(known.boosts) if value],
            # ``item_known`` and ``ability_known`` are the observation saying
            # whether it has *seen* these, not whether they exist.
            "item": (self.names.item(known.item)
                     if known.item and known.item_known else None),
            "ability": (self.names.ability(known.ability)
                        if known.ability and known.ability_known else None),
            "moves": [self.names.move(move) for move in known.moves] if not mine else [],
        })
        return entry

    def _side(self, conditions) -> list[dict]:
        return [{"name": SIDE_KO.get(name, name), "value": value,
                 "unit": "층" if name in LAYERED else "턴"}
                for name, value in conditions]

    def _field(self, observation) -> list[str]:
        parts = []
        if observation.weather:
            parts.append(f"{WEATHER_KO.get(observation.weather, observation.weather)} "
                         f"{observation.weather_turns}턴")
        if observation.terrain:
            parts.append(f"{TERRAIN_KO.get(observation.terrain, observation.terrain)} "
                         f"{observation.terrain_turns}턴")
        for name, turns in observation.rooms:
            parts.append(f"{ROOM_KO.get(name, name)} {turns}턴")
        return parts

    def _choices(self) -> list[dict]:
        """What we may submit, with the calculator's read on each move."""
        if self.state.finished:
            return []
        if self.state.phase is Phase.TEAM_PREVIEW:
            return [{"kind": "select", "brought": self.config.brought,
                     "team": [self._named(self.state.parties[YOU][slot].species.id)
                              | {"slot": slot,
                                 "moves": [self.names.move(move.id) for move in
                                           self.state.parties[YOU][slot].moves],
                                 "item": (self.names.item(
                                     self.state.parties[YOU][slot].item)
                                     if self.state.parties[YOU][slot].item else None)}
                              for slot in range(self.config.registered)]}]

        notes = self._damage_notes()
        positions = []
        taken: set[int] = set()
        for position in range(self.config.active_count):
            options = self._options(position, taken)
            positions.append({
                "kind": "act",
                "position": position,
                "speed": notes.get(position, {}).get("speed"),
                "options": [self._option(position, index, action, notes)
                            for index, action in enumerate(options)],
            })
        return positions

    def _option(self, position: int, index: int, action: Action, notes) -> dict:
        entry = {"index": index, "kind": action.kind.name.lower(),
                 "label": self._label(position, action)}
        if action.kind is ActionKind.MOVE:
            entry["note"] = notes.get(position, {}).get(action.index)
        elif action.kind is ActionKind.SWITCH:
            species = self.state.pokemon(YOU, action.index).species.id
            entry["sprite"] = self.sprites.get(species)
        return entry

    def _label(self, position: int, action: Action) -> str:
        if action.kind is ActionKind.MOVE:
            slot = self.state.sides[YOU].active[position]
            moves = self.state.moves(YOU, slot)
            name = (self.names.move(moves[action.index].id)
                    if action.index < len(moves) else f"기술{action.index}")
            parts = ["메가진화 + " + name] if action.mega else [name]
            if action.target == TARGET_SELF:
                parts.append("(자신)")
            elif action.target == TARGET_ALLY:
                parts.append("(파트너)")
            elif action.target:
                parts.append(f"(상대 {action.target}번)")
            return " ".join(parts)
        if action.kind is ActionKind.SWITCH:
            return self.names.species(self.state.pokemon(YOU, action.index).species.id)
        if action.kind is ActionKind.STRUGGLE:
            return "발버둥"
        return "넘김"

    def _damage_notes(self) -> dict:
        """From ``envs.analysis`` -- the same calculator the agent reads."""
        from pkcm.envs.analysis import assess
        from pkcm.envs.encoding import Vocabulary
        from pkcm.envs.reference import sheet_for

        observation = Observation.of(self.state, YOU)
        sheet = sheet_for(self.dex, Vocabulary.of(self.dex))
        found: dict[int, dict] = {}
        for position in range(self.config.active_count):
            assessment = assess(observation, sheet, self.dex, position)
            if assessment is None:
                continue
            attacker = next((k for k in observation.own
                             if k.position == position), None)
            if attacker is None:
                continue
            order = {move_id: index for index, move_id in enumerate(attacker.moves)}
            notes: dict = {}
            for _slot, estimate in assessment.damage:
                index = order.get(estimate.move_id)
                if index is None:
                    continue
                low, high = estimate.percent.low, estimate.percent.high
                marks = [f"{low}~{high}%" if low != high else f"{low}%"]
                if estimate.effectiveness == 0:
                    marks = ["효과 없음"]
                elif estimate.guaranteed_ko:
                    marks.append("확정 1타")
                elif estimate.ko_chance > 0:
                    marks.append(f"난수 1타 {estimate.ko_chance * 100:.0f}%")
                else:
                    hits = estimate.hits_to_ko
                    marks.append(f"{hits.low}타" if hits.certain
                                 else f"{hits.low}~{hits.high}타")
                if estimate.hit_chance < 1.0:
                    marks.append(f"명중 {estimate.hit_chance * 100:.0f}%")
                notes[index] = "  ".join(marks)
            for _slot, faster in assessment.outspeeds:
                notes["speed"] = ("선공" if faster else "후공") if faster is not None \
                    else "선후공 불명"
            found[position] = notes
        return found


class Handler(BaseHTTPRequestHandler):
    battle: Battle = None  # type: ignore[assignment]

    def log_message(self, *args) -> None:  # quiet
        pass

    def _send(self, payload: bytes, kind: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        route = urlparse(self.path)
        if route.path in ("/", "/index.html"):
            self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
        elif route.path == "/state":
            with self.battle.lock:
                view = self.battle.view()
            self._send(json.dumps(view, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
        elif route.path.startswith("/sprite/"):
            _, _, rest = route.path.partition("/sprite/")
            view, _, name = rest.partition("/")
            target = (SPRITE_DIR / view / name).resolve()
            # A path from the network never gets to name a file outside here.
            if not target.is_file() or SPRITE_DIR.resolve() not in target.parents:
                self.send_error(404)
                return
            self._send(target.read_bytes(), "image/png")
        elif route.path == "/act":
            query = parse_qs(route.query)
            picks = [int(value) for value in query.get("pick", [])]
            with self.battle.lock:
                if not self.battle.state.finished and picks:
                    self.battle.submit(picks)
                view = self.battle.view()
            self._send(json.dumps(view, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
        elif route.path == "/new":
            query = parse_qs(route.query)
            seed = int(query.get("seed", [self.battle.seed + 1])[0])
            with self.battle.lock:
                self.battle.start(seed)
                view = self.battle.view()
            self._send(json.dumps(view, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
        else:
            self.send_error(404)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):  # pragma: no cover
            pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None,
                        help="a saved network for the agent's prior and leaf "
                             "value. Without one it plays the handcrafted search")
    parser.add_argument("--search-iterations", type=int, default=None,
                        help="defaults to the deploy budget the measurements "
                             "say to play at")
    parser.add_argument("--trust", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--format", default="singles", choices=("singles", "doubles"))
    parser.add_argument("--teams", default="ranker", choices=("random", "ranker"))
    parser.add_argument("--port", type=int, default=8760)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if args.search_iterations is None:
        try:
            from pkcm.search.mcts import DEPLOY_ITERATIONS

            args.search_iterations = DEPLOY_ITERATIONS
        except ImportError:  # pragma: no cover - older checkout
            args.search_iterations = 800
    if args.seed is None:
        import time as _time

        args.seed = int(_time.time()) % 100000

    print(f"상대: {'망 + 탐색 ' + args.checkpoint if args.checkpoint else '손으로 짠 탐색'}"
          f" ({args.search_iterations} iterations, {args.teams} 팀)")
    Handler.battle = Battle(args)

    address = f"http://127.0.0.1:{args.port}"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"열림: {address}   (Ctrl+C로 종료)")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(address)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n종료합니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
