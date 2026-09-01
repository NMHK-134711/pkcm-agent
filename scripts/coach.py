"""Advise on a real Champions game, played in the actual client.

``play_web.py`` plays a battle we own. This one watches a battle we do not: the
game is running in Champions on this machine, and everything we know about the
opponent arrives by being typed in. Our own party is fixed -- the tournament
picked it -- and their six comes off team preview.

    python scripts/coach.py --party 7 --checkpoint runs/curriculum4/best.pt

Each turn is three things and they are ordered by how much they cost:

1. **The advice.** The search's recommendation, with how its visits were
   actually spread -- because "58% / 31% / 11%" is a different instruction from
   "94% / 4% / 2%" and the second one is worth trusting further.
2. **What they did.** One move or one switch. Moves are matched on the Korean
   name; a move we had not guessed is taught to their placeholder on the spot.
3. **What the bars say.** Our damage roll is not the game's, so HP drifts.
   Correcting it is two numbers and the page pre-fills them with the engine's
   own guess, so an uneventful turn is a single click.

**The advice is built from ``Observation.of(state, us)``, never from the
mirror's opponent.** That half of the state is a placeholder -- see
``pkcm.live.mirror`` -- and the point of the whole arrangement is that nothing
reads it. What the search knows about them is what has been entered.
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
from pkcm.engine.actions import Action, ActionKind  # noqa: E402
from pkcm.engine.legality import ranker_parties  # noqa: E402
from pkcm.engine.state import (  # noqa: E402
    BattleConfig,
    Phase,
    legal_actions,
)
from pkcm.envs.observation import Observation  # noqa: E402
from pkcm.live import Mirror, MirrorError  # noqa: E402
from pkcm.render.names import Names  # noqa: E402
from pkcm.render.text import Renderer  # noqa: E402
from pkcm.search import MCTS, SearchConfig  # noqa: E402

SPRITE_DIR = ROOT / "data" / "raw" / "sprites"
PAGE = ROOT / "scripts" / "coach.html"

STATUS_KO = {"brn": "화상", "par": "마비", "psn": "독", "tox": "맹독",
             "slp": "잠듦", "frz": "얼음"}
BOOST_KO = ("공격", "방어", "특공", "특방", "스피드", "명중", "회피")
SIDE_KO = {"spikes": "압정뿌리기", "toxicspikes": "독압정", "stealthrock": "스텔스록",
           "stickyweb": "끈적끈적네트", "reflect": "리플렉터", "lightscreen": "빛의장막",
           "auroraveil": "오로라베일", "tailwind": "순풍", "safeguard": "신비의부적",
           "mist": "하얀안개", "luckychant": "행운의주문"}
WEATHER_KO = {"sunnyday": "쨍쨍햇살", "raindance": "비", "sandstorm": "모래바람",
              "snowscape": "눈", "hail": "싸라기눈", "desolateland": "큰햇살",
              "primordialsea": "큰비", "deltastream": "델타스트림"}
TERRAIN_KO = {"electricterrain": "일렉트릭필드", "grassyterrain": "그래스필드",
              "mistyterrain": "미스트필드", "psychicterrain": "사이코필드"}
ROOM_KO = {"trickroom": "트릭룸", "magicroom": "매직룸", "wonderroom": "원더룸",
           "gravity": "중력"}
LAYERED = frozenset({"spikes", "toxicspikes", "stealthrock", "stickyweb"})

US, THEM = 0, 1

#: Which parties each shipped network was trained on, by the name in its path.
#:
#: These are specialists and the specialisation is the point: off its own
#: parties curriculum4 scores 42.7% [37.9, 47.6] against the handcrafted
#: search -- separably *worse* than no network at all. pilot43 is narrower
#: still, one network for one team. So a mismatch is worth saying out loud
#: rather than silently making the advice worse.
TRAINED_ON = {
    "pilot43": frozenset({43}),
    "curriculum4": frozenset({7, 10, 14, 17}),
}


def trained_on(checkpoint: str | None) -> frozenset[int] | None:
    """The parties this checkpoint was trained on, if we know it."""
    if not checkpoint:
        return None
    for name, parties in TRAINED_ON.items():
        if name in checkpoint.replace("\\", "/"):
            return parties
    return None


class Coach:
    """One live game being advised on."""

    def __init__(self, args) -> None:
        self.lock = threading.Lock()
        self.args = args
        self.dex = load_dex()
        self.regulation = self.dex.regulation("m_b")
        self.names = Names("ko", self.dex)
        self.renderer = Renderer("ko", self.dex, ("우리", "상대"))
        self.sprites = self._sprite_map()
        self.search = MCTS(
            SearchConfig(iterations=args.search_iterations,
                         determinizations=max(4, args.search_iterations // 20)),
            evaluator=self._evaluator())
        self.our_party = ranker_parties()[args.party]
        self.lookup = self._lookup()
        self.mirror: Mirror | None = None
        self.log: list[str] = []
        self.advice: dict | None = None
        self.error: str | None = None

    def _evaluator(self):
        """The trained network, if one was named.

        ``runs/curriculum4/best.pt`` is the first network here measured
        separably stronger than the handcrafted search -- but only on the four
        parties it was trained on, and separably *worse* on drawn teams. So
        this is an option rather than a default, and which party is being
        played decides whether it is the right one.
        """
        if not self.args.checkpoint:
            return None
        from pkcm.envs.encoding import SCALAR_SIZE, action_space_size
        from pkcm.train.evaluator import from_checkpoint

        config = BattleConfig(dex=self.dex, regulation=self.regulation,
                              battle_format=self.args.format)
        return from_checkpoint(
            self.args.checkpoint, self.dex,
            action_space_size(config.registered, config.brought),
            SCALAR_SIZE, device="cpu", trust=self.args.trust)

    def _sprite_map(self) -> dict[str, int]:
        manifest = SPRITE_DIR / "MANIFEST.json"
        if not manifest.exists():
            print("  (스프라이트 없음 -- scripts/fetch_sprites.py 실행; "
                  "이름으로만 표시됩니다)")
            return {}
        return json.loads(manifest.read_text(encoding="utf-8"))["species"]

    def _lookup(self) -> dict:
        """Korean name -> id, for everything that gets typed in.

        Only what the format allows, so a typo lands on a legal Pokemon or on
        nothing rather than on a Pokemon that cannot be in this game.
        """
        roster = self.regulation.legal_species | self.regulation.legal_megas
        species = [{"id": one, "ko": self.names.species(one)}
                   for one in sorted(roster)
                   if not self.dex.species[one].is_mega]
        moves = [{"id": one.id, "ko": self.names.move(one.id)}
                 for one in sorted(self.dex.moves.values(), key=lambda m: m.id)
                 if self.dex.exists_in_champions(one)]
        abilities = [{"id": one, "ko": self.names.ability(one)}
                     for one in sorted(self.dex.abilities)]
        from pkcm.engine.items import champions_items

        items = [{"id": one, "ko": self.names.item(one)}
                 for one in sorted(champions_items())]
        return {"species": species, "moves": moves,
                "abilities": abilities, "items": items}

    # -- the game ------------------------------------------------------------ #

    def begin(self, six: list[str]) -> None:
        self.mirror = Mirror.begin(self.dex, self.regulation,
                                   self.our_party.team, six,
                                   battle_format=self.args.format,
                                   seed=self.args.seed)
        self.log = [f"상대 등록: {', '.join(self.names.species(s) for s in six)}"]
        self.advice = None

    def think(self) -> None:
        """Ask the search what to play here."""
        if self.mirror is None or self.mirror.finished:
            return
        options = legal_actions(self.mirror.state, US)
        if all(action.kind is ActionKind.PASS for action in options):
            # Their replacement, our nothing. Reporting "pass, 100%, confident"
            # is noise at the moment the clock is running, and worse, it reads
            # as advice.
            self.advice = {"waiting": True, "best": "", "value": 0.0,
                           "options": []}
            return
        result = self.mirror.advise(self.search)
        ranked = sorted(result.distribution, key=lambda pair: -pair[1])
        self.advice = {
            "waiting": False,
            "best": self._describe(result.action),
            "value": round(getattr(result, "value", 0.0), 3),
            "options": [{"label": self._describe(choice),
                         "share": round(share, 3)}
                        for choice, share in ranked[:5] if share > 0.005],
        }

    def _describe(self, choice) -> str:
        action = choice[0]
        if action.kind is ActionKind.SELECT:
            return " → ".join(self.names.species(
                self.our_party.team[slot].species) for slot in action.selection)
        if action.kind is ActionKind.SWITCH:
            side = self.mirror.state.sides[US]
            slot = side.selection[action.index]
            return f"교체 → {self.names.species(self.our_party.team[slot].species)}"
        if action.kind is ActionKind.MOVE:
            slot = self.mirror.state.sides[US].active[0]
            move = self.mirror.state.moves(US, slot)[action.index]
            return ("메가진화 + " if action.mega else "") + self.names.move(move.id)
        if action.kind is ActionKind.PASS:
            return "대기"
        return str(action)

    # -- the view ------------------------------------------------------------ #

    def view(self) -> dict:
        payload = {
            "party": {"index": self.args.party, "title": self.our_party.title,
                      "team": [self._sheet(slot) for slot in
                               range(len(self.our_party.team))]},
            "lookup": self.lookup,
            "log": self.log[-40:],
            "error": self.error,
            "advice": self.advice,
            "started": self.mirror is not None,
        }
        self.error = None
        if self.mirror is None:
            return payload

        observation = Observation.of(self.mirror.state, US)
        payload.update({
            "phase": self.mirror.phase.name,
            "turn": self.mirror.state.turn,
            "finished": self.mirror.finished,
            "winner": self.mirror.state.winner,
            "their_six": [{"id": one, "ko": self.names.species(one),
                           "sprite": self.sprites.get(one)}
                          for one in self.mirror.their_six],
            "ours": [self._side_entry(observation, known, US)
                     for known in observation.own],
            "theirs": [self._side_entry(observation, known, THEM)
                       for known in observation.foe],
            "our_options": self._our_options(),
            "their_decision": self._their_decision(),
            "their_mega_used": self.mirror.state.mega_used[THEM],
            "our_active": self.active_name(US),
            "our_item": self.names.item(self.mirror.our_item()),
            "their_active": self.active_name(THEM),
            "field": self._field(),
            "our_conditions": self._conditions(observation.own_conditions),
            "their_conditions": self._conditions(observation.foe_conditions),
        })
        return payload

    def _sheet(self, slot: int) -> dict:
        mon = self.our_party.team[slot]
        return {"slot": slot, "ko": self.names.species(mon.species),
                "sprite": self.sprites.get(mon.species),
                "item": self.names.item(mon.item),
                "moves": [self.names.move(m) for m in mon.moves]}

    def _side_entry(self, observation, known, side: int) -> dict:
        """One Pokemon as the observation has it -- silhouette included."""
        species = known.species_id
        return {
            "slot": known.slot,
            "id": species,
            "ko": self.names.species(species) if species else "???",
            "sprite": self.sprites.get(species) if species else None,
            "active": known.position is not None,
            "fainted": known.fainted,
            "hp": round(known.hp_fraction * 100),
            "status": STATUS_KO.get(known.status or "", known.status or ""),
            # The id as well as the label: the correction form has to be able
            # to pre-select what is already there.
            "rawStatus": known.status or "",
            "boosts": [{"name": BOOST_KO[index], "value": value}
                       for index, value in enumerate(known.boosts) if value],
            "moves": [self.names.move(m) for m in known.moves],
            "item": self.names.item(known.item) if known.item_known else None,
        }

    def active_name(self, side: int) -> str:
        """Who the HP field is about. After a faint that is not who it was."""
        state = self.mirror.state
        side_state = state.sides[side]
        if not side_state.active or side_state.active[0] < 0:
            return ""
        slot = side_state.active[0]
        if side == THEM and slot not in state.revealed[THEM].species:
            return "???"
        return self.names.species(state.species_id(side, slot))

    def _our_options(self) -> list[dict]:
        """Everything we may legally submit, labelled.

        Built from ``legal_actions`` rather than from "four moves and a bench",
        because the phases where those differ are exactly the ones a hand-rolled
        list gets wrong: a forced switch offers only switches, and a side with
        nothing to decide offers only a pass.
        """
        options = []
        for action in legal_actions(self.mirror.state, US):
            options.append({"code": _encode(action),
                            "label": self._describe((action,)),
                            "pp": self._pp_of(action)})
        return options

    def _pp_of(self, action: Action) -> int | None:
        if action.kind is not ActionKind.MOVE:
            return None
        side = self.mirror.state.sides[US]
        slot = side.active[0] if side.active else -1
        if slot < 0 or action.index >= len(side.pp[slot]):
            return None
        return side.pp[slot][action.index]

    def _their_decision(self) -> bool:
        """Whether they have anything to decide this step.

        During our forced switch they do not, and asking the person to type
        what the opponent did when the opponent did nothing is how a coach
        gets abandoned two turns in.
        """
        return any(action.kind is not ActionKind.PASS
                   for action in legal_actions(self.mirror.state, THEM))

    def _field(self) -> list[str]:
        field = self.mirror.state.field
        out = []
        if field.weather:
            out.append(f"{WEATHER_KO.get(field.weather, field.weather)} "
                       f"{field.weather_turns}턴")
        if field.terrain:
            out.append(f"{TERRAIN_KO.get(field.terrain, field.terrain)} "
                       f"{field.terrain_turns}턴")
        for room, turns in field.rooms.items():
            out.append(f"{ROOM_KO.get(room, room)} {turns}턴")
        return out

    def _conditions(self, conditions) -> list[dict]:
        return [{"name": SIDE_KO.get(name, name), "value": value,
                 "unit": "겹" if name in LAYERED else "턴"}
                for name, value in conditions]


def _routes(coach: Coach, path: str, query: dict) -> None:
    """Everything the page can ask for that changes the game."""
    mirror = coach.mirror
    if path == "/begin":
        coach.begin(query.get("species", []))
        return
    if mirror is None:
        raise MirrorError("배틀이 아직 시작되지 않았습니다")

    if path == "/open":
        mirror.choose_ours([int(v) for v in query.get("pick", [])])
        mirror.their_lead(query["lead"][0])
        mirror.open()
        coach.log.append(
            f"선봉: 우리 {coach.names.species(coach.our_party.team[mirror.state.sides[US].selection[0]].species)}"
            f" / 상대 {coach.names.species(query['lead'][0])}")
        return
    if path == "/turn":
        ours = _decode(query["ours"][0])
        if not coach._their_decision():
            theirs = Action.PASS
        elif query.get("their_switch"):
            theirs = mirror.report_switch(query["their_switch"][0])
        else:
            theirs = mirror.report_move(
                query["their_move"][0],
                mega=query.get("their_mega", ["0"])[0] == "1")
        # Before the step, not after: an ability that announces itself mostly
        # does it by doing something, and Intimidate is a -1 on our Attack that
        # the engine can only apply if it knows about it when the switch
        # happens. A switch aims these at the Pokemon arriving, not the one
        # leaving.
        landing = theirs.index if theirs.kind is ActionKind.SWITCH else None
        _learned(coach, query, landing)
        events = mirror.advance(ours, theirs)
        coach.log.extend(coach.renderer.render_log(events).splitlines())
        # The correction rides along with the turn rather than following it.
        # Two buttons meant the first one produced advice computed on HP the
        # engine had guessed, and the second produced different advice on the
        # HP that was true -- and no way to tell from the page which of the two
        # to play. There is one number that matters and it is the one on the
        # screen, so it is entered with the turn.
        _correct(coach, query)
        return
    if path == "/observe":
        _learned(coach, query, None)
        _correct(coach, query)
        return
    raise MirrorError(f"알 수 없는 요청: {path}")


def _learned(coach: Coach, query: dict, slot: int | None) -> None:
    """Anything new about either set, from the turn we just watched."""
    ability = (query.get("their_ability") or [""])[0].strip()
    if ability:
        coach.mirror.report_ability(ability, slot)
    item = (query.get("their_item") or [""])[0].strip()
    if item:
        coach.mirror.report_item(
            item, slot,
            consumed=query.get("their_item_used", ["0"])[0] == "1")
    # Ours is not a guess, so the only thing to say about it is that it went.
    if query.get("our_item_used", ["0"])[0] == "1":
        coach.mirror.report_our_item(consumed=True)


def _correct(coach: Coach, query: dict) -> None:
    """Overwrite HP and status with what the person read off the screen."""
    for side_name, side in (("our", US), ("their", THEM)):
        hp = query.get(f"{side_name}_hp")
        status = query.get(f"{side_name}_status")
        if not hp and status is None:
            continue
        coach.mirror.observe(
            side,
            hp_fraction=float(hp[0]) / 100 if hp else None,
            status=(status[0] or None) if status is not None else ...)


def _encode(action: Action) -> str:
    """One action as a string the page can hand straight back."""
    if action.kind is ActionKind.MOVE:
        return f"m:{action.index}:{1 if action.mega else 0}"
    if action.kind is ActionKind.SWITCH:
        return f"s:{action.index}"
    return "p"


def _decode(code: str) -> Action:
    kind, _, rest = code.partition(":")
    if kind == "m":
        index, _, mega = rest.partition(":")
        return Action.move(int(index), mega=mega == "1")
    if kind == "s":
        return Action.switch(int(rest))
    return Action.PASS


class Handler(BaseHTTPRequestHandler):
    coach: Coach = None  # type: ignore[assignment]

    def log_message(self, *args) -> None:  # quiet
        pass

    def _send(self, payload: bytes, kind: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, view: dict) -> None:
        self._send(json.dumps(view, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self) -> None:
        route = urlparse(self.path)
        if route.path in ("/", "/index.html"):
            self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
            return
        if route.path.startswith("/sprite/"):
            _, _, rest = route.path.partition("/sprite/")
            view, _, name = rest.partition("/")
            target = (SPRITE_DIR / view / name).resolve()
            # A path from the network never gets to name a file outside here.
            if not target.is_file() or SPRITE_DIR.resolve() not in target.parents:
                self.send_error(404)
                return
            self._send(target.read_bytes(), "image/png")
            return
        if route.path == "/state":
            with self.coach.lock:
                self._json(self.coach.view())
            return

        query = parse_qs(route.query, keep_blank_values=True)
        with self.coach.lock:
            try:
                _routes(self.coach, route.path, query)
                # Re-think after anything that moved the game on. A report the
                # person is still typing has not moved it, which is why
                # /observe asks for advice too: the corrected HP is the thing
                # the leaf value reads.
                self.coach.think()
            except (MirrorError, KeyError, ValueError, IndexError) as error:
                self.coach.error = f"{type(error).__name__}: {error}"
            self._json(self.coach.view())


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):  # pragma: no cover
            pass

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--party", type=int, default=43,
                        help="which imported party we are playing; "
                             "scripts/tournament.py prints the indices. 43 is "
                             "the round robin's only team separably above the "
                             "field, and what runs/pilot43 was specialised on")
    parser.add_argument("--checkpoint", default=None,
                        help="a trained network for the prior and leaf value, "
                             "e.g. runs/pilot43/best.pt. Every one of these is "
                             "a specialist and separably *weaker* than the "
                             "handcrafted search off the parties it trained "
                             "on -- so pass it with a --party it knows, and "
                             "leave it off otherwise")
    parser.add_argument("--trust", type=float, default=1.0)
    parser.add_argument("--search-iterations", type=int, default=None,
                        help="defaults to the deploy budget. A real turn timer "
                             "is 45-90s and this costs a couple of seconds, so "
                             "there is room above it")
    parser.add_argument("--format", default="singles", choices=("singles", "doubles"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=8761)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if args.search_iterations is None:
        from pkcm.search.mcts import DEPLOY_ITERATIONS

        args.search_iterations = DEPLOY_ITERATIONS

    Handler.coach = Coach(args)
    print(f"우리 파티: {args.party} — {Handler.coach.our_party.title}")
    print(f"탐색: {args.search_iterations} iterations"
          + (f" + 망 {args.checkpoint}" if args.checkpoint
             else " (손으로 짠 탐색)"))
    known = trained_on(args.checkpoint)
    if known is not None and args.party not in known:
        print(f"  경고: 이 망은 파티 {sorted(known)} 에서 학습됐습니다. "
              f"파티 {args.party} 에서는 손으로 짠 탐색보다 약하다고 측정됐으니 "
              f"--checkpoint 없이 쓰는 쪽이 낫습니다.")
    elif args.checkpoint and known is None:
        print("  참고: 이 체크포인트가 어느 파티에서 학습됐는지 모릅니다 — "
              "coach.TRAINED_ON 에 없습니다.")
    address = f"http://127.0.0.1:{args.port}"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"열림: {address}   (Ctrl+C로 종료)")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(address)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
