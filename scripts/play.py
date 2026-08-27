"""Play a battle against the agent yourself.

An arena win rate is one number over eighty games and it hides the shape of the
mistakes. A rate of 27% can mean "slightly worse everywhere" or "fine in battle
and picks its team at random", and those want completely different fixes -- the
second one was in fact true of the first pre-trained network and the win rate
alone never said so. Twenty minutes at the keyboard finds that kind of thing
faster than another run does.

This is a check, not a measurement: one person's impression of a few games is
not a confidence interval, and nothing here should overrule ``scripts/arena.py``.

Usage:
    python scripts/play.py                                   # handcrafted search
    python scripts/play.py --checkpoint runs/imitate2/net.pt # the network
    python scripts/play.py --search-iterations 200           # a faster opponent

**You are shown exactly what a policy would be shown** -- the observation, not
the state. The opponent's unrevealed Pokemon, their items, their spreads and
their PP stay hidden, because a game where one side can read the other is not
the game the agent is being measured on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pkcm.data.dex import load_dex  # noqa: E402
from pkcm.engine.actions import TARGET_ALLY, TARGET_SELF, Action, ActionKind  # noqa: E402
from pkcm.engine.battle import step  # noqa: E402
from pkcm.engine.legality import random_team  # noqa: E402
from pkcm.engine.rng import Rng  # noqa: E402
from pkcm.engine.state import BattleConfig, Phase, legal_actions, new_battle  # noqa: E402
from pkcm.envs.observation import Observation  # noqa: E402
from pkcm.render.names import Names  # noqa: E402
from pkcm.render.text import Renderer  # noqa: E402
from pkcm.search import MCTS, SearchConfig  # noqa: E402

TRAINERS = ("당신", "에이전트")
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
#: Hazards stack; screens and Tailwind count down. The number means different
#: things and the label has to say which, or "압정뿌리기 2" reads as two turns.
LAYERED = frozenset({"spikes", "toxicspikes", "stealthrock", "stickyweb"})


# --------------------------------------------------------------------------- #
# What the human is allowed to see
# --------------------------------------------------------------------------- #


def describe(known, names: Names, mine: bool) -> str:
    """One Pokemon, from the observation. Never from the state."""
    if known is None:
        return "  (빈 자리)"
    # ``species_id is None`` means unknown, not absent: they brought something
    # to this slot and we have not seen it. Printing a blank name would read as
    # an empty slot, which is a different fact.
    name = names.species(known.species_id) if known.species_id else "???"
    bar = "기절" if known.fainted else f"HP {known.hp_fraction * 100:5.1f}%"
    marks = [bar]
    if known.status:
        marks.append(STATUS_KO.get(known.status, known.status))
    boosts = [f"{BOOST_KO[i]}{value:+d}" for i, value in enumerate(known.boosts)
              if value]
    if boosts:
        marks.append(" ".join(boosts))
    if known.item and known.item_known:
        marks.append(f"@{names.item(known.item)}")
    if known.ability and known.ability_known:
        marks.append(f"[{names.ability(known.ability)}]")
    if not mine and known.moves:
        marks.append("본 기술: " + ", ".join(names.move(m) for m in known.moves))
    where = "* " if known.position is not None else "  "
    return f"  {where}{name:<14} {'  '.join(marks)}"


def side_line(conditions, names: Names) -> str:
    parts = []
    for name, value in conditions:
        korean = SIDE_KO.get(name, name)
        parts.append(f"{korean} {value}층" if name in LAYERED
                     else f"{korean} {value}턴")
    return "  ".join(parts)


def show_weather(observation: Observation, names: Names) -> None:
    """Weather, terrain and rooms, with the turns left on each.

    The real game shows these and a player counts on them -- a Trick Room with
    one turn left and one with four are completely different positions.
    """
    parts = []
    if observation.weather:
        parts.append(f"{WEATHER_KO.get(observation.weather, observation.weather)} "
                     f"{observation.weather_turns}턴")
    if observation.terrain:
        parts.append(f"{TERRAIN_KO.get(observation.terrain, observation.terrain)} "
                     f"{observation.terrain_turns}턴")
    for name, turns in observation.rooms:
        parts.append(f"{ROOM_KO.get(name, name)} {turns}턴")
    if parts:
        print("  [ " + " | ".join(parts) + " ]")


def show_field(observation: Observation, names: Names) -> None:
    print()
    print(f"--- {observation.turn}턴 " + "-" * 44)
    show_weather(observation, names)
    theirs = side_line(observation.foe_conditions, names)
    print(f"{TRAINERS[1]}" + (f"    {theirs}" if theirs else ""))
    for known in sorted(observation.foe,
                        key=lambda k: (k.position is None, k.slot)):
        print(describe(known, names, mine=False))
    seen = {known.species_id for known in observation.foe if known.species_id}
    hidden = sum(1 for known in observation.foe if not known.species_id)
    if hidden:
        rest = [names.species(s) for s in observation.registered[1] if s not in seen]
        print(f"    ??? {hidden}마리는 등록 6마리 중 하나: {', '.join(rest)}")
    ours = side_line(observation.own_conditions, names)
    print(f"{TRAINERS[0]}" + (f"    {ours}" if ours else ""))
    for known in sorted(observation.own,
                        key=lambda k: (k.position is None, k.slot)):
        print(describe(known, names, mine=True))


# --------------------------------------------------------------------------- #
# Asking
# --------------------------------------------------------------------------- #


def label(state, player: int, action: Action, position: int,
          names: Names) -> str:
    if action.kind is ActionKind.MOVE:
        slot = state.sides[player].active[position]
        moves = state.moves(player, slot)
        name = names.move(moves[action.index].id) if action.index < len(moves) \
            else f"기술{action.index}"
        parts = [name]
        if action.mega:
            parts.insert(0, "메가진화 +")
        if action.target == TARGET_SELF:
            parts.append("(자신)")
        elif action.target == TARGET_ALLY:
            parts.append("(파트너)")
        elif action.target:
            parts.append(f"(상대 {action.target}번 자리)")
        return " ".join(parts)
    if action.kind is ActionKind.SWITCH:
        selection = state.sides[player].selection
        species = (state.pokemon(player, action.index).species.id
                   if action.index < len(selection) else None)
        return f"교체 → {names.species(species) if species else action.index}"
    if action.kind is ActionKind.STRUGGLE:
        return "발버둥"
    if action.kind is ActionKind.PASS:
        return "넘김"
    return str(action)


def damage_notes(state, player: int, position: int, names: Names) -> dict[int, str]:
    """What each of our moves would do, per move index.

    Straight from ``pkcm.envs.analysis`` -- the same calculator the agent reads,
    on the same observation, so the two of us are looking at the same numbers.
    The brackets are wide because the opponent's spread is hidden; that is the
    honest width, not a rounding.
    """
    from pkcm.envs.analysis import assess
    from pkcm.envs.encoding import Vocabulary
    from pkcm.envs.reference import sheet_for

    dex = state.config.dex
    observation = Observation.of(state, player)
    found = assess(observation, sheet_for(dex, Vocabulary.of(dex)), dex, position)
    if found is None:
        return {}
    attacker = next((k for k in observation.own if k.position == position), None)
    if attacker is None:
        return {}
    order = {move_id: index for index, move_id in enumerate(attacker.moves)}

    notes: dict[int, str] = {}
    for _slot, estimate in found.damage:
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
            marks.append(f"{hits.low}타" if hits.certain else f"{hits.low}~{hits.high}타")
        if estimate.hit_chance < 1.0:
            marks.append(f"명중 {estimate.hit_chance * 100:.0f}%")
        if estimate.survivable:
            marks.append("버틸 수단 있을 수 있음")
        elif estimate.blunted_possible:
            marks.append("특성으로 반감될 수 있음")
        notes[index] = "  ".join(marks)

    for _slot, faster in found.outspeeds:
        if faster is not None:
            notes[-1] = "선공" if faster else "후공"
        else:
            notes[-1] = "선후공 불명"
    return notes


def ask_number(prompt: str, low: int, high: int) -> int:
    while True:
        try:
            raw = input(prompt).strip()
        except EOFError:
            raise SystemExit("\n(입력이 끊겼습니다)")
        if raw.lower() in ("q", "quit", "exit"):
            raise SystemExit("\n그만둡니다.")
        if raw.isdigit() and low <= int(raw) <= high:
            return int(raw)
        print(f"  {low}에서 {high} 사이의 숫자를 넣어주세요 (q로 종료).")


def ask_pick(state, player: int, names: Names) -> tuple[Action, ...]:
    """Team preview. Enumerating all 120 orderings would be unreadable, so the
    three are asked for one at a time, which is how the game asks too."""
    registered = state.config.registered
    brought = state.config.brought
    print()
    print(f"=== 팀 프리뷰 — {brought}마리를 순서대로 고르세요 (1번이 선봉) ===")
    observation = Observation.of(state, player)
    print(f"{TRAINERS[1]}의 등록 6마리")
    for species_id in observation.registered[1]:
        print(f"    {names.species(species_id)}")
    print(f"{TRAINERS[0]}의 등록 6마리")
    party = state.parties[player]
    for slot in range(registered):
        pokemon = party[slot]
        moves = ", ".join(names.move(move.id) for move in pokemon.moves)
        item = names.item(pokemon.item) if pokemon.item else "-"
        print(f"  [{slot}] {names.species(pokemon.species.id):<14} @ {item:<12} "
              f"HP {pokemon.max_hp:<4} 스피드 {pokemon.stats[5]:<4}")
        print(f"      {moves}")

    order: list[int] = []
    while len(order) < brought:
        remaining = [s for s in range(registered) if s not in order]
        which = "선봉" if not order else f"{len(order) + 1}번째"
        print(f"  고를 수 있는 번호: {remaining}")
        choice = ask_number(f"  {which}> ", 0, registered - 1)
        if choice in order:
            print("  이미 고른 번호입니다.")
            continue
        order.append(choice)
    return (Action.select(*order),)


def ask_actions(state, player: int, names: Names) -> tuple[Action, ...]:
    if state.phase is Phase.TEAM_PREVIEW:
        return ask_pick(state, player, names)

    positions = state.config.active_count
    picks: list[Action] = []
    taken: set[int] = set()
    for position in range(positions):
        # The one rule no per-position mask can carry: the same Pokemon may not
        # be sent to two positions. ``battle.step`` would reject the pair.
        options = [action for action in legal_actions(state, player, position)
                   if not (action.kind is ActionKind.SWITCH and action.index in taken)]
        if not options:
            picks.append(Action.PASS)
            continue
        if len(options) == 1:
            picks.append(options[0])
            continue
        if positions > 1:
            print(f"  -- {position}번 자리 --")
        notes = damage_notes(state, player, position, names)
        if notes.get(-1):
            print(f"    ({notes[-1]})")
        for number, action in enumerate(options):
            note = notes.get(action.index, "") if action.kind is ActionKind.MOVE else ""
            text = label(state, player, action, position, names)
            print(f"    [{number}] {text:<24} {note}")
        chosen = options[ask_number("  > ", 0, len(options) - 1)]
        if chosen.kind is ActionKind.SWITCH:
            taken.add(chosen.index)
        picks.append(chosen)
    return tuple(picks)


# --------------------------------------------------------------------------- #


def use_utf8() -> None:
    """The Windows console defaults to cp949 here, which cannot encode an em
    dash and mangles the Korean. Everything this prints is UTF-8."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):  # pragma: no cover - a pipe, or a tty
            pass                           # that is already UTF-8


def main() -> int:
    use_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None,
                        help="a saved network for the agent's prior and leaf "
                             "value. Without one it plays the handcrafted search")
    parser.add_argument("--search-iterations", type=int, default=800)
    parser.add_argument("--trust", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None,
                        help="fix the teams. Omit for a different pair each time")
    parser.add_argument("--format", default="singles", choices=("singles", "doubles"))
    parser.add_argument("--lang", default="ko", choices=("ko", "en"))
    args = parser.parse_args()

    seed = args.seed
    if seed is None:
        # Not Rng: this only has to differ between sittings.
        import time as _time

        seed = int(_time.time()) % 100000
        print(f"(팀 시드 {seed} — 같은 팀을 다시 보려면 --seed {seed})")

    dex = load_dex()
    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"),
                          battle_format=args.format)
    teams = tuple(
        random_team(dex, config.regulation,
                    Rng.from_seed(seed * 2 + offset).cursor(), args.format)
        for offset in (1, 2))
    state = new_battle(config, teams, seed=seed)

    search = SearchConfig(iterations=args.search_iterations,
                          determinizations=max(4, args.search_iterations // 20))
    evaluator = None
    if args.checkpoint:
        from pkcm.envs.encoding import SCALAR_SIZE, action_space_size
        from pkcm.train.evaluator import from_checkpoint

        evaluator = from_checkpoint(
            args.checkpoint, dex,
            action_space_size(config.registered, config.brought),
            SCALAR_SIZE, device="cpu", trust=args.trust)
    agent = MCTS(search, evaluator=evaluator)
    cursor = Rng.from_seed(seed ^ 0xA9E27).cursor()
    renderer = Renderer(args.lang, dex, TRAINERS)

    print(f"상대: {'망 + 탐색 ' + args.checkpoint if args.checkpoint else '손으로 짠 탐색'}"
          f" ({args.search_iterations} iterations)")

    human, machine = 0, 1
    while not state.finished:
        if state.phase is not Phase.TEAM_PREVIEW:
            show_field(Observation.of(state, human), renderer.names)
        mine = ask_actions(state, human, renderer.names)
        theirs = agent.choose(state, machine, cursor).action
        state, log = step(state, mine, theirs)
        text = renderer.render_log(log)
        if text.strip():
            print()
            print(text)

    print()
    print("=" * 54)
    if state.winner is None:
        print(f"무승부 ({state.turn}턴)")
    else:
        print(f"{TRAINERS[state.winner]} 승 ({state.turn}턴)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
