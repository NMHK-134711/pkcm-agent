"""Turn the engine's event log into readable lines.

A consumer of the log, not part of the engine (docs/DESIGN.md §1e). The engine
emits **ids**; this is where they become names, and the only place that knows
what language anyone reads. A sprite renderer would be a sibling of this module
and would need no engine changes.
"""

from __future__ import annotations

from pkcm.engine.events import Event
from pkcm.render.names import DEFAULT_LANGUAGE, Names


#: Korean particles agree with the last syllable of the word before them, so
#: "핫삼" takes 은/을/이 and "루차불" takes... also those, because it ends in a
#: consonant too. Picking one and living with it produces "핫삼를", which reads
#: as badly in Korean as "a apple" does in English.
_HANGUL_START, _HANGUL_END, _JONGSEONG = 0xAC00, 0xD7A3, 28

_PARTICLES = {
    "\uc740": ("\uc740", "\ub294"),   # 은 / 는
    "\uc774": ("\uc774", "\uac00"),   # 이 / 가
    "\uc744": ("\uc744", "\ub97c"),   # 을 / 를
    "\uacfc": ("\uacfc", "\uc640"),   # 과 / 와
    "\uc73c\ub85c": ("\uc73c\ub85c", "\ub85c"),  # 으로 / 로
}


def has_final_consonant(word: str) -> bool | None:
    """``None`` when the last character is not a Hangul syllable at all."""
    if not word:
        return None
    code = ord(word[-1])
    if not _HANGUL_START <= code <= _HANGUL_END:
        return None
    return (code - _HANGUL_START) % _JONGSEONG != 0


def josa(word: str, particle: str) -> str:
    """``word`` followed by whichever form of ``particle`` fits it."""
    forms = _PARTICLES.get(particle)
    if forms is None:
        return f"{word}{particle}"
    final = has_final_consonant(word)
    if final is None:
        # Latin fallback: assume no final consonant, which is what a reader
        # would say aloud for an English word.
        final = False
    with_final, without = forms
    # 으로/로 is the exception: a word ending in ㄹ takes the short form.
    if particle == "\uc73c\ub85c" and final and (ord(word[-1]) - _HANGUL_START) % _JONGSEONG == 8:
        return f"{word}{without}"
    return f"{word}{with_final if final else without}"


EFFECTIVENESS_TEXT = {
    "ko": {0.25: "효과가 별로인 것 같다...", 0.5: "효과가 별로인 것 같다...",
           2.0: "효과가 굉장했다!", 4.0: "효과가 굉장했다!"},
    "en": {0.25: "It's not very effective...", 0.5: "It's not very effective...",
           2.0: "It's super effective!", 4.0: "It's super effective!"},
}

STATUS_TEXT = {
    "ko": {"brn": "화상을 입었다", "par": "마비되었다", "psn": "독에 걸렸다",
           "tox": "맹독에 걸렸다", "slp": "잠들어 버렸다", "frz": "얼어붙었다"},
    "en": {"brn": "was burned", "par": "was paralyzed", "psn": "was poisoned",
           "tox": "was badly poisoned", "slp": "fell asleep", "frz": "was frozen solid"},
}

CANT_MOVE_TEXT = {
    "ko": {"par": "몸이 저려서 움직일 수 없다!", "slp": "쿨쿨 잠들어 있다.",
           "frz": "얼어붙어서 움직일 수 없다!", "flinch": "풀이 죽어서 움직일 수 없다!",
           "attract": "헤롱헤롱해서 움직일 수 없다!"},
    "en": {"par": "is paralyzed! It can't move!", "slp": "is fast asleep.",
           "frz": "is frozen solid!", "flinch": "flinched!", "attract": "is in love!"},
}

RESIDUAL_TEXT = {
    "ko": {"brn": "화상 데미지를 입었다", "psn": "독 데미지를 입었다",
           "tox": "맹독 데미지를 입었다", "leechseed": "씨뿌리기에 체력을 빼앗겼다",
           "sandstorm": "모래바람에 시달렸다", "spikes": "압정에 찔렸다",
           "stealthrock": "스텔스록에 부딪혔다", "substitute": "분신을 만들었다",
           "lifeorb": "생명의구슬에 체력을 깎였다", "dryskin": "건조피부로 체력이 줄었다"},
    "en": {"brn": "was hurt by its burn", "psn": "was hurt by poison",
           "tox": "was hurt by poison", "leechseed": "had its health sapped",
           "sandstorm": "was buffeted by the sandstorm", "spikes": "was hurt by Spikes",
           "stealthrock": "was hurt by Stealth Rock", "substitute": "put in a substitute"},
}

FIELD_TEXT = {
    "ko": {"sunnyday": "햇살이 강해졌다!", "raindance": "비가 내리기 시작했다!",
           "sandstorm": "모래바람이 불기 시작했다!", "snowscape": "눈이 내리기 시작했다!",
           "electricterrain": "발밑에 전기가 흐르기 시작했다!",
           "grassyterrain": "발밑에 풀이 무성해졌다!",
           "mistyterrain": "발밑에 안개가 자욱해졌다!",
           "psychicterrain": "발밑이 이상해졌다!",
           "trickroom": "차원이 뒤틀렸다!"},
    "en": {"sunnyday": "The sunlight turned harsh!", "raindance": "It started to rain!",
           "sandstorm": "A sandstorm kicked up!", "snowscape": "It started to snow!",
           "electricterrain": "An electric current ran across the battlefield!",
           "grassyterrain": "Grass grew to cover the battlefield!",
           "mistyterrain": "Mist swirled around the battlefield!",
           "psychicterrain": "The battlefield got weird!",
           "trickroom": "The dimensions were twisted!"},
}

BOOST_STAT_TEXT = {
    "ko": {"atk": "공격", "def": "방어", "spa": "특수공격", "spd": "특수방어",
           "spe": "스피드", "accuracy": "명중률", "evasion": "회피율"},
}

SIDE_CONDITION_TEXT = {
    "ko": {"reflect": "리플렉터", "lightscreen": "빛의장막", "auroraveil": "오로라베일",
           "tailwind": "순풍", "spikes": "압정뿌리기", "toxicspikes": "독압정",
           "stealthrock": "스텔스록", "stickyweb": "끈적끈적네트", "safeguard": "안전지대"},
}


class Renderer:
    """Turns events into lines. One per language, cheap to make."""

    __slots__ = ("language", "names", "trainers")

    def __init__(self, language: str = DEFAULT_LANGUAGE, dex=None,
                 trainers: tuple[str, str] | None = None) -> None:
        self.language = language
        self.names = Names(language, dex)
        self.trainers = trainers

    # -- helpers ---------------------------------------------------------- #

    def _phrase(self, table: dict, key, default=None):
        return table.get(self.language, table.get("en", {})).get(key, default)

    def _who(self, event: Event) -> str:
        if self.trainers is None or event.side is None:
            return f"P{event.side}"
        return self.trainers[event.side]

    def _hp(self, event: Event) -> str:
        return f"-> {event.hp}/{event.max_hp}"

    def _pokemon(self, event: Event) -> str:
        """Whoever the event is about, by name.

        Doubles made this necessary: with four on the field, "it" is ambiguous
        and the side alone no longer identifies anyone.
        """
        if event.species:
            return self.names.species(event.species)
        if event.side is None or event.slot is None:
            return "?"
        return f"{self._who(event)}의 {event.slot + 1}번" if self.language == "ko" \
            else f"{self._who(event)}'s #{event.slot + 1}"

    def _stat(self, key: str | None) -> str:
        return self._phrase(BOOST_STAT_TEXT, key, key) or (key or "")

    def _side_condition(self, key: str | None) -> str:
        return self._phrase(SIDE_CONDITION_TEXT, key, key) or (key or "")

    # -- the log ---------------------------------------------------------- #

    def render(self, event: Event) -> list[str]:
        handler = getattr(self, f"_on_{event.kind}", None)
        if handler is None:
            return [repr(event)]
        return handler(event)

    def render_log(self, log: list[Event]) -> str:
        lines: list[str] = []
        for event in log:
            lines.extend(self.render(event))
        return "\n".join(lines)

    # -- per event -------------------------------------------------------- #

    def _on_turn_start(self, e):
        return ["", f"--- {e.turn}턴 ---" if self.language == "ko" else f"--- Turn {e.turn} ---"]

    def _on_team_preview(self, e):
        return [f"{self._who(e)} 선출: {e.detail}"]

    def _on_switch_in(self, e):
        name = self.names.species(e.species)
        if self.language == "ko":
            subject = josa(self._who(e), "\uc740")
            return [f"{subject} {josa(name, '\uc744')} 내보냈다! ({e.hp}/{e.max_hp})"]
        return [f"{self._who(e)} sent out {name}! ({e.hp}/{e.max_hp})"]

    # -- doubles ---------------------------------------------------------- #
    #
    # Reading the log is how most of this engine's bugs were caught, so an
    # event that renders as ``repr(Event(...))`` is a blind spot rather than a
    # cosmetic problem -- and doubles put four Pokemon in every turn to lose
    # track of.

    def _on_redirected(self, e):
        who = self._pokemon(e)
        if self.language == "ko":
            return [f"  {josa(who, '\uac00')} 공격을 대신 받아냈다!"]
        return [f"  {who} drew the attack!"]

    def _on_ally_switch(self, e):
        if self.language == "ko":
            return ["  파트너와 자리를 바꿨다!"]
        return ["  It swapped places with its ally!"]

    def _on_move_order(self, e):
        who = self._pokemon(e)
        later = e.detail == "quash"
        if self.language == "ko":
            return [f"  {josa(who, '\uc758')} 순서가 " + ("뒤로 밀렸다!" if later else "앞당겨졌다!")]
        return [f"  {who} will move " + ("last!" if later else "next!")]

    def _on_instructed(self, e):
        who, move = self._pokemon(e), self.names.move(e.move)
        if self.language == "ko":
            return [f"  {josa(who, '\uc774')} {josa(move, '\uc744')} 다시 사용했다!"]
        return [f"  {who} used {move} again!"]

    def _on_position_empty(self, e):
        if self.language == "ko":
            return [f"  {self._who(e)}는 더 이상 내보낼 포켓몬이 없다!"]
        return [f"  {self._who(e)} has nobody left to send out!"]

    # -- everything else that used to print as a repr ---------------------- #

    def _on_charging(self, e):
        move = self.names.move(e.move)
        if self.language == "ko":
            return [f"  {josa(move, '\uc744')} 준비하고 있다!"]
        return [f"  It is charging {move}!"]

    def _on_recharging(self, e):
        return ["  움직일 수 없다! 반동으로 쉬고 있다!" if self.language == "ko"
                else "  It must recharge!"]

    def _on_avoided(self, e):
        return ["  하지만 닿지 않았다!" if self.language == "ko"
                else "  ...but it could not be reached!"]

    def _on_endured(self, e):
        return ["  공격을 버텨냈다!" if self.language == "ko"
                else "  It endured the hit!"]

    def _on_dragged_out(self, e):
        who = self._pokemon(e)
        if self.language == "ko":
            return [f"  {josa(who, '\uc774')} 강제로 교체되었다!"]
        return [f"  {who} was dragged out!"]

    def _on_self_switch(self, e):
        who = self._pokemon(e)
        if self.language == "ko":
            return [f"  {josa(who, '\uc774')} 돌아왔다!"]
        return [f"  {who} went back!"]

    def _on_self_destruct(self, e):
        who = self._pokemon(e)
        if self.language == "ko":
            return [f"  {josa(who, '\uc740')} 자폭했다!"]
        return [f"  {who} blew itself up!"]

    def _on_set_hp(self, e):
        return [f"  {self._hp(e)}"]

    def _on_pp_lost(self, e):
        move = self.names.move(e.move)
        if self.language == "ko":
            return [f"  {josa(move, '\uc758')} PP가 줄어들었다!"]
        return [f"  The PP of {move} was reduced!"]

    def _on_ability_swapped(self, e):
        return ["  특성이 바뀌었다!" if self.language == "ko"
                else "  Abilities were swapped!"]

    def _on_items_swapped(self, e):
        return ["  도구를 교환했다!" if self.language == "ko"
                else "  Items were swapped!"]

    def _on_boosts_copied(self, e):
        return ["  랭크 변화를 복사했다!" if self.language == "ko"
                else "  It copied the stat changes!"]

    def _on_copy_boosts(self, e):
        return self._on_boosts_copied(e)

    def _on_boosts_swapped(self, e):
        return ["  랭크 변화를 교환했다!" if self.language == "ko"
                else "  Stat changes were swapped!"]

    def _on_boosts_cleared(self, e):
        return ["  랭크 변화가 사라졌다!" if self.language == "ko"
                else "  Stat changes were removed!"]

    def _on_clear_boosts(self, e):
        return self._on_boosts_cleared(e)

    def _on_team_cured(self, e):
        return ["  파티 전원의 상태이상이 회복되었다!" if self.language == "ko"
                else "  The whole team was cured!"]

    def _on_called_move(self, e):
        move = self.names.move(e.move)
        if self.language == "ko":
            return [f"  {josa(move, '\uc744')} 불러냈다!"]
        return [f"  It called {move}!"]

    def _on_heal_blocked(self, e):
        return ["  회복이 봉인되어 있다!" if self.language == "ko"
                else "  It cannot heal!"]

    def _on_cant_move(self, e):
        who = self._pokemon(e)
        if self.language == "ko":
            return [f"  {josa(who, '\uc740')} 움직일 수 없다!"]
        return [f"  {who} could not move!"]

    def _on_move_used(self, e):
        species, move = self.names.species(e.species), self.names.move(e.move)
        if self.language == "ko":
            return [f"{species}의 {move}!"]
        return [f"{species} used {move}!"]

    def _on_missed(self, e):
        return ["  하지만 빗나갔다!" if self.language == "ko" else "  ...but it missed!"]

    def _on_immune(self, e):
        return ["  효과가 없는 것 같다..." if self.language == "ko"
                else "  It doesn't affect the target..."]

    def _on_protected(self, e):
        move = self.names.move(e.move)
        return [f"  {josa(move, '\uc744')} 막아냈다!" if self.language == "ko"
                else f"  It protected itself from {move}!"]

    def _on_move_failed(self, e):
        detail = f" ({e.detail})" if e.detail else ""
        return [f"  하지만 실패했다!{detail}" if self.language == "ko"
                else f"  ...but it failed!{detail}"]

    def _on_unimplemented(self, e):
        return [f"  (미구현: {e.detail})" if self.language == "ko"
                else f"  (not implemented yet: {e.detail})"]

    def _on_cant_move(self, e):
        return [f"  {self._phrase(CANT_MOVE_TEXT, e.detail, e.detail)}"]

    def _on_confused(self, e):
        return ["  혼란에 빠져 자신을 공격했다!" if self.language == "ko"
                else "  It hurt itself in its confusion!"]

    def _on_damage(self, e):
        lines = []
        if e.crit:
            lines.append("  급소에 맞았다!" if self.language == "ko" else "  A critical hit!")
        note = self._phrase(EFFECTIVENESS_TEXT, e.effectiveness)
        if note:
            lines.append(f"  {note}")
        percent = 100 * e.amount / e.max_hp if e.max_hp else 0
        lines.append(f"  -{e.amount} ({percent:.0f}%) {self._hp(e)}")
        return lines

    def _residual(self, e):
        text = self._phrase(RESIDUAL_TEXT, e.detail, e.detail)
        return [f"  {text}. -{e.amount} {self._hp(e)}"]

    _on_status_damage = _residual
    _on_weather_damage = _residual
    _on_hazard_damage = _residual

    def _on_recoil(self, e):
        detail = self._phrase(RESIDUAL_TEXT, e.detail)
        if detail:
            return [f"  {detail}. -{e.amount} {self._hp(e)}"]
        return [f"  반동 데미지를 입었다! -{e.amount} {self._hp(e)}" if self.language == "ko"
                else f"  It was hurt by recoil! -{e.amount} {self._hp(e)}"]

    def _on_heal(self, e):
        source = f" ({self.names.item(e.detail) if e.detail else ''})" if e.detail else ""
        return [f"  체력을 {e.amount} 회복했다{source}. {self._hp(e)}" if self.language == "ko"
                else f"  It restored {e.amount} HP{source}. {self._hp(e)}"]

    def _on_substitute_hit(self, e):
        return [f"  분신이 대신 맞았다! (-{e.amount})" if self.language == "ko"
                else f"  The substitute took the hit! (-{e.amount})"]

    def _on_multi_hit(self, e):
        return [f"  {e.amount}번 맞았다!" if self.language == "ko"
                else f"  Hit {e.amount} times!"]

    def _on_status(self, e):
        return [f"  {self._phrase(STATUS_TEXT, e.detail, e.detail)}!"]

    def _on_cure_status(self, e):
        return [f"  상태이상이 회복되었다." if self.language == "ko"
                else f"  It shook off its {e.detail}."]

    def _on_status_immune(self, e):
        return ["  효과가 없는 것 같다..." if self.language == "ko"
                else "  It can't be affected by that."]

    def _on_boost(self, e):
        stat = self._stat(e.detail)
        if self.language == "ko":
            size = {1: "", 2: "크게 ", 3: "매우 크게 "}.get(abs(e.amount), "")
            verb = "올라갔다" if e.amount > 0 else "떨어졌다"
            return [f"  {josa(stat, '\uc774')} {size}{verb}! ({e.hp:+d})"]
        direction = "rose" if e.amount > 0 else "fell"
        return [f"  Its {stat} {direction}! (now {e.hp:+d})"]

    def _on_boost_failed(self, e):
        stat = self._stat(e.detail)
        limit = "더 오르지 않는다" if e.amount > 0 else "더 내려가지 않는다"
        return [f"  {josa(stat, '\uc740')} {limit}!" if self.language == "ko"
                else f"  Its {stat} won't go any {'higher' if e.amount > 0 else 'lower'}!"]

    def _on_boost_restored(self, e):
        return ["  떨어진 능력이 원래대로 돌아왔다!" if self.language == "ko"
                else "  Its lowered stats were restored!"]

    def _on_volatile_start(self, e):
        return [f"  {e.detail} 시작." if self.language == "ko" else f"  {e.detail} started."]

    def _on_volatile_end(self, e):
        return [f"  {e.detail} 종료." if self.language == "ko" else f"  {e.detail} ended."]

    def _field_start(self, e):
        return [f"  {self._phrase(FIELD_TEXT, e.detail, e.detail)}"]

    _on_weather_start = _field_start
    _on_terrain_start = _field_start
    _on_room_start = _field_start

    def _field_end(self, e):
        return [f"  {self._phrase(FIELD_TEXT, e.detail, e.detail)} 효과가 사라졌다."
                if self.language == "ko" else f"  The {e.detail} ended."]

    _on_weather_end = _field_end
    _on_terrain_end = _field_end
    _on_room_end = _field_end

    #: Screens and Tailwind count down turns; hazards stack layers. The number
    #: means different things and should not be printed the same way.
    TIMED_SIDE_CONDITIONS = ("reflect", "lightscreen", "auroraveil", "tailwind")

    def _on_side_condition(self, e):
        name = self._side_condition(e.detail)
        timed = e.detail in self.TIMED_SIDE_CONDITIONS
        if self.language == "ko":
            measure = f"{e.amount}턴" if timed else f"{e.amount}겹"
            return [f"  {self._who(e)} 쪽에 {name} ({measure})"]
        measure = f"{e.amount} turns" if timed else f"x{e.amount}"
        return [f"  {self._who(e)}'s side: {name} ({measure})"]

    def _on_side_condition_end(self, e):
        return [f"  {self._who(e)} 쪽의 {self._side_condition(e.detail)} 효과가 사라졌다."
                if self.language == "ko"
                else f"  {self._who(e)}'s {e.detail} wore off."]

    def _on_hazard_absorbed(self, e):
        return [f"  {self._who(e)} 쪽의 {josa(self._side_condition(e.detail), '\uc774')} "
                f"흡수되었다."]

    def _on_mega_evolve(self, e):
        name = self.names.species(e.species)
        return [f"  {josa(name, '\uc73c\ub85c')} 메가진화했다!" if self.language == "ko"
                else f"  {name}! It Mega Evolved!"]

    def _on_forme_change(self, e):
        return []  # whatever caused it says something more useful

    def _on_transform(self, e):
        return [f"  {josa(self.names.species(e.detail), '\uc73c\ub85c')} 변신했다!"
                if self.language == "ko"
                else f"  It transformed into {self.names.species(e.detail)}!"]

    def _on_type_change(self, e):
        return [f"  타입이 {josa(self.names.type(e.detail), '\uc73c\ub85c')} 바뀌었다!"
                if self.language == "ko"
                else f"  Its type changed to {e.detail}!"]

    def _on_ability(self, e):
        return [f"  [{self.names.ability(e.detail)}]"]

    def _on_ability_block(self, e):
        ability, move = self.names.ability(e.detail), self.names.move(e.move)
        return [f"  {josa(ability, '\uc73c\ub85c')} {josa(move, '\uc744')} 막았다!"
                if self.language == "ko"
                else f"  {ability} blocked {move}!"]

    def _on_ability_suppressed(self, e):
        return [f"  ({josa(self.names.ability(e.detail), '\uc774')} 무시되었다)"]

    def _on_ability_change(self, e):
        return [f"  특성이 {josa(self.names.ability(e.detail), '\uc73c\ub85c')} 바뀌었다!"]

    def _on_item(self, e):
        return [f"  [{self.names.item(e.detail)}]"]

    def _on_item_used(self, e):
        return [f"  {josa(self.names.item(e.detail), '\uc744')} 다 썼다."
                if self.language == "ko"
                else f"  Its {self.names.item(e.detail)} was used up."]

    def _on_faint(self, e):
        return [f"  {josa(self.names.species(e.species), '\uc740')} 쓰러졌다!"
                if self.language == "ko"
                else f"  {self.names.species(e.species)} fainted!"]

    def _on_battle_end(self, e):
        if e.side is None:
            return ["", f"무승부. ({e.detail})" if self.language == "ko"
                    else f"Draw. ({e.detail})"]
        return ["", f"{self._who(e)} 승리! ({e.detail})" if self.language == "ko"
                else f"{self._who(e)} wins! ({e.detail})"]


def render_log(log: list[Event], trainers: tuple[str, str] | None = None,
               language: str = DEFAULT_LANGUAGE, dex=None) -> str:
    return Renderer(language, dex, trainers).render_log(log)


def render(event: Event, trainers: tuple[str, str] | None = None,
           language: str = DEFAULT_LANGUAGE, dex=None) -> list[str]:
    return Renderer(language, dex, trainers).render(event)
