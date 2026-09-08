# 효과가 있는데, 그 문장을 직접 재보지는 않은 기술

`python scripts/coverage_audit.py`가 만듭니다. 포맷의 기술 497개 중 설명이 없거나 "No additional effect."뿐인 것 31개를 빼면 **466개**가 효과 문장을 갖고, 그 중 **436개**는 모든 효과 문장마다 그 문장을 읽고 쓴 probe가 붙어 있습니다.

아래 **30개**가 나머지입니다. `clause_check`가 그 문장을 다른 검사에 *위임*했고, "그 검사가 이 문장을 덮는다"는 건 제 판단이지 측정이 아닙니다. 전부 초록이지만, 초록의 근거가 문장이 아니라 제 짐작입니다.

드래곤옐이 정확히 이 자리에 있었습니다: 문장은 급소율을 말하는데 위임된 검사는 능력치 랭크를 재고 있었고, 잴 것이 없으니 조용히 통과시켰습니다.

(테라스탈·블루오브·하늘가르기처럼 이 포맷에 없는 기전만 언급하는 문장은 도달 불가로 기록되어 있고, 여기서는 세지 않았습니다.)

## B등급 -- 25개

그 기술 전용 수기 검사가 있습니다. 다만 그 검사가 아래 문장을 덮는다는 것은 확인된 적이 없습니다.

- **헤롱헤롱** (Attract, `attract`, Status)
  - The effect ends when either the user or the target is no longer active.
- **오로라베일** (Aurora Veil, `auroraveil`, Status)
  - Brick Break and Psychic Fangs remove the effect before damage is calculated.
  - Fails unless the weather is Snow.
- **배턴터치** (Baton Pass, `batonpass`, Status)
  - The user is replaced with another Pokemon in its party.
  - The selected Pokemon has the user's stat stage changes transferred to it, as well as the effects of confusion, Aqua Ring, Curse, Dragon Cheer, Embargo, Focus Energy, Gastro Acid, Heal Block, Ingrain, Leech Seed, Lock-On (Mind Reader), Magnet Rise, Perish Song, Power Trick, Telekinesis, and a substitute with its remaining HP.
  - The effect of Gastro Acid is not transferred if the recipient has an Ability that cannot be affected.
- **벌레먹기** (Bug Bite, `bugbite`, Physical)
  - If this move is successful and the user has not fainted, it steals the target's held Berry if it is holding one and eats it immediately, gaining its effects even if the user's item is being ignored.
- **흉내쟁이** (Copycat, `copycat`, Status)
  - The user uses the last move used by any Pokemon, including itself.
- **안개제거** (Defog, `defog`, Status)
  - If this move is successful and whether or not the target's evasiveness was affected, the effects of Reflect, Light Screen, Aurora Veil, Safeguard, Mist, Spikes, Toxic Spikes, Stealth Rock, and Sticky Web end for the target's side, and the effects of Spikes, Toxic Spikes, Stealth Rock, and Sticky Web end for the user's side.
  - Ignores a target's substitute, although a substitute will still block the lowering of evasiveness.
- **길동무** (Destiny Bond, `destinybond`, Status)
  - Until the user's next move, if an opposing Pokemon's attack knocks the user out, that Pokemon faints as well, unless the attack was Doom Desire or Future Sight.
- **송전** (Electrify, `electrify`, Status)
  - Among effects that can change a move's type, this effect happens last.
- **치유소원** (Healing Wish, `healingwish`, Status)
  - The user faints, and if the Pokemon brought out to replace it does not have full HP or has a non-volatile status condition, its HP is fully restored along with having any non-volatile status condition cured.
  - The replacement is sent out at the end of the turn, and the healing happens before hazards take effect.
  - This effect continues until a Pokemon that meets either of these conditions switches in at the user's position or gets swapped into the position with Ally Switch.
- **치유파동** (Heal Pulse, `healpulse`, Status)
  - If the user has the Mega Launcher Ability, the target instead restores 3/4 of its maximum HP, rounded half down.
- **봉인** (Imprison, `imprison`, Status)
  - The user prevents all opposing Pokemon from using any moves that the user also knows as long as the user remains active.
- **록온** (Lock-On, `lockon`, Status)
  - Until the end of the next turn, the target cannot avoid the user's moves, even if the target is in the middle of a two-turn move.
- **마법가루** (Magic Powder, `magicpowder`, Status)
  - Fails if the target is an Arceus or a Silvally, if the target is already purely Psychic type, or if the target is Terastallized.
- **매직룸** (Magic Room, `magicroom`, Status)
  - An item's effect of causing forme changes is unaffected, but any other effects from such items are negated.
- **막말내뱉기** (Parting Shot, `partingshot`, Status)
  - The user does not switch out if the target's Attack and Special Attack stat stages were both unchanged, or if there are no unfainted party members.
- **쪼아대기** (Pluck, `pluck`, Physical)
  - If this move is successful and the user has not fainted, it steals the target's held Berry if it is holding one and eats it immediately, gaining its effects even if the user's item is being ignored.
- **리사이클** (Recycle, `recycle`, Status)
  - The user regains the item it last used.
  - Items thrown with Fling can be regained.
- **미러타입** (Reflect Type, `reflecttype`, Status)
  - If the target's current types include typeless and a non-added type, typeless is ignored.
  - If the target's current types include typeless and an added type from Forest's Curse or Trick-or-Treat, typeless is copied as the Normal type instead.
- **역할** (Role Play, `roleplay`, Status)
  - The user's Ability changes to match the target's Ability.
- **꼬리자르기** (Shed Tail, `shedtail`, Status)
  - The user takes 1/2 of its maximum HP, rounded up, and creates a substitute that has 1/4 of the user's maximum HP, rounded down.
  - The user is replaced with another Pokemon in its party and the selected Pokemon has the substitute transferred to it.
- **스킬스왑** (Skill Swap, `skillswap`, Status)
  - The user swaps its Ability with the target's Ability.
  - Fails if either the user or the target's Ability is As One, Battle Bond, Comatose, Commander, Disguise, Embody Aspect, Hunger Switch, Ice Face, Illusion, Multitype, Neutralizing Gas, Poison Puppeteer, Power Construct, Protosynthesis, Quark Drive, RKS System, Schooling, Shields Down, Stance Change, Tera Shell, Tera Shift, Teraform Zero, Wonder Guard, Zen Mode, or Zero to Hero.
- **다과회** (Teatime, `teatime`, Status)
  - This effect is not prevented by substitutes, the Klutz or Unnerve Abilities, or the effects of Embargo or Magic Room.
- **독압정** (Toxic Spikes, `toxicspikes`, Status)
  - Safeguard prevents the opposing party from being poisoned on switch-in, but a substitute does not.
- **변신** (Transform, `transform`, Status)
  - The user transforms into the target.
  - The target's current stats, stat stages, types, moves, Ability, weight, gender, and sprite are copied.
  - The user's level and HP remain the same and each copied move receives only 5 PP, with a maximum of 5 PP each.
- **소란피기** (Uproar, `uproar`, Special)
  - The user spends three turns locked into this move.
  - This move targets an opponent at random on each turn.

## C등급 -- 5개

기술 데이터의 필드(secondary/boosts/status)가 실제 배틀에 나타나는지 자동 검사합니다. 문장이 필드보다 더 말하는 부분은 측정되지 않습니다.

- **썰렁개그** (Chilly Reception, `chillyreception`, Status)
  - The user switches out even if it is trapped and is replaced immediately by a selected party member.
- **힘껏펀치** (Focus Punch, `focuspunch`, Physical)
  - The user loses its focus and does nothing if it is hit by a damaging attack this turn before it can execute the move.
- **미래예지** (Future Sight, `futuresight`, Special)
  - If the user is no longer active at the time, damage is calculated based on the user's natural Special Attack stat, types, and level, with no boosts from its held item or Ability.
- **비장의무기** (Last Resort, `lastresort`, Physical)
  - This move fails unless the user knows this move and at least one other move, and has used all the other moves it knows at least once each since it became active or Transformed.
- **셸암즈** (Shell Side Arm, `shellsidearm`, Special)
  - This move becomes a physical attack that makes contact if the value of ((((2 * the user's level / 5 + 2) * 90 * X) / Y) / 50), where X is the user's Attack stat and Y is the target's Defense stat, is greater than the same value where X is the user's Special Attack stat and Y is the target's Special Defense stat.
  - No stat modifiers other than stat stage changes are considered for this purpose.

