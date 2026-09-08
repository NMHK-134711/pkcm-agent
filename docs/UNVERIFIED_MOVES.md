# 효과가 있는데, 그 문장을 직접 재보지는 않은 기술

`python scripts/coverage_audit.py`가 만듭니다. 포맷의 기술 497개 중 설명이 없거나 "No additional effect."뿐인 것 29개를 빼면 **468개**가 효과 문장을 갖고, 그 중 **188개**는 모든 효과 문장마다 그 문장을 읽고 쓴 probe가 붙어 있습니다.

아래 **280개**가 나머지입니다. `clause_check`가 그 문장을 다른 검사에 *위임*했고, "그 검사가 이 문장을 덮는다"는 건 제 판단이지 측정이 아닙니다. 전부 초록이지만, 초록의 근거가 문장이 아니라 제 짐작입니다.

드래곤옐이 정확히 이 자리에 있었습니다: 문장은 급소율을 말하는데 위임된 검사는 능력치 랭크를 재고 있었고, 잴 것이 없으니 조용히 통과시켰습니다.

(테라스탈·블루오브·하늘가르기처럼 이 포맷에 없는 기전만 언급하는 문장은 도달 불가로 기록되어 있고, 여기서는 세지 않았습니다.)

## A등급 -- 2개

위임된 검사가 '싱글에서 아군이 없어 실패한다'만 확인합니다. 기술이 무엇을 하는지는 아무도 재지 않았습니다 -- 드래곤옐이 여기 있었습니다.

- **아로마미스트** (Aromatic Mist, `aromaticmist`, Status)
  - Raises the target's Special Defense by 1 stage.
- **코칭** (Coaching, `coaching`, Status)
  - Raises the target's Attack and Defense by 1 stage.

## B등급 -- 87개

그 기술 전용 수기 검사가 있습니다. 다만 그 검사가 아래 문장을 덮는다는 것은 확인된 적이 없습니다.

### 설명이 기술 그 자체 (mechanic_check의 수기 검사에 통째로 위임) -- 53개

- **아쿠아링** (Aqua Ring, `aquaring`, Status)
  - The user has 1/16 of its maximum HP, rounded down, restored at the end of each turn while it remains active.
- **헤롱헤롱** (Attract, `attract`, Status)
  - The effect ends when either the user or the target is no longer active.
- **토치카** (Baneful Bunker, `banefulbunker`, Status)
  - The user is protected from most attacks made by other Pokemon during this turn, and Pokemon making contact with the user become poisoned.
- **배턴터치** (Baton Pass, `batonpass`, Status)
  - The user is replaced with another Pokemon in its party.
  - The selected Pokemon has the user's stat stage changes transferred to it, as well as the effects of confusion, Aqua Ring, Curse, Dragon Cheer, Embargo, Focus Energy, Gastro Acid, Heal Block, Ingrain, Leech Seed, Lock-On (Mind Reader), Magnet Rise, Perish Song, Power Trick, Telekinesis, and a substitute with its remaining HP.
  - The effect of Gastro Acid is not transferred if the recipient has an Ability that cannot be affected.
- **집단구타** (Beat Up, `beatup`, Physical)
  - The power of each hit is equal to 5+(X/10), where X is each participating Pokemon's base Attack; each hit is considered to come from the user.
- **바디프레스** (Body Press, `bodypress`, Physical)
  - Other effects that modify the Attack stat are used as normal.
- **벌레먹기** (Bug Bite, `bugbite`, Physical)
  - If this move is successful and the user has not fainted, it steals the target's held Berry if it is holding one and eats it immediately, gaining its effects even if the user's item is being ignored.
- **흉내쟁이** (Copycat, `copycat`, Status)
  - The user uses the last move used by any Pokemon, including itself.
- **부식가스** (Corrosive Gas, `corrosivegas`, Status)
  - This move cannot cause Pokemon with the Sticky Hold Ability to lose their held item or cause a Kyogre, Groudon, Dialga, Palkia, Giratina, Arceus, Genesect, Silvally, Zacian, Zamazenta, Paradox Pokemon, or Ogerpon to lose their Blue Orb, Red Orb, Adamant Crystal, Lustrous Globe, Griseous Core, Plate, Drive, Memory, Rusted Sword, Rusted Shield, Booster Energy, or Mask, respectively.
- **길동무** (Destiny Bond, `destinybond`, Status)
  - Until the user's next move, if an opposing Pokemon's attack knocks the user out, that Pokemon faints as well, unless the attack was Doom Desire or Future Sight.
- **드래곤테일** (Dragon Tail, `dragontail`, Physical)
  - This effect fails if the target used Ingrain previously, has the Suction Cups Ability, or this move hit a substitute.
- **송전** (Electrify, `electrify`, Status)
  - Among effects that can change a move's type, this effect happens last.
- **버티기** (Endure, `endure`, Status)
  - The user will survive attacks made by other Pokemon during this turn with at least 1 HP.
- **객기** (Facade, `facade`, Physical)
  - The physical damage halving effect from the user's burn is ignored.
- **가드셰어** (Guard Split, `guardsplit`, Status)
  - The user and the target have their Defense and Special Defense stats set to be equal to the average of the user and the target's Defense and Special Defense stats, respectively, rounded down.
- **가드스왑** (Guard Swap, `guardswap`, Status)
  - The user swaps its Defense and Special Defense stat stage changes with the target.
- **자이로볼** (Gyro Ball, `gyroball`, Physical)
  - Power is equal to (25 * target's current Speed / user's current Speed) + 1, rounded down, but not more than 150.
- **치유소원** (Healing Wish, `healingwish`, Status)
  - The user faints, and if the Pokemon brought out to replace it does not have full HP or has a non-volatile status condition, its HP is fully restored along with having any non-volatile status condition cured.
  - The replacement is sent out at the end of the turn, and the healing happens before hazards take effect.
  - This effect continues until a Pokemon that meets either of these conditions switches in at the user's position or gets swapped into the position with Ally Switch.
- **치유파동** (Heal Pulse, `healpulse`, Status)
  - If the user has the Mega Launcher Ability, the target instead restores 3/4 of its maximum HP, rounded half down.
- **봉인** (Imprison, `imprison`, Status)
  - The user prevents all opposing Pokemon from using any moves that the user also knows as long as the user remains active.
- **킹실드** (King's Shield, `kingsshield`, Status)
  - The user is protected from most attacks made by other Pokemon during this turn, and Pokemon trying to make contact with the user have their Attack lowered by 1 stage.
- **탁쳐서떨구기** (Knock Off, `knockoff`, Physical)
  - This move's power is multiplied by 1.5 if the target is holding an item, and the target loses its held item if the user has not fainted.
  - This move does not increase in power or remove the target's item if it is a Blue Orb, Red Orb, Adamant Crystal, Lustrous Globe, Griseous Core, Plate, Drive, Memory, Rusted Sword, Rusted Shield, Booster Energy, or Mask held by a Kyogre, Groudon, Dialga, Palkia, Giratina, Arceus, Genesect, Silvally, Zacian, Zamazenta, Paradox Pokemon, or Ogerpon, respectively, or if the user is one of those species and the target is holding the respective item.
- **록온** (Lock-On, `lockon`, Status)
  - Until the end of the next turn, the target cannot avoid the user's moves, even if the target is in the middle of a two-turn move.
- **마법가루** (Magic Powder, `magicpowder`, Status)
  - Fails if the target is an Arceus or a Silvally, if the target is already purely Psychic type, or if the target is Terastallized.
- **매직룸** (Magic Room, `magicroom`, Status)
  - An item's effect of causing forme changes is unaffected, but any other effects from such items are negated.
- **아픔나누기** (Pain Split, `painsplit`, Status)
  - The user and the target's HP become the average of their current HP, rounded down, but not more than the maximum HP of either one.
- **보복** (Payback, `payback`, Physical)
  - Switching in does not count as an action.
- **쪼아대기** (Pluck, `pluck`, Physical)
  - If this move is successful and the user has not fainted, it steals the target's held Berry if it is holding one and eats it immediately, gaining its effects even if the user's item is being ignored.
- **파워셰어** (Power Split, `powersplit`, Status)
  - The user and the target have their Attack and Special Attack stats set to be equal to the average of the user and the target's Attack and Special Attack stats, respectively, rounded down.
- **파워스왑** (Power Swap, `powerswap`, Status)
  - The user swaps its Attack and Special Attack stat stage changes with the target.
- **파워트릭** (Power Trick, `powertrick`, Status)
  - If the user has its stats recalculated by changing forme while its stats are swapped, this effect is ignored but is still active for the purposes of Baton Pass.
- **자기암시** (Psych Up, `psychup`, Status)
  - The user copies all of the target's current stat stage changes.
- **분노의주먹** (Rage Fist, `ragefist`, Physical)
  - Power is equal to 50+(X*50), where X is the total number of times the user has been hit by a damaging attack during the battle, even if the user did not lose HP from the attack.
  - X cannot be greater than 6 and does not reset upon switching out or fainting.
- **리사이클** (Recycle, `recycle`, Status)
  - The user regains the item it last used.
  - Items thrown with Fling can be regained.
- **미러타입** (Reflect Type, `reflecttype`, Status)
  - If the target's current types include typeless and a non-added type, typeless is ignored.
  - If the target's current types include typeless and an added type from Forest's Curse or Trick-or-Treat, typeless is copied as the Normal type instead.
  - Fails if the user is an Arceus or a Silvally, if the user is Terastallized, or if the target's current type is typeless alone.
- **잠자기** (Rest, `rest`, Status)
  - The user falls asleep for the next two turns and restores all of its HP, curing itself of any non-volatile status condition in the process.
- **울부짖기** (Roar, `roar`, Status)
  - The target is forced to switch out and be replaced with a random unfainted ally.
- **역할** (Role Play, `roleplay`, Status)
  - The user's Ability changes to match the target's Ability.
- **꼬리자르기** (Shed Tail, `shedtail`, Status)
  - The user takes 1/2 of its maximum HP, rounded up, and creates a substitute that has 1/4 of the user's maximum HP, rounded down.
  - The user is replaced with another Pokemon in its party and the selected Pokemon has the substitute transferred to it.
- **스킬스왑** (Skill Swap, `skillswap`, Status)
  - The user swaps its Ability with the target's Ability.
  - Fails if either the user or the target's Ability is As One, Battle Bond, Comatose, Commander, Disguise, Embody Aspect, Hunger Switch, Ice Face, Illusion, Multitype, Neutralizing Gas, Poison Puppeteer, Power Construct, Protosynthesis, Quark Drive, RKS System, Schooling, Shields Down, Stance Change, Tera Shell, Tera Shift, Teraform Zero, Wonder Guard, Zen Mode, or Zero to Hero.
- **솔라빔** (Solar Beam, `solarbeam`, Special)
  - If the user is holding Utility Umbrella and the weather is Desolate Land or Sunny Day, the move still requires a turn to charge.
- **솔라블레이드** (Solar Blade, `solarblade`, Physical)
  - If the user is holding Utility Umbrella and the weather is Desolate Land or Sunny Day, the move still requires a turn to charge.
- **스피드스왑** (Speed Swap, `speedswap`, Status)
  - The user swaps its Speed stat with the target.
- **니들가드** (Spiky Shield, `spikyshield`, Status)
  - The user is protected from most attacks made by other Pokemon during this turn, and Pokemon making contact with the user lose 1/8 of their maximum HP, rounded down.
- **토해내기** (Spit Up, `spitup`, Special)
  - Whether or not this move is successful, the user's Defense and Special Defense decrease by as many stages as Stockpile had increased them, and the user's Stockpile count resets to 0.
- **볼가득넣기** (Stuff Cheeks, `stuffcheeks`, Status)
  - This move cannot be selected unless the user is holding a Berry.
  - The user eats its Berry and raises its Defense by 2 stages.
  - This effect is not prevented by the Klutz or Unnerve Abilities, or the effects of Embargo or Magic Room.
- **대타출동** (Substitute, `substitute`, Status)
  - The user takes 1/4 of its maximum HP, rounded down, and puts it into a substitute to take its place in battle.
  - The substitute is removed once enough damage is inflicted on it, if the user switches out or faints, or if any Pokemon uses Tidy Up.
  - Until the substitute is broken, it receives damage from all attacks made by other Pokemon and shields the user from status effects and stat stage changes caused by other Pokemon.
- **꿀꺽** (Swallow, `swallow`, Status)
  - The user restores its HP based on its Stockpile count.
  - The user's Defense and Special Defense decrease by as many stages as Stockpile had increased them, and the user's Stockpile count resets to 0.
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
- **날려버리기** (Whirlwind, `whirlwind`, Status)
  - The target is forced to switch out and be replaced with a random unfainted ally.

### 선언된 랭크 변화 (boosts 필드) -- 10개

- **경혈찌르기** (Acupressure, `acupressure`, Status)
  - Raises a random stat by 2 stages as long as the stat is not already at stage 6.
- **배북** (Belly Drum, `bellydrum`, Status)
  - Raises the user's Attack by 12 stages in exchange for the user losing 1/2 of its maximum HP, rounded down.
- **명상** (Calm Mind, `calmmind`, Status)
  - Raises the user's Special Attack and Special Defense by 1 stage.
- **용의춤** (Dragon Dance, `dragondance`, Status)
  - Raises the user's Attack and Speed by 1 stage.
- **마지막일침** (Fell Stinger, `fellstinger`, Physical)
  - Raises the user's Attack by 3 stages if this move knocks out the target.
- **철벽** (Iron Defense, `irondefense`, Status)
  - Raises the user's Defense by 2 stages.
- **자기장조작** (Magnetic Flux, `magneticflux`, Status)
  - Raises the Defense and Special Defense of Pokemon on the user's side with the Plus or Minus Abilities by 1 stage.
- **배수의진** (No Retreat, `noretreat`, Status)
  - Raises the user's Attack, Defense, Special Attack, Special Defense, and Speed by 1 stage, but it becomes prevented from switching out.
- **껍질깨기** (Shell Smash, `shellsmash`, Status)
  - Lowers the user's Defense and Special Defense by 1 stage.
  - Raises the user's Attack, Special Attack, and Speed by 2 stages.
- **정리정돈** (Tidy Up, `tidyup`, Status)
  - Raises the user's Attack and Speed by 1 stage.

### 선언된 랭크 변화 (boosts 필드), 설명이 기술 그 자체 (mechanic_check의 수기 검사에 통째로 위임) -- 5개

- **안개제거** (Defog, `defog`, Status)
  - Lowers the target's evasiveness by 1 stage.
  - If this move is successful and whether or not the target's evasiveness was affected, the effects of Reflect, Light Screen, Aurora Veil, Safeguard, Mist, Spikes, Toxic Spikes, Stealth Rock, and Sticky Web end for the target's side, and the effects of Spikes, Toxic Spikes, Stealth Rock, and Sticky Web end for the user's side.
  - Ignores a target's substitute, although a substitute will still block the lowering of evasiveness.
- **작아지기** (Minimize, `minimize`, Status)
  - Raises the user's evasiveness by 2 stages.
  - Whether or not the user's evasiveness was changed, Body Slam, Dragon Rush, Flying Press, Heat Crash, Heavy Slam, Malicious Moonsault, Steamroller, Stomp, and Supercell Slam will not check accuracy and have their damage doubled if used against the user while it is active.
- **막말내뱉기** (Parting Shot, `partingshot`, Status)
  - Lowers the target's Attack and Special Attack by 1 stage.
  - The user does not switch out if the target's Attack and Special Attack stat stages were both unchanged, or if there are no unfainted party members.
- **비축하기** (Stockpile, `stockpile`, Status)
  - Raises the user's Defense and Special Defense by 1 stage.
  - The user's Stockpile count increases by 1.
  - The user's Stockpile count is reset to 0 when it is no longer active.
- **힘흡수** (Strength Sap, `strengthsap`, Status)
  - Lowers the target's Attack by 1 stage.
  - The user restores its HP equal to the target's Attack stat calculated with its stat stage before this move was used.

### 추가효과 확률 (데이터의 secondary 필드) -- 5개

- **속이기** (Fake Out, `fakeout`, Physical)
  - Has a 100% chance to make the target flinch.
- **G의힘** (Grav Apple, `gravapple`, Physical)
  - Has a 100% chance to lower the target's Defense by 1 stage.
- **백귀야행** (Infernal Parade, `infernalparade`, Special)
  - Has a 30% chance to burn the target.
- **킬러스핀** (Mortal Spin, `mortalspin`, Physical)
  - Has a 100% chance to poison the target.
- **고속스핀** (Rapid Spin, `rapidspin`, Physical)
  - Has a 100% chance to raise the user's Speed by 1 stage.

### 위력 계산 -- 4개

- **승부굳히기** (Assurance, `assurance`, Physical)
  - Power doubles if the target has already taken damage this turn, other than direct damage from Belly Drum, confusion, Curse, or Pain Split.
- **눈사태** (Avalanche, `avalanche`, Physical)
  - Power doubles if the user was hit by the target this turn.
- **성묘** (Last Respects, `lastrespects`, Physical)
  - Power is equal to 50+(X*50), where X is the total number of times any Pokemon has fainted on the user's side, and X cannot be greater than 100.
- **대지의파동** (Terrain Pulse, `terrainpulse`, Special)
  - Power doubles if the user is grounded and a terrain is active, and this move's type changes to match.

### 조건부 1.5배 -- 2개

- **앙갚음** (Comeuppance, `comeuppance`, Physical)
  - Deals damage to the last opposing Pokemon to hit the user with a physical or special attack this turn equal to 1.5 times the HP lost by the user from that attack, rounded down.
- **메탈버스트** (Metal Burst, `metalburst`, Physical)
  - Deals damage to the last opposing Pokemon to hit the user with a physical or special attack this turn equal to 1.5 times the HP lost by the user from that attack, rounded down.

### 조건부 2배 -- 2개

- **카운터** (Counter, `counter`, Physical)
  - Deals damage to the last opposing Pokemon to hit the user with a physical attack this turn equal to twice the HP lost by the user from that attack.
- **미러코트** (Mirror Coat, `mirrorcoat`, Special)
  - Deals damage to the last opposing Pokemon to hit the user with a special attack this turn equal to twice the HP lost by the user from that attack.

### 설명이 기술 그 자체 (mechanic_check의 수기 검사에 통째로 위임), 벽 파괴, 지속 턴 수 -- 1개

- **오로라베일** (Aurora Veil, `auroraveil`, Status)
  - For 5 turns, the user and its party members take 0.5x damage from physical and special attacks, or 0.66x damage if in a Double Battle; does not reduce damage further with Reflect or Light Screen.
  - Critical hits ignore this protection.
  - It is removed from the user's side if the user or an ally is successfully hit by Brick Break, Psychic Fangs, or Defog.

### 설명이 기술 그 자체 (mechanic_check의 수기 검사에 통째로 위임), 지속 턴 수 -- 1개

- **사이코노이즈** (Psychic Noise, `psychicnoise`, Special)
  - For 2 turns, the target is prevented from restoring any HP as long as it remains active.
  - Pain Split and the Regenerator Ability are unaffected.

### 벽 파괴 -- 1개

- **레이징불** (Raging Bull, `ragingbull`, Physical)
  - If this attack does not miss, the effects of Reflect, Light Screen, and Aurora Veil end for the target's side of the field before damage is calculated.

### 선언된 상태이상 (status 필드) -- 1개

- **맹독** (Toxic, `toxic`, Status)
  - Badly poisons the target.

### 지속 턴 수 -- 1개

- **트릭룸** (Trick Room, `trickroom`, Status)
  - For 5 turns, the Speed of every Pokemon is recalculated for the purposes of determining turn order.

### 위력 계산, 설명이 기술 그 자체 (mechanic_check의 수기 검사에 통째로 위임) -- 1개

- **웨더볼** (Weather Ball, `weatherball`, Special)
  - Power doubles if a weather condition other than Delta Stream is active, and this move's type changes to match.
  - If the user is holding Utility Umbrella and uses Weather Ball during Primordial Sea, Rain Dance, Desolate Land, or Sunny Day, this move remains Normal type and does not double in power.

## C등급 -- 191개

기술 데이터의 필드(secondary/boosts/status)가 실제 배틀에 나타나는지 자동 검사합니다. 문장이 필드보다 더 말하는 부분은 측정되지 않습니다.

### 추가효과 확률 (데이터의 secondary 필드) -- 106개

- **애시드봄** (Acid Spray, `acidspray`, Special)
  - Has a 100% chance to lower the target's Special Defense by 2 stages.
- **에어슬래시** (Air Slash, `airslash`, Special)
  - Has a 30% chance to make the target flinch.
- **매혹의보이스** (Alluring Voice, `alluringvoice`, Special)
  - Has a 100% chance to confuse the target if it had a stat stage raised this turn.
- **원시의힘** (Ancient Power, `ancientpower`, Special)
  - Has a 10% chance to raise the user's Attack, Defense, Special Attack, Special Defense, and Speed by 1 stage.
- **사과산** (Apple Acid, `appleacid`, Special)
  - Has a 100% chance to lower the target's Special Defense by 1 stage.
- **아쿠아스텝** (Aqua Step, `aquastep`, Physical)
  - Has a 100% chance to raise the user's Speed by 1 stage.
- **오라휠** (Aura Wheel, `aurawheel`, Physical)
  - Has a 100% chance to raise the user's Speed by 1 stage.
- **발꿈치찍기** (Axe Kick, `axekick`, Physical)
  - Has a 30% chance to confuse the target.
- **독침천발** (Barb Barrage, `barbbarrage`, Physical)
  - Has a 50% chance to poison the target.
- **물기** (Bite, `bite`, Physical)
  - Has a 30% chance to make the target flinch.
- **천추지한** (Bitter Malice, `bittermalice`, Special)
  - Has a 100% chance to lower the target's Attack by 1 stage.
- **블레이즈킥** (Blaze Kick, `blazekick`, Physical)
  - Has a 10% chance to burn the target and a higher chance for a critical hit.
- **눈보라** (Blizzard, `blizzard`, Special)
  - Has a 10% chance to freeze the target.
- **누르기** (Body Slam, `bodyslam`, Physical)
  - Has a 30% chance to paralyze the target.
- **뛰어오르기** (Bounce, `bounce`, Physical)
  - Has a 30% chance to paralyze the target.
- **와이드브레이커** (Breaking Swipe, `breakingswipe`, Physical)
  - Has a 100% chance to lower the target's Attack by 1 stage.
- **벌레의야단법석** (Bug Buzz, `bugbuzz`, Special)
  - Has a 10% chance to lower the target's Special Defense by 1 stage.
- **땅고르기** (Bulldoze, `bulldoze`, Physical)
  - Has a 100% chance to lower the target's Speed by 1 stage.
- **질투의불꽃** (Burning Jealousy, `burningjealousy`, Special)
  - Has a 100% chance to burn the target if it had a stat stage raised this turn.
- **차지빔** (Charge Beam, `chargebeam`, Special)
  - Has a 70% chance to raise the user's Special Attack by 1 stage.
- **찬물끼얹기** (Chilling Water, `chillingwater`, Special)
  - Has a 100% chance to lower the target's Attack by 1 stage.
- **크로스포이즌** (Cross Poison, `crosspoison`, Physical)
  - Has a 10% chance to poison the target and a higher chance for a critical hit.
- **깨물어부수기** (Crunch, `crunch`, Physical)
  - Has a 20% chance to lower the target's Defense by 1 stage.
- **브레이크클로** (Crush Claw, `crushclaw`, Physical)
  - Has a 50% chance to lower the target's Defense by 1 stage.
- **악의파동** (Dark Pulse, `darkpulse`, Special)
  - Has a 20% chance to make the target flinch.
- **페이탈클로** (Dire Claw, `direclaw`, Physical)
  - Has a 50% chance to cause the target to either fall asleep, become poisoned, or become paralyzed.
- **방전** (Discharge, `discharge`, Special)
  - Has a 30% chance to paralyze the target.
- **드래곤다이브** (Dragon Rush, `dragonrush`, Physical)
  - Has a 20% chance to make the target flinch.
- **폭발펀치** (Dynamic Punch, `dynamicpunch`, Physical)
  - Has a 100% chance to confuse the target.
- **대지의힘** (Earth Power, `earthpower`, Special)
  - Has a 10% chance to lower the target's Special Defense by 1 stage.
- **일렉트릭네트** (Electroweb, `electroweb`, Special)
  - Has a 100% chance to lower the target's Speed by 1 stage.
- **에너지볼** (Energy Ball, `energyball`, Special)
  - Has a 10% chance to lower the target's Special Defense by 1 stage.
- **신통력** (Extrasensory, `extrasensory`, Special)
  - Has a 10% chance to make the target flinch.
- **변덕레이저** (Fickle Beam, `ficklebeam`, Special)
  - Has a 30% chance this move's power is doubled.
- **불꽃춤** (Fiery Dance, `fierydance`, Special)
  - Has a 50% chance to raise the user's Special Attack by 1 stage.
- **불대문자** (Fire Blast, `fireblast`, Special)
  - Has a 10% chance to burn the target.
- **불꽃엄니** (Fire Fang, `firefang`, Physical)
  - Has a 10% chance to burn the target and a 10% chance to make it flinch.
- **불꽃채찍** (Fire Lash, `firelash`, Physical)
  - Has a 100% chance to lower the target's Defense by 1 stage.
- **불꽃펀치** (Fire Punch, `firepunch`, Physical)
  - Has a 10% chance to burn the target.
- **니트로차지** (Flame Charge, `flamecharge`, Physical)
  - Has a 100% chance to raise the user's Speed by 1 stage.
- **화염방사** (Flamethrower, `flamethrower`, Special)
  - Has a 10% chance to burn the target.
- **플레어드라이브** (Flare Blitz, `flareblitz`, Physical)
  - Has a 10% chance to burn the target.
- **러스터캐논** (Flash Cannon, `flashcannon`, Special)
  - Has a 10% chance to lower the target's Special Defense by 1 stage.
- **기합구슬** (Focus Blast, `focusblast`, Special)
  - Has a 10% chance to lower the target's Special Defense by 1 stage.
- **더스트슈트** (Gunk Shot, `gunkshot`, Physical)
  - Has a 30% chance to poison the target.
- **열풍** (Heat Wave, `heatwave`, Special)
  - Has a 10% chance to burn the target.
- **폭풍** (Hurricane, `hurricane`, Special)
  - Has a 30% chance to confuse the target.
- **냉동빔** (Ice Beam, `icebeam`, Special)
  - Has a 10% chance to freeze the target.
- **얼음엄니** (Ice Fang, `icefang`, Physical)
  - Has a 10% chance to freeze the target and a 10% chance to make it flinch.
- **냉동펀치** (Ice Punch, `icepunch`, Physical)
  - Has a 10% chance to freeze the target.
- **고드름떨구기** (Icicle Crash, `iciclecrash`, Physical)
  - Has a 30% chance to make the target flinch.
- **얼어붙은바람** (Icy Wind, `icywind`, Special)
  - Has a 100% chance to lower the target's Speed by 1 stage.
- **연옥** (Inferno, `inferno`, Special)
  - Has a 100% chance to burn the target.
- **아이언헤드** (Iron Head, `ironhead`, Physical)
  - Has a 30% chance to make the target flinch.
- **아이언테일** (Iron Tail, `irontail`, Physical)
  - Has a 30% chance to lower the target's Defense by 1 stage.
- **분연** (Lava Plume, `lavaplume`, Special)
  - Has a 30% chance to burn the target.
- **아쿠아브레이크** (Liquidation, `liquidation`, Physical)
  - Has a 20% chance to lower the target's Defense by 1 stage.
- **로킥** (Low Sweep, `lowsweep`, Physical)
  - Has a 100% chance to lower the target's Speed by 1 stage.
- **루미나콜리전** (Lumina Crash, `luminacrash`, Special)
  - Has a 100% chance to lower the target's Special Defense by 2 stages.
- **덤벼들기** (Lunge, `lunge`, Physical)
  - Has a 100% chance to lower the target's Attack by 1 stage.
- **휘적휘적포** (Matcha Gotcha, `matchagotcha`, Special)
  - Has a 20% chance to burn the target.
- **코멧펀치** (Meteor Mash, `meteormash`, Physical)
  - Has a 20% chance to raise the user's Attack by 1 stage.
- **문포스** (Moonblast, `moonblast`, Special)
  - Has a 30% chance to lower the target's Special Attack by 1 stage.
- **빙산바람** (Mountain Gale, `mountaingale`, Physical)
  - Has a 30% chance to make the target flinch.
- **탁류** (Muddy Water, `muddywater`, Special)
  - Has a 30% chance to lower the target's accuracy by 1 stage.
- **머드샷** (Mud Shot, `mudshot`, Special)
  - Has a 100% chance to lower the target's Speed by 1 stage.
- **진흙뿌리기** (Mud-Slap, `mudslap`, Special)
  - Has a 100% chance to lower the target's accuracy by 1 stage.
- **매지컬플레임** (Mystical Fire, `mysticalfire`, Special)
  - Has a 100% chance to lower the target's Special Attack by 1 stage.
- **나이트버스트** (Night Daze, `nightdaze`, Special)
  - Has a 40% chance to lower the target's accuracy by 1 stage.
- **볼부비부비** (Nuzzle, `nuzzle`, Physical)
  - Has a 100% chance to paralyze the target.
- **치근거리기** (Play Rough, `playrough`, Physical)
  - Has a 10% chance to lower the target's Attack by 1 stage.
- **맹독엄니** (Poison Fang, `poisonfang`, Physical)
  - Has a 50% chance to badly poison the target.
- **독찌르기** (Poison Jab, `poisonjab`, Physical)
  - Has a 30% chance to poison the target.
- **달려들기** (Pounce, `pounce`, Physical)
  - Has a 100% chance to lower the target's Speed by 1 stage.
- **사이코키네시스** (Psychic, `psychic`, Special)
  - Has a 10% chance to lower the target's Special Defense by 1 stage.
- **배리어러시** (Psyshield Bash, `psyshieldbash`, Physical)
  - Has a 100% chance to raise the user's Defense by 1 stage.
- **셸블레이드** (Razor Shell, `razorshell`, Physical)
  - Has a 50% chance to lower the target's Defense by 1 stage.
- **스톤샤워** (Rock Slide, `rockslide`, Physical)
  - Has a 30% chance to make the target flinch.
- **암석봉인** (Rock Tomb, `rocktomb`, Physical)
  - Has a 100% chance to lower the target's Speed by 1 stage.
- **열탕** (Scald, `scald`, Special)
  - Has a 30% chance to burn the target.
- **열사의대지** (Scorching Sands, `scorchingsands`, Special)
  - Has a 30% chance to burn the target.
- **섀도볼** (Shadow Ball, `shadowball`, Special)
  - Has a 20% chance to lower the target's Special Defense by 1 stage.
- **엄습하는일격** (Skitter Smack, `skittersmack`, Physical)
  - Has a 100% chance to lower the target's Special Attack by 1 stage.
- **불새** (Sky Attack, `skyattack`, Physical)
  - Has a 30% chance to make the target flinch and a higher chance for a critical hit.
- **오물폭탄** (Sludge Bomb, `sludgebomb`, Special)
  - Has a 30% chance to poison the target.
- **오물웨이브** (Sludge Wave, `sludgewave`, Special)
  - Has a 10% chance to poison the target.
- **바크아웃** (Snarl, `snarl`, Special)
  - Has a 100% chance to lower the target's Special Attack by 1 stage.
- **코골기** (Snore, `snore`, Special)
  - Has a 30% chance to make the target flinch.
- **소울크래시** (Spirit Break, `spiritbreak`, Physical)
  - Has a 100% chance to lower the target's Special Attack by 1 stage.
- **강철날개** (Steel Wing, `steelwing`, Physical)
  - Has a 10% chance to raise the user's Defense by 1 stage.
- **벌레의저항** (Struggle Bug, `strugglebug`, Special)
  - Has a 100% chance to lower the target's Special Attack by 1 stage.
- **번개** (Thunder, `thunder`, Special)
  - Has a 30% chance to paralyze the target.
- **10만볼트** (Thunderbolt, `thunderbolt`, Special)
  - Has a 10% chance to paralyze the target.
- **번개엄니** (Thunder Fang, `thunderfang`, Physical)
  - Has a 10% chance to paralyze the target and a 10% chance to make it flinch.
- **번개펀치** (Thunder Punch, `thunderpunch`, Physical)
  - Has a 10% chance to paralyze the target.
- **플레어송** (Torch Song, `torchsong`, Special)
  - Has a 100% chance to raise the user's Special Attack by 1 stage.
- **개척하기** (Trailblaze, `trailblaze`, Physical)
  - Has a 100% chance to raise the user's Speed by 1 stage.
- **트라이어택** (Tri Attack, `triattack`, Special)
  - Has a 20% chance to either burn, freeze, or paralyze the target.
- **3연화살** (Triple Arrows, `triplearrows`, Physical)
  - Has a 50% chance to lower the target's Defense by 1 stage, a 30% chance to make it flinch, and a higher chance for a critical hit.
- **트로피컬킥** (Trop Kick, `tropkick`, Physical)
  - Has a 100% chance to lower the target's Attack by 1 stage.
- **기선제압** (Upper Hand, `upperhand`, Physical)
  - Has a 100% chance to make the target flinch.
- **볼트태클** (Volt Tackle, `volttackle`, Physical)
  - Has a 10% chance to paralyze the target.
- **폭포오르기** (Waterfall, `waterfall`, Physical)
  - Has a 20% chance to make the target flinch.
- **물의파동** (Water Pulse, `waterpulse`, Special)
  - Has a 20% chance to confuse the target.
- **전자포** (Zap Cannon, `zapcannon`, Special)
  - Has a 100% chance to paralyze the target.
- **사념의박치기** (Zen Headbutt, `zenheadbutt`, Physical)
  - Has a 20% chance to make the target flinch.

### 선언된 랭크 변화 (boosts 필드) -- 46개

- **녹기** (Acid Armor, `acidarmor`, Status)
  - Raises the user's Defense by 2 stages.
- **고속이동** (Agility, `agility`, Status)
  - Raises the user's Speed by 2 stages.
- **망각술** (Amnesia, `amnesia`, Status)
  - Raises the user's Special Defense by 2 stages.
- **아머캐논** (Armor Cannon, `armorcannon`, Special)
  - Lowers the user's Defense and Special Defense by 1 stage.
- **초롱초롱눈동자** (Baby-Doll Eyes, `babydolleyes`, Status)
  - Lowers the target's Attack by 1 stage.
- **벌크업** (Bulk Up, `bulkup`, Status)
  - Raises the user's Attack and Defense by 1 stage.
- **애교부리기** (Charm, `charm`, Status)
  - Lowers the target's Attack by 2 stages.
- **스케일노이즈** (Clanging Scales, `clangingscales`, Special)
  - Lowers the user's Defense by 1 stage.
- **인파이트** (Close Combat, `closecombat`, Physical)
  - Lowers the user's Defense and Special Defense by 1 stage.
- **똬리틀기** (Coil, `coil`, Status)
  - Raises the user's Attack, Defense, and accuracy by 1 stage.
- **코스믹파워** (Cosmic Power, `cosmicpower`, Status)
  - Raises the user's Defense and Special Defense by 1 stage.
- **코튼가드** (Cotton Guard, `cottonguard`, Status)
  - Raises the user's Defense by 3 stages.
- **목화포자** (Cotton Spore, `cottonspore`, Status)
  - Lowers the target's Speed by 2 stages.
- **데코레이션** (Decorate, `decorate`, Status)
  - Raises the target's Attack and Special Attack by 2 stages.
- **그림자분신** (Double Team, `doubleteam`, Status)
  - Raises the user's evasiveness by 1 stage.
- **용성군** (Draco Meteor, `dracometeor`, Special)
  - Lowers the user's Special Attack by 2 stages.
- **괴전파** (Eerie Impulse, `eerieimpulse`, Status)
  - Lowers the target's Special Attack by 2 stages.
- **거짓울음** (Fake Tears, `faketears`, Status)
  - Lowers the target's Special Defense by 2 stages.
- **깃털댄스** (Feather Dance, `featherdance`, Status)
  - Lowers the target's Attack by 2 stages.
- **부추기기** (Flatter, `flatter`, Status)
  - Raises the target's Special Attack by 1 stage and confuses it.
- **암해머** (Hammer Arm, `hammerarm`, Physical)
  - Lowers the user's Speed by 1 stage.
- **들이받기** (Headlong Rush, `headlongrush`, Physical)
  - Lowers the user's Defense and Special Defense by 1 stage.
- **멀리짖기** (Howl, `howl`, Status)
  - Raises the Attack of the user and all allies 1 stage.
- **아이스해머** (Ice Hammer, `icehammer`, Physical)
  - Lowers the user's Speed by 1 stage.
- **리프스톰** (Leaf Storm, `leafstorm`, Special)
  - Lowers the user's Special Attack by 2 stages.
- **골드러시** (Make It Rain, `makeitrain`, Special)
  - Lowers the user's Special Attack by 1 stage.
- **금속음** (Metal Sound, `metalsound`, Status)
  - Lowers the target's Special Defense by 2 stages.
- **메테오빔** (Meteor Beam, `meteorbeam`, Special)
  - Raises the user's Special Attack by 1 stage on the first turn.
- **나쁜음모** (Nasty Plot, `nastyplot`, Status)
  - Raises the user's Special Attack by 2 stages.
- **부르짖기** (Noble Roar, `nobleroar`, Status)
  - Lowers the target's Attack and Special Attack by 1 stage.
- **오버히트** (Overheat, `overheat`, Special)
  - Lowers the user's Special Attack by 2 stages.
- **나비춤** (Quiver Dance, `quiverdance`, Status)
  - Raises the user's Special Attack, Special Defense, and Speed by 1 stage.
- **록커트** (Rock Polish, `rockpolish`, Status)
  - Raises the user's Speed by 2 stages.
- **스케일샷** (Scale Shot, `scaleshot`, Physical)
  - Lowers the user's Defense by 1 stage and raises the user's Speed by 1 stage after the last hit.
- **겁나는얼굴** (Scary Face, `scaryface`, Status)
  - Lowers the target's Speed by 2 stages.
- **싫은소리** (Screech, `screech`, Status)
  - Lowers the target's Defense by 2 stages.
- **농성** (Shelter, `shelter`, Status)
  - Raises the user's Defense by 2 stages.
- **하바네로엑기스** (Spicy Extract, `spicyextract`, Status)
  - Raises the target's Attack by 2 stages and lowers its Defense by 2 stages.
- **실뿜기** (String Shot, `stringshot`, Status)
  - Lowers the target's Speed by 2 stages.
- **엄청난힘** (Superpower, `superpower`, Physical)
  - Lowers the user's Attack and Defense by 1 stage.
- **뽐내기** (Swagger, `swagger`, Status)
  - Raises the target's Attack by 2 stages and confuses it.
- **달콤한향기** (Sweet Scent, `sweetscent`, Status)
  - Lowers the target's evasiveness by 2 stages.
- **칼춤** (Swords Dance, `swordsdance`, Status)
  - Raises the user's Attack by 2 stages.
- **눈물그렁그렁** (Tearful Look, `tearfullook`, Status)
  - Lowers the target's Attack and Special Attack by 1 stage.
- **간지르기** (Tickle, `tickle`, Status)
  - Lowers the target's Attack and Defense by 1 stage.
- **독실** (Toxic Thread, `toxicthread`, Status)
  - Lowers the target's Speed by 1 stage and poisons it.

### 설명이 기술 그 자체 (mechanic_check의 수기 검사에 통째로 위임) -- 10개

- **썰렁개그** (Chilly Reception, `chillyreception`, Status)
  - The user switches out even if it is trapped and is replaced immediately by a selected party member.
- **배대뒤치기** (Circle Throw, `circlethrow`, Physical)
  - This effect fails if the target is under the effect of Ingrain, has the Suction Cups Ability, or this move hit a substitute.
- **플라잉프레스** (Flying Press, `flyingpress`, Physical)
  - This move combines Flying in its type effectiveness against the target.
- **힘껏펀치** (Focus Punch, `focuspunch`, Physical)
  - The user loses its focus and does nothing if it is hit by a damaging attack this turn before it can execute the move.
- **속임수** (Foul Play, `foulplay`, Physical)
  - The user's Ability, item, and burn are used as normal.
- **미래예지** (Future Sight, `futuresight`, Special)
  - If the user is no longer active at the time, damage is calculated based on the user's natural Special Attack stat, types, and level, with no boosts from its held item or Ability.
  - Fails if this move or Doom Desire is already in effect for the target's position.
- **뿌리박기** (Ingrain, `ingrain`, Status)
  - The user has 1/16 of its maximum HP restored at the end of each turn, but it is prevented from switching out and other Pokemon cannot force the user to switch out.
- **비장의무기** (Last Resort, `lastresort`, Physical)
  - This move fails unless the user knows this move and at least one other move, and has used all the other moves it knows at least once each since it became active or Transformed.
- **철제광선** (Steel Beam, `steelbeam`, Special)
  - Whether or not this move is successful and even if it would cause fainting, the user loses 1/2 of its maximum HP, rounded up, unless the user has the Magic Guard Ability.
- **발버둥** (Struggle, `struggle`, Physical)
  - This move is automatically used if none of the user's known moves can be selected.

### 선언된 상태이상 (status 필드) -- 10개

- **이상한빛** (Confuse Ray, `confuseray`, Status)
  - Causes the target to become confused.
- **뱀눈초리** (Glare, `glare`, Status)
  - Paralyzes the target.
- **최면술** (Hypnosis, `hypnosis`, Status)
  - Causes the target to fall asleep.
- **독가루** (Poison Powder, `poisonpowder`, Status)
  - Poisons the target.
- **노래하기** (Sing, `sing`, Status)
  - Causes the target to fall asleep.
- **수면가루** (Sleep Powder, `sleeppowder`, Status)
  - Causes the target to fall asleep.
- **저리가루** (Stun Spore, `stunspore`, Status)
  - Paralyzes the target.
- **천사의키스** (Sweet Kiss, `sweetkiss`, Status)
  - Causes the target to become confused.
- **흔들흔들댄스** (Teeter Dance, `teeterdance`, Status)
  - Causes the target to become confused.
- **도깨비불** (Will-O-Wisp, `willowisp`, Status)
  - Burns the target.

### 지속 턴 수 -- 7개

- **일렉트릭필드** (Electric Terrain, `electricterrain`, Status)
  - For 5 turns, the terrain becomes Electric Terrain.
- **그래스필드** (Grassy Terrain, `grassyterrain`, Status)
  - For 5 turns, the terrain becomes Grassy Terrain.
- **미스트필드** (Misty Terrain, `mistyterrain`, Status)
  - For 5 turns, the terrain becomes Misty Terrain.
- **사이코필드** (Psychic Terrain, `psychicterrain`, Status)
  - For 5 turns, the terrain becomes Psychic Terrain.
- **비바라기** (Rain Dance, `raindance`, Status)
  - For 5 turns, the weather becomes Rain Dance.
- **모래바람** (Sandstorm, `sandstorm`, Status)
  - For 5 turns, the weather becomes Sandstorm.
- **쾌청** (Sunny Day, `sunnyday`, Status)
  - For 5 turns, the weather becomes Sunny Day.

### 선언된 랭크 변화 (boosts 필드), 설명이 기술 그 자체 (mechanic_check의 수기 검사에 통째로 위임) -- 5개

- **충전** (Charge, `charge`, Status)
  - Raises the user's Special Defense by 1 stage.
  - The user's next Electric-type attack will have its power doubled; the effect ends when the user is no longer active, or after the user attempts to use any Electric-type move besides Charge, even if it is not successful.
- **소울비트** (Clangorous Soul, `clangoroussoul`, Status)
  - Raises the user's Attack, Defense, Special Attack, Special Defense, and Speed by 1 stage in exchange for the user losing 33% of its maximum HP, rounded down.
  - Fails if the user would faint or if its Attack, Defense, Special Attack, Special Defense, and Speed stat stages would not change.
- **일렉트로빔** (Electro Shot, `electroshot`, Special)
  - Raises the user's Special Attack by 1 stage on the first turn.
  - If the user is holding Utility Umbrella and the weather is Primordial Sea or Rain Dance, the move still requires a turn to charge.
- **성장** (Growth, `growth`, Status)
  - Raises the user's Attack and Special Attack by 1 stage.
  - If the user is holding Utility Umbrella, this move will only raise the user's Attack and Special Attack by 1 stage, even if the weather is Sunny Day or Desolate Land.
- **추억의선물** (Memento, `memento`, Status)
  - Lowers the target's Attack and Special Attack by 2 stages.
  - The user faints unless this move misses or there is no target.

### 벽 파괴 -- 2개

- **깨뜨리다** (Brick Break, `brickbreak`, Physical)
  - If this attack does not miss, the effects of Reflect, Light Screen, and Aurora Veil end for the target's side of the field before damage is calculated.
- **사이코팽** (Psychic Fangs, `psychicfangs`, Physical)
  - If this attack does not miss, the effects of Reflect, Light Screen, and Aurora Veil end for the target's side of the field before damage is calculated.

### 설명이 기술 그 자체 (mechanic_check의 수기 검사에 통째로 위임), 추가효과 확률 (데이터의 secondary 필드) -- 2개

- **프리즈드라이** (Freeze-Dry, `freezedry`, Special)
  - Has a 10% chance to freeze the target.
  - This move's type effectiveness against Water is changed to be super effective no matter what this move's type is.
- **셸암즈** (Shell Side Arm, `shellsidearm`, Special)
  - Has a 20% chance to poison the target.
  - This move becomes a physical attack that makes contact if the value of ((((2 * the user's level / 5 + 2) * 90 * X) / Y) / 50), where X is the user's Attack stat and Y is the target's Defense stat, is greater than the same value where X is the user's Special Attack stat and Y is the target's Special Defense stat.
  - No stat modifiers other than stat stage changes are considered for this purpose.

### 벽 파괴, 지속 턴 수 -- 2개

- **빛의장막** (Light Screen, `lightscreen`, Status)
  - For 5 turns, the user and its party members take 0.5x damage from special attacks, or 0.66x damage if in a Double Battle.
  - Damage is not reduced further with Aurora Veil.
  - Critical hits ignore this effect.
- **리플렉터** (Reflect, `reflect`, Status)
  - For 5 turns, the user and its party members take 0.5x damage from physical attacks, or 0.66x damage if in a Double Battle.
  - Damage is not reduced further with Aurora Veil.
  - Critical hits ignore this effect.

### 선언된 상태이상 (status 필드), 설명이 기술 그 자체 (mechanic_check의 수기 검사에 통째로 위임) -- 1개

- **전기자석파** (Thunder Wave, `thunderwave`, Status)
  - Paralyzes the target.
  - This move does not ignore type immunity.

