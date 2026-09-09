# 인수인계 — 2026-09-10, 개인PC 작성

랩실PC 앞으로. **두 대가 같은 탐색을 나눠 도는 게 아니라, 다른 씨앗에서 각자 돌고
결과 장부를 합친다.** 라운드로빈처럼 일정표를 나누는 방식이 아니다.

이전 인수인계(09-08, 기술 전수 검사)는 끝났다. 이 문서가 지금 할 일이다.

---

## 0. 먼저 당겨라

```bash
git pull
```

`.venv`는 옮기지 않는다. 없으면 [RESUME.md](RESUME.md) §1.

---

## 1. 지금 하는 것 — 메가보만다 주축 파티 생성

주축 포켓몬 하나를 고정하고 나머지 다섯 자리를 진화 알고리즘으로 찾는다. 적합도는
**저점**(최악 1/4 매치업의 평균, `party_floor`)이다. hk의 목표가 "6마리 중 3마리를
뽑는 걸 감안해서, 왠만한 파티에 대응 방법이 있는 파티"이기 때문이다. 평균 승률이
아니다.

**랩실PC에서 돌릴 명령 (이 한 줄만 바뀐다: `--seed`, `--log`, `--workers`)**

```bash
python scripts/evolve_party.py --core salamence --regulation m_c --population 24 --generations 8 --budget 3000 --opponents 40 --search-iterations 200 --judge 4 --workers 9 --seed 910000 --log runs/notebook_salamence_lab.jsonl --out runs/evolve_salamence_lab.json
```

개인PC는 `--seed 900000`, `--log runs/notebook_salamence_pc.jsonl`, `--workers 19`로
돈다. 20코어에서 약 3시간, 10코어면 그 두 배로 잡아라.

### 세 플래그만 다르고 나머지는 반드시 같아야 한다

| 플래그 | 왜 |
|---|---|
| `--seed` | **다르게.** 같으면 두 대가 완전히 같은 탐색을 한다. 개체군 초기화부터 돌연변이까지 전부 이 씨앗 하나에서 나온다 |
| `--log` | **다르게.** 각자 자기 파일에 append 한다. 한 파일을 두 대가 쓰면 git에서 줄마다 충돌한다 |
| `--workers` | 기계 코어 수 |
| **그 외 전부** | **같게.** `--opponents`, `--search-iterations`, `--regulation`, `--parties`, `--seed-games`, `--judge-*`, `--rollout-turns` |

마지막 줄이 핵심이다. 장부의 각 줄은 자기가 **어떤 조건에서 측정됐는지**(basis)를
같이 적는다 — `parties_field.json/m_c/singles/opp40/sim200/rollout0/seed700000`.
합칠 때 basis가 같은 줄끼리만 비교한다. 플래그가 하나라도 다르면 두 대의 결과가
영원히 서로 다른 표에 남아서, 두 대를 돌린 의미가 없어진다.

([memory: recipe flags are variables] — 플래그 하나 차이는 전부 실험 변수다.
`--foe-teams` 때 이미 한 번 당했다.)

---

## 2. 중간 결과는 계속 쌓인다

채점되는 모든 후보가 **채점되는 즉시** `--log`에 적힌다. 한 세대가 24개를 채점하고
3개만 데려가지만, 탈락한 21개도 배틀 값을 이미 치렀다. 컴퓨터를 꺼야 하면 그냥
Ctrl+C 하면 된다. 거기까지 잰 건 전부 남는다.

```bash
python scripts/party_notebook.py runs/notebook_salamence_lab.jsonl --top 15
```

**점수가 아니라 신뢰구간 아래쪽으로 정렬된다.** 1라운드 탈락 후보는 생존자의 1/5
게임으로 채점됐고, 운이 나빠서 탈락한 면도 있다. 점수로 줄 세우면 그 소수 표본
대박이 위로 온다. 253파티 라운드로빈에서 +39.6pp로 확인한 승자의 저주가 한 층
아래에서 똑같이 반복되는 것이다. 아래쪽 경계로 정렬하면 **측정이 돼야 순위에 든다.**

`--min-games 200`을 주면 판정 단계를 통과한 줄만 본다. 그게 읽을 만한 숫자다.

---

## 3. 두 대의 장부 합치기

장부는 git에 올라간다(`.gitignore`에 예외를 넣었다). 라운드로빈 진행 파일과 같은
이유다 — 두 기계 사이를 오가는 측정값이다.

```bash
git add runs/notebook_salamence_lab.jsonl && git commit -m "lab: salamence notebook" && git push
```

합쳐 읽는 건 인자를 여러 개 주면 된다. 파일은 건드리지 않는다. 각자 자기 파일에
계속 쓴다.

```bash
python scripts/party_notebook.py runs/notebook_salamence_*.jsonl --top 20 --min-games 200 --show 3
```

같은 파티가 양쪽에 있으면 **게임 수가 많은 쪽**이 남는다. 점수가 높은 쪽이 아니다.
최댓값을 취하면 방금 피한 그 편향이 되돌아온다.

상위 파티를 판정이나 라운드로빈에 넣으려면:

```bash
python scripts/party_notebook.py runs/notebook_salamence_*.jsonl --top 12 --export data/champions/parties_evolved.json
```

M-C에서 합법으로 읽히는 것까지 확인했다.

---

## 4. 끝없이 돌리려면

`--forever`를 붙이면 씨앗을 1씩 올리며 새 탐색을 계속 시작한다. 같은 장부에 쌓인다.
넘어가는 건 개체군이 아니라 기록이다. 밤새 켜둘 때 쓴다.

---

## 5. 이 탐색에서 믿으면 안 되는 숫자

**세대별 floor를 승률로 읽지 마라.** 예산 안에서는 꼬리를 고른 게임과 채점한 게임이
같아서 아래로 편향돼 있다. 모든 후보에게 똑같이 걸린 편향이라 **순위는 살아남고
값은 안 산다.**

**탐색을 하는 에이전트는 중립이 아니다.** 2026-09-09 엔진에서 잰 값으로, 지평선
너머를 보는 롤아웃이 물량 계산 대비 **벽깔이 파티에 +10.9pp, 랭크업 파티에 +9.7pp,
순수 공격 파티에 +0.0pp**다. 즉 싼 에이전트는 hk가 찾으려는 종류의 파티를 체계적으로
과소평가한다. 그래서 2단계다 — **싼 걸로 탐색하고, 살아남은 넷만 비싼 걸로 판정한다.**
읽어야 할 숫자는 `judge` 쪽 basis(`rollout20`)에 있는 줄이다.

---

## 6. 아직 남은 것

- **M-C 엔진에서 필드 라운드로빈 재측정.** 09-09 이전에 잰 승률은 전부 낡았다.
- **나머지 30종의 특성 검증.** 아직 쇼다운 출처이고 게임으로 확인 안 했다.
  ([memory: 챔피언스는 코드만 참조] — 쇼다운은 메커니즘 참조지 규칙 출처가 아니다)
- **`build_regulation_mc.py`를 `fetch_regulation.py m_c`로 교체.** 벌바피디아에
  M-C 페이지가 올라오는 날.
- **ROM 아이템 표를 이름으로 읽기.** `champout/item.json`이 `ITEMNAME_214` 같은
  숫자 라벨로만 키를 잡고 이름 파일이 없다. 그래서 아이템 존재 여부만 아직 쇼다운으로
  떨어진다.
