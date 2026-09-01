# 다음 학습 실행 — 자동 선정 (라운드로빈 v2 기준)

엔진 수정(바디프레스 계열·조건부 위력 17종·메가 후 belief) 이후의
46파티 라운드로빈 v2에서 자동 선정됨. 선정 규칙: **v2 승률 상위 4** —
규칙이 단순한 것은 의도이며, 근거는 아래 표가 전부다.

## v2 순위 (상위 12) — v1 대비 이동

| v2 | 파티 | 승률 [CI] | v1 순위 | 이동 |
|---|---|---|---|---|
| 1 | 39 (pokedb 38) | 62.0% [56.9%, 66.9%] | 3 | +2 |
| 2 | 43 (pokedb 45) | 61.6% [56.5%, 66.5%] | 1 | -1 |
| 3 | 14 ([2200]가람 Mega Gengar and) | 61.5% [56.4%, 66.4%] | 2 | -1 |
| 4 | 42 (pokedb 44) | 61.2% [56.1%, 66.2%] | 13 | +9 |
| 5 | 37 (pokedb 34) | 60.4% [55.3%, 65.4%] | 20 | +15 |
| 6 | 10 ([2403] 엑자몽 중심 10일전 랭크 13) | 58.8% [53.7%, 63.8%] | 12 | +6 |
| 7 | 32 (pokedb 25) | 58.5% [53.3%, 63.5%] | 8 | +1 |
| 8 | 21 (pokedb 03) | 58.4% [53.2%, 63.4%] | 5 | -3 |
| 9 | 23 (pokedb 05) | 58.1% [52.9%, 63.1%] | 6 | -3 |
| 10 | 30 (pokedb 19) | 57.0% [51.8%, 62.1%] | 28 | +18 |
| 11 | 3 ([2780] らいざ Mega Gyarados) | 56.5% [51.4%, 61.6%] | 9 | -2 |
| 12 | 34 (pokedb 31) | 56.5% [51.3%, 61.5%] | 19 | +7 |

1위 하한 56.9% — 필드 위로 분리됨.
엔진 수정 전후 순위 상관(Spearman): **0.88** — 순위 대체로 유지.

## 개인 PC에서 실행할 것

> **큐 확정** (hk가 개인PC 세션에 직접 지시): pilot43_v2 판정 → 3200시뮬 A/B
> 판정 → 이 커리큘럼 순서. 조건 하나 — **3200 A/B 결과가 커리큘럼의 자가대전
> 예산을 정합니다.** 3200이 이기면 아래 명령의 `--search-iterations 800`을
> 개인PC 세션이 3200(+적은 배틀)으로 바꿔 끼우고 무엇을 왜 바꿨는지 보고합니다.
> 800이거나 구분 안 되면 그대로 800. 나머지 인자는 **적힌 그대로** 실행.
> 참고 수치: 고쳐진 엔진에서 기존 pilot43 미러가 54.5% → 47.7%로 재측정됨
> (개인PC 세션 측정) — 기존 체크포인트의 우세는 엔진 수정으로 사라졌습니다.

```bash
git pull
```

**추천 (검증된 레시피)** — curriculum4를 만든 그 설정, 파티만 새 상위 4(39,43,14,42)로:

```bash
python scripts/train_loop.py --init runs/imitate8/net.pt --iterations 8 \
    --battles 250 --search-iterations 800 --evaluate-every 2 \
    --evaluate-battles 60 --teams parties:39,43,14,42 --foe-teams parties \
    --hidden 512 --blocks 4 --workers 12 --bootstrap-weight 1 \
    --root-noise 0.25 --sample-turns 12 --gate 200 --train-steps 256 \
    --learning-rate 3e-4 --out runs/curriculum_v2 --name curriculum_v2
```

이 레시피가 이 프로젝트에서 손짜기 탐색을 분리 가능하게 넘은 유일한
방법입니다(curriculum4, 59.4% [54.5, 64.2]). 다른 시도들(8파티 확장, 1파티
특화, 3200시뮬 선생)은 전부 그보다 못했거나 미검증입니다.
`--search-iterations 800`만 위 큐 확정 블록의 규칙에 따라 교체 가능하고,
`--init`(imitate8 고정)·`--gate 200`·`--teams` 목록은 그대로 씁니다.

주의:
- `--workers`는 개인 PC 물리 코어 수에 맞추세요.
- `--foe-teams parties`는 상대를 46파티 전체에서 뽑습니다.
- 끝나면 400게임 판정: `python scripts/judge.py runs/curriculum_v2/best.pt \
    --teams parties:39,43,14,42 --matches 200`

**대안 (미검증, 관심 있으면)** — 새 1위(39) 단독 + 3200시뮬 선생 + GPU 서버:

```bash
python scripts/train_loop.py --init runs/imitate8/net.pt --iterations 8 \
    --battles 250 --search-iterations 3200 --leaf-batch 64 --gpu-server \
    --evaluate-every 2 --evaluate-battles 60 --teams parties:39 \
    --foe-teams parties --hidden 512 --blocks 4 --workers 12 \
    --bootstrap-weight 1 --root-noise 0.25 --sample-turns 12 --gate 200 \
    --train-steps 256 --learning-rate 3e-4 --out runs/pilot_v2 --name pilot_v2
```
