# Extra credit: `_join.cell` 파일을 어떻게 쓸까

## 데이터 구성

조교님이 준 zip(mon / unmon)에는 방문 1회마다 파일이 6개씩 들어 있다.

| 파일 | 뜻 |
|---|---|
| `*_split_0` ~ `*_split_4.cell` | TrafficSliver BWR로 나뉜 5개 경로 각각의 트래픽 |
| `*_join.cell` | 5개 조각을 다시 합친 방어 전 원본 트래픽 |

`mon_25`와 `unmon_5000`의 방문 10,000개 모두에서 join의 셀 수가 split 5개의 셀 수 합과 정확히 같았다.
`load_data.build_sliver`는 이 조건을 assert로 다시 확인한다.

## split과 섞어 쓰면 안 되는 이유

1. **답이 새어 나간다.** join은 split 5개를 그대로 이어 붙인 것이다.
   어떤 방문의 join이 train에 있고 같은 방문의 split이 test에 있으면, 모델은 test 트래픽을 이미 본 셈이다.
2. **푸는 문제가 다르다.** 공격자는 입구 노드 하나만 감시하므로 split 하나만 볼 수 있다. join은 볼 수 없는 데이터다.
   join을 test에 섞으면 방어 없는 쉬운 문제가 끼어들어 방어를 깬 정도가 부풀려진다.

## 어떻게 쓰나

모든 실험에서 train/test는 **`group` 단위**로 나눈다. `group`은 같은 방문의 split 5개와 join이 공유하는 값이다.
sklearn에서는 `GroupKFold`나 `GroupShuffleSplit`을 쓰면 된다.
최종 test는 `ld.sliver_is_test(group)`가 True인 방문(10%)으로 고정한다. 튜닝 중에는 이 방문을 보지 않는다.

| 실험 | 학습 | 평가 | 무엇을 보여주나 |
|---|---|---|---|
| A. 기준선 (방어 없음) | join | join | 방어가 없을 때 정확도 X% |
| B. 본 실험 (방어 있음) | split | split | 공격자가 실제로 얻는 정확도 Y% |
| C. 선택 | join | split | 방어 전 데이터로 학습하면 split에도 통하는가 |

보고할 때는 A와 B를 나란히 놓아 방어 효과(X → Y)를 보이고, B를 얼마나 끌어올렸는지를 extra credit 결과로 내세운다.

## 코드

```python
import load_data as ld
splits = ld.iter_sliver('C:/Users/user/Downloads/sliver_mon.npz')             # part 0~4 (기본)
joins  = ld.iter_sliver('C:/Users/user/Downloads/sliver_mon.npz', parts=[5])  # join만
for times, sizes, label, group, part in splits: ...
```

- `label`: 사이트 번호 0–94. unmonitored는 -1이다.
- `part`: 0–4는 split, 5는 join이다.
- 빈 split 파일(약 8–10%)은 변환할 때 뺐다. join은 비어 있는 경우가 없다.
- 시간은 파일마다 첫 셀이 0이 되도록 맞췄다. 원본 split은 join의 시계를 그대로 가지고 있어서 시작 시간이 0이 아니다.

## 첫 결과 (2026-11-10, `logs/feature_importance_sliver.log`)

RF(200), 기존 피처 375개, test를 뺀 방문으로 group 5-fold, 95-class (찍으면 1.05%).

| 실험 | 정확도 |
|---|---|
| A join→join (방어 없음) | 89.3% ± 0.6 |
| B split→split (방어 있음) | 7.5% ± 0.1 |
| C join→split | 1.4% ± 0.1 |

- TrafficSliver가 89% → 7.5%로 막는다. C가 찍는 수준이라 학습도 split으로 해야 한다.
- split에 남은 정보는 주로 **시간** 피처다. 그룹을 섞었을 때 하락은 `@timing` −5.0%p로 가장 크다.
  그다음은 `@cumul100` −2.6%p로, 방향 누적합 곡선의 전체 모양을 본다. 나머지 그룹은 0에 가깝다.
  불순도 상위 16개도 모두 시간 피처다(`iat_*`, `t_*pct`, `duration`, `pps_*`).
- 방어가 없을 때 핵심이던 방향·위치 피처(`@early`, `@bseq`, `@dir`)는 split에서 쓸모가 없다.
  BWR이 셀을 무작위 경로로 보내서 한 경로의 셀 순서가 사이트와 무관해지기 때문으로 보인다.
- 이후 시간 피처를 추가해 15.5%까지 올렸다. 전체 정리는 [EXTRA_CREDIT.md](EXTRA_CREDIT.md)에 있다.
