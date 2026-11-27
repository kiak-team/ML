# Feature Importance

## 1. Closed-world

(기존 `code/feature_importance_cv.py`, `logs/feature_importance_cv_*.log` 결과 정리 — 작성 예정)

## 2. Open-world — 그룹 permutation 중요도

### 2.1 설정

| 항목 | 내용 |
|---|---|
| 피처 | leak 제거(skip14) 피처 중 A팀 후보 134개 (`ow_models.load`, `@all @early @bseq @dir50 @qconst`) |
| 모델 | RF 200그루 (`ow_run.rf`, B 파트 E7~E10과 동일) |
| 라벨 | multi 96클래스 (monitored 95 + unmon -1) |
| 점수 | TPR@FPR≤1% (규칙 B, `ow_models.best_at`) |
| 분할 | visit 기준 train 5-fold. **test는 사용하지 않음** |
| 방법 | 검증 fold에서 그룹의 열들을 같은 순서로 함께 섞고 점수 하락을 잼. 그룹당 3회 평균 |

그룹은 `experiment.groups` 정의를 그대로 썼고, 134개가 9개 그룹에 중복·누락 없이 들어간다.

코드: `code/feature_importance_ow.py` (Colab 노트북에서 돌린 셀들을 스크립트로 정리한 것, 시드 동일)
결과: `results/fi/ow_group_perm.csv`(fold × 그룹별 하락), `results/fi/ow_group_summary.csv`(그룹별 평균·표준편차) · 그림: `figures/fi_ow_group.png`

### 2.2 결과 (134개)

기준 TPR@FPR≤1% = **0.892** (fold 평균; fold별 0.901 / 0.895 / 0.901 / 0.906 / 0.856)

| 그룹 | 개수 | 섞었을 때 하락 (%p, 평균 ± sd) |
|---|---:|---:|
| @qconst | 13 | 35.43 ± 1.57 |
| @burst | 11 | 20.29 ± 2.34 |
| @cumul | 12 | 11.24 ± 1.51 |
| @timing | 18 | 10.82 ± 1.07 |
| @bseq | 24 | 8.36 ± 1.19 |
| @early | 6 | 5.57 ± 0.89 |
| @order | 10 | 4.71 ± 0.92 |
| @count | 4 | 2.02 ± 0.55 |
| @dir50 | 36 | 1.17 ± 0.41 |

![](../figures/fi_ow_group.png)

### 2.3 해석

- **@qconst가 1위지만 사이트 정보라고 보기 어렵다.** 이 13개는 mon에서 거의 상수이고 unmon에서만 값이 변해서, 이것만으로 mon/unmon이 AUC 0.9999로 갈린다(`experiment.py` 주석). 즉 모델이 "monitored인가"를 가르는 데 크게 기대고 있다는 뜻이다. 수집 환경 차이일 가능성이 있어 121개(−@qconst) 결과를 따로 본다 → 2.4.
- **사이트 정보 중에서는 @burst가 가장 크다(20.3%p).** B 파트 E8에서도 Burst는 빼고 재학습했을 때 유일하게 크게 떨어진 그룹(−25.8%p)이라 결론이 일치한다.
- **@dir50은 36개로 가장 많지만 영향은 가장 작다(1.2%p).** E8에서 빼도 같은 수준이었던 것과 일치한다.
- permutation은 재학습 없이 "학습된 모델이 얼마나 기대는가"를 잰다. 그래서 E8의 제거 실험(다른 피처가 대신 채움)보다 값이 크게 나온다. 예: @qconst는 제거 시 −2.6%p, 섞으면 −35.4%p. 두 결과는 서로 보완 관계로 읽는다.
- fold 4는 기준 점수부터 0.856으로 낮다.

### 2.4 @qconst를 뺀 121개

재현: `python code/feature_importance_ow.py --drop-qconst`
결과: `results/fi/ow_group_perm_121.csv`, `results/fi/ow_group_summary_121.csv` · 비교 그림: `figures/fi_ow_group_compare.png`

기준 TPR@FPR≤1% = **0.870** (fold별 0.873 / 0.873 / 0.885 / 0.902 / 0.818). 134개보다 2.2%p 낮고, E8에서 @qconst를 뺐을 때(−2.6%p)와 같은 수준이다.

| 그룹 | 개수 | 134개 (%p) | 121개 (%p) |
|---|---:|---:|---:|
| @burst | 11 | 20.29 ± 2.34 | **50.63 ± 2.57** |
| @timing | 18 | 10.82 ± 1.07 | 16.62 ± 0.83 |
| @cumul | 12 | 11.24 ± 1.51 | 15.69 ± 1.76 |
| @bseq | 24 | 8.36 ± 1.19 | 13.99 ± 1.67 |
| @early | 6 | 5.57 ± 0.89 | 8.61 ± 0.96 |
| @order | 10 | 4.71 ± 0.92 | 8.46 ± 1.09 |
| @dir50 | 36 | 1.17 ± 0.41 | 2.69 ± 0.74 |
| @count | 4 | 2.02 ± 0.55 | 1.57 ± 0.60 |
| @qconst | 13 | 35.43 ± 1.57 | — |

![](../figures/fi_ow_group_compare.png)

- @qconst가 없으면 모델은 mon/unmon 구분까지 다른 그룹에 맡기게 되고, 그 몫을 대부분 **@burst**가 가져간다(20.3 → 50.6%p). 섞었을 때 TPR이 절반 넘게 떨어지므로 121개 모델에서는 burst가 사실상 핵심이다.
- 나머지 그룹도 모두 커졌지만, 순서는 거의 그대로다: burst > timing ≈ cumul > bseq > early ≈ order > dir50 ≈ count. 134개에서 본 사이트 정보 순위가 @qconst 때문에 생긴 착시는 아니라는 뜻이다.
- @dir50과 @count는 두 설정 모두에서 3%p 이하라 영향이 작다.
- fold 4가 두 설정 모두 가장 낮다(0.856 / 0.818).
