# closed-world 모델링 기록 (C파트)

감시 대상 95개 사이트 분류(closed-world)의 모델 선택·튜닝·딥러닝·앙상블·오류 분석을 정리한 문서다.
피처와 전처리는 A파트 결과를 그대로 썼다([WORK_SUMMARY.md](WORK_SUMMARY.md)). 작성 2026-11-27.

## 요약

- **모든 점수는 train 5-fold CV(OOF) 점수다. test는 아직 보지 않았다.** 분할은 11/15 새 분할([TEST_SPLIT.md](TEST_SPLIT.md)), fold는 `inst % 5`다(B파트 `splits/pkl_mon.csv`와 같음을 확인).
- 10월의 SoftVote 0.9716 등은 옛 분할(inst 18~19 = test) 기준이라 **전부 새 분할로 다시 쟀다.** 새 분할에서는 inst 19가 train fold 4에 들어가 점수가 0.5~1%p 낮다.
- **단일 모델 최고: LightGBM 튜닝 96.56%.** sklearn 모델 중에서는 HGB 95.84%가 가장 높았다.
- **딥러닝(DF 1D-CNN):** 논문 그대로(방향만) 94.81% → **방향 + 시간 간격 2채널 96.20%**(+1.39%p). 시간을 곱해 넣는 Tik-Tok 방식은 93.28%로 오히려 떨어졌다.
- **최종 후보: 7개 모델 stacking 98.09% ± 0.59** (LightGBM, ExtraTrees, RandomForest, DF 방향, DF 방향+시간 간격, k-NN, LR).
  가중치·메타 모델은 fold마다 나머지 4 fold OOF로만 학습한 중첩 CV 점수다.
- **LightGBM 없이도 97.69%**(HGB·ET·RF·DF·k-NN·LR·MLP, DF 방향+시간 간격 추가 전 기준)라, 수업 모델만 써야 해도 차이가 작다.
- **앙상블은 개수보다 다양성이 중요했다.** 트리 모델 위주 12개(97.57%)보다 성격이 다른 7개(98.09%)가 높았다.
- **남은 오류(327개, 1.9%)는 느리거나 덜 로딩된 기록에 몰려 있다.** inst 19는 94.4%(나머지 98.3%)이고, train의 5%인데 오류의 14.7%다.

## 1. 실험 설정

| 항목 | 내용 |
|---|---|
| 데이터 | mon 19,000개 중 train 17,100개 (test 1,900개는 쓰지 않음) |
| 피처 | A파트 후보 131개 `@all -@const @early @bseq @cumulabs @dir50` (트리 모델 비교용으로 181개 `@dir` 버전도 시험) |
| 전처리 | `experiment.build()` 그대로: 트리는 원본, 그 외는 signed log + 표준화, k-NN·SVM은 그룹 가중치 선택 |
| DF 입력 | `code/sequences.py`로 만든 방향 시퀀스 (길이 5,000, mon 33% 잘림) |
| 평가 | 5-fold OOF 정확도. 모든 모델의 OOF 확률을 저장해 앙상블은 재학습 없이 계산 |
| 코드 | `code/cw_tuning.py`(튜닝·앙상블·test), `code/cw_df.py`(DF, Colab GPU), `code/cw_error.py`(오류 분석), `code/cw_plots.py`(그래프) |

## 2. 단일 모델

| 모델 | 기본값 | 튜닝 후 | 최고 설정 |
|---|---|---|---|
| **LightGBM** | 95.56% | **96.56%** ± 0.75 | lr 0.03, 400그루, num_leaves 63, colsample 0.5, subsample 0.9 |
| HistGB (sklearn) | 95.84% | 95.81% | 기본값(lr 0.05, 400회)이 최적 |
| ExtraTrees | 95.28% | 95.28% | 기본값(300그루, max_features 0.1)이 최적 |
| RandomForest | 95.25% | 95.25% | 기본값(300그루, max_features 0.1)이 최적 |
| SVM (RBF) | 94.63% | 94.32% | 기본값(C=10, 그룹 가중치 켬)이 더 좋음 |
| k-NN | 90.69% | **94.05%** | k=3, 맨해튼 거리, 거리 가중 (+3.36%p) |
| MLP | — | 93.31% | 은닉 512 한 층, alpha 0.001 |
| 로지스틱 회귀 | 89.06% | 89.06% | C=1(기본값)이 최적 |

- 튜닝은 random search(모델별 4~20회), 기준은 CV 정확도다.
- **RF·ET·LR·HGB는 기본값이 이미 최적에 가까웠다.** B파트 E10(RF 튜닝 +0.1%p)과 같은 결론이다.
- **부스팅은 학습률에 민감하다.** HGB를 lr 0.1로 돌린 3회는 모두 발산했다(정확도 17.8~20.1%). EXPERIMENTS.md 고도화 재실험의 관찰과 같다.
- LightGBM은 시드를 바꿔도(96.53%, 96.56%) 피처를 181개로 바꿔도(96.56%) 같았다. 트리 모델에 181개를 쓰면 ET +0.09%p, RF +0.05%p로 시드 변동 수준이다.
- 모든 모델이 fold 4(inst 4·9·14·19)에서 2~4%p 떨어진다. inst 19 때문이다(5절).

## 3. 딥러닝: DF 1D-CNN

DF(Sirinam et al., CCS 2018) 구조를 PyTorch로 구현했다. conv 블록 4개(32/64/128/256, kernel 8) + FC 512·512,
Adamax lr 0.002, batch 128, 60 epoch 고정(검증 fold로 early stopping 하지 않아 OOF가 깨끗하다). Colab T4에서 epoch당 약 5초.

| 변형 | 바꾼 것 | CV | 기본 대비 |
|---|---|---|---|
| **iat** | **방향 + log(1 + 시간 간격) 2채널** | **96.20%** ± 0.69 | **+1.39%p** |
| s1 | 시드만 바꿈 | 95.04% | +0.23 |
| cos | 학습률 cosine 감소 | 95.03% | +0.22 |
| 기본 | 방향만 (논문 그대로) | 94.81% ± 1.34 | — |
| shift | 학습 증강: 시작 위치 0~20칸 밀기 | 94.56% | −0.25 |
| L10k | 길이 10,000 (잘림 0%) | 94.03% | −0.78 |
| tt | 방향 × 시각 (Tik-Tok, PETS 2020) | 93.28% | −1.53 |

- 30 epoch에서는 아직 오르는 중이었고(fold 0 94.1%), 50 epoch 이후 94.7~94.8%에서 멈췄다 → 60 epoch로 정했다.
- **시간 정보는 어떻게 넣느냐가 중요하다.** 패킷 사이 간격을 log로 따로 넣으면 크게 오르고, 시각을 방향에 곱하면 떨어진다.
- 길이를 10,000으로 늘리면 오히려 떨어졌다. 사이트 정보가 앞부분에 몰려 있다는 A파트 분석과 맞는다.
- **피처 131개 + LightGBM(96.56%)이 원본 시퀀스 딥러닝보다 여전히 높다.** 그러나 DF는 앙상블에서 대체 불가능한 역할을 한다(4·5절).
- TAM 입력(Robust Fingerprinting, USENIX Security 2023: 44ms 구간별 송신·수신 셀 수)은 11/27 Colab GPU 사용량 한도로 중간에 멈췄다. fold별 결과가 Drive에 저장돼 있어 이어서 돌릴 수 있다.

## 4. 앙상블

섞는 방법 4가지를 같은 멤버로 비교한다. 가중 평균과 stacking은 가중치·메타 모델을 학습하므로 **fold k의 점수는 나머지 4 fold의 OOF로만 학습해서 낸다(중첩 CV).**

| 방법 | 내용 |
|---|---|
| mean | 확률 평균 |
| geo | 확률의 기하 평균 (log 평균) |
| wvote | 가중 평균. 가중치는 Dirichlet 무작위 600개 + 같은 비중 중 정확도 최고 |
| stack | 멤버 확률을 이어 붙여 로지스틱 회귀(C=1) 학습. log 확률보다 확률 그대로가 1%p 이상 좋았다 |

| 멤버 | mean | geo | wvote | stack |
|---|---|---|---|---|
| **최종 7개**: LightGBM, ET, RF, DF, DF-iat, k-NN, LR | 97.95 | 97.81 | 98.02 | **98.09** ± 0.59 |
| 6개: LightGBM, ET, RF, DF, k-NN, LR | 97.51 | 97.30 | 97.63 | 97.78 |
| LightGBM 없이 7개: HGB, ET, RF, DF, k-NN, LR, MLP | 97.26 | 97.37 | 97.63 | 97.69 |
| 6개 + HGB + MLP | 97.38 | 97.49 | 97.61 | 97.66 |
| 다양하게 9개: LightGBM ×3(시드), ET·RF(181), DF, SVM, k-NN, LR | 97.43 | 97.61 | 97.71 | 97.73 |
| 확률 상위 12개 (트리 위주) | 97.19 | 97.49 | 97.57 | 97.50 |

- 최종 7개 stacking의 fold별 정확도: 98.48 / 98.07 / 98.54 / 98.39 / **96.96**. fold 4도 단일 모델(92~95%)보다 크게 높다.
- 전체 OOF로 맞춘 가중 평균 가중치: LightGBM 0.20, ET 0.26, RF 0.06, DF 0.08, **DF-iat 0.21**, k-NN 0.08, LR 0.11.
- **비슷한 모델을 더 넣으면 오히려 떨어진다.** HGB는 LightGBM과, LightGBM 시드 변형은 서로 같은 기록을 틀린다.
- 여러 조합을 CV로 비교해 골랐으므로 최고 조합의 CV는 약간 낙관적이다. 최종 성능은 test 1회로 본다.

## 5. 오류 분석 (최종 7개 stacking, `code/cw_error.py`)

**더 올릴 여지**
- 멤버 중 하나라도 맞힌 비율(오라클) **99.04%**. 섞는 방법만 바꿔서는 최대 +0.95%p다.
- **7개 모델이 모두 틀린 기록 165개(0.96%)**는 새 입력이나 새 모델이 있어야 맞힐 수 있다.

**어디서 틀리나**

| 조건 | 앙상블 정확도 |
|---|---|
| inst 19 (마지막 방문) | **94.4%** (단일 LightGBM 91.7%, 나머지 방문 평균 98.3%) |
| 로딩 시간 5분위: 아주 빠름 → 아주 느림 | 98.9 / 99.1 / 98.0 / 98.6 / **95.8** |
| 기록 길이 5분위: 아주 짧음 → 아주 김 | **94.4** / 98.6 / 99.0 / 99.3 / 99.1 |
| 약 10,000셀에서 잘린 긴 기록(11.6%) | 99.3% (나머지 97.9%) |

- inst 19는 train의 5.0%인데 오류의 **14.7%**다. 남은 오류는 **느리거나 덜 로딩된 방문**이라는 데이터 상태에서 나온다.
- 특정 사이트 문제는 아니다. 가장 어려운 사이트도 93%(사이트 61, 82)이고, 100%인 사이트가 10개다. 자주 헷갈리는 상위 10쌍(4↔30, 61↔67, 82↔88 등)이 오류의 11%뿐이다.

**멤버별 기여**: 다른 멤버가 모두 틀렸는데 혼자 맞힌 기록 수

| LightGBM | ET | RF | DF | **DF-iat** | k-NN | LR |
|---|---|---|---|---|---|---|
| 27 | 11 | 7 | 13 | **38** | 8 | 17 |

DF-iat가 가장 대체 불가능하다. 단독 정확도가 낮은 LR(89%)도 17개를 혼자 맞혀 앙상블에 남는다.

## 6. 그래프 (`figures/`)

| 파일 | 내용 |
|---|---|
| `cw_models.png` | 모델별 CV 정확도 (튜닝 전후, DF 변형) |
| `cw_ensemble.png` | 단일 최고 vs 섞는 방법 4가지 |
| `cw_folds.png` | fold별 정확도. fold 4(inst 19)에서 모두 떨어지고 앙상블이 메운다 |
| `cw_diversity.png` | 두 모델의 예측이 서로 다른 비율 (다양성) |
| `cw_errors.png` | 방문 번호별 정확도, 가장 어려운 사이트 10개 |
| `cw_df_curve.png` | DF 학습 곡선 (fold별, 60 epoch) |

## 7. 재현 방법

```
# 레포 루트에서 (경고 숨김: set PYTHONWARNINGS=ignore)
python -I -W ignore -X utf8 code/cw_tuning.py base -m lr knn svm rf et lgbm hgb
python -I -W ignore -X utf8 code/cw_tuning.py tune -m lgbm -n 6        # 모델별 random search
python -I -W ignore -X utf8 code/cw_tuning.py best -m svm --set C=10 _gw=True probability=True --name svm_proba
python -I -W ignore -X utf8 code/cw_tuning.py summary
python -I -W ignore -X utf8 code/cw_tuning.py ensemble --members lgbm_tuned et_base rf_base df_base df_iat knn_tuned lr_base
python -I -W ignore -X utf8 code/cw_error.py --members lgbm_tuned et_base rf_base df_base df_iat knn_tuned lr_base
python -I -W ignore -X utf8 code/cw_plots.py

# DF (Colab T4). 입력: sequences.py data --length 5000 --time
python -I -W ignore -X utf8 code/cw_df.py --epochs 60 --tag base
python -I -W ignore -X utf8 code/cw_df.py --epochs 60 --input dir+iat --tag iat
```

결과는 `results/closedworld_results.csv`(B파트와 같은 20개 컬럼), `results/cw_best_params.json`, `results/cw_ensemble.json`,
`results/cw_errors_*.csv`에 쌓인다. OOF 예측(`results/cw_oof/*.npz`, 모델당 수 MB)은 용량 때문에 커밋하지 않고 Drive에 둔다.

## 8. test 평가 계획 (아직 하지 않음)

1. 최종 조합은 CV로 미리 정하고 이 문서에 적은 뒤 test를 돌린다. test를 보고 다시 고르지 않는다.
2. DF 멤버는 Colab에서 전체 train으로 다시 학습해 test 확률을 만든다.
   `cw_df.py --final --epochs 60 --tag base`, `cw_df.py --final --epochs 60 --input dir+iat --tag iat` → `results/cw_test/df_*_test.npz`
3. `cw_tuning.py test --final stack:lgbm_tuned+et_base+rf_base+df_base+df_iat+knn_tuned+lr_base --compare (비교표용 단일 모델들)`
   한 번 실행하면 `results/cw_test_LOCK`이 생겨 다시 실행되지 않는다(B파트 `ow_test.py`와 같은 방식).
4. 보고 항목: test 정확도(1,900개), inst 19 정확도(95개), macro F1, 혼동행렬(`results/cw_test/`).

