# 데이터 전처리 작업 정리

Tor 웹사이트 핑거프린팅(WF) 프로젝트의 **데이터 전처리** 작업 베이스다. 데이터 읽기, 데이터 점검, 피처 추출, 피처 유효성 간단 확인까지 포함한다.

- 데이터 구조와 통계 → [DATA.md](DATA.md)
- 피처 55개의 정의 → [FEATURES.md](FEATURES.md)
- 피처 선택 실험의 단계별 기록 → [EXPERIMENTS.md](EXPERIMENTS.md)

## 폴더 구조와 파일별 역할

```
wf_project/
├─ README.md                  프로젝트 전체 소개 (작성 예정)
├─ requirements.txt           ⑥⑦에 필요한 패키지 (numpy, pandas, scipy, scikit-learn)
├─ docs/
│  ├─ PREPROCESSING.md        ← 이 파일 (전처리 작업 정리)
│  ├─ DATA.md                 데이터 설명 (구조, 통계, 품질 점검)
│  ├─ FEATURES.md             피처 55개 설명
│  └─ EXPERIMENTS.md          피처 선택 실험 기록 (단계별 가설·결과·결정)
├─ code/
│  ├─ load_data.py            ① 공용 데이터 로더
│  ├─ check_data.py           ② 데이터 품질 점검
│  ├─ explore.py              ③ 기본 통계 (길이·시간·송신 비율)
│  ├─ features.py             ④ 피처 추출 + 피처 분석
│  ├─ eval_nearest_centroid.py ⑤ 피처 유효성 간단 평가
│  ├─ feature_importance.py   ⑥ 랜덤 포레스트 피처 중요도 (sklearn 필요)
│  ├─ experiment.py           ⑦ 피처·모델을 골라 직접 실험 (sklearn 필요)
│  ├─ features_extra.py       ⑧ 추가 피처 150개 추출 (3단계 실험용)
│  ├─ features_extra2.py      ⑧-2 추가 피처 2차 추출 (4단계 실험용)
│  └─ run_stages.py           ⑨ EXPERIMENTS.md의 단계별 실험 배치
├─ features/
│  ├─ features_mon.csv        ④가 저장한 mon 피처 (19,000행 × site,url,inst + 55열)
│  ├─ features_unmon.csv      ④가 저장한 unmon 피처 (10,000행 × 55열)
│  ├─ features_extra_mon.csv  ⑧이 저장한 추가 피처 (19,000행 × site,url,inst + 150열)
│  ├─ features_extra_unmon.csv ⑧이 저장한 추가 피처 (10,000행 × 150열)
│  ├─ features_extra2_mon.csv ⑧-2가 저장한 추가 피처 (19,000행 × site,url,inst + 171열)
│  └─ features_extra2_unmon.csv ⑧-2가 저장한 추가 피처 (10,000행 × 171열)
└─ logs/                      각 코드의 실행 출력
   ├─ check_data.log          ②의 출력
   ├─ explore.log             ③의 출력
   ├─ features_run.log        ④의 출력
   ├─ eval_nearest_centroid.log ⑤의 출력
   ├─ feature_importance.log  ⑥의 출력
   ├─ experiments.csv         ⑦의 실험 기록 (각자 로컬, git에 올리지 않음)
   ├─ features_extra2_run.log ⑧-2의 출력
   └─ stage1~4.log            ⑨의 단계별 결과
```

### ① `code/load_data.py`: 공용 데이터 로더 (**팀 전원이 이 파일로 데이터를 읽는다**)
교수님이 제공한 `load_pickle_code.ipynb`와 **같은 규칙**으로 pkl을 읽는다.
- 라벨 `i // 10` (사이트 95개), 방향 `c > 0`이면 송신(+1) 아니면 수신(-1), 시간 `abs(c)`, 크기는 방향 × 512
- `load_mon(경로)` → 노트북과 같은 `X1`(시간), `X2`(방향×512), `y`(라벨)
- `load_unmon(경로)` → `X1`, `X2`
- `iter_mon(경로)` → 기록을 하나씩 반환해 메모리를 덜 쓴다. URL 번호와 방문 순번(inst)도 함께 반환한다.

```python
import sys; sys.path.insert(0, 'code')
import load_data as ld
X1, X2, y = ld.load_mon('C:/Users/user/Downloads')
```

### ② `code/check_data.py`: 데이터 품질 점검
빈 기록, NaN, 시간 역행, 길이 상한, 0.01초 단위 여부 등을 확인한다. 또한 **`load_data.py`가 교수님 노트북 코드와 같은 결과를 내는지** 29,000개 기록 전부를 비교한다(현재 불일치 0개).

### ③ `code/explore.py`: 기본 통계
길이 분위수, 지속 시간, 송신 비율을 mon과 unmon 각각 출력한다. [DATA.md](DATA.md) 3절의 표가 이 결과다.

### ④ `code/features.py`: 피처 추출 + 분석
1. 기록마다 피처 55개를 추출해 CSV 2개로 저장한다. **모델 학습에 쓰는 결과물이 이 CSV다.**
2. 피처별로 사이트 구분력(η²)과 mon/unmon 구분력(AUC)을 계산한다.
3. 거의 같은 정보를 담은 피처 쌍(상관계수 0.9 초과)을 찾는다.
4. 간단한 분류기(⑤와 동일)로 정확도를 출력한다.

### ⑤ `code/eval_nearest_centroid.py`: 피처 유효성 평가
④가 저장한 CSV를 읽어 아주 단순한 분류기로 정확도를 잰다. **최종 모델이 아니라** "이 피처들로 사이트를 구분할 수 있는가"를 빠르게 확인하는 용도다.
- **분류기 (nearest centroid)**: train 데이터로 사이트별 피처 평균점을 구하고, test 기록을 가장 가까운 평균점의 사이트로 예측한다.
- **분할**: URL마다 방문 20회 중 0~17번째는 train(17,100개), 18~19번째는 test(1,900개). 무작위 요소가 없다.
- **전처리**: `sign(x)·log(1+|x|)`로 값의 크기를 줄인 뒤 train의 평균과 표준편차로 표준화한다.
- ④와의 차이: 피처 선택용 η²를 **train으로만** 계산한다. ④는 test까지 포함해 계산하므로 데이터 누수가 있다.

### ⑥ `code/feature_importance.py`: 피처 중요도 분석 (sklearn 필요)
⑤와 같은 분할로 랜덤 포레스트를 학습해 "모델이 실제로 어떤 피처를 쓰는가"를 본다. η²는 피처를 하나씩 따로 보지만, 이 분석은 피처끼리 정보가 겹치는 것까지 반영한다.
- **불순도 중요도**: 트리가 분기할 때 그 피처를 얼마나 효과적으로 썼는가
- **permutation 중요도**: test에서 그 피처 값을 섞었을 때 정확도가 얼마나 떨어지는가
- **상관 묶음 permutation**: 서로 대신하는 피처(상관 높음)를 묶음째 섞는다. 하나씩 섞으면 다른 피처가 대신해 중요도가 낮게 나오기 때문이다.
- **그룹 실험**: 그룹만 쓸 때 / 그룹을 뺄 때 정확도 (재학습)
- **mon vs unmon**: open-world에서 어떤 피처가 mon과 unmon을 가르는가
- test 결과를 보고 피처를 고르면 누수가 생긴다. 이 결과는 **방향을 잡는 탐색용**이고, 실제 선택은 train 내부 교차검증으로 한다.

### ⑦ `code/experiment.py`: 피처·모델을 골라 직접 실험 (sklearn 필요)
피처 조합과 모델을 바꿔 가며 정확도를 비교한다. **기본은 train 내부 5-fold 교차검증(CV)이고, test는 `--test`를 줄 때만 쓴다.** 피처를 고르는 동안에는 CV만 보고, 최종 조합이 정해지면 test를 한 번만 본다. test를 보면서 피처를 바꾸면 test에 우연히 맞는 조합을 고르게 된다.

- **피처 지정**: 이름(`first30_out`), 그룹(`@all @count @timing @order @burst @cumul @length`), 제외(`-이름`, `-@그룹`)
  - `@length`는 ⑥에서 한 묶음으로 나온 "기록 길이" 계열 17개다.
- **모델**: 수업에서 다룬 모델 `lr nb svm dt rf et knn mlp`. 하이퍼파라미터는 `-p 키=값`
- **전처리**: 트리 모델(dt, rf, et)은 그대로 쓴다. 나머지는 `sign(x)·log(1+|x|)` 후 표준화한다(기준은 학습 fold로만 계산). `--no-log`, `--no-scale`로 끌 수 있다.
- **CV fold**: `inst % 5`
- 실행할 때마다 `logs/experiments.csv`에 한 줄씩 쌓인다(`--note`로 메모 가능).

```
python -I -X utf8 code/experiment.py --list                          # 피처·그룹·모델 목록
python -I -X utf8 code/experiment.py @all -@timing -m knn -p n_neighbors=3
python -I -X utf8 code/experiment.py n_total frac_out first30_out -m svm -p C=10
python -I -X utf8 code/experiment.py @all -m rf --test               # 최종 1회 평가
```

노트북에서는 다음처럼 쓴다.

```python
import sys; sys.path.insert(0, 'code')
from experiment import run
r = run(['@order', 'n_total'], model='knn', n_neighbors=3)   # r['cv_mean'], r['features']
```

### ⑧ `code/features_extra.py`, ⑨ `code/run_stages.py`: 단계별 피처 실험
⑧은 원본 pkl에서 추가 피처 150개(앞부분 송신 수, 고정 위치 누적합, 버스트 순서, CUMUL식 보간 누적합)를 뽑아 별도 CSV로 저장한다. ④의 CSV는 그대로 둔다. ⑦은 이 CSV가 있으면 자동으로 붙여 `@early @cumulabs @bseq @cumul100 @extra` 그룹으로 쓸 수 있다. ⑨는 [EXPERIMENTS.md](EXPERIMENTS.md)의 단계별 실험을 그대로 다시 돌린다.

```
python -I -X utf8 code/features_extra.py <pkl 폴더> features         # 약 1분
python -I -X utf8 code/run_stages.py 3 | tee logs/stage3.log          # 약 1시간
```

⑧-2 `code/features_extra2.py`는 4단계용 추가 피처 171개(31~100번째 버스트 길이 70개, 처음 100셀의 방향값 100개, 기록의 버스트 수 1개)를 같은 방식으로 별도 CSV에 저장한다. ⑦에서 `@bseqlong @dir @dir50` 그룹으로 쓴다.

```
python -I -X utf8 code/features_extra2.py <pkl 폴더> features        # 약 1분
python -I -X utf8 code/run_stages.py 4 | tee logs/stage4.log          # 약 30분
```

## 실행 방법

①~⑤는 Python 3 표준 라이브러리만 사용한다(numpy·sklearn 불필요). `<pkl 폴더>`는 pkl 3개를 둔 폴더다. 아래 명령은 **프로젝트 최상위 폴더(`wf_project/`)에서** 실행한다.

```
python -I code/check_data.py <pkl 폴더>                          # 수 분 소요
python -I code/explore.py    <pkl 폴더>
python -I -X utf8 code/features.py <pkl 폴더> features            # 약 2~3분, CSV 2개 생성
python -I -X utf8 code/eval_nearest_centroid.py features/features_mon.csv
```

⑥은 패키지가 필요하다(`pip install -r requirements.txt`). pkl 없이 `features/`의 CSV만 읽는다.

```
python -I -X utf8 code/feature_importance.py .                  # 약 10~15분
```

## 현재까지 결과 (nearest centroid, 사이트 95개, test 1,900개)

우연히 맞힐 확률은 1/95 = 1.05%다.

| 피처 조합 | 정확도 |
|---|---|
| 전체 55개 | 25.7% |
| 순서·집중도 (10개) | 25.5% |
| 버스트 (11개) | 17.5% |
| 시간 (18개) | 11.4% |
| 개수 (4개) | 10.2% |
| CUMUL (12개) | 8.2% |
| η² 상위 10개 (train 기준) | 10.7% |
| η² 상위 20개 (train 기준) | 9.7% |
| η² 하위 20개 (train 기준) | 15.3% |

- 아주 단순한 분류기로도 우연(1%)보다 훨씬 높으므로, 피처에 사이트 정보가 담겨 있다.
- η² 상위 20개가 하위 20개보다 정확도가 **낮다**. 상위 피처끼리 같은 정보("기록 길이")를 반복해서 담고 있기 때문이다. 피처 선택 시 상관관계도 함께 봐야 한다.
- 강의에서 소개된 기법(k-FP, CUMUL, DF)은 90% 이상이다. 실제 모델(랜덤 포레스트, SVM, k-NN 등)은 모델링 단계에서 이 CSV로 학습한다.

## 랜덤 포레스트 피처 중요도 (⑥, 트리 300개)

전체 55개 피처로 **73.2%** 다. nearest centroid(25.7%)보다 훨씬 높으므로, 피처가 쓸 만하고 이전 결과는 분류기가 너무 단순했던 탓이다.

**1. 기록 길이(트래픽 양)가 가장 중요하지만, 피처 17개가 같은 정보다.**
`n_total`, `n_in`, `cumul_*`, `*_order_*` 등 17개를 묶음째 섞으면 정확도가 **51.7%p** 떨어진다. 하나씩 섞으면 각각 0.5%p 안팎이다. 서로 대신하므로 대표 1~2개면 충분하다.

**2. 기록 앞부분의 방향 패턴이 그다음이다.**

| 피처 | 섞었을 때 정확도 하락 |
|---|---|
| `first30_out` (처음 30셀 중 송신 수) | 5.0%p (단일 피처 1위) |
| `first100_out` | 3.5%p |
| `cumul_at_500` (500번째 셀의 누적합) | 2.7%p |

고정된 셀 위치의 값(`cumul_at_500`)이 길이 대비 비율 위치의 값(`cumul_10pct` 등)보다 유용하다.

**3. 송신 비율과 버스트 리듬**: `frac_out`, `in_burst_mean`, `out_burst_mean`, `conc_std`가 각각 1~2%p를 기여한다.

**그룹 실험**

| 그룹 | 이 그룹만 | 이 그룹을 빼면 |
|---|---|---|
| 순서·집중도 (10개) | 69.2% | 69.0% (유일하게 빼면 손해) |
| CUMUL (12개) | 60.5% | 70.7% |
| 버스트 (11개) | 52.5% | 72.8% |
| 시간 (18개) | 47.1% | 73.3% (빼도 그대로) |
| 개수 (4개) | 38.6% | 72.6% |

- 시간 피처는 사이트 분류에서 다른 피처와 정보가 겹친다. 0.01초 해상도 때문으로 보인다.
- `in_burst_1~3`은 중요도 0이다.

**mon vs unmon**: AUC 0.968. 1위는 `first30_out`이고, `duration`, `t_25pct` 같은 시간 피처가 상위권이다. mon과 unmon의 수집 환경 차이를 배운 것일 수 있으므로 주의한다.

## 마지막 방문(inst 19)은 수집 상태가 다르다

RF(트리 100개)를 학습할 때마다 방문 2개(2,000 → 1,900개)를 test로 빼면, **어느 2개를 빼도 87~90%인데 18~19만 73%** 다.

| | inst 0~17 | inst 18 | inst 19 |
|---|---|---|---|
| 길이 중앙값 | 약 3,300~3,500 | 3,184 | **2,340** |
| 지속 시간 중앙값 | 약 22~26초 | 29.9초 | **40.7초** |
| 초당 셀 수 중앙값 | 약 124~131 | 117 | **76** |
| RF 정확도 (inst 0~17로 학습) | - | 82.2% | **63.7%** |

- inst 19는 더 느리고 짧다. 네트워크가 느렸거나 페이지 로딩이 덜 끝난 기록이 많은 것으로 보인다. 수집 코드가 없어 원인을 확정할 수는 없다.
- `first30_out`, `cumul_at_500` 같은 앞부분 피처는 inst 19에서도 값이 같다. 기록이 덜 끝나도 앞부분은 영향을 받지 않기 때문이다. 길이 계열 피처는 영향을 크게 받는다.
- 따라서 **현재 test(inst 18~19)는 일반적인 방문보다 어려운 세트**다. ⑤⑥의 test 정확도는 낮게 나온 값이고, ⑦의 CV(약 88%)와 직접 비교하면 안 된다.
- **팀 결정 필요**: test 분할을 유지할지(보수적, 환경 변화에 대한 강건성까지 평가), 바꿀지(예: 방문마다 골고루 뽑기).

**주의**: 분할 1개, 시드 1개 결과다. test 1,900개 기준 ±1%p는 오차 범위이므로 비슷한 순위끼리의 차이는 의미가 없다.

## 변경 기록

- **2026-10-12**: 최초 작성. 피처 추출과 평가.
- **2026-10-13**: 교수님 노트북과 규칙 통일.
  - 공용 로더 `load_data.py` 추가.
  - 첫 셀 `0.0`의 방향을 **송신 → 수신**으로 변경. 이전 코드는 `+0.0`을 송신으로 처리해 노트북과 달랐다.
  - 그 결과 모든 기록에서 `n_out`이 1 감소, `n_in`이 1 증가했다. 방향을 쓰는 피처는 조금씩 바뀌었고 시간만 쓰는 피처는 그대로다. 정확도 변화는 ±0.3%p 이내다.
  - 피처 CSV와 로그를 모두 다시 생성했다.
- **2026-10-15**: 랜덤 포레스트 피처 중요도 분석(⑥) 추가. `requirements.txt` 추가.
- **2026-10-15**: 직접 실험용 `experiment.py`(⑦) 추가. 마지막 방문(inst 19)의 수집 상태가 다르다는 것을 확인했다.
- **2026-10-15**: 단계별 피처 실험(⑧⑨, [EXPERIMENTS.md](EXPERIMENTS.md)) 추가. 2단계 결과로 상수 피처 3개 제거를 채택했다.
- **2026-10-15**: 4단계 실험(⑧-2) 추가. 처음 50셀의 방향값(@dir50)을 채택해 후보 피처셋을 131개로 바꿨다. 31~100번째 버스트 길이(@bseqlong)는 기각했다.

## 다음에 할 일 (전처리 담당)

- [ ] 그래프: mon/unmon 길이 분포, 지속 시간 분포, 사이트별 길이 분포
- [ ] 시퀀스 모델(DF 등)용 입력: 고정 길이(예: 5,000)로 자르기/0 채우기. 잘리는 기록 비율도 확인
- [ ] open-world용 데이터: unmon 라벨(예: -1) 결정과 train/test 분할
- [x] 피처 정리: 상수 피처(`in_burst_1~3`) 제거 채택. 길이 계열 축소와 시간 그룹 제거는 모델에 따라 손해라 기각 ([EXPERIMENTS.md](EXPERIMENTS.md) 2단계)
- [x] 앞부분 피처 추가 → `features_extra.py`. 후보 피처셋은 기존 52개 + @early + @bseq (86개), CV RF 95.5% ([EXPERIMENTS.md](EXPERIMENTS.md) 3단계)
- [x] train 내부 교차검증으로 피처 선택 (test는 마지막 1회만) → `experiment.py`, `run_stages.py`
- [x] 앞부분 패턴 확장 → `features_extra2.py`. @dir50 채택, @bseqlong 기각. 후보 피처셋 131개 ([EXPERIMENTS.md](EXPERIMENTS.md) 4단계)
- [ ] 후보 피처셋으로 모델 튜닝 ([EXPERIMENTS.md](EXPERIMENTS.md) 5단계)
- [ ] test 분할 유지 여부 결정 (inst 19 문제)
