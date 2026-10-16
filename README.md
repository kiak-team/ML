# Kiak!

Tor 웹사이트 핑거프린팅(WF) 프로젝트

> 프로젝트 전체 소개는 추후 작성 예정이다.

## 문서

- [docs/WORK_SUMMARY.md](docs/WORK_SUMMARY.md): **데이터 전처리 담당 작업 전체 정리** (여기부터 읽기)
- [docs/SCORES.md](docs/SCORES.md): **지금 점수가 무슨 뜻인지 쉽게 정리** (CV vs test, 현재 점수, 최종 점수 내는 순서)
- [docs/PREPROCESSING.md](docs/PREPROCESSING.md): 데이터 전처리 작업 정리 (폴더 구조, 파일별 역할, 실행 방법, 결과, 변경 기록, 다음에 할 일)
- [docs/DATA.md](docs/DATA.md): 데이터 설명
- [docs/FEATURES.md](docs/FEATURES.md): 피처 55개 설명
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md): 피처 선택 실험 기록 (단계별 가설·결과·결정)

## 현재 후보 피처셋

`@all -@const @early @bseq @cumulabs @dir50` = **131개** (closed-world, 기본 하이퍼파라미터 CV: RF 95.7%, ET 95.8%, SVM 94.9%, k-NN 90.4%).
6단계(10/16)에서 이전 후보 118개에 `@cumulabs` 13개를 더했다. 5개 모델 전부에서 118개보다 높았다 → EXPERIMENTS.md 6단계.

> 개수가 4단계의 131개와 같지만 구성이 다르다. 4단계 131개 = 118개 + 준상수 13개(4b단계에서 제거),
> 지금 131개 = 118개 + `@cumulabs` 13개. 겹치는 피처가 없다.

k-NN·SVM은 그룹 가중치(`experiment.py --gw`)를 켜는 것을 제안한다(LR은 손해라 끈다) — EXPERIMENTS.md 6c단계.

제안서용 그래프는 `figures/`에 있다(목록: PREPROCESSING.md "그래프" 절). 시퀀스 모델 입력은 `code/sequences.py`로 만든다.

open-world는 피처셋이 다르다 (`@all @early @bseq @dir50 @qconst`, 앞 14셀 버림, 134개) — docs/OPENWORLD.md.
