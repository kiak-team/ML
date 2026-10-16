"""피처셋 × 모델 후보군 비교 (proposal용).

사용법: python -I -X utf8 code/run_model_stages.py a > logs/stage6a.log   (약 40분)
      python -I -X utf8 code/run_model_stages.py b > logs/stage6b.log   (약 10분)
  a  피처셋 6개 × 모델 5개, 기본 하이퍼파라미터
  b  a단계 상위 3개를 시드 3개로 반복해 차이가 재현되는지 확인 (튜닝 아님)

피처셋 6개 × 모델 5개를 **기본 하이퍼파라미터로** 비교한다. 4단계가 비교하지 않은
조합(@cumulabs와 @dir50을 동시에 넣은 것)을 메우는 것이 목적이다.

**하이퍼파라미터 튜닝은 하지 않는다.** 튜닝과 최종 모델 선택은 6단계(모델 선정 담당)다.
이 표는 "proposal에 후보로 올릴 피처셋·모델"을 고르기 위한 것이고, 모든 모델이
기본값이므로 **모델 순위로 읽으면 안 된다**(1단계부터 지켜 온 규칙).

평가는 train(inst 0~17) 내부 5-fold CV다. test는 쓰지 않는다.
비교 기준: 4b단계에서 시드 변동폭이 최대 0.09%p로 측정됐다. fold 표준편차(약 0.4%p)
근거의 1%p 규칙은 짝지은 비교에는 쓰지 않는다.

피처셋 정의는 아래 SETS에 있다. 그룹 정의는 4b단계 이후(준상수 13개 제외)다.
"""
import os, sys, warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment import run

B = ['@all', '-@const']                       # 52
C = B + ['@early', '@bseq']                   # 82

SETS = {
    'B': B,                                   # 52  기존 피처만
    'C': C,                                   # 82  3b단계 후보
    'C+cumulabs': C + ['@cumulabs'],          # 95  RF에서 가장 높았다 (4b단계)
    'C+dir50': C + ['@dir50'],                # 118 현재 공식 후보 (4단계)
    'C+cumulabs+dir50': C + ['@cumulabs', '@dir50'],   # 131 미검증 조합
    'C+cumulabs+dir': C + ['@cumulabs', '@dir'],       # 181 앞 100셀까지
}


def go(spec, model, note, **params):
    try:
        r = run(spec, model=model, note=note, **params)
        p = ' '.join(f'{k}={v}' for k, v in params.items())
        print(f'RESULT\t{note}\t{model}\t{p}\t{len(r["features"])}\t'
              f'{r["cv_mean"]:.4f}\t{r["cv_std"]:.4f}\t{r["sec"]:.0f}s', flush=True)
    except Exception as e:
        print(f'FAIL\t{note}\t{model}\t{e!r}', flush=True)


def stage_a():
    # 빠른 모델부터 돌려 결과를 먼저 본다
    for models in (['knn', 'et', 'rf'], ['svm'], ['lr']):
        for name, spec in SETS.items():
            for m in models:
                go(spec, m, f'S6a {name}')


def stage_b():
    """a단계 상위 3개를 시드 3개로 반복한다. 튜닝이 아니라 '차이가 재현되는지' 확인이다.

    a단계에서 피처셋 간 차이가 0.2~0.4%p로 나왔다. 4b단계에서 측정한 시드 변동폭은
    최대 0.09%p이므로 재현되어야 한다. k-NN과 SVM은 시드에 의존하지 않아 1회만 돌린다.
    """
    for name in ('C+dir50', 'C+cumulabs+dir50', 'C+cumulabs+dir'):
        for m in ('rf', 'et'):
            for seed in (0, 1, 2):
                go(SETS[name], m, f'S6b {name} seed{seed}', random_state=seed)


if __name__ == '__main__':
    {'a': stage_a, 'b': stage_b}[sys.argv[1]]()
    print('DONE', flush=True)
