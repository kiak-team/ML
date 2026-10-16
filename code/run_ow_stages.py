"""open-world 단계별 피처 실험 배치. closed-world의 run_stages.py와 같은 방식이다.

사용법: python -I -X utf8 code/run_ow_stages.py <단계> [svm|nosvm] | tee logs/ow_stage<단계>.log
  단계에 1,2,3처럼 여러 개를 쉼표로 줄 수 있다. svm = SVM만, nosvm = SVM 빼고.
  SVM은 한 번에 약 9분 걸려서(unmon 9,000개가 한 클래스) 나머지와 따로 동시에 돌리면 빠르다.
  1   기준점: 기본 55개 전체 × 모델 5개
  2   빼도 되는 피처가 있나 (시간 그룹, 길이 계열)
  3   새 피처를 더하면 좋아지나 (closed-world 3·4단계의 추가 그룹)
  3b  3단계 결과로 고른 조합

공통: features_ow.py로 앞 14셀을 버린 피처(--skip 14), 96클래스 방식(사이트 95개 + unmon),
      train 내부 5-fold CV(test는 쓰지 않는다). 모델은 기본 하이퍼파라미터.
주 지표는 ROC-AUC, 보조 지표는 FPR 1%일 때의 TPR이다.
판단 기준(실험 전에 정함): 기준 조합 대비 ROC-AUC 차이가 fold 간 표준편차 이내면 같은 수준으로 보고
피처가 적은 쪽을 고른다. 각 단계의 해석은 docs/OPENWORLD.md에 있다.
"""
import os, sys, warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from openworld import run

SKIP = 14
B = ['@all']  # 앞 14셀을 버리면 상수 피처가 없다
MODELS = (['knn', 'et', 'rf'], ['lr'], ['svm'])  # 빠른 모델부터


def go(spec, model, note):
    try:
        r = run(spec, model=model, mode='multi', skip=SKIP)
        fpr1 = dict(r['rows'])['FPR≤1%']
        print(f'RESULT\t{note}\t{model}\t{len(r["features"])}\tAUC {r["ROC-AUC"]:.4f}\tstd {r["auc_std"]:.4f}\t'
              f'PR-AUC {r["PR-AUC"]:.4f}\tTPR@1% {fpr1["TPR"]:.4f}\tTPRsite@1% {fpr1["TPR_site"]:.4f}', flush=True)
    except Exception as e:
        print(f'FAIL\t{note}\t{model}\t{e!r}', flush=True)


def stage(sets):
    for models in MODELS:
        for name, spec in sets.items():
            for m in models:
                go(spec, m, name)


def stage1():
    stage({'OW1 B (기본 55개)': B})


def stage2():
    stage({
        'OW2 B-timing': B + ['-@timing'],
        'OW2 B length->n_total': B + ['-@length', 'n_total'],
    })


def stage3():
    stage({
        'OW3 B+early': B + ['@early'],
        'OW3 B+cumulabs': B + ['@cumulabs'],
        'OW3 B+bseq': B + ['@bseq'],
        'OW3 B+cumul100': B + ['@cumul100'],
        'OW3 B+dir50': B + ['@dir50'],
        'OW3 B+bseqlong': B + ['@bseqlong'],
    })


def stage3b():
    # 3단계: @early·@bseq·@cumulabs·@dir50은 RF·ET·LR·SVM에서 올랐고, @cumul100·@bseqlong은 떨어졌다
    stage({
        'OW3b B+early+bseq': B + ['@early', '@bseq'],
        'OW3b B+early+bseq+dir50': B + ['@early', '@bseq', '@dir50'],
        'OW3b B+early+bseq+dir50+cumulabs': B + ['@early', '@bseq', '@dir50', '@cumulabs'],
        # closed-world 후보 131개와 같은 조합 (in_burst_1~3 제외). 같으면 두 실험에 한 피처셋을 쓸 수 있다
        'OW3b closed-world 131': ['@all', '-@const', '@early', '@bseq', '@dir50'],
    })


def stage3c():
    # @qconst 13개(first10_out, bseq_4~6, dir_006~014)는 mon 원본에서 앞 14셀 흔적 때문에 준상수라 그룹에서 빠져 있다.
    # 앞 14셀을 버린 피처에서는 같은 이름이 15번째 셀 이후를 가리켜 상수가 아니므로 다시 넣어 본다
    stage({
        'OW3c B+early+bseq+qconst': B + ['@early', '@bseq', 'first10_out', 'bseq_4', 'bseq_5', 'bseq_6'],
        'OW3c B+early+bseq+dir50+qconst': B + ['@early', '@bseq', '@dir50', '@qconst'],
    })


if __name__ == '__main__':
    only = sys.argv[2] if len(sys.argv) > 2 else ''
    if only == 'svm':
        MODELS = (['svm'],)
    elif only == 'nosvm':
        MODELS = MODELS[:-1]
    for st in sys.argv[1].split(','):
        {'1': stage1, '2': stage2, '3': stage3, '3b': stage3b, '3c': stage3c}[st]()
    print('DONE', flush=True)
