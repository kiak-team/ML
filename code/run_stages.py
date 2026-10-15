"""단계별 피처 실험 배치. 보고서의 표는 이 스크립트와 logs/stage*.log로 다시 만들 수 있다.

사용법: python -I -X utf8 code/run_stages.py <단계 번호> | tee logs/stage<번호>.log
  1  기준점: 기존 55개 전체 × 모델 8개 (약 25분, mlp가 대부분)
  2  피처 정리 가설 검증 (약 30분, mlp 제외)
  3  추가 피처 가설 검증. features_extra.py를 먼저 실행해야 한다 (약 1시간)
  3b 3단계 후속: @cumul100 없이도 같은 수준인지 (약 10분)

모든 평가는 train 내부 5-fold CV(experiment.py)다. test는 쓰지 않는다.
판단 기준(실험 전에 정함): 기준 조합 대비 차이가 1%p(≈ fold 표준편차) 이내면 같은 수준으로 보고 피처가 적은 쪽을 고른다.
각 단계의 가설과 결과 해석은 docs/EXPERIMENTS.md에 있다.
"""
import os, sys, warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment import run

B = ['@all', '-@const']  # 2단계에서 채택한 기준 조합 (52개)


def go(spec, model, note):
    try:
        r = run(spec, model=model, note=note)
        print(f'RESULT\t{note}\t{model}\t{len(r["features"])}\t{r["cv_mean"]:.4f}\t{r["cv_std"]:.4f}\t{r["sec"]:.0f}s', flush=True)
    except Exception as e:
        print(f'FAIL\t{note}\t{model}\t{e!r}', flush=True)


def stage1():
    for m in ['nb', 'lr', 'dt', 'knn', 'rf', 'et', 'svm', 'mlp']:
        go(['@all'], m, 'S1 baseline')


def stage2():
    # mlp는 한 번에 10분~1시간 이상 걸려 2단계에서 제외했다 (B만 실행됨)
    sets = {
        'S2 B: -const': B,
        'S2 C: -const -timing': B + ['-@timing'],
        'S2 D: -const length->n_total': B + ['-@length', 'n_total'],
        'S2 E: -const -timing length->n_total': B + ['-@timing', '-@length', 'n_total'],
    }
    for name, spec in sets.items():
        for m in ['lr', 'knn', 'rf', 'et', 'svm']:
            go(spec, m, name)


def stage3():
    # 비교 기준은 2단계의 'S2 B: -const' 결과다
    sets = {
        'S3 B+early': B + ['@early'],
        'S3 B+cumulabs': B + ['@cumulabs'],
        'S3 B+bseq': B + ['@bseq'],
        'S3 B+cumul100': B + ['@cumul100'],
        'S3 B+extra': B + ['@extra'],
        # CUMUL 논문 재현에 가까운 조합: 송수신 개수 + 보간 누적합 100개
        'S3 CUMUL-like': ['n_in', 'n_out', '@cumul100'],
    }
    # 빠른 모델부터 돌려 결과를 먼저 본다
    for models in (['knn', 'et', 'rf'], ['svm'], ['lr']):
        for name, spec in sets.items():
            for m in models:
                go(spec, m, name)


def stage3b():
    # 3단계 결과: @cumul100은 단독으로 효과가 거의 없었다 → 빼도 @extra와 같은 수준인지 확인
    sets = {
        'S3b B+early+bseq': B + ['@early', '@bseq'],
        'S3b B+early+bseq+cumulabs': B + ['@early', '@bseq', '@cumulabs'],
    }
    for models in (['knn', 'et', 'rf'], ['svm'], ['lr']):
        for name, spec in sets.items():
            for m in models:
                go(spec, m, name)


if __name__ == '__main__':
    {'1': stage1, '2': stage2, '3': stage3, '3b': stage3b}[sys.argv[1]]()
    print('DONE', flush=True)
