"""open-world 피처셋 교차 확인 (proposal용).

사용법: python -I -X utf8 code/run_ow_model_stages.py feat > logs/ow_model_feat.log  (약 15분)

closed-world에서 좋았던 피처셋이 open-world에서도 좋은지 RF 기본값으로 확인한다.
모델 5개의 기본 설정 비교는 이미 docs/OPENWORLD.md 3.3에 있다.

**하이퍼파라미터 튜닝은 하지 않는다.** 튜닝과 모델 선택은 모델 선정 담당 몫이다.

규칙 (OPENWORLD.md 3.1과 같다)
  피처   앞 14셀을 버린 것 (--skip 14). 앞 14셀은 mon/unmon 수집 환경 차이가 드러난다
  방식   96클래스 (3.2에서 이진 분류보다 좋았다)
  평가   train 내부 5-fold CV의 out-of-fold 점수. test는 쓰지 않는다
  지표   주 지표 ROC-AUC, 보조 TPR@FPR1%. fold 간 표준편차는 RF 0.0007~0.0022, ET 0.0006~0.0037
"""
import os, sys, warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openworld as ow

OW = ['@all', '@early', '@bseq', '@dir50', '@qconst']  # 134개, OPENWORLD.md 3.7 후보
SKIP = 14


def go(spec, model, note, mode='multi', **params):
    try:
        r = ow.run(spec, model=model, mode=mode, skip=SKIP, **params)
        rows = dict(r['rows'])
        tpr1, tpr5 = rows['FPR≤1%'], rows['FPR≤5%']
        p = ' '.join(f'{k}={v}' for k, v in params.items())
        print(f'RESULT\t{note}\t{model}\t{p}\t{len(r["features"])}\t'
              f'{r["ROC-AUC"]:.4f}\t{r.get("auc_std", 0):.4f}\t{r["PR-AUC"]:.4f}\t'
              f'{tpr1["TPR"]:.4f}\t{tpr1.get("TPR_site", float("nan")):.4f}\t{tpr1["precision"]:.4f}\t'
              f'{tpr5["TPR"]:.4f}', flush=True)
    except Exception as e:
        print(f'FAIL\t{note}\t{model}\t{e!r}', flush=True)


def feat():
    # closed-world 6단계에서 좋았던 조합이 open-world에서도 좋은지 (RF 기본값 고정)
    sets = {
        'OW 후보 134': OW,
        'OW -qconst 121': OW + ['-@qconst'],
        'OW +cumulabs': OW + ['@cumulabs'],
        'OW +dir (앞100셀)': OW + ['-@dir50', '@dir'],
        'OW -timing': OW + ['-@timing'],
    }
    for name, spec in sets.items():
        go(spec, 'rf', f'OWf {name}')


def modes(models=('rf', 'et', 'knn', 'lr', 'svm')):
    # 이진 vs 96클래스를 후보 134개, 모델 5개(기본값)로 비교한다. 3.2는 RF와 4b 이전 131개로만 비교했다
    for model in models:
        for mode in ('binary', 'multi'):
            go(OW, model, f'OWm {mode}', mode=mode)


if __name__ == '__main__':
    # 사용법: ... modes [모델 ...]  (기본 rf et knn lr svm, 약 1시간) > logs/ow_modes.log
    stage = sys.argv[1]
    if stage == 'modes' and len(sys.argv) > 2:
        modes(sys.argv[2:])
    else:
        {'feat': feat, 'modes': modes}[stage]()
    print('DONE', flush=True)
