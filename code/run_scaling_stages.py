"""6c단계: k-NN·SVM·LR용 전처리 점검 (sklearn 필요).

사용법: python -I -X utf8 code/run_scaling_stages.py knn svm > logs/stage6c.log   (약 15분)
      python -I -X utf8 code/run_scaling_stages.py lr > logs/stage6c_lr.log        (약 1시간)

질문: 피처가 131개로 늘었다. 거리(k-NN)·커널(SVM)·선형 결합(LR)에는 모든 피처가 똑같이 들어가므로,
지금의 전처리(sign·log 변환 후 표준화)만으로 충분한가? 아니면 피처 가중치나 선택이 필요한가?

비교하는 전처리 (피처셋은 후보 131개로 고정, 모델은 기본 하이퍼파라미터):
  std          표준화만 (log 없음)
  log+std      sign(x)·log(1+|x|) 후 표준화 — 지금 experiment.py 기본값
  quantile     순위를 정규분포로 바꾼다(QuantileTransformer). 치우친 분포·이상치에 가장 강하다
  log+std+gw   log+std 후 그룹마다 1/√(그룹 크기)를 곱한다. @dir50(36개)처럼 큰 그룹이
               거리를 독차지하지 않도록 그룹마다 비중을 맞춘다
  log+std+k60  log+std 후 ANOVA F값 상위 60개만 쓴다(SelectKBest). fold마다 학습 데이터로만 고른다
  log+std+k90  같은 방식으로 상위 90개
  log+std+pca  log+std 후 PCA로 분산 99%만 남긴다. 서로 겹치는(상관 높은) 피처를 합친다

평가: train(inst 0~17) 내부 5-fold CV. experiment.py와 같은 분할이다. test는 쓰지 않는다.
변환기(표준화, 분위수, F값 선택, PCA)는 모두 파이프라인 안에서 fold의 학습 부분으로만 맞춘다.
"""
import os, sys, time, warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import PredefinedSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, QuantileTransformer, StandardScaler

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment as ex

CAND = ['@all', '-@const', '@early', '@bseq', '@cumulabs', '@dir50']  # closed-world 후보 131개


group_weights = ex.group_weights  # 기존 52개는 소그룹(@count 등), 추가 피처는 @early 등 그룹 단위


def pipes(feats, names):
    w = group_weights(feats, names)
    return {
        'std': lambda: [StandardScaler()],
        'log+std': lambda: [FunctionTransformer(ex.signed_log), StandardScaler()],
        'quantile': lambda: [QuantileTransformer(output_distribution='normal', random_state=0)],
        'log+std+gw': lambda: [FunctionTransformer(ex.signed_log), StandardScaler(), ex.GroupWeight(w)],
        'log+std+k60': lambda: [FunctionTransformer(ex.signed_log), StandardScaler(), SelectKBest(f_classif, k=60)],
        'log+std+k90': lambda: [FunctionTransformer(ex.signed_log), StandardScaler(), SelectKBest(f_classif, k=90)],
        'log+std+pca': lambda: [FunctionTransformer(ex.signed_log), StandardScaler(),
                                PCA(n_components=0.99, svd_solver='full', random_state=0)],
    }


# LR은 1회 15~20분이라 지금 기준과 대안 중 가능성이 있는 것만 돌린다
LR_VARIANTS = ['log+std', 'quantile', 'log+std+gw', 'log+std+k90']


def main(models):
    df, names = ex.load()
    feats = ex.select(CAND, names)
    tr = df[df.inst <= 17]
    X, y, cv = tr[feats].values, tr.site.values, PredefinedSplit(tr.inst.values % 5)
    P = pipes(feats, names)
    print(f'features={len(feats)}  train={len(y)}  그룹 가중치(1/√크기): '
          + ' '.join(sorted({f'{v:.2f}' for v in group_weights(feats, names)})), flush=True)
    for m in models:
        for name, steps in P.items():
            if m == 'lr' and name not in LR_VARIANTS:
                continue
            pipe = make_pipeline(*steps(), ex.MODELS[m][0]())
            t0 = time.time()
            s = cross_val_score(pipe, X, y, cv=cv, n_jobs=5 if m != 'lr' else 1)
            print(f'RESULT\t{m}\t{name}\t{s.mean():.4f}\t{s.std():.4f}\t{time.time() - t0:.0f}s\t'
                  f'[{" ".join(f"{a:.3f}" for a in s)}]', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:] or ['knn', 'svm'])
