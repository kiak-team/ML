"""개별 피처 점검 (4b단계). 정보가 없는 피처를 걸러내고, 피처셋 비교가 재현되는지 본다.

사용법: python -I -X utf8 code/check_features.py > logs/check_features.log   (약 4분)
pkl 없이 features/*.csv만으로 돌아간다.

왜 개별 "순위"로 고르지 않는가: 0단계에서 eta^2 상위 20개가 하위 20개보다 정확도가 낮았다
(9.7% vs 15.3%). 상위권이 전부 "기록 길이"라는 같은 정보였기 때문이다. 그래서 이 스크립트는
피처를 줄 세우지 않고 "값이 거의 변하지 않아 정보가 아예 없는 것"만 걸러낸다.

점검 3개
  [1] mon train 기준 준상수 탐색 (최빈값 비율 >= 0.99 또는 표준편차 0) + train-only eta^2
  [2] 준상수를 빼면 정확도가 변하는지 (k-NN, RF)
  [3] 시드 3개 반복으로 피처셋 간 차이가 노이즈인지 재현되는지
  [4] 준상수가 unmon에서도 상수인지 (open-world 영향)

결과 해석은 docs/EXPERIMENTS.md 4b단계에 있다.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I 에서도 experiment를 찾도록
from experiment import ROOT, groups, load, run, select

MODE_THRESHOLD = 0.99  # 한 값이 이 비율 이상이면 준상수로 본다 (임계값은 임의로 정했다)


def eta2(col, y):
    """피처 하나의 사이트 구분력. 우연 수준은 약 0.005다."""
    gm = col.mean()
    sst = ((col - gm) ** 2).sum()
    if sst == 0:
        return 0.0
    ssb = sum(len(g) * (g.mean() - gm) ** 2 for g in (col[y == c] for c in np.unique(y)))
    return ssb / sst


def mode_frac(col):
    _, counts = np.unique(col, return_counts=True)
    return counts.max() / len(col)


def main():
    df, names = load()
    # @qconst를 포함한 4단계 당시의 후보 131개를 대상으로 점검한다
    feats = select(['@all', '-@const', '@early', '@bseq', '@dir50', '@qconst'], names)
    tr = df[df.inst <= 17]
    X, y = tr[feats].values.astype(float), tr.site.values
    print(f'[점검 대상] {len(feats)}개 피처, train {len(tr)}개 (inst 0~17)\n')

    print('=' * 70)
    print(f'[1] 준상수 탐색 (최빈값 비율 >= {MODE_THRESHOLD} 또는 표준편차 0)')
    print('=' * 70)
    mf = np.array([mode_frac(X[:, j]) for j in range(X.shape[1])])
    sd = X.std(axis=0)
    e2 = np.array([eta2(X[:, j], y) for j in range(X.shape[1])])
    hit = [j for j in range(len(feats)) if sd[j] == 0 or mf[j] >= MODE_THRESHOLD]
    print(f'{"피처":16s} {"최빈값비율":>10s} {"값종류":>7s} {"표준편차":>9s} {"eta^2":>8s}')
    for j in hit:
        print(f'{feats[j]:16s} {mf[j]:10.4f} {len(np.unique(X[:, j])):7d} {sd[j]:9.4f} {e2[j]:8.4f}')
    print(f'\n준상수 {len(hit)}개. eta^2 우연 수준은 약 0.005다.')
    print('\n[경계 근처] 최빈값 비율 0.90~0.99 → eta^2가 높으면 남긴다')
    for j in np.argsort(-mf):
        if 0.90 <= mf[j] < MODE_THRESHOLD:
            print(f'  {feats[j]:16s} 최빈값비율={mf[j]:.4f} eta^2={e2[j]:.4f}')
    qconst = [feats[j] for j in hit]

    print('\n' + '=' * 70)
    print('[2] 준상수를 빼면 정확도가 변하는가')
    print('=' * 70)
    with_q = ['@all', '-@const', '@early', '@bseq', '@dir50', '@qconst']
    without_q = ['@all', '-@const', '@early', '@bseq', '@dir50']
    for model in ('knn', 'rf'):
        for spec, tag in ((with_q, '준상수 포함'), (without_q, '준상수 제외')):
            r = run(spec, model=model, note=f'4b [2] {tag}', save=False)
            print(f'  -> {tag}: {len(r["features"])}개  CV {r["cv_mean"]:.4f} +- {r["cv_std"]:.4f}  {r["sec"]:.0f}s\n')

    print('=' * 70)
    print('[3] 시드 3개 반복: 피처셋 간 차이가 재현되는가 (RF)')
    print('=' * 70)
    B = ['@all', '-@const']
    sets = {
        'C (early+bseq)': B + ['@early', '@bseq'],
        'C + cumulabs': B + ['@early', '@bseq', '@cumulabs'],
        'C + dir50': B + ['@early', '@bseq', '@dir50'],
    }
    table = {}
    for name, spec in sets.items():
        row = []
        for seed in (0, 1, 2):
            r = run(spec, model='rf', random_state=seed, note=f'4b [3] {name} seed{seed}', save=False)
            row.append(r['cv_mean'])
        table[name] = (len(r['features']), row)
        print()
    print(f'{"조합":18s} {"피처수":>6s} {"seed0":>8s} {"seed1":>8s} {"seed2":>8s} {"평균":>8s}')
    for name, (n, row) in table.items():
        print(f'{name:18s} {n:6d} ' + ' '.join(f'{v:8.4f}' for v in row) + f' {np.mean(row):8.4f}')
    spread = max(max(r) - min(r) for _, r in table.values())
    print(f'\n시드 변동폭(최대) = {spread * 100:.2f}%p. 이보다 큰 차이는 재현된다고 본다.')

    print('\n' + '=' * 70)
    print('[4] 준상수가 unmon에서도 상수인가 (open-world 영향)')
    print('=' * 70)
    unm = pd.concat([pd.read_csv(os.path.join(ROOT, 'features', f)) for f in
                     ('features_unmon.csv', 'features_extra_unmon.csv', 'features_extra2_unmon.csv')], axis=1)
    print(f'{"피처":16s} {"mon최빈비율":>12s} {"unmon최빈비율":>14s} {"mon평균":>10s} {"unmon평균":>11s}')
    for f in qconst:
        a, b = tr[f].values.astype(float), unm[f].values.astype(float)
        flag = '  <- unmon에서 변함' if mode_frac(b) < MODE_THRESHOLD else ''
        print(f'{f:16s} {mode_frac(a):12.4f} {mode_frac(b):14.4f} {a.mean():10.4f} {b.mean():11.4f}{flag}')

    Xq = np.vstack([tr[qconst].values.astype(float), unm[qconst].values.astype(float)])
    yq = np.r_[np.zeros(len(tr)), np.ones(len(unm))]
    auc = cross_val_score(RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0),
                          Xq, yq, cv=3, scoring='roc_auc').mean()
    print(f'\n준상수 {len(qconst)}개만으로 mon/unmon 분류: AUC = {auc:.4f} (0.5 = 무정보)')
    print('사이트 구분력이 우연 수준인 피처가 mon/unmon을 거의 완벽히 가른다.')
    print('수집 환경 차이로 보이지만 수집 코드가 없어 확정할 수 없다.')
    print('→ closed-world에서는 뺀다. open-world에서 쓸지는 따로 판단한다 (@qconst).')


if __name__ == '__main__':
    main()
