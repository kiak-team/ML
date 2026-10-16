"""피처 중요도를 train 내부에서만 다시 계산한다 (sklearn 필요). feature_importance.py(0단계)의 누수 없는 버전.

사용법: python -I -X utf8 code/feature_importance_cv.py cand > logs/feature_importance_cv_cand.log   (약 10분)
      python -I -X utf8 code/feature_importance_cv.py base > logs/feature_importance_cv_base.log   (약 5분)
  cand  closed-world 후보 131개 (@all -@const @early @bseq @cumulabs @dir50)
  base  0단계와 같은 55개 (@all). 0단계(test 기준) 순위와 비교한다

0단계와 다른 점:
  0단계는 permutation 중요도를 test(inst 18~19)에서 쟀다. test를 보고 피처를 판단하면 누수가 된다.
  여기서는 train(inst 0~17) 안에서 experiment.py와 같은 5-fold(inst % 5)로 나눠
  fold마다 나머지 4개로 RF를 학습하고, 빼 둔 fold에서 피처를 섞어 정확도 하락을 잰다. 5개 fold 평균이다.

출력:
  1. fold별 RF 정확도 (CV)
  2. RF 불순도 중요도 상위 20 (fold 평균)
  3. permutation 중요도 상위 20 (빼 둔 fold에서 섞었을 때 정확도 하락, fold 평균 ± 표준편차)
  4. 그룹 permutation: 그룹 전체를 함께 섞었을 때 하락 (서로 대신하는 피처가 있어도 그룹의 몫이 보인다)
  5. 상관 묶음 permutation: 0단계와 같은 방식(|spearman| 계층 군집)
  6. (base만) 0단계 test 기준 순위와 비교 — 순위 상관, 상위 15개 겹침
  CSV: logs/feature_importance_cv_<tag>.csv (피처별 불순도·permutation 평균·표준편차). plots.py가 읽는다
"""
import os, sys, time, warnings

import numpy as np, pandas as pd
from scipy.cluster import hierarchy
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment as ex

SPECS = {'cand': ['@all', '-@const', '@early', '@bseq', '@cumulabs', '@dir50'], 'base': ['@all']}
GROUPS = ['count', 'timing', 'order', 'burst', 'cumul', 'early', 'bseq', 'cumulabs', 'dir50', 'const']
REPEATS = 3

# 0단계(logs/feature_importance.log) test 기준 permutation 상위 15개. base 비교용
STAGE0_TOP15 = ['first30_out', 'first100_out', 'cumul_at_500', 'frac_out', 'in_burst_mean', 'out_burst_mean',
                'n_out', 'cumul_10pct', 'n_total', 'conc_std', 'out_order_std', 'out_burst_std', 'in_order_mean',
                'cumul_20pct', 'cumul_80pct']


def shuffled_drop(rf, X, y, cols, base, rng):
    drops = []
    for _ in range(REPEATS):
        Xp = X.copy()
        Xp[:, cols] = Xp[rng.permutation(len(Xp))][:, cols]
        drops.append(base - rf.score(Xp, y))
    return np.mean(drops)


def main(tag):
    df, names = ex.load()
    feats = ex.select(SPECS[tag], names)
    tr = df[df.inst <= 17]
    X, y, fold = tr[feats].values, tr.site.values, tr.inst.values % 5
    g = ex.groups(names)
    named = {k: [feats.index(f) for f in g[k] if f in feats] for k in GROUPS}
    named = {k: v for k, v in named.items() if v}

    # 상관 묶음 (0단계와 같은 기준: 1-|spearman|, ward, 거리 0.3). train 전체로 한 번만 정한다
    corr = spearmanr(X).correlation
    corr = np.nan_to_num((corr + corr.T) / 2, nan=0.0); np.fill_diagonal(corr, 1)
    cl = hierarchy.fcluster(hierarchy.ward(1 - np.abs(corr)), 0.3, criterion='distance')
    clusters = {}
    for i, c in enumerate(cl):
        clusters.setdefault(c, []).append(i)

    rng = np.random.default_rng(0)
    acc, imp, perm = [], [], []
    grp = {k: [] for k in named}
    clu = {c: [] for c in clusters}
    t0 = time.time()
    for k in range(5):
        rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0).fit(X[fold != k], y[fold != k])
        Xv, yv = X[fold == k], y[fold == k]
        base = rf.score(Xv, yv)
        acc.append(base); imp.append(rf.feature_importances_)
        perm.append([shuffled_drop(rf, Xv, yv, [j], base, rng) for j in range(len(feats))])
        for name, cols in named.items():
            grp[name].append(shuffled_drop(rf, Xv, yv, cols, base, rng))
        for c, cols in clusters.items():
            clu[c].append(shuffled_drop(rf, Xv, yv, cols, base, rng))
        print(f'fold {k}: acc {base:.4f}  ({time.time() - t0:.0f}s)', flush=True)

    imp, perm = np.array(imp), np.array(perm)
    res = pd.DataFrame({'feature': feats, 'impurity': imp.mean(0), 'perm_mean': perm.mean(0),
                        'perm_std': perm.std(0)}).set_index('feature')
    res.sort_values('perm_mean', ascending=False).to_csv(
        os.path.join(ex.ROOT, 'logs', f'feature_importance_cv_{tag}.csv'), float_format='%.5f')

    print(f'\n[{tag}] features={len(feats)}  RF(300) CV acc = {np.mean(acc):.4f} ± {np.std(acc):.4f}')
    print('\n== RF 불순도 중요도 상위 20 (fold 평균) ==')
    for f, v in res.impurity.sort_values(ascending=False).head(20).items():
        print(f'  {f:16s} {v:.4f}')
    print('\n== permutation 중요도 상위 20 (빼 둔 fold에서 섞었을 때 정확도 하락, fold 평균 ± 표준편차) ==')
    for f, r in res.sort_values('perm_mean', ascending=False).head(20).iterrows():
        print(f'  {f:16s} {r.perm_mean:+.4f} ± {r.perm_std:.4f}')
    print(f'  ... 하락이 0 이하인 피처: {(res.perm_mean <= 0).sum()}개')
    print('\n== 그룹 permutation (그룹 전체를 함께 섞음, fold 평균) ==')
    for name, v in sorted(grp.items(), key=lambda kv: -np.mean(kv[1])):
        print(f'  @{name:9s} ({len(named[name]):2d}개)  {np.mean(v):+.4f} ± {np.std(v):.4f}')
    print(f'\n== 상관 묶음 permutation 상위 15 ({len(clusters)}개 묶음) ==')
    for c, v in sorted(clu.items(), key=lambda kv: -np.mean(kv[1]))[:15]:
        fs = [feats[i] for i in clusters[c]]
        print(f'  {np.mean(v):+.4f}  ({len(fs):2d}) {", ".join(fs[:8])}{" ..." if len(fs) > 8 else ""}')

    if tag == 'base':
        top = res.sort_values('perm_mean', ascending=False).index[:15].tolist()
        rank_new = res.perm_mean.rank(ascending=False)
        rank_old = pd.Series(np.arange(1, 16), STAGE0_TOP15)
        rho = spearmanr(rank_old.values, rank_new[rank_old.index].values).correlation
        print('\n== 0단계(test 기준)와 비교 ==')
        print(f'  상위 15개 겹침: {len(set(top) & set(STAGE0_TOP15))}/15')
        print(f'  0단계 상위 15개의 순위 상관(spearman): {rho:.3f}')
        print(f'  0단계에만 있음: {", ".join(f for f in STAGE0_TOP15 if f not in top) or "없음"}')
        print(f'  여기에만 있음:  {", ".join(f for f in top if f not in STAGE0_TOP15) or "없음"}')
        print('  상위 5 (0단계 → 지금): ' + ' '.join(STAGE0_TOP15[:5]) + '  →  ' + ' '.join(top[:5]))
    print('DONE', flush=True)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'cand')
