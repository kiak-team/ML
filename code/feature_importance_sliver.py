"""Extra credit 피처 선택: TrafficSliver split 데이터로 피처 중요도를 train 안에서만 다시 잰다 (sklearn 필요).

사용법: python -I -W ignore -X utf8 code/feature_importance_sliver.py features/sliver_mon.csv > logs/feature_importance_sliver.log

방식 (feature_importance_cv.py와 같고 다른 점만 적는다)
- test(ld.sliver_is_test, 방문의 10%)는 쓰지 않는다. 나머지 방문을 group 단위 5-fold로 나눈다.
  같은 방문의 split 5개와 join이 서로 다른 fold에 들어가지 않는다 (docs/EXTRA_JOIN.md).
- 피처 하나씩 섞는 permutation은 375개 × 9만 행이라 너무 느려서 뺐다. 대신 그룹·상관 묶음 단위로 섞는다.
- 기준선 (fold 평균 RF 정확도, 95-class closed-world):
    A join→join (방어 없음)   B split→split (공격자가 실제로 얻는 값)   C join→split
출력: 위 로그 + logs/feature_importance_sliver.csv (피처별 RF 불순도 중요도, train-only η²)
"""
import os, sys, time

import numpy as np, pandas as pd
from scipy.cluster import hierarchy
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment as ex
import load_data as ld
from feature_importance_cv import shuffled_drop
from features import eta2

N_TREES = 200


def rf():
    return RandomForestClassifier(n_estimators=N_TREES, n_jobs=-1, random_state=0)


def main(path):
    df = pd.read_csv(path)
    df = df[~ld.sliver_is_test(df.group.values)].reset_index(drop=True)
    names = [c for c in df.columns if c not in ('label', 'group', 'part', 'n_bursts_total')]
    sp, jn = df[df.part < ld.JOIN_PART], df[df.part == ld.JOIN_PART]
    X, y, g = sp[names].values, sp.label.values, sp.group.values
    Xj, yj, gj = jn[names].values, jn.label.values, jn.group.values
    print(f'train split {len(sp)}행 / join {len(jn)}행, 피처 {len(names)}개, 사이트 {len(set(y))}개', flush=True)

    G = ex.groups(names)
    named = {k: [names.index(f) for f in v] for k, v in G.items() if k not in ('all', 'extra') and v}

    corr = spearmanr(X).correlation
    corr = np.nan_to_num((corr + corr.T) / 2, nan=0.0); np.fill_diagonal(corr, 1)
    cl = hierarchy.fcluster(hierarchy.ward(1 - np.abs(corr)), 0.3, criterion='distance')
    clusters = {}
    for i, c in enumerate(cl):
        clusters.setdefault(c, []).append(i)

    rng = np.random.default_rng(0)
    A, B, C, imp = [], [], [], []
    grp = {k: [] for k in named}; clu = {c: [] for c in clusters}
    t0 = time.time()
    # 같은 group 번호로 split과 join을 함께 나눈다
    ug = np.unique(g)
    for k, (tri, vai) in enumerate(GroupKFold(5).split(ug, groups=ug)):
        tr_g, va_g = ug[tri], ug[vai]
        mtr, mva = np.isin(g, tr_g), np.isin(g, va_g)
        mjtr, mjva = np.isin(gj, tr_g), np.isin(gj, va_g)
        m = rf().fit(X[mtr], y[mtr])
        Xv, yv = X[mva], y[mva]
        base = m.score(Xv, yv); B.append(base); imp.append(m.feature_importances_)
        mj = rf().fit(Xj[mjtr], yj[mjtr])
        A.append(mj.score(Xj[mjva], yj[mjva])); C.append(mj.score(Xv, yv))
        for name, cols in named.items():
            grp[name].append(shuffled_drop(m, Xv, yv, cols, base, rng))
        for c, cols in clusters.items():
            clu[c].append(shuffled_drop(m, Xv, yv, cols, base, rng))
        print(f'fold {k}: A {A[-1]:.4f}  B {base:.4f}  C {C[-1]:.4f}  ({time.time() - t0:.0f}s)', flush=True)

    imp = np.array(imp).mean(0)
    e2 = [eta2(X[:, j].tolist(), y.tolist()) for j in range(len(names))]
    res = pd.DataFrame({'feature': names, 'impurity': imp, 'eta2': e2}).set_index('feature')
    res.sort_values('impurity', ascending=False).to_csv(
        os.path.join(ex.ROOT, 'logs', 'feature_importance_sliver.csv'), float_format='%.5f')

    print(f'\n== 기준선 RF({N_TREES}) 5-fold, 95-class ==')
    for tag, v in (('A join→join (방어 없음)', A), ('B split→split (방어 있음)', B), ('C join→split', C)):
        print(f'  {tag:26s} {np.mean(v):.4f} ± {np.std(v):.4f}')
    print('\n== RF 불순도 중요도 상위 25 (split 학습, fold 평균) / η² ==')
    for f, r in res.sort_values('impurity', ascending=False).head(25).iterrows():
        print(f'  {f:16s} {r.impurity:.4f}  η²={r.eta2:.3f}')
    print('\n== 그룹 permutation (그룹 전체를 섞었을 때 B 정확도 하락, fold 평균) ==')
    for name, v in sorted(grp.items(), key=lambda kv: -np.mean(kv[1])):
        print(f'  @{name:9s} ({len(named[name]):3d}개)  {np.mean(v):+.4f} ± {np.std(v):.4f}')
    print(f'\n== 상관 묶음 permutation 상위 20 ({len(clusters)}개 묶음) ==')
    for c, v in sorted(clu.items(), key=lambda kv: -np.mean(kv[1]))[:20]:
        fs = [names[i] for i in clusters[c]]
        print(f'  {np.mean(v):+.4f}  ({len(fs):3d}) {", ".join(fs[:8])}{" ..." if len(fs) > 8 else ""}')
    print('DONE', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
