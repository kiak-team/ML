"""closed_world_experiments.ipynb(PR #10) 셀 8·10 재실험: 피처 버그·부스팅 설정·KMeans 누수 수정본.

  python -I -X utf8 code/rerun_cw_experiments.py        # 1~4단계 전부
  python -I -X utf8 code/rerun_cw_experiments.py 34     # 3·4단계만

수정 사항
  - 셀 10: base 피처를 df_all이 아니라 features_mon.csv에서만 뽑는다 (373개 → 52개, 합계 131개)
  - 부스팅: lr 0.1/100회 → lr 0.05/400회 + 규제 (95클래스에서 lr 0.1은 fold에 따라 발산)
  - KMeans·스케일러를 fold 안에서만 fit (검증 fold 누수 제거)
  - Soft voting은 fold마다 각 모델의 predict_proba를 평균해서 계산 (재학습 없이 같은 모델 재사용)
  - URL(950클래스) ET는 트리 하나가 ~200MB라 10그루씩 만들어 proba만 누적한다 (100그루 숲과 같음)
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
STAGES = sys.argv[1] if len(sys.argv) > 1 else '1234'

META = ['site', 'url', 'inst']
df_base = pd.read_csv('features/features_mon.csv')
df_ex1 = pd.read_csv('features/features_extra_mon.csv')
df_ex2 = pd.read_csv('features/features_extra2_mon.csv')
df_all = pd.concat([df_base, df_ex1.drop(columns=META), df_ex2.drop(columns=META)], axis=1)

groups = {
    'base': [c for c in df_base.columns if c not in META + ['in_burst_1', 'in_burst_2', 'in_burst_3']],
    'early': ['first20_out', 'first50_out', 'first200_out', 'first300_out', 'first500_out', 'first1000_out'],
    'bseq': [f'bseq_{i}' for i in range(7, 31)],
    'cumulabs': [f'cumul_at_{x}' for x in [100, 200, 300, 400, 600, 700, 800, 900, 1000, 1500, 2000, 3000, 5000]],
    'dir50': [f'dir_{i:03d}' for i in range(15, 51)],
}
feats = sum(groups.values(), [])
assert len(feats) == len(set(feats)) == 131, len(feats)

df = df_all[df_all['inst'] < 18].reset_index(drop=True)
X = df[feats].values.astype(np.float32)
y = df['site'].values
y_url = df['url'].values
inst = df['inst'].values
classes = np.unique(y)
folds = [(np.where(inst % 5 != k)[0], np.where(inst % 5 == k)[0]) for k in range(5)]
url2site = df.drop_duplicates('url').set_index('url')['site']
print(f'train {X.shape}, sites {len(classes)}, urls {len(np.unique(y_url))}')


def et(**kw):
    p = dict(n_estimators=300, max_features=0.1, random_state=0, n_jobs=-1)
    p.update(kw)
    return ExtraTreesClassifier(**p)


def chunked_proba(Xtr, ytr, Xva, n_trees=100, chunk=10):
    """ET n_trees그루의 predict_proba를 chunk그루씩 나눠 만들어 평균 (메모리 절약)."""
    total = 0
    for c in range(n_trees // chunk):
        total = total + et(n_estimators=chunk, random_state=c).fit(Xtr, ytr).predict_proba(Xva)
    return np.unique(ytr), total / (n_trees // chunk)


def report(name, scores, t0):
    s = np.array(scores)
    print(f'  {name:<34} {s.mean():.4f} ± {s.std():.4f}   {np.round(s, 3)}   {time.time() - t0:.0f}s', flush=True)


# ---------------------------------------------------------------- 1. 모델 비교 (fold마다 proba 저장)
if '1' in STAGES:
    print('\n[1] 모델 비교 (131개, inst%5 기준 5-fold)', flush=True)
    models = {
        'ET (300, mf=0.1)': lambda: et(),
        'RF (300, mf=0.1)': lambda: RandomForestClassifier(n_estimators=300, max_features=0.1, random_state=0, n_jobs=-1),
        'LightGBM (lr .05, 400)': lambda: lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=10,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
            random_state=0, n_jobs=-1, verbose=-1),
        'HistGB (lr .05, 400)': lambda: HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=400, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=False, random_state=0),
    }
    probas = {m: [] for m in models}
    for name, make in models.items():
        t0, scores = time.time(), []
        for tr, va in folds:
            clf = make().fit(X[tr], y[tr])
            p = clf.predict_proba(X[va])
            probas[name].append(p)
            scores.append((clf.classes_[p.argmax(1)] == y[va]).mean())
        report(name, scores, t0)

    for combo in [('ET (300, mf=0.1)', 'RF (300, mf=0.1)', 'LightGBM (lr .05, 400)'),
                  ('ET (300, mf=0.1)', 'LightGBM (lr .05, 400)')]:
        scores = [(classes[np.mean([probas[m][k] for m in combo], axis=0).argmax(1)] == y[va]).mean()
                  for k, (_, va) in enumerate(folds)]
        report('SoftVote ' + '+'.join(m.split()[0] for m in combo), scores, time.time())

# ---------------------------------------------------------------- 2. KMeans 거리 피처 (fold 안에서 fit)
if '2' in STAGES:
    print('\n[2] KMeans(15) 거리 피처 + ET, 누수 제거', flush=True)
    t0, scores = time.time(), []
    for tr, va in folds:
        sc = StandardScaler().fit(X[tr])
        km = KMeans(n_clusters=15, n_init=5, random_state=0).fit(sc.transform(X[tr]))
        aug = lambda idx: np.hstack([X[idx], km.transform(sc.transform(X[idx]))])
        clf = et().fit(aug(tr), y[tr])
        scores.append((clf.predict(aug(va)) == y[va]).mean())
    report('KMeans + ET', scores, t0)

# ---------------------------------------------------------------- 3. site 직접 vs URL 계층
if '3' in STAGES:
    print('\n[3] Site 직접 분류 vs URL(950) 예측 후 site로 합산 (둘 다 ET 100그루)', flush=True)
    t0, s_direct, s_url = time.time(), [], []
    for tr, va in folds:
        cs, ps = chunked_proba(X[tr], y[tr], X[va])
        s_direct.append((cs[ps.argmax(1)] == y[va]).mean())
        cu, pu = chunked_proba(X[tr], y_url[tr], X[va])
        ps = np.zeros((len(va), len(classes)))
        np.add.at(ps.T, np.searchsorted(classes, url2site.loc[cu].values), pu.T)
        s_url.append((classes[ps.argmax(1)] == y[va]).mean())
        print(f'    fold {len(s_url)}/5  {time.time() - t0:.0f}s', flush=True)
    report('Site 직접 (ET)', s_direct, t0)
    report('URL 계층 (ET)', s_url, t0)

# ---------------------------------------------------------------- 4. 피처 그룹 기여도
if '4' in STAGES:
    print('\n[4] 피처 그룹: 단독 사용 / 131개에서 제거', flush=True)
    for g, cols in groups.items():
        for mode, use in [('단독', cols), ('제거', [f for f in feats if f not in cols])]:
            idx = [feats.index(f) for f in use]
            t0, scores = time.time(), []
            for tr, va in folds:
                clf = et(max_features='sqrt' if mode == '단독' else 0.1).fit(X[tr][:, idx], y[tr])
                scores.append((clf.predict(X[va][:, idx]) == y[va]).mean())
            report(f'{g:<9}({len(cols):>2}개) {mode} → {len(use)}개', scores, t0)
