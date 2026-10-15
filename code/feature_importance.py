"""랜덤 포레스트로 피처 중요도 분석 (sklearn 필요).

사용법: python -I code/feature_importance.py <wf_project 폴더>
  예) python -I code/feature_importance.py .   (약 10~15분)

- 분할은 eval_nearest_centroid.py와 같다: inst 0~17 → train, 18~19 → test.
- 출력 (logs/feature_importance.log):
  1. 피처 55개 전체 RF 정확도
  2. RF 불순도 중요도 (train 기준)
  3. permutation 중요도: test에서 피처 하나를 섞었을 때 정확도 하락
  4. 상관된 피처 묶음 단위 permutation: 서로 대신하는 피처를 함께 섞는다
  5. 그룹 실험: 그룹만 썼을 때 / 그룹을 뺐을 때 정확도 (모델 재학습)
  6. mon vs unmon 구분 RF의 AUC와 중요도
- 탐색용 분석이다. test를 보고 피처를 고르면 누수가 생기므로, 실제 선택은
  train 내부 교차검증으로 한다.
"""
import sys
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from scipy.cluster import hierarchy
from scipy.stats import spearmanr

root = sys.argv[1]
mon = pd.read_csv(f"{root}/features/features_mon.csv")
unmon = pd.read_csv(f"{root}/features/features_unmon.csv")
F = [c for c in mon.columns if c not in ("site", "url", "inst")]
tr, te = mon[mon.inst <= 17], mon[mon.inst >= 18]
Xtr, ytr, Xte, yte = tr[F].values, tr.site.values, te[F].values, te.site.values

rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0)
rf.fit(Xtr, ytr)
print(f"[closed-world RF] all 55 acc = {rf.score(Xte, yte):.4f}")

imp = pd.Series(rf.feature_importances_, F).sort_values(ascending=False)
print("\n== RF impurity importance top 20 ==")
for k, v in imp.head(20).items(): print(f"  {k:16s} {v:.4f}")
print("  ... bottom 5:", ", ".join(f"{k}={v:.4f}" for k, v in imp.tail(5).items()))

pi = permutation_importance(rf, Xte, yte, n_repeats=5, random_state=0, n_jobs=-1)
pis = pd.Series(pi.importances_mean, F).sort_values(ascending=False)
print("\n== permutation importance on test (acc drop) top 15 ==")
for k, v in pis.head(15).items(): print(f"  {k:16s} {v:+.4f}")

# 상관 묶음: 1-|spearman|을 거리로 계층 군집 → 묶음째 섞어서 정확도 하락을 잰다
corr = spearmanr(Xtr).correlation
corr = np.nan_to_num((corr + corr.T) / 2, nan=0.0); np.fill_diagonal(corr, 1)
link = hierarchy.ward(1 - np.abs(corr))
cl = hierarchy.fcluster(link, 0.3, criterion="distance")
groups = {}
for f, c in zip(F, cl): groups.setdefault(c, []).append(f)
base = rf.score(Xte, yte); rng = np.random.default_rng(0)
rows = []
for c, fs in groups.items():
    idx = [F.index(f) for f in fs]; drops = []
    for _ in range(3):
        Xp = Xte.copy(); perm = rng.permutation(len(Xp))
        Xp[:, idx] = Xp[perm][:, idx]; drops.append(base - rf.score(Xp, yte))
    rows.append((np.mean(drops), fs))
print(f"\n== correlated clusters (|spearman| based), grouped permutation drop; {len(groups)} clusters ==")
for d, fs in sorted(rows, reverse=True):
    print(f"  {d:+.4f}  ({len(fs):2d}) {', '.join(fs[:8])}{' ...' if len(fs) > 8 else ''}")

G = {
 "count": F[0:4], "timing": [f for f in F if f.startswith(("duration","cells_per","t_","iat","pps"))],
 "order/conc": [f for f in F if f.startswith(("out_order","in_order","first","last30","conc"))],
 "burst": [f for f in F if "burst" in f], "cumul": [f for f in F if f.startswith("cumul")],
}
print("\n== group: only-this-group acc / drop-this-group acc ==")
for g, fs in G.items():
    keep = [F.index(f) for f in fs]; rest = [i for i in range(len(F)) if i not in keep]
    a = RandomForestClassifier(300, n_jobs=-1, random_state=0).fit(Xtr[:, keep], ytr).score(Xte[:, keep], yte)
    b = RandomForestClassifier(300, n_jobs=-1, random_state=0).fit(Xtr[:, rest], ytr).score(Xte[:, rest], yte)
    print(f"  {g:11s} ({len(fs):2d})  only={a:.4f}  without={b:.4f}")

# mon vs unmon: unmon은 앞 80% train, 뒤 20% test (무작위 없음)
u = unmon[F].values; ntr = int(len(u) * 0.8)
Xb = np.vstack([Xtr, u[:ntr]]); yb = np.r_[np.ones(len(Xtr)), np.zeros(ntr)]
Xbt = np.vstack([Xte, u[ntr:]]); ybt = np.r_[np.ones(len(Xte)), np.zeros(len(u) - ntr)]
rb = RandomForestClassifier(300, n_jobs=-1, random_state=0, class_weight="balanced").fit(Xb, yb)
print(f"\n[mon vs unmon RF] AUC = {roc_auc_score(ybt, rb.predict_proba(Xbt)[:, 1]):.4f}")
ib = pd.Series(rb.feature_importances_, F).sort_values(ascending=False)
print("  top 10:", ", ".join(f"{k}={v:.3f}" for k, v in ib.head(10).items()))
