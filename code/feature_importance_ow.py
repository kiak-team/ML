"""Open-world 피처 그룹 permutation 중요도 (RF, 후보 134개, multi 96클래스).

모델·피처·평가는 open-world(B 파트)와 똑같이 맞춘다.
  - 피처: ow_models.load() = leak 제거(skip14) 피처 중 A팀 후보 134개 (@all @early @bseq @dir50 @qconst)
  - 모델: ow_run.rf() = RandomForest(200그루, random_state 고정)
  - 점수: TPR@FPR<=1% (규칙 B, ow_models.best_at) — B 파트 E7~E10과 같은 계산
  - 분할: visit 기준 train 5-fold. test(fold -1)는 쓰지 않는다.

방법: fold k로 학습한 모델을 고정해 두고, 검증 fold에서 한 그룹의 열들을 "같은 행 순서로 함께" 섞었을 때
TPR@FPR<=1%가 얼마나 떨어지는지 잰다(REPEATS번 평균). 그룹 안 피처끼리 상관이 높아도 함께 섞으므로
서로 대신해 주는 효과가 없다. 재학습하지 않으므로 "이 모델이 얼마나 기대는가"이지 "빼도 되는가"가 아니다.
(빼고 재학습한 결과는 B 파트 E8 참고.)

실행 (레포 루트에서):
    python code/feature_importance_ow.py                  # 134개
    python code/feature_importance_ow.py --drop-qconst    # @qconst 13개를 뺀 121개
    python code/feature_importance_ow.py --compare        # 위 두 결과를 한 그림에 (학습 없음)
출력:
    results/fi/ow_group_perm{_121}.csv     fold × 그룹별 하락
    results/fi/ow_group_summary{_121}.csv  그룹별 평균 ± 표준편차
    figures/fi_ow_group{_121}.png, figures/fi_ow_group_compare.png
로그는 tee로 남긴다: ... | tee logs/feature_importance_ow{_121}.log
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment as ex  # noqa: E402
import ow_models as M  # noqa: E402
import ow_run as R  # noqa: E402

GROUPS = ['count', 'timing', 'order', 'burst', 'cumul', 'early', 'bseq', 'dir50', 'qconst']


def group_columns(names, all_names, drop=()):
    """ex.groups 정의를 names(고른 피처) 안의 열 번호로 바꾼다. 모든 열이 정확히 한 그룹에 들어가는지 확인."""
    g = ex.groups(all_names)
    cols = {k: [names.index(f) for f in g[k] if f in names] for k in GROUPS if k not in drop}
    flat = [c for v in cols.values() for c in v]
    assert len(flat) == len(set(flat)) == len(names), f'그룹 배정 오류: {len(flat)}열 / 중복 {len(flat) - len(set(flat))} / 전체 {len(names)}'
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--drop-qconst', action='store_true', help='@qconst 13개를 빼고 121개로 학습·평가')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--compare', action='store_true', help='학습 없이 134개·121개 요약 CSV로 비교 그림만 그린다')
    a = ap.parse_args()
    if a.compare:
        return compare()

    d = M.load(False)  # 전체 데이터, 후보 134개
    names, X = list(d['names']), d['X']
    if a.drop_qconst:
        q = set(ex.groups(d['all_names'])['qconst'])
        keep = [i for i, f in enumerate(names) if f not in q]
        X, names = X[:, keep], [names[i] for i in keep]
    groups = group_columns(names, d['all_names'], drop=('qconst',) if a.drop_qconst else ())
    tag = f'_{len(names)}' if a.drop_qconst else ''
    y, fold = d['y'], d['fold_visit']
    print(f'피처 {len(names)}개, X {X.shape}, 그룹: ' + ' '.join(f'{k}({len(v)})' for k, v in groups.items()), flush=True)

    rng = np.random.default_rng(a.seed)
    rows, t_all = [], time.time()
    for k in range(5):
        fit = np.flatnonzero((fold >= 0) & (fold != k))
        val = np.flatnonzero(fold == k)
        model = R.rf().fit(X[fit], y[fit])
        Xv, yv = X[val], y[val]

        def score(Xm):
            S = model.predict_proba(Xm)
            return M.best_at(R.rule_curve(S, model.classes_, yv, 'B'), 'TPR', 0.01)

        base = score(Xv)
        rows.append(dict(fold=k, group='(기준)', base=base, drop=0.0))
        for gname, cols in groups.items():
            drops = []
            for _ in range(a.repeats):
                perm = rng.permutation(len(Xv))
                Xp = Xv.copy()
                Xp[:, cols] = Xv[np.ix_(perm, cols)]  # 그룹 열들을 같은 순서로 함께 섞기
                drops.append(base - score(Xp))
            rows.append(dict(fold=k, group=gname, base=base, drop=float(np.mean(drops))))
        print(f'fold {k}: 기준 TPR@FPR1% {base:.4f} (누적 {time.time() - t_all:.0f}초)', flush=True)

    os.makedirs('results/fi', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(f'results/fi/ow_group_perm{tag}.csv', index=False)

    base_mean = df[df.group == '(기준)'].base.mean()
    s = (df[df.group != '(기준)'].groupby('group')['drop'].agg(['mean', 'std'])
         .sort_values('mean', ascending=False))
    s.insert(0, 'n_feats', [len(groups[g]) for g in s.index])
    s.to_csv(f'results/fi/ow_group_summary{tag}.csv')

    print(f'\n## 요약 — 피처 {len(names)}개, 기준 TPR@FPR1% {base_mean:.4f} (fold 평균), 섞었을 때 하락 %p (fold 평균 ± sd)')
    for g, r in s.iterrows():
        print(f'  @{g:7s} ({int(r.n_feats):2d}개)  {r["mean"] * 100:6.2f} ± {r["std"] * 100:.2f}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = s.sort_values('mean')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh([f'{g} ({n})' for g, n in zip(p.index, p.n_feats)], p['mean'] * 100, xerr=p['std'] * 100, capsize=3)
    ax.set_xlabel('Drop in TPR@FPR1% when group is shuffled (%p)')
    ax.set_title(f'Open-world (RF, {len(names)} features): group permutation importance\n'
                 f'visit 5-fold CV, mean ± std, base {base_mean:.3f}')
    plt.tight_layout()
    plt.savefig(f'figures/fi_ow_group{tag}.png', dpi=200)
    print('DONE', flush=True)


def compare():
    """results/fi의 134개·121개 요약을 한 그림에 (figures/fi_ow_group_compare.png)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a = pd.read_csv('results/fi/ow_group_summary.csv', index_col=0) * [1, 100, 100]
    b = pd.read_csv('results/fi/ow_group_summary_121.csv', index_col=0) * [1, 100, 100]
    order = list(b.sort_values('mean', ascending=False).index) + ['qconst']
    yy, h = np.arange(len(order)), 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    for off, s, lab in [(-h / 2, a, '134 features (with qconst)'), (h / 2, b, '121 features (without qconst)')]:
        ax.barh(yy + off, [s['mean'].get(g, np.nan) for g in order], h,
                xerr=[s['std'].get(g, 0) for g in order], capsize=2, label=lab)
    ax.set_yticks(yy)
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel('Drop in TPR@FPR1% when group is shuffled (%p)')
    ax.set_title('Open-world (RF): group permutation importance, visit 5-fold CV mean ± std')
    ax.legend()
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/fi_ow_group_compare.png', dpi=200)
    print('saved figures/fi_ow_group_compare.png')


if __name__ == '__main__':
    main()


