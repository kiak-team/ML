"""closed-world(C파트) 발표용 그래프. results/cw_oof/*.npz와 results/cw_ensemble.json을 읽는다.

사용법 (레포 루트에서, matplotlib 필요: pip install matplotlib):
  python -I -W ignore -X utf8 code/cw_plots.py            # 전부
  python -I -W ignore -X utf8 code/cw_plots.py models folds

그래프 (figures/에 저장):
  models    cw_models.png     모델별 CV 정확도 (기본값 → 튜닝, DF 변형)
  ensemble  cw_ensemble.png   단일 최고 vs 섞는 방법 4가지 (마지막으로 돌린 ensemble 결과 기준)
  folds     cw_folds.png      fold별 정확도. fold 4(inst 19 포함)에서 모두 떨어지는 것과 앙상블이 메우는 것
  diversity cw_diversity.png  두 모델이 서로 다르게 예측하는 비율. 앙상블이 왜 오르는지(다양성) 설명

※ ensemble 그래프는 results/cw_ensemble.json(마지막 ensemble 실행)을 그린다.
  최종 후보로 ensemble을 한 번 더 돌린 뒤 그리면 된다.
"""
import json, os, sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plots as P  # 팀 그래프 스타일(색·글꼴)을 그대로 쓴다
import cw_tuning as cw

C, INK, INK2 = P.C, P.INK, P.INK2
LABEL = {'lgbm': 'LightGBM', 'et': 'ExtraTrees', 'rf': 'RandomForest', 'svm': 'SVM (RBF)', 'knn': 'k-NN',
         'lr': '로지스틱 회귀', 'hgb': 'HistGB', 'mlp': 'MLP', 'df': 'DF 1D-CNN'}
DF_TAG = {'base': '기본', 'cos': '학습률 cosine', 's1': '시드 1', 'iat': '+시간 간격 2채널', 'tt': 'Tik-Tok(방향×시각)',
          'L10k': '길이 10,000', 'shift': '시작 위치 증강'}


def oofs():
    o = cw.load_oofs()
    y = next(iter(o.values()))[0]
    z = np.load(os.path.join(cw.OOF_DIR, sorted(f for f in os.listdir(cw.OOF_DIR) if f.endswith('.npz'))[0]))
    fold = z['inst'][np.argsort(z['url'].astype(np.int64) * 20 + z['inst'])] % 5
    return o, y, fold


def acc(o, y, n):
    return (o[n][1] == y).mean()


def fig_models():
    o, y, _ = oofs()
    rows = []  # (라벨, 기본값, 튜닝 후)
    for m in ['lgbm', 'et', 'rf', 'svm', 'knn', 'lr', 'hgb', 'mlp']:
        b = acc(o, y, f'{m}_base') if f'{m}_base' in o else None
        t = acc(o, y, f'{m}_tuned') if f'{m}_tuned' in o else None
        if b is not None or t is not None:
            rows.append((LABEL[m], b, t))
    dfs = sorted((n for n in o if n.startswith('df_')), key=lambda n: -acc(o, y, n))
    rows += [(f'DF · {DF_TAG.get(n[3:], n[3:])}', acc(o, y, n), None) for n in dfs]
    rows.sort(key=lambda r: max(v for v in r[1:] if v is not None))
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(rows) + 1.2))
    for i, (lab, b, t) in enumerate(rows):
        best = max(v for v in (b, t) if v is not None)
        col = C[2] if lab.startswith('DF') else (C[0] if t is not None else '#c9c8c3')
        ax.barh(i, best * 100, color=col, height=0.6)
        if b is not None and t is not None and abs(t - b) > 0.0005:  # 기본값 위치를 검은 눈금으로
            ax.plot([b * 100] * 2, [i - 0.36, i + 0.36], color=INK, lw=2)
        txt = f'{best * 100:.2f}%' + (f'  (기본 {b * 100:.2f} → 튜닝 {t * 100:.2f})' if b and t and abs(t - b) > 0.0005 else '')
        ax.text(best * 100 + 0.05, i, txt, va='center', fontsize=8.5, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    lo = min(v for r in rows for v in r[1:] if v is not None) * 100
    ax.set_xlim(max(lo - 2, 0), 100.5)
    ax.set_xlabel('train 5-fold CV 정확도 (%)  ·  새 test 분할(11/15)  ·  피처 131개 / DF는 방향 시퀀스 5,000')
    ax.set_title('closed-world 95개 사이트: 모델별 정확도')
    ax.legend(handles=[Patch(color=C[0], label='튜닝 후 (검은 눈금 = 기본값)'), Patch(color='#c9c8c3', label='기본값만'),
                       Patch(color=C[2], label='DF 1D-CNN (원본 시퀀스)')], loc='lower right', fontsize=8.5)
    P.save(fig, 'cw_models.png')


def fig_ensemble():
    path = os.path.join(cw.RES, 'cw_ensemble.json')
    if not os.path.exists(path):
        print('cw_ensemble.json 없음 → ensemble 먼저'); return
    e = json.load(open(path, encoding='utf-8'))
    o, y, _ = oofs()
    single = max(e['members'], key=lambda n: acc(o, y, n))
    names = [f'단일 최고: {single}', '단순 평균', '기하 평균', '가중 평균', 'stacking']
    vals = [acc(o, y, single)] + [e['results'][k]['cv'] for k in ['mean', 'geo', 'wvote', 'stack']]
    errs = [np.std([(o[single][1][f] == y[f]).mean() for f in [_fold() == k for k in range(5)]])] + \
           [e['results'][k]['std'] for k in ['mean', 'geo', 'wvote', 'stack']]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    cols = ['#c9c8c3'] + [C[0]] * 4
    cols[int(np.argmax(vals))] = C[1]
    ax.bar(range(5), [v * 100 for v in vals], color=cols, width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v * 100 + 0.06, f'{v * 100:.2f}%', ha='center', fontsize=9.5, color=INK)
    ax.set_xticks(range(5), names, fontsize=9)
    ax.set_ylim(min(vals) * 100 - 1.0, max(vals) * 100 + 0.6)
    ax.set_ylabel('CV 정확도 (%)')
    ax.set_title('모델을 섞으면: 단일 모델 대비 앙상블')
    mem = ', '.join(e['members'])
    fig.text(0.01, -0.12, f'멤버: {mem}\n가중 평균·stacking은 fold마다 나머지 4 fold로만 학습(중첩 CV). '
             f'fold 간 표준편차: 단일 {errs[0] * 100:.2f}%p, 앙상블 {min(errs[1:]) * 100:.2f}~{max(errs[1:]) * 100:.2f}%p',
             color=INK2, fontsize=8)
    P.save(fig, 'cw_ensemble.png')


_FOLD = None


def _fold():
    global _FOLD
    if _FOLD is None:
        _FOLD = oofs()[2]
    return _FOLD


def fig_folds():
    o, y, fold = oofs()
    pick = [n for n in ['lgbm_tuned', 'rf_base', 'et_base', 'df_base', 'svm_base', 'knn_tuned'] if n in o]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for i, n in enumerate(pick):
        fa = [(o[n][1][fold == k] == y[fold == k]).mean() * 100 for k in range(5)]
        ax.plot(range(5), fa, marker='o', color=C[i % len(C)] if i < len(C) else '#999', label=n, alpha=0.85)
    path = os.path.join(cw.RES, 'cw_ensemble.json')
    if os.path.exists(path):
        e = json.load(open(path, encoding='utf-8'))
        best = max(e['results'], key=lambda k: e['results'][k]['cv'])
        ax.plot(range(5), [v * 100 for v in e['results'][best]['folds']], marker='s', color=INK, lw=2.6,
                label=f'앙상블 ({best})')
    ax.set_xticks(range(5), [f'fold {k}\n(inst {k},{k + 5},{k + 10},{k + 15})' for k in range(5)], fontsize=8.5)
    ax.axvspan(3.6, 4.4, color='#f3e1d6', zorder=0)
    ax.text(4, ax.get_ylim()[1] - 0.1, 'inst 19 포함\n(짧고 느린 마지막 방문)', ha='center', va='top', fontsize=8, color=INK2)
    ax.set_ylabel('정확도 (%)')
    ax.set_title('fold별 정확도: 모든 모델이 fold 4에서 떨어진다')
    ax.legend(fontsize=8, ncol=2, loc='lower left')
    P.save(fig, 'cw_folds.png')


def fig_diversity():
    o, y, _ = oofs()
    path = os.path.join(cw.RES, 'cw_ensemble.json')
    names = json.load(open(path, encoding='utf-8'))['members'] if os.path.exists(path) else \
        sorted(o, key=lambda n: -acc(o, y, n))[:8]
    n = len(names)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = (o[names[i]][1] != o[names[j]][1]).mean() * 100  # 두 모델의 예측이 다른 비율
    fig, ax = plt.subplots(figsize=(0.75 * n + 2.2, 0.62 * n + 1.6))
    im = ax.imshow(M, cmap='Blues')
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f'{M[i, j]:.1f}', ha='center', va='center', fontsize=8,
                        color='white' if M[i, j] > M.max() * 0.6 else INK)
    ax.set_xticks(range(n), names, rotation=40, ha='right', fontsize=8.5)
    ax.set_yticks(range(n), names, fontsize=8.5)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label='예측이 서로 다른 기록 비율 (%)')
    ax.set_title('두 모델이 다르게 예측하는 비율 (클수록 섞을 때 이득)')
    P.save(fig, 'cw_diversity.png')


FIGS = {'models': fig_models, 'ensemble': fig_ensemble, 'folds': fig_folds, 'diversity': fig_diversity}

if __name__ == '__main__':
    for k in (sys.argv[1:] or FIGS):
        FIGS[k]()
