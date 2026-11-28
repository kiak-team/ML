"""open-world test holdout 평가 (B 파트, 단 1회). E10까지 CV로 정한 구성을 그대로 적용한다.

사용법 (레포 루트에서):
  python -I -W ignore -X utf8 code/ow_test.py --dry   > logs/ow_test_dry.log   # 파이프라인 확인: CV fold 0을 test 대신 쓴다 (test 안 봄)
  python -I -W ignore -X utf8 code/ow_test.py         > logs/ow_test.log       # 실제 test 1회
  python -I -W ignore -X utf8 code/ow_test.py --plot                           # 그림만 (test 재계산 없음)

미리 정한 구성 (reports/openworld.md "최종 모델·피처셋 선정", 2026-11-26 확인)
  주 결과   multi(96클래스), RF(n=200, rs=42, 그 외 기본값), A팀 후보 134개(leak 제거 피처), 불균형 처리 없음
  보조      binary, 같은 RF·피처
  민감도    후보 134 − @qconst (121개), multi·binary
  학습      train 전체(mon 17,100 + unmon 9,000)로 한 번
  판정점    argmax, 규칙 B(monitored 점수 최댓값 >= t)의 FPR 0.1% / 1% / 5%.
            t는 train 5-fold OOF 점수의 unmon 9,000개에서 정한다(ow_metrics.fpr_threshold). test를 보고 고르지 않는다
  보고      개수와 Wilson 95% 신뢰구간. test unmon은 1,000개라 FPR 0.1% = 오탐 1개
안전장치: 결과 CSV에 TEST- 행이 이미 있으면 실행하지 않는다(test를 두 번 보지 않도록).
"""
import argparse, os, sys, time

import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment as ex
import ow_models as M
import ow_run as R
from ow_metrics import UNMON, fpr_threshold, mon_score, ow_metrics, unmon_confusion

CONFIGS = [  # (이름, 방식, 피처 지정) — 순서대로 평가. 첫 줄이 주 결과
    ('cand134', 'multi', M.CAND134),
    ('cand134', 'binary', M.CAND134),
    ('cand121', 'multi', M.CAND134 + ['-@qconst']),
    ('cand121', 'binary', M.CAND134 + ['-@qconst']),
]
TARGETS = ((0.001, 'fpr0.1'), (0.01, 'fpr1'), (0.05, 'fpr5'))


def make_rf():
    return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=R.RANDOM_STATE)


def wilson(k, n, z=1.96):
    """이항 비율 k/n의 Wilson 95% 신뢰구간. n = 0이면 (nan, nan)."""
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def ci_str(k, n):
    lo, hi = wilson(k, n)
    return f'{k}/{n} [{lo:.4f}, {hi:.4f}]'


def main(dry):
    if not dry and os.path.exists(R.RESULTS) and \
            pd.read_csv(R.RESULTS, usecols=['exp_id']).exp_id.str.startswith('TEST-').any():
        raise SystemExit('결과 CSV에 TEST- 행이 이미 있다. test는 한 번만 본다. 다시 보려면 이유를 리포트에 적고 행을 지운 뒤 실행')
    d = R.load_pkl(14)
    names = d['names']
    pos = {f: i for i, f in enumerate(names)}
    fold = d['fold_visit']
    if dry:  # fold 0을 test 대신: 학습 = fold 1~4, CV = fold 1~4 안의 4-fold, "test" = fold 0
        is_te = fold == 0
        cv_fold = np.where(fold >= 1, fold - 1, -1)
        tag, dataset = 'DRY-', 'pkl_cvfold0_as_test'
    else:
        is_te = d['is_test'].astype(bool)
        cv_fold = fold
        tag, dataset = 'TEST-', 'pkl_test'
    is_tr = cv_fold >= 0
    assert not (is_tr & is_te).any()
    n_folds = int(cv_fold.max()) + 1
    print(f'{"[DRY RUN: fold 0을 test 대신 사용, 실제 test는 보지 않음]" if dry else "[TEST holdout 1회 평가]"}')
    print(f'학습 {is_tr.sum()}행 (mon {(is_tr & (d["y"] != UNMON)).sum()}, unmon {(is_tr & (d["y"] == UNMON)).sum()}) / '
          f'평가 {is_te.sum()}행 (mon {(is_te & (d["y"] != UNMON)).sum()}, unmon {(is_te & (d["y"] == UNMON)).sum()}). '
          f'임계값은 학습 행의 {n_folds}-fold OOF 점수로 정한다')
    summary = []
    for fs_name, mode, spec in CONFIGS:
        feats = ex.select(spec, names)
        X = d['X'][:, [pos[f] for f in feats]]
        y = M.labels(d, mode)
        exp_base = f'{tag}{mode}-rf-{fs_name}'
        print(f'\n## {exp_base}: 피처 {len(feats)}개', flush=True)
        # (1) 임계값: 학습 행의 OOF 점수 (E7·E9와 같은 계산)
        t0 = time.time()
        oof, classes = _oof(X, y, cv_fold, n_folds)
        if not dry and fs_name == 'cand134':  # E7 캐시와 같은지 (같은 시드·fold → 같아야 한다)
            z = np.load(os.path.join(R.CACHE, 'oof', f'E7-{mode}-rf.npz'))
            assert np.allclose(z['proba'], oof), 'E7 OOF와 다르다'
            print('  임계값용 OOF 점수 = E7 RF OOF와 같음 확인')
        s_oof, _ = mon_score(oof, classes)
        u_oof = s_oof[y[is_tr] == UNMON]
        ts = {suf: fpr_threshold(u_oof, x) for x, suf in TARGETS}
        print(f'  임계값 (train OOF unmon {len(u_oof)}개): ' + ', '.join(f'{k} t={v:.4f}' for k, v in ts.items())
              + f'  ({time.time() - t0:.0f}s)')
        # (2) 학습 전체로 1회 학습 → 평가 행 예측
        m = make_rf().fit(X[is_tr], y[is_tr])
        assert (m.classes_ == classes).all()
        P = m.predict_proba(X[is_te])
        yt = y[is_te]
        s_te, site = mon_score(P, classes)
        is_mon = yt != UNMON
        auc, ap = roc_auc_score(is_mon, s_te), average_precision_score(is_mon, s_te)
        print(f'  탐지 ROC-AUC {auc:.4f}  PR-AUC {ap:.4f}')
        decisions = [('argmax', '', classes[P.argmax(1)])] + \
                    [(suf, round(ts[suf], 6), np.where(s_te >= ts[suf], site, UNMON)) for _, suf in TARGETS]
        for point, thr, pred in decisions:
            r = ow_metrics(yt, pred, labels=classes)
            e = R.extra_metrics(r, yt, pred, len(classes))
            flagged = r['TP'] + r['wrong_site'] + r['FP']
            k_r = r['n_unmon'] / r['n_mon']
            # 현실 비율 precision (E3와 같은 환산): TPR / (det_TPR + k·FPR), k = unmon/mon
            p100 = r['TPR'] / (r['det_TPR'] + 100 * r['FPR']) if (r['det_TPR'] + 100 * r['FPR']) > 0 else float('nan')
            exp_id = f'{exp_base}-{point}'
            cm, labs, cs = unmon_confusion(yt, pred)
            pd.DataFrame(cm, index=labs, columns=labs).to_csv(os.path.join(R.ROOT, 'results', 'confusion', f'{exp_id}.csv'))
            notes = (f'{"DRY RUN fold0-as-test" if dry else "TEST holdout, evaluated once"}; trained on all train rows; '
                     f'decision={"argmax" if point == "argmax" else "rule B, t from train OOF unmon scores for " + point}; '
                     f'TP {ci_str(r["TP"], r["n_mon"])}; FP {ci_str(r["FP"], r["n_unmon"])}; '
                     f'precision {ci_str(r["TP"], flagged)}; wrong_site {r["wrong_site"]}; FN {r["FN"]}; '
                     f'precision at mon:unmon=1:100 {p100:.4f}; skip14 (leak removed) features')
            R.record({'exp_id': exp_id, 'dataset': dataset, 'split_mode': 'test(visit holdout)' if not dry else 'dry',
                      'feature_set': f'{fs_name}({len(feats)})_skip14', 'model': f'RF(n=200,rs=42) {mode}',
                      'imbalance': 'none', 'threshold': thr, 'n_classes': len(classes), 'TPR': r['TPR'],
                      'FPR': r['FPR'], 'precision': r['precision'], 'recall': r['recall'], 'macro_f1': r['macro_f1'],
                      'notes': notes, **e, 'roc_auc': auc, 'pr_auc': ap})
            lo, hi = wilson(r['TP'], r['n_mon'])
            flo, fhi = wilson(r['FP'], r['n_unmon'])
            print(f'  [{exp_id}] TPR {r["TPR"]:.4f} [{lo:.4f}, {hi:.4f}]  FPR {r["FPR"]:.4f} ({r["FP"]}/{r["n_unmon"]}) '
                  f'[{flo:.4f}, {fhi:.4f}]  precision {r["precision"]:.4f}  macro_f1 {r["macro_f1"]:.4f}  acc {e["accuracy"]:.4f} '
                  f'| det_TPR {r["det_TPR"]:.4f} wrong_site {r["wrong_site"]} | prec@1:100 {p100:.4f}', flush=True)
            if point == 'fpr1':
                print(f'      -1 행: unmon {r["n_unmon"]} 중 mon으로 {cs["unmon_as_mon"]} (FP가 많은 사이트 {cs["top_FP_sites"]})')
            summary.append({'config': f'{mode}-{fs_name}', 'point': point, 'TPR': r['TPR'], 'TPR_lo': lo, 'TPR_hi': hi,
                            'FPR': r['FPR'], 'FP': r['FP'], 'FPR_lo': flo, 'FPR_hi': fhi, 'precision': r['precision'],
                            'det_TPR': r['det_TPR'], 'p100': p100, 'auc': auc})
    df = pd.DataFrame(summary)
    out = os.path.join(R.ROOT, 'results', f'ow_{"test_dry" if dry else "test"}_summary.csv')
    df.to_csv(out, index=False)
    print(f'\n요약 표: {out}')


def _oof(X, y, cv_fold, n_folds):
    tr = cv_fold >= 0
    Xt, yt, ft = X[tr], y[tr], cv_fold[tr]
    classes = np.unique(yt)
    P = np.zeros((len(yt), len(classes)), np.float32)
    for k in range(n_folds):
        m = make_rf().fit(Xt[ft != k], yt[ft != k])
        assert (m.classes_ == classes).all()
        P[ft == k] = m.predict_proba(Xt[ft == k])
    return P, classes


CV_SRC = {'multi-cand134': 'E7-multi-rf', 'binary-cand134': 'E7-binary-rf',
          'multi-cand121': 'E9-multi-rf-cand134-qconst', 'binary-cand121': 'E9-binary-rf-cand134-qconst'}


def plot():
    """CV(nested 판정점, E7·E9 행) vs test(Wilson 95% CI). test를 다시 계산하지 않고 저장된 CSV만 읽는다."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    df = pd.read_csv(R.RESULTS)
    t = pd.read_csv(os.path.join(R.ROOT, 'results', 'ow_test_summary.csv'))
    ink, muted, grid, base = '#0b0b0b', '#898781', '#e1e0d9', '#c3c2b7'
    plt.rcParams.update({'font.family': ['AppleGothic', 'sans-serif'], 'axes.unicode_minus': False, 'font.size': 9.5})
    pts = [('fpr0.1', 'FPR ≤ 0.1%'), ('fpr1', 'FPR ≤ 1%'), ('fpr5', 'FPR ≤ 5%'), ('argmax', 'argmax')]
    cfgs = [('multi-cand134', 'multi · 후보 134 (주 결과)'), ('multi-cand121', 'multi · 121 (@qconst 제외)'),
            ('binary-cand134', 'binary · 후보 134'), ('binary-cand121', 'binary · 121')]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9), sharey=True, facecolor='#fcfcfb')
    y = np.arange(len(pts))[::-1]
    for ax, (cfg, title) in zip(axes, cfgs):
        for yy, (p, lab) in zip(y, pts):
            cv = df[df.exp_id == f'{CV_SRC[cfg]}-{p}'].iloc[0]
            s = t[(t.config == cfg) & (t.point == p)].iloc[0]
            ax.plot([s.TPR_lo, s.TPR_hi], [yy - 0.12] * 2, color='#9ec5f4', lw=4, solid_capstyle='round', zorder=2)
            ax.scatter(s.TPR, yy - 0.12, s=55, color='#2a78d6', edgecolor='#fcfcfb', lw=1.5, zorder=3,
                       label='test (가로 막대 = 95% CI)' if yy == y[0] else None)
            ax.scatter(cv.TPR, yy + 0.12, s=55, color='#898781', marker='D', edgecolor='#fcfcfb', lw=1.5, zorder=3,
                       label='CV (train 5-fold OOF)' if yy == y[0] else None)
            ax.annotate(f'{s.TPR * 100:.1f}  (FPR {s.FPR * 100:.1f}%)', (s.TPR_hi, yy - 0.12), xytext=(5, 0),
                        textcoords='offset points', va='center', color=ink, fontsize=8)
        ax.set_xlim(0.35, 1.08)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v * 100:g}%' if v <= 1 else ''))
        ax.set_title(title, color=ink, loc='left')
        ax.set_xlabel('TPR' + (' (사이트까지 맞힘)' if cfg.startswith('multi') else ' (탐지)'), color=ink)
        R._style(ax, ink, muted, grid, base)
        ax.tick_params(axis='y', colors=ink)
    axes[0].set_yticks(y, [lab for _, lab in pts])
    axes[0].set_ylim(-0.6, len(pts) - 0.4)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc='lower center', ncol=2, labelcolor=ink, fontsize=9)
    fig.suptitle('test holdout 1회 평가 — RF(기본값), leak 제거 피처. 임계값은 train OOF로 미리 정함. 숫자 = test TPR (실제 FPR)',
                 color=ink, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = os.path.join(R.ROOT, 'figures', 'ow_test.png')
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f'그림: {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry', action='store_true', help='CV fold 0을 test 대신 쓴다 (실제 test는 보지 않음)')
    ap.add_argument('--plot', action='store_true', help='저장된 test 결과로 그림만 그린다 (test 재계산 없음)')
    a = ap.parse_args()
    plot() if a.plot else main(a.dry)
