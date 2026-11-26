"""open-world 측정 축 실험 E7~E10 (B 파트). 결과는 ow_run.py와 같은 results/openworld_results.csv에 누적한다.

사용법 (레포 루트에서):
  python -I -W ignore -X utf8 code/ow_models.py time            > logs/ow_e7_timing.log   # 20% 층화 서브샘플 소요 시간
  python -I -W ignore -X utf8 code/ow_models.py e7 --subset     > logs/ow_e7_subset.log   # 파이프라인 검증 (mon 25 / unmon 5000)
  python -I -W ignore -X utf8 code/ow_models.py e7              > logs/ow_e7.log
  python -I -W ignore -X utf8 code/ow_models.py e8 [--subset]   > logs/ow_e8[_subset].log  # 피처 그룹 (RF, LR)
  python -I -W ignore -X utf8 code/ow_models.py e9 [--subset]   > logs/ow_e9[_subset].log  # 피처 선택 (RF)
  python -I -W ignore -X utf8 code/ow_models.py e10 [--subset]  > logs/ow_e10[_subset].log # RF 격자 탐색 + 튜닝 모델 OOF
  python -I -W ignore -X utf8 code/ow_models.py e10b [--subset] > logs/ow_e10b[_subset].log # 불균형 재확인 + subpage

공통
  피처 = features/ow_skip14_*.csv (앞 14셀 버림 = leak 제거). 이 파일의 실험은 전부 leak 제거 피처 기준이다
  split = splits/pkl_*.csv의 visit 모드(fold_visit). train 내부 5-fold CV의 out-of-fold 점수. test는 쓰지 않는다
  클래스 = monitored 95개(데이터에서 셈) + unmon -1 = 96 (multi), mon 1 / unmon -1 (binary)
  전처리 = A팀 experiment.build와 같다. fold 안에서 fit하는 Pipeline이라 검증 fold 통계가 학습에 새지 않는다
    RF / ET / HGB        원본 값
    LR                   sign(x)·log(1+|x|) → StandardScaler
    SVM / k-NN           sign(x)·log(1+|x|) → StandardScaler → 그룹 가중치 1/√(그룹 크기) (experiment.group_weights)
  하이퍼파라미터 = A팀 experiment.MODELS의 기본값 (SVM C=10 gamma=scale, LR max_iter=2000, RF·ET 200그루, k-NN k=5).
    트리 random_state만 42 (이 레포 split·E1~E6과 같게). LightGBM이 없어 HistGradientBoosting(기본값)으로 대체
  판정점 = argmax, 그리고 규칙 B(monitored 점수 최댓값 >= t)의 FPR 0.1% / 1% / 5% 임계값.
    임계값은 평가 fold를 뺀 나머지 4개 fold의 unmon OOF 점수로 정한다(ow_run.nested_fpr_pred, 누수 없음)
  점수 = predict_proba. SVM은 기본으로 decision_function(A팀과 같음, probability=True는 time 결과로 판단)
"""
import argparse, os, sys, time

import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment as ex
import ow_run as R
from ow_metrics import UNMON, mon_score

RS = R.RANDOM_STATE
CAND134 = ['@all', '@early', '@bseq', '@dir50', '@qconst']  # A팀 open-world 후보 (docs/OPENWORLD.md 3.7)
MODES = ('binary', 'multi')
FPR_TARGETS = ((0.001, 'fpr0.1'), (0.01, 'fpr1'), (0.05, 'fpr5'))
TREES = ('rf', 'et', 'hgb', 'hgbl2')  # 원본 값. 나머지는 log + 표준화
LABEL = {'lr': 'LR', 'rf': 'RF', 'svm': 'SVM(RBF)', 'svmp': 'SVM(RBF)+Platt', 'et': 'ET', 'hgb': 'HGB', 'hgbl2': 'HGB(l2=1)', 'knn': 'k-NN'}


# ---------------- 모델 ----------------
def make_model(name, feats, names, **params):
    """A팀 experiment.build와 같은 전처리 + 모델. params는 마지막 추정기에 넣는다."""
    est = {
        'rf': lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RS),
        'et': lambda: ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=RS),
        'hgb': lambda: HistGradientBoostingClassifier(random_state=RS),
        # 기본값(l2 0)은 96클래스 전체 데이터에서 학습 손실이 발산한다(E7 로그, reports 3절 E7). l2만 1로 바꾼 안정화판
        'hgbl2': lambda: HistGradientBoostingClassifier(l2_regularization=1.0, random_state=RS),
        'lr': lambda: LogisticRegression(max_iter=2000),
        'svm': lambda: SVC(C=10, gamma='scale'),
        # Platt 확률 (sklearn 1.9에서 deprecated, 1.11에서 제거. 내부 5-fold CV라 시드 고정)
        'svmp': lambda: SVC(C=10, gamma='scale', probability=True, random_state=RS),
        'knn': lambda: KNeighborsClassifier(n_neighbors=5),
    }[name]().set_params(**params)
    if name in TREES:
        return est
    steps = [FunctionTransformer(ex.signed_log), StandardScaler()]
    if name in ('svm', 'svmp', 'knn'):  # A팀 6c단계: k-NN·SVM은 그룹 가중치가 이득, LR은 손해라 쓰지 않는다
        steps.append(ex.GroupWeight(ex.group_weights(feats, names)))
    return make_pipeline(*steps, est)


def model_desc(name, **params):
    base = {'rf': 'RF(n=200,rs=42)', 'et': 'ET(n=200,rs=42)', 'hgb': 'HGB(default,rs=42)', 'hgbl2': 'HGB(l2_regularization=1,rs=42)',
            'lr': 'LR(max_iter=2000) log+std', 'svm': 'SVC(C=10,gamma=scale) log+std+gw',
            'svmp': 'SVC(C=10,gamma=scale,probability=True) log+std+gw',
            'knn': 'kNN(k=5) log+std+gw'}[name]
    return base + (f' {params}' if params else '')


def score(m, X):
    """(n, 클래스 수) 점수. predict_proba가 없거나(SVC 기본) 꺼져 있으면 decision_function.
    binary decision_function(1차원)은 [-s, s]로 만들어 classes [-1, 1]과 맞춘다(argmax = s > 0)."""
    # sklearn 1.9에서 SVC.probability 기본값은 문자열 'deprecated'라 참/거짓으로 보면 안 된다 → 메서드 유무로 판단
    if not hasattr(m, 'predict_proba'):
        s = m.decision_function(X)
        return np.column_stack([-s, s]) if s.ndim == 1 else s
    return m.predict_proba(X)


def _fit_fold(make, Xf, yf, Xv):
    t0 = time.time()
    m = make().fit(Xf, yf)
    t1 = time.time()
    return score(m, Xv).astype(np.float32), m.classes_, t1 - t0, time.time() - t1


def cv_scores(X, y, fold, make, parallel):
    """fold >= 0인 행에서 5-fold OOF 점수. parallel=True면 fold 5개를 프로세스 5개로 동시에 돌린다
    (모델 자체가 단일 스레드인 SVM·LR·k-NN). 반환: (점수, classes, fold별 fit 시간, fold별 predict 시간)."""
    tr = fold >= 0
    Xt, yt, ft = X[tr], y[tr], fold[tr]
    classes = np.unique(yt)
    jobs = [delayed(_fit_fold)(make, Xt[ft != k], yt[ft != k], Xt[ft == k]) for k in range(5)]
    out = Parallel(n_jobs=5)(jobs) if parallel else [j[0](*j[1], **j[2]) for j in jobs]
    S = np.zeros((len(yt), len(classes)), np.float32)
    for k, (s, cls, tf, tp) in enumerate(out):
        assert (cls == classes).all(), f'fold {k}: train에 빠진 클래스가 있다'
        S[ft == k] = s
        print(f'    fold {k}: fit {int((ft != k).sum())} / val {int((ft == k).sum())}  fit {tf:.0f}s predict {tp:.0f}s',
              flush=True)
    return S, classes, [o[2] for o in out], [o[3] for o in out]


# ---------------- 데이터 ----------------
def load(subset, spec=CAND134):
    """ow_run.load_pkl(skip=14)에서 spec의 피처만 고른 dict. names = 고른 피처 이름, all_names = 376개 전체."""
    d = R.load_pkl(14, subset)
    feats = ex.select(spec, d['names'])
    idx = [d['names'].index(f) for f in feats]
    return dict(d, X=d['X'][:, idx], names=feats, all_names=d['names'])


def labels(d, mode):
    return d['y'] if mode == 'multi' else np.where(d['y'] == UNMON, UNMON, 1)


# ---------------- 평가 ----------------
def best_at(c, key, x):
    m = c['FPR'] <= x
    return float(c[key][m].max()) if m.any() else 0.0


def assess(exp_id, d, mode, S, classes, feature_set, model, imbalance='none', split_mode='visit', extra_notes=''):
    """argmax + FPR 목표 3개 판정점을 CSV에 기록하고, 곡선 요약 dict를 반환한다."""
    db = dict(d, y=labels(d, mode))
    fold = db[f'fold_{split_mode}']
    tr = fold >= 0
    y, f = db['y'][tr], fold[tr]
    s_det, _ = mon_score(S, classes)
    is_mon = y != UNMON
    auc, ap = roc_auc_score(is_mon, s_det), average_precision_score(is_mon, s_det)
    auc_sd = float(np.std([roc_auc_score(is_mon[f == k], s_det[f == k]) for k in range(5)]))
    curve = R.rule_curve(S, classes, y, 'B')
    # fold마다 그 fold의 unmon으로 FPR <= 1%가 되는 최고 TPR (E10 scorer와 같은 계산) → fold 간 표준편차
    per_fold = [best_at(R.rule_curve(S[f == k], classes, y[f == k], 'B'), 'TPR', 0.01) for k in range(5)]
    summary = {'roc_auc': auc, 'roc_auc_sd': auc_sd, 'pr_auc': ap, 'curve': curve,
               'tpr_fpr1_folds': per_fold, **{f'best_{suf}': best_at(curve, 'TPR', x) for x, suf in FPR_TARGETS}}
    ex_ = {'roc_auc': auc, 'pr_auc': ap}
    base = (f'; ROC-AUC fold std {auc_sd:.4f}; pooled best TPR@FPR<=0.1/1/5% '
            f'{summary["best_fpr0.1"]:.4f}/{summary["best_fpr1"]:.4f}/{summary["best_fpr5"]:.4f}; '
            f'per-fold TPR@FPR<=1% mean {np.mean(per_fold):.4f} std {np.std(per_fold):.4f}; '
            f'skip14 (leak removed) features' + extra_notes)
    print(f'  ROC-AUC {auc:.4f} (fold std {auc_sd:.4f})  PR-AUC {ap:.4f}  | pooled best TPR@FPR≤0.1/1/5% '
          f'{summary["best_fpr0.1"]:.4f} / {summary["best_fpr1"]:.4f} / {summary["best_fpr5"]:.4f}  '
          f'| per-fold TPR@FPR≤1% {np.mean(per_fold):.4f} ± {np.std(per_fold):.4f}', flush=True)
    pred = np.asarray(classes)[S.argmax(1)]
    summary['argmax'] = R.evaluate(exp_id + '-argmax', db, split_mode, pred, classes, '', feature_set, model,
                                   imbalance, base, decision='argmax of scores', extras=ex_)
    summary['argmax']['accuracy'] = float(np.mean(y == pred))
    for x, suf in FPR_TARGETS:
        pred, ts = R.nested_fpr_pred(S, classes, y, f, x)
        summary[suf] = R.evaluate(f'{exp_id}-{suf}', db, split_mode, pred, classes, round(float(np.mean(ts)), 6),
                                  feature_set, model, imbalance,
                                  base + f'; threshold per fold {[round(t, 4) for t in ts]}',
                                  decision=f'rule B: max monitored score >= t, t from other 4 folds for FPR<={x:.1%}',
                                  extras=ex_)
    return summary


def save_scores(exp_id, d, mode, S, classes, split_mode='visit'):
    os.makedirs(os.path.join(R.CACHE, 'oof'), exist_ok=True)
    fold = d[f'fold_{split_mode}']
    tr = fold >= 0
    np.savez_compressed(os.path.join(R.CACHE, 'oof', f'{exp_id}.npz'), proba=S, classes=classes,
                        y=labels(d, mode)[tr], fold=fold[tr], row=d['row'][tr])


# ---------------- 소요 시간 (20% 층화 서브샘플) ----------------
def timing(_subset):
    d = load(False)
    fold = d['fold_visit']
    tr = fold >= 0
    X, y = d['X'][tr], d['y'][tr]
    n_full_fit = int(round(tr.sum() * 4 / 5))
    print(f'## 소요 시간 측정: 후보 {len(d["names"])}개 피처, train {tr.sum()}행 (mon {(y != UNMON).sum()}, '
          f'unmon {(y == UNMON).sum()}). 실제 CV의 fold당 학습 = 약 {n_full_fit}행, 검증 = 약 {tr.sum() // 5}행')
    idx = np.arange(len(y))
    sub = {}
    for frac in (0.1, 0.2):
        sub[frac], _ = train_test_split(idx, train_size=frac, stratify=y, random_state=RS)
    rest = np.setdiff1d(idx, sub[0.2])
    val = np.random.default_rng(RS).choice(rest, 2000, replace=False)
    print(f'  20% 층화 서브샘플 {len(sub[0.2])}행 (10%: {len(sub[0.1])}행), predict = 서브샘플 밖 2,000행\n')
    rows = []
    plan = [(n, {}) for n in ('lr', 'rf', 'et', 'hgb', 'knn', 'svm')] + [('svm', {'probability': True})]
    for mode in MODES:
        yy = y if mode == 'multi' else np.where(y == UNMON, UNMON, 1)
        for name, params in plan:
            fracs = (0.1, 0.2) if name == 'svm' else (0.2,)
            t = {}
            for frac in fracs:
                make = lambda: make_model(name, d['names'], d['all_names'], **params)
                t0 = time.time(); m = make().fit(X[sub[frac]], yy[sub[frac]]); t1 = time.time()
                score(m, X[val]); t2 = time.time()
                t[frac] = (t1 - t0, t2 - t1)
            # 외삽: 학습 시간 ∝ n^a. SVM은 10%·20%에서 a를 잰다. 나머지는 a = 1(대략 선형)로 둔다
            a = np.log(t[0.2][0] / t[0.1][0]) / np.log(2) if 0.1 in t and t[0.1][0] > 0.05 else 1.0
            fit_full = t[0.2][0] * (n_full_fit / len(sub[0.2])) ** a
            # predict ∝ 검증 행 수 × (SVM은 서포트 벡터 수 ∝ 학습 행 수)
            pred_full = t[0.2][1] * (tr.sum() / 5 / 2000) * ((n_full_fit / len(sub[0.2])) if name in ('svm', 'knn') else 1)
            per_fold = fit_full + pred_full
            total = per_fold * (1 if name in ('svm', 'lr', 'knn') else 5)  # SVM·LR·k-NN은 fold 5개 병렬
            desc = model_desc(name, **params)
            rows.append((mode, desc, t[0.2][0], t[0.2][1], a, per_fold, total))
            print(f'  {mode:6s} {desc:42s} 20% fit {t[0.2][0]:7.1f}s predict(2k) {t[0.2][1]:6.1f}s | '
                  f'지수 a {a:.2f} | 전체 추정 fold당 {per_fold / 60:6.1f}분, 5-fold {total / 60:6.1f}분'
                  + ('' if name not in ('svm', 'lr', 'knn') else ' (fold 병렬)'), flush=True)
    print('\n추정은 대략값이다(학습 시간 ∝ n^a, SVM·k-NN의 predict ∝ 학습 행 수). 병렬 fold 5개는 코어 8개 중 5개를 쓴다.')


# ---------------- E7 모델 비교 ----------------
E7_MODELS = ('lr', 'rf', 'svm', 'svmp', 'et', 'hgb', 'hgbl2', 'knn')
E7_PLOT = tuple(n for n in E7_MODELS if n != 'hgb')  # 기본 HGB는 multi에서 발산해 그림에서 뺀다 (CSV에는 남김)
PARALLEL = ('svm', 'svmp', 'lr', 'knn')  # 단일 스레드 모델 → fold 5개 병렬


def e7(subset, models=E7_MODELS):
    tag = '-subset' if subset else ''
    d = load(subset)
    n_mon = len(np.unique(d['y'][d['y'] != UNMON]))
    fs = f'cand{len(d["names"])}_skip14'
    print(f'E7 dataset={d["dataset"]}: 피처 {fs} ({" ".join(CAND134)}), monitored 사이트 {n_mon}개(데이터에서 셈) '
          f'+ unmon -1 → multi {n_mon + 1}클래스 / binary 2클래스. split visit, CV5 OOF. HGB = LightGBM 대체')
    out = {}
    for mode in MODES:
        for name in models:
            exp_id = f'E7{tag}-{mode}-{name}'
            print(f'\n## {exp_id}: {model_desc(name)}', flush=True)
            make = lambda: make_model(name, d['names'], d['all_names'])
            S, classes, tf, tp = cv_scores(d['X'], labels(d, mode), d['fold_visit'], make,
                                           parallel=name in PARALLEL)
            save_scores(exp_id, d, mode, S, classes)
            note = (f'; fit {np.mean(tf):.0f}s/fold predict {np.mean(tp):.0f}s/fold'
                    + ('; scores = SVC decision_function (not probability)' if name == 'svm' else '')
                    + ('; scores = Platt probability (internal 5-fold, rs=42)' if name == 'svmp' else '')
                    + ('; k=5 -> 6 distinct scores, coarse thresholds' if name == 'knn' else '')
                    + ('; HGB replaces LightGBM (not installed)' if name in ('hgb', 'hgbl2') else '')
                    + ('; only l2_regularization=1 changed from default (default diverges in multi)' if name == 'hgbl2' else ''))
            out[(mode, name)] = assess(exp_id, d, mode, S, classes, fs, model_desc(name) + f' {mode}', extra_notes=note)
            out[(mode, name)]['fit_s'] = float(np.mean(tf))
    print('\n## 요약 (visit, CV5 OOF, 후보 134개). TPR = 사이트까지 맞힘(binary는 탐지)')
    print(f'  {"mode":6s} {"model":8s} | ROC-AUC (sd) | best TPR@FPR≤0.1% 1% 5% | per-fold TPR@1% (sd) | '
          f'argmax TPR FPR prec macroF1 acc | fit s/fold')
    for (mode, name), s in out.items():
        a = s['argmax']
        print(f'  {mode:6s} {LABEL[name]:8s} | {s["roc_auc"]:.4f} ({s["roc_auc_sd"]:.4f}) | '
              f'{s["best_fpr0.1"]:.4f} {s["best_fpr1"]:.4f} {s["best_fpr5"]:.4f} | '
              f'{np.mean(s["tpr_fpr1_folds"]):.4f} ({np.std(s["tpr_fpr1_folds"]):.4f}) | '
              f'{a["TPR"]:.4f} {a["FPR"]:.4f} {a["precision"]:.4f} {a["macro_f1"]:.4f} {a["accuracy"]:.4f} '
              f'| {s["fit_s"]:.0f}')
    if set(models) >= set(E7_PLOT):
        plot_e7(out, tag)


def plot_e7(out, tag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    ink, muted, grid, base = '#0b0b0b', '#898781', '#e1e0d9', '#c3c2b7'
    pct = FuncFormatter(lambda v, _: f'{v * 100:g}%')
    plt.rcParams.update({'font.family': ['AppleGothic', 'sans-serif'], 'axes.unicode_minus': False, 'font.size': 10})
    # 판정점 3개 = 같은 지표의 세 조건 → 한 색상 계열(진하기)로. 모델은 y축 이름이 식별한다
    pts = [('fpr0.1', 'FPR ≤ 0.1%', '#9ec5f4'), ('fpr1', 'FPR ≤ 1%', '#2a78d6'), ('fpr5', 'FPR ≤ 5%', '#0d3a73')]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, facecolor='#fcfcfb')
    order = sorted(E7_PLOT, key=lambda n: out[('multi', n)]['best_fpr1'])
    for ax, mode in zip(axes, MODES):
        for i, n in enumerate(order):
            s = out[(mode, n)]
            xs = [s[f'best_{k}'] for k, _, _ in pts]
            ax.plot([min(xs), max(xs)], [i, i], color=base, lw=2, zorder=1, solid_capstyle='round')
            for (k, lab, c), x in zip(pts, xs):
                ax.scatter(x, i, s=60, color=c, edgecolor='#fcfcfb', linewidth=1.5, zorder=3,
                           label=lab if i == 0 else None)
            ax.annotate(f'{s["best_fpr1"] * 100:.1f}', (s['best_fpr1'], i), xytext=(0, 8), textcoords='offset points',
                        ha='center', color=ink, fontsize=8.5)
        ax.set_yticks(range(len(order)), [LABEL[n] for n in order])
        ax.set_xlim(0, 1); ax.xaxis.set_major_formatter(pct)
        ax.set_xlabel('TPR (pooled CV 곡선 위 최고값)' + (' — 사이트까지 맞힘' if mode == 'multi' else ' — 탐지'), color=ink)
        ax.set_title(f'{mode} ({"96" if mode == "multi" else "2"}클래스)', color=ink, loc='left')
        R._style(ax, ink, muted, grid, base)
        ax.tick_params(axis='y', colors=ink)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc='lower center', ncol=3, labelcolor=ink, fontsize=9)
    fig.suptitle('E7 모델 비교 — 후보 134개(leak 제거 피처), 기본 하이퍼파라미터, visit CV5 OOF. 숫자 = TPR@FPR≤1% (%)',
                 color=ink, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out1 = os.path.join(R.ROOT, 'figures', f'ow_e7_models{tag}.png')
    fig.savefig(out1, dpi=150); plt.close(fig)

    # ROC (binary·multi, 로그 FPR) + PR (binary만)
    colors = {'rf': '#2a78d6', 'et': '#eb6834', 'hgb': '#1baf7a', 'svm': '#7a5bd6', 'svmp': '#d65b9e', 'hgbl2': '#1baf7a',
              'lr': '#c9a227', 'knn': '#898781'}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor='#fcfcfb')
    for ax, mode in zip(axes[:2], MODES):
        for n in E7_PLOT:
            c = out[(mode, n)]['curve']
            keep = c['FPR'] >= 1e-4
            ax.plot(c['FPR'][keep], c['TPR'][keep], color=colors[n], lw=2, label=LABEL[n])
        ax.set_xscale('log'); ax.set_xlim(1e-4, 1); ax.set_ylim(0, 1.02)
        ax.xaxis.set_major_formatter(pct); ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ''))
        ax.yaxis.set_major_formatter(pct)
        for x in (0.001, 0.01, 0.05):
            ax.axvline(x, color=muted, lw=1, ls=(0, (3, 3)))
        ax.set_title(f'ROC — {mode}' + (' (TPR = 사이트까지 맞힘)' if mode == 'multi' else ''), color=ink, loc='left')
        ax.set_xlabel('FPR (로그 축)', color=ink)
        R._style(ax, ink, muted, grid, base)
    axes[0].set_ylabel('TPR', color=ink)
    axes[0].legend(frameon=False, loc='lower right', labelcolor=ink, fontsize=9)
    ax = axes[2]
    for n in E7_PLOT:
        c = out[('binary', n)]['curve']
        ax.plot(c['TPR'], c['precision'], color=colors[n], lw=2, label=LABEL[n])
    ax.set_xlim(0, 1); ax.set_ylim(0.6, 1.005)
    ax.xaxis.set_major_formatter(pct); ax.yaxis.set_major_formatter(pct)
    ax.set_title('PR — binary (실험 비율 mon:unmon ≈ 1.9:1)', color=ink, loc='left')
    ax.set_xlabel('Recall = TPR', color=ink); ax.set_ylabel('Precision', color=ink)
    R._style(ax, ink, muted, grid, base)
    fig.suptitle('E7 곡선 — 후보 134개(leak 제거), visit CV5 OOF, 규칙 B 점수(monitored 최댓값). SVM(RBF) 점수 = decision_function, +Platt = 확률',
                 color=ink, x=0.01, ha='left')
    fig.tight_layout()
    out2 = os.path.join(R.ROOT, 'figures', f'ow_e7_roc_pr{tag}.png')
    fig.savefig(out2, dpi=150); plt.close(fig)
    print(f'\n그림: {out1}\n      {out2}', flush=True)


def replot_e7(subset):
    """cache/oof의 E7 점수로 그림만 다시 그린다 (CSV에 쓰지 않는다)."""
    tag = '-subset' if subset else ''
    out = {}
    for mode in MODES:
        for name in E7_PLOT:
            z = np.load(os.path.join(R.CACHE, 'oof', f'E7{tag}-{mode}-{name}.npz'))
            c = R.rule_curve(z['proba'], z['classes'], z['y'], 'B')
            out[(mode, name)] = {'curve': c, **{f'best_{suf}': best_at(c, 'TPR', x) for x, suf in FPR_TARGETS}}
    plot_e7(out, tag)


# ---------------- E8 피처 그룹 분석 (모델 고정: RF, LR) ----------------
BASE_GROUPS = ('count', 'timing', 'order', 'burst', 'cumul')     # A팀 기본 55개의 원래 구분 (4/18/10/11/12)
EXTRA_GROUPS = ('early', 'bseq', 'cumulabs', 'dir50', 'qconst')  # 추가 피처군. qconst = leak 제거 후 15~28번째 셀 구간
FULL147 = ['@all'] + [f'@{g}' for g in EXTRA_GROUPS]             # 55 + 6 + 24 + 13 + 36 + 13
E8_MODELS = ('rf', 'lr')
GROUP_LABEL = {'count': 'Count', 'timing': 'Timing', 'order': 'Order·Conc.', 'burst': 'Burst', 'cumul': 'CUMUL',
               'early': '@early', 'bseq': '@bseq', 'cumulabs': '@cumulabs', 'dir50': '@dir50', 'qconst': '@qconst'}


def e8_sets(names):
    """(set_id, 피처 이름 리스트, kind, group, 기준 set_id). 그룹 정의는 A팀 experiment.groups 그대로.
    기본 그룹: 단독 / All55에서 제거. 추가 그룹: 단독 / All55에 추가 / Full147에서 제거."""
    g = ex.groups(names)
    all55, full = g['all'], ex.select(FULL147, names)
    sets = [('all55', all55, 'ref', None, None)]
    for k in BASE_GROUPS:
        sets.append((f'only-{k}', g[k], 'alone', k, None))
        sets.append((f'all55-{k}', [f for f in all55 if f not in g[k]], 'remove', k, 'all55'))
    sets.append(('full147', full, 'ref', None, None))
    for k in EXTRA_GROUPS:
        sets.append((f'only-{k}', g[k], 'alone', k, None))
        sets.append((f'all55+{k}', all55 + g[k], 'add', k, 'all55'))
        sets.append((f'full147-{k}', [f for f in full if f not in g[k]], 'remove', k, 'full147'))
    return sets


def e8(subset):
    import json
    tag = '-subset' if subset else ''
    d = load(subset, FULL147)
    sets = e8_sets(d['all_names'])
    pos = {f: i for i, f in enumerate(d['names'])}
    print(f'E8 dataset={d["dataset"]}: leak 제거 피처(skip14). 그룹 크기 '
          + ', '.join(f'{GROUP_LABEL[k]} {len(f)}' for s, f, kind, k, _ in sets if kind == 'alone')
          + f' | All55 {len(sets[0][1])}, Full147 {len(d["names"])}. 모델 RF, LR. split visit, CV5 OOF')
    print('그룹 정의: A팀 experiment.groups. bseq_1~3·dir_001~005·n_bursts_total은 A팀 정의대로 어느 그룹에도 넣지 않았다')
    summ = {}
    for mode in MODES:
        for name in E8_MODELS:
            for sid, feats, kind, grp, ref in sets:
                exp_id = f'E8{tag}-{mode}-{name}-{sid}'
                print(f'\n## {exp_id}: {len(feats)}개', flush=True)
                X = d['X'][:, [pos[f] for f in feats]]
                make = lambda: make_model(name, feats, d['all_names'])
                S, classes, tf, _ = cv_scores(X, labels(d, mode), d['fold_visit'], make, parallel=name in PARALLEL)
                s = assess(exp_id, d, mode, S, classes, f'{sid}({len(feats)})_skip14',
                           model_desc(name) + f' {mode}', extra_notes=f'; E8 {kind} group={grp}; fit {np.mean(tf):.0f}s/fold')
                summ[f'{mode}|{name}|{sid}'] = {
                    'mode': mode, 'model': name, 'set': sid, 'kind': kind, 'group': grp, 'ref': ref, 'n': len(feats),
                    'roc_auc': s['roc_auc'], 'roc_auc_sd': s['roc_auc_sd'],
                    **{f'best_{suf}': s[f'best_{suf}'] for _, suf in FPR_TARGETS},
                    'folds': [float(v) for v in s['tpr_fpr1_folds']], 'argmax_TPR': s['argmax']['TPR']}
    out = os.path.join(R.CACHE, f'e8_summary{tag}.json')
    json.dump(summ, open(out, 'w'), indent=1)
    e8_report(summ)
    plot_e8(summ, tag)


def e8_report(summ):
    """요약 표. Δ = 기준 대비 fold 평균 TPR@FPR≤1% 차이. 유의 = |Δ| > max(1%p, 기준 fold sd) (A팀 기준)."""
    for mode in MODES:
        for name in E8_MODELS:
            S = {v['set']: v for v in summ.values() if v['mode'] == mode and v['model'] == name}
            m = lambda sid: float(np.mean(S[sid]['folds']))
            print(f'\n## 요약 {mode} {LABEL[name]} — TPR@FPR≤1% (fold 평균 ± sd), ROC-AUC. '
                  f'Δ 유의 기준 = max(1%p, 기준 fold sd)')
            for ref in ('all55', 'full147'):
                r = S[ref]
                print(f'  기준 {ref:8s} ({r["n"]:3d}개): {m(ref):.4f} ± {np.std(r["folds"]):.4f}  AUC {r["roc_auc"]:.4f}')
            print(f'  {"그룹":12s} | 단독: 개수 TPR@1% AUC | 제거: Δ (유의) | 추가(55+g): Δ (유의)')
            for g in BASE_GROUPS + EXTRA_GROUPS:
                a = S[f'only-{g}']
                rm = S[f'all55-{g}'] if g in BASE_GROUPS else S[f'full147-{g}']
                thr = lambda ref: max(0.01, float(np.std(S[ref]['folds'])))
                dr = m(rm['set']) - m(rm['ref'])
                line = (f'  {GROUP_LABEL[g]:12s} | {a["n"]:3d} {m(a["set"]):.4f} {a["roc_auc"]:.4f} | '
                        f'{rm["ref"]}-g {dr:+.4f} ({"유의" if abs(dr) > thr(rm["ref"]) else "같은 수준"})')
                if g in EXTRA_GROUPS:
                    ad = S[f'all55+{g}']
                    da = m(ad['set']) - m('all55')
                    line += f' | {da:+.4f} ({"유의" if abs(da) > thr("all55") else "같은 수준"})'
                print(line)


def plot_e8(summ, tag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    ink, muted, grid, base = '#0b0b0b', '#898781', '#e1e0d9', '#c3c2b7'
    plt.rcParams.update({'font.family': ['AppleGothic', 'sans-serif'], 'axes.unicode_minus': False, 'font.size': 9.5})
    pct = FuncFormatter(lambda v, _: f'{v * 100:g}%')
    pp = FuncFormatter(lambda v, _: f'{v * 100:+.0f}%p' if v else '0')
    groups = BASE_GROUPS + EXTRA_GROUPS
    cols = [(m, n) for m in MODES for n in E8_MODELS]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.6), sharey=True, facecolor='#fcfcfb')
    y = np.arange(len(groups))[::-1]
    for j, (mode, name) in enumerate(cols):
        S = {v['set']: v for v in summ.values() if v['mode'] == mode and v['model'] == name}
        m = lambda sid: float(np.mean(S[sid]['folds']))
        # 위: 단독 사용
        ax = axes[0, j]
        vals = [m(f'only-{g}') for g in groups]
        ax.barh(y, vals, height=0.6, color=['#2a78d6' if g in BASE_GROUPS else '#eb6834' for g in groups])
        for yy, v in zip(y, vals):
            ax.annotate(f'{v * 100:.0f}', (v, yy), xytext=(3, 0), textcoords='offset points', va='center',
                        color=ink, fontsize=8)
        for ref, ls in (('all55', (0, (3, 3))), ('full147', '-')):
            ax.axvline(m(ref), color=muted, lw=1, ls=ls)
        ax.set_xlim(0, 1.08); ax.xaxis.set_major_formatter(pct)
        ax.set_title(f'{mode} · {LABEL[name]} — 단독 사용', color=ink, loc='left')
        # 아래: 제거(기본 그룹: All55에서 / 추가 그룹: Full147에서) Δ, 회색 띠 = 유의 기준
        ax2 = axes[1, j]
        dr = [m(f'all55-{g}') - m('all55') if g in BASE_GROUPS else m(f'full147-{g}') - m('full147') for g in groups]
        da = [np.nan if g in BASE_GROUPS else m(f'all55+{g}') - m('all55') for g in groups]
        ax2.barh(y + 0.17, dr, height=0.32, color='#2a78d6', zorder=3, label='제거 (기본: All55−g / 추가: Full147−g)')
        ax2.barh(y - 0.17, da, height=0.32, color='#eb6834', zorder=3, label='추가 (All55+g)')
        band = max(0.01, float(np.std(S['all55']['folds'])), float(np.std(S['full147']['folds'])))
        ax2.axvspan(-band, band, color='#efeee8', lw=0, zorder=0)
        ax2.axvline(0, color=base, lw=1, zorder=2)
        lim = max(0.05, np.nanmax(np.abs(dr + da)) * 1.15)
        ax2.set_xlim(-lim, lim); ax2.xaxis.set_major_formatter(pp)
        ax2.xaxis.set_major_locator(MaxNLocator(nbins=4, symmetric=True))
        ax2.set_title(f'{mode} · {LABEL[name]} — 기준 대비 Δ', color=ink, loc='left')
        for a in (ax, ax2):
            R._style(a, ink, muted, grid, base)
            a.tick_params(axis='y', colors=ink)
    axes[0, 0].set_yticks(y, [GROUP_LABEL[g] for g in groups])
    axes[0, 0].set_ylabel('그룹 (파랑 = 기본 55개 구분, 주황 = 추가 피처군)', color=ink)
    h, l = axes[1, 0].get_legend_handles_labels()
    fig.legend(h, l + [], frameon=False, loc='lower center', ncol=2, labelcolor=ink, fontsize=9)
    fig.suptitle('E8 피처 그룹 — TPR@FPR≤1% (fold 평균, multi = 사이트까지 맞힘). 위 점선 = All55, 실선 = Full147. '
                 '아래 회색 띠 = 유의 기준 max(1%p, fold sd). leak 제거 피처, visit CV5 OOF',
                 color=ink, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = os.path.join(R.ROOT, 'figures', f'ow_e8_groups{tag}.png')
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f'\n그림: {out}', flush=True)


def replot_e8(subset):
    import json
    tag = '-subset' if subset else ''
    summ = json.load(open(os.path.join(R.CACHE, f'e8_summary{tag}.json')))
    e8_report(summ)
    plot_e8(summ, tag)


# ---------------- E9 피처 선택 분석 (모델 고정: RF) ----------------
from sklearn.base import BaseEstimator, TransformerMixin


def eta_squared(X, y):
    """열마다 η² = 클래스 간 제곱합 / 전체 제곱합 (A팀 check_features.eta2와 같은 값, 벡터화). 분산 0이면 0."""
    X = np.asarray(X, np.float64)
    _, inv = np.unique(y, return_inverse=True)
    n = np.bincount(inv).astype(np.float64)
    mu = X.mean(0)
    sums = np.zeros((len(n), X.shape[1]))
    np.add.at(sums, inv, X)
    ssb = (n[:, None] * (sums / n[:, None] - mu) ** 2).sum(0)
    sst = ((X - mu) ** 2).sum(0)
    return np.divide(ssb, sst, out=np.zeros_like(ssb), where=sst > 0)


class EtaSelect(TransformerMixin, BaseEstimator):
    """학습 fold의 (X, y)로만 η²를 계산해 상위(bottom=True면 하위) k개 열을 고른다. 분산 0인 열은 하위 후보에서 뺀다.
    mon_only=True: unmon(-1) 행을 빼고 사이트 라벨만으로 계산한다 (A팀 정의 = 사이트 구분력)"""
    def __init__(self, k=20, bottom=False, mon_only=False):
        self.k, self.bottom, self.mon_only = k, bottom, mon_only

    def fit(self, X, y):
        m = (np.asarray(y) != UNMON) if self.mon_only else slice(None)
        self.eta_ = eta_squared(np.asarray(X)[m], np.asarray(y)[m])
        if self.bottom:
            ok = np.flatnonzero(np.asarray(X).std(0) > 0)
            self.idx_ = ok[np.argsort(self.eta_[ok], kind='stable')[:self.k]]
        else:
            self.idx_ = np.argsort(-self.eta_, kind='stable')[:self.k]
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.idx_]


def corr_reduce_idx(X, thr=0.95):
    """열 순서대로 훑으며, 이미 남긴 열과 |상관| >= thr인 열은 버린다 (쌍마다 앞의 것을 남김). 분산 0인 열의 상관은 0으로 본다."""
    X = np.asarray(X, np.float64)
    sd = X.std(0)
    Z = np.where(sd > 0, (X - X.mean(0)) / np.where(sd > 0, sd, 1), 0)
    C = np.abs(Z.T @ Z / len(X))
    keep = []
    for j in range(X.shape[1]):
        if not keep or C[j, keep].max() < thr:
            keep.append(j)
    return np.array(keep)


class CorrReduce(TransformerMixin, BaseEstimator):
    """학습 fold로만 상관을 계산해 |r| >= thr인 쌍에서 하나씩 뺀다."""
    def __init__(self, thr=0.95):
        self.thr = thr

    def fit(self, X, y=None):
        self.idx_ = corr_reduce_idx(X, self.thr)
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.idx_]


E9_SETS = [  # (set_id, 피처 지정, 선택기, 방식) — 선택기는 fold 안에서 fit한다
    ('all376', None, None, MODES),
    ('cand134', CAND134, None, MODES),
    ('cand134-qconst', CAND134 + ['-@qconst'], None, MODES),
    ('eta-top20', None, lambda: EtaSelect(20), MODES),        # η² 라벨 = 그 방식의 학습 라벨 (multi는 unmon 포함 96클래스)
    ('eta-top50', None, lambda: EtaSelect(50), MODES),
    ('eta-top100', None, lambda: EtaSelect(100), MODES),
    ('eta-bottom20', None, lambda: EtaSelect(20, bottom=True), MODES),  # 대조: A팀 함정(상위 20 < 하위 20) 확인용
    ('eta-site-top20', None, lambda: EtaSelect(20, mon_only=True), ('multi',)),  # A팀 정의(사이트만). binary는 사이트 라벨이 없다
    ('eta-site-top50', None, lambda: EtaSelect(50, mon_only=True), ('multi',)),
    ('eta-site-top100', None, lambda: EtaSelect(100, mon_only=True), ('multi',)),
    ('reduced095', None, lambda: CorrReduce(0.95), MODES),
]
# 앞 14셀을 버린 직후 구간(15~28번째 셀)을 가리키는 피처: @qconst 13 + A팀 그룹에 없는 dir_001~005·bseq_1~3 + in_burst_1~3
EARLY_WINDOW = (['first10_out'] + [f'bseq_{j}' for j in range(1, 7)] + [f'dir_{j:03d}' for j in range(1, 15)]
                + ['in_burst_1', 'in_burst_2', 'in_burst_3'])


def e9(subset):
    import json
    from sklearn.pipeline import make_pipeline as mp
    tag = '-subset' if subset else ''
    full = R.load_pkl(14, subset)
    d = dict(full, all_names=full['names'])
    names = d['names']
    pos = {f: i for i, f in enumerate(names)}
    fold = d['fold_visit']
    tr = fold >= 0
    print(f'E9 dataset={d["dataset"]}: leak 제거 피처(skip14) 전체 {len(names)}개, 모델 RF(n=200, rs=42) 고정, visit CV5 OOF')
    print('η²: 각 fold의 학습 행으로만 계산, 라벨 = 그 방식의 학습 라벨 (multi 96클래스 / binary 2클래스). '
          'Reduced: 학습 fold의 |r| >= 0.95 쌍에서 열 순서상 뒤의 것을 뺀다')
    summ, sel_log = {}, {}
    for mode in MODES:
        y_all = labels(d, mode)
        for sid, spec, selector, modes in E9_SETS:
            if mode not in modes:
                continue
            feats = names if spec is None else ex.select(spec, names)
            X = d['X'][:, [pos[f] for f in feats]]
            exp_id = f'E9{tag}-{mode}-rf-{sid}'
            print(f'\n## {exp_id}: 입력 {len(feats)}개' + (f' → fold 안에서 {selector().__class__.__name__}' if selector else ''),
                  flush=True)
            make = (lambda: make_model('rf', feats, names)) if selector is None else \
                (lambda: mp(selector(), make_model('rf', feats, names)))
            # fold별 선택 결과 기록 (모델 학습과 같은 학습 행, 같은 결정적 계산)
            chosen = []
            if selector is not None:
                for k in range(5):
                    fit = tr & (fold != k)
                    chosen.append([feats[i] for i in selector().fit(X[fit], y_all[fit]).idx_])
                n_sel = [len(c) for c in chosen]
                inter = set.intersection(*map(set, chosen))
                print(f'  fold별 선택 개수 {n_sel}, 5개 fold 공통 {len(inter)}개')
                sel_log[f'{mode}|{sid}'] = chosen
            S, classes, tf, _ = cv_scores(X, y_all, fold, make, parallel=False)
            fs = f'{sid}({np.mean([len(c) for c in chosen]):.0f} of {len(feats)})_skip14' if chosen else f'{sid}({len(feats)})_skip14'
            s = assess(exp_id, d, mode, S, classes, fs, model_desc('rf') + f' {mode}',
                       extra_notes=f'; E9 selection inside each training fold; fit {np.mean(tf):.0f}s/fold')
            summ[f'{mode}|{sid}'] = {'mode': mode, 'set': sid, 'n': (float(np.mean([len(c) for c in chosen]))
                                                                     if chosen else len(feats)),
                                     'roc_auc': s['roc_auc'], 'roc_auc_sd': s['roc_auc_sd'],
                                     **{f'best_{suf}': s[f'best_{suf}'] for _, suf in FPR_TARGETS},
                                     'folds': [float(v) for v in s['tpr_fpr1_folds']],
                                     'precision_fpr1': s['fpr1']['precision'], 'fit_s': float(np.mean(tf))}
    json.dump({'summary': summ, 'selected': sel_log}, open(os.path.join(R.CACHE, f'e9_summary{tag}.json'), 'w'), indent=1)
    corr = e9_corr(d, sel_log, tag)
    e9_report(summ)
    plot_e9(summ, corr, tag)


def e9_corr(d, sel_log, tag):
    """η² 상위 20개(fold 0 선택)의 상관행렬 — fold 0의 학습 행으로 계산. A팀 함정('상위 20개 = 기록 길이') 확인용."""
    names = d['names']
    pos = {f: i for i, f in enumerate(names)}
    fold = d['fold_visit']
    fit = (fold >= 0) & (fold != 0)
    out = {}
    print('\n## η² 상위/하위 20개의 상관 (fold 0 학습 행). |r| 통계는 서로 다른 쌍 190개 기준')
    for mode in MODES:
        for sid in ('eta-top20', 'eta-site-top20', 'eta-bottom20'):
            if f'{mode}|{sid}' not in sel_log:
                continue
            f = sel_log[f'{mode}|{sid}'][0]
            C = np.corrcoef(d['X'][fit][:, [pos[c] for c in f]].astype(np.float64).T)
            C = np.nan_to_num(C)
            iu = np.triu_indices(len(f), 1)
            a = np.abs(C[iu])
            g = ex.groups(names)
            grp = {c: k for k in GW_GROUPS for c in g.get(k, [])}
            grp.update({c: '15~28셀' for c in EARLY_WINDOW})
            comp = pd.Series([grp.get(c, '기타') for c in f]).value_counts()
            length = np.mean([c in g['length'] for c in f])
            print(f'  {mode:6s} {sid:14s}: 평균 |r| {a.mean():.3f}, |r| >= 0.9 쌍 {np.mean(a >= 0.9):.0%}, >= 0.95 {np.mean(a >= 0.95):.0%} '
                  f'| 길이 계열 {length:.0%} | 구성 ' + ', '.join(f'{k} {v}' for k, v in comp.items()))
            print(f'      {" ".join(f)}')
            out[(mode, sid)] = (f, C)
    return out


GW_GROUPS = ['count', 'timing', 'order', 'burst', 'cumul', 'early', 'bseq', 'cumulabs', 'cumul100', 'bseqlong', 'dir', 'qconst', 'const']


def e9_report(summ):
    print('\n## 요약 E9 (RF 고정) — TPR@FPR≤1% fold 평균 ± sd. Δ = cand134 대비, 유의 = |Δ| > max(1%p, cand134 fold sd)')
    for mode in MODES:
        ref = summ[f'{mode}|cand134']
        thr = max(0.01, float(np.std(ref['folds'])))
        print(f'  {mode}')
        for sid, _, _, modes in E9_SETS:
            if mode not in modes:
                continue
            s = summ[f'{mode}|{sid}']
            dlt = np.mean(s['folds']) - np.mean(ref['folds'])
            print(f'    {sid:15s} {s["n"]:6.1f}개 | {np.mean(s["folds"]):.4f} ± {np.std(s["folds"]):.4f} | '
                  f'Δ {dlt:+.4f} ({"기준" if sid == "cand134" else "유의" if abs(dlt) > thr else "같은 수준"}) | '
                  f'AUC {s["roc_auc"]:.4f} | TPR@0.1% {s["best_fpr0.1"]:.4f} @5% {s["best_fpr5"]:.4f} | '
                  f'prec@1% {s["precision_fpr1"]:.4f} | fit {s["fit_s"]:.0f}s')


def plot_e9(summ, corr, tag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    ink, muted, grid, base = '#0b0b0b', '#898781', '#e1e0d9', '#c3c2b7'
    plt.rcParams.update({'font.family': ['AppleGothic', 'sans-serif'], 'axes.unicode_minus': False, 'font.size': 9.5})
    pct = FuncFormatter(lambda v, _: f'{v * 100:g}%')
    sets = [s for s, *_ in E9_SETS]
    lab = {'all376': 'All 376', 'cand134': 'A팀 후보 134', 'cand134-qconst': '후보 134 − @qconst (121)',
           'eta-top20': '$\\eta^2$ 상위 20', 'eta-top50': '$\\eta^2$ 상위 50', 'eta-top100': '$\\eta^2$ 상위 100',
           'eta-bottom20': '$\\eta^2$ 하위 20 (대조)', 'reduced095': 'Reduced (|r|<0.95)',
           'eta-site-top20': '$\\eta^2$(사이트만) 상위 20', 'eta-site-top50': '$\\eta^2$(사이트만) 상위 50',
           'eta-site-top100': '$\\eta^2$(사이트만) 상위 100'}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True, facecolor='#fcfcfb')
    y = np.arange(len(sets))[::-1]
    for ax, mode in zip(axes, MODES):
        for yy, sid in zip(y, sets):
            if f'{mode}|{sid}' not in summ:
                continue
            s = summ[f'{mode}|{sid}']
            m, sd = np.mean(s['folds']), np.std(s['folds'])
            ref = sid == 'cand134'
            ax.errorbar(m, yy, xerr=sd, fmt='o', ms=8, color='#2a78d6' if not ref else '#eb6834',
                        ecolor=base, elinewidth=2, capsize=0, mec='#fcfcfb', mew=1.5, zorder=3)
            ax.annotate(f'{m * 100:.1f}  ({s["n"]:.0f}개)', (m, yy), xytext=(0, 9), textcoords='offset points',
                        ha='center', color=ink, fontsize=8.5)
        rm = np.mean(summ[f'{mode}|cand134']['folds'])
        ax.axvline(rm, color=muted, lw=1, ls=(0, (3, 3)))
        ax.set_xlim(0, 1); ax.xaxis.set_major_formatter(pct)
        ax.set_title(f'{mode} · RF', color=ink, loc='left')
        ax.set_xlabel('TPR@FPR≤1% (fold 평균, 가로선 = ±1 sd)' + (' — 사이트까지 맞힘' if mode == 'multi' else ' — 탐지'), color=ink)
        R._style(ax, ink, muted, grid, base)
        ax.tick_params(axis='y', colors=ink)
    axes[0].set_yticks(y, [lab[s] for s in sets])
    fig.suptitle('E9 피처셋 비교 — RF 고정, leak 제거 피처, visit CV5 OOF. $\\eta^2$·Reduced는 학습 fold 안에서 선택. 주황 = 기준(후보 134), 점선 = 기준값',
                 color=ink, x=0.01, ha='left')
    fig.tight_layout()
    out1 = os.path.join(R.ROOT, 'figures', f'ow_e9_featuresets{tag}.png')
    fig.savefig(out1, dpi=150); plt.close(fig)

    # $\\eta^2$ 상위 20 상관행렬 (multi, binary)
    panels = [('binary', 'eta-top20', 'binary $\\eta^2$(2클래스)'), ('multi', 'eta-top20', 'multi $\\eta^2$(96클래스)'),
              ('multi', 'eta-site-top20', '$\\eta^2$(사이트만, A팀 정의)')]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.8), facecolor='#fcfcfb')
    for ax, (mode, sid, title) in zip(axes, panels):
        f, C = corr[(mode, sid)]
        im = ax.imshow(np.abs(C), cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(len(f)), f, rotation=90, fontsize=7.5, color=ink)
        ax.set_yticks(range(len(f)), f, fontsize=7.5, color=ink)
        a = np.abs(C[np.triu_indices(len(f), 1)])
        ax.set_title(f'{title} 상위 20\n평균 |r| {a.mean():.2f}, |r| ≥ 0.9인 쌍 {np.mean(a >= 0.9):.0%}', color=ink, loc='left', fontsize=10)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.015)
    cb.set_label('|상관계수|', color=ink); cb.outline.set_visible(False)
    fig.suptitle('E9 $\\eta^2$ 상위 20개 피처의 상관 (fold 0 학습 행, fold 0에서 선택된 20개, $\\eta^2$ 순서)', color=ink, x=0.01, y=0.98, ha='left')
    out2 = os.path.join(R.ROOT, 'figures', f'ow_e9_eta_corr{tag}.png')
    fig.savefig(out2, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f'\n그림: {out1}\n      {out2}', flush=True)


def replot_e9(subset):
    import json
    tag = '-subset' if subset else ''
    z = json.load(open(os.path.join(R.CACHE, f'e9_summary{tag}.json')))
    full = R.load_pkl(14, subset)
    corr = e9_corr(full, z['selected'], tag)
    e9_report(z['summary'])
    plot_e9(z['summary'], corr, tag)


# ---------------- E10 하이퍼파라미터 튜닝 (RF + 후보 134) ----------------
RF_GRID = {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, None],
           'min_samples_split': [2, 5, 10], 'class_weight': [None, 'balanced']}
RF_DEFAULT = {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 2, 'class_weight': None}


def scorer_tpr_fpr1(est, X, y):
    """open-world scorer: 이 검증 fold에서 규칙 B로 FPR <= 1%가 되는 최고 TPR (multi는 사이트까지 맞힘).
    E7~E9의 'fold별 TPR@FPR≤1%'와 같은 계산이다. accuracy가 아니다."""
    return best_at(R.rule_curve(est.predict_proba(X), est.classes_, np.asarray(y), 'B'), 'TPR', 0.01)


def scorer_macro_f1(est, X, y):
    from sklearn.metrics import f1_score
    return f1_score(y, est.predict(X), labels=est.classes_, average='macro', zero_division=0)


def fold_tpr1(S, classes, y, f):
    return [best_at(R.rule_curve(S[f == k], classes, y[f == k], 'B'), 'TPR', 0.01) for k in range(5)]


def e10(subset):
    import json
    from sklearn.model_selection import GridSearchCV, PredefinedSplit
    tag = '-subset' if subset else ''
    d = load(subset)
    fs = f'cand{len(d["names"])}_skip14'
    fold = d['fold_visit']
    tr = fold >= 0
    out = {}
    print(f'E10 dataset={d["dataset"]}: RF + 후보 {len(d["names"])}개 (leak 제거 피처). 격자 {RF_GRID} = '
          f'{np.prod([len(v) for v in RF_GRID.values()])}개 조합, visit 5-fold (기존 split 그대로).')
    print('scoring = fold별 TPR@FPR≤1% (multi는 사이트까지 맞힘). 보조 = macro F1(argmax). C팀 튜닝(closed-world accuracy)과 기준이 다르다')
    for mode in MODES:
        y = labels(d, mode)
        print(f'\n######## {mode} ########', flush=True)
        t0 = time.time()
        gs = GridSearchCV(RandomForestClassifier(n_jobs=-1, random_state=RS), RF_GRID,
                          scoring={'tpr1': scorer_tpr_fpr1, 'macro_f1': scorer_macro_f1}, refit=False,
                          cv=PredefinedSplit(fold[tr]), n_jobs=1, error_score='raise')
        gs.fit(d['X'][tr], y[tr])
        cv = pd.DataFrame(gs.cv_results_)
        cols = ['param_' + k for k in RF_GRID]
        cv = cv[cols + ['mean_test_tpr1', 'std_test_tpr1'] + [f'split{k}_test_tpr1' for k in range(5)]
                + ['mean_test_macro_f1', 'std_test_macro_f1', 'mean_fit_time']].sort_values('mean_test_tpr1', ascending=False)
        os.makedirs(os.path.join(R.ROOT, 'results'), exist_ok=True)
        cv.to_csv(os.path.join(R.ROOT, 'results', f'ow_e10_grid_{mode}{tag}.csv'), index=False)
        print(f'  격자 탐색 {time.time() - t0:.0f}s. 상위 8개 (TPR@FPR≤1% fold 평균 ± sd | macro F1 | fit s/fold):')
        def pstr(r):
            return '{' + ', '.join(f'{k}={None if pd.isna(r["param_" + k]) else r["param_" + k]}' for k in RF_GRID) + '}'
        for _, r in cv.head(8).iterrows():
            print(f'    {pstr(r)}  {r.mean_test_tpr1:.4f} ± {r.std_test_tpr1:.4f} | '
                  f'{r.mean_test_macro_f1:.4f} | {r.mean_fit_time:.1f}')
        is_def = np.all([(cv['param_' + k].astype(object).where(cv['param_' + k].notna(), None) == v)
                         if v is not None else cv['param_' + k].isna() for k, v in RF_DEFAULT.items()], axis=0)
        rdef = cv[is_def].iloc[0]
        rank_def = int(np.flatnonzero(is_def)[0]) + 1
        print(f'  기본값 {RF_DEFAULT}: {rdef.mean_test_tpr1:.4f} ± {rdef.std_test_tpr1:.4f} (54개 중 {rank_def}위)')
        # 기본값 조합이 E7 RF와 같은 값인지 (같은 시드·같은 fold → 같아야 한다)
        if not subset:
            z = np.load(os.path.join(R.CACHE, 'oof', f'E7-{mode}-rf.npz'))
            e7f = fold_tpr1(z['proba'], z['classes'], z['y'], z['fold'])
            gdf = [rdef[f'split{k}_test_tpr1'] for k in range(5)]
            print(f'  일관성: 격자의 기본값 fold별 {np.round(gdf, 4).tolist()} vs E7 RF {np.round(e7f, 4).tolist()} '
                  f'→ {"같다" if np.allclose(gdf, e7f) else "다르다"}')
        best = cv.iloc[0]
        bp = {k: (None if pd.isna(best['param_' + k]) else best['param_' + k]) for k in RF_GRID}
        bp = {k: (int(v) if isinstance(v, (int, float, np.integer, np.floating)) and float(v).is_integer() else v)
              for k, v in bp.items()}
        # 격자 경계: 범위 끝에 붙은 값. min_samples_split=2는 sklearn의 최솟값이라 더 내릴 수 없다(경계 아님)
        edge = [k for k, v in bp.items() if k != 'class_weight' and v is not None and v in (RF_GRID[k][0], RF_GRID[k][-1])
                and not (k == 'min_samples_split' and v == 2)]
        # 판정: 개선폭(fold 평균)이 max(1%p, 기본값 fold sd)보다 큰가. 짝지은 fold 차이도 함께
        bf = np.array([best[f'split{k}_test_tpr1'] for k in range(5)])
        dfold = np.array([rdef[f'split{k}_test_tpr1'] for k in range(5)])
        delta, thr = bf.mean() - dfold.mean(), max(0.01, dfold.std())
        sig = delta > thr
        print(f'  최고 {bp}: {bf.mean():.4f} ± {bf.std():.4f} | Δ vs 기본값 {delta:+.4f} (기준 {thr:.4f}) → '
              f'{"유의한 개선" if sig else "유의한 개선 아님"} | 짝지은 fold 차이 {np.round(bf - dfold, 4).tolist()} '
              f'(5개 중 {int(np.sum(bf > dfold))}개 fold에서 개선)')
        print(f'  격자 경계: {"없음" if not edge else ", ".join(f"{k}={bp[k]} (범위 끝)" for k in edge)}'
              + ('' if bp['max_depth'] is not None else ' | max_depth=None은 제한 없음(확장 불필요)')
              + (' | min_samples_split=2는 허용 최솟값' if bp['min_samples_split'] == 2 else '')
              + ' — 확장 여부는 아래 파라미터별 최고값이 범위 끝으로 갈수록 계속 오르는지로 판단')
        # 파라미터별 주변 최댓값 (그림·해석용)
        marg = {k: {str(v): float(cv[(cv['param_' + k].isna() if v is None else cv['param_' + k] == v)].mean_test_tpr1.max())
                    for v in RF_GRID[k]} for k in RF_GRID}
        print(f'  파라미터별 최고값: {marg}')
        out[mode] = {'best': bp, 'best_folds': bf.tolist(), 'default_folds': dfold.tolist(), 'delta': float(delta),
                     'thr': float(thr), 'significant': bool(sig), 'edge': edge, 'marginal': marg, 'rank_default': rank_def,
                     'best_macro_f1': float(best.mean_test_macro_f1), 'default_macro_f1': float(rdef.mean_test_macro_f1)}
        # 튜닝 모델의 전체 OOF → 판정점 4개를 CSV에 (튜닝 전 = E7-*-rf 행)
        print(f'\n## E10{tag}-{mode}-rf-tuned: {bp}', flush=True)
        make = lambda: RandomForestClassifier(n_jobs=-1, random_state=RS, **bp)
        S, classes, tf, _ = cv_scores(d['X'], y, fold, make, parallel=False)
        save_scores(f'E10{tag}-{mode}-rf-tuned', d, mode, S, classes)
        desc = f'RF tuned {bp} {mode}'
        out[mode]['tuned'] = _slim(assess(f'E10{tag}-{mode}-rf-tuned', d, mode, S, classes, fs, desc,
                                          extra_notes=f'; E10 GridSearchCV(54, scoring=TPR@FPR<=1%) best; '
                                          f'delta vs default {delta:+.4f} ({"significant" if sig else "not significant"})'))
    json.dump(out, open(os.path.join(R.CACHE, f'e10_summary{tag}.json'), 'w'), indent=1, default=str)
    print('\n격자 결과: results/ow_e10_grid_{binary,multi}' + tag + '.csv')


def _slim(s):
    keep = ('roc_auc', 'roc_auc_sd', 'pr_auc', 'best_fpr0.1', 'best_fpr1', 'best_fpr5', 'tpr_fpr1_folds')
    r = {k: s[k] for k in keep}
    for p in ('argmax', 'fpr0.1', 'fpr1', 'fpr5'):
        r[p] = {k: s[p][k] for k in ('TPR', 'FPR', 'precision', 'macro_f1', 'det_TPR', 'wrong_site', 'FP') if k in s[p]}
    return r


def _params(p):
    """JSON에서 읽은 파라미터의 숫자 문자열('300')을 int로."""
    return {k: (int(v) if isinstance(v, str) and v.isdigit() else v) for k, v in p.items()}


def e10b(subset):
    """E10 후속: 튜닝된 최종 모델로 (1) E2 불균형 기법 재확인 (multi), (2) subpage 모드 1회 (binary·multi)."""
    import json
    tag = '-subset' if subset else ''
    z = json.load(open(os.path.join(R.CACHE, f'e10_summary{tag}.json')))
    d = load(subset)
    fs = f'cand{len(d["names"])}_skip14'
    out = {'imbalance': {}, 'subpage': {}}
    # (1) 불균형: 튜닝 파라미터 고정, 불균형 처리만 바꾼다. 언더샘플링은 학습 fold의 unmon만 줄인다(E2와 같은 함수)
    bp = _params(z['multi']['best'])
    print(f'E10b dataset={d["dataset"]}: 최종 모델 RF {bp} + {fs}')
    print('\n######## (1) 불균형 처리 재확인 (multi, visit) — 튜닝 파라미터 고정, 처리만 바꾼다 ########')
    y = labels(d, 'multi')
    variants = [('none', {'class_weight': None}, None), ('balanced', {'class_weight': 'balanced'}, None),
                ('under1x', {'class_weight': None}, 1), ('under5x', {'class_weight': None}, 5),
                ('under10x', {'class_weight': None}, 10)]
    for name, cw, ratio in variants:
        p = dict(bp, **cw)
        exp_id = f'E10{tag}-multi-rf-tuned-imb-{name}'
        print(f'\n## {exp_id}: {p}' + (f', unmon 언더샘플링 ×{ratio}' if ratio else ''), flush=True)
        S, classes, _ = R.cv_oof(d['X'], y, d['fold_visit'],
                                 lambda: RandomForestClassifier(n_jobs=-1, random_state=RS, **p),
                                 sampler=R.undersampler(ratio) if ratio else None)
        s = assess(exp_id, d, 'multi', S, classes, fs, f'RF tuned {p} multi', imbalance=name,
                   extra_notes='; E10 imbalance re-check with tuned params')
        out['imbalance'][name] = _slim(s)
    # (2) subpage 모드: 같은 최종 모델 (binary는 binary 튜닝 파라미터)
    print('\n######## (2) subpage 모드 (처음 보는 URL로 일반화) ########')
    for mode in MODES:
        p = _params(z[mode]['best'])
        exp_id = f'E10{tag}-{mode}-rf-tuned-subpage'
        print(f'\n## {exp_id}: {p}', flush=True)
        S, classes, _, _ = cv_scores(d['X'], labels(d, mode), d['fold_subpage'],
                                     lambda: RandomForestClassifier(n_jobs=-1, random_state=RS, **p), parallel=False)
        out['subpage'][mode] = _slim(assess(exp_id, d, mode, S, classes, fs, f'RF tuned {p} {mode}', split_mode='subpage',
                                            extra_notes='; E10 final model in subpage mode'))
    json.dump(out, open(os.path.join(R.CACHE, f'e10b_summary{tag}.json'), 'w'), indent=1, default=str)
    e10_report(tag)
    plot_e10(tag)


def e10_report(tag):
    import json
    a = json.load(open(os.path.join(R.CACHE, f'e10_summary{tag}.json')))
    b = json.load(open(os.path.join(R.CACHE, f'e10b_summary{tag}.json')))
    print('\n## 요약 E10 — 튜닝 전(기본값) vs 후, visit. TPR@FPR≤1% = fold 평균 ± sd')
    for mode in MODES:
        s = a[mode]
        print(f'  {mode}: 기본값 {np.mean(s["default_folds"]):.4f} ± {np.std(s["default_folds"]):.4f} → '
              f'튜닝 {np.mean(s["best_folds"]):.4f} ± {np.std(s["best_folds"]):.4f} | Δ {s["delta"]:+.4f} (기준 {s["thr"]:.4f}) '
              f'{"유의" if s["significant"] else "유의한 개선 아님"} | macro F1 {s["default_macro_f1"]:.4f} → {s["best_macro_f1"]:.4f} '
              f'| 최고 {s["best"]} | 기본값 순위 {s["rank_default"]}/54')
    print('\n## 불균형 재확인 (multi, 튜닝 파라미터): TPR@FPR≤1% fold 평균 | pooled 0.1/1/5% | argmax TPR FPR')
    for k, s in b['imbalance'].items():
        print(f'  {k:9s} {np.mean(s["tpr_fpr1_folds"]):.4f} ± {np.std(s["tpr_fpr1_folds"]):.4f} | '
              f'{s["best_fpr0.1"]:.4f} {s["best_fpr1"]:.4f} {s["best_fpr5"]:.4f} | {s["argmax"]["TPR"]:.4f} {s["argmax"]["FPR"]:.4f}')
    print('\n## subpage 모드 (최종 모델): visit 대비')
    for mode in MODES:
        sv, ss = a[mode]['tuned'], b['subpage'][mode]
        print(f'  {mode}: visit {np.mean(sv["tpr_fpr1_folds"]):.4f} → subpage {np.mean(ss["tpr_fpr1_folds"]):.4f} ± '
              f'{np.std(ss["tpr_fpr1_folds"]):.4f} | AUC {sv["roc_auc"]:.4f} → {ss["roc_auc"]:.4f} | '
              f'argmax TPR {sv["argmax"]["TPR"]:.4f} → {ss["argmax"]["TPR"]:.4f}')


def plot_e10(tag):
    import json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    a = json.load(open(os.path.join(R.CACHE, f'e10_summary{tag}.json')))
    b = json.load(open(os.path.join(R.CACHE, f'e10b_summary{tag}.json')))
    ink, muted, grid, base = '#0b0b0b', '#898781', '#e1e0d9', '#c3c2b7'
    plt.rcParams.update({'font.family': ['AppleGothic', 'sans-serif'], 'axes.unicode_minus': False, 'font.size': 9.5})
    pct = FuncFormatter(lambda v, _: f'{v * 100:g}%')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor='#fcfcfb', gridspec_kw={'width_ratios': [1, 1, 1.25]})
    # (a)(b) 튜닝 전후: fold별 점 + 평균
    for ax, mode in zip(axes[:2], MODES):
        s = a[mode]
        for i, (key, lab, c) in enumerate((('default_folds', '기본값', '#898781'), ('best_folds', '튜닝', '#2a78d6'))):
            v = np.array(s[key])
            ax.scatter(np.full(5, i) + np.linspace(-0.12, 0.12, 5), v, s=28, color=c, alpha=0.55, lw=0, zorder=3)
            ax.plot([i - 0.25, i + 0.25], [v.mean()] * 2, color=c, lw=2.5, zorder=4)
            ax.annotate(f'{v.mean() * 100:.1f}', (i + 0.27, v.mean()), va='center', color=ink, fontsize=9)
        for k in range(5):
            ax.plot([-0.12 + 0.06 * k, 1 - 0.12 + 0.06 * k], [s['default_folds'][k], s['best_folds'][k]],
                    color=grid, lw=1, zorder=2)
        ax.set_xticks([0, 1], ['기본값', '튜닝'])
        ax.set_xlim(-0.5, 1.6)
        ax.yaxis.set_major_formatter(pct)
        ax.set_title(f'{mode}: Δ {s["delta"] * 100:+.1f}%p — '
                     + ('유의' if s['significant'] else '유의한 개선 아님'), color=ink, loc='left')
        R._style(ax, ink, muted, grid, base)
        ax.tick_params(axis='x', colors=ink)
    axes[0].set_ylabel('TPR@FPR≤1% (fold별 점, 가로선 = 평균)', color=ink)
    # (c) 불균형 재확인 (multi)
    ax = axes[2]
    names = list(b['imbalance'])
    lab = {'none': '처리 없음', 'balanced': "class_weight='balanced'", 'under1x': '언더샘플링 ×1',
           'under5x': '언더샘플링 ×5', 'under10x': '언더샘플링 ×10'}
    y = np.arange(len(names))[::-1]
    for yy, n in zip(y, names):
        s = b['imbalance'][n]
        v = np.array(s['tpr_fpr1_folds'])
        ax.errorbar(v.mean(), yy, xerr=v.std(), fmt='o', ms=8, color='#eb6834' if n == 'none' else '#2a78d6',
                    ecolor=base, elinewidth=2, mec='#fcfcfb', mew=1.5, zorder=3)
        ax.annotate(f'{v.mean() * 100:.1f}', (v.mean(), yy), xytext=(0, 9), textcoords='offset points', ha='center',
                    color=ink, fontsize=8.5)
    ax.set_yticks(y, [lab[n] for n in names])
    ax.set_ylim(-0.6, len(names) - 0.3)
    lo = min(np.mean(b['imbalance'][n]['tpr_fpr1_folds']) for n in names)
    ax.set_xlim(max(0, lo - 0.1), 1); ax.xaxis.set_major_formatter(pct)
    ax.set_xlabel('TPR@FPR≤1% (fold 평균 ± sd, 사이트까지 맞힘)', color=ink)
    ax.set_title('multi: 불균형 처리 재확인 (튜닝 파라미터)', color=ink, loc='left')
    R._style(ax, ink, muted, grid, base)
    ax.tick_params(axis='y', colors=ink)
    fig.suptitle('E10 RF 튜닝 (후보 134, leak 제거 피처, visit CV5) — scoring = TPR@FPR≤1%. 회색 선 = 같은 fold끼리 연결',
                 color=ink, x=0.01, ha='left')
    fig.tight_layout()
    out = os.path.join(R.ROOT, 'figures', f'ow_e10_tuning{tag}.png')
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f'\n그림: {out}', flush=True)


def replot_e10(subset):
    tag = '-subset' if subset else ''
    e10_report(tag)
    plot_e10(tag)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('exp', choices=['time', 'e7', 'e7plot', 'e8', 'e8plot', 'e9', 'e9plot', 'e10', 'e10b', 'e10plot'])
    ap.add_argument('--subset', action='store_true')
    ap.add_argument('--models', nargs='+', default=None, help='e7: 이 모델만 돌린다')
    a = ap.parse_args()
    if a.exp == 'e7' and a.models:
        e7(a.subset, tuple(a.models))
    else:
        {'time': timing, 'e7': e7, 'e7plot': replot_e7, 'e8': e8, 'e8plot': replot_e8, 'e9': e9, 'e9plot': replot_e9, 'e10': e10, 'e10b': e10b,
         'e10plot': replot_e10}[a.exp](a.subset)
