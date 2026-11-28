"""open-world 실험 러너 (B 파트). 결과는 results/openworld_results.csv에 한 줄씩 누적한다.

사용법 (레포 루트에서, stdout만 로그로):
  python -I -W ignore -X utf8 code/ow_run.py e1 --subset   > logs/ow_e1_subset.log   # 파이프라인 검증 (mon 25 / unmon 5000)
  python -I -W ignore -X utf8 code/ow_run.py e1            > logs/ow_e1.log

데이터 (pkl)
  피처 = features/ow_skip14_*.csv (앞 14셀 버림, 376개. reports/openworld.md 0절)
  split = splits/pkl_*.csv (reports/openworld.md 1절). 평가는 train 내부 5-fold CV의 out-of-fold 예측. test는 쓰지 않는다
  --subset: mon site < 25, unmon row < 5000. TrafficSliver 축소판(mon_25 / unmon_5000)과 같은 규모의 pkl 부분집합
  피처 행렬은 cache/pkl_skip14_*.npy로 캐싱한다 (git 제외)

출력
  results/openworld_results.csv  고정 컬럼 COLUMNS (C팀과 공유. 바꾸지 말 것)
  results/confusion/<exp_id>.csv confusion matrix (행 = 실제, 열 = 예측, 첫 행/열 = -1)
  cache/oof/<exp_id>.npz         out-of-fold predict_proba (E2 임계값·E3 sweep에서 재학습 없이 쓴다)
"""
import argparse, json, os, sys, time

import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ow_metrics import UNMON, apply_threshold, fpr_threshold, mon_score, ow_metrics, unmon_confusion

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results', 'openworld_results.csv')
CACHE = os.path.join(ROOT, 'cache')
BASE_COLUMNS = ['exp_id', 'dataset', 'split_mode', 'feature_set', 'model', 'imbalance', 'threshold', 'n_classes',
                'TPR', 'FPR', 'precision', 'recall', 'macro_f1', 'notes']
# E7부터 뒤에 추가한 컬럼 (기존 14개는 순서·이름 그대로. E1~E6 행은 빈 값). 정의는 reports/openworld.md 2절
#   accuracy     전체 샘플 중 라벨(-1 포함)이 정확히 맞은 비율. unmon이 많으면 부풀려지므로 해석 근거로 쓰지 않는다
#   f1           2·precision·TPR/(precision+TPR) (TP 정의는 기존 그대로: 사이트까지 맞힘)
#   roc_auc      탐지 ROC-AUC: 점수 = monitored 최댓값(binary는 P(mon)), 정답 = mon인가. 임계값과 무관
#   pr_auc       같은 점수의 average precision
#   det_TPR      (TP + wrong_site) / mon 수. 사이트는 틀려도 monitored로 잡은 비율
#   site_id_rate TP / (TP + wrong_site). monitored로 잡은 mon 중 사이트까지 맞힌 비율 (multi만)
EXTRA_COLUMNS = ['accuracy', 'f1', 'roc_auc', 'pr_auc', 'det_TPR', 'site_id_rate']
COLUMNS = BASE_COLUMNS + EXTRA_COLUMNS
RANDOM_STATE = 42
SPLIT_MODES = ('visit', 'subpage')


# ---------------- 데이터 ----------------
def _cached_matrix(stem):
    """features/<stem>.csv → (X float32, 피처 이름). cache/<stem>.npy에 캐싱."""
    npy, js = os.path.join(CACHE, f'{stem}.npy'), os.path.join(CACHE, f'{stem}.names.json')
    csv = os.path.join(ROOT, 'features', f'{stem}.csv')
    if os.path.exists(npy) and os.path.getmtime(npy) >= os.path.getmtime(csv):
        return np.load(npy), json.load(open(js))
    df = pd.read_csv(csv)
    names = [c for c in df.columns if c not in ('site', 'url', 'inst')]
    X = df[names].to_numpy(np.float32)
    os.makedirs(CACHE, exist_ok=True)
    np.save(npy, X); json.dump(names, open(js, 'w'))
    if 'site' in df:
        np.save(os.path.join(CACHE, f'{stem}.ids.npy'), df[['site', 'url', 'inst']].to_numpy())
    return X, names


PKL_SKIP0 = ('features', 'features_extra', 'features_extra2')  # leak 포함 피처 CSV (features/<stem>_mon.csv)


def load_pkl(skip=14, subset=False):
    """mon + unmon을 이어 붙인 dict. 행 순서 = [mon 19,000 (CSV 순서), unmon 10,000 (CSV 순서)]."""
    if skip == 0:  # leak 포함: A팀 기존 CSV 3개(55 + 150 + 171)를 같은 순서로 이어 붙인다 = ow_skip14와 같은 이름·순서
        parts_m = [_cached_matrix(f'{s}_mon') for s in PKL_SKIP0]
        parts_u = [_cached_matrix(f'{s}_unmon') for s in PKL_SKIP0]
        Xm, names = np.hstack([x for x, _ in parts_m]), sum((n for _, n in parts_m), [])
        Xu, names_u = np.hstack([x for x, _ in parts_u]), sum((n for _, n in parts_u), [])
        ids = np.load(os.path.join(CACHE, f'{PKL_SKIP0[0]}_mon.ids.npy'))
        for st in PKL_SKIP0[1:]:
            assert (np.load(os.path.join(CACHE, f'{st}_mon.ids.npy')) == ids).all(), f'{st}: 행 순서 불일치'
    else:
        Xm, names = _cached_matrix(f'ow_skip{skip}_mon')
        Xu, names_u = _cached_matrix(f'ow_skip{skip}_unmon')
        ids = np.load(os.path.join(CACHE, f'ow_skip{skip}_mon.ids.npy'))
    assert names == names_u, 'mon/unmon 피처 열이 다르다'
    sm = pd.read_csv(os.path.join(ROOT, 'splits', 'pkl_mon.csv'))
    su = pd.read_csv(os.path.join(ROOT, 'splits', 'pkl_unmon.csv'))
    assert (sm[['site', 'url', 'inst']].to_numpy() == ids).all(), 'split 파일과 피처 CSV의 행이 어긋난다'
    assert len(su) == len(Xu) and (su.row.values == np.arange(len(Xu))).all()
    d = {
        'X': np.vstack([Xm, Xu]), 'names': names,
        'y': np.concatenate([sm.site.values, np.full(len(Xu), UNMON)]),
        'is_test': np.concatenate([sm.is_test.values, su.is_test.values]),
        'fold_visit': np.concatenate([sm.fold_visit.values, su.fold.values]),
        'fold_subpage': np.concatenate([sm.fold_subpage.values, su.fold.values]),
        'row': np.concatenate([sm.row.values, su.row.values]),
    }
    if subset:
        keep = np.concatenate([sm.site.values < 25, su.row.values < 5000])
        d = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    d['dataset'] = 'pkl_m25_u5000' if subset else 'pkl'
    return d


# ---------------- 실험 ----------------
def rf():
    return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)


def cv_oof(X, y, fold, make_model, sampler=None):
    """fold >= 0인 행(train)에서 5-fold out-of-fold predict_proba. 반환: (proba, classes, fold별 시간).
    sampler(y_fit, k) → bool mask: 학습 fold 안에서만 행을 고른다(언더샘플링). 검증 fold는 항상 전부 쓴다."""
    tr = fold >= 0
    Xt, yt, ft = X[tr], y[tr], fold[tr]
    classes = np.unique(yt)
    proba = np.zeros((len(yt), len(classes)), np.float32)
    secs = []
    for k in range(5):
        t0 = time.time()
        fit = np.flatnonzero(ft != k)
        if sampler is not None:
            fit = fit[sampler(yt[fit], k)]
        m = clone(make_model()).fit(Xt[fit], yt[fit])
        assert (m.classes_ == classes).all(), f'fold {k}: train에 빠진 클래스가 있다'
        proba[ft == k] = m.predict_proba(Xt[ft == k])
        secs.append(time.time() - t0)
        nu = int((yt[fit] == UNMON).sum())
        print(f'    fold {k}: fit {len(fit)} (unmon {nu}) / val {int((ft == k).sum())}  {secs[-1]:.0f}s', flush=True)
    return proba, classes, secs


def undersampler(ratio):
    """학습 fold의 unmon을 'monitored 클래스 하나의 평균 크기 × ratio'개로 무작위로 줄인다 (시드 42 + fold)."""
    def f(y_fit, k):
        mon = y_fit != UNMON
        n_keep = int(round(ratio * mon.sum() / len(np.unique(y_fit[mon]))))
        u = np.flatnonzero(~mon)
        keep = np.zeros(len(y_fit), bool)
        keep[mon] = True
        keep[np.random.default_rng(RANDOM_STATE + k).choice(u, min(n_keep, len(u)), replace=False)] = True
        return keep
    return f


def nested_fpr_pred(proba, classes, y, fold, target):
    """규칙 B 판정. 임계값은 fold k를 뺀 나머지 4개 fold의 unmon OOF 점수로 FPR <= target이 되게 정하고 fold k에 적용한다.
    평가하는 fold를 보고 임계값을 고르지 않으므로 누수가 없다. 반환: (pred, fold별 임계값)."""
    s, site = mon_score(proba, classes)
    pred, ts = np.full(len(y), UNMON), []
    for k in range(5):
        t = fpr_threshold(s[(fold != k) & (y == UNMON)], target)
        sel = fold == k
        pred[sel] = np.where(s[sel] >= t, site[sel], UNMON)
        ts.append(t)
    return pred, ts


def rule_curve(proba, classes, y, rule):
    """임계값을 점수 경계마다 훑은 곡선 (pooled). 반환: dict(t, FPR, TPR, precision, wrong_site_rate).
    TPR은 사이트까지 맞힌 것만 센다. 같은 점수는 한 점으로 묶는다(RF 확률은 1/200 단위라 동점이 많다).
      rule 'A': argmax가 monitored 사이트이고 전체 확률 최댓값 >= t일 때만 monitored (지시 규칙: max < t → -1)
      rule 'B': monitored 확률 최댓값 >= t이면 그 사이트"""
    proba, classes = np.asarray(proba), np.asarray(classes)
    if rule == 'A':
        site = classes[proba.argmax(1)]
        s = np.where(site != UNMON, proba.max(1), -np.inf)
    else:
        s, site = mon_score(proba, classes)
    o = np.argsort(-s, kind='stable')
    s, site, yy = s[o], site[o], y[o]
    ok = np.isfinite(s)
    s, site, yy = s[ok], site[ok], yy[ok]
    is_u, hit = yy == UNMON, (yy != UNMON) & (site == yy)
    wrong = (yy != UNMON) & (site != yy)
    last = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]  # 같은 점수 묶음의 마지막 인덱스
    n_u, n_m = (y == UNMON).sum(), (y != UNMON).sum()
    tp, fp, ws = np.cumsum(hit)[last], np.cumsum(is_u)[last], np.cumsum(wrong)[last]
    return {'t': s[last], 'FPR': fp / n_u, 'TPR': tp / n_m, 'precision': tp / (last + 1), 'wrong_site_rate': ws / n_m,
            'det_TPR': (tp + ws) / n_m}  # det_TPR: 사이트는 틀려도 monitored로 잡은 비율


def strict_curve(proba, classes, y):
    """규칙 B 곡선의 (FPR, TPR). E2 그림용."""
    c = rule_curve(proba, classes, y, 'B')
    return c['FPR'], c['TPR']


def fold_spread(y, pred, fold):
    """fold별 TPR·FPR의 표준편차 (판단 기준용)."""
    per = [ow_metrics(y[fold == k], pred[fold == k]) for k in range(5)]
    return np.std([p['TPR'] for p in per]), np.std([p['FPR'] for p in per])


def _migrate_columns():
    """기존 CSV(14개 컬럼)에 EXTRA_COLUMNS를 맨 뒤에 빈 값으로 붙인다. 기존 컬럼·값은 건드리지 않는다."""
    head = list(pd.read_csv(RESULTS, nrows=0).columns)
    if head == COLUMNS:
        return
    assert head == BASE_COLUMNS, f'결과 CSV 컬럼이 고정 컬럼과 다르다: {head}'
    df = pd.read_csv(RESULTS, dtype=str, keep_default_na=False)
    for c in EXTRA_COLUMNS:
        df[c] = ''
    df.to_csv(RESULTS, index=False)


def record(row):
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    row = {c: row.get(c, '') for c in COLUMNS}
    for c in ('TPR', 'FPR', 'precision', 'recall', 'macro_f1'):
        row[c] = round(float(row[c]), 6)
    for c in EXTRA_COLUMNS:
        if row[c] != '' and row[c] is not None:
            row[c] = round(float(row[c]), 6)
    new = not os.path.exists(RESULTS)
    if not new:
        _migrate_columns()
        assert list(pd.read_csv(RESULTS, nrows=0).columns) == COLUMNS, '결과 CSV 컬럼이 고정 컬럼과 다르다'
    pd.DataFrame([row], columns=COLUMNS).to_csv(RESULTS, mode='a', header=new, index=False)


def extra_metrics(r, y, pred, n_classes):
    """EXTRA_COLUMNS 중 판정(pred)에서 바로 나오는 것. roc_auc·pr_auc는 점수가 필요해 호출하는 쪽이 넣는다."""
    p, t = r['precision'], r['TPR']
    return {'accuracy': float(np.mean(np.asarray(y) == np.asarray(pred))),
            'f1': 2 * p * t / (p + t) if (p + t) > 0 else (0.0 if p == p else float('nan')),
            'det_TPR': r['det_TPR'],
            'site_id_rate': (r['TP'] / (r['TP'] + r['wrong_site']) if r['TP'] + r['wrong_site'] else float('nan'))
            if n_classes > 2 else ''}


def evaluate(exp_id, d, split_mode, pred, classes, threshold, feature_set, model, imbalance, extra_notes='',
             decision='argmax', extras=None):
    """pred = train 행의 OOF 판정. threshold = CSV에 적을 값('' = argmax). extras = EXTRA_COLUMNS에 더 넣을 값."""
    fold = d[f'fold_{split_mode}']
    tr = fold >= 0
    y, f = d['y'][tr], fold[tr]
    r = ow_metrics(y, pred, labels=classes)
    sd_tpr, sd_fpr = fold_spread(y, pred, f)
    cm, labels, cs = unmon_confusion(y, pred)
    os.makedirs(os.path.join(ROOT, 'results', 'confusion'), exist_ok=True)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(os.path.join(ROOT, 'results', 'confusion', f'{exp_id}.csv'))
    notes = (f'CV5 OOF train only (mon {r["n_mon"]}, unmon {r["n_unmon"]}); '
             f'decision={decision}; '
             f'TP {r["TP"]}, wrong_site {r["wrong_site"]} ({r["wrong_site_rate"]:.4f}), FN {r["FN"]}, FP {r["FP"]}; '
             f'det_TPR {r["det_TPR"]:.4f}; fold std TPR {sd_tpr:.4f} FPR {sd_fpr:.4f}' + extra_notes)
    record({'exp_id': exp_id, 'dataset': d['dataset'], 'split_mode': split_mode, 'feature_set': feature_set,
            'model': model, 'imbalance': imbalance, 'threshold': threshold, 'n_classes': len(classes),
            'TPR': r['TPR'], 'FPR': r['FPR'], 'precision': r['precision'], 'recall': r['recall'],
            'macro_f1': r['macro_f1'], 'notes': notes, **extra_metrics(r, y, pred, len(classes)), **(extras or {})})
    print(f'  [{exp_id}] TPR {r["TPR"]:.4f}  FPR {r["FPR"]:.4f}  precision {r["precision"]:.4f}  '
          f'macro_f1 {r["macro_f1"]:.4f}  | det_TPR {r["det_TPR"]:.4f}  wrong_site {r["wrong_site"]} '
          f'({r["wrong_site_rate"]:.2%})  FN {r["FN"]}  FP {r["FP"]}  | fold std TPR {sd_tpr:.4f} FPR {sd_fpr:.4f}')
    print(f'      -1 행(실제 unmon {r["n_unmon"]}): -1로 {cs["unmon_as_unmon"]}, mon으로 {cs["unmon_as_mon"]} '
          f'(FP가 많은 사이트 {cs["top_FP_sites"]})')
    print(f'      -1 열(-1로 예측된 mon): {cs["mon_as_unmon"]} = FN', flush=True)
    return r


def save_oof(exp_id, d, split_mode, proba, classes):
    os.makedirs(os.path.join(CACHE, 'oof'), exist_ok=True)
    fold = d[f'fold_{split_mode}']
    tr = fold >= 0
    np.savez_compressed(os.path.join(CACHE, 'oof', f'{exp_id}.npz'), proba=proba, classes=classes,
                        y=d['y'][tr], fold=fold[tr], row=d['row'][tr])


def feature_set_name(d, skip):
    return f'all{len(d["names"])}_' + ('skip14' if skip == 14 else 'leak(skip0)' if skip == 0 else f'skip{skip}')


def e1(subset, skip=14, exp='E1'):
    """E1 베이스라인: (N+1) 클래스, 전체 피처, RF, 불균형 처리 없음, argmax 판정.
    E4는 같은 함수를 skip=0(leak 포함 피처), exp='E4-E1'로 부른다."""
    d = load_pkl(skip, subset)
    n_mon = len(np.unique(d['y'][d['y'] != UNMON]))
    print(f'{exp} dataset={d["dataset"]} ({feature_set_name(d, skip)}): 피처 {len(d["names"])}개, monitored 사이트 {n_mon}개(데이터에서 셈) + unmon -1 '
          f'= {n_mon + 1}클래스, 행 {len(d["y"])} (test {int(d["is_test"].sum())}개는 쓰지 않음)', flush=True)
    tag = '-subset' if subset else ''
    for mode in SPLIT_MODES:
        exp_id = f'{exp}{tag}-{mode}'
        print(f'\n## {exp_id}', flush=True)
        proba, classes, secs = cv_oof(d['X'], d['y'], d[f'fold_{mode}'], rf)
        assert len(classes) == n_mon + 1
        save_oof(exp_id, d, mode, proba, classes)
        evaluate(exp_id, d, mode, apply_threshold(proba, classes, 0.0), classes, '', feature_set_name(d, skip),
                 'RF(n=200,rs=42)', 'none', f'; fit+predict {sum(secs):.0f}s')


E2_FPR = 0.01
E2_METHODS = {  # name → (imbalance 표기, 모델 생성, sampler). 'none'은 E1의 OOF를 그대로 쓴다
    'balanced': ('class_weight=balanced', lambda: rf().set_params(class_weight='balanced'), None),
    'under1x': ('undersample_unmon_1x', rf, undersampler(1)),
    'under5x': ('undersample_unmon_5x', rf, undersampler(5)),    # 보조: 언더샘플링 비율의 영향
    'under10x': ('undersample_unmon_10x', rf, undersampler(10)),  # 보조
}


def e2(subset):
    """E2 불균형 처리 비교. 같은 split·같은 RF·같은 피처에서 불균형 처리만 바꾼다.
    각 방법을 (1) argmax 판정 (2) FPR <= 1% 판정점(규칙 B, 다른 fold로 임계값 결정) 두 곳에서 잰다.
    임계값 조정 방법 = 처리 없는 모델(E1)에 (2)를 적용한 것."""
    d = load_pkl(14, subset)
    tag = '-subset' if subset else ''
    fs, model = f'all{len(d["names"])}_skip14', 'RF(n=200,rs=42)'
    for mode in SPLIT_MODES:
        fold = d[f'fold_{mode}']
        tr = fold >= 0
        y, f = d['y'][tr], fold[tr]
        e1z = np.load(os.path.join(CACHE, 'oof', f'E1{tag}-{mode}.npz'))
        assert (e1z['y'] == y).all() and (e1z['fold'] == f).all(), 'E1 OOF와 split이 다르다. E1을 먼저 다시 돌릴 것'
        runs = {'none': (e1z['proba'], e1z['classes'])}
        for name, (imb, make, sampler) in E2_METHODS.items():
            exp_id = f'E2{tag}-{name}-{mode}'
            print(f'\n## {exp_id}', flush=True)
            proba, classes, secs = cv_oof(d['X'], d['y'], fold, make, sampler)
            save_oof(exp_id, d, mode, proba, classes)
            runs[name] = (proba, classes)
            evaluate(exp_id, d, mode, apply_threshold(proba, classes, 0.0), classes, '', fs, model, imb,
                     f'; fit+predict {sum(secs):.0f}s')
        print(f'\n## {mode}: FPR <= {E2_FPR:.0%} 판정점 (규칙 B, 임계값은 나머지 4개 fold로 결정)', flush=True)
        for name, (proba, classes) in runs.items():
            imb = 'threshold(FPR<=1%)' if name == 'none' else E2_METHODS[name][0]
            exp_id = f'E2{tag}-{"threshold" if name == "none" else name}-{mode}-fpr1'
            pred, ts = nested_fpr_pred(proba, classes, y, f, E2_FPR)
            evaluate(exp_id, d, mode, pred, classes, round(float(np.mean(ts)), 4), fs, model, imb,
                     f'; threshold per fold {[round(t, 4) for t in ts]}; source OOF = '
                     f'{"E1" if name == "none" else "E2-" + name}', decision='max monitored proba >= t (rule B)')
    plot_e2(tag)


def plot_e2(tag=''):
    """TPR(사이트까지 맞힘)–FPR 곡선. 방법 3개(처리 없음 / balanced / 언더샘플링 1x), visit·subpage 나란히.
    cache/oof의 OOF 확률로 그린다(재학습 없음): python code/ow_run.py e2plot [--subset]"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    curves = {}
    for mode in SPLIT_MODES:
        for name in ('none', 'balanced', 'under1x'):
            src = f'E1{tag}-{mode}' if name == 'none' else f'E2{tag}-{name}-{mode}'
            z = np.load(os.path.join(CACHE, 'oof', f'{src}.npz'))
            curves[(mode, name)] = strict_curve(z['proba'], z['classes'], z['y'])
    ink, muted, grid, base = '#0b0b0b', '#898781', '#e1e0d9', '#c3c2b7'
    series = [('none', '처리 없음 (E1)', '#2a78d6'), ('balanced', 'class_weight=balanced', '#eb6834'),
              ('under1x', 'unmon 언더샘플링 1x', '#1baf7a')]
    plt.rcParams.update({'font.family': ['AppleGothic', 'sans-serif'], 'axes.unicode_minus': False, 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, facecolor='#fcfcfb')
    for ax, mode in zip(axes, SPLIT_MODES):
        ax.set_facecolor('#fcfcfb')
        marks = []
        for name, label, color in series:
            fpr, tpr = curves[(mode, name)]
            keep = fpr >= 1e-4
            ax.plot(fpr[keep], tpr[keep], color=color, lw=2, label=label)
            v = tpr[min(np.searchsorted(fpr, E2_FPR, side='right') - 1, len(tpr) - 1)]  # FPR <= 1%인 마지막 점
            marks.append([v, v, color])
        # FPR 1% 지점의 값 라벨: 위에서부터 최소 간격 0.045를 두어 겹치지 않게 한다
        marks.sort(key=lambda m: -m[0])
        for j in range(1, len(marks)):
            marks[j][1] = min(marks[j][1], marks[j - 1][1] - 0.045)
        for v, ty, color in marks:
            ax.plot([E2_FPR], [v], 'o', ms=8, color=color, mec='#fcfcfb', mew=2, zorder=3)
            ax.annotate(f'{v:.3f}', (E2_FPR, v), xytext=(E2_FPR * 0.55, ty), textcoords='data', ha='right',
                        va='center', color=ink, fontsize=8, zorder=4,
                        bbox=dict(boxstyle='round,pad=0.2', fc='#fcfcfb', ec='none'))
        ax.axvline(E2_FPR, color=muted, lw=1, ls=(0, (3, 3)))
        ax.text(E2_FPR, 0.02, ' FPR 1%', color=muted, fontsize=8)
        ax.set_xscale('log'); ax.set_xlim(1e-3, 1); ax.set_ylim(0, 1)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v * 100:g}%'))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ''))
        ax.set_title(f'{mode} 모드', color=ink, loc='left')
        ax.set_xlabel('FPR (unmon을 monitored로 판정한 비율, 로그 축)', color=ink)
        ax.grid(True, color=grid, lw=0.6); ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(base)
        ax.tick_params(colors=muted)
    axes[0].set_ylabel('TPR (사이트까지 맞힌 비율)', color=ink)
    axes[0].legend(frameon=False, loc='lower right', labelcolor=ink)
    fig.suptitle('E2 불균형 처리 비교 — 96클래스 RF, CV OOF (판정 규칙 B)', color=ink, x=0.01, ha='left')
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', f'ow_e2_imbalance{tag}.png')
    fig.savefig(out, dpi=150)
    print(f'\n그림: {out}', flush=True)


E3_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)
E3_FPR_TARGETS = (0.001, 0.005, 0.01, 0.02, 0.05)
E3_RATIOS = (1, 10, 100)  # mon : unmon = 1 : k 로 바꿨을 때의 precision
RULES = {'A': ('max proba < t -> -1 (rule A)', apply_threshold),
         'B': ('max monitored proba >= t (rule B)', None)}


def e3(subset, exp='E3', src='E1', fs='all376_skip14', plot=True):
    """E3 임계값 sweep. E2 승자(처리 없음 = E1 모델)의 OOF 확률, 재학습 없음.
    t = 0.05 ~ 0.95 (0.05 간격) × 규칙 A·B × visit·subpage → 결과 CSV. 곡선은 점수 경계마다 촘촘히."""
    from ow_metrics import apply_mon_threshold
    tag = '-subset' if subset else ''
    dataset = 'pkl_m25_u5000' if subset else 'pkl'
    curves = {}
    for mode in SPLIT_MODES:
        z = np.load(os.path.join(CACHE, 'oof', f'{src}{tag}-{mode}.npz'))
        P, C, y, f = z['proba'], z['classes'], z['y'], z['fold']
        print(f'\n## {exp}{tag}-{mode}: mon {(y != UNMON).sum()}, unmon {(y == UNMON).sum()}, 클래스 {len(C)} ({src} OOF)')
        print(f'  {"t":>5s} | {"규칙 A: TPR":>11s} {"FPR":>7s} {"prec":>6s} {"wrong":>6s} | '
              f'{"규칙 B: TPR":>11s} {"FPR":>7s} {"prec":>6s} {"wrong":>6s}')
        rows = {}
        for rule, (desc, fn) in RULES.items():
            fn = fn or apply_mon_threshold
            for t in E3_GRID:
                pred = fn(P, C, t)
                r = ow_metrics(y, pred, labels=C)
                sd_tpr, sd_fpr = fold_spread(y, pred, f)
                rows[(rule, t)] = r
                record({'exp_id': f'{exp}{tag}-{rule}-{mode}-t{t:.2f}', 'dataset': dataset, 'split_mode': mode,
                        'feature_set': fs, 'model': 'RF(n=200,rs=42)', 'imbalance': 'none',
                        'threshold': float(t), 'n_classes': len(C), 'TPR': r['TPR'], 'FPR': r['FPR'],
                        'precision': r['precision'], 'recall': r['recall'], 'macro_f1': r['macro_f1'],
                        'notes': f'CV5 OOF train only, source {src}{tag}-{mode}; decision={desc}; TP {r["TP"]}, '
                                 f'wrong_site {r["wrong_site"]}, FN {r["FN"]}, FP {r["FP"]}; det_TPR {r["det_TPR"]:.4f}; '
                                 f'fold std TPR {sd_tpr:.4f} FPR {sd_fpr:.4f}'})
        for t in E3_GRID:
            a, b = rows[('A', t)], rows[('B', t)]
            fmt = lambda r: f'{r["TPR"]:11.4f} {r["FPR"]:7.4f} {r["precision"]:6.4f} {r["wrong_site"]:6d}'
            print(f'  {t:5.2f} | {fmt(a)} | {fmt(b)}')
        for rule in RULES:
            curves[(mode, rule)] = c = rule_curve(P, C, y, rule)
            for t in E3_GRID:  # 촘촘한 곡선과 그리드 판정(ow_metrics)이 같은 점을 주는지 대조
                i = np.flatnonzero(c['t'] >= t)
                tpr, fpr = (c['TPR'][i.max()], c['FPR'][i.max()]) if len(i) else (0.0, 0.0)
                r = rows[(rule, t)]
                assert abs(tpr - r['TPR']) < 1e-12 and abs(fpr - r['FPR']) < 1e-12, (rule, t, tpr, r['TPR'])
            print(f'  규칙 {rule} 촘촘한 곡선: FPR 목표별 최고 TPR (pooled) — ' + ', '.join(
                f'≤{x:.1%}: ' + (f'{c["TPR"][c["FPR"] <= x].max():.4f}' if (c['FPR'] <= x).any() else '-')
                for x in E3_FPR_TARGETS) + f'  | 도달 가능한 최대 FPR {c["FPR"].max():.4f}')
        # 대표 임계값의 confusion matrix (규칙 A·B가 같아지는 0.5)
        cm, labels, cs = unmon_confusion(y, apply_threshold(P, C, 0.5))
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(
            os.path.join(ROOT, 'results', 'confusion', f'{exp}{tag}-{mode}-t0.50.csv'))
        # 현실 비율로 환산한 precision: TP·k... = TPR / (TPR + wrong_site_rate + k·FPR)
        print(f'  mon:unmon 비율별 precision (규칙 A·B 중 FPR 목표를 만족하는 최고 TPR 점, 실험 비율은 '
              f'{(y != UNMON).sum() / (y == UNMON).sum():.1f}:1)')
        for x in (0.001, 0.01, 0.05):
            for rule in RULES:
                c = curves[(mode, rule)]
                ok = np.flatnonzero(c['FPR'] <= x)
                if not len(ok):
                    continue
                i = ok[np.argmax(c['TPR'][ok])]
                tpr, fpr, wsr = c['TPR'][i], c['FPR'][i], c['wrong_site_rate'][i]
                prec = [tpr / (tpr + wsr + k * fpr) for k in E3_RATIOS]
                print(f'    FPR≤{x:.1%} 규칙 {rule}: t={c["t"][i]:.3f} TPR {tpr:.4f} FPR {fpr:.4f} → precision '
                      + ', '.join(f'1:{k} {p:.3f}' for k, p in zip(E3_RATIOS, prec)))
    if plot:
        plot_e3(curves, tag)
    return curves


def e4(subset):
    """E4 leak-free 비교. E1·E3를 leak 포함 피처(A팀 기존 CSV, 앞 14셀 그대로)로 다시 돌린다.
    피처 이름·순서·행·split·모델이 모두 같고 바뀌는 것은 '앞 14셀을 버렸는가' 하나다."""
    tag = '-subset' if subset else ''
    d0, d14 = load_pkl(0, subset), load_pkl(14, subset)
    assert d0['names'] == d14['names'] and (d0['y'] == d14['y']).all() and (d0['row'] == d14['row']).all()
    e1(subset, skip=0, exp='E4-E1')
    leak = e3(subset, exp='E4-E3', src='E4-E1', fs=feature_set_name(d0, 0), plot=False)
    free = {}
    for mode in SPLIT_MODES:
        z = np.load(os.path.join(CACHE, 'oof', f'E1{tag}-{mode}.npz'))
        for rule in RULES:
            free[(mode, rule)] = rule_curve(z['proba'], z['classes'], z['y'], rule)
    print('\n## 비교: FPR 목표별 최고 TPR (사이트까지 맞힘, pooled) — leak 포함 → leak 제거')
    for mode in SPLIT_MODES:
        for rule in RULES:
            a, b = leak[(mode, rule)], free[(mode, rule)]
            best = lambda c, x: c['TPR'][c['FPR'] <= x].max() if (c['FPR'] <= x).any() else float('nan')
            print(f'  {mode:7s} 규칙 {rule}: ' + ', '.join(
                f'FPR≤{x:.1%} {best(a, x):.4f} → {best(b, x):.4f}' for x in E3_FPR_TARGETS))
    # 모델이 무엇에 기대는가: train 전체(visit 기준 train 행)로 한 번 학습한 RF의 불순도 중요도 상위 10개
    print('\n## RF 중요도 상위 10 (train 전체, 96클래스). dir_008 = 8번째 셀 방향 (skip14에서는 22번째 셀)')
    for name, d in (('leak 포함', d0), ('leak 제거', d14)):
        tr = d['fold_visit'] >= 0
        m = rf().fit(d['X'][tr], d['y'][tr])
        o = np.argsort(-m.feature_importances_)[:10]
        print(f'  {name}: ' + ', '.join(f'{d["names"][i]} {m.feature_importances_[i]:.4f}' for i in o))
    plot_e4(leak, free, tag)


def plot_e4(leak, free, tag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    ink, muted, grid, base = '#0b0b0b', '#898781', '#e1e0d9', '#c3c2b7'
    plt.rcParams.update({'font.family': ['AppleGothic', 'sans-serif'], 'axes.unicode_minus': False, 'font.size': 10})
    series = [('leak 제거 (앞 14셀 버림, E1·E3)', free, '#2a78d6'), ('leak 포함 (A팀 기존 CSV, E4)', leak, '#eb6834')]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, facecolor='#fcfcfb')
    for ax, mode in zip(axes, SPLIT_MODES):
        for label, cv, color in series:
            cB, cA = cv[(mode, 'B')], cv[(mode, 'A')]
            keep = cB['FPR'] >= 1e-4
            ax.plot(cB['FPR'][keep], cB['TPR'][keep], color=color, lw=2, label=label)
            e = len(cA['FPR']) - 1  # 규칙 A 끝 = argmax 판정점
            ax.plot([max(cA['FPR'][e], 1e-4)], [cA['TPR'][e]], 'o', ms=8, color=color, mec='#fcfcfb', mew=2, zorder=3)
            ax.annotate(f'argmax {cA["TPR"][e]:.3f} @ {cA["FPR"][e]:.2%}', (max(cA['FPR'][e], 1e-4), cA['TPR'][e]),
                        xytext=(10, -14), textcoords='offset points', color=ink, fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.2', fc='#fcfcfb', ec='none'))
        ax.set_xscale('log'); ax.set_xlim(1e-4, 1); ax.set_ylim(0, 1)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v * 100:g}%'))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ''))
        ax.axvline(0.01, color=muted, lw=1, ls=(0, (3, 3)))
        ax.set_title(f'{mode} 모드', color=ink, loc='left')
        ax.set_xlabel('FPR (로그 축)', color=ink)
        _style(ax, ink, muted, grid, base)
    axes[0].set_ylabel('TPR (사이트까지 맞힘)', color=ink)
    axes[0].legend(frameon=False, loc='lower right', labelcolor=ink, fontsize=9)
    fig.suptitle('E4 leak 포함 vs 제거 — 96클래스 RF, CV OOF (선 = 규칙 B, 점 = argmax)', color=ink, x=0.01, ha='left')
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', f'ow_e4_leak{tag}.png')
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f'\n그림: {out}', flush=True)


E5_CAVEAT = 'CAVEAT binary: may be inflated by data artifact (mon/unmon collection & site-population gap; see report 0, E4, E5)'
LEAK_NAME = {14: 'free', 0: 'leak'}


def e5(subset):
    """E5 binary (보조): mon = 1 / unmon = -1. 같은 RF, 불균형 처리 없음. leak 제거·포함 × visit·subpage.
    점수 = P(mon). argmax(= t 0.5)와 FPR <= 1%·0.1% 판정점(나머지 4개 fold로 임계값). multi(E1·E4)와 같은 FPR에서 비교."""
    from sklearn.metrics import roc_auc_score
    tag = '-subset' if subset else ''
    curves = {}
    for skip in (14, 0):
        d = load_pkl(skip, subset)
        db = dict(d, y=np.where(d['y'] == UNMON, UNMON, 1))
        fs = feature_set_name(d, skip)
        for mode in SPLIT_MODES:
            exp_id = f'E5{tag}-{LEAK_NAME[skip]}-{mode}'
            print(f'\n## {exp_id} ({fs})', flush=True)
            fold = db[f'fold_{mode}']
            tr = fold >= 0
            y, f = db['y'][tr], fold[tr]
            proba, classes, secs = cv_oof(db['X'], db['y'], fold, rf)
            assert list(classes) == [UNMON, 1]
            save_oof(exp_id, db, mode, proba, classes)
            p1 = proba[:, 1]
            auc = roc_auc_score(y == 1, p1)
            auc_sd = np.std([roc_auc_score(y[f == k] == 1, p1[f == k]) for k in range(5)])
            print(f'  ROC-AUC {auc:.4f} (fold std {auc_sd:.4f})')
            base = f'; ROC-AUC {auc:.4f} (fold std {auc_sd:.4f}); {E5_CAVEAT}'
            evaluate(exp_id, db, mode, apply_threshold(proba, classes, 0.0), classes, '', fs, 'RF(n=200,rs=42) binary',
                     'none', base + f'; fit+predict {sum(secs):.0f}s', decision='argmax = P(mon) >= 0.5')
            for target, suf in ((0.01, 'fpr1'), (0.001, 'fpr0.1')):
                pred, ts = nested_fpr_pred(proba, classes, y, f, target)
                evaluate(f'{exp_id}-{suf}', db, mode, pred, classes, round(float(np.mean(ts)), 4), fs,
                         'RF(n=200,rs=42) binary', 'none',
                         base + f'; threshold per fold {[round(t, 4) for t in ts]}',
                         decision=f'P(mon) >= t, t from other 4 folds for FPR<={target:.1%}')
            curves[('bin', skip, mode)] = rule_curve(proba, classes, y, 'B')
            src = 'E1' if skip == 14 else 'E4-E1'
            z = np.load(os.path.join(CACHE, 'oof', f'{src}{tag}-{mode}.npz'))
            curves[('multi', skip, mode)] = rule_curve(z['proba'], z['classes'], z['y'], 'B')
            curves[('multiA', skip, mode)] = rule_curve(z['proba'], z['classes'], z['y'], 'A')
            s_multi, _ = mon_score(z['proba'], z['classes'])
            print(f'  (비교) multi {src} 점수의 탐지 ROC-AUC {roc_auc_score(z["y"] != UNMON, s_multi):.4f}')

    best = lambda c, key, x: c[key][c['FPR'] <= x].max() if (c['FPR'] <= x).any() else float('nan')
    print('\n## 같은 FPR에서 비교 (pooled): binary TPR | multi 탐지(det_TPR, 사이트 무관) | multi TPR(사이트까지)')
    for skip in (14, 0):
        for mode in SPLIT_MODES:
            b, m, mA = curves[('bin', skip, mode)], curves[('multi', skip, mode)], curves[('multiA', skip, mode)]
            print(f'  {LEAK_NAME[skip]:4s} {mode:7s} ' + ' | '.join(
                f'FPR≤{x:.1%}: bin {best(b, "TPR", x):.4f}, multi det {max(best(m, "det_TPR", x), best(mA, "det_TPR", x)):.4f}, '
                f'multi site {max(best(m, "TPR", x), best(mA, "TPR", x)):.4f}' for x in (0.001, 0.01, 0.05)))
    plot_e5(curves, tag)


def plot_e5(curves, tag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    ink, muted, grid, base = '#0b0b0b', '#898781', '#e1e0d9', '#c3c2b7'
    plt.rcParams.update({'font.family': ['AppleGothic', 'sans-serif'], 'axes.unicode_minus': False, 'font.size': 10})
    series = [(('bin', 14), 'TPR', 'binary, leak 제거', '#2a78d6'),
              (('multi', 14), 'det_TPR', 'multi 탐지(사이트 무관), leak 제거', '#eb6834'),
              (('bin', 0), 'TPR', 'binary, leak 포함', '#1baf7a')]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, facecolor='#fcfcfb')
    for ax, mode in zip(axes, SPLIT_MODES):
        for (kind, skip), key, label, color in series:
            c = curves[(kind, skip, mode)]
            keep = c['FPR'] >= 1e-4
            ax.plot(c['FPR'][keep], c[key][keep], color=color, lw=2, label=label)
        ax.set_xscale('log'); ax.set_xlim(1e-4, 1); ax.set_ylim(0, 1.02)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v * 100:g}%'))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ''))
        for x in (0.001, 0.01):
            ax.axvline(x, color=muted, lw=1, ls=(0, (3, 3)))
        ax.set_title(f'{mode} 모드', color=ink, loc='left')
        ax.set_xlabel('FPR (로그 축)', color=ink)
        _style(ax, ink, muted, grid, base)
    axes[0].set_ylabel('TPR (monitored로 탐지한 비율)', color=ink)
    axes[0].legend(frameon=False, loc='lower right', labelcolor=ink, fontsize=9)
    fig.suptitle('E5 binary (보조) — 탐지 곡선, CV OOF. leak 포함 곡선은 데이터 아티팩트로 부풀려진 값',
                 color=ink, x=0.01, ha='left')
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', f'ow_e5_binary{tag}.png')
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f'\n그림: {out}', flush=True)


def _style(ax, ink, muted, grid, base):
    ax.set_facecolor('#fcfcfb')
    ax.grid(True, color=grid, lw=0.6); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(base)
    ax.tick_params(colors=muted)


def plot_e3(curves, tag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    ink, muted, grid, base = '#0b0b0b', '#898781', '#e1e0d9', '#c3c2b7'
    col = {'A': '#2a78d6', 'B': '#eb6834'}
    lab = {'A': '규칙 A: 최댓값 < t → −1 (지시 규칙)', 'B': '규칙 B: monitored 최댓값 ≥ t'}
    pct = FuncFormatter(lambda v, _: f'{v * 100:g}%')
    plt.rcParams.update({'font.family': ['AppleGothic', 'sans-serif'], 'axes.unicode_minus': False, 'font.size': 10})

    def at_grid(c, rule):
        """0.05 간격 임계값에서의 점 (곡선 위). 규칙 A는 t 이상인 점수만 판정하므로 같은 계산이다."""
        idx = [np.flatnonzero(c['t'] >= t).max() if (c['t'] >= t).any() else None for t in E3_GRID]
        return [(t, i) for t, i in zip(E3_GRID, idx) if i is not None]

    # ---- 그림 1: 임계값 → TPR, FPR (small multiples: 위 TPR / 아래 FPR 로그) ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, facecolor='#fcfcfb')
    for j, mode in enumerate(SPLIT_MODES):
        for rule in RULES:
            c = curves[(mode, rule)]
            pts = at_grid(c, rule)
            ts = [t for t, _ in pts]
            tpr = [c['TPR'][i] for _, i in pts]
            fpr = [c['FPR'][i] for _, i in pts]
            axes[0, j].plot(ts, tpr, '-o', color=col[rule], lw=2, ms=5, mec='#fcfcfb', mew=0.8, label=lab[rule])
            keep = [k for k, v in enumerate(fpr) if v > 0]
            axes[1, j].plot([ts[k] for k in keep], [fpr[k] for k in keep], '-o', color=col[rule], lw=2, ms=5,
                            mec='#fcfcfb', mew=0.8, label=lab[rule])
        axes[0, j].set_title(f'{mode} 모드', color=ink, loc='left')
        axes[0, j].set_ylim(0, 1)
        axes[1, j].set_yscale('log'); axes[1, j].set_ylim(1e-4, 1)
        axes[1, j].yaxis.set_major_formatter(pct)
        axes[1, j].set_xlabel('임계값 t', color=ink)
        for i in range(2):
            _style(axes[i, j], ink, muted, grid, base)
            axes[i, j].axvline(0.5, color=muted, lw=1, ls=(0, (3, 3)))
        axes[0, j].text(0.51, 0.95, 't ≥ 0.5: 두 규칙이 같다', color=muted, fontsize=8, va='top')
    axes[0, 0].set_ylabel('TPR (사이트까지 맞힘)', color=ink)
    axes[1, 0].set_ylabel('FPR (로그 축, 0은 생략)', color=ink)
    axes[0, 0].legend(frameon=False, loc='lower left', labelcolor=ink, fontsize=9)
    fig.suptitle('E3 임계값 sweep (t = 0.05 간격) — E1 모델, CV OOF', color=ink, x=0.01, ha='left')
    fig.tight_layout()
    out1 = os.path.join(ROOT, 'figures', f'ow_e3_threshold{tag}.png')
    fig.savefig(out1, dpi=150); plt.close(fig)

    # ---- 그림 2: TPR–FPR 곡선 (위) · PR 곡선 (아래) ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor='#fcfcfb')
    for j, mode in enumerate(SPLIT_MODES):
        for rule in ('B', 'A'):  # A를 위에 그린다 (FPR 1% 이하에서 겹치는 구간)
            c = curves[(mode, rule)]
            pts = at_grid(c, rule)
            ax = axes[0, j]
            keep = c['FPR'] >= 1e-4
            ax.plot(c['FPR'][keep], c['TPR'][keep], color=col[rule], lw=2, label=lab[rule])
            gi = [i for _, i in pts if c['FPR'][i] >= 1e-4]
            ax.plot(c['FPR'][gi], c['TPR'][gi], 'o', color=col[rule], ms=5, mec='#fcfcfb', mew=0.8)
            ax = axes[1, j]
            ax.plot(c['TPR'], c['precision'], color=col[rule], lw=2, label=lab[rule])
            gi = [i for _, i in pts]
            ax.plot(c['TPR'][gi], c['precision'][gi], 'o', color=col[rule], ms=5, mec='#fcfcfb', mew=0.8)
        # 규칙 A의 끝점(= argmax) 표시
        cA = curves[(mode, 'A')]
        e = len(cA['FPR']) - 1
        axes[0, j].annotate('규칙 A 끝 = argmax', (cA['FPR'][e], cA['TPR'][e]), xytext=(14, -40),
                            textcoords='offset points', color=ink, fontsize=8,
                            arrowprops=dict(arrowstyle='-', color=muted, lw=0.8))
        ax = axes[0, j]
        ax.set_xscale('log'); ax.set_xlim(1e-4, 1); ax.set_ylim(0, 1)
        ax.xaxis.set_major_formatter(pct); ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ''))
        ax.axvline(0.01, color=muted, lw=1, ls=(0, (3, 3)))
        ax.set_title(f'{mode} 모드 — TPR–FPR', color=ink, loc='left')
        ax.set_xlabel('FPR (로그 축)', color=ink)
        ax = axes[1, j]
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_title(f'{mode} 모드 — Precision–Recall', color=ink, loc='left')
        ax.set_xlabel('Recall = TPR (사이트까지 맞힘)', color=ink)
        for i in range(2):
            _style(axes[i, j], ink, muted, grid, base)
    axes[0, 0].set_ylabel('TPR (사이트까지 맞힘)', color=ink)
    axes[1, 0].set_ylabel('Precision (실험 비율 mon:unmon ≈ 1.9:1)', color=ink)
    axes[0, 0].legend(frameon=False, loc='lower right', labelcolor=ink, fontsize=9)
    fig.suptitle('E3 곡선 — 선 = 점수 경계마다, 점 = t 0.05 간격', color=ink, x=0.01, ha='left')
    fig.tight_layout()
    out2 = os.path.join(ROOT, 'figures', f'ow_e3_roc_pr{tag}.png')
    fig.savefig(out2, dpi=150); plt.close(fig)
    print(f'\n그림: {out1}\n      {out2}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('exp', choices=['e1', 'e2', 'e2plot', 'e3', 'e4', 'e5'])
    ap.add_argument('--subset', action='store_true')
    a = ap.parse_args()
    if a.exp == 'e2plot':
        plot_e2('-subset' if a.subset else '')
    else:
        {'e1': e1, 'e2': e2, 'e3': e3, 'e4': e4, 'e5': e5}[a.exp](a.subset)
