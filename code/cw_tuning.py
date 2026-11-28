"""closed-world(95개 사이트) 모델 튜닝·비교·앙상블·최종 test (C파트, 이지원).

레포 규칙을 그대로 따른다.
  - 피처: experiment.py의 load()/select(). 기본 후보 131개 = @all -@const @early @bseq @cumulabs @dir50
  - 분할: test = load_data.mon_is_test (11/15 새 분할, URL마다 방문 2개), CV fold = inst % 5
          (splits/pkl_mon.csv의 is_test·fold_visit와 같다. 확인함)
  - 전처리: experiment.build() (트리는 원본, 그 외는 signed log + 표준화, --gw는 그룹 가중치)
  - test는 마지막 stage `test`에서 한 번만 쓴다. 한 번 돌리면 results/cw_test_LOCK이 생겨 다시 안 돈다.

모든 CV 결과는 OOF(out-of-fold) 예측을 results/cw_oof/<이름>.npz에 저장한다.
→ soft voting은 다시 학습하지 않고 OOF 확률을 평균해서 바로 계산한다(fold가 같으므로 VotingClassifier의 CV와 같다).
→ DF 1D-CNN(code/cw_df.py)도 같은 형식으로 OOF를 저장하므로 앙상블에 넣을 수 있다.

사용법 (레포 루트에서):
  python -I -W ignore -X utf8 code/cw_tuning.py base                       # ① 기본값 재측정 (새 분할)
  python -I -W ignore -X utf8 code/cw_tuning.py base -m rf et lgbm         #    일부 모델만
  python -I -W ignore -X utf8 code/cw_tuning.py tune -m lgbm -n 12         # ② random search (모델별)
  python -I -W ignore -X utf8 code/cw_tuning.py tune -m et -n 20 --fs tree181
  python -I -W ignore -X utf8 code/cw_tuning.py best -m lgbm --set random_state=1 --name lgbm_s1   # 최고 설정 재사용(시드·확률·피처셋)
  python -I -W ignore -X utf8 code/cw_tuning.py vote                       # ③ OOF로 soft voting 조합 비교
  python -I -W ignore -X utf8 code/cw_tuning.py ensemble --members lgbm_tuned et_tuned df_base   # mean/geo/가중/stacking 비교
  python -I -W ignore -X utf8 code/cw_tuning.py summary                    #    지금까지 결과 표
  python -I -W ignore -X utf8 code/cw_tuning.py test --final stack:lgbm_tuned+et_tuned+df_base   # ④ 월요일 1회 (방법: mean geo wvote stack)

결과:
  results/closedworld_results.csv   B파트 openworld_results.csv와 같은 20개 컬럼 (closed-world라 TPR/FPR 칸은 비움)
  results/cw_best_params.json       모델별 최고 하이퍼파라미터 (tune이 갱신)
  results/cw_oof/*.npz              OOF 예측 (url, inst, y, pred, proba)
  results/cw_test/*                 test 예측·혼동행렬 (test stage)
"""
import argparse, csv, itertools, json, os, sys, time
from datetime import datetime

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import ParameterSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment as ex
import load_data as ld

ROOT = ex.ROOT
RES = os.path.join(ROOT, 'results')
OOF_DIR = os.path.join(RES, 'cw_oof')
TEST_DIR = os.path.join(RES, 'cw_test')
CSV_OUT = os.path.join(RES, 'closedworld_results.csv')
BEST_JSON = os.path.join(RES, 'cw_best_params.json')
LOCK = os.path.join(RES, 'cw_test_LOCK')
COLS = ['exp_id', 'dataset', 'split_mode', 'feature_set', 'model', 'imbalance', 'threshold', 'n_classes',
        'TPR', 'FPR', 'precision', 'recall', 'macro_f1', 'notes',
        'accuracy', 'f1', 'roc_auc', 'pr_auc', 'det_TPR', 'site_id_rate']

# 피처셋 프리셋. 모델별로 다르게 둘 수 있다(WORK_SUMMARY 6.3: RF는 @dir50 없어도 같음, ET는 필요, 트리는 181이 +0.1%p)
FEATURESETS = {
    'cand131': ['@all', '-@const', '@early', '@bseq', '@cumulabs', '@dir50'],
    'tree181': ['@all', '-@const', '@early', '@bseq', '@cumulabs', '@dir'],
    'nodir95': ['@all', '-@const', '@early', '@bseq', '@cumulabs'],
}

# ---------------------------------------------------------------- 모델 정의
# experiment.MODELS(수업 모델)에 부스팅 2개를 더한다. (make, is_tree)
try:
    from lightgbm import LGBMClassifier
    HAVE_LGBM = True
except ImportError:
    HAVE_LGBM = False
from sklearn.ensemble import HistGradientBoostingClassifier

EXTRA_MODELS = {
    # 고도화 재실험(EXPERIMENTS.md)에서 lr 0.1/100회는 일부 fold가 발산했다 → 0.05/400회가 기본
    'hgb': (lambda: HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, early_stopping=False,
                                                   random_state=0), True),
}
if HAVE_LGBM:
    EXTRA_MODELS['lgbm'] = (lambda: LGBMClassifier(learning_rate=0.05, n_estimators=400, verbose=-1,
                                                   n_jobs=-1, random_state=0), True)
ex.MODELS.update(EXTRA_MODELS)

# 기본값 측정에서 쓰는 설정. 고도화 재실험과 맞춘다(ET/RF 300그루 mf 0.1). svm·knn은 gw 켬(6c단계 제안)
BASE = {
    'lr':   ({}, False),
    'knn':  ({}, True),
    'svm':  ({}, True),
    'rf':   ({'n_estimators': 300, 'max_features': 0.1}, False),
    'et':   ({'n_estimators': 300, 'max_features': 0.1}, False),
    'hgb':  ({}, False),
    'lgbm': ({}, False),
    'mlp':  ({}, False),
}

# random search 공간. 메모리: 트리 노드 수 × 95클래스 × 8바이트(WORK_SUMMARY 6.4 ③) → min_samples_leaf로 조절
# '_gw'는 모델 파라미터가 아니라 그룹 가중치 켬/끔이다.
SPACE = {
    'rf':  {'n_estimators': [300, 500], 'max_features': ['sqrt', 0.1, 0.2, 0.3, 0.5],
            'min_samples_leaf': [1, 2], 'criterion': ['gini', 'entropy']},
    'et':  {'n_estimators': [300, 500], 'max_features': ['sqrt', 0.1, 0.2, 0.3, 0.5],
            'min_samples_leaf': [1, 2], 'criterion': ['gini', 'entropy']},
    'lgbm': {'n_estimators': [400, 800], 'learning_rate': [0.03, 0.05, 0.1], 'num_leaves': [15, 31, 63],
             'colsample_bytree': [0.3, 0.5, 0.8], 'subsample': [0.7, 0.9, 1.0], 'subsample_freq': [1],
             'min_child_samples': [10, 20, 40], 'reg_lambda': [0.0, 1.0, 5.0]},
    # HGB는 95클래스라 반복마다 나무 95개 → 느리다(기본값 CV 약 35분). n_iter를 작게(3~5)
    'hgb': {'learning_rate': [0.05, 0.1], 'max_iter': [300, 500], 'max_leaf_nodes': [15, 31, 63],
            'l2_regularization': [0.0, 1.0], 'max_features': [0.3, 0.5, 1.0]},
    'svm': {'C': [1, 3, 10, 30, 100], 'gamma': ['scale', 0.003, 0.01, 0.03], '_gw': [True, False]},
    'knn': {'n_neighbors': [1, 3, 5, 7, 10, 15], 'weights': ['uniform', 'distance'], 'p': [1, 2],
            '_gw': [True, False]},
    'lr':  {'C': [0.1, 0.3, 1, 3, 10, 30], 'max_iter': [3000]},
    'mlp': {'hidden_layer_sizes': [(256,), (512,), (512, 256)], 'alpha': [1e-4, 1e-3, 1e-2],
            'learning_rate_init': [1e-3, 3e-4]},
}


# ---------------------------------------------------------------- 데이터
def data(fs='cand131'):
    df, names = ex.load()
    feats = ex.select(FEATURESETS[fs], names)
    is_te = ld.mon_is_test(df.url.values, df.inst.values)
    tr, te = df[~is_te], df[is_te]
    return dict(names=names, feats=feats, Xtr=tr[feats].values, ytr=tr.site.values, fold=tr.inst.values % 5,
                url_tr=tr.url.values, inst_tr=tr.inst.values,
                Xte=te[feats].values, yte=te.site.values, url_te=te.url.values, inst_te=te.inst.values)


def make_pipe(model, params, D):
    params = dict(params)
    gw = params.pop('_gw', False)
    pipe, _, _ = ex.build(model, params, weights=ex.group_weights(D['feats'], D['names']) if gw else None)
    return pipe


def proba_of(pipe, X, n_cls=95):
    if hasattr(pipe, 'predict_proba'):
        try:
            p = pipe.predict_proba(X)
            out = np.zeros((len(X), n_cls), np.float32)
            out[:, pipe.classes_] = p  # fold에 없는 클래스가 있어도 열 위치를 맞춘다
            return out
        except AttributeError:  # SVC(probability=False)
            pass
    return None


def cv_oof(model, params, D):
    """5-fold OOF. 반환: acc_per_fold, pred, proba(or None), 초"""
    pred = np.full(len(D['ytr']), -1)
    proba = np.zeros((len(D['ytr']), 95), np.float32)
    has_p, accs, t0 = True, [], time.time()
    for k in range(5):
        va = D['fold'] == k
        pipe = make_pipe(model, params, D).fit(D['Xtr'][~va], D['ytr'][~va])
        pred[va] = pipe.predict(D['Xtr'][va])
        p = proba_of(pipe, D['Xtr'][va])
        if p is None:
            has_p = False
        else:
            proba[va] = p
        accs.append((pred[va] == D['ytr'][va]).mean())
        print(f'    fold {k}: {accs[-1]:.4f}  ({time.time() - t0:.0f}s)', flush=True)
    return np.array(accs), pred, (proba if has_p else None), time.time() - t0


def save_oof(name, D, pred, proba, params, fs):
    os.makedirs(OOF_DIR, exist_ok=True)
    kw = dict(url=D['url_tr'], inst=D['inst_tr'], y=D['ytr'], pred=pred,
              params=json.dumps(params, default=str), fs=fs)
    if proba is not None:
        kw['proba'] = proba
    np.savez_compressed(os.path.join(OOF_DIR, f'{name}.npz'), **kw)


def metrics(y, pred):
    return dict(accuracy=accuracy_score(y, pred), macro_f1=f1_score(y, pred, average='macro'),
                precision=precision_score(y, pred, average='macro', zero_division=0),
                recall=recall_score(y, pred, average='macro'))


def log_row(exp_id, model, fs, n_feats, m, notes):
    os.makedirs(RES, exist_ok=True)
    new = not os.path.exists(CSV_OUT)
    with open(CSV_OUT, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new:
            w.writerow(COLS)
        row = dict(exp_id=exp_id, dataset='pkl_mon95', split_mode='visit', feature_set=f'{fs}({n_feats})',
                   model=model, imbalance='none', n_classes=95,
                   precision=f'{m["precision"]:.6f}', recall=f'{m["recall"]:.6f}', macro_f1=f'{m["macro_f1"]:.6f}',
                   accuracy=f'{m["accuracy"]:.6f}', f1=f'{m["macro_f1"]:.6f}',
                   notes=f'{notes}; {datetime.now():%m-%d %H:%M}')
        w.writerow([row.get(c, '') for c in COLS])


def fmt_params(p):
    return ','.join(f'{k}={v}' for k, v in p.items()) or 'default'


# ---------------------------------------------------------------- stage
def stage_base(args):
    models = args.models or [m for m in BASE if m in ex.MODELS and m != 'mlp']
    D = data(args.fs)
    print(f'피처 {len(D["feats"])}개 ({args.fs}), train {len(D["ytr"])}개, 새 분할(11/15)\n')
    for m in models:
        if m not in ex.MODELS:
            print(f'[건너뜀] {m}: 설치 안 됨'); continue
        params, gw = BASE[m]
        params = {**params, '_gw': True} if gw else params
        print(f'[{m}] {fmt_params(params)}', flush=True)
        accs, pred, proba, sec = cv_oof(m, params, D)
        name = f'{m}_base' + ('' if args.fs == 'cand131' else f'_{args.fs}')
        save_oof(name, D, pred, proba, params, args.fs)
        mt = metrics(D['ytr'], pred)
        print(f'  → CV {accs.mean():.4f} ± {accs.std():.4f}  macroF1 {mt["macro_f1"]:.4f}  {sec:.0f}s\n')
        log_row(f'CW-base-{name}', f'{m}({fmt_params(params)})', args.fs, len(D['feats']), mt,
                f'CV5 OOF train only; fold acc {" ".join(f"{a:.4f}" for a in accs)}; std {accs.std():.4f}; {sec:.0f}s')


def stage_tune(args):
    assert len(args.models) == 1, 'tune은 모델 하나씩: -m lgbm'
    m = args.models[0]
    D = data(args.fs)
    trials = list(ParameterSampler(SPACE[m], n_iter=args.n_iter, random_state=args.seed))
    print(f'[{m}] random search {len(trials)}회, 피처 {len(D["feats"])}개 ({args.fs})\n')
    best = json.load(open(BEST_JSON, encoding='utf-8')) if os.path.exists(BEST_JSON) else {}
    key = m if args.fs == 'cand131' else f'{m}_{args.fs}'
    for i, p in enumerate(trials):
        print(f'  ({i + 1}/{len(trials)}) {fmt_params(p)}', flush=True)
        accs, pred, proba, sec = cv_oof(m, p, D)
        mt = metrics(D['ytr'], pred)
        print(f'    → CV {accs.mean():.4f} ± {accs.std():.4f}  {sec:.0f}s', flush=True)
        log_row(f'CW-tune-{key}-{args.seed}-{i}', f'{m}({fmt_params(p)})', args.fs, len(D['feats']), mt,
                f'CV5 OOF train only; random search seed {args.seed}; std {accs.std():.4f}; {sec:.0f}s')
        if accs.mean() > best.get(key, {}).get('cv', -1):
            best[key] = dict(model=m, fs=args.fs, params=p, cv=float(accs.mean()), std=float(accs.std()))
            save_oof(f'{key}_tuned', D, pred, proba, p, args.fs)
            json.dump(best, open(BEST_JSON, 'w', encoding='utf-8'), indent=2, ensure_ascii=False, default=str)
            print('    ★ 최고 갱신 → OOF 저장', flush=True)
    print(f'\n최고: {best[key]}')


def load_oofs():
    """results/cw_oof/*.npz → {이름: (y, pred, proba)}. 행은 (url, inst) 기준으로 맞춘다(DF 결과도 섞기 위해)."""
    out, ref = {}, None
    for fn in sorted(os.listdir(OOF_DIR)):
        if not fn.endswith('.npz'):
            continue
        z = np.load(os.path.join(OOF_DIR, fn), allow_pickle=True)
        key = z['url'].astype(np.int64) * 20 + z['inst']
        order = np.argsort(key)
        if ref is None:
            ref = key[order]
        assert (key[order] == ref).all(), f'{fn}: 행 구성이 다르다'
        out[fn[:-4]] = (z['y'][order], z['pred'][order], z['proba'][order] if 'proba' in z else None)
    return out


def stage_vote(args):
    oofs = load_oofs()
    y = next(iter(oofs.values()))[0]
    single = sorted(((name, (v[1] == y).mean()) for name, v in oofs.items()), key=lambda t: -t[1])
    print('단일 모델 (OOF 정확도):')
    for n, a in single:
        print(f'  {a:.4f}  {n}' + ('' if oofs[n][2] is not None else '   (확률 없음 → 앙상블 제외)'))
    cands = [n for n, _ in single if oofs[n][2] is not None]
    if args.only:
        cands = [n for n in cands if any(s in n for s in args.only)]
    cands = cands[:args.top]  # 조합 폭발 방지: 상위 top개 안에서만
    res = []
    for r in range(2, min(args.max_k, len(cands)) + 1):
        for combo in itertools.combinations(cands, r):
            p = np.mean([oofs[n][2] for n in combo], axis=0)
            res.append(((p.argmax(1) == y).mean(), combo))
    res.sort(key=lambda t: -t[0])
    print(f'\nsoft voting 상위 {args.show}개 (후보 {len(cands)}개, 최대 {args.max_k}개 조합):')
    for a, combo in res[:args.show]:
        print(f'  {a:.4f}  ' + ' + '.join(combo))
    if res:
        a, combo = res[0]
        mt = metrics(y, np.mean([oofs[n][2] for n in combo], axis=0).argmax(1))
        log_row('CW-vote-best', 'SoftVote(' + '+'.join(combo) + ')', 'mixed', 0, mt,
                f'OOF 확률 평균, 재학습 없음; 후보 {len(cands)}개 중 최고')
        print('\n※ 조합을 CV로 여러 개 비교해 고른 것이라 최고 조합의 CV는 약간 낙관적이다. 최종 성능은 test 1회로 본다.')


def stage_summary(args):
    oofs = load_oofs()
    y = next(iter(oofs.values()))[0]
    best = json.load(open(BEST_JSON, encoding='utf-8')) if os.path.exists(BEST_JSON) else {}
    print(f'{"이름":28s} {"CV acc":>7s}  fold별')
    # fold는 inst % 5. OOF 행은 url*20+inst 순서로 정렬돼 있다
    z = np.load(os.path.join(OOF_DIR, sorted(os.listdir(OOF_DIR))[0]), allow_pickle=True)
    order = np.argsort(z['url'].astype(np.int64) * 20 + z['inst'])
    fold = z['inst'][order] % 5
    for name, (yy, pred, _) in sorted(oofs.items(), key=lambda t: -(t[1][1] == y).mean()):
        fa = [(pred[fold == k] == yy[fold == k]).mean() for k in range(5)]
        print(f'{name:28s} {(pred == yy).mean():.4f}  ' + ' '.join(f'{a:.3f}' for a in fa))
    if best:
        print('\n최고 하이퍼파라미터 (cw_best_params.json):')
        for k, v in best.items():
            print(f'  {k:14s} CV {v["cv"]:.4f} ± {v["std"]:.4f}  {fmt_params(v["params"])}')


# ---------------------------------------------------------------- best: 최고 설정 재사용 (시드·SVM 확률·피처셋 바꾸기)
def parse_set(items):
    out = {}
    for it in items or []:
        k, v = it.split('=', 1)
        out[k] = ex.parse_value(v) if v not in ('True', 'False', 'None') else {'True': True, 'False': False, 'None': None}[v]
    return out


def stage_best(args):
    """cw_best_params.json의 최고 설정을 가져와 일부만 바꿔 CV → OOF 저장.
    예) best -m lgbm --set random_state=1 --name lgbm_s1         (시드 앙상블용)
        best -m svm --set probability=True --name svm_proba       (SVM을 앙상블에 넣기 위한 확률 출력)
        best -m et --fs tree181 --name et_tree181                 (같은 설정, 피처셋만 181개)"""
    assert len(args.models) == 1 and args.name, 'best -m 모델 --name 저장이름 [--set k=v ...] [--fs ...] [--from 키]'
    m = args.models[0]
    best = json.load(open(BEST_JSON, encoding='utf-8')) if os.path.exists(BEST_JSON) else {}
    key = args.from_key or m
    if key in best:
        params = dict(best[key]['params'])
        src = f'cw_best_params.json[{key}] (CV {best[key]["cv"]:.4f})'
    else:
        params, gw = BASE[m]
        params = {**params, '_gw': True} if gw else dict(params)
        src = '기본값 (튜닝 기록 없음)'
    if 'hidden_layer_sizes' in params:
        params['hidden_layer_sizes'] = tuple(params['hidden_layer_sizes'])
    params.update(parse_set(args.set))
    D = data(args.fs)
    print(f'[{args.name}] {m}  출발점: {src}\n  설정: {fmt_params(params)}  피처 {len(D["feats"])}개 ({args.fs})', flush=True)
    accs, pred, proba, sec = cv_oof(m, params, D)
    mt = metrics(D['ytr'], pred)
    print(f'  → CV {accs.mean():.4f} ± {accs.std():.4f}  {sec:.0f}s' + ('' if proba is not None else '  (확률 없음)'))
    save_oof(args.name, D, pred, proba, params, args.fs)
    log_row(f'CW-best-{args.name}', f'{m}({fmt_params(params)})', args.fs, len(D['feats']), mt,
            f'CV5 OOF train only; from {key}; fold acc {" ".join(f"{a:.4f}" for a in accs)}; {sec:.0f}s')


# ---------------------------------------------------------------- ensemble: 섞는 방법 4가지를 정직하게 비교
EPS = 1e-6


def combine(method, P, w=None, meta=None):
    """P: [n_model] × (n, 95) 확률. 반환 (n, 95) 점수."""
    if method == 'mean':
        return np.mean(P, axis=0)
    if method == 'geo':  # 확률의 기하평균 = log 확률 평균. 한 모델이라도 확신 없이 낮게 준 클래스를 깎는다
        return np.mean(np.log(np.asarray(P) + EPS), axis=0)
    if method == 'wvote':
        return np.tensordot(w, np.asarray(P), axes=1)
    if method == 'stack':
        return meta.predict_proba(np.hstack(P))  # 확률 그대로 이어 붙임 (log보다 1%p 이상 좋았다)
    raise ValueError(method)


def fit_weights(P, y, n_samples=600, seed=0):
    """가중치 탐색: 같은 비중 + Dirichlet 무작위 n_samples개 중 정확도 최고. 모델이 적어 이 정도면 충분하다."""
    rng = np.random.default_rng(seed)
    M = len(P)
    P = np.asarray(P)
    cands = [np.full(M, 1 / M)] + list(rng.dirichlet(np.ones(M), n_samples))
    best_w, best_a = None, -1
    for w in cands:
        a = (np.tensordot(w, P, axes=1).argmax(1) == y).mean()
        if a > best_a:
            best_w, best_a = w, a
    return best_w


def fit_stack(P, y, C):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, max_iter=300).fit(np.hstack(P), y)


def stage_ensemble(args):
    """같은 멤버로 mean / geo / wvote / stack을 비교한다.
    wvote·stack은 가중치·메타 모델을 '학습'하므로, fold k의 점수는 나머지 4 fold의 OOF로만 학습해서 낸다(중첩 CV).
    그래서 표의 숫자는 그대로 비교해도 공정하다."""
    oofs = load_oofs()
    y = next(iter(oofs.values()))[0]
    z = np.load(os.path.join(OOF_DIR, sorted(f for f in os.listdir(OOF_DIR) if f.endswith('.npz'))[0]), allow_pickle=True)
    fold = z['inst'][np.argsort(z['url'].astype(np.int64) * 20 + z['inst'])] % 5
    if args.members:
        members = args.members
    else:  # 확률이 있는 모델 중 OOF 상위 top개 (같은 모델의 시드 변형은 모두 포함)
        ranked = sorted((n for n in oofs if oofs[n][2] is not None), key=lambda n: -(oofs[n][1] == y).mean())
        members = ranked[:args.top]
    miss = [n for n in members if n not in oofs or oofs[n][2] is None]
    assert not miss, f'확률 OOF가 없는 멤버: {miss}'
    P = [oofs[n][2] for n in members]
    print('멤버 (단독 OOF 정확도):')
    for n in members:
        print(f'  {(oofs[n][1] == y).mean():.4f}  {n}')
    rows = []
    for method in ['mean', 'geo', 'wvote', 'stack']:
        t0, pred = time.time(), np.zeros_like(y)
        for k in range(5):
            tr, va = fold != k, fold == k
            Ptr, Pva = [p[tr] for p in P], [p[va] for p in P]
            w = fit_weights(Ptr, y[tr], seed=k) if method == 'wvote' else None
            meta = fit_stack(Ptr, y[tr], args.stack_c) if method == 'stack' else None
            pred[va] = combine(method, Pva, w, meta).argmax(1)
        fa = [(pred[fold == k] == y[fold == k]).mean() for k in range(5)]
        rows.append((method, np.mean(fa), np.std(fa), fa))
        print(f'  {method:6s} {np.mean(fa):.4f} ± {np.std(fa):.4f}  [{" ".join(f"{a:.4f}" for a in fa)}]  {time.time() - t0:.0f}s',
              flush=True)
        log_row(f'CW-ens-{method}', f'{method}(' + '+'.join(members) + ')', 'mixed', 0, metrics(y, pred),
                f'중첩 CV (가중치·메타는 나머지 4 fold OOF로만 학습); fold acc {" ".join(f"{a:.4f}" for a in fa)}')
    w_all = fit_weights(P, y)
    print('\n전체 OOF로 맞춘 가중치 (test에서 wvote:에 쓰는 값): ' + ', '.join(f'{n} {w:.2f}' for n, w in zip(members, w_all)))
    best = max(rows, key=lambda r: r[1])
    print(f'\n최고: {best[0]} {best[1]:.4f}  →  test 때: --final {best[0]}:{"+".join(members)}')
    json.dump(dict(members=members, results={r[0]: dict(cv=r[1], std=r[2], folds=r[3]) for r in rows},
                   weights=dict(zip(members, map(float, w_all))), stack_c=args.stack_c),
              open(os.path.join(RES, 'cw_ensemble.json'), 'w', encoding='utf-8'), indent=2, ensure_ascii=False,
              default=float)
    print('기록 → results/cw_ensemble.json')


def fit_full(spec, D_cache):
    """spec: 'lgbm_tuned' | 'rf_base' | 'et_tree181_tuned' ... → test 확률(또는 예측)"""
    z = np.load(os.path.join(OOF_DIR, f'{spec}.npz'), allow_pickle=True)
    fs = str(z['fs'])
    params = json.loads(str(z['params']))
    if 'hidden_layer_sizes' in params:
        params['hidden_layer_sizes'] = tuple(params['hidden_layer_sizes'])
    m = spec.split('_')[0]
    if m == 'df':
        p = os.path.join(TEST_DIR, f'{spec}_test.npz')  # df_base → df_base_test.npz (cw_df.py --final --tag base)
        assert os.path.exists(p), f'{p}가 없다: Colab에서 code/cw_df.py --final --tag {spec[3:]} 먼저 돌려 옮길 것'
        t = np.load(p)
        order = np.argsort(t['url'].astype(np.int64) * 20 + t['inst'])  # 피처 CSV 행 순서(url*20+inst)에 맞춘다
        return t['proba'][order], t['proba'][order].argmax(1)
    D = D_cache.setdefault(fs, data(fs))
    pipe = make_pipe(m, params, D).fit(D['Xtr'], D['ytr'])
    proba = proba_of(pipe, D['Xte'])
    return proba, pipe.predict(D['Xte'])


def stage_test(args):
    if os.path.exists(LOCK) and not args.force:
        raise SystemExit(f'test는 이미 돌렸다 ({open(LOCK, encoding="utf-8").read().strip()}). '
                         'test를 보고 모델을 다시 고르면 안 된다. 정말 필요하면 --force (보고서에 밝힐 것)')
    assert args.final, ('--final 필요. 예: --final lgbm_tuned  /  --final mean:et_tuned+rf_tuned+lgbm_tuned  '
                        '(앙상블 방법: mean·vote(=mean)·geo·wvote·stack)')
    method, combo = (args.final.split(':', 1)[0], args.final.split(':', 1)[1].split('+')) if ':' in args.final \
        else (None, [args.final])
    method = 'mean' if method == 'vote' else method
    os.makedirs(TEST_DIR, exist_ok=True)
    with open(LOCK, 'w', encoding='utf-8') as f:
        f.write(f'{datetime.now():%Y-%m-%d %H:%M} final={args.final} compare={args.compare}\n')
    D_cache, probs = {}, {}
    names = list(dict.fromkeys((args.compare or []) + combo))
    ref = data('cand131')
    for n in names:
        print(f'[{n}] 전체 train으로 학습 → test 예측', flush=True)
        probs[n] = fit_full(n, D_cache)
    rows = [(n, probs[n][1]) for n in names]
    if method:
        Pte = [probs[n][0] for n in combo]
        assert all(p is not None for p in Pte), '앙상블 멤버는 모두 확률을 내야 한다'
        w = meta = None
        if method in ('wvote', 'stack'):  # 가중치·메타 모델은 train OOF 전체로 맞춘다 (test는 쓰지 않음)
            oofs = load_oofs()
            y_oof, Poof = next(iter(oofs.values()))[0], [oofs[n][2] for n in combo]
            w = fit_weights(Poof, y_oof) if method == 'wvote' else None
            meta = fit_stack(Poof, y_oof, args.stack_c) if method == 'stack' else None
            if w is not None:
                print('가중치: ' + ', '.join(f'{n} {v:.2f}' for n, v in zip(combo, w)))
        rows.append((args.final, combine(method, Pte, w, meta).argmax(1)))
    i19 = ref['inst_te'] == 19
    print(f'\n{"모델":45s} {"test acc":>8s} {"inst19":>7s} {"macroF1":>8s}')
    for n, pred in rows:
        mt = metrics(ref['yte'], pred)
        a19 = (pred[i19] == ref['yte'][i19]).mean()
        star = '  ← 최종 (CV로 미리 정함)' if n == args.final else ''
        print(f'{n:45s} {mt["accuracy"]:8.4f} {a19:7.4f} {mt["macro_f1"]:8.4f}{star}')
        log_row(f'TEST-{n}', n, 'see_oof', 0, mt, f'TEST 1회 (1,900개, URL마다 방문 2개); inst19(95개) acc {a19:.4f}'
                + ('; FINAL' if star else ''))
        safe = n.replace(':', '_').replace('+', '-')
        np.savez_compressed(os.path.join(TEST_DIR, f'{safe}.npz'), pred=pred, y=ref['yte'],
                            url=ref['url_te'], inst=ref['inst_te'])
        if star:
            np.savetxt(os.path.join(TEST_DIR, f'confusion_{safe}.csv'),
                       confusion_matrix(ref['yte'], pred, labels=range(95)), fmt='%d', delimiter=',')
    print('\n이 결과를 보고 모델·피처·하이퍼파라미터를 다시 고르지 않는다.')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['base', 'tune', 'best', 'vote', 'ensemble', 'summary', 'test'])
    ap.add_argument('-m', '--models', nargs='+')
    ap.add_argument('--fs', default='cand131', choices=FEATURESETS)
    ap.add_argument('-n', '--n-iter', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--top', type=int, default=8, help='vote: 확률 있는 상위 몇 개 모델 안에서 조합할지')
    ap.add_argument('--max-k', type=int, default=4, help='vote: 최대 몇 개까지 섞을지')
    ap.add_argument('--show', type=int, default=15)
    ap.add_argument('--only', nargs='+', help='vote: 이름에 이 문자열이 든 OOF만 (예: tuned)')
    ap.add_argument('--final', help='test: 최종 모델 (OOF 이름 또는 vote:a+b+c)')
    ap.add_argument('--compare', nargs='+', help='test: 비교표에 같이 넣을 모델들 (OOF 이름)')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--name', help='best: 저장 이름 (results/cw_oof/<name>.npz)')
    ap.add_argument('--set', nargs='+', help='best: 바꿀 설정 k=v (예: random_state=1 probability=True)')
    ap.add_argument('--from', dest='from_key', help='best: cw_best_params.json에서 가져올 키 (기본: 모델 이름)')
    ap.add_argument('--members', nargs='+', help='ensemble: 섞을 OOF 이름들 (기본: 확률 있는 상위 --top개)')
    ap.add_argument('--stack-c', type=float, default=1.0, help='ensemble/test: stacking 메타 LR의 C')
    args = ap.parse_args()
    if not HAVE_LGBM:
        print('※ lightgbm이 없어 lgbm은 건너뛴다 (pip install lightgbm)\n')
    {'base': stage_base, 'tune': stage_tune, 'best': stage_best, 'vote': stage_vote, 'ensemble': stage_ensemble,
     'summary': stage_summary, 'test': stage_test}[args.stage](args)


if __name__ == '__main__':
    main()
