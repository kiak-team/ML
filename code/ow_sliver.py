"""E6 TrafficSliver extra credit (open-world, multi 96클래스). reports/openworld.md E6 절.

사용법 (레포 루트에서):
  python -I -X utf8 code/ow_sliver.py build "<extra credit>/mon.zip" "<extra credit>/unmon.zip"   # zip → cache/sliver_raw_*.npz (A팀 build_sliver)
  python -I -W ignore -X utf8 code/ow_sliver.py feats  > logs/ow_e6_feats.log                     # 피처 → cache/sliver_feat_*.npz
  python -I -W ignore -X utf8 code/ow_sliver.py run --subset > logs/ow_e6_subset.log                # mon 사이트 25 / unmon 5,000
  python -I -W ignore -X utf8 code/ow_sliver.py run > logs/ow_e6.log

인스턴스 (공격자가 보는 것)
  split  (a) split 파일 하나 = 인스턴스 하나. 공격자가 경로 5개 중 하나만 관찰 (TrafficSliver의 방어 전제)
  merged (b) 한 방문의 split 5개를 합쳐 시각순 정렬. 원본 split 파일은 시각이 파일마다 0부터 시작하고
             join 시계 기준 offset이 없다(logs/ow_leak_check.log). 그래서 "다섯 경로가 같은 시각에 시작했다"고
             가정하고 파일의 시각 그대로 합친다. 같은 시각이면 split 번호 순서(안정 정렬). 실제 offset은 0~8초 어긋난다
  join   방어 없음. 5개 경로를 실제 시계로 합친 것과 같다 (join 길이 = split 합, build_sliver가 assert)

leakage (reports/openworld.md 0.3절, 결정 2026-11-26): 모든 인스턴스에서 앞 14셀을 버린다(skip 14, 주 조건).
  민감도로 버리지 않은 것(skip 0)도 같은 인스턴스·같은 분할로 잰다. 두 조건을 같은 인스턴스로 비교하려고
  15셀 미만 인스턴스는 양쪽 모두에서 뺀다(빠진 수를 로그에 남긴다). 시각은 버린 뒤에도 원래 값 그대로다
  (A팀 features_ow.py와 같다. features_time은 내부에서 첫 셀 기준으로 다시 맞춘다).

피처: A팀 extra credit 최종 202개 = @timing 18 (features.py) + tcum 38 + tbin 40 + gap 26 + tnorm 80 (features_time.py).
분할: splits/sliver_*.csv (방문 group 단위). train = fold 1~4, 검증 = fold 0, test(fold -1)는 쓰지 않는다.
모델: RF 200그루 = 25그루 × 8번(random_state 42+c), predict_proba를 그루 수로 가중 평균 → RF(200)과 같은 앙상블.
      96클래스 split 데이터는 나무 한 그루가 수십 MB라 한 번에 200그루를 메모리에 둘 수 없다(A팀 4.9).
"""
import argparse, os, sys, time
from multiprocessing import Pool

import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment as ex
import load_data as ld
from features import feats
from features_time import feats_time
from ow_metrics import UNMON, apply_threshold, ow_metrics, unmon_confusion
import ow_run as R

CACHE = R.CACHE
KINDS = ('split', 'merged', 'join')
SKIPS = (14, 0)
MIN_LEN = 15
TIME_GROUPS = ('tcum_', 'tbin_', 'gap_', 'tnorm_')
N_CHUNK, CHUNK = 8, 25


def raw_path(side):
    return os.path.join(CACHE, f'sliver_raw_{side}.npz')


def feat_path(kind, skip, side):
    return os.path.join(CACHE, f'sliver_feat_{kind}_skip{skip}_{side}.npz')


# ---------------- 인스턴스 ----------------
def instances(side, kind):
    """(times, dirs, label, group, part) 리스트. part: split 0~4, merged -1, join 5."""
    d = dict(np.load(raw_path(side)))
    off, T, D = d['offsets'], d['times'], d['dirs']
    lab, grp, part = d['label'], d['group'], d['part']
    seg = lambda i: (T[off[i]:off[i + 1]], D[off[i]:off[i + 1]])
    out = []
    if kind == 'split':
        for i in np.flatnonzero(part < ld.JOIN_PART):
            out.append((*seg(i), int(lab[i]), int(grp[i]), int(part[i])))
    elif kind == 'join':
        for i in np.flatnonzero(part == ld.JOIN_PART):
            out.append((*seg(i), int(lab[i]), int(grp[i]), 5))
    else:  # merged
        sp = np.flatnonzero(part < ld.JOIN_PART)
        order = np.lexsort((part[sp], grp[sp]))  # group, 그 안에서 split 번호 순
        sp = sp[order]
        bounds = np.r_[0, np.flatnonzero(np.diff(grp[sp])) + 1, len(sp)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            idx = sp[a:b]
            t = np.concatenate([seg(i)[0] for i in idx]); dr = np.concatenate([seg(i)[1] for i in idx])
            o = np.argsort(t, kind='stable')  # 파일 시각 그대로, 같은 시각이면 split 번호 순
            out.append((t[o], dr[o], int(lab[idx[0]]), int(grp[idx[0]]), -1))
    return out


# ---------------- 피처 ----------------
_SKIP = 0


def _init(skip):
    global _SKIP
    _SKIP = skip


def _feat(td):
    t, dr = td
    t, s = t[_SKIP:].astype(float).tolist(), (dr[_SKIP:].astype(int) * ld.CELL_SIZE).tolist()
    f = {**feats((t, s)), **feats_time((t, s))}
    return f, (int(dr[_SKIP + 7]) if len(dr) > _SKIP + 7 else 0)


def build_feats():
    names = None
    for side in ('mon', 'unmon'):
        for kind in KINDS:
            inst = instances(side, kind)
            lens = np.array([len(x[0]) for x in inst])
            keep = lens >= MIN_LEN
            inst = [x for x, k in zip(inst, keep) if k]
            print(f'[{side} {kind}] 인스턴스 {len(lens)}, {MIN_LEN}셀 미만 제외 {(~keep).sum()} '
                  f'({(~keep).mean():.2%}) → {len(inst)}', flush=True)
            for skip in SKIPS:
                t0 = time.time()
                with Pool(os.cpu_count(), initializer=_init, initargs=(skip,)) as p:
                    res = p.map(_feat, [(x[0], x[1]) for x in inst], chunksize=200)
                if names is None:
                    all_names = list(res[0][0])
                    timing = ex.groups(all_names[:55], pkl_qconst=False)['timing']
                    names = timing + [n for n in all_names if n.startswith(TIME_GROUPS)]
                    assert len(names) == 202, len(names)
                X = np.array([[r[0][n] for n in names] for r in res], np.float32)
                np.savez(feat_path(kind, skip, side), X=X, names=np.array(names),
                         label=np.array([x[2] for x in inst]), group=np.array([x[3] for x in inst]),
                         part=np.array([x[4] for x in inst]), length=lens[keep],
                         d7=np.array([r[1] for r in res], np.int8))
                print(f'   skip {skip}: {X.shape}  {time.time() - t0:.0f}s', flush=True)


# ---------------- 실험 ----------------
def load(kind, skip, subset):
    m, u = dict(np.load(feat_path(kind, skip, 'mon'))), dict(np.load(feat_path(kind, skip, 'unmon')))
    fm = pd.read_csv(os.path.join(R.ROOT, 'splits', 'sliver_mon.csv')).set_index('group')['fold']
    fu = pd.read_csv(os.path.join(R.ROOT, 'splits', 'sliver_unmon.csv')).set_index('group')['fold']
    fold = np.concatenate([fm.loc[m['group']].values, fu.loc[u['group']].values])
    d = {'X': np.vstack([m['X'], u['X']]), 'y': np.concatenate([m['label'], np.full(len(u['label']), UNMON)]),
         'group': np.concatenate([m['group'], u['group']]), 'fold': fold,
         'd7': np.concatenate([m['d7'], u['d7']]), 'names': list(m['names'])}
    if subset:
        keep = np.concatenate([m['label'] < 25, (u['group'] - 10**6) < 5000])
        d = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    return d


def chunked_rf(Xtr, ytr, Xva):
    from sklearn.ensemble import RandomForestClassifier
    proba, classes = None, None
    for c in range(N_CHUNK):
        m = RandomForestClassifier(n_estimators=CHUNK, n_jobs=-1, random_state=R.RANDOM_STATE + c).fit(Xtr, ytr)
        p = m.predict_proba(Xva)
        if proba is None:
            proba, classes = p, m.classes_
        else:
            assert (m.classes_ == classes).all()
            proba += p
        del m
    return proba / N_CHUNK, classes


def run(subset):
    tag = '-subset' if subset else ''
    dataset = 'sliver_m25_u5000' if subset else 'sliver'
    curves, summary = {}, []
    for skip in SKIPS:
        for kind in KINDS:
            d = load(kind, skip, subset)
            tr, va = d['fold'] >= 1, d['fold'] == 0
            assert not set(d['group'][tr]) & set(d['group'][va]), '방문이 train과 검증 양쪽에 있다'
            y = d['y'][va]
            n_mon = len(np.unique(d['y'][d['y'] != UNMON]))
            # 처리 후 남은 leak: 검증 인스턴스에서 'idx7 outgoing → mon' 규칙
            rp, ru = (d['d7'][va][y != UNMON] == 1).mean(), (d['d7'][va][y == UNMON] == 1).mean()
            t0 = time.time()
            proba, classes = chunked_rf(d['X'][tr], d['y'][tr], d['X'][va])
            secs = time.time() - t0
            assert len(classes) == n_mon + 1
            exp_id = f'E6{tag}-{kind}-skip{skip}'
            np.savez_compressed(os.path.join(CACHE, 'oof', f'{exp_id}.npz'), proba=proba.astype(np.float32),
                                classes=classes, y=y, group=d['group'][va])
            pred = apply_threshold(proba, classes, 0.0)
            r = ow_metrics(y, pred, labels=classes)
            mon = y != UNMON
            cw = (classes[classes != UNMON][proba[mon][:, classes != UNMON].argmax(1)] == y[mon]).mean()
            cA, cB = R.rule_curve(proba, classes, y, 'A'), R.rule_curve(proba, classes, y, 'B')
            curves[(kind, skip)] = (cA, cB)
            best = lambda x: max([c['TPR'][c['FPR'] <= x].max() for c in (cA, cB) if (c['FPR'] <= x).any()] or [0.0])
            b = {x: best(x) for x in (0.001, 0.01, 0.05)}
            # FPR <= 1% 최고 점 (검증 fold에서 고른 점이라 약간 낙관적)
            cands = [(c['TPR'][i], c['t'][i], name, c['FPR'][i], c['precision'][i])
                     for name, c in (('A', cA), ('B', cB)) for i in np.flatnonzero(c['FPR'] <= 0.01)]
            tp1, t1, rule1, fp1, pr1 = max(cands)
            print(f'\n## {exp_id}: train {tr.sum()} / val {va.sum()} (mon {mon.sum()}, unmon {(~mon).sum()}), '
                  f'{n_mon + 1}클래스, {secs:.0f}s', flush=True)
            print(f'  잔여 leak 규칙(idx7 outgoing → mon) val: TPR {rp:.4f} FPR {ru:.4f}')
            print(f'  argmax: TPR {r["TPR"]:.4f} FPR {r["FPR"]:.4f} precision {r["precision"]:.4f} macro_f1 '
                  f'{r["macro_f1"]:.4f} det_TPR {r["det_TPR"]:.4f} wrong_site {r["wrong_site"]} | closed-world acc(mon) {cw:.4f}')
            print(f'  최고 TPR: FPR≤0.1% {b[0.001]:.4f}, ≤1% {b[0.01]:.4f}, ≤5% {b[0.05]:.4f}')
            common = (f'fold0 holdout (train fold1-4 {tr.sum()}, val mon {mon.sum()} unmon {(~mon).sum()}); '
                      f'instance={kind}; skip{skip}; min_len {MIN_LEN}; closed-world acc(mon) {cw:.4f}; '
                      f'residual idx7 rule val TPR {rp:.4f} FPR {ru:.4f}')
            fs = f'sliver202_skip{skip}'
            model = f'RF(n=200 as {N_CHUNK}x{CHUNK},rs=42+c)'
            R.record({'exp_id': exp_id, 'dataset': f'{dataset}_{kind}', 'split_mode': 'visit(group)', 'feature_set': fs,
                      'model': model, 'imbalance': 'none', 'threshold': '', 'n_classes': len(classes),
                      'TPR': r['TPR'], 'FPR': r['FPR'], 'precision': r['precision'], 'recall': r['recall'],
                      'macro_f1': r['macro_f1'], 'notes': common + f'; decision=argmax; det_TPR {r["det_TPR"]:.4f}; '
                      f'wrong_site {r["wrong_site"]}; best TPR FPR<=0.1% {b[0.001]:.4f} <=5% {b[0.05]:.4f}; {secs:.0f}s'})
            pred1 = apply_threshold(proba, classes, t1) if rule1 == 'A' else np.where(
                R.mon_score(proba, classes)[0] >= t1, R.mon_score(proba, classes)[1], UNMON)
            r1 = ow_metrics(y, pred1, labels=classes)
            assert abs(r1['TPR'] - tp1) < 1e-9 and abs(r1['FPR'] - fp1) < 1e-9
            R.record({'exp_id': exp_id + '-fpr1', 'dataset': f'{dataset}_{kind}', 'split_mode': 'visit(group)',
                      'feature_set': fs, 'model': model, 'imbalance': 'none', 'threshold': round(float(t1), 4),
                      'n_classes': len(classes), 'TPR': r1['TPR'], 'FPR': r1['FPR'], 'precision': r1['precision'],
                      'recall': r1['recall'], 'macro_f1': r1['macro_f1'],
                      'notes': common + f'; decision=rule {rule1}, t chosen on val for FPR<=1% (optimistic); '
                      f'wrong_site {r1["wrong_site"]}'})
            cm, labels, _ = unmon_confusion(y, pred)
            pd.DataFrame(cm, index=labels, columns=labels).to_csv(
                os.path.join(R.ROOT, 'results', 'confusion', f'{exp_id}.csv'))
            summary.append((kind, skip, r['TPR'], r['FPR'], cw, b[0.001], b[0.01], b[0.05], r['det_TPR']))
    print('\n## 요약 (검증 fold 0): kind skip | argmax TPR FPR | closed-world | 최고 TPR @FPR≤0.1% 1% 5% | argmax det_TPR')
    for row in summary:
        print('  {:6s} {:2d} | {:.4f} {:.4f} | {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f}'.format(*row))
    plot(curves, tag)


def plot(curves, tag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    ink, muted, grid, base = '#0b0b0b', '#898781', '#e1e0d9', '#c3c2b7'
    plt.rcParams.update({'font.family': ['AppleGothic', 'sans-serif'], 'axes.unicode_minus': False, 'font.size': 10})
    series = [('join', '방어 없음 (join)', '#2a78d6'), ('merged', '(b) split 5개 합침', '#eb6834'),
              ('split', '(a) split 하나', '#1baf7a')]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, facecolor='#fcfcfb')
    for ax, skip in zip(axes, SKIPS):
        for kind, label, color in series:
            cA, cB = curves[(kind, skip)]
            keep = cB['FPR'] >= 1e-4
            ax.plot(cB['FPR'][keep], cB['TPR'][keep], color=color, lw=2, label=label)
            e = len(cA['FPR']) - 1
            ax.plot([max(cA['FPR'][e], 1e-4)], [cA['TPR'][e]], 'o', ms=8, color=color, mec='#fcfcfb', mew=2, zorder=3)
        ax.set_xscale('log'); ax.set_xlim(1e-4, 1); ax.set_ylim(0, 1)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v * 100:g}%'))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ''))
        ax.axvline(0.01, color=muted, lw=1, ls=(0, (3, 3)))
        ax.set_title('앞 14셀 버림 (주 조건)' if skip == 14 else '앞 14셀 그대로 (민감도, leak 포함)', color=ink, loc='left')
        ax.set_xlabel('FPR (로그 축)', color=ink)
        R._style(ax, ink, muted, grid, base)
    axes[0].set_ylabel('TPR (사이트까지 맞힘)', color=ink)
    axes[0].legend(frameon=False, loc='upper left', labelcolor=ink, fontsize=9)
    fig.suptitle('E6 TrafficSliver open-world — 96클래스 RF, 검증 fold 0 (선 = 규칙 B, 점 = argmax)',
                 color=ink, x=0.01, ha='left')
    fig.tight_layout()
    out = os.path.join(R.ROOT, 'figures', f'ow_e6_sliver{tag}.png')
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f'\n그림: {out}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build', 'feats', 'run'])
    ap.add_argument('zips', nargs='*')
    ap.add_argument('--subset', action='store_true')
    a = ap.parse_args()
    if a.cmd == 'build':
        os.makedirs(CACHE, exist_ok=True)
        for side, zp in zip(('mon', 'unmon'), a.zips):
            ld.build_sliver(zp, raw_path(side))
    elif a.cmd == 'feats':
        build_feats()
    else:
        run(a.subset)
