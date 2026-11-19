"""Extra credit ⑤: 끝까지 아껴둔 최종 test로 한 번만 평가한다 (sklearn 필요).

사용법: python -I -W ignore -X utf8 code/final_test_sliver.py > logs/final_test_sliver.log
        (stdout만 로그로. 2>&1 금지: joblib 경고가 로그를 GB로 불린다)

test는 `ld.sliver_is_test(group)`, 방문의 10%다. 피처·세트·모델을 고르는 동안 한 번도 보지 않았다
(4.6–4.9는 모두 test를 뺀 train 안에서 group 5-fold로 했다). 그래서 이 값이 보고서에 쓸 숫자다.

재는 것
  B split→split (202개)     : 공격자가 입구 노드 하나만 보는 상황. 우리 결과다
  A join→join   (202개)     : 세트를 고정했을 때 방어 자체의 효과
  A join→join   (피처 전체)  : 방어가 없으면 공격자는 202개에 묶이지 않는다. 202개는 split용으로 고른
                              시간 피처 위주여서 join의 핵심인 방향·순서 피처(@early @bseq @dir)가 빠져
                              있다. 보고서에 쓸 "실제 격차"는 양쪽이 각자 최선을 쓴 이 비교다.

시드 0·1·2로 세 번 돌려 변동폭도 같이 낸다. 학습은 fold가 아니라 train 전체이므로 fit 한 번이다.
"""
import os, sys, time

import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment as ex
import load_data as ld

N_TREES = 200
SEEDS = (0, 1, 2)
TIME_NEW = ('tcum_', 'tbin_', 'tnorm_', 'gap_')
FINAL_EXTRA = ('tnout_', 'tnqin_', 'tnqout_')
EDGES = [0, 10, 25, 50, 100, 200, 400, 800, 1600, 10**9]


def load():
    """short_splits.load와 달리 test 행과 join 행을 버리지 않는다."""
    a = pd.read_csv(os.path.join(ex.ROOT, 'features', 'sliver_mon.csv'))
    b = pd.read_csv(os.path.join(ex.ROOT, 'features', 'sliver_mon_time2.csv')).drop(columns=['label'])
    df = a.merge(b, on=['group', 'part'], how='inner', validate='one_to_one')
    assert len(df) == len(a), f'(group, part) 합치기 실패: {len(df)} != {len(a)}'
    names = [c for c in df.columns if c not in ('label', 'group', 'part', 'n_bursts_total')]
    df[names] = df[names].astype(np.float32)
    return df.reset_index(drop=True), names


def main():
    df, names = load()
    G = ex.groups(names)
    fs = G['timing'] + [f for f in names if f.startswith(TIME_NEW + FINAL_EXTRA)]
    assert len(fs) == 202, len(fs)

    is_test = ld.sliver_is_test(df.group.values)
    is_join = df.part.values == ld.JOIN_PART
    X, y = df[names].values, df.label.values  # 세트별로 열을 골라 쓴다
    col = {f: i for i, f in enumerate(names)}
    n_cells = df.n_total.values.astype(int)

    # train과 test에 같은 방문이 걸치지 않는지 직접 확인한다 (group 단위 분리)
    gtr, gte = set(df.group.values[~is_test]), set(df.group.values[is_test])
    assert not (gtr & gte), f'누수: 방문 {len(gtr & gte)}개가 train과 test 양쪽에 있다'
    print(f'전체 {len(df)}행 (split {(~is_join).sum()}, join {is_join.sum()})')
    print(f'방문 train {len(gtr)}개 / test {len(gte)}개 ({len(gte) / (len(gtr) + len(gte)) * 100:.1f}%)'
          f' — 겹치는 방문 0개')
    print(f'세트 {len(fs)}개, RF({N_TREES}), 시드 {SEEDS}\n', flush=True)

    t0 = time.time()
    runs = {}

    def run(tag, rows, seed, cols):
        tr, te = rows & ~is_test, rows & is_test
        ci = [col[f] for f in cols]
        m = RandomForestClassifier(n_estimators=N_TREES, n_jobs=-1, random_state=seed)
        m.fit(X[tr][:, ci], y[tr])
        pred = m.predict(X[te][:, ci])
        a = (pred == y[te]).mean()
        runs[(tag, seed)] = (te, pred == y[te])
        print(f'  {tag:24s} 시드 {seed}  피처 {len(cols):3d}  train {tr.sum():6d} / test {te.sum():5d}'
              f'  {a:.4f}  ({time.time() - t0:.0f}s)', flush=True)
        return a

    print('===== 최종 test =====', flush=True)
    res = {}
    for tag, rows, cols in (('B split→split (202개)', ~is_join, fs),
                            ('A join→join (202개)', is_join, fs),
                            ('A join→join (피처 전체)', is_join, names)):
        v = [run(tag, rows, s, cols) for s in SEEDS]
        res[tag] = (np.mean(v), max(v) - min(v))

    print('\n===== 정리 =====')
    print(f'{"":24s} {"test 정확도":>10s} {"시드변동폭":>9s}')
    for tag, (m, sp) in res.items():
        print(f'{tag:24s} {m * 100:9.2f}% {sp * 100:8.2f}%')
    print(f'{"찍기 (1/95)":24s} {100 / 95:9.2f}%')
    b = res['B split→split (202개)'][0]
    a2 = res['A join→join (202개)'][0]
    a9 = res['A join→join (피처 전체)'][0]
    print(f'\n세트 고정(202개): {a2 * 100:.1f}%p → {b * 100:.1f}%p — 방어 자체의 효과')
    print(f'각자 최선:        {a9 * 100:.1f}%p → {b * 100:.1f}%p — 실제 격차'
          f' (남은 것은 방어 없을 때의 {b / a9 * 100:.0f}%)')
    print(f'찍기 대비 {b / (1 / 95):.1f}배')

    print('\n===== 길이별 (B, 시드 평균) =====')
    print(f'{"길이(셀)":>14s} {"행 수":>7s} {"정확도":>8s}')
    te0 = runs[('B split→split (202개)', 0)][0]
    oks = [runs[('B split→split (202개)', s)][1] for s in SEEDS]
    nt = n_cells[te0]
    for lo, hi in zip(EDGES, EDGES[1:]):
        sel = (nt >= lo) & (nt < hi)
        if not sel.any():
            continue
        tag = f'{lo}~{hi}' if hi < 10**8 else f'{lo}+'
        print(f'{tag:>14s} {sel.sum():7d} {np.mean([o[sel].mean() for o in oks]) * 100:7.2f}%')
    print('\n5-fold 검증값(15.98%)과 비교하면 test가 얼마나 비관/낙관인지 알 수 있다.', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
