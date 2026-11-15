"""Extra credit 2단계: 시간 피처 추가 후보 + 짧은 split 처리 (sklearn 필요).

사용법: python -I -W ignore -X utf8 code/sliver_stage2.py <Downloads>/sliver_mon.npz > logs/sliver_stage2.log   (stdout만 로그로)

screen_sliver.py와 같은 holdout(최종 test 제외, 나머지 방문의 20%를 group 단위 검증)과 RF(200)를 쓴다.
기준은 '@timing + 새 시간' 142개(15.5%)다.
11/15 결과로 features_time.py의 tnorm을 40등분 수신·송신으로 바꿨다. 이 스크립트는 그 전 CSV(tnorm_in 20등분)를 전제로 한 기록용이다.

1. 시간 피처 후보 (CSV에 없으므로 npz에서 바로 계산해 group·part로 붙인다)
   - tnorm_out (20): 기록 길이를 20등분한 시각까지 보낸 셀 비율 (기존 tnorm_in의 송신 쪽)
   - tnorm40_in / tnorm40_out (40씩): 40등분 (2.5% 간격, 이름 끝 4자리 = 천분율). 기존 tnorm_in(20등분)을 대신한다
   검증 정확도 표준오차는 약 0.3%p(15,771행, 15%)라, +0.5%p 이상이고 시드 2개에서 같은 방향이면 채택한다
2. 짧은 split: 길이 구간별 검증 정확도, 50셀 미만을 학습에서 뺐을 때와 비교
"""
import os, sys, time

import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment as ex
import load_data as ld
from screen_sliver import TIME_NEW, is_val, load, rf

N20, N40 = 20, 40


def tnorm(x, dur, n, name):
    """x(정렬된 시각)에서 dur*j/n까지 들어온 비율, j=1..n."""
    q = np.searchsorted(x, dur * np.arange(1, n + 1) / n, 'right') / max(len(x), 1)
    return {(f'{name}_{j * 5:03d}' if n == 20 else f'{name}_{j * 25:04d}'): v for j, v in enumerate(q, 1)}


def new_feats(npz):
    d = dict(np.load(npz))
    off, keep = d['offsets'], (d['part'] < ld.JOIN_PART) & ~ld.sliver_is_test(d['group'])
    rows = []
    for i in np.flatnonzero(keep):
        t, dr = d['times'][off[i]:off[i + 1]].astype(float), d['dirs'][off[i]:off[i + 1]]
        dur, ti, to = t[-1], t[dr < 0], t[dr > 0]
        rows.append({'group': d['group'][i], 'part': d['part'][i], **tnorm(to, dur, N20, 'tnorm_out'),
                     **tnorm(ti, dur, N40, 'tnorm40_in'), **tnorm(to, dur, N40, 'tnorm40_out')})
    return pd.DataFrame(rows)


def fit_score(fs, tr, va, seed=0):
    m = rf(); m.set_params(random_state=seed).fit(mon.loc[tr, fs].values, y[tr])
    return m, m.predict(mon.loc[va, fs].values) == y[va]


if __name__ == '__main__':
    t0 = time.time()
    mon, names = load('mon')
    add = new_feats(sys.argv[1])
    mon = mon.merge(add, on=['group', 'part'], how='left', validate='one_to_one')
    assert mon[add.columns].notna().all().all(), 'npz와 CSV의 행이 맞지 않는다'
    # 자가 확인: 새로 계산한 값이 CSV(features_time.py)와 같은 규칙인지
    # 40등분의 50% = 20등분의 50%. 경계에 걸린 셀은 float32/64 차이로 하나 어긋날 수 있어 99.9%로 본다
    assert np.isclose(mon['tnorm40_in_0500'], mon['tnorm_in_050'], atol=1e-4).mean() > 0.999
    print(f'mon split {len(mon)}행, 새 피처 {add.shape[1] - 2}개 계산 ({time.time() - t0:.0f}s)', flush=True)

    G = ex.groups(names)
    timenew = [f for f in names if f.startswith(TIME_NEW)]
    base = G['timing'] + timenew
    out20 = [c for c in add if c.startswith('tnorm_out_')]
    in40 = [c for c in add if c.startswith('tnorm40_in_')]
    out40 = [c for c in add if c.startswith('tnorm40_out_')]
    no_in20 = [f for f in base if not f.startswith('tnorm_in_')]
    sets = {
        'S0 기준 @timing + 새 시간': base,
        'S1 + tnorm_out': base + out20,
        'S2 tnorm_in → tnorm40_in': no_in20 + in40,
        'S3 tnorm40_in + tnorm40_out (tnorm_in 대신)': no_in20 + in40 + out40,
    }
    va = is_val(mon.group.values)
    tr, y = ~va, mon.label.values
    print(f'train {tr.sum()}행 / 검증 {va.sum()}행\n\n== 1. 시간 피처 후보 (RF 200, 시드 0) ==')
    score = {}
    for k, fs in sets.items():
        _, p = fit_score(fs, tr, va)
        score[k] = p.mean()
        print(f'  {k:44s} ({len(fs):3d}개)  {p.mean():.4f}  ({p.mean() - score["S0 기준 @timing + 새 시간"]:+.4f})  ({time.time() - t0:.0f}s)', flush=True)
    best = max(score, key=score.get)
    print(f'\n  시드 1로 재확인 (S0와 {best}):')
    for k in dict.fromkeys(['S0 기준 @timing + 새 시간', best]):
        _, p = fit_score(sets[k], tr, va, seed=1)
        print(f'    {k:44s}  {p.mean():.4f}', flush=True)

    # 채택 규칙(+0.5%p 이상 + 시드 2개 같은 방향)은 로그를 보고 사람이 판단한다. 짧은 split 분석은 S0 기준으로 고정
    fs = sets['S0 기준 @timing + 새 시간']
    n = mon.n_total.values
    bins = [0, 10, 20, 50, 100, 200, 500, 1000, 2000, np.inf]
    lab = [f'{int(a)}~{"" if b == np.inf else int(b) - 1}' for a, b in zip(bins[:-1], bins[1:])]
    cut = pd.cut(n, bins, right=False, labels=lab)
    print('\n== 2. 짧은 split: 길이 구간별 검증 정확도 (S0, 시드 0) ==')
    _, p_all = fit_score(fs, tr, va)
    long_tr = tr & (n >= 50)
    _, p_long = fit_score(fs, long_tr, va)
    print(f'  학습: 전체 {tr.sum()}행 / 50셀 이상만 {long_tr.sum()}행 ({long_tr.sum() / tr.sum():.1%})')
    print(f'  {"길이(셀)":>10s} {"검증 수":>7s} {"비율":>6s}  {"전체로 학습":>10s}  {"50셀 이상만 학습":>14s}')
    cv = cut[va]
    for b in lab:
        s = np.asarray(cv == b)
        if s.sum():
            print(f'  {b:>10s} {s.sum():7d} {s.mean():6.1%}  {p_all[s].mean():10.4f}  {p_long[s].mean():14.4f}')
    s = n[va] >= 50
    print(f'  {"50 미만":>10s} {(~s).sum():7d} {(~s).mean():6.1%}  {p_all[~s].mean():10.4f}  {p_long[~s].mean():14.4f}')
    print(f'  {"50 이상":>10s} {s.sum():7d} {s.mean():6.1%}  {p_all[s].mean():10.4f}  {p_long[s].mean():14.4f}')
    print(f'  {"전체":>10s} {len(s):7d} {1:6.1%}  {p_all.mean():10.4f}  {p_long.mean():14.4f}')
    print(f'  찍기 수준: {1 / 95:.4f}\nDONE ({time.time() - t0:.0f}s)', flush=True)
