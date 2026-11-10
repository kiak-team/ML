"""Extra credit 시간 피처 (features_sliver.py가 기존 피처 뒤에 붙인다).

근거 (logs/feature_importance_sliver.log): split에서는 방향·순서 피처가 쓸모없고 시간 피처(@timing)만 남는다.
기존 시간 피처는 요약값 18개뿐이라, 시간 축 패턴을 직접 담는다. split 길이: 중앙값 9초, 95% 48초.

- @tcum  (38개) 고정 시각 T초까지 받은/보낸 셀 수 (누적). 셀 순서가 흩어져도 "언제 로딩되나"는 남는다
- @tbin  (40개) 처음 10초를 0.5초 칸으로 나눈 칸별 수신/송신 셀 수
- @tnorm (20개) 기록 길이를 20등분한 시각까지 받은 셀 비율 (시간 축 CUMUL, 길이와 무관한 모양)
- @gap   (26개) 셀 사이 빈 시간: 임계값별 개수, 가장 긴 10개, 0.2초 넘는 빈 시간이 처음 10번 시작한 시각
                 BWR은 셀을 50~70개 묶음으로 경로를 고르므로, 한 경로에서는 묶음 사이가 빈 시간으로 보인다
"""
import os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I 에서도 load_data를 찾도록
import load_data as ld

T_CUM = [0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60]
BIN, N_BIN = 0.5, 20
N_NORM = 20
GAP_TH = [0.05, 0.1, 0.2, 0.5, 1, 2]
N_TOPGAP, N_GAPAT, GAPAT_TH = 10, 10, 0.2


def feats_time(t):
    ts, sizes = ld.as_seq(t)
    ts, out = np.asarray(ts, dtype=float), np.asarray(sizes) > 0
    ts = ts - ts[0]  # pkl 기록도 같은 기준이 되도록
    f = {}
    ti, to = ts[~out], ts[out]
    for T in T_CUM:
        f[f'tcum_in_{T:g}'] = int(np.searchsorted(ti, T, 'right'))
        f[f'tcum_out_{T:g}'] = int(np.searchsorted(to, T, 'right'))
    edges = np.arange(N_BIN + 1) * BIN
    for name, x in (('in', ti), ('out', to)):
        for j, c in enumerate(np.histogram(x, edges)[0]):
            f[f'tbin_{name}_{j + 1:02d}'] = int(c)
    dur = ts[-1]
    for j in range(1, N_NORM + 1):
        f[f'tnorm_in_{j * 5:03d}'] = np.searchsorted(ti, dur * j / N_NORM, 'right') / max(len(ti), 1)
    gaps = np.diff(ts)
    for th in GAP_TH:
        f[f'gap_n_{th:g}'] = int((gaps > th).sum())
    top = np.sort(gaps)[::-1][:N_TOPGAP]
    for j in range(N_TOPGAP):
        f[f'gap_top_{j + 1:02d}'] = float(top[j]) if j < len(top) else 0.0
    at = ts[:-1][gaps > GAPAT_TH][:N_GAPAT]
    for j in range(N_GAPAT):
        f[f'gap_at_{j + 1:02d}'] = float(at[j]) if j < len(at) else -1.0  # -1 = 그만큼 빈 시간이 없다
    return f


if __name__ == '__main__':
    # 자가 확인: 손으로 만든 기록
    f = feats_time(([0.0, 0.1, 0.3, 1.2, 1.25], [-512, 512, -512, -512, 512]))
    assert f['tcum_in_0.25'] == 1 and f['tcum_out_0.25'] == 1 and f['tcum_in_1.5'] == 3
    assert f['tbin_in_01'] == 2 and f['tbin_in_03'] == 1 and f['tbin_out_03'] == 1
    assert f['gap_n_0.5'] == 1 and abs(f['gap_top_01'] - 0.9) < 1e-9 and f['gap_at_01'] == 0.3 and f['gap_at_02'] == -1
    assert f['tnorm_in_100'] == 1.0
    print(len(f), 'features OK')
