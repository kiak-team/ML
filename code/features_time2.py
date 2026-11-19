"""Extra credit 시간 피처 2차 후보: features_time.py의 @tnorm을 넓힌다.

근거 (logs/screen_sliver.log, logs/verify_sliver_sets.log)
  @tnorm은 피처 20개로 @timing 10.7% → 14.0%, +3.3%p를 올려 네 그룹 중 효율이 가장 높다.
  open-world 단일 피처 AUC 상위 10개도 모두 tnorm_in_*이다. 그런데 지금 @tnorm에는
  (1) 수신만 있고 송신(tnorm_out)이 없다  (2) 20등분뿐이다  (3) "시각 → 받은 비율" 방향만 있다.
  이 세 가지를 각각 피처 그룹으로 만들어, 어느 쪽이 실제로 정확도를 올리는지 따로 잰다.

그룹 (verify_sliver_sets2.py가 조합을 비교한다)
- @tnout  (20개) 송신 셀 비율. tnorm_in의 송신 짝
- @tnqin  (20개) 수신 셀의 k%가 도착한 시각 / 전체 지속 시간. tnorm_in의 역함수 방향
                  tnorm_in은 "이 시각까지 몇 % 받았나", 이쪽은 "k%를 받기까지 얼마 걸렸나"다
- @tnqout (20개) 송신 쪽 같은 것
- @tnfine (40개) 수신 비율을 2.5% 간격으로. 20등분이 거친지 보기 위한 것 (기존 20개와 겹친다)

지속 시간이 0인 기록(셀이 전부 같은 시각에 찍힌 짧은 split)에서는 비율 자체가 정의되지 않아 -1을 쓴다.
features_time.py의 gap_at_*와 같은 규칙이다. split의 약 16%가 50셀 미만이라 무시할 양이 아니다.
"""
import os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I 에서도 load_data를 찾도록
import load_data as ld

N_NORM = 20
N_FINE = 40
UNDEF = -1.0  # 지속 시간이 0이라 정의할 수 없을 때


def _frac_at(ts_dir, dur, n_div):
    """시각 dur*j/n_div까지 받은(보낸) 셀의 비율. ts_dir는 그 방향 셀의 시각(오름차순)."""
    d = max(len(ts_dir), 1)
    return [np.searchsorted(ts_dir, dur * j / n_div, 'right') / d for j in range(1, n_div + 1)]


def _time_at(ts_dir, dur, n_div):
    """그 방향 셀의 j/n_div 만큼이 도착한 시각 / dur. 셀이 없으면 -1."""
    out = []
    for j in range(1, n_div + 1):
        if len(ts_dir) == 0:
            out.append(UNDEF); continue
        k = int(np.ceil(len(ts_dir) * j / n_div)) - 1
        out.append(float(ts_dir[k]) / dur)
    return out


def feats_time2(t):
    ts, sizes = ld.as_seq(t)
    ts, out = np.asarray(ts, dtype=float), np.asarray(sizes) > 0
    ts = ts - ts[0]
    ti, to = ts[~out], ts[out]
    dur = ts[-1]
    f = {}
    if dur <= 0:  # 셀이 전부 같은 시각 → 모양이 정의되지 않는다
        for j in range(1, N_NORM + 1):
            f[f'tnout_{j * 5:03d}'] = UNDEF
            f[f'tnqin_{j * 5:03d}'] = UNDEF
            f[f'tnqout_{j * 5:03d}'] = UNDEF
        for j in range(1, N_FINE + 1):
            f[f'tnfine_{j * 25:04d}'] = UNDEF
        return f
    for j, v in enumerate(_frac_at(to, dur, N_NORM), 1):
        f[f'tnout_{j * 5:03d}'] = v
    for j, v in enumerate(_time_at(ti, dur, N_NORM), 1):
        f[f'tnqin_{j * 5:03d}'] = v
    for j, v in enumerate(_time_at(to, dur, N_NORM), 1):
        f[f'tnqout_{j * 5:03d}'] = v
    for j, v in enumerate(_frac_at(ti, dur, N_FINE), 1):
        f[f'tnfine_{j * 25:04d}'] = v
    return f


if __name__ == '__main__':
    # 자가 확인: 손으로 만든 기록 (수신 0.0 / 0.3 / 1.2, 송신 0.1 / 1.25, 지속 1.25초)
    f = feats_time2(([0.0, 0.1, 0.3, 1.2, 1.25], [-512, 512, -512, -512, 512]))
    assert len(f) == 2 * N_NORM + N_NORM + N_FINE, len(f)
    assert f['tnout_100'] == 1.0 and f['tnfine_1000'] == 1.0
    assert abs(f['tnout_050'] - 0.5) < 1e-9          # 0.625초까지 송신 1/2개
    assert abs(f['tnqin_100'] - 1.2 / 1.25) < 1e-9   # 수신 마지막 셀은 1.2초
    assert abs(f['tnqout_050'] - 0.1 / 1.25) < 1e-9  # 송신 1/2개째는 0.1초
    # 지속 시간 0: 전부 -1
    z = feats_time2(([0.0, 0.0, 0.0], [-512, 512, -512]))
    assert set(z.values()) == {UNDEF}, set(z.values())
    # 기존 tnorm_in과 겹치는 자리는 값이 같아야 한다 (@tnfine 2.5% 간격의 짝수 번째)
    from features_time import feats_time
    g = feats_time(([0.0, 0.1, 0.3, 1.2, 1.25], [-512, 512, -512, -512, 512]))
    for j in range(1, N_NORM + 1):
        assert abs(f[f'tnfine_{j * 50:04d}'] - g[f'tnorm_in_{j * 5:03d}']) < 1e-9, j
    print(len(f), 'features OK')
