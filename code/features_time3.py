"""11/15 세트(5.1)의 @tnorm 80개를 팀원 공식 그대로 재현한다 — 두 202개 세트 맞대결용.

사용법: python -I -W ignore -X utf8 code/features_sliver3.py <npz> <out.csv>

왜 따로 뽑나
  5.1은 features_time.py의 @tnorm을 40등분 수신·송신 80개로 바꿨다. 4.7(11/19)은 20등분 수신을
  두고 송신 쪽 60개를 더했다. 둘은 다른 세트인데 개수가 우연히 같고, 측정도 holdout 1번 대
  group 5-fold × 시드 3개로 달라서 직접 비교할 수 없다(docs/EXTRA_CREDIT.md 6절).

  features_time2.py의 @tnfine(40등분 수신)은 5.1의 tnorm_in과 값·이름이 같지만, 지속 시간이 0인
  기록에서 -1(UNDEF)을 넣는 점이 다르다. 팀원 코드는 그 경우도 그냥 계산한다(dur*j/N = 0이므로
  시각 0의 셀이 세어져 1.0이 된다). 맞대결에서 그 차이가 섞이면 안 되므로, 여기서는 features_time.py
  (origin/main 기준)의 식을 그대로 쓴다.

이름은 5.1과 같은 규칙(끝 4자리 = 천분율)이되 접두사만 tfin_/tfout_으로 둔다. 기존 CSV의
tnorm_in_*(20등분)과 열 이름이 겹치지 않게 하려는 것이다.
"""
import os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I 에서도 load_data를 찾도록
import load_data as ld

N_NORM = 40


def feats_time3(t):
    ts, sizes = ld.as_seq(t)
    ts, out = np.asarray(ts, dtype=float), np.asarray(sizes) > 0
    ts = ts - ts[0]
    dur = ts[-1]
    f = {}
    for name, x in (('tfin', ts[~out]), ('tfout', ts[out])):
        for j in range(1, N_NORM + 1):
            f[f'{name}_{j * 1000 // N_NORM:04d}'] = (
                np.searchsorted(x, dur * j / N_NORM, 'right') / max(len(x), 1))
    return f


if __name__ == '__main__':
    # 자가 확인: 수신 0.0 / 0.3 / 1.2, 송신 0.1 / 1.25, 지속 1.25초 (features_time.py와 같은 기록)
    f = feats_time3(([0.0, 0.1, 0.3, 1.2, 1.25], [-512, 512, -512, -512, 512]))
    assert len(f) == 2 * N_NORM, len(f)
    assert f['tfin_1000'] == 1.0 and f['tfout_1000'] == 1.0
    assert f['tfout_0500'] == 0.5 and abs(f['tfin_0250'] - 2 / 3) < 1e-9
    # origin/main의 features_time.py와 값이 같은지 직접 대조 (그 파일이 40등분이면)
    from features_time import feats_time, N_NORM as M
    g = feats_time(([0.0, 0.1, 0.3, 1.2, 1.25], [-512, 512, -512, -512, 512]))
    if M == N_NORM and 'tnorm_in_1000' in g:
        for j in range(1, N_NORM + 1):
            k = f'{j * 1000 // N_NORM:04d}'
            assert abs(f[f'tfin_{k}'] - g[f'tnorm_in_{k}']) < 1e-12, k
            assert abs(f[f'tfout_{k}'] - g[f'tnorm_out_{k}']) < 1e-12, k
        print('features_time.py의 tnorm 80개와 값이 전부 같다')
    else:
        print(f'features_time.py가 {M}등분이라 대조는 건너뛴다')
    print(len(f), 'features OK')
