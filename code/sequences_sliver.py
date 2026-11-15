"""Extra credit: split(조각) 하나를 NN 입력 시퀀스로 만든다. numpy만 필요.

사용법:
  python -I -X utf8 code/sequences_sliver.py <Downloads>/sliver_mon.npz --stats            # 길이 분포·잘리는 비율만
  python -I -X utf8 code/sequences_sliver.py <Downloads>/sliver_mon.npz <out.npz> [--length 3000]

입력 형식 (sequences.py와 같은 방식 + 시간):
  방향 dir   (n, L) int8    +1 송신, -1 수신, 0 = 셀 없음(뒤를 0으로 채움). L보다 길면 뒤를 자른다
  시각 t_ms  (n, L) uint16  첫 셀 기준 밀리초. 65535 = 65.5초 이상(split의 99% 이상은 그 안이다). 채운 칸은 0
             split에서는 방향 순서가 사이트와 거의 무관하고 시간에 정보가 남으므로(EXTRA_CREDIT.md 4.2) 시간을 함께 넣는다
             float32 대신 uint16(ms)로 저장해 크기를 반으로 줄였다. 쓸 때는 t_ms / 1000.0 (초)
  label (n,) int16 (unmon = -1), group, part, length(자르기 전 길이), is_test(ld.sliver_is_test)
  split만 넣는다(join 제외). 빈 split은 npz에서 이미 빠져 있다. train/test·fold는 반드시 group 단위로 나눈다.

L 기본 3000: mon split의 94.4%가 잘리지 않고 셀의 92.8%가 남는다 (중앙값 529셀, 95% 3,153셀).
출력은 크니(mon 약 0.8GB) 저장소 밖(Downloads)에 둔다.
"""
import argparse, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import load_data as ld

CHECK_L = [500, 1000, 2000, 3000, 5000]


def encode(times, dirs, off, idx, L):
    n = len(idx)
    D, T, lens = np.zeros((n, L), np.int8), np.zeros((n, L), np.uint16), np.zeros(n, np.int32)
    for r, i in enumerate(idx):
        a, b = off[i], off[i + 1]
        lens[r] = b - a
        k = min(L, b - a)
        D[r, :k] = dirs[a:a + k]
        T[r, :k] = np.minimum(np.round(times[a:a + k].astype(np.float64) * 1000), 65535)
    return D, T, lens


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('npz')
    ap.add_argument('out', nargs='?')
    ap.add_argument('--length', type=int, default=3000)
    ap.add_argument('--stats', action='store_true')
    a = ap.parse_args()
    d = dict(np.load(a.npz))
    off, idx = d['offsets'], np.flatnonzero(d['part'] < ld.JOIN_PART)
    lens = np.diff(off)[idx]
    print(f'{a.npz}: split {len(idx)}개, 길이 분위수 5/25/50/75/95/99%: '
          + ' / '.join(f'{v:.0f}' for v in np.percentile(lens, [5, 25, 50, 75, 95, 99])))
    for L in CHECK_L:
        print(f'  L={L:5d}: 잘리는 split {(lens > L).mean():6.1%}, 남는 셀 {np.minimum(lens, L).sum() / lens.sum():6.1%}')
    dur = d['times'][off[idx + 1] - 1]
    print(f'  지속 시간 65.5초 초과(t_ms 상한에 걸림): {(dur > 65.535).mean():.2%}')
    if a.stats or not a.out:
        sys.exit()
    D, T, n = encode(d['times'], d['dirs'], off, idx, a.length)
    # 자가 확인: 첫 행이 원본과 같은지, 패딩이 0인지
    k = min(a.length, n[0])
    assert (D[0, :k] == d['dirs'][off[idx[0]]:off[idx[0]] + k]).all() and (D[0, k:] == 0).all() and T[0, 0] == 0
    np.savez_compressed(a.out, dir=D, t_ms=T, label=d['label'][idx], group=d['group'][idx], part=d['part'][idx],
                        length=n, is_test=ld.sliver_is_test(d['group'][idx]))
    print(f'저장: {a.out}  dir/t_ms {D.shape}, 크기 {os.path.getsize(a.out) / 1e6:.0f}MB')
