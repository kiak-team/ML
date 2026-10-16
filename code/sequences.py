"""시퀀스 모델(DF 등 1D-CNN)용 고정 길이 입력을 만든다. 표준 라이브러리 + numpy.

사용법:
  python -I -X utf8 code/sequences.py <pkl 폴더> --stats                       # 길이 분포·잘리는 비율만 출력 (약 1분)
  python -I -X utf8 code/sequences.py <pkl 폴더> --length 5000                 # sequences/seq_L5000_skip0.npz
  python -I -X utf8 code/sequences.py <pkl 폴더> --length 5000 --skip 14       # open-world용 (앞 14셀 버림)
  python -I -X utf8 code/sequences.py <pkl 폴더> --length 5000 --time          # 시각 시퀀스도 함께 저장

입력 형식 (DF 논문과 같은 방식):
  기록 하나 = 방향 시퀀스 (+1 송신, -1 수신). load_data.py의 방향 규칙을 그대로 쓴다(0.0 → 수신)
  길이 L로 맞춘다: L보다 길면 뒤를 자르고, 짧으면 뒤를 0으로 채운다. 0 = "셀 없음"
  --time: 같은 위치의 시각(초, 첫 셀 기준)을 float32로 함께 저장한다. 채운 칸은 0
  --skip K: 앞 K셀을 버린 뒤 길이를 맞춘다. open-world에서는 앞 14셀에 수집 환경 흔적이 있으므로
            --skip 14를 쓴다(docs/OPENWORLD.md 2절). closed-world는 0이면 된다

출력 (sequences/, git 제외 — 용량이 커서 각자 만든다):
  seq_L<L>_skip<K>.npz
    mon_dir   (19000, L) int8     mon_site (19000,) 0~94   mon_url  mon_inst (0~19)
    unmon_dir (10000, L) int8
    mon_time, unmon_time (19000/10000, L) float32   ← --time일 때만
    mon_len, unmon_len   원래 길이 (skip 이후)

분할은 피처 실험과 같게 쓰면 된다: mon train = inst 0~17, test = inst 18~19. CV fold = inst % 5.
unmon은 openworld.py와 같은 시드 0 무작위 분할(test 1,000개)을 쓰려면 openworld.split()의 순서를 따른다.
"""
import argparse, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import load_data as ld

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECK_L = [1000, 2000, 3000, 5000, 7000, 10000]


def encode(traces, L, skip, with_time):
    n = len(traces)
    D = np.zeros((n, L), np.int8)
    T = np.zeros((n, L), np.float32) if with_time else None
    lens = np.zeros(n, np.int32)
    for i, tr in enumerate(traces):
        tr = tr[skip:]
        lens[i] = len(tr)
        k = min(L, len(tr))
        D[i, :k] = [ld.direction(c) for c in tr[:k]]
        if with_time and k:
            t = np.abs(np.asarray(tr[:k], np.float64))
            T[i, :k] = t - t[0]
    return D, T, lens


def stats(name, lens):
    p = np.percentile(lens, [5, 25, 50, 75, 95, 99])
    print(f'{name}: n={len(lens)}  길이 평균 {lens.mean():.0f}  최소 {lens.min()}  최대 {lens.max()}')
    print('  분위수 5/25/50/75/95/99%: ' + ' / '.join(f'{v:.0f}' for v in p))
    for L in CHECK_L:
        cut = lens > L
        kept = np.minimum(lens, L).sum() / lens.sum()
        pad = np.mean(np.maximum(L - lens, 0) / L)
        print(f'  L={L:>6}: 잘리는 기록 {cut.mean():6.1%}  보존되는 셀 {kept:6.1%}  평균 0 채움 비율 {pad:6.1%}')


def main():
    ap = argparse.ArgumentParser(description='시퀀스 모델용 고정 길이 입력. 자세한 사용법은 파일 맨 위 설명 참고.')
    ap.add_argument('pkl_dir')
    ap.add_argument('--length', type=int, default=5000)
    ap.add_argument('--skip', type=int, default=0)
    ap.add_argument('--time', action='store_true')
    ap.add_argument('--stats', action='store_true', help='길이 분포만 출력하고 저장하지 않는다')
    a = ap.parse_args()

    raw = ld.load_raw_mon(a.pkl_dir)
    mon, site, url, inst = [], [], [], []
    for i in range(ld.TOTAL_URLS):
        for j, tr in enumerate(raw[i]):
            mon.append(tr); site.append(ld.site_label(i)); url.append(i); inst.append(j)
    unmon = ld.load_raw_unmon(a.pkl_dir)

    if a.stats:
        for skip in sorted({0, a.skip}):
            print(f'--- skip={skip}')
            stats('mon', np.array([max(len(t) - skip, 0) for t in mon]))
            stats('unmon', np.array([max(len(t) - skip, 0) for t in unmon]))
            m = np.array([max(len(t) - skip, 0) for t in mon]); ii = np.array(inst)
            print(f'  참고: mon 길이 중앙값 inst 0~17 {np.median(m[ii <= 17]):.0f} / inst 18 {np.median(m[ii == 18]):.0f}'
                  f' / inst 19 {np.median(m[ii == 19]):.0f}')
        return

    Dm, Tm, lm = encode(mon, a.length, a.skip, a.time)
    Du, Tu, lu = encode(unmon, a.length, a.skip, a.time)
    out = dict(mon_dir=Dm, mon_site=np.array(site, np.int16), mon_url=np.array(url, np.int16),
               mon_inst=np.array(inst, np.int8), mon_len=lm, unmon_dir=Du, unmon_len=lu)
    if a.time:
        out.update(mon_time=Tm, unmon_time=Tu)
    os.makedirs(os.path.join(ROOT, 'sequences'), exist_ok=True)
    path = os.path.join(ROOT, 'sequences', f'seq_L{a.length}_skip{a.skip}.npz')
    np.savez_compressed(path, **out)
    print(f'저장: {path}  ({os.path.getsize(path) / 1e6:.0f} MB)')
    for name, lens in (('mon', lm), ('unmon', lu)):
        print(f'  {name}: {len(lens)}개, L={a.length}에서 잘린 기록 {np.mean(lens > a.length):.1%}')


if __name__ == '__main__':
    main()
