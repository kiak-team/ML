"""공용 데이터 로더: 교수님 제공 load_pickle_code.ipynb와 같은 규칙.

팀 전체가 이 파일만 import 해서 쓰면 라벨·방향·크기 규칙이 어긋나지 않는다.

규칙 (노트북과 동일)
- 라벨: URL 번호 i (0~949) → 사이트 번호 i // 10 (0~94, 95-class)
         USE_SUBLABEL=True 면 URL 번호 i 그대로 (950-class)
- 방향: c > 0 이면 +1(송신, outgoing), 그 외(0.0 포함)는 -1(수신, incoming)
- 시간: abs(c)
- 크기: pkl에 크기 정보가 없으므로 방향 * 512 로 고정

사용 예
    import load_data as ld
    X1, X2, y = ld.load_mon('C:/Users/user/Downloads')          # 노트북과 같은 배열
    X1u, X2u = ld.load_unmon('C:/Users/user/Downloads')
    for times, sizes, label, url, inst in ld.iter_mon(D): ...   # 메모리를 덜 쓰는 방식
"""
import os
import pickle

URL_PER_SITE = 10
TOTAL_URLS = 950
CELL_SIZE = 512
MON_FILE = 'mon_standard.pkl'
UNMON_FILE = 'unmon_standard10.pkl'


def direction(c):
    """+1 = 송신(outgoing), -1 = 수신(incoming). 노트북의 `dr = 1 if c > 0 else -1`."""
    return 1 if c > 0 else -1


def convert(trace):
    """pkl 기록 하나 → (time_seq, size_seq). 노트북 반복문 한 번과 같다."""
    time_seq = [abs(c) for c in trace]
    size_seq = [direction(c) * CELL_SIZE for c in trace]
    return time_seq, size_seq


def as_seq(t):
    """pkl 기록(부호 붙은 시간 리스트) 또는 이미 바꾼 (times, sizes) 튜플 → (times, sizes). 피처 함수가 둘 다 받도록."""
    return t if isinstance(t, tuple) else convert(t)


def site_label(url_idx, use_sublabel=False):
    return url_idx if use_sublabel else url_idx // URL_PER_SITE


def load_raw_mon(data_dir):
    """원본 dict {url 번호: [기록 20개]} 그대로."""
    with open(os.path.join(data_dir, MON_FILE), 'rb') as f:
        return pickle.load(f)


def load_raw_unmon(data_dir, filename=UNMON_FILE):
    """원본 list [기록 10,000개] 그대로."""
    with open(os.path.join(data_dir, filename), 'rb') as f:
        return pickle.load(f)


def iter_mon(data_dir, use_sublabel=False):
    """(time_seq, size_seq, label, url, inst)를 하나씩 돌려준다. inst = URL 안에서 몇 번째 방문인지(0~19)."""
    data = load_raw_mon(data_dir)
    for i in range(TOTAL_URLS):
        label = site_label(i, use_sublabel)
        for j, sample in enumerate(data[i]):
            times, sizes = convert(sample)
            yield times, sizes, label, i, j


def load_mon(data_dir, use_sublabel=False):
    """노트북 1번 셀과 같은 X1(시간), X2(방향*512), y(라벨). 19,000개."""
    X1, X2, y = [], [], []
    for times, sizes, label, _, _ in iter_mon(data_dir, use_sublabel):
        X1.append(times); X2.append(sizes); y.append(label)
    return X1, X2, y


def load_unmon(data_dir, filename=UNMON_FILE):
    """노트북 2번 셀과 같은 X1(시간), X2(방향*512). 10,000개, 라벨 없음."""
    X1, X2 = [], []
    for sample in load_raw_unmon(data_dir, filename):
        times, sizes = convert(sample)
        X1.append(times); X2.append(sizes)
    return X1, X2


# ---------------- Extra credit: BigEnough + TrafficSliver BWR (조교 제공 zip) ----------------
# zip 안 파일: mon = '{사이트}-{방문 0~199}_split_{0~4}.cell' / '_join.cell', unmon = '{번호}_split_k.cell' / '_join.cell'
# 한 줄 = '시간\t방향(±1)\t크기(±512)'. join = split 5개를 합친 방어 전 원본(셀 수 = split 합).
JOIN_PART = 5


def _sliver_name(n):
    """'mon_25/3-139_split_0.cell' → (label, group, part). unmon label = -1."""
    stem = n.rsplit('/', 1)[-1][:-len('.cell')]
    key, part = stem.rsplit('_split_', 1) if '_split_' in stem else (stem[:-len('_join')], JOIN_PART)
    if '-' in key:
        site, inst = map(int, key.split('-'))
        return site, site * 1000 + inst, int(part)
    return -1, 10**6 + int(key), int(part)


def build_sliver(zip_path, out_path):
    """zip을 풀지 않고 읽어 npz 하나로 저장. 시간은 파일마다 0부터, 빈 split 파일은 뺀다.
    저장: times(float32), dirs(int8), offsets(파일 i = [offsets[i], offsets[i+1])), label, group, part.
    group = 같은 방문(split 5개 + join)이 같은 값 → train/test는 반드시 group 단위로 나눈다."""
    import zipfile
    import numpy as np
    z = zipfile.ZipFile(zip_path)
    T, Dr, meta, empty = [], [], [], 0
    lens = {}  # group → [split 합, join 길이] 검증용
    for n in z.namelist():
        if not n.endswith('.cell'):
            continue
        label, group, part = _sliver_name(n)
        a = np.array(z.read(n).split(), dtype=np.float64).reshape(-1, 3)
        s = lens.setdefault(group, [0, 0])
        s[part == JOIN_PART] += len(a)
        if len(a) == 0:
            empty += 1
            continue
        T.append((a[:, 0] - a[0, 0]).astype(np.float32))
        Dr.append(a[:, 1].astype(np.int8))
        meta.append((label, group, part))
    bad = sum(1 for sp, jn in lens.values() if sp != jn)
    assert bad == 0, f'join 길이 ≠ split 합인 방문 {bad}개'
    meta = np.array(meta, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum([len(t) for t in T])])
    np.savez(out_path, times=np.concatenate(T), dirs=np.concatenate(Dr), offsets=offsets,
             label=meta[:, 0].astype(np.int16), group=meta[:, 1], part=meta[:, 2].astype(np.int8))
    print(f'{zip_path}: 방문 {len(lens)}, 저장 파일 {len(meta)} (빈 split {empty}개 제외), 셀 {offsets[-1]:,}')


def iter_sliver(npz_path, parts=range(JOIN_PART)):
    """(times, sizes, label, group, part)를 하나씩. 기본은 split만(join 제외), parts=[5]면 join만."""
    import numpy as np
    d = dict(np.load(npz_path))  # npz는 d['키']마다 디스크에서 다시 읽으므로 한 번에 메모리로
    off, keep = d['offsets'], np.isin(d['part'], list(parts))
    for i in np.flatnonzero(keep):
        a, b = off[i], off[i + 1]
        yield d['times'][a:b], d['dirs'][a:b].astype(np.int16) * CELL_SIZE, int(d['label'][i]), int(d['group'][i]), int(d['part'][i])



def sliver_is_test(group):
    """test로 떼어 둘 방문(약 10%). 방문(group) 단위라 split 5개와 join이 함께 움직인다.
    행 순서와 무관한 고정 규칙(정수 해시)이라 누가 언제 돌려도 같은 test가 나온다. 튜닝 중에는 보지 않는다."""
    import numpy as np
    return (np.asarray(group, dtype=np.uint64) * np.uint64(2654435761) % np.uint64(2**32)) % 10 == 0

if __name__ == '__main__':
    # python -I -X utf8 code/load_data.py <zip> <out.npz>
    import sys
    build_sliver(sys.argv[1], sys.argv[2])
