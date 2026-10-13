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
