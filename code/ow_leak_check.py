"""open-world 데이터 누수(앞 14셀 방향 패턴) 검증. reports/openworld.md 0절의 근거.

사용법:
  python -I -X utf8 code/ow_leak_check.py                              # [A] pkl 피처 CSV만 (원본 불필요)
  python -I -X utf8 code/ow_leak_check.py <mon.zip> <unmon.zip>        # [A] + [B] TrafficSliver 원본
  > logs/ow_leak_check.log  (stdout만. 약 20초)

조교 공지: mon trace의 99.98%가 앞 14셀 방향이 같고 unmon에서는 1만 개 중 1개뿐이다.
"인덱스 7번(8번째) 셀이 outgoing이면 monitored" 규칙 하나로 이진 분류가 99.99%가 된다.

[A] pkl: A팀이 만든 leak 포함(features_extra2_*) / 제거(ow_skip14_*) 피처에서 같은 규칙을 잰다.
    dir_008 = 8번째 셀 방향. ow_skip14에서는 22번째 셀을 가리킨다(앞 14셀을 버린 뒤 8번째).
[B] TrafficSliver: zip을 풀지 않고 파일마다 앞 14셀 방향을 읽어
    join(방어 없음) / split 전체 / split_k별 / "앞 7셀이 핸드셰이크로 시작하는 split" 부분집합에서
    mon·unmon 최빈 패턴과 그 비율, idx7 규칙의 acc·TPR·FPR을 낸다.
    split 시각이 join 시계를 갖는지(파일마다 0부터인지)도 함께 센다 (E6 merge 기준에 필요).
"""
import collections, os, re, sys, zipfile

import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = 14
HANDSHAKE7 = '--+-+-+'  # join/pkl 기록이 모두 이렇게 시작한다


def pat(h):
    return ''.join('+' if x == 1 else '-' if x == -1 else '.' for x in h)


def rule(hm, hu):
    """'idx7 outgoing → mon' 규칙. hm/hu = (n, >=8) 방향 배열. 14셀 미만 기록은 0(=outgoing 아님)."""
    pm, pu = hm[:, 7] == 1, hu[:, 7] == 1
    acc = (pm.sum() + (~pu).sum()) / (len(pm) + len(pu))
    return acc, pm.mean(), pu.mean()


def report(name, hm, hu):
    print(f'\n### {name}: mon {len(hm)}, unmon {len(hu)}')
    for lab, h in [('mon', hm), ('unmon', hu)]:
        c = collections.Counter(map(pat, h)).most_common(3)
        print(f'  {lab:5s} 최빈 패턴 top3: ' + ', '.join(f'{p} {n / len(h):.2%}' for p, n in c))
    top = collections.Counter(map(pat, hm)).most_common(1)[0][0]
    print(f'  mon 최빈 패턴이 unmon에 나오는 비율: {np.mean([pat(x) == top for x in hu]):.4%}')
    acc, tpr, fpr = rule(hm, hu)
    print(f'  규칙 "idx7 outgoing → mon": acc {acc:.4f}  TPR {tpr:.4f}  FPR {fpr:.4f}')


def check_pkl_csv():
    print('=' * 20, '[A] pkl 피처 CSV (A팀 산출물)', '=' * 20)
    cols = [f'dir_{j:03d}' for j in range(1, K + 1)]
    for tag, stem in [('leak 포함', 'features_extra2'), ('leak 제거 (앞 14셀 버림)', 'ow_skip14')]:
        m = pd.read_csv(os.path.join(ROOT, 'features', f'{stem}_mon.csv'), usecols=cols).to_numpy()
        u = pd.read_csv(os.path.join(ROOT, 'features', f'{stem}_unmon.csv'), usecols=cols).to_numpy()
        report(f'{tag}: {stem}_*.csv', m, u)


def scan(zip_path):
    """파일마다 (앞 K셀 방향, 길이, 첫 시각, label, group, part). part 5 = join."""
    pm = re.compile(r'(\d+)-(\d+)_(?:split_([0-4])|join)$')
    pu = re.compile(r'(\d+)_(?:split_([0-4])|join)$')
    z = zipfile.ZipFile(zip_path)
    H, L, T0, meta = [], [], [], []
    for n in z.namelist():
        if not n.endswith('.cell'):
            continue
        stem = n.rsplit('/', 1)[-1][:-len('.cell')]
        m = pm.match(stem)
        if m:
            lab, grp, part = int(m[1]), int(m[1]) * 1000 + int(m[2]), int(m[3]) if m[3] else 5
        else:
            u = pu.match(stem)
            assert u, f'파일명 파싱 실패: {n}'
            lab, grp, part = -1, 10**6 + int(u[1]), int(u[2]) if u[2] else 5
        rows = [r for r in z.read(n).split(b'\n') if r.strip()]
        h = np.zeros(K, np.int8)
        for i, r in enumerate(rows[:K]):
            h[i] = int(r.split(b'\t')[1])
        H.append(h); L.append(len(rows)); meta.append((lab, grp, part))
        T0.append(float(rows[0].split(b'\t')[0]) if rows else np.nan)
    meta = np.array(meta)
    return dict(head=np.array(H), length=np.array(L), t0=np.array(T0),
                label=meta[:, 0], group=meta[:, 1], part=meta[:, 2])


def check_sliver(mon_zip, unmon_zip):
    print('\n' + '=' * 20, '[B] TrafficSliver 원본 zip', '=' * 20)
    m, u = scan(mon_zip), scan(unmon_zip)
    for lab, d in [('mon', m), ('unmon', u)]:
        sp = d['part'] < 5
        print(f'{lab}: 파일 {len(d["part"])} (join {(~sp).sum()}, split {sp.sum()}, 빈 split {(sp & (d["length"] == 0)).sum()})'
              + (f', 사이트 {len(set(d["label"]))}개 [{d["label"].min()}..{d["label"].max()}]' if lab == 'mon' else ''))
        ne = sp & (d['length'] > 0)
        print(f'   split 첫 시각 |t0| < 1e-9: {np.mean(np.abs(d["t0"][ne]) < 1e-9):.2%}  '
              f'→ split 시각은 파일마다 0부터 다시 시작한다 (join 시계 offset 없음)')
    for name, parts in [('join (방어 없음)', [5]), ('split 0-4 전체 (방어 있음)', [0, 1, 2, 3, 4])] + \
                       [(f'split_{k}', [k]) for k in range(5)]:
        km = np.isin(m['part'], parts) & (m['length'] > 0)
        ku = np.isin(u['part'], parts) & (u['length'] > 0)
        report(name, m['head'][km], u['head'][ku])
    # join의 시작 구간을 그대로 받은 split
    sub = {}
    for lab, d in [('mon', m), ('unmon', u)]:
        k = (d['part'] < 5) & (d['length'] >= K)
        hs = np.array([pat(x[:7]) == HANDSHAKE7 for x in d['head'][k]])
        per_visit = collections.Counter(collections.Counter(d['group'][k][hs]).values())
        sub[lab] = d['head'][k][hs]
        tot = ((d['part'] < 5) & (d['length'] > 0)).sum()
        print(f'\n{lab}: 앞 7셀이 "{HANDSHAKE7}"인 split {hs.sum()} / 비어있지 않은 split {tot} ({hs.sum() / tot:.2%}), '
              f'방문당 개수 분포 {dict(per_visit)}')
    report(f'split 중 앞 7셀이 "{HANDSHAKE7}"인 부분집합', sub['mon'], sub['unmon'])


if __name__ == '__main__':
    check_pkl_csv()
    if len(sys.argv) == 3:
        check_sliver(sys.argv[1], sys.argv[2])
