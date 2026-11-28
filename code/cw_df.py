"""closed-world DF 1D-CNN (Sirinam et al., CCS 2018) — PyTorch. Colab GPU에서 돌리는 것을 전제로 한다.

입력: code/sequences.py로 만든 sequences/seq_L5000_skip0.npz (방향 시퀀스, 길이 5,000, mon 33% 잘림)
분할: cw_tuning.py와 같다. test = load_data.mon_is_test, CV fold = inst % 5
학습: epoch 수를 고정한다(검증 fold로 early stopping 하지 않음 → OOF가 깨끗하다).
      fold별 epoch마다 검증 정확도를 찍어 두니, 그걸 보고 --epochs를 정한다.
출력: results/cw_oof/df_base.npz  (5 fold 다 돌렸을 때만. cw_tuning.py vote/summary에 그대로 섞인다)
      results/cw_test/df_test.npz (--final일 때. 정확도는 출력하지 않는다 → cw_tuning.py test에서 한 번에 본다)

Colab 순서:
  !git clone https://github.com/kiak-team/ml.git && cd ml
  # pkl 두 파일(mon_standard.pkl 등)을 Drive에서 data/ 로 복사
  !python -I -X utf8 code/sequences.py data --length 5000            # → sequences/seq_L5000_skip0.npz (몇 분)
  !python -I -W ignore -X utf8 code/cw_df.py --folds 0 --epochs 30   # ① fold 0만 빠르게: 몇 epoch에서 수렴하는지 확인
  !python -I -W ignore -X utf8 code/cw_df.py --epochs 30             # ② 5-fold 전체 → OOF 저장 (앙상블 후보)
  !python -I -W ignore -X utf8 code/cw_df.py --final --epochs 30     # ③ 월요일: 전체 train 학습 → test 확률 저장
  결과 파일(results/cw_oof/df_base.npz 등)은 로컬 레포의 같은 위치로 내려받는다.

성능 올리기 변형 (--tag로 이름을 달리하면 결과가 따로 저장되고, cw_tuning.py vote/ensemble에 모두 들어간다):
  --seed 1 --tag s1                     같은 설정, 다른 시드 → 시드 앙상블 (딥러닝은 시드만 바꿔 평균해도 오른다)
  --sched cosine --tag cos              학습률을 점점 줄이기 → 끝부분 진동(±0.5%p)을 줄인다
  --shift 20 --tag shift                학습 증강: 시퀀스 시작을 0~20칸 무작위로 밀기
  sequences.py --length 10000 → --npz sequences/seq_L10000_skip0.npz --tag L10k   잘림 33% → 0%
  sequences.py --length 5000 --time → --npz sequences/seq_L5000_skip0.npz --input tiktok --tag tt  방향×시각 (Tik-Tok)
                                      → --input dir+iat --tag iat  방향 + 시간 간격 2채널
  sequences.py --length 10000 --time → --npz sequences/seq_L10000_skip0.npz --input tam --tag tam
                                      시간 구간(44ms)별 송신·수신 셀 수 (Robust Fingerprinting, USENIX Security 2023)
"""
import argparse, json, os, sys, time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import load_data as ld

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DF(nn.Module):
    """DF 논문 구조: conv 블록 4개(32/64/128/256, kernel 8) + FC 512·512. 블록 1만 ELU, 나머지 ReLU.
    C = 입력 채널 수 (방향만 1, 방향+시간 간격 2)."""
    def __init__(self, L, n_cls=95, C=1):
        super().__init__()
        def block(cin, cout, act):
            return nn.Sequential(
                nn.Conv1d(cin, cout, 8, padding='same'), nn.BatchNorm1d(cout), act(),
                nn.Conv1d(cout, cout, 8, padding='same'), nn.BatchNorm1d(cout), act(),
                nn.MaxPool1d(8, stride=4, padding=2), nn.Dropout(0.1))
        self.features = nn.Sequential(block(C, 32, nn.ELU), block(32, 64, nn.ReLU),
                                      block(64, 128, nn.ReLU), block(128, 256, nn.ReLU))
        with torch.no_grad():
            n_flat = self.features(torch.zeros(1, C, L)).numel()
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(n_flat, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.7),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, n_cls))

    def forward(self, x):
        return self.head(self.features(x))


def make_tam(D, T, sec=80.0, bins=1800, chunk=1000):
    """방향 D (n, L) int8과 시각 T (n, L) → (n, 2, bins) float32. 0행 = 송신(+1) 수, 1행 = 수신(-1) 수.
    채운 칸(D == 0)은 세지 않는다. sec를 넘는 셀은 마지막 구간에 넣는다. 메모리를 아끼려고 chunk행씩 계산한다."""
    n = len(D)
    out = np.zeros((n, 2, bins), np.float32)
    for s in range(0, n, chunk):
        d, t = D[s:s + chunk], T[s:s + chunk]
        b = np.minimum((t / sec * bins).astype(np.int64), bins - 1)
        r = np.broadcast_to(np.arange(len(d))[:, None], d.shape)
        keep = d != 0
        np.add.at(out[s:s + chunk], (r[keep], (d[keep] < 0).astype(np.int64), b[keep]), 1)
    return out


def as_batch(xb):
    """(n, L) int8 방향 → (n, 1, L) float / (n, C, L) 그대로 float"""
    xb = xb.float()
    return xb.unsqueeze(1) if xb.dim() == 2 else xb


def predict(model, X, dev, bs=512):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = as_batch(torch.from_numpy(X[i:i + bs])).to(dev)
            with torch.autocast(dev.type, enabled=dev.type == 'cuda'):
                out.append(torch.softmax(model(xb).float(), 1).cpu().numpy())
    return np.concatenate(out)


def train(Xtr, ytr, args, dev, Xva=None, yva=None, tag=''):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    C, L = (1, Xtr.shape[1]) if Xtr.ndim == 2 else Xtr.shape[1:]
    model = DF(L, C=C).to(dev)
    opt = torch.optim.Adamax(model.parameters(), lr=args.lr)  # 논문 설정
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs) if args.sched == 'cosine' else None
    scaler = torch.cuda.amp.GradScaler(enabled=dev.type == 'cuda')
    lossf = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(Xtr)  # int8 그대로 두고 배치마다 float로
    yt = torch.from_numpy(ytr.astype(np.int64))
    curve = []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, perm, tot = time.time(), torch.randperm(len(Xt)), 0.0
        for i in range(0, len(perm), args.bs):
            idx = perm[i:i + args.bs]
            xb = Xt[idx]
            if args.shift:  # 증강: 앞쪽을 0~shift칸 무작위로 밀기 (기록 시작이 조금 늦게 잡힌 상황 흉내)
                s_ = np.random.randint(0, args.shift + 1)
                if s_:
                    xb = torch.cat([torch.zeros_like(xb[..., :s_]), xb[..., :-s_]], dim=-1)
            xb = as_batch(xb).to(dev, non_blocking=True)
            yb = yt[idx].to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, enabled=dev.type == 'cuda'):
                loss = lossf(model(xb), yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(idx)
        if sched:
            sched.step()
        msg = f'  {tag} epoch {ep:2d}  loss {tot / len(Xt):.4f}'
        if Xva is not None and (ep % args.eval_every == 0 or ep == args.epochs):
            acc = (predict(model, Xva, dev).argmax(1) == yva).mean()
            curve.append((ep, float(acc)))
            msg += f'  val {acc:.4f}'
        print(msg + f'  ({time.time() - t0:.0f}s)', flush=True)
    return model, curve


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--npz', default=os.path.join(ROOT, 'sequences', 'seq_L5000_skip0.npz'))
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=0.002)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--eval-every', type=int, default=2)
    ap.add_argument('--part-dir', help='fold별 결과를 저장할 폴더 (Colab이면 Drive 폴더). 이미 있는 fold는 건너뛴다')
    ap.add_argument('--out', help='OOF를 저장할 경로 (기본: results/cw_oof/df_base.npz)')
    ap.add_argument('--final', action='store_true', help='전체 train 학습 → test 확률 저장 (월요일 1회)')
    ap.add_argument('--tag', default='base', help='결과 이름: OOF results/cw_oof/df_<tag>.npz, test df_<tag>_test.npz')
    ap.add_argument('--input', default='dir', choices=['dir', 'tiktok', 'dir+iat', 'tam'],
                    help='dir: 방향(DF 논문) / tiktok: 방향×시각(Tik-Tok 논문) / dir+iat: 방향과 log 시간 간격 2채널. '
                         '/ tam: 시간 구간별 송신·수신 셀 수 2채널 (Robust Fingerprinting, USENIX Sec 2023). '
                         'tiktok·dir+iat·tam은 sequences.py --time으로 만든 npz가 필요 (tam은 --length 10000 --time 권장: 잘림 0%)')
    ap.add_argument('--tam-sec', type=float, default=80, help='tam: 앞에서 몇 초까지 볼지 (논문 80초)')
    ap.add_argument('--tam-bins', type=int, default=1800, help='tam: 구간 수 (논문 1800 → 약 44ms)')
    ap.add_argument('--sched', default='none', choices=['none', 'cosine'], help='학습률 스케줄 (cosine: 점점 줄임)')
    ap.add_argument('--shift', type=int, default=0, help='학습 증강: 시퀀스를 0~N칸 무작위로 뒤로 밀기 (0이면 끔)')
    args = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {dev}' + ('' if dev.type == 'cuda' else '  ※ GPU가 아니면 매우 느리다 (Colab: 런타임 → T4 GPU)'))
    z = np.load(args.npz)
    X, y, url, inst = z['mon_dir'], z['mon_site'], z['mon_url'], z['mon_inst']
    if args.input != 'dir':
        assert 'mon_time' in z, 'sequences.py --time 으로 만든 npz가 필요하다 (예: seq_L5000_skip0.npz를 --time으로 다시 생성)'
        T = z['mon_time']
        if args.input == 'tam':  # Traffic Aggregation Matrix: [송신 수, 수신 수] × 시간 구간
            real = X != 0
            print(f'TAM: {args.tam_sec:.0f}초 / {args.tam_bins}구간 ({args.tam_sec / args.tam_bins * 1000:.0f}ms 단위). '
                  f'{args.tam_sec:.0f}초 넘는 셀 {((T > args.tam_sec) & real).sum() / real.sum() * 100:.2f}%는 마지막 구간에 모은다')
            X = make_tam(X, T, args.tam_sec, args.tam_bins)
        elif args.input == 'tiktok':  # 방향 × 시각(초). Tik-Tok(Rahman et al., PETS 2020) 방식
            X = (X.astype(np.float32) * T).astype(np.float32)
        else:  # 2채널: 방향, log(1 + 직전 셀과의 시간 간격 × 100)  (0.01초 단위라 ×100)
            iat = np.diff(T, prepend=T[:, :1], axis=1).clip(min=0)
            iat[X == 0] = 0
            X = np.stack([X.astype(np.float32), np.log1p(iat * 100).astype(np.float32)], axis=1)
    is_te = ld.mon_is_test(url, inst)
    Xtr, ytr, utr, itr = X[~is_te], y[~is_te], url[~is_te], inst[~is_te]
    fold = itr % 5
    L = X.shape[-1]
    print(f'train {len(ytr)}, test {is_te.sum()}, L={L}, input={args.input}, sched={args.sched}, shift={args.shift}, '
          f'seed={args.seed}, tag={args.tag}')
    params = dict(epochs=args.epochs, bs=args.bs, lr=args.lr, seed=args.seed, L=int(L), input=args.input,
                  sched=args.sched, shift=args.shift)

    if args.final:
        model, _ = train(Xtr, ytr, args, dev, tag='full')
        p = predict(model, X[is_te], dev)
        os.makedirs(os.path.join(ROOT, 'results', 'cw_test'), exist_ok=True)
        out = args.out or os.path.join(ROOT, 'results', 'cw_test', f'df_{args.tag}_test.npz')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(out, proba=p, url=url[is_te], inst=inst[is_te], params=json.dumps(params))
        print(f'test 확률 저장 → {out}  (컴퓨터 레포 results/cw_test/df_{args.tag}_test.npz로 옮긴 뒤 cw_tuning.py test에서 확인)')
        return

    oof = np.zeros((len(ytr), 95), np.float32)
    curves = {}
    # fold마다 결과를 --part-dir에 바로 저장한다. Colab이 끊겨도 끝난 fold는 남고, 다시 돌리면 건너뛴다.
    if args.part_dir:
        os.makedirs(args.part_dir, exist_ok=True)
    for k in args.folds:
        va = fold == k
        part = os.path.join(args.part_dir, (f'df_fold{k}_e{args.epochs}.npz' if args.tag == 'base'
                                            else f'df_{args.tag}_fold{k}_e{args.epochs}.npz')) if args.part_dir else None
        if part and os.path.exists(part):
            p = np.load(part)
            oof[va], curves[k] = p['proba'], [tuple(c) for c in p['curve']]
            print(f'fold {k}: 이미 끝남 → 불러옴 ({part})  {(oof[va].argmax(1) == ytr[va]).mean():.4f}\n', flush=True)
            continue
        model, curve = train(Xtr[~va], ytr[~va], args, dev, Xtr[va], ytr[va], tag=f'fold{k}')
        oof[va] = predict(model, Xtr[va], dev)
        curves[k] = curve
        if part:
            np.savez_compressed(part, proba=oof[va], curve=np.array(curve))
            print(f'  fold {k} 저장 → {part}', flush=True)
        print(f'fold {k}: {(oof[va].argmax(1) == ytr[va]).mean():.4f}\n', flush=True)
    accs = [(oof[fold == k].argmax(1) == ytr[fold == k]).mean() for k in args.folds]
    print(f'CV ({len(args.folds)} fold): {np.mean(accs):.4f} ± {np.std(accs):.4f}   [{" ".join(f"{a:.4f}" for a in accs)}]')
    print('epoch별 검증 정확도 (fold 평균):')
    for i, (ep, _) in enumerate(next(iter(curves.values()))):
        print(f'  epoch {ep:2d}: {np.mean([c[i][1] for c in curves.values()]):.4f}')
    if sorted(args.folds) == [0, 1, 2, 3, 4]:
        out = args.out or os.path.join(ROOT, 'results', 'cw_oof', f'df_{args.tag}.npz')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(out, url=utr, inst=itr, y=ytr, pred=oof.argmax(1), proba=oof,
                            params=json.dumps(params), fs=f'seq_L{L}')
        print(f'OOF 저장 → {out}  (컴퓨터 레포의 results/cw_oof/df_{args.tag}.npz로 옮기면 cw_tuning.py vote에 들어간다)')
    else:
        print('fold 일부만 돌려 OOF는 저장하지 않았다.')


if __name__ == '__main__':
    main()
