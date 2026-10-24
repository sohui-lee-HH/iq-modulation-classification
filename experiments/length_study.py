"""
Experiment 1 -- How much does observation length matter?

Hypothesis from the baseline run: the remaining 16QAM <-> 64QAM confusion is
caused by the short frame (128 samples = 16 symbols cannot visit all 64
constellation points), not by the model.

Method: generate 1024-sample frames once, then train the same CNN and the
cumulant baseline on the first L = 128, 256, 512, 1024 samples of the *same*
frames. Only the observation length changes.

    python experiments/length_study.py            # ~40 min on a 2-core CPU
    python experiments/length_study.py --quick    # ~3 min smoke test
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from iqmodrec import CLASSES, IQCNN, cumulant_features, generate
from iqmodrec.training import accuracy_by_snr, predict, train_model

SPS = 8
QAM = [CLASSES.index("16QAM"), CLASSES.index("64QAM")]


def crop(X, L):
    Xc = X[:, :, :L].copy()
    p = (Xc ** 2).sum(axis=1).mean(axis=1)
    return (Xc / np.sqrt(p)[:, None, None]).astype(np.float32)  # re-apply AGC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--per-snr", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--out", default="results/length_study")
    args = ap.parse_args()
    lengths = [128, 256, 512, 1024]
    if args.quick:
        args.per_snr, args.epochs, lengths = 30, 3, [128, 1024]

    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X, y, snr = generate(n_per_class_per_snr=args.per_snr, n_samples=max(lengths), seed=1)
    strat = y * 100 + snr
    tr, te = train_test_split(np.arange(len(X)), test_size=0.2, stratify=strat, random_state=0)
    tr, va = train_test_split(tr, test_size=0.1, stratify=strat[tr], random_state=0)
    hi = snr[te] >= 10
    qam_hi = hi & np.isin(y[te], QAM)
    print(f"{len(X)} frames of {max(lengths)} samples; train {len(tr)}, test {len(te)}")

    res = {}
    for L in lengths:
        XL = crop(X, L)
        torch.manual_seed(0)
        model = IQCNN(len(CLASSES)).to(device)
        t0 = time.time()
        model, _ = train_model(model, XL[tr], y[tr], XL[va], y[va], epochs=args.epochs,
                               device=device, tag=f"[L={L}] ")
        train_s = time.time() - t0

        # CPU inference latency per frame, batch of 256
        xb = torch.from_numpy(XL[te[:256]])
        model.cpu().eval()
        with torch.no_grad():
            model(xb)
            t0 = time.time()
            for _ in range(5):
                model(xb)
        lat_us = (time.time() - t0) / (5 * 256) * 1e6
        model.to(device)

        p_cnn = predict(model, XL[te], device)
        F = cumulant_features(XL)
        gb = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(F[tr], y[tr])
        p_cum = gb.predict(F[te])

        levels, acc_snr = accuracy_by_snr(p_cnn, y[te], snr[te])
        res[L] = {
            "symbols_per_frame": L // SPS,
            "cnn_overall": float((p_cnn == y[te]).mean()),
            "cnn_snr>=10dB": float((p_cnn[hi] == y[te][hi]).mean()),
            "cnn_16/64QAM_snr>=10dB": float((p_cnn[qam_hi] == y[te][qam_hi]).mean()),
            "cumulants_overall": float((p_cum == y[te]).mean()),
            "cumulants_snr>=10dB": float((p_cum[hi] == y[te][hi]).mean()),
            "cumulants_16/64QAM_snr>=10dB": float((p_cum[qam_hi] == y[te][qam_hi]).mean()),
            "train_time_s": round(train_s, 1),
            "cpu_latency_us_per_frame": round(lat_us, 1),
            "acc_by_snr": {int(s): float(a) for s, a in zip(levels, acc_snr)},
        }
        print(f"L={L}: " + ", ".join(f"{k} {v:.3f}" for k, v in res[L].items()
                                      if isinstance(v, float) and "time" not in k and "latency" not in k))

    with open(f"{args.out}/summary.json", "w") as f:
        json.dump({"per_snr": args.per_snr, "epochs": args.epochs, "results": res}, f, indent=2)

    # ---- figure
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3))
    for L in lengths:
        d = res[L]["acc_by_snr"]
        a1.plot(list(map(int, d)), list(d.values()), "o-", ms=3,
                label=f"{L} samples ({L // SPS} symbols)")
    a1.axhline(1 / len(CLASSES), color="grey", lw=0.8, ls=":")
    a1.set(xlabel="SNR (dB)", ylabel="Accuracy", ylim=(0, 1.02), title="CNN accuracy vs SNR by frame length")
    a1.grid(alpha=0.3)
    a1.legend(fontsize=8)

    syms = [res[L]["symbols_per_frame"] for L in lengths]
    a2.plot(syms, [res[L]["cnn_snr>=10dB"] for L in lengths], "o-", label="CNN, all classes")
    a2.plot(syms, [res[L]["cnn_16/64QAM_snr>=10dB"] for L in lengths], "o-", label="CNN, 16QAM/64QAM only")
    a2.plot(syms, [res[L]["cumulants_snr>=10dB"] for L in lengths], "s--", label="Cumulants, all classes")
    a2.plot(syms, [res[L]["cumulants_16/64QAM_snr>=10dB"] for L in lengths], "s--",
            label="Cumulants, 16QAM/64QAM only")
    a2.set_xscale("log", base=2)
    a2.set_xticks(syms, [str(s) for s in syms])
    a2.set(xlabel="Symbols per frame", ylabel="Accuracy at SNR ≥ 10 dB", ylim=(0, 1.02),
           title="High-SNR accuracy vs observation length")
    a2.grid(alpha=0.3)
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{args.out}/length_study.png", dpi=150)
    print(f"saved to {args.out}/")


if __name__ == "__main__":
    main()
