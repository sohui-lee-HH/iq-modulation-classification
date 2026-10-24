"""
Train a 1D CNN for modulation classification on raw IQ samples, compare it
to a cumulant-based baseline, and save the standard AMC plots.

    python train.py                      # full run (~15 min on a laptop CPU)
    python train.py --quick              # smoke test (~1 min)
    python train.py --length 1024        # longer frames, RadioML 2018 style
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from iqmodrec import CLASSES, IQCNN, cumulant_features, generate
from iqmodrec.training import accuracy_by_snr, predict, train_model


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--per-snr", type=int, default=500, help="samples per class per SNR")
    ap.add_argument("--length", type=int, default=128, help="IQ samples per example (128 = RadioML 2016, 1024 = RadioML 2018)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    if args.quick:
        args.epochs, args.per_snr = 5, 120

    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- data
    t0 = time.time()
    X, y, snr = generate(n_per_class_per_snr=args.per_snr, n_samples=args.length)
    print(f"generated {len(X)} samples in {time.time() - t0:.1f}s, shape {X.shape}")
    idx = np.arange(len(X))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y * 100 + snr, random_state=0)
    tr, va = train_test_split(tr, test_size=0.1, stratify=(y * 100 + snr)[tr], random_state=0)

    # ---- CNN training
    model = IQCNN(len(CLASSES)).to(device)
    print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")
    model, history = train_model(model, X[tr], y[tr], X[va], y[va], epochs=args.epochs,
                                 device=device, ckpt=f"{args.out}/iqcnn.pt")
    pred_cnn = predict(model, X[te], device)

    # ---- cumulant baseline
    F = cumulant_features(X)
    gb = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(F[tr], y[tr])
    pred_cum = gb.predict(F[te])

    # ---- evaluation
    levels, acc_cnn = accuracy_by_snr(pred_cnn, y[te], snr[te])
    _, acc_cum = accuracy_by_snr(pred_cum, y[te], snr[te])
    hi = snr[te] >= 10
    summary = {
        "overall_acc_cnn": float((pred_cnn == y[te]).mean()),
        "overall_acc_cumulants": float((pred_cum == y[te]).mean()),
        "acc_cnn_snr>=10dB": float((pred_cnn[hi] == y[te][hi]).mean()),
        "acc_cumulants_snr>=10dB": float((pred_cum[hi] == y[te][hi]).mean()),
        "acc_by_snr": {int(s): {"cnn": float(a), "cumulants": float(b)}
                       for s, a, b in zip(levels, acc_cnn, acc_cum)},
        "history": history,
    }
    with open(f"{args.out}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    for k in list(summary)[:4]:
        print(f"{k:28s} {summary[k]:.3f}")

    # plot 1: accuracy vs SNR
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(levels, acc_cnn, "o-", label="1D CNN on raw IQ")
    ax.plot(levels, acc_cum, "s--", label="Cumulants + gradient boosting")
    ax.axhline(1 / len(CLASSES), color="grey", lw=0.8, ls=":", label="chance")
    ax.set(xlabel="SNR (dB)", ylabel="Accuracy", ylim=(0, 1.02),
           title="Modulation classification accuracy vs SNR")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{args.out}/accuracy_vs_snr.png", dpi=150)

    # plot 2: confusion matrices (all SNR / high SNR)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mask, title in [(axes[0], np.ones_like(hi), "All SNRs"), (axes[1], hi, "SNR ≥ 10 dB")]:
        cm = confusion_matrix(y[te][mask], pred_cnn[mask], normalize="true")
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45)
        ax.set_yticks(range(len(CLASSES)), CLASSES)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                if cm[i, j] >= 0.01:
                    ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if cm[i, j] > 0.5 else "black")
        ax.set(xlabel="Predicted", ylabel="True", title=f"CNN confusion matrix — {title}")
    fig.tight_layout()
    fig.savefig(f"{args.out}/confusion.png", dpi=150)

    # plot 3: example constellations at 20 dB
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.5))
    for k, ax in enumerate(axes.ravel()):
        i = np.where((y == k) & (snr == 20))[0][0]
        ax.plot(X[i, 0], X[i, 1], ".", ms=3)
        ax.set(title=CLASSES[k], aspect="equal", xticks=[], yticks=[])
    fig.suptitle("Example IQ samples at 20 dB SNR (with CFO, before any sync)")
    fig.tight_layout()
    fig.savefig(f"{args.out}/examples.png", dpi=150)
    print(f"saved plots and model to {args.out}/")


if __name__ == "__main__":
    main()
