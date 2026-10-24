"""
Experiment 2 -- Domain shift: from a clean simulation to a realistic channel.

A model trained on simulated AWGN data is deployed on a channel it never saw:
multipath fading + oscillator phase noise + receiver IQ imbalance
(profile TARGET in iqmodrec/dataset.py). How much accuracy is lost, and which
remedy recovers it?

  A  Source only          train on AWGN                     -> test on target
  B  + AdaBN              A, BatchNorm statistics re-estimated on UNLABELLED
                          target data (unsupervised domain adaptation)
  C  + Fine-tuning        A, fine-tuned on a small LABELLED target set
                          (10 frames per class per SNR, 4 % of the source data)
  D  Impairment augment.  train on synthetic data with randomised impairments
                          (profile AUGMENT: wider, different parameter ranges)
  O  Oracle               train on target data (upper bound, not available
                          in practice)

    python experiments/domain_shift.py            # ~25 min on a 2-core CPU
    python experiments/domain_shift.py --quick    # ~2 min smoke test
"""

import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from iqmodrec import CLASSES, IQCNN, generate
from iqmodrec.dataset import AUGMENT, TARGET
from iqmodrec.training import accuracy_by_snr, adapt_batchnorm, predict, train_model


def split(X, y, snr, frac=0.1):
    tr, va = train_test_split(np.arange(len(X)), test_size=frac, stratify=y * 100 + snr, random_state=0)
    return X[tr], y[tr], X[va], y[va]


def fresh_model(device):
    torch.manual_seed(0)
    return IQCNN(len(CLASSES)).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--per-snr", type=int, default=250)
    ap.add_argument("--test-per-snr", type=int, default=100)
    ap.add_argument("--ft-per-snr", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--out", default="results/domain_shift")
    args = ap.parse_args()
    if args.quick:
        args.per_snr, args.test_per_snr, args.epochs = 40, 20, 3

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = args.per_snr

    # ---- data (different seeds = independent draws)
    src = generate(n, seed=10)                              # AWGN
    aug = generate(n, seed=11, impairments=AUGMENT)         # randomised impairments
    tgt = generate(n, seed=12, impairments=TARGET)          # target, for oracle / AdaBN (labels unused)
    ft = generate(args.ft_per_snr, seed=13, impairments=TARGET)  # small labelled target set
    src_te = generate(args.test_per_snr, seed=20)
    tgt_te = generate(args.test_per_snr, seed=21, impairments=TARGET)
    print(f"source {len(src[0])}, augmented {len(aug[0])}, target {len(tgt[0])}, "
          f"fine-tune {len(ft[0])}, test {len(tgt_te[0])} per domain", flush=True)

    models = {}
    # A: source only
    m = fresh_model(device)
    models["A  Source only (AWGN)"], _ = train_model(m, *split(*src), epochs=args.epochs,
                                                      device=device, tag="[A] ")
    # B: AdaBN on unlabelled target data
    models["B  + AdaBN (unlabelled target)"] = adapt_batchnorm(
        copy.deepcopy(models["A  Source only (AWGN)"]), tgt[0], device)
    # C: fine-tune on small labelled target set
    m = copy.deepcopy(models["A  Source only (AWGN)"])
    models["C  + Fine-tuning (4 % labelled target)"], _ = train_model(
        m, ft[0], ft[1], epochs=10, lr=5e-4, device=device, tag="[C] ")
    # D: impairment augmentation
    m = fresh_model(device)
    models["D  Impairment augmentation"], _ = train_model(m, *split(*aug), epochs=args.epochs,
                                                           device=device, tag="[D] ")
    # O: oracle
    m = fresh_model(device)
    models["O  Oracle (trained on target)"], _ = train_model(m, *split(*tgt), epochs=args.epochs,
                                                              device=device, tag="[O] ")

    # ---- evaluation
    summary = {"config": vars(args), "results": {}}
    curves = {}
    for name, model in models.items():
        r = {}
        for dom, (Xt, yt, st) in {"source": src_te, "target": tgt_te}.items():
            p = predict(model, Xt, device)
            hi = st >= 10
            r[f"{dom}_overall"] = float((p == yt).mean())
            r[f"{dom}_snr>=10dB"] = float((p[hi] == yt[hi]).mean())
            if dom == "target":
                curves[name] = accuracy_by_snr(p, yt, st)
        summary["results"][name] = r
        print(f"{name:42s} " + "  ".join(f"{k} {v:.3f}" for k, v in r.items()), flush=True)

    p = predict(models["A  Source only (AWGN)"], src_te[0], device)
    curves_ref = accuracy_by_snr(p, src_te[1], src_te[2])
    with open(f"{args.out}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- figure
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(*curves_ref, color="grey", ls="--", lw=1.2, label="Reference: AWGN model on AWGN test")
    styles = ["o-", "^-", "v-", "s-", "D:"]
    for (name, (lv, acc)), st in zip(curves.items(), styles):
        ax.plot(lv, acc, st, ms=4, label=name.split("  ", 1)[1] + " → target")
    ax.axhline(1 / len(CLASSES), color="grey", lw=0.8, ls=":")
    ax.set(xlabel="SNR (dB)", ylabel="Accuracy on target channel", ylim=(0, 1.02),
           title="Domain shift: AWGN-trained model on multipath + phase noise + IQ imbalance")
    ax.title.set_fontsize(10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{args.out}/domain_shift.png", dpi=150)
    print(f"saved to {args.out}/")


if __name__ == "__main__":
    main()
