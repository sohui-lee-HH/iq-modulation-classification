"""Shared training, inference and adaptation helpers."""

import numpy as np
import torch
import torch.nn as nn


def rotate_iq(xb: torch.Tensor) -> torch.Tensor:
    """Random carrier-phase rotation per example (label-preserving augmentation)."""
    th = torch.rand(len(xb), 1, device=xb.device) * 2 * np.pi
    c, s = torch.cos(th), torch.sin(th)
    return torch.stack([c * xb[:, 0] - s * xb[:, 1], s * xb[:, 0] + c * xb[:, 1]], 1)


def predict(model, X, device="cpu", bs=1024):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(model(torch.from_numpy(X[i:i + bs]).to(device)).argmax(1).cpu())
    return torch.cat(out).numpy()


def accuracy_by_snr(pred, y, snr):
    levels = np.unique(snr)
    return levels, np.array([(pred[snr == s] == y[snr == s]).mean() for s in levels])


def train_model(model, X, y, Xva=None, yva=None, epochs=25, lr=2e-3, bs=256,
                device="cpu", ckpt=None, verbose=True, tag=""):
    """AdamW + one-cycle LR, phase-rotation augmentation.

    If a validation set is given, the weights with the best validation
    accuracy are kept (and saved to ckpt if given); otherwise the last epoch.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * int(np.ceil(len(X) / bs)))
    loss_fn = nn.CrossEntropyLoss()
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)

    best_va, best_state, history = -1.0, None, []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        tot = 0.0
        for i in range(0, len(X), bs):
            b = perm[i:i + bs]
            xb, yb = rotate_iq(Xt[b].to(device)), yt[b].to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        rec = {"epoch": ep + 1, "train_loss": tot / len(X)}
        if Xva is not None:
            va = float((predict(model, Xva, device) == yva).mean())
            rec["val_acc"] = va
            if va > best_va:
                best_va = va
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        history.append(rec)
        if verbose:
            msg = f"{tag}epoch {ep + 1:2d}  loss {rec['train_loss']:.3f}"
            print(msg + (f"  val acc {rec['val_acc']:.3f}" if "val_acc" in rec else ""), flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    if ckpt:
        torch.save(model.state_dict(), ckpt)
    return model, history


def adapt_batchnorm(model, X_unlabelled, device="cpu", bs=512):
    """AdaBN (Li et al., 2016): unsupervised domain adaptation.

    Re-estimates every BatchNorm layer's running mean/variance on unlabelled
    target-domain data; all learned weights stay unchanged. Needs no labels
    and only forward passes.
    """
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative average over all batches
    model.train()
    with torch.no_grad():
        for i in range(0, len(X_unlabelled), bs):
            model(torch.from_numpy(X_unlabelled[i:i + bs]).to(device))
    for m in bns:
        m.momentum = 0.1
    model.eval()
    return model
