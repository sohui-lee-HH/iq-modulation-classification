"""
Classical DSP baseline: higher-order cumulants + gradient boosting.

Higher-order statistics (Swami & Sadler, 2000) are the textbook
feature-based approach to modulation classification. They are invariant
to phase rotation (when using |C|) and separate PSK/QAM/FSK families well
at high SNR. Comparing against this baseline answers the interview question
"why use deep learning at all?"
"""

import numpy as np


def cumulant_features(X: np.ndarray) -> np.ndarray:
    """X: (N, 2, L) -> features (N, F)."""
    x = X[:, 0, :] + 1j * X[:, 1, :]
    x = x / np.sqrt(np.mean(np.abs(x) ** 2, axis=1, keepdims=True))
    xc = np.conj(x)

    M20 = np.mean(x * x, axis=1)
    M21 = np.mean(x * xc, axis=1)
    M40 = np.mean(x**4, axis=1)
    M41 = np.mean(x**3 * xc, axis=1)
    M42 = np.mean(x**2 * xc**2, axis=1)
    M60 = np.mean(x**6, axis=1)
    M63 = np.mean(x**3 * xc**3, axis=1)

    C40 = M40 - 3 * M20**2
    C41 = M41 - 3 * M20 * M21
    C42 = M42 - np.abs(M20) ** 2 - 2 * M21**2
    C60 = M60 - 15 * M20 * M40 + 30 * M20**3
    C63 = M63 - 9 * M42 * M21 + 12 * M21**3

    amp = np.abs(x)
    inst_phase = np.unwrap(np.angle(x), axis=1)
    inst_freq = np.diff(inst_phase, axis=1)

    feats = [
        np.abs(M20), np.abs(C40), np.abs(C41), np.abs(C42),
        np.abs(C60), np.abs(C63),
        amp.std(axis=1) / amp.mean(axis=1),          # envelope variation (constant envelope -> 0)
        np.abs(np.mean(np.abs(x) ** 4, axis=1)),     # kurtosis-like
        inst_freq.std(axis=1),                       # FSK vs linear
        np.abs(inst_freq - inst_freq.mean(axis=1, keepdims=True)).mean(axis=1),
    ]
    return np.stack([np.real(f) for f in feats], axis=1).astype(np.float32)
