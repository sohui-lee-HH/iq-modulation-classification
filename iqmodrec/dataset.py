"""
Synthetic IQ dataset for automatic modulation classification (AMC).

Every sample goes through a small but realistic transmit/channel chain:

    random bits -> symbol mapping -> pulse shaping (RRC or Gaussian FSK)
    -> carrier frequency offset + random phase + timing offset
    -> AWGN at a chosen SNR -> unit-power normalisation

Output per sample: a (2, N) float32 array [I; Q], a class label and its SNR.
The format matches RadioML 2016.10a (2 x 128), so the same model can be
trained on that dataset later without changes.
"""

import numpy as np

CLASSES = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "PAM4", "GFSK", "CPFSK"]


# ---------------------------------------------------------------- pulse shapes
def rrc_taps(beta: float, sps: int, span: int = 8) -> np.ndarray:
    """Root-raised-cosine filter taps (unit energy)."""
    t = np.arange(-span * sps / 2, span * sps / 2 + 1) / sps
    taps = np.zeros_like(t)
    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            taps[i] = 1.0 + beta * (4 / np.pi - 1)
        elif beta > 0 and np.isclose(abs(ti), 1 / (4 * beta)):
            taps[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            taps[i] = num / den
    return taps / np.sqrt(np.sum(taps**2))


def gaussian_taps(bt: float, sps: int, span: int = 4) -> np.ndarray:
    """Gaussian frequency pulse for GFSK/GMSK (sums to 1)."""
    t = np.arange(-span * sps / 2, span * sps / 2 + 1) / sps
    sigma = np.sqrt(np.log(2)) / (2 * np.pi * bt)
    h = np.exp(-(t**2) / (2 * sigma**2))
    return h / h.sum()


# ---------------------------------------------------------------- constellations
def _psk(m):
    return np.exp(1j * (2 * np.pi * np.arange(m) / m + (np.pi / m if m == 4 else 0)))


def _qam(m):
    k = int(np.sqrt(m))
    levels = np.arange(-(k - 1), k, 2)
    pts = (levels[:, None] + 1j * levels[None, :]).ravel()
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


CONSTELLATIONS = {
    "BPSK": np.array([1, -1], dtype=complex),
    "QPSK": _psk(4),
    "8PSK": _psk(8),
    "16QAM": _qam(16),
    "64QAM": _qam(64),
    "PAM4": np.array([-3, -1, 1, 3], dtype=complex) / np.sqrt(5),
}


# ---------------------------------------------------------------- modulators
def modulate(name: str, n_samples: int, sps: int, rng: np.random.Generator) -> np.ndarray:
    """Return a clean, oversampled complex baseband waveform of length >= n_samples."""
    n_sym = n_samples // sps + 32  # extra symbols so filter transients can be cut off

    if name in CONSTELLATIONS:
        const = CONSTELLATIONS[name]
        symbols = const[rng.integers(0, len(const), n_sym)]
        up = np.zeros(n_sym * sps, dtype=complex)
        up[::sps] = symbols
        beta = rng.uniform(0.2, 0.5)  # roll-off varies per sample
        return np.convolve(up, rrc_taps(beta, sps), mode="same")

    # Continuous-phase FSK family: integrate a frequency pulse into phase
    bits = rng.integers(0, 2, n_sym) * 2 - 1
    freq = np.repeat(bits.astype(float), sps)
    if name == "GFSK":
        freq = np.convolve(freq, gaussian_taps(bt=0.3, sps=sps), mode="same")
        h = rng.uniform(0.45, 0.55)  # modulation index (~GMSK / BLE-like)
    elif name == "CPFSK":
        h = rng.uniform(0.9, 1.1)  # wider deviation, rectangular pulse
    else:
        raise ValueError(name)
    phase = np.pi * h * np.cumsum(freq) / sps
    return np.exp(1j * phase)


# ---------------------------------------------------------------- impairments
def multipath(x: np.ndarray, rng: np.random.Generator, delay_spread=(1.0, 4.0),
              k_factor=(0.0, 5.0), max_delay: int = 12) -> np.ndarray:
    """Tapped-delay-line fading channel, one realisation per frame.

    Taps at 1-sample spacing with an exponential power-delay profile
    (rms delay spread tau in samples). The first tap is Rician with factor K
    (line of sight + diffuse part); all others are Rayleigh. The impulse
    response is normalised to unit energy so that the average received
    power, and therefore the SNR definition, is unchanged.
    """
    tau = rng.uniform(*delay_spread)
    K = rng.uniform(*k_factor)
    d = np.arange(max_delay + 1)
    pdp = np.exp(-d / tau)
    diffuse = (rng.standard_normal(len(d)) + 1j * rng.standard_normal(len(d))) / np.sqrt(2)  # CN(0, 1)
    los = np.exp(1j * rng.uniform(0, 2 * np.pi))
    h = np.sqrt(pdp) * diffuse
    h[0] = np.sqrt(pdp[0]) * (np.sqrt(K / (K + 1)) * los + np.sqrt(1 / (K + 1)) * diffuse[0])
    h = h / np.linalg.norm(h)
    return np.convolve(x, h)[:len(x)]


def phase_noise(x: np.ndarray, rng: np.random.Generator, sigma=(0.01, 0.03)) -> np.ndarray:
    """Oscillator phase noise as a Wiener process (random walk), sigma in rad/sample."""
    s = rng.uniform(*sigma)
    return x * np.exp(1j * np.cumsum(s * rng.standard_normal(len(x))))


def iq_imbalance(x: np.ndarray, rng: np.random.Generator, amp_db=(0.5, 1.5),
                 phase_deg=(3.0, 10.0)) -> np.ndarray:
    """Receiver IQ imbalance: Q branch has gain g and phase skew phi relative to I.

    Equivalent to y = mu*x + nu*conj(x): creates an image of the signal.
    g = 1, phi = 0 is the identity.
    """
    g = 10 ** (rng.choice([-1, 1]) * rng.uniform(*amp_db) / 20)
    phi = np.deg2rad(rng.choice([-1, 1]) * rng.uniform(*phase_deg))
    return x.real + 1j * g * (x.imag * np.cos(phi) + x.real * np.sin(phi))


# Two impairment profiles used in experiments/domain_shift.py.
# TARGET: the "deployment" channel -- every impairment always present.
# AUGMENT: what we can model at training time -- each impairment applied with
# probability p and with deliberately wider, different parameter ranges.
TARGET = {
    "p": 1.0,
    "multipath": {"delay_spread": (1.0, 4.0), "k_factor": (0.0, 5.0), "max_delay": 12},
    "phase_noise": {"sigma": (0.01, 0.03)},
    "iq": {"amp_db": (0.5, 1.5), "phase_deg": (3.0, 10.0)},
}
AUGMENT = {
    "p": 0.7,
    "multipath": {"delay_spread": (0.5, 6.0), "k_factor": (0.0, 10.0), "max_delay": 16},
    "phase_noise": {"sigma": (0.0, 0.05)},
    "iq": {"amp_db": (0.0, 2.0), "phase_deg": (0.0, 15.0)},
}


# ---------------------------------------------------------------- channel
def channel(x: np.ndarray, snr_db: float, rng: np.random.Generator,
            max_cfo: float = 0.01, impairments: dict | None = None) -> np.ndarray:
    """Optional multipath -> CFO + random phase (+ phase noise) -> AWGN (-> IQ imbalance).

    max_cfo is in cycles/sample (0.01 = 1 % of the sample rate).
    SNR is defined as average signal power / noise power over the full sampled band.
    impairments: None (AWGN only) or a profile such as TARGET / AUGMENT.
    """
    imp = impairments or {}
    p = imp.get("p", 1.0)

    def use(key):
        return key in imp and rng.random() < p

    if use("multipath"):
        x = multipath(x, rng, **imp["multipath"])

    n = np.arange(len(x))
    cfo = rng.uniform(-max_cfo, max_cfo)
    phi = rng.uniform(0, 2 * np.pi)
    x = x * np.exp(1j * (2 * np.pi * cfo * n + phi))
    if use("phase_noise"):
        x = phase_noise(x, rng, **imp["phase_noise"])

    p_sig = np.mean(np.abs(x) ** 2)
    p_noise = p_sig / (10 ** (snr_db / 10))
    noise = np.sqrt(p_noise / 2) * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
    y = x + noise

    if use("iq"):  # receiver front end, so it acts on signal + noise
        y = iq_imbalance(y, rng, **imp["iq"])
    return y


# ---------------------------------------------------------------- dataset
def generate(n_per_class_per_snr: int = 400,
             snrs=range(-10, 21, 2),
             n_samples: int = 128,
             sps: int = 8,
             seed: int = 0,
             impairments: dict | None = None):
    """Return X (N, 2, n_samples) float32, y (N,) int64, snr (N,) int."""
    rng = np.random.default_rng(seed)
    X, y, s = [], [], []
    for label, name in enumerate(CLASSES):
        for snr in snrs:
            for _ in range(n_per_class_per_snr):
                clean = modulate(name, n_samples, sps, rng)
                noisy = channel(clean, snr, rng, impairments=impairments)
                # random timing offset: start anywhere after the filter transient
                start = rng.integers(8 * sps, len(noisy) - n_samples - 8 * sps)
                seg = noisy[start:start + n_samples]
                seg = seg / np.sqrt(np.mean(np.abs(seg) ** 2))  # unit power (AGC)
                X.append(np.stack([seg.real, seg.imag]).astype(np.float32))
                y.append(label)
                s.append(snr)
    return np.stack(X), np.array(y, dtype=np.int64), np.array(s)


if __name__ == "__main__":
    X, y, s = generate(n_per_class_per_snr=10)
    print(X.shape, y.shape, s.shape, "classes:", CLASSES)
