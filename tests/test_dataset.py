"""Sanity checks for the signal generator and model. Run with: pytest"""

import numpy as np
import torch

from iqmodrec import CLASSES, IQCNN, cumulant_features, generate
from iqmodrec.dataset import CONSTELLATIONS, channel, modulate, rrc_taps


def test_shapes_and_labels():
    X, y, snr = generate(n_per_class_per_snr=2, snrs=[0, 10])
    assert X.shape == (len(CLASSES) * 2 * 2, 2, 128)
    assert X.dtype == np.float32
    assert set(y) == set(range(len(CLASSES)))
    assert set(snr) == {0, 10}


def test_unit_power_after_agc():
    X, _, _ = generate(n_per_class_per_snr=3, snrs=[5])
    power = (X[:, 0] ** 2 + X[:, 1] ** 2).mean(axis=1)
    assert np.allclose(power, 1.0, atol=1e-4)


def test_constellations_have_unit_power():
    for name, c in CONSTELLATIONS.items():
        if name != "BPSK":
            assert np.isclose(np.mean(np.abs(c) ** 2), 1.0), name


def test_rrc_unit_energy():
    taps = rrc_taps(beta=0.35, sps=8)
    assert np.isclose(np.sum(taps**2), 1.0)


def test_fsk_is_constant_envelope():
    rng = np.random.default_rng(0)
    for name in ["GFSK", "CPFSK"]:
        x = modulate(name, 256, 8, rng)
        assert np.allclose(np.abs(x), 1.0), name


def test_measured_snr_matches_target():
    rng = np.random.default_rng(1)
    clean = np.exp(1j * rng.uniform(0, 2 * np.pi, 200_000))  # unit power
    noisy = channel(clean, snr_db=10, rng=rng, max_cfo=0.0)
    # with max_cfo=0 the only rotation is a constant phase; estimate and remove it
    phi = np.angle(np.vdot(clean, noisy))
    noise = noisy - clean * np.exp(1j * phi)
    snr_est = 10 * np.log10(1.0 / np.mean(np.abs(noise) ** 2))
    assert abs(snr_est - 10) < 0.2


def test_cumulant_features_finite():
    X, _, _ = generate(n_per_class_per_snr=2, snrs=[0])
    F = cumulant_features(X)
    assert F.shape[0] == len(X) and np.isfinite(F).all()


def test_model_forward_any_length():
    model = IQCNN(len(CLASSES)).eval()
    for L in (128, 1024):
        out = model(torch.randn(4, 2, L))
        assert out.shape == (4, len(CLASSES))


# ---------------------------------------------------------------- impairments
from iqmodrec.dataset import AUGMENT, TARGET, iq_imbalance, multipath, phase_noise
from iqmodrec.training import adapt_batchnorm


def test_multipath_preserves_average_power():
    rng = np.random.default_rng(2)
    x = np.exp(1j * rng.uniform(0, 2 * np.pi, 50_000))
    powers = [np.mean(np.abs(multipath(x, rng)) ** 2) for _ in range(200)]
    assert abs(np.mean(powers) - 1.0) < 0.02  # unit-energy impulse response


def test_phase_noise_keeps_envelope():
    rng = np.random.default_rng(3)
    x = np.exp(1j * rng.uniform(0, 2 * np.pi, 1000))
    assert np.allclose(np.abs(phase_noise(x, rng)), 1.0)


def test_iq_imbalance_identity_and_image():
    rng = np.random.default_rng(4)
    n = np.arange(4096)
    tone = np.exp(2j * np.pi * 0.1 * n)
    assert np.allclose(iq_imbalance(tone, rng, amp_db=(0, 0), phase_deg=(0, 0)), tone)
    # imbalance creates an image at -f
    spec = np.abs(np.fft.fft(iq_imbalance(tone, rng)))
    k = int(0.1 * len(n))
    assert spec[-k] > 1e-3 * spec[k]


def test_impaired_dataset_and_awgn_reproducible():
    X1, _, _ = generate(n_per_class_per_snr=2, snrs=[10], impairments=TARGET)
    X2, _, _ = generate(n_per_class_per_snr=2, snrs=[10], impairments=AUGMENT)
    assert X1.shape == X2.shape and np.isfinite(X1).all() and np.isfinite(X2).all()
    A, _, _ = generate(n_per_class_per_snr=2, snrs=[10], seed=5)
    B, _, _ = generate(n_per_class_per_snr=2, snrs=[10], seed=5)
    assert np.array_equal(A, B)


def test_adabn_changes_only_bn_statistics():
    model = IQCNN(len(CLASSES)).eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    adapt_batchnorm(model, np.random.randn(64, 2, 128).astype(np.float32))
    after = model.state_dict()
    for k in before:
        same = torch.equal(before[k], after[k])
        assert same == ("running" not in k and "num_batches" not in k), k
