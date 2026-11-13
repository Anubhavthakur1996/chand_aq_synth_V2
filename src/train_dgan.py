"""
GAN training + generation (Gretel timeseries DGAN, PyTorch - same implementation as the base paper)
and post-processing of the generated data.
"""
import time
import numpy as np
import config as C
from .seeding import set_seed


def fit_and_generate(train_windows, epochs, n_sequences, seed):
    """Train a DGAN on `train_windows` (n, SEQ_LEN, F) and return n_sequences synthetic sequences.
    Full-batch training, exactly like the base paper's code (batch_size = number of windows)."""
    from gretel_synthetics.timeseries_dgan.dgan import DGAN
    from gretel_synthetics.timeseries_dgan.config import DGANConfig, OutputType

    set_seed(seed)
    n, L, F = train_windows.shape
    cfg = DGANConfig(
        max_sequence_len=L, sample_len=L,
        batch_size=n,
        apply_feature_scaling=True, apply_example_scaling=False,
        use_attribute_discriminator=False,
        generator_learning_rate=C.LEARNING_RATE, discriminator_learning_rate=C.LEARNING_RATE,
        epochs=int(epochs),
    )
    model = DGAN(cfg)
    model.train_numpy(features=train_windows.astype(np.float32),
                      feature_types=[OutputType.CONTINUOUS] * F)
    _, synth = model.generate_numpy(int(n_sequences))
    return np.asarray(synth, dtype=np.float64)


def clock_diagnostics(synth, cols):
    """How valid is the generated clock?  (hour_sin, hour_cos) should lie on the unit circle and
    advance ~1 h per row.  Reported for the paper; computed BEFORE projection."""
    out = {}
    for name, period in (("hour", 24.0), ("dow", 7.0)):
        if f"{name}_sin" not in cols:
            continue
        s = synth[:, :, cols.index(f"{name}_sin")]
        c = synth[:, :, cols.index(f"{name}_cos")]
        r2 = s ** 2 + c ** 2
        d = {"radius_sq_mean": float(r2.mean()), "radius_sq_min": float(r2.min()), "radius_sq_max": float(r2.max())}
        if name == "hour":
            ang = (np.arctan2(s, c) % (2 * np.pi)) * period / (2 * np.pi)
            step = np.diff(ang, axis=1)
            step = (step + period / 2) % period - period / 2
            d["share_steps_0.5_to_1.5h"] = float(np.mean((step > 0.5) & (step < 1.5)))
        out[name] = d
    return out


def postprocess(synth, cols):
    """Physical clipping + projection of generated (sin, cos) pairs back onto the unit circle."""
    x = synth.copy()
    for name in C.NON_NEGATIVE:
        if name in cols:
            i = cols.index(name); x[:, :, i] = np.clip(x[:, :, i], 0.0, None)
    for name, hi in C.UPPER_BOUNDS.items():
        if name in cols:
            i = cols.index(name); x[:, :, i] = np.clip(x[:, :, i], 0.0, hi)
    for name in ("hour", "dow"):
        if f"{name}_sin" in cols:
            si, ci = cols.index(f"{name}_sin"), cols.index(f"{name}_cos")
            norm = np.sqrt(x[:, :, si] ** 2 + x[:, :, ci] ** 2)
            ok = norm > 1e-6
            x[:, :, si] = np.where(ok, x[:, :, si] / np.where(ok, norm, 1.0), 0.0)
            x[:, :, ci] = np.where(ok, x[:, :, ci] / np.where(ok, norm, 1.0), 1.0)
    return x


def generate_synthetic(train_windows, cols, epochs, n_sequences, seed):
    t0 = time.time()
    raw = fit_and_generate(train_windows, epochs, n_sequences, seed)
    seconds = time.time() - t0
    diag = clock_diagnostics(raw, cols)
    return postprocess(raw, cols), diag, seconds
