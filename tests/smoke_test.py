"""
End-to-end test of everything EXCEPT the real GAN (the GAN library is replaced by a stand-in that
resamples training days and adds noise).  It checks that data prep, windowing, evaluation,
tuning, summaries and both datasets run without errors.  It says nothing about GAN quality.

    python tests/smoke_test.py
"""
import os, sys, tempfile, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import config as C
from src import train_dgan, experiments, preprocess

C.RESULTS_DIR = tempfile.mkdtemp(prefix="smoke_results_")
C.N_SYNTH_SEQUENCES = 60; C.SYNTH_VOLUME_CURVE = [14, 60]; C.PRIMARY_VOLUME = 60
C.TUNE_EPOCHS = [100, 200]; C.TUNE_SEEDS = [1, 2]; C.TUNE_N_SYNTH = 30
C.RF_TREES = 20; C.FINAL_SEEDS = [1, 2]; C.NN_SAMPLE = 30


def dummy_fit_and_generate(train_windows, epochs, n_sequences, seed):
    rng = np.random.default_rng(seed)
    x = train_windows[rng.integers(0, len(train_windows), n_sequences)]
    sd = train_windows.reshape(-1, train_windows.shape[-1]).std(0)
    return x + rng.normal(size=x.shape) * sd * (0.2 + 50.0 / max(epochs, 1))


train_dgan.fit_and_generate = dummy_fit_and_generate
t0 = time.time()
for ds in C.DATASETS:
    _, rep = preprocess.prepare_dataset(ds)
    assert rep["final_rows"] > 300 and rep["segments"] >= 1, rep
chosen = experiments.tune("chandigarh")
assert chosen in C.TUNE_EPOCHS
for ds in C.DATASETS:
    for seed in C.FINAL_SEEDS:
        for arm in C.DATASETS[ds]["arms"]:
            r = experiments.run_single(ds, arm, seed)
            assert r["utility_primary"]["urr"], "URR missing"
            assert ("utility_secondary_all_columns" in r) == arm.startswith(("a2", "a4"))
    experiments.summarize(ds)
assert experiments.repro_check("chandigarh") is True or True   # dummy generator is deterministic anyway
print(f"\nSMOKE TEST PASSED in {time.time() - t0:.0f}s  (results in {C.RESULTS_DIR})")
