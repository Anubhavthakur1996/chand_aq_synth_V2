"""Checks that every number added in the camera-ready revision is reproduced from the data and the
saved runs.  Run after scripts/paper/*.py:   python tests/test_paper_revision.py"""
import os, json, subprocess, sys
import pandas as pd
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "chandigarh", "revision")
for s in ["paper_checks.py", "rescore_independent_blocks.py"]:
    subprocess.run([sys.executable, "-I", os.path.join(REPO, "scripts", "paper", s)], check=True, capture_output=True)

c = json.load(open(os.path.join(OUT, "paper_checks.json")))
assert c["usable_hours"] == 11219
assert round(c["interp_share_all_pct"], 1) == 4.5 and round(c["interp_hours_any_pct"], 1) == 11.3      # Sec. III-A
assert round(c["interp_share_per_var_pct"]["o3"], 1) == 7.2 and round(c["interp_share_per_var_pct"]["co"], 1) == 5.7
assert round(c["interp_train_share_pct"], 1) == 1.7 and round(c["interp_train_hours_any_pct"], 1) == 7.7
acf = c["acf_observed_only_vs_all"]
assert max(v["mean_abs_change"] for v in acf.values()) <= 0.0095 and round(acf["no2"]["mean_abs_change"], 3) == 0.009
assert round(acf["so2"]["max_abs_change"], 3) == 0.023
assert c["co_source_scale"] == {"mean": 1.22, "std": 0.49, "min": 0.0, "max": 5.82}                     # Table I
ck = c["clock_diagnostics"]
assert ck["chandigarh/a2_time_long"]["hour_share_steps_0.5_1.5_pct"] == [91.5, 1.0]
assert ck["chandigarh/a4_time_short"]["hour_share_steps_0.5_1.5_pct"] == [67.4, 0.9]
assert ck["bilbao/a4_time_short"]["hour_share_steps_0.5_1.5_pct"] == [93.7, 1.5]

t = pd.read_csv(os.path.join(OUT, "rescore_independent_blocks.csv"))
assert (t["emd_val"] - t["emd_val_orig"]).abs().max() < 1e-6, "re-scoring must reproduce the saved validation EMD"
assert (t["nn_val"] - t["nn_orig"]).abs().max() < 1e-6, "re-scoring must reproduce the saved memorisation ratio"
m = t.groupby("arm")["emd_testall"].mean().round(4)
assert m.to_dict() == {"a1_base_long": 0.5059, "a2_time_long": 0.5028, "a3_base_short": 0.5109, "a4_time_short": 0.5328}  # Table II
p = t.pivot(index="seed", columns="arm")["emd_testall"]
d = p["a4_time_short"] - p["a3_base_short"]
assert (d > 0).all() and round(d.mean(), 3) == 0.022
print("all revision numbers reproduced")
