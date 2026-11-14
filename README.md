# Training Budget and Cyclical Time Encoding in DoppelGANger-Based Synthetic Air-Quality Data: A Multi-Seed Evaluation

## Description
This repository contains the code, training setup, evaluation framework and results for a multi-seed study on generating synthetic air-quality data for Sector 22, Chandigarh, India, with the DoppelGANger architecture.

The study builds on Morales-García et al. (2025), *Exploiting synthetic data generation to enhance pollution prediction* (Applied Soft Computing 175, 113076), and its released code ([github.com/JuMoGar/synthetic_pollution_data_ua_upct_uca](https://github.com/JuMoGar/synthetic_pollution_data_ua_upct_uca)). Like that work, it uses the Gretel `timeseries_dgan` implementation of DoppelGANger (PyTorch). Please cite and respect the licence of the original repository.

The base study trains on about two weeks of real data for 100,000 epochs, from a single run. This study tests two modifications to that setup, each on its own and together, with three random seeds per configuration: cyclical (sine/cosine) hour and day-of-week encodings, and a training budget cut to 7,500 epochs (13 times cheaper). Results are mixed and reported as such. Synthetic data lowered random-forest forecasting error by 11–17% compared with training on the real days alone. Time encoding gave a small but consistent fidelity gain at 100,000 epochs and no downstream benefit. The 7,500-epoch budget kept most random-forest utility but lost fidelity and gradient-boosting utility.

## Features
**The Four-Configuration Design**

To measure each modification on its own and in combination, the study compares four configurations (feature set × training budget):

| Configuration | Code name | Features | Training epochs |
|---|---|---|---|
| Base-100k | `a1_base_long` | Base | 100,000 (base paper) |
| Time-100k | `a2_time_long` | Base + cyclical hour/day-of-week | 100,000 |
| Base-7.5k | `a3_base_short` | Base | 7,500 (selected on validation data) |
| Time-7.5k | `a4_time_short` | Base + cyclical hour/day-of-week | 7,500 |

1. **Base features (the base paper's variables)**
   - **Features:** Pollutants (PM2.5, PM10, O3, CO, SO2, NO, NO2) + meteorological data (temperature, relative humidity, wind speed, wind direction): 11 variables.
   - **Rationale:** Replicates the base study's feature set on Chandigarh data. The station's NOx column is excluded because it is unusable (it never exceeds NO + NO2; see `CHANGES.md`). CO is used as reported by OpenAQ (labelled ppb; its magnitude matches the mg/m³ scale used by CPCB, the original provider).

2. **Time features (cyclical encoding)**
   - **Features:** Base + sin/cos of the hour of day and of the day of week, in local time (Asia/Kolkata).
   - **Rationale:** Gives the generator an explicit clock for diurnal and weekly cycles. The time features are generated like any other variable and projected back onto the unit circle afterwards. How often the generated hour advances by about one hour is measured and reported.

3. **Training budget (100,000 vs 7,500 epochs)**
   - **Selection:** The short budget is chosen once, on the validation block, for the Base configuration only, from 1,000 to 10,000 epochs with separate tuning seeds (11, 22, 33). 7,500 epochs gave the lowest held-out EMD; between 5,000 and 10,000 epochs the difference is below 0.002.
   - **Final runs:** Every configuration is trained with the same three seeds (42, 123, 789).

4. **Second dataset (Bilbao)**
   - The base study's Bilbao data, with its vehicle and 8-hour columns removed so it matches Chandigarh. Only Base-100k and Time-7.5k are run there, as a transfer check.

**Data and Split (Chandigarh)**

- Hourly grid in local time. Gaps of at most 3 consecutive hours are linearly interpolated (4.5% of all values) and the remaining incomplete hours are dropped, leaving 11,219 usable hours in 147 contiguous segments. No window or lag sample crosses a gap.
- First 336 usable hours: training (13 non-overlapping 24-hour windows). Next 336: validation (budget selection and held-out fidelity). All later hours: test (forecasting, and an extra fidelity check on data never used for selection).

**Evaluation Methodology: Fidelity, Memorisation and TSTR**

Synthetic data is judged both on how closely it resembles real data and on whether it is useful for training a forecaster.

1. **Fidelity:** Normalised Earth Mover's Distance (EMD divided by the real standard deviation), Jensen–Shannon distance, mean autocorrelation error (lags 1–12) and the Frobenius norm of the correlation-matrix difference. Each is computed against the training days, the held-out validation days and all complete days of the test block.

2. **Memorisation Check:** The mean distance of synthetic days to their nearest real training day, divided by the same quantity for unseen real days. A ratio well below 1 means synthetic days stay close to the training days.

3. **Downstream Utility (TSTR):** Train-on-Synthetic, Test-on-Real with random forest and gradient boosting, predicting PM2.5 one hour ahead from the previous 12 hours. The reference is the same model trained on the 13 real days (TRTR). Utility Retention Ratio URR = TSTR MAE / TRTR MAE, so below 1 means the synthetic data beat the real days.

4. **Controls:** Synthetic rows shuffled in time (to check that the forecaster uses temporal structure) and a persistence forecast (next hour = this hour). Persistence beats every data source at this one-hour horizon. The study compares data sources under a fixed forecaster, not forecasting methods.

```
==================================================
CHANDIGARH: mean ± std over 3 seeds (42, 123, 789)
==================================================
                              Base-100k        Time-100k        Base-7.5k        Time-7.5k
--- FIDELITY (lower = better) ---
EMD vs held-out days        0.3327 ± 0.0020  0.3290 ± 0.0023  0.3434 ± 0.0061  0.3642 ± 0.0030
EMD vs test-block days      0.5059 ± 0.0014  0.5028 ± 0.0018  0.5109 ± 0.0063  0.5328 ± 0.0061
ΔACF vs held-out days       0.1689 ± 0.0023  0.1650 ± 0.0009  0.1818 ± 0.0054  0.1819 ± 0.0041
Correlation Frobenius       1.3588 ± 0.0259  1.3928 ± 0.0108  1.4881 ± 0.0394  1.5246 ± 0.0178
Memorisation ratio          0.360 ± 0.007    0.319 ± 0.013    0.632 ± 0.003    0.627 ± 0.007

--- TSTR DOWNSTREAM UTILITY (next-hour PM2.5 MAE, µg/m³) ---
Random forest               6.98 ± 0.35      6.92 ± 0.10      7.42 ± 0.19      7.15 ± 0.38
Gradient boosting           6.79 ± 0.03      6.85 ± 0.04      7.71 ± 0.11      7.86 ± 0.51
URR (random forest)         0.838 ± 0.042    0.830 ± 0.012    0.891 ± 0.023    0.858 ± 0.045
Shuffled-rows control       29.40 ± 0.55     27.79 ± 0.57     31.29 ± 2.73     31.21 ± 1.42

[References] Train on real only (TRTR, RF): 8.34 | Persistence: 4.68
[Real training vs real held-out days] EMD 0.326 | vs test-block days 0.496
[Generated hour advancing 0.5–1.5 h] Time-100k 91.5% | Time-7.5k 67.4%
```

## Installation
```bash
git clone https://github.com/Anubhavthakur1996/chand_aq_synth.git
cd chand_aq_synth
pip install -r requirements.txt
pip install --no-deps gretel-synthetics==0.20.0   # must be installed WITHOUT its dependencies
```
See `INSTALL_WSL.md` for a step-by-step WSL2 + CUDA setup and the comments in `requirements.txt` for a CPU-only install.

## Usage
```bash
# 1. Tests
python tests/test_windows.py        # unit tests of the windowing code
python tests/test_metrics.py        # evaluation formulas on data with known answers
python tests/smoke_test.py          # whole pipeline with a stand-in generator (no GAN, ~2 min)

# 2. Data preparation and reproducibility check
python main.py prep                 # cleaning; writes results/<dataset>/data_report.json
python main.py repro                # does the same seed give identical output on your machine?

# 3. Budget selection and final runs
python main.py tune                 # picks the short epoch budget -> results/chosen_epochs.json
python main.py run                  # Chandigarh, 4 configurations x 3 seeds (resumable; skips finished runs)
python main.py run --dataset bilbao # Bilbao: Base-100k and Time-7.5k

# 4. Reports
python main.py summarize            # results/<dataset>/summary.txt, summary.csv, all_runs.csv

# 5. Replication check against the base paper (after the Bilbao a1 runs)
python main.py replicate --base-synth "<base repo>/3. AI models/1. Real vs Synth vs RealSynth/datasets/synth/60_Bilbao_synth.csv"

# 6. Paper revision checks and figures (no retraining; read the saved runs)
python -I scripts/paper/paper_checks.py                # interpolation shares, ACF check, CO source scale, generated-hour diagnostics
python -I scripts/paper/rescore_independent_blocks.py  # fidelity vs an independent block and the whole test block
python -I scripts/paper/make_fig1_fig3.py              # Figs. 1 and 3
python -I scripts/paper/make_fig4.py                   # Fig. 4
python tests/test_paper_revision.py                    # checks every number added in the revision
```
`python main.py all` runs steps 2–5 in order. All settings live in `config.py`.

## Project Structure
```text
chand_aq_synth/
├── archive/                         # Earlier three-tier version (code, data and results), kept unchanged
├── data/
│   ├── raw/                         # Chandigarh Sector 22 (OpenAQ) and Bilbao source data
│   └── processed/                   # Cleaned hourly data written by `main.py prep`
├── results/
│   ├── chosen_epochs.json           # Short budget selected by `main.py tune`
│   ├── chandigarh/
│   │   ├── runs/                    # One JSON per configuration and seed (metrics and diagnostics)
│   │   ├── synthetic/               # Generated data per run (.npz)
│   │   ├── tuning/                  # Budget-selection runs and plot
│   │   ├── revision/                # Outputs of scripts/paper (figures, checks, re-scoring)
│   │   ├── summary.txt / .csv       # Mean ± std over seeds and paired differences
│   │   ├── all_runs.csv             # Every run in one table
│   │   └── data_report.json         # Usable hours, segments, interpolation counts
│   └── bilbao/                      # Same layout for the second dataset
├── scripts/paper/                   # Revision checks and figure scripts for the paper
├── src/                             # Core pipeline modules
│   ├── preprocess.py                # Cleaning, unit conversion, interpolation, segments
│   ├── windows.py                   # 24-hour windows and lag samples that never cross a gap
│   ├── train_dgan.py                # DoppelGANger training, generation and post-processing
│   ├── evaluate.py                  # Fidelity metrics, memorisation check, TSTR/TRTR utility
│   ├── experiments.py               # Budget tuning and final runs per configuration and seed
│   ├── replication_check.py         # Comparison with the base paper's released synthetic data
│   └── seeding.py                   # Seed control for reproducible runs
├── tests/                           # Unit tests, smoke test and revision checks
├── utils/
│   └── fetchdata.py                 # OpenAQ download of the raw Chandigarh data
├── config.py                        # All settings (seeds, budgets, columns, split sizes)
├── main.py                          # Command-line entry point
├── CHANGES.md                       # What changed from the earlier version, and why
├── INSTALL_WSL.md                   # Step-by-step installation on WSL2 with CUDA
├── requirements.txt                 # Pinned dependencies
├── LICENCE                          # MIT licence
└── README.md                        # Repository documentation
```

## Abstract
The air quality monitoring is important in city management and prediction of public health, but the sensor data tends to be sparse due to failures in calibration and deployment of sensors. In order to handle the problem of missing data, GANs have often been utilised for generation of high-quality multivariate synthetic data. Therefore, this paper builds on the DoppelGANger pipeline of Morales-García et al. and studies two modifications on hourly data from one monitoring station in Chandigarh, India: cyclical (sine/cosine) hour and day-of-week encodings, and reduced training budget of 7,500 instead of 100,000 epochs. The four resulting configurations are trained on 13 days of real data with three random seeds each and are assessed with normalised Earth Mover's Distance, Jensen–Shannon distance, autocorrelation and correlation-structure errors, a nearest-neighbour memorisation check, and Train-on-Synthetic-Test-on-Real (TSTR) forecasting of next-hour PM2.5 with random forest and gradient boosting. Synthetic data lowered random-forest forecasting error by 11–17% compared with training on the real days alone. Time encoding produced a small but consistent fidelity gain at 100,000 epochs (held-out EMD 0.329 versus 0.333) and no downstream benefit. The 7,500-epoch budget, 13 times cheaper, retained most random-forest utility (MAE 7.42 versus 6.98, within seed spread) with a weaker memorisation signal, but with lower fidelity and lower gradient-boosting utility. On the Bilbao data of the base study, the combined configuration was worse than the baseline. These mixed results are reported together with an open, seed-controlled evaluation protocol.

## How to Cite
If you use this code, data or evaluation protocol in your research, please cite our paper:

**Plain Text:**
> Singh, T. A., Gupta, M., & Kumar, R. (2027). *Training Budget and Cyclical Time Encoding in DoppelGANger-Based Synthetic Air-Quality Data: A Multi-Seed Evaluation*. International Conference on Reasoning AI in Future Communication Systems (ICRFCS-2027). Accepted, to appear.

**BibTeX:**
```bibtex
@inproceedings{singh2027doppelganger,
  title     = {Training Budget and Cyclical Time Encoding in DoppelGANger-Based Synthetic Air-Quality Data: A Multi-Seed Evaluation},
  author    = {Singh, Thakur Anubhav and Gupta, Meenu and Kumar, Rakesh},
  booktitle = {International Conference on Reasoning AI in Future Communication Systems (ICRFCS-2027)},
  year      = {2027},
  note      = {Accepted, to appear}
}
```

Please also cite the base study:
> Morales-García, J., Ramos-Sorroche, E., Balderas-Díaz, S., Guerrero-Contreras, G., Muñoz, A., Santa, J., & Terroso-Sáenz, F. (2025). Exploiting synthetic data generation to enhance pollution prediction. *Applied Soft Computing*, 175, 113076. https://doi.org/10.1016/j.asoc.2025.113076

## Requirements
- Python 3.10 (tested on WSL2 with CUDA 12.6)
- PyTorch (CUDA recommended; CPU works but is slow for 100,000 epochs)
- gretel-synthetics 0.20.0 (installed with `--no-deps`)
- numpy, pandas, pyarrow, scipy, scikit-learn, statsmodels, matplotlib, einops, tqdm, requests, category-encoders
- python-dotenv (only for `utils/fetchdata.py`)

Exact pinned versions are in `requirements.txt`.

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the `LICENCE` file for details.

## Authors
Thakur Anubhav Singh, Department of Computer Science, Chandigarh University
(anubhavthakur1996@outlook.com)

Meenu Gupta, School of Computer Science Engineering, Geeta University

Rakesh Kumar, Department of Computer Science, Chandigarh University
