# Dam Water Level Forecasting — Morocco

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-41%20passed-2E7D32)](#testing)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

Multi-horizon forecasting of **dam water storage levels in Morocco**, built as a rigorous
time-series research pipeline: seasonal-naive and XGBoost walk-forward baselines,
zero-shot vs **LoRA fine-tuned Chronos** (time-series foundation model), and
**conformal prediction intervals** with calibrated coverage — all tracked in MLflow
and validated with statistical tests.

---

## Why it matters

Morocco is one of the most water-stressed countries in the world. Reliable medium-term
forecasts of dam storage (1–6 months ahead) directly support water allocation decisions,
agriculture planning and drought early warning. This project forecasts storage for
**100 dam time series** across 5 hydrological basins (Oum Er-Rbia, Draa, Tensift,
Inaouen, Moulouya), including flagship reservoirs such as **Al Massira** and
**Bin el Ouidane**.

---

## Results

**MASE (lower = better, < 1.0 means beating the seasonal-naive baseline)** — mean over all dams:

| Horizon | Naive Seasonal | XGBoost (walk-forward) | Chronos (zero-shot) | **Chronos + LoRA** |
|---------|---------------:|----------------------:|--------------------:|-------------------:|
| 1 month | 1.56 | 0.73 | 0.71 | **0.75** |
| 3 months | 3.34 | 0.89 | 2.17 | **1.92** |
| 6 months | 4.32 | **0.95** | 4.27 | 4.01 |

- **~52 % MASE reduction** over the seasonal-naive baseline at the 1-month horizon (Chronos-LoRA).
- Walk-forward validation (5 rolling windows) and **Diebold-Mariano tests** for significance.
- **Conformal prediction intervals** (target 90 %) with empirical coverage ≈ **0.86** and CRPS reported per dam/horizon.

---

## Data

| Source | Content |
|--------|---------|
| Moroccan basin agencies (Excel exports) | Monthly dam storage levels for Souss-Massa, Tensift and other basins |
| [ERA5](https://cds.climate.copernicus.eu/) reanalysis | Climate covariates (precipitation, temperature, ...) aligned per dam |
| `configs/barrages.yaml` | Dam registry: geolocation, basin, max capacity, dam height, commissioning year |

Engineered features (`src/data/dam_features.py`): seasonal lags, rolling aggregates,
calendar features and basin-level covariates.

---

## Models

1. **Naive Seasonal** — seasonal naive baseline (seasonality = 12 months).
2. **XGBoost** — gradient boosting with walk-forward validation, per-horizon models.
3. **Chronos (zero-shot)** — Amazon's time-series foundation model, no training.
4. **Chronos + LoRA** — parameter-efficient fine-tuning of Chronos on the dam corpus (single consumer GPU).
5. **Conformal wrapper** — split-conformal calibration of residuals for prediction intervals at any target coverage.

---

## Project structure

```
src/
├── config.py             # Central configuration (paths, horizons, seeds, device)
├── data/                 # Ingestion (Excel, ERA5), cleaning, feature engineering, synthetic mode
├── models/               # baseline_xgboost · chronos_zeroshot · finetune_lora · eval_lora · conformal
├── evaluation/           # MAE, RMSE, MASE, Diebold-Mariano, coverage
└── dashboard/            # Visualization (WIP)
reports/                  # Comparison tables (MASE/RMSE/coverage per dam × horizon)
tests/                    # 41 unit tests (features, metrics, conformal)
```

---

## Quickstart

```bash
git clone https://github.com/Mohamed-ettaouil12/dam-water-level-forecasting.git
cd dam-water-level-forecasting

python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .

# Baseline + foundation models
.venv/bin/python -m src.models.baseline_xgboost      # XGBoost walk-forward
.venv/bin/python -m src.models.chronos_zeroshot      # Chronos zero-shot
.venv/bin/python -m src.models.finetune_lora         # LoRA fine-tuning (GPU)
.venv/bin/python -m src.models.eval_lora             # Evaluate fine-tuned model
.venv/bin/python -m src.models.run_conformal         # Conformal intervals + coverage

# Tests
.venv/bin/python -m pytest -q
```

All runs are logged to **MLflow** (`mlruns/`) and produce CSV reports in `reports/`.

---

## Testing

```bash
.venv/bin/python -m pytest -q   # 41 tests passed in ~4 s
```

---

## Author

**Mohamed Ettaouil** — Engineering student in AI & Data Science (ENSIAS, Rabat)
· [LinkedIn](https://www.linkedin.com/in/mohamed-ettaouil) · [GitHub profile](https://github.com/Mohamed-ettaouil12)
