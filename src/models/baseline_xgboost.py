"""Baseline XGBoost avec validation walk-forward."""

import logging
import warnings
from pathlib import Path
from typing import List, Tuple

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import SEED, DATA_PROCESSED, MLRUNS_DIR, REPORTS_DIR
from src.data.dam_features import load_dam_features
from src.evaluation.metrics import (
    coverage,
    diebold_mariano,
    mae,
    mase,
    rmse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 3, 6]
N_WINDOWS = 5
SEASONALITY = 12


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Prépare les features pour XGBoost."""
    df = df.copy()
    df = df.sort_values(["barrage", "date"]).reset_index(drop=True)

    feature_cols = []

    for barrage, group in df.groupby("barrage"):
        idx = group.index

        for lag in [1, 3, 6, 12]:
            col = f"niveau_lag_{lag}"
            df.loc[idx, col] = group["niveau_pct"].shift(lag).values
            feature_cols.append(col)

        for window in [3, 6]:
            col = f"niveau_roll_mean_{window}"
            df.loc[idx, col] = group["niveau_pct"].rolling(window, min_periods=1).mean().values
            feature_cols.append(col)

        df.loc[idx, "month_sin"] = np.sin(2 * np.pi * group["date"].dt.month / 12).values
        df.loc[idx, "month_cos"] = np.cos(2 * np.pi * group["date"].dt.month / 12).values
        feature_cols.extend(["month_sin", "month_cos"])

        if "precip_3m" in group.columns:
            df.loc[idx, "precip_3m"] = group["precip_3m"].values
            feature_cols.append("precip_3m")
        if "precip_6m" in group.columns:
            df.loc[idx, "precip_6m"] = group["precip_6m"].values
            feature_cols.append("precip_6m")
        if "temp_3m_avg" in group.columns:
            df.loc[idx, "temp_3m_avg"] = group["temp_3m_avg"].values
            feature_cols.append("temp_3m_avg")

    static_cols = ["capacite_max_m3", "hauteur_m", "annee_mise_en_service"]
    for col in static_cols:
        if col in df.columns:
            feature_cols.append(col)

    feature_cols = list(dict.fromkeys(feature_cols))
    return df, feature_cols


def create_targets(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    """Crée les colonnes cibles pour chaque horizon."""
    df = df.copy()
    df = df.sort_values(["barrage", "date"]).reset_index(drop=True)

    for h in horizons:
        df[f"target_h{h}"] = df.groupby("barrage")["niveau_pct"].shift(-h)
    return df


def naive_seasonal_forecast(df: pd.DataFrame, horizon: int) -> np.ndarray:
    """Prévision naïve saisonnière = même mois année précédente."""
    df = df.sort_values(["barrage", "date"])
    preds = []
    for barrage, group in df.groupby("barrage"):
        group = group.sort_values("date")
        for i in range(len(group)):
            target_date = group.iloc[i]["date"]
            ref_date = target_date - pd.DateOffset(years=1)
            ref_row = group[group["date"] == ref_date]
            if len(ref_row) > 0:
                preds.append(ref_row["niveau_pct"].values[0])
            else:
                preds.append(np.nan)
    return np.array(preds)


def walk_forward_split(
    df: pd.DataFrame,
    n_windows: int = N_WINDOWS,
    horizon: int = 1,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Génère les splits walk-forward expanding window."""
    df = df.sort_values("date").reset_index(drop=True)
    dates = df["date"].unique()
    dates.sort()

    n_dates = len(dates)
    test_size = 12
    min_train = n_dates - test_size * n_windows

    splits = []
    for i in range(n_windows):
        train_end_idx = min_train + i * test_size
        test_start_idx = train_end_idx
        test_end_idx = min(train_end_idx + test_size, n_dates)

        if test_start_idx >= test_end_idx:
            break

        train_dates = dates[:train_end_idx]
        test_dates = dates[test_start_idx:test_end_idx]

        train = df[df["date"].isin(train_dates)].copy()
        test = df[df["date"].isin(test_dates)].copy()

        if len(train) > 0 and len(test) > 0:
            splits.append((train, test))

    return splits


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict = None,
) -> xgb.XGBRegressor:
    """Entraîne un modèle XGBoost."""
    if params is None:
        params = {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": SEED,
            "n_jobs": -1,
            "objective": "reg:squarederror",
            "verbosity": 0,
        }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)
    return model


def evaluate_window(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    horizon: int,
) -> dict:
    """Évalue sur une fenêtre walk-forward."""
    train_clean = train.dropna(subset=feature_cols + [target_col])
    test_clean = test.dropna(subset=feature_cols + [target_col])

    if len(train_clean) < 50 or len(test_clean) < 5:
        return None

    X_train = train_clean[feature_cols]
    y_train = train_clean[target_col]
    X_test = test_clean[feature_cols]
    y_test = test_clean[target_col]

    model = train_xgboost(X_train, y_train)
    y_pred = model.predict(X_test)

    naive_pred = naive_seasonal_forecast(test_clean, horizon)

    metrics = {
        "MAE": mae(y_test.values, y_pred),
        "RMSE": rmse(y_test.values, y_pred),
        "MASE": mase(y_test.values, y_pred, train_clean["niveau_pct"].values, SEASONALITY),
    }

    if "target_lower" in test_clean.columns and "target_upper" in test_clean.columns:
        metrics["Coverage"] = coverage(y_test.values, test_clean["target_lower"].values, test_clean["target_upper"].values)

    return metrics


def run_baseline() -> pd.DataFrame:
    """Lance la baseline complète."""
    logger.info("=== Début baseline XGBoost walk-forward ===")

    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR.absolute()}")
    mlflow.set_experiment("barrages-baseline-xgboost")

    data_path = DATA_PROCESSED / "dataset_complet.parquet"
    if not data_path.exists():
        logger.error("Fichier dataset_complet.parquet non trouvé. Lancez d'abord src.data.era5 et src.data.clean")
        return pd.DataFrame()

    df = pd.read_parquet(data_path)
    logger.info("Données chargées: %d lignes, %d barrages", len(df), df["barrage"].nunique())

    df = prepare_features(df)
    df = create_targets(df, HORIZONS)

    all_results = []

    for barrage in sorted(df["barrage"].unique()):
        df_barrage = df[df["barrage"] == barrage].copy()
        feature_cols = [c for c in df.columns if c.startswith(("niveau_", "month_", "precip_", "temp_", "capacite", "hauteur", "annee"))]

        for horizon in HORIZONS:
            target_col = f"target_h{horizon}"
            if target_col not in df_barrage.columns:
                continue

            logger.info("Barrage: %s, Horizon: %d mois", barrage, horizon)

            splits = walk_forward_split(df_barrage, N_WINDOWS, horizon)

            window_results = []
            for window_idx, (train, test) in enumerate(splits):
                logger.info("  Fenêtre %d/%d: train=%d, test=%d", window_idx + 1, len(splits), len(train), len(test))

                with mlflow.start_run(run_name=f"{barrage}_h{horizon}_w{window_idx}") as run:
                    mlflow.log_param("barrage", barrage)
                    mlflow.log_param("horizon", horizon)
                    mlflow.log_param("window", window_idx)
                    mlflow.log_param("train_size", len(train))
                    mlflow.log_param("test_size", len(test))

                    metrics = evaluate_window(train, test, feature_cols, target_col, horizon)

                    if metrics is None:
                        logger.warning("  Fenêtre ignorée (pas assez de données)")
                        continue

                    for k, v in metrics.items():
                        mlflow.log_metric(k, v)

                    window_results.append(metrics)
                    logger.info("  MAE=%.4f, RMSE=%.4f, MASE=%.4f", metrics["MAE"], metrics["RMSE"], metrics["MASE"])

            if window_results:
                mean_metrics = {k: np.mean([r[k] for r in window_results]) for k in window_results[0].keys()}
                std_metrics = {k: np.std([r[k] for r in window_results]) for k in window_results[0].keys()}

                for k, v in mean_metrics.items():
                    all_results.append({
                        "barrage": barrage,
                        "horizon": horizon,
                        "metric": k,
                        "mean": v,
                        "std": std_metrics[k],
                        "n_windows": len(window_results),
                    })

                logger.info("  Moyenne ± std: MAE=%.4f±%.4f, RMSE=%.4f±%.4f, MASE=%.4f±%.4f",
                           mean_metrics["MAE"], std_metrics["MAE"],
                           mean_metrics["RMSE"], std_metrics["RMSE"],
                           mean_metrics["MASE"], std_metrics["MASE"])

    results_df = pd.DataFrame(all_results)
    output_path = REPORTS_DIR / "metrics_xgboost.csv"
    results_df.to_csv(output_path, index=False)
    logger.info("Résultats sauvegardés: %s", output_path)

    logger.info("=== Fin baseline XGBoost ===")
    return results_df


def main() -> int:
    results = run_baseline()
    if results.empty:
        logger.error("Aucun résultat généré")
        return 1

    print("\n=== RÉSUMÉ BASELINE XGBOOST ===")
    pivot = results.pivot_table(index=["barrage", "horizon"], columns="metric", values=["mean", "std"])
    print(pivot.to_string())

    return 0


if __name__ == "__main__":
    exit(main())