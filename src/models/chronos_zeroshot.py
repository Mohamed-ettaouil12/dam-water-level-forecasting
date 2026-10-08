"""Évaluation zero-shot Chronos-Bolt avec validation walk-forward (même protocole que XGBoost)."""

import logging
import warnings
from pathlib import Path
from typing import List, Tuple

import mlflow
import numpy as np
import pandas as pd
import torch

from src.config import DATA_PROCESSED, MLRUNS_DIR, REPORTS_DIR, DATA_SOURCE, SEED
from src.evaluation.metrics import (
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
QUANTILE_IDX = 4  # Index de la médiane (0.5 quantile) dans les 9 quantiles [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def load_data_by_source() -> pd.DataFrame:
    """Charge les données nettoyées selon DATA_SOURCE."""
    if DATA_SOURCE == "real":
        path = DATA_PROCESSED / "barrages_clean.parquet"
        logger.info("Chargement données RÉELLES depuis %s", path)
    else:
        path = DATA_PROCESSED / "synthetic" / "barrages_clean_synthetic.parquet"
        logger.info("Chargement données SYNTHÉTIQUES depuis %s", path)

    if not path.exists():
        raise FileNotFoundError(f"Fichier non trouvé : {path}")

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["barrage", "date"]).reset_index(drop=True)
    logger.info("Données chargées : %d lignes, %d barrages", len(df), df["barrage"].nunique())
    return df


def load_xgboost_results() -> pd.DataFrame:
    """Recharge les résultats XGBoost et naïve pour comparaison."""
    path = REPORTS_DIR / "metrics_xgboost.csv"
    if not path.exists() or path.stat().st_size == 0:
        logger.warning("Fichier XGBoost non trouvé ou vide : %s", path)
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        logger.info("Résultats XGBoost/Naïve rechargés : %d lignes", len(df))
        return df
    except pd.errors.EmptyDataError:
        logger.warning("Fichier XGBoost vide : %s", path)
        return pd.DataFrame()


def get_chronos_pipeline():
    """Charge le pipeline Chronos une seule fois (singleton)."""
    global _chronos_pipeline
    if "_chronos_pipeline" not in globals() or _chronos_pipeline is None:
        from chronos import BaseChronosPipeline
        logger.info("Chargement modèle Chronos-Bolt (amazon/chronos-bolt-small)...")
        _chronos_pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-small",
            device_map="cpu",
            torch_dtype=torch.bfloat16,
        )
        logger.info("Modèle Chronos chargé sur CPU")
    return _chronos_pipeline


def chronos_predict_series(
    pipeline,
    series: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Prédit avec Chronos pour une série (retourne médiane)."""
    context = torch.tensor(series.astype(np.float32))
    forecast = pipeline.predict(inputs=context, prediction_length=horizon)
    # forecast shape: [batch=1, num_quantiles=9, prediction_length]
    # On prend la médiane (quantile 0.5 = index 4)
    median_forecast = forecast[0, QUANTILE_IDX, :].cpu().numpy()
    # S'assurer que c'est un array 1D de longueur horizon
    median_forecast = np.atleast_1d(median_forecast)
    logger.debug(f"chronos_predict_series: forecast shape={forecast.shape}, median_forecast shape={median_forecast.shape}")
    return median_forecast


def walk_forward_split(
    df: pd.DataFrame,
    n_windows: int = N_WINDOWS,
    horizon: int = 1,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Génère les splits walk-forward expanding window."""
    df = df.sort_values("date").reset_index(drop=True)
    dates = np.sort(df["date"].unique())

    n_dates = len(dates)
    test_size = 12
    min_train = n_dates - test_size * n_windows

    if min_train <= 0:
        logger.warning("Pas assez de données pour %d fenêtres", n_windows)
        return []

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


def evaluate_window(
    pipeline,
    train: pd.DataFrame,
    test: pd.DataFrame,
    horizon: int,
) -> dict:
    """Évalue sur une fenêtre walk-forward pour une série.
    
    Pour chaque mois du test, prédit h mois en avant (rolling forecast).
    """
    logger.debug(f"evaluate_window called with horizon={horizon}, type={type(horizon)}")
    train_clean = train.dropna(subset=["niveau_pct"])
    test_clean = test.dropna(subset=["niveau_pct"])

    if len(train_clean) < 50 or len(test_clean) < 5:
        return None

    y_train = train_clean["niveau_pct"].values
    y_test = test_clean["niveau_pct"].values

    # Rolling forecast: pour chaque mois du test, prédit h mois en avant
    y_pred = []
    naive_pred = []
    
    for i in range(len(test_clean)):
        # Contexte = train + test précédents
        context_end_idx = len(train_clean) + i
        context = np.concatenate([y_train, test_clean["niveau_pct"].values[:i]])
        
        # Prédiction Chronos h mois en avant
        pred = chronos_predict_series(pipeline, context, horizon)
        if len(pred) >= horizon:
            y_pred.append(pred[horizon - 1])  # Prédiction h mois en avant
        else:
            y_pred.append(np.nan)
        
        # Prévision naïve (même mois année précédente)
        target_date = test_clean.iloc[i]["date"]
        ref_date = target_date - pd.DateOffset(years=1)
        ref_row = train_clean[train_clean["date"] == ref_date]
        if len(ref_row) > 0:
            naive_pred.append(ref_row["niveau_pct"].values[0])
        else:
            naive_pred.append(np.nan)

    y_pred = np.array(y_pred)
    naive_pred = np.array(naive_pred)
    
    logger.debug(f"y_pred shape: {y_pred.shape}, y_test shape: {y_test.shape}, horizon={horizon}")

    # Prévision naïve saisonnière (même mois année précédente)
    naive_pred = []
    for i in range(len(test_clean)):
        target_date = test_clean.iloc[i]["date"]
        ref_date = target_date - pd.DateOffset(years=1)
        ref_row = train_clean[train_clean["date"] == ref_date]
        if len(ref_row) > 0:
            naive_pred.append(ref_row["niveau_pct"].values[0])
        else:
            naive_pred.append(np.nan)
    naive_pred = np.array(naive_pred)
    logger.debug(f"naive_pred shape: {naive_pred.shape}")

    # Métriques
    mask = ~np.isnan(y_test) & ~np.isnan(y_pred)
    logger.debug(f"mask shape: {mask.shape}, y_pred shape: {y_pred.shape}, y_test shape: {y_test.shape}")
    if not mask.any():
        return None

    metrics = {
        "MAE": mae(y_test[mask], y_pred[mask]),
        "RMSE": rmse(y_test[mask], y_pred[mask]),
        "MASE": mase(y_test[mask], y_pred[mask], y_train, SEASONALITY),
        "MAE_naive": mae(y_test, naive_pred),
        "RMSE_naive": rmse(y_test, naive_pred),
        "MASE_naive": mase(y_test, naive_pred, y_train, SEASONALITY),
    }

    return metrics


def run_chronos() -> pd.DataFrame:
    """Lance l'évaluation Chronos zero-shot."""
    logger.info("=== Début évaluation Chronos zero-shot (source=%s) ===", DATA_SOURCE)

    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR.absolute()}")
    mlflow.set_experiment("barrages-chronos-zeroshot")

    df = load_data_by_source()
    pipeline = get_chronos_pipeline()
    xgb_results = load_xgboost_results()

    all_results = []

    for barrage in sorted(df["barrage"].unique()):
        df_barrage = df[df["barrage"] == barrage].copy()

        for horizon in HORIZONS:
            logger.info("Barrage: %s, Horizon: %d mois", barrage, horizon)

            splits = walk_forward_split(df_barrage, N_WINDOWS, horizon)

            window_results = []
            window_naive_results = []

            for window_idx, (train, test) in enumerate(splits):
                logger.info("  Fenêtre %d/%d: train=%d, test=%d", window_idx + 1, len(splits), len(train), len(test))

                with mlflow.start_run(run_name=f"{barrage}_h{horizon}_w{window_idx}") as run:
                    mlflow.log_param("barrage", barrage)
                    mlflow.log_param("horizon", horizon)
                    mlflow.log_param("window", window_idx)
                    mlflow.log_param("train_size", len(train))
                    mlflow.log_param("test_size", len(test))
                    mlflow.log_param("source", DATA_SOURCE)

                    metrics = evaluate_window(pipeline, train, test, horizon)

                    if metrics is None:
                        logger.warning("  Fenêtre ignorée (pas assez de données)")
                        continue

                    for k, v in metrics.items():
                        mlflow.log_metric(k, v)

                    window_results.append({
                        "MAE": metrics["MAE"],
                        "RMSE": metrics["RMSE"],
                        "MASE": metrics["MASE"],
                    })
                    window_naive_results.append({
                        "MAE": metrics["MAE_naive"],
                        "RMSE": metrics["RMSE_naive"],
                        "MASE": metrics["MASE_naive"],
                    })
                    logger.info("  Chronos: MAE=%.4f, RMSE=%.4f, MASE=%.4f", metrics["MAE"], metrics["RMSE"], metrics["MASE"])
                    logger.info("  Naive:   MAE=%.4f, RMSE=%.4f, MASE=%.4f", metrics["MAE_naive"], metrics["RMSE_naive"], metrics["MASE_naive"])

            if window_results:
                mean_chr = {k: np.mean([r[k] for r in window_results]) for k in window_results[0].keys()}
                std_chr = {k: np.std([r[k] for r in window_results]) for k in window_results[0].keys()}
                mean_naive = {k: np.mean([r[k] for r in window_naive_results]) for k in window_naive_results[0].keys()}
                std_naive = {k: np.std([r[k] for r in window_naive_results]) for k in window_naive_results[0].keys()}

                for k in mean_chr.keys():
                    all_results.append({
                        "barrage": barrage,
                        "horizon": horizon,
                        "metric": k,
                        "model": "Chronos_ZeroShot",
                        "mean": mean_chr[k],
                        "std": std_chr[k],
                        "n_windows": len(window_results),
                    })
                    all_results.append({
                        "barrage": barrage,
                        "horizon": horizon,
                        "metric": k,
                        "model": "Naive_Seasonal",
                        "mean": mean_naive[k],
                        "std": std_naive[k],
                        "n_windows": len(window_naive_results),
                    })

                logger.info("  Chronos Moyenne ± std: MAE=%.4f±%.4f, RMSE=%.4f±%.4f, MASE=%.4f±%.4f",
                           mean_chr["MAE"], std_chr["MAE"],
                           mean_chr["RMSE"], std_chr["RMSE"],
                           mean_chr["MASE"], std_chr["MASE"])
                logger.info("  Naive Moyenne ± std:   MAE=%.4f±%.4f, RMSE=%.4f±%.4f, MASE=%.4f±%.4f",
                           mean_naive["MAE"], std_naive["MAE"],
                           mean_naive["RMSE"], std_naive["RMSE"],
                           mean_naive["MASE"], std_naive["MASE"])

    results_df = pd.DataFrame(all_results)
    output_path = REPORTS_DIR / "metrics_chronos_zeroshot.csv"
    results_df.to_csv(output_path, index=False)
    logger.info("Résultats Chronos sauvegardés: %s", output_path)

    logger.info("=== Fin évaluation Chronos ===")
    return results_df


def create_comparison_table(chronos_df: pd.DataFrame, xgb_df: pd.DataFrame) -> pd.DataFrame:
    """Crée le tableau comparatif final."""
    # Filtrer MASE
    chronos_mase = chronos_df[chronos_df["metric"] == "MASE"]
    naive_df = chronos_df[chronos_df["model"] == "Naive_Seasonal"]
    
    # Moyennes globales par modèle et horizon
    comparison = []

    for model_name, model_df in [("Chronos_ZeroShot", chronos_mase), ("Naive_Seasonal", naive_df)]:
        for horizon in HORIZONS:
            hdf = model_df[model_df["horizon"] == horizon]
            if len(hdf) > 0:
                comparison.append({
                    "model": model_name,
                    "horizon": horizon,
                    "MASE_mean": hdf["mean"].mean(),
                    "MASE_std": hdf["std"].mean(),
                })

    # Ajouter XGBoost si disponible
    if not xgb_df.empty and "metric" in xgb_df.columns:
        xgb_mase = xgb_df[xgb_df["metric"] == "MASE"]
        for horizon in HORIZONS:
            hdf = xgb_mase[xgb_mase["horizon"] == horizon]
            if len(hdf) > 0:
                comparison.append({
                    "model": "XGBoost",
                    "horizon": horizon,
                    "MASE_mean": hdf["mean"].mean(),
                    "MASE_std": hdf["std"].mean(),
                })

    comparison_df = pd.DataFrame(comparison)
    return comparison_df


def main() -> int:
    chronos_df = run_chronos()
    if chronos_df.empty:
        logger.error("Aucun résultat Chronos généré")
        return 1

    xgb_df = load_xgboost_results()
    if xgb_df.empty:
        logger.warning("Résultats XGBoost non trouvés")

    # Tableau comparatif
    comparison = create_comparison_table(chronos_df, xgb_df)

    output_path = REPORTS_DIR / "comparison_zero_vs_xgb.csv"
    comparison.to_csv(output_path, index=False)
    logger.info("Tableau comparatif sauvegardé: %s", output_path)

    print("\n=== TABLEAU COMPARATIF MASE (moyenne ± écart-type) ===")
    for horizon in HORIZONS:
        print(f"\nHorizon {horizon} mois:")
        hdf = comparison[comparison["horizon"] == horizon]
        for _, row in hdf.iterrows():
            print(f"  {row['model']:20s} : MASE = {row['MASE_mean']:.4f} ± {row['MASE_std']:.4f}")

    # Verdict
    print("\n=== VERDICT ===")
    for horizon in HORIZONS:
        hdf = comparison[comparison["horizon"] == horizon]
        best = hdf.loc[hdf["MASE_mean"].idxmin()]
        worst = hdf.loc[hdf["MASE_mean"].idxmax()]
        print(f"Horizon {horizon}m: Meilleur = {best['model']} (MASE={best['MASE_mean']:.4f}), Pire = {worst['model']} (MASE={worst['MASE_mean']:.4f})")

    return 0


if __name__ == "__main__":
    exit(main())