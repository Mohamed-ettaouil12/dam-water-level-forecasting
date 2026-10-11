"""Évaluation Chronos LoRA fine-tuné (évaluation sur GPU)."""

import argparse
import logging
import warnings
from pathlib import Path
from typing import List, Tuple

import mlflow
import numpy as np
import pandas as pd
import torch
from peft import PeftModel

from src.config import DATA_PROCESSED, MLRUNS_DIR, REPORTS_DIR, DATA_SOURCE, SEED
from src.evaluation.metrics import mae, mase, rmse

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
CONTEXT_LENGTH = 240


def get_gpu_info() -> dict:
    """Retourne les infos GPU."""
    if not torch.cuda.is_available():
        return {"name": "CPU", "total_vram_gb": 0, "free_vram_gb": 0}

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    total = props.total_memory / 1e9
    free = (props.total_memory - torch.cuda.memory_reserved()) / 1e9
    return {
        "name": props.name,
        "total_vram_gb": round(total, 1),
        "free_vram_gb": round(free, 1),
    }


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


def load_pipeline_with_lora(
    horizon: int,
    device: str = "cuda",
) -> Tuple:
    """Charge le pipeline base + adaptateur LoRA."""
    import torch
    from chronos.chronos_bolt import BaseChronosPipeline
    from peft import PeftModel

    logger.info("Chargement modèle base Chronos-t5-small...")
    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map=device if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
    )

    # Charger adaptateur LoRA
    adapter_path = Path("models/chronos-lora") / f"horizon_{horizon}"
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adaptateur LoRA non trouvé : {adapter_path}")

    logger.info(f"Chargement adaptateur LoRA depuis {adapter_path}...")
    pipeline.model.model = PeftModel.from_pretrained(pipeline.model.model, adapter_path)
    pipeline.model.to(device if torch.cuda.is_available() else "cpu")
    pipeline.model.eval()

    logger.info("Modèle LoRA chargé et prêt pour l'inférence")
    return pipeline


def chronos_predict_series(
    pipeline,
    series: np.ndarray,
    horizon: int,
    device: str,
) -> np.ndarray:
    """Prédit avec Chronos LoRA pour une série (retourne médiane)."""
    import torch
    context = torch.tensor(series.astype(np.float32)).to(device)
    forecast = pipeline.predict(inputs=context, prediction_length=horizon)
    # forecast shape: [batch=1, num_samples, prediction_length]
    median_forecast = forecast[0, :, :].median(dim=0).values.cpu().numpy()
    return np.atleast_1d(median_forecast)


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
    device: str,
) -> dict:
    """Évalue sur une fenêtre walk-forward pour une série (rolling forecast)."""
    train_clean = train.dropna(subset=["niveau_pct"])
    test_clean = test.dropna(subset=["niveau_pct"])

    if len(train_clean) < 50 or len(test_clean) < 5:
        return None

    y_train = train_clean["niveau_pct"].values
    y_test = test_clean["niveau_pct"].values

    # Rolling forecast
    y_pred = []
    naive_pred = []

    for i in range(len(test_clean)):
        # Contexte = train + test précédents
        context_end_idx = len(train_clean) + i
        context = np.concatenate([y_train, test_clean["niveau_pct"].values[:i]])

        # Prédiction LoRA h mois en avant
        pred = chronos_predict_series(pipeline, context, horizon, device)
        if len(pred) >= horizon:
            y_pred.append(pred[horizon - 1])
        else:
            y_pred.append(np.nan)

        # Prévision naïve saisonnière (même mois année précédente, ajustée pour l'horizon)
        target_date = test_clean.iloc[i]["date"] + pd.DateOffset(months=horizon)
        ref_date = target_date - pd.DateOffset(years=1)
        ref_row = train_clean[train_clean["date"] == ref_date]
        if len(ref_row) > 0:
            naive_pred.append(ref_row["niveau_pct"].values[0])
        else:
            naive_pred.append(np.nan)

    y_pred = np.array(y_pred)
    naive_pred = np.array(naive_pred)
    y_test = test_clean["niveau_pct"].values

    # Métriques
    from src.evaluation.metrics import mae, mase, rmse

    mask = ~np.isnan(y_test) & ~np.isnan(y_pred)
    if not mask.any():
        return None

    train_clean_vals = train_clean["niveau_pct"].values
    naive_pred_arr = np.array([
        train_clean[train_clean["date"] == test_clean.iloc[i]["date"] + pd.DateOffset(months=horizon) - pd.DateOffset(years=1)]["niveau_pct"].values[0]
        if len(train_clean[train_clean["date"] == test_clean.iloc[i]["date"] + pd.DateOffset(months=horizon) - pd.DateOffset(years=1)]) > 0 else np.nan
        for i in range(len(test_clean))
    ])

    mask = ~np.isnan(y_test) & ~np.isnan(y_pred)
    if not mask.any():
        return None

    y_train = train_clean["niveau_pct"].values
    naive_pred_arr = np.array([
        train_clean[train_clean["date"] == test_clean.iloc[i]["date"] + pd.DateOffset(months=horizon) - pd.DateOffset(years=1)]["niveau_pct"].values[0]
        if len(train_clean[train_clean["date"] == test_clean.iloc[i]["date"] + pd.DateOffset(months=horizon) - pd.DateOffset(years=1)]) > 0 else np.nan
        for i in range(len(test_clean))
    ])

    metrics = {
        "MAE": mae(y_test[mask], np.array(y_pred)[mask]),
        "RMSE": rmse(y_test[mask], np.array(y_pred)[mask]),
        "MASE": mase(y_test[mask], np.array(y_pred)[mask], train_clean["niveau_pct"].values, 12),
        "MAE_naive": mae(y_test, naive_pred_arr),
        "RMSE_naive": rmse(y_test, naive_pred_arr),
        "MASE_naive": mase(y_test, naive_pred_arr, train_clean["niveau_pct"].values, 12),
    }

    return metrics


def load_xgboost_results() -> pd.DataFrame:
    """Recharge les résultats XGBoost pour comparaison."""
    path = Path("reports") / "metrics_xgboost.csv"
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        return df
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_chronos_results() -> pd.DataFrame:
    """Recharge les résultats Chronos zero-shot."""
    path = Path("reports") / "metrics_chronos_zeroshot.csv"
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        return df
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def create_final_comparison(
    lora_df: pd.DataFrame,
    chronos_df: pd.DataFrame,
    xgb_df: pd.DataFrame,
) -> pd.DataFrame:
    """Crée le tableau comparatif final des 4 modèles."""
    comparison = []

    # Chronos LoRA
    lora_mase = lora_df[(lora_df["metric"] == "MASE") & (lora_df["model"] == "Chronos_LoRA")]
    naive_df = lora_df[lora_df["model"] == "Naive_Seasonal"]

    # Recharge Chronos zero-shot
    chronos_zs = pd.read_csv("reports/metrics_chronos_zeroshot.csv")
    zs_mase = chronos_zs[(chronos_zs["metric"] == "MASE") & (chronos_zs["model"] == "Chronos_ZeroShot")]

    # XGBoost
    xgb_df = pd.read_csv("reports/metrics_xgboost.csv") if Path("reports/metrics_xgboost.csv").exists() else pd.DataFrame()
    if not xgb_df.empty:
        xgb_mase = xgb_df[(xgb_df["metric"] == "MASE") & (xgb_df["model"] == "XGBoost")]
    else:
        xgb_mase = pd.DataFrame()

    all_models = {
        "Chronos_LoRA": lora_df[(lora_df["model"] == "Chronos_LoRA") & (lora_df["metric"] == "MASE")],
        "Chronos_ZeroShot": chronos_zs[(chronos_zs["metric"] == "MASE") & (chronos_zs["model"] == "Chronos_ZeroShot")],
        "XGBoost": xgb_mase,
        "Naive_Seasonal": naive_df[naive_df["metric"] == "MASE"],
    }

    comparison = []
    for model_name, model_df in all_models.items():
        if model_df.empty:
            continue
        for h in [1, 3, 6]:
            hdf = model_df[model_df["horizon"] == h]
            if len(hdf) > 0:
                comparison.append({
                    "model": model_name,
                    "horizon": h,
                    "MASE_mean": round(hdf["mean"].mean(), 4),
                    "MASE_std": round(hdf["std"].mean(), 4),
                })

    return pd.DataFrame(comparison)


def main(
    horizon: int = 1,
    device: str = "auto",
    seed: int = SEED,
) -> int:
    """Point d'entrée principal."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    gpu_info = get_gpu_info()
    logger.info(f"Device: {device}")
    logger.info(f"GPU: {gpu_info['name']}")
    logger.info(f"VRAM totale: {gpu_info['total_vram_gb']} GB")
    logger.info(f"VRAM libre: {gpu_info['free_vram_gb']} GB")

    # MLflow
    import os
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR.absolute()}")
    mlflow.set_experiment("barrages-chronos-lora")

    # Chargement données
    df = load_data_by_source()

    # Chargement pipeline LoRA
    pipeline = load_pipeline_with_lora(horizon, device)
    device_obj = torch.device(device)

    # MLflow
    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR.absolute()}")
    mlflow.set_experiment("barrages-chronos-lora")

    all_results = []

    for barrage in sorted(df["barrage"].unique()):
        df_barrage = df[df["barrage"] == barrage].copy()

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
                mlflow.log_param("model", "Chronos_LoRA")

                metrics = evaluate_window(pipeline, train, test, horizon, device)

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
                logger.info("  LoRA: MAE=%.4f, RMSE=%.4f, MASE=%.4f", metrics["MAE"], metrics["RMSE"], metrics["MASE"])
                logger.info("  Naive: MAE=%.4f, RMSE=%.4f, MASE=%.4f", metrics["MAE_naive"], metrics["RMSE_naive"], metrics["MASE_naive"])

        if window_results:
            mean_lora = {k: np.mean([r[k] for r in window_results]) for k in window_results[0].keys()}
            std_lora = {k: np.std([r[k] for r in window_results]) for k in window_results[0].keys()}
            mean_naive = {k: np.mean([r[k] for r in window_naive_results]) for k in window_naive_results[0].keys()}
            std_naive = {k: np.std([r[k] for r in window_naive_results]) for k in window_naive_results[0].keys()}

            for k in mean_lora.keys():
                all_results.append({
                    "barrage": barrage,
                    "horizon": horizon,
                    "metric": k,
                    "model": "Chronos_LoRA",
                    "mean": mean_lora[k],
                    "std": std_lora[k],
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

            logger.info("  LoRA Moyenne ± std: MAE=%.4f±%.4f, RMSE=%.4f±%.4f, MASE=%.4f±%.4f",
                       mean_lora["MAE"], std_lora["MAE"],
                       mean_lora["RMSE"], std_lora["RMSE"],
                       mean_lora["MASE"], std_lora["MASE"])
            logger.info("  Naive Moyenne ± std: MAE=%.4f±%.4f, RMSE=%.4f±%.4f, MASE=%.4f±%.4f",
                       mean_naive["MAE"], std_naive["MAE"],
                       mean_naive["RMSE"], std_naive["RMSE"],
                       mean_naive["MASE"], std_naive["MASE"])

    # Sauvegarde résultats LoRA
    results_df = pd.DataFrame(all_results)
    output_path = Path("reports") / "metrics_chronos_lora.csv"
    results_df.to_csv(output_path, index=False)
    logger.info("Résultats LoRA sauvegardés: %s", output_path)

    # Tableau comparatif final
    chronos_df = pd.read_csv("reports/metrics_chronos_zeroshot.csv")
    xgb_df = pd.read_csv("reports/metrics_xgboost.csv") if Path("reports/metrics_xgboost.csv").exists() else pd.DataFrame()

    comparison = create_final_comparison(
        pd.DataFrame(all_results),
        pd.read_csv("reports/metrics_chronos_zeroshot.csv"),
        pd.read_csv("reports/metrics_xgboost.csv") if Path("reports/metrics_xgboost.csv").exists() else pd.DataFrame(),
    )

    output_path = Path("reports") / "comparison_zero_vs_xgb.csv"
    comparison.to_csv(output_path, index=False)
    logger.info("Tableau comparatif sauvegardé: %s", output_path)

    print("\n=== TABLEAU COMPARATIF FINAL MASE (moyenne ± écart-type) ===")
    for horizon in [1, 3, 6]:
        print(f"\nHorizon {horizon} mois:")
        hdf = comparison[comparison["horizon"] == horizon]
        for _, row in hdf.iterrows():
            print(f"  {row['model']:20s} : MASE = {row['MASE_mean']:.4f} ± {row['MASE_std']:.4f}")

    # Verdict final
    print("\n=== VERDICT FINAL (4 modèles) ===")
    for h in [1, 3, 6]:
        hdf = comparison[comparison["horizon"] == h]
        best = hdf.loc[hdf["MASE_mean"].idxmin()]
        worst = hdf.loc[hdf["MASE_mean"].idxmax()]
        print(f"Horizon {h}m: Meilleur = {best['model']} (MASE={best['MASE_mean']:.4f}), Pire = {worst['model']} (MASE={worst['MASE_mean']:.4f})")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation Chronos LoRA fine-tuné")
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 3, 6])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    exit(main(
        horizon=args.horizon,
        device=args.device,
        seed=args.seed,
    ))