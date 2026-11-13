"""Conformal prediction intervals for all models with walk-forward validation."""

import argparse
import logging
import warnings
from pathlib import Path
from typing import List, Tuple, Dict

import mlflow
import numpy as np
import pandas as pd
import torch

from src.config import (
    DATA_PROCESSED,
    MLRUNS_DIR,
    REPORTS_DIR,
    DATA_SOURCE,
    SEED,
    DEVICE,
    HORIZONS,
)
from src.models.conformal import conformal_intervals, compute_coverage, compute_crps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

N_WINDOWS = 5
SEASONALITY = 12
TEST_SIZE = 12
CALIB_SIZE = 12

PRED_CACHE_PATH = DATA_PROCESSED / "preds_cache.parquet"
OUTPUT_PREDS_PATH = DATA_PROCESSED / "predictions.parquet"
OUTPUT_COVERAGE_PATH = REPORTS_DIR / "coverage.csv"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run test mode: 1 barrage, 1 horizon, 1 window")
    parser.add_argument("--barrage", type=str, default=None, help="Specific barrage to run")
    parser.add_argument("--horizon", type=int, default=None, help="Specific horizon to run")
    parser.add_argument("--window", type=int, default=None, help="Specific window to run")
    return parser.parse_args()


def load_data() -> pd.DataFrame:
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


def walk_forward_split(
    df: pd.DataFrame,
    n_windows: int = N_WINDOWS,
    horizon: int = 1,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Génère les splits walk-forward expanding window."""
    df = df.sort_values("date").reset_index(drop=True)
    dates = np.sort(df["date"].unique())

    n_dates = len(dates)
    min_train = n_dates - TEST_SIZE * n_windows

    if min_train <= 0:
        logger.warning("Pas assez de données pour %d fenêtres", n_windows)
        return []

    splits = []
    for i in range(n_windows):
        train_end_idx = min_train + i * TEST_SIZE
        test_start_idx = train_end_idx
        test_end_idx = min(train_end_idx + TEST_SIZE, n_dates)

        if test_start_idx >= test_end_idx:
            break

        train_dates = dates[:train_end_idx]
        test_dates = dates[test_start_idx:test_end_idx]

        train = df[df["date"].isin(train_dates)].copy()
        test = df[df["date"].isin(test_dates)].copy()

        if len(train) > 0 and len(test) > 0:
            splits.append((train, test))

    return splits


def naive_seasonal_forecast(
    train: pd.DataFrame,
    test: pd.DataFrame,
    horizon: int,
) -> np.ndarray:
    """Prévision naïve saisonnière = même mois année précédente."""
    train_clean = train.dropna(subset=["niveau_pct"])
    test_clean = test.dropna(subset=["niveau_pct"])

    preds = []
    for i in range(len(test_clean)):
        target_date = test_clean.iloc[i]["date"] + pd.DateOffset(months=horizon)
        ref_date = target_date - pd.DateOffset(years=1)
        ref_row = train_clean[train_clean["date"] == ref_date]
        if len(ref_row) > 0:
            preds.append(ref_row["niveau_pct"].values[0])
        else:
            preds.append(np.nan)
    return np.array(preds)


def prepare_xgboost_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Prépare les features pour XGBoost."""
    from src.models.baseline_xgboost import prepare_features, create_targets

    df = prepare_features(df)[0]
    df = create_targets(df, HORIZONS)
    return df


def get_xgboost_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> np.ndarray:
    """Entraîne XGBoost et prédit sur test."""
    import xgboost as xgb

    train_clean = train.dropna(subset=feature_cols + [target_col])
    test_clean = test.dropna(subset=feature_cols + [target_col])

    if len(train_clean) < 50 or len(test_clean) < 5:
        return np.full(len(test), np.nan)

    X_train = train_clean[feature_cols]
    y_train = train_clean[target_col]
    X_test = test_clean[feature_cols]

    params = {
        "n_estimators": 300,
        "max_depth": 6,
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
    return model.predict(X_test)


def get_chronos_zeroshot_pipeline():
    """Charge le pipeline Chronos zero-shot."""
    global _chronos_zs_pipeline
    if "_chronos_zs_pipeline" not in globals() or _chronos_zs_pipeline is None:
        from chronos.chronos_bolt import BaseChronosPipeline
        logger.info("Chargement modèle Chronos-Bolt (amazon/chronos-bolt-small)...")
        _chronos_zs_pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-small",
            device_map="cpu",
            torch_dtype=torch.bfloat16,
        )
        logger.info("Modèle Chronos ZeroShot chargé")
    return _chronos_zs_pipeline


def get_chronos_zeroshot_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    horizon: int,
) -> np.ndarray:
    """Prédit avec Chronos ZeroShot (rolling forecast)."""
    pipeline = get_chronos_zeroshot_pipeline()
    QUANTILE_IDX = 4

    train_clean = train.dropna(subset=["niveau_pct"])
    test_clean = test.dropna(subset=["niveau_pct"])

    if len(train_clean) < 50 or len(test_clean) < 5:
        return np.full(len(test), np.nan)

    y_train = train_clean["niveau_pct"].values
    y_pred = []

    for i in range(len(test_clean)):
        context = np.concatenate([y_train, test_clean["niveau_pct"].values[:i]])
        context_t = torch.tensor(context.astype(np.float32))
        forecast = pipeline.predict(inputs=context_t, prediction_length=horizon)
        median_forecast = forecast[0, QUANTILE_IDX, :].cpu().numpy()
        median_forecast = np.atleast_1d(median_forecast)
        if len(median_forecast) >= horizon:
            y_pred.append(median_forecast[horizon - 1])
        else:
            y_pred.append(np.nan)

    return np.array(y_pred)


def get_chronos_lora_pipeline(horizon: int):
    """Charge le pipeline Chronos LoRA."""
    global _chronos_lora_pipelines
    if "_chronos_lora_pipelines" not in globals():
        _chronos_lora_pipelines = {}

    if horizon not in _chronos_lora_pipelines:
        from chronos.chronos_bolt import BaseChronosPipeline
        from peft import PeftModel
        import peft.tuners.lora.layer as lora_layer

        # Monkey-patch
        def _patched_get_in_out_features(module):
            if hasattr(module, 'weight'):
                if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                    return module.in_features, module.out_features
                elif module.weight.dim() == 2:
                    return module.weight.shape[1], module.weight.shape[0]
            raise ValueError(f'Cannot get in/out features for {type(module)}')
        lora_layer._get_in_out_features = _patched_get_in_out_features

        logger.info(f"Chargement modèle Chronos LoRA horizon {horizon}...")
        pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small",
            device_map="cpu",
            torch_dtype=torch.bfloat16,
        )
        adapter_path = Path("models/chronos-lora") / f"horizon_{horizon}"
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adaptateur LoRA non trouvé : {adapter_path}")

        logger.info(f"Chargement adaptateur LoRA depuis {adapter_path}...")
        pipeline.model.model = PeftModel.from_pretrained(pipeline.model.model, adapter_path)

        if hasattr(pipeline.tokenizer, 'boundaries') and pipeline.tokenizer.boundaries.device.type != 'cpu':
            pipeline.tokenizer.boundaries = pipeline.tokenizer.boundaries.cpu()
        if hasattr(pipeline.tokenizer, 'centers') and pipeline.tokenizer.centers.device.type != 'cpu':
            pipeline.tokenizer.centers = pipeline.tokenizer.centers.cpu()

        pipeline.model = pipeline.model.to(DEVICE).eval()
        _chronos_lora_pipelines[horizon] = pipeline
        logger.info(f"Modèle LoRA horizon {horizon} chargé sur {DEVICE}")

    return _chronos_lora_pipelines[horizon]


def get_chronos_lora_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    horizon: int,
) -> np.ndarray:
    """Prédit avec Chronos LoRA (rolling forecast)."""
    pipeline = get_chronos_lora_pipeline(horizon)

    train_clean = train.dropna(subset=["niveau_pct"])
    test_clean = test.dropna(subset=["niveau_pct"])

    if len(train_clean) < 50 or len(test_clean) < 5:
        return np.full(len(test), np.nan)

    y_train = train_clean["niveau_pct"].values
    y_pred = []

    for i in range(len(test_clean)):
        context = np.concatenate([y_train, test_clean["niveau_pct"].values[:i]])
        context_t = torch.tensor(context.astype(np.float32))
        forecast = pipeline.predict(inputs=context_t, prediction_length=horizon)
        median_forecast = forecast[0, :, :].median(dim=0).values.cpu().numpy()
        median_forecast = np.atleast_1d(median_forecast)
        if len(median_forecast) >= horizon:
            y_pred.append(median_forecast[horizon - 1])
        else:
            y_pred.append(np.nan)

    return np.array(y_pred)


def generate_all_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Génère les prédictions pour tous les modèles, barrages, horizons, fenêtres."""
    from tqdm import tqdm

    # Préparer features XGBoost une seule fois
    logger.info("Préparation features XGBoost...")
    df_feat = prepare_xgboost_features(df)

    cache_rows = []

    barrages = sorted(df["barrage"].unique())

    for barrage in tqdm(barrages, desc="Barrages"):
        df_barrage = df_feat[df_feat["barrage"] == barrage].copy()

        feature_cols = [c for c in df_barrage.columns if c.startswith(("niveau_", "month_", "precip_", "temp_", "capacite", "hauteur", "annee"))]
        feature_cols = [c for c in feature_cols if c in df_barrage.columns]

        for horizon in HORIZONS:
            target_col = f"target_h{horizon}"
            if target_col not in df_barrage.columns:
                continue

            splits = walk_forward_split(df_barrage, N_WINDOWS, horizon)

            for window_idx, (train, test) in enumerate(splits):
                logger.info(f"  {barrage} h{horizon} w{window_idx}: train={len(train)}, test={len(test)}")

                # Prédictions XGBoost
                y_pred_xgb = get_xgboost_predictions(train, test, feature_cols, target_col)

                # Prédictions Chronos ZeroShot
                y_pred_zs = get_chronos_zeroshot_predictions(train, test, horizon)

                # Prédictions Chronos LoRA
                y_pred_lora = get_chronos_lora_predictions(train, test, horizon)

                # Prédictions Naive
                y_pred_naive = naive_seasonal_forecast(train, test, horizon)

                # Vraies valeurs
                y_true = test["niveau_pct"].values

                # Stocker
                for i in range(len(test)):
                    row = {
                        "barrage": barrage,
                        "horizon": horizon,
                        "window": window_idx,
                        "date": test.iloc[i]["date"],
                        "y_true": y_true[i],
                        "y_pred_xgb": y_pred_xgb[i] if i < len(y_pred_xgb) else np.nan,
                        "y_pred_zeroshot": y_pred_zs[i] if i < len(y_pred_zs) else np.nan,
                        "y_pred_lora": y_pred_lora[i] if i < len(y_pred_lora) else np.nan,
                        "y_pred_naive": y_pred_naive[i] if i < len(y_pred_naive) else np.nan,
                    }
                    cache_rows.append(row)

    cache_df = pd.DataFrame(cache_rows)
    logger.info(f"Cache généré : {len(cache_df)} lignes")
    return cache_df


def load_or_generate_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Charge le cache ou génère les prédictions."""
    if PRED_CACHE_PATH.exists():
        logger.info(f"Chargement cache depuis {PRED_CACHE_PATH}")
        cache_df = pd.read_parquet(PRED_CACHE_PATH)
        logger.info(f"Cache chargé : {len(cache_df)} lignes")
        return cache_df

    logger.info("Cache non trouvé, génération des prédictions...")
    cache_df = generate_all_predictions(df)
    PRED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cache_df.to_parquet(PRED_CACHE_PATH, index=False)
    logger.info(f"Cache sauvegardé : {PRED_CACHE_PATH}")
    return cache_df


def compute_calibration_residuals(
    cache_df: pd.DataFrame,
    barrage: str,
    horizon: int,
    window: int,
    model: str,
) -> np.ndarray:
    """Calcule les résidus sur les derniers 12 mois du TRAIN (calibration)."""
    # Recharger les données pour recréer les splits exacts
    df = load_data()
    df_barrage = df[df["barrage"] == barrage].copy()

    splits = walk_forward_split(df_barrage, N_WINDOWS, horizon)
    if window >= len(splits):
        return np.array([])

    train, _ = splits[window]

    # Derniers 12 mois du train = calibration
    train_clean = train.dropna(subset=["niveau_pct"]).sort_values("date")
    if len(train_clean) < CALIB_SIZE:
        logger.warning(f"Pas assez de données pour calibration: {len(train_clean)} < {CALIB_SIZE}")
        return np.array([])

    calib_data = train_clean.tail(CALIB_SIZE)

    # Pour la calibration, on a besoin des prédictions du modèle sur ces données
    # On recalcule les prédictions sur la calibration
    if model == "XGBoost":
        # Préparer features
        from src.models.baseline_xgboost import prepare_features, create_targets
        df_feat = prepare_features(calib_data)[0]
        df_feat = create_targets(df_feat, [horizon])
        target_col = f"target_h{horizon}"
        feature_cols = [c for c in df_feat.columns if c.startswith(("niveau_", "month_", "precip_", "temp_", "capacite", "hauteur", "annee"))]
        feature_cols = [c for c in feature_cols if c in df_feat.columns]

        # Entraîner sur train complet, prédire sur calibration
        full_train = train.dropna(subset=["niveau_pct"])
        full_train_feat = prepare_features(full_train)[0]
        full_train_feat = create_targets(full_train_feat, [horizon])

        preds = get_xgboost_predictions(full_train_feat, df_feat, feature_cols, target_col)
        y_true = calib_data["niveau_pct"].values
        return y_true - preds

    elif model == "Chronos_ZeroShot":
        # Rolling forecast sur calibration
        full_train = train.dropna(subset=["niveau_pct"]).sort_values("date")
        calib_dates = calib_data["date"].values
        preds = []
        for cal_date in calib_dates:
            # Contexte = données avant cal_date
            context = full_train[full_train["date"] < cal_date]["niveau_pct"].values
            if len(context) < 50:
                preds.append(np.nan)
                continue
            context_t = torch.tensor(context.astype(np.float32))
            pipeline = get_chronos_zeroshot_pipeline()
            QUANTILE_IDX = 4
            forecast = pipeline.predict(inputs=context_t, prediction_length=horizon)
            median_forecast = forecast[0, QUANTILE_IDX, :].cpu().numpy()
            median_forecast = np.atleast_1d(median_forecast)
            if len(median_forecast) >= horizon:
                preds.append(median_forecast[horizon - 1])
            else:
                preds.append(np.nan)
        preds = np.array(preds)
        y_true = calib_data["niveau_pct"].values
        return y_true - preds

    elif model == "Chronos_LoRA":
        full_train = train.dropna(subset=["niveau_pct"]).sort_values("date")
        calib_dates = calib_data["date"].values
        preds = []
        pipeline = get_chronos_lora_pipeline(horizon)
        for cal_date in calib_dates:
            context = full_train[full_train["date"] < cal_date]["niveau_pct"].values
            if len(context) < 50:
                preds.append(np.nan)
                continue
            context_t = torch.tensor(context.astype(np.float32))
            forecast = pipeline.predict(inputs=context_t, prediction_length=horizon)
            median_forecast = forecast[0, :, :].median(dim=0).values.cpu().numpy()
            median_forecast = np.atleast_1d(median_forecast)
            if len(median_forecast) >= horizon:
                preds.append(median_forecast[horizon - 1])
            else:
                preds.append(np.nan)
        preds = np.array(preds)
        y_true = calib_data["niveau_pct"].values
        return y_true - preds

    elif model == "Naive_Seasonal":
        # Prédiction naïve sur calibration
        preds = naive_seasonal_forecast(train, calib_data, horizon)
        y_true = calib_data["niveau_pct"].values
        return y_true - preds

    return np.array([])


def run_conformal_analysis(cache_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Applique la conformal prediction et calcule coverage + CRPS."""
    from tqdm import tqdm

    models = ["XGBoost", "Chronos_ZeroShot", "Chronos_LoRA", "Naive_Seasonal"]
    pred_cols = {
        "XGBoost": "y_pred_xgb",
        "Chronos_ZeroShot": "y_pred_zeroshot",
        "Chronos_LoRA": "y_pred_lora",
        "Naive_Seasonal": "y_pred_naive",
    }

    results_rows = []
    coverage_rows = []

    # Grouper par barrage, horizon, fenêtre
    grouped = cache_df.groupby(["barrage", "horizon", "window"])

    for (barrage, horizon, window), group in tqdm(grouped, desc="Conformal analysis"):
        group = group.sort_values("date")
        y_true = group["y_true"].values

        for model in models:
            pred_col = pred_cols[model]
            y_pred = group[pred_col].values

            # Skip si pas de prédictions
            if np.all(np.isnan(y_pred)):
                continue

            # Résidus de calibration
            calib_residuals = compute_calibration_residuals(cache_df, barrage, horizon, window, model)
            calib_residuals = calib_residuals[~np.isnan(calib_residuals)]

            if len(calib_residuals) < 10:
                logger.warning(f"Pas assez de résidus calibration pour {model} {barrage} h{horizon} w{window}: {len(calib_residuals)}")
                continue

            # Intervalles conformal
            low, high = conformal_intervals(calib_residuals, y_pred, coverage=0.90)

            # Coverage et CRPS sur test
            mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & ~np.isnan(low) & ~np.isnan(high)
            if not mask.any():
                continue

            y_true_valid = y_true[mask]
            y_pred_valid = y_pred[mask]
            low_valid = low[mask]
            high_valid = high[mask]

            coverage = compute_coverage(y_true_valid, low_valid, high_valid)
            crps = compute_crps(y_true_valid, y_pred_valid, high_valid, low_valid)

            # Sauvegarder prédictions avec intervalles
            for i, idx in enumerate(group.index):
                if mask[i]:
                    results_rows.append({
                        "barrage": barrage,
                        "horizon": horizon,
                        "window": window,
                        "date": group.loc[idx, "date"],
                        "model": model,
                        "y_true": y_true_valid[i],
                        "y_pred": y_pred_valid[i],
                        "y_pred_low": low_valid[i],
                        "y_pred_high": high_valid[i],
                    })

            # Sauvegarder métriques de couverture
            coverage_rows.append({
                "barrage": barrage,
                "horizon": horizon,
                "window": window,
                "model": model,
                "coverage": coverage,
                "crps": crps,
                "n_calib": len(calib_residuals),
                "n_test": mask.sum(),
            })

    results_df = pd.DataFrame(results_rows)
    coverage_df = pd.DataFrame(coverage_rows)

    return results_df, coverage_df


def main():
    args = parse_args()

    logger.info("=== Début conformal prediction ===")

    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR.absolute()}")
    mlflow.set_experiment("barrages-conformal")

    # 1. Charger données
    df = load_data()

    # Filtrer pour le mode test
    if args.test or args.barrage:
        test_barrage = args.barrage or "generic_Draa_55"
        test_horizon = args.horizon or 1
        test_window = args.window or 0
        logger.info(f"MODE TEST: barrage={test_barrage}, horizon={test_horizon}, window={test_window}")
        df = df[df["barrage"] == test_barrage].copy()

    # 2. Charger ou générer prédictions
    cache_df = load_or_generate_predictions(df)

    # Filtrer cache pour test
    if args.test or args.barrage:
        test_barrage = args.barrage or "generic_Draa_55"
        test_horizon = args.horizon or 1
        test_window = args.window or 0
        cache_df = cache_df[
            (cache_df["barrage"] == test_barrage) &
            (cache_df["horizon"] == test_horizon) &
            (cache_df["window"] == test_window)
        ].copy()
        logger.info(f"Cache filtré pour test: {len(cache_df)} lignes")

    # 3. Analyse conformal
    results_df, coverage_df = run_conformal_analysis(cache_df)

    # 4. Sauvegarder résultats
    OUTPUT_PREDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(OUTPUT_PREDS_PATH, index=False)
    logger.info(f"Prédictions avec intervalles sauvées : {OUTPUT_PREDS_PATH}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    coverage_df.to_csv(OUTPUT_COVERAGE_PATH, index=False)
    logger.info(f"Coverage sauvegardé : {OUTPUT_COVERAGE_PATH}")

    # 5. Résumé
    print("\n=== RÉSUMÉ COVERAGE (moyenne par modèle/horizon) ===")
    if not coverage_df.empty:
        summary = coverage_df.groupby(["model", "horizon"]).agg(
            coverage_mean=("coverage", "mean"),
            coverage_std=("coverage", "std"),
            crps_mean=("crps", "mean"),
            crps_std=("crps", "std"),
        ).reset_index()
        print(summary.to_string(index=False))

    logger.info("=== Fin conformal prediction ===")
    return 0


if __name__ == "__main__":
    exit(main())