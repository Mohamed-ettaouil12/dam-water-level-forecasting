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

from src.config import SEED, DATA_PROCESSED, MLRUNS_DIR, REPORTS_DIR, DATA_SOURCE
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


def load_weather() -> pd.DataFrame:
    """Charge la météo pour les features exogènes."""
    weather_path = DATA_PROCESSED / "weather.csv"
    if not weather_path.exists():
        logger.warning("Fichier météo non trouvé : %s", weather_path)
        return pd.DataFrame()

    df = pd.read_csv(weather_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"bassin": "bassin_versant"})

    # Aggréger par bassin et date (moyenne sur les barrages du bassin)
    df = df.groupby(["bassin_versant", "date"], as_index=False).agg({
        "precip_mm": "mean",
        "temp_c": "mean",
    })
    logger.info("Météo chargée et agrégée par bassin : %d lignes", len(df))
    return df


def join_weather(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Joint les features météo au dataset des barrages."""
    if weather.empty:
        logger.warning("Météo vide, pas de jointure")
        return df

    # Gérer les suffixes _x/_y d'une jointure précédente
    basin_col = None
    for col in ["bassin_versant", "bassin_versant_x", "bassin_versant_y"]:
        if col in df.columns and df[col].notna().any():
            basin_col = col
            break

    # Si colonne bassin existe mais tout NaN, ou n'existe pas -> extraire du nom de barrage (synthétique)
    if basin_col is None or (basin_col in df.columns and df[basin_col].isna().all()):
        # Pour synthétique : extraire le bassin du nom du barrage
        # Format: "generic_Draa_55" -> "Draa", "generic_Oum Er-Rbia_0" -> "Oum Er-Rbia"
        if "barrage" in df.columns:
            logger.info("Extraction du bassin depuis le nom du barrage...")
            # Format: generic_{basin}_{id} ou generic_{basin}_{id} avec espaces dans le bassin
            def extract_basin(name: str) -> str:
                if name.startswith("generic_"):
                    parts = name.split("_")
                    # Rejoindre les parties du milieu (le bassin peut avoir des espaces)
                    # Format: generic_{basin_part1}_{basin_part2}..._{id}
                    if len(parts) >= 3:
                        return "_".join(parts[1:-1]).replace("_", " ")
                return "Unknown"

            df["bassin_versant"] = df["barrage"].apply(extract_basin)
            logger.info("Bassin extrait du nom barrage pour %d lignes, valeurs uniques: %s",
                        len(df), df["bassin_versant"].unique())
            basin_col = "bassin_versant"
        else:
            logger.warning("Pas de colonne barrage, tentative join dam_features")
            dams = load_dam_features()
            dams_subset = dams[["nom", "bassin_versant"]].drop_duplicates()
            df = df.merge(dams_subset, left_on="barrage", right_on="nom", how="left")
            basin_col = "bassin_versant"
    else:
        # Renommer la colonne trouvée en bassin_versant
        if basin_col != "bassin_versant":
            df["bassin_versant"] = df[basin_col]

    logger.info("bassin_versant présent: %s, valeurs uniques: %s",
                "bassin_versant" in df.columns, df["bassin_versant"].unique() if "bassin_versant" in df.columns else "N/A")

    df = df.merge(
        weather[["date", "bassin_versant", "precip_mm", "temp_c"]],
        on=["date", "bassin_versant"],
        how="left",
    )

    missing_precip = df["precip_mm"].isna().sum()
    missing_temp = df["temp_c"].isna().sum()
    logger.info("Jointure météo : %d lignes, precip manquants=%d, temp manquants=%d",
                len(df), missing_precip, missing_temp)
    return df


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

        if "precip_mm" in group.columns:
            for window in [3, 6]:
                col = f"precip_roll_sum_{window}"
                df.loc[idx, col] = group["precip_mm"].rolling(window, min_periods=1).sum().values
                feature_cols.append(col)

        if "temp_c" in group.columns:
            col = "temp_roll_mean_3"
            df.loc[idx, col] = group["temp_c"].rolling(3, min_periods=1).mean().values
            feature_cols.append(col)

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


def naive_seasonal_forecast(test_df: pd.DataFrame, train_df: pd.DataFrame, horizon: int) -> np.ndarray:
    """Prévision naïve saisonnière = même mois année précédente (depuis train)."""
    test_df = test_df.sort_values(["barrage", "date"])
    train_df = train_df.sort_values(["barrage", "date"])
    preds = []
    for barrage, test_group in test_df.groupby("barrage"):
        train_group = train_df[train_df["barrage"] == barrage].sort_values("date")
        for i in range(len(test_group)):
            target_date = test_group.iloc[i]["date"]
            ref_date = target_date - pd.DateOffset(years=1)
            ref_row = train_group[train_group["date"] == ref_date]
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


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict = None,
) -> xgb.XGBRegressor:
    """Entraîne un modèle XGBoost."""
    if params is None:
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
    return model


def evaluate_window(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    horizon: int,
) -> dict:
    """Évalue sur une fenêtre walk-forward."""
    # Debug: voir quelles colonnes ont des NaN
    logger.info("  Colonnes features: %s", feature_cols)
    logger.info("  NaN dans train par feature:")
    for col in feature_cols:
        if col in train.columns:
            nan_count = train[col].isna().sum()
            if nan_count > 0:
                logger.info("    %s: %d NaN", col, nan_count)
        else:
            logger.info("    %s: MANQUANTE", col)

    train_clean = train.dropna(subset=feature_cols + [target_col])
    test_clean = test.dropna(subset=feature_cols + [target_col])

    logger.info("  train=%d -> train_clean=%d, test=%d -> test_clean=%d",
                len(train), len(train_clean), len(test), len(test_clean))

    if len(train_clean) < 50 or len(test_clean) < 5:
        logger.warning("  Pas assez de données: train_clean=%d, test_clean=%d", len(train_clean), len(test_clean))
        return None

    X_train = train_clean[feature_cols]
    y_train = train_clean[target_col]
    X_test = test_clean[feature_cols]
    y_test = test_clean[target_col]

    model = train_xgboost(X_train, y_train)
    y_pred = model.predict(X_test)

    naive_pred = naive_seasonal_forecast(test_clean, train_clean, horizon)

    metrics = {
        "MAE": mae(y_test.values, y_pred),
        "RMSE": rmse(y_test.values, y_pred),
        "MASE": mase(y_test.values, y_pred, train_clean["niveau_pct"].values, SEASONALITY),
        "MAE_naive": mae(y_test.values, naive_pred),
        "RMSE_naive": rmse(y_test.values, naive_pred),
        "MASE_naive": mase(y_test.values, naive_pred, train_clean["niveau_pct"].values, SEASONALITY),
    }

    return metrics


def run_baseline() -> pd.DataFrame:
    """Lance la baseline complète."""
    logger.info("=== Début baseline XGBoost walk-forward (source=%s) ===", DATA_SOURCE)

    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR.absolute()}")
    mlflow.set_experiment("barrages-baseline-xgboost")

    df = load_data_by_source()
    weather = load_weather()
    df = join_weather(df, weather)

    # Pour synthétique : remplir features statiques manquantes
    if "scenario" in df.columns:
        logger.info("Vérification features statiques: capacite_max_m3 NaN=%s", df["capacite_max_m3"].isna().all())
        if df["capacite_max_m3"].isna().all():
            logger.info("Remplissage features statiques depuis scenario...")
            profile_map = {
                "grand": {"capacite_max_m3": 2_650_000_000, "hauteur_m": 102, "annee_mise_en_service": 1979},
                "moyen_grand": {"capacite_max_m3": 1_200_000_000, "hauteur_m": 80, "annee_mise_en_service": 1950},
                "moyen": {"capacite_max_m3": 600_000_000, "hauteur_m": 55, "annee_mise_en_service": 1970},
                "petit_moyen": {"capacite_max_m3": 100_000_000, "hauteur_m": 35, "annee_mise_en_service": 1990},
                "petit": {"capacite_max_m3": 30_000_000, "hauteur_m": 25, "annee_mise_en_service": 2000},
            }
            for idx, row in df.iterrows():
                profile = row["scenario"].split("_")[0]
                if profile in profile_map:
                    for k, v in profile_map[profile].items():
                        df.at[idx, k] = v
            logger.info("Features statiques remplies. capacite_max_m3 NaN=%s", df["capacite_max_m3"].isna().all())
        else:
            logger.info("Features statiques déjà présentes")
    else:
        logger.info("Pas de scenario, pas de remplissage features statiques")

    df = prepare_features(df)[0]
    df = create_targets(df, HORIZONS)

    all_results = []

    for barrage in sorted(df["barrage"].unique()):
        df_barrage = df[df["barrage"] == barrage].copy()

        feature_cols = [c for c in df.columns if c.startswith(("niveau_", "month_", "precip_", "temp_", "capacite", "hauteur", "annee"))]
        feature_cols = [c for c in feature_cols if c in df_barrage.columns]

        for horizon in HORIZONS:
            target_col = f"target_h{horizon}"
            if target_col not in df_barrage.columns:
                continue

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

                    metrics = evaluate_window(train, test, feature_cols, target_col, horizon)

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
                    logger.info("  XGB: MAE=%.4f, RMSE=%.4f, MASE=%.4f", metrics["MAE"], metrics["RMSE"], metrics["MASE"])
                    logger.info("  Naive: MAE=%.4f, RMSE=%.4f, MASE=%.4f", metrics["MAE_naive"], metrics["RMSE_naive"], metrics["MASE_naive"])

            if window_results:
                mean_xgb = {k: np.mean([r[k] for r in window_results]) for k in window_results[0].keys()}
                std_xgb = {k: np.std([r[k] for r in window_results]) for k in window_results[0].keys()}
                mean_naive = {k: np.mean([r[k] for r in window_naive_results]) for k in window_naive_results[0].keys()}
                std_naive = {k: np.std([r[k] for r in window_naive_results]) for k in window_naive_results[0].keys()}

                for k in mean_xgb.keys():
                    all_results.append({
                        "barrage": barrage,
                        "horizon": horizon,
                        "metric": k,
                        "model": "XGBoost",
                        "mean": mean_xgb[k],
                        "std": std_xgb[k],
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

                logger.info("  XGB Moyenne ± std: MAE=%.4f±%.4f, RMSE=%.4f±%.4f, MASE=%.4f±%.4f",
                           mean_xgb["MAE"], std_xgb["MAE"],
                           mean_xgb["RMSE"], std_xgb["RMSE"],
                           mean_xgb["MASE"], std_xgb["MASE"])
                logger.info("  Naive Moyenne ± std: MAE=%.4f±%.4f, RMSE=%.4f±%.4f, MASE=%.4f±%.4f",
                           mean_naive["MAE"], std_naive["MAE"],
                           mean_naive["RMSE"], std_naive["RMSE"],
                           mean_naive["MASE"], std_naive["MASE"])

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

    print("\n=== RÉSUMÉ BASELINE XGBOOST vs NAIVE ===")
    pivot = results.pivot_table(
        index=["barrage", "horizon", "model"],
        columns="metric",
        values=["mean", "std"]
    )
    print(pivot.to_string())

    print("\n=== VERDICT (XGBoost vs Naive) ===")
    xgb_mae = results[(results["model"] == "XGBoost") & (results["metric"] == "MAE")]["mean"].mean()
    naive_mae = results[(results["model"] == "Naive_Seasonal") & (results["metric"] == "MAE")]["mean"].mean()
    xgb_mase = results[(results["model"] == "XGBoost") & (results["metric"] == "MASE")]["mean"].mean()
    naive_mase = results[(results["model"] == "Naive_Seasonal") & (results["metric"] == "MASE")]["mean"].mean()

    print(f"XGBoost MAE moyen: {xgb_mae:.4f}")
    print(f"Naive MAE moyen:   {naive_mae:.4f}")
    print(f"XGBoost MASE moyen: {xgb_mase:.4f}")
    print(f"Naive MASE moyen:   {naive_mase:.4f}")

    if xgb_mae < naive_mae and xgb_mase < naive_mase:
        print("✅ SUCCÈS : XGBoost bat la naïve sur MAE et MASE")
    elif xgb_mae < naive_mae:
        print("⚠️ PARTIEL : XGBoost bat la naïve sur MAE mais pas MASE")
    else:
        print("❌ ÉCHEC : XGBoost ne bat pas la naïve - BUG à corriger")

    return 0


if __name__ == "__main__":
    exit(main())