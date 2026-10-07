"""Nettoyage et enrichissement des séries temporelles des barrages."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pandera as pa
from pandera.typing import DataFrame

from src.config import DATA_INTERIM, DATA_PROCESSED
from src.data.dam_features import load_dam_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

MAX_CONSECUTIVE_NA = 3
JUMP_THRESHOLD = 30.0


CleanSchema = pa.DataFrameSchema(
    {
        "date": pa.Column(pa.DateTime, nullable=False),
        "barrage": pa.Column(pa.String, nullable=False),
        "niveau_pct": pa.Column(pa.Float, nullable=True),
        "reserve_m3": pa.Column(pa.Float, nullable=True),
        "aberration_flag": pa.Column(pa.Bool, nullable=False),
        "capacite_max_m3": pa.Column(pa.Float, nullable=True),
        "hauteur_m": pa.Column(pa.Float, nullable=True),
        "annee_mise_en_service": pa.Column(pa.Int, nullable=True),
        "bassin_versant": pa.Column(pa.String, nullable=True),
        "latitude": pa.Column(pa.Float, nullable=True),
        "longitude": pa.Column(pa.Float, nullable=True),
    },
    strict=False,
)


def load_raw_data() -> DataFrame:
    """Charge les données brutes depuis data/interim/."""
    path = DATA_INTERIM / "barrages_raw.parquet"
    if not path.exists():
        logger.warning("Fichier non trouvé : %s", path)
        return pd.DataFrame()
    df = pd.read_parquet(path)
    logger.info("Chargé %s : %d lignes", path.name, len(df))
    return df


def interpolate_missing(df: DataFrame) -> tuple[DataFrame, int]:
    """Interpolation temporelle par barrage (max 3 mois consécutifs)."""
    df = df.copy()
    df = df.sort_values(["barrage", "date"]).reset_index(drop=True)

    interpolated_count = 0

    for barrage, group in df.groupby("barrage"):
        idx = group.index
        series = group["niveau_pct"].copy()
        dates = group["date"].copy()

        na_mask = series.isna()
        if not na_mask.any():
            continue

        na_groups = (na_mask != na_mask.shift()).cumsum()

        for grp_id in na_groups[na_mask].unique():
            grp_mask = na_groups == grp_id
            if grp_mask.sum() <= MAX_CONSECUTIVE_NA:
                grp_positions = np.where(grp_mask)[0]
                grp_start = grp_positions[0]
                grp_end = grp_positions[-1]

                left_idx = grp_start - 1
                right_idx = grp_end + 1

                if left_idx >= 0 and right_idx < len(series):
                    sub_series = series.iloc[left_idx:right_idx + 1].copy()
                    sub_dates = dates.iloc[left_idx:right_idx + 1]
                    sub_series.index = sub_dates
                    interpolated = sub_series.interpolate(method="time")
                    series.iloc[left_idx:right_idx + 1] = interpolated.values
                    interpolated_count += grp_mask.sum()

        df.loc[idx, "niveau_pct"] = series.values

    logger.info("Interpolation : %d valeurs remplies (max %d mois consécutifs)", interpolated_count, MAX_CONSECUTIVE_NA)
    return df, interpolated_count


def clip_out_of_bounds(df: DataFrame) -> tuple[DataFrame, int]:
    """Rejette valeurs < 0 ou > 100 → NaN + log."""
    df = df.copy()
    out_of_bounds = (df["niveau_pct"] < 0) | (df["niveau_pct"] > 100)
    count = out_of_bounds.sum()
    if count > 0:
        logger.warning("%d valeurs hors bornes [0, 100] mises à NaN", count)
        df.loc[out_of_bounds, "niveau_pct"] = np.nan
    else:
        logger.info("Aucune valeur hors bornes [0, 100]")
    return df, int(count)


def detect_jumps(df: DataFrame) -> DataFrame:
    """Ajoute aberration_flag si saut > 30 points en 1 mois pour même barrage."""
    df = df.copy()
    df = df.sort_values(["barrage", "date"]).reset_index(drop=True)

    df["aberration_flag"] = False

    for barrage, group in df.groupby("barrage"):
        idx = group.index
        if len(group) < 2:
            continue
        diff = group["niveau_pct"].diff().abs()
        jumps = diff > JUMP_THRESHOLD
        df.loc[idx[jumps], "aberration_flag"] = True

    jump_count = df["aberration_flag"].sum()
    logger.info("Aberrations détectées (saut > %.0f pts) : %d", JUMP_THRESHOLD, jump_count)
    return df


def join_static_features(df: DataFrame) -> DataFrame:
    """Joint les caractéristiques statiques des barrages."""
    df = df.copy()
    features = load_dam_features()

    feature_cols = [
        "nom", "capacite_max_m3", "hauteur_m", "annee_mise_en_service",
        "bassin_versant", "latitude", "longitude",
    ]
    features = features[feature_cols].rename(columns={"nom": "barrage"})

    df = df.merge(features, on="barrage", how="left")
    missing = df["capacite_max_m3"].isna().sum()
    if missing > 0:
        logger.warning("%d lignes sans caractéristiques statiques (barrage inconnu)", missing)
    else:
        logger.info("Jointure caractéristiques statiques : OK (%d barrages uniques)", df["barrage"].nunique())
    return df


def validate_and_save(df: DataFrame) -> Path:
    """Valide avec pandera et sauvegarde."""
    try:
        validated = CleanSchema.validate(df, lazy=True)
        logger.info("Validation pandera : OK")
    except pa.errors.SchemaErrors as e:
        logger.warning("Validation pandera : %d erreurs", len(e.failure_cases))
        validated = df

    output_path = DATA_PROCESSED / "barrages_clean.parquet"
    validated.to_parquet(output_path, index=False)
    logger.info("Sauvegardé : %s (%.2f MB)", output_path, output_path.stat().st_size / 1e6)
    return output_path


def main() -> int:
    """Pipeline de nettoyage complet."""
    logger.info("=== Début nettoyage données barrages ===")

    df = load_raw_data()
    if df.empty:
        logger.error("Aucune donnée à nettoyer")
        return 1

    initial_count = len(df)
    initial_na = df["niveau_pct"].isna().sum()

    df, interp_count = interpolate_missing(df)
    df, oob_count = clip_out_of_bounds(df)
    df = detect_jumps(df)
    df = join_static_features(df)

    final_na = df["niveau_pct"].isna().sum()
    logger.info("Résumé : lignes=%d, NA initial=%d, interpolées=%d, hors bornes=%d, NA final=%d",
                initial_count, initial_na, interp_count, oob_count, final_na)

    validate_and_save(df)
    logger.info("=== Fin nettoyage ===")
    return 0


if __name__ == "__main__":
    exit(main())