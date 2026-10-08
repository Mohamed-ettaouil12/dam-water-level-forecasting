"""Nettoyage et validation des données de barrages (réelles et synthétiques)."""

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pandera as pa
from pandera import Column, Check, DataFrameSchema

from src.config import DATA_INTERIM, DATA_PROCESSED, DATA_SOURCE
from src.data.dam_features import load_dam_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SCHEMA = DataFrameSchema(
    {
        "date": Column(pa.DateTime, nullable=False),
        "barrage": Column(pa.String, nullable=False, checks=Check.str_length(min_value=1)),
        "niveau_pct": Column(
            pa.Float,
            nullable=False,
            checks=Check.between(-1, 101),
        ),
        "source": Column(pa.String, nullable=False, checks=Check.isin(["real", "synthetic"])),
    },
    strict=False,
    coerce=True,
)

MAX_CONSECUTIVE_NAN = 3
MAX_JUMP_PCT = 30


def load_raw_data(source: Literal["real", "synthetic"]) -> pd.DataFrame:
    """Charge les données brutes selon la source."""
    if source == "real":
        path = DATA_INTERIM / "barrages_raw.parquet"
        logger.info("Chargement données RÉELLES depuis %s", path)
        if not path.exists():
            raise FileNotFoundError(f"Fichier réel non trouvé : {path}")
        df = pd.read_parquet(path)
        df["source"] = "real"
        if "reserve_m3" in df.columns:
            df = df[["date", "barrage", "niveau_pct", "source"]]
    elif source == "synthetic":
        path = DATA_PROCESSED / "synthetic" / "synthetic_dams.parquet"
        logger.info("Chargement données SYNTHÉTIQUES depuis %s", path)
        if not path.exists():
            raise FileNotFoundError(f"Fichier synthétique non trouvé : {path}. Lancez src.data.synthetic")
        df = pd.read_parquet(path)
        # Conserver scenario pour les features statiques
        df = df[["date", "barrage", "niveau_pct", "source", "scenario"]].copy()
    else:
        raise ValueError(f"Source inconnue : {source}. Attendu 'real' ou 'synthetic'")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["barrage", "date"]).reset_index(drop=True)
    logger.info("Données chargées : %d lignes, %d barrages", len(df), df["barrage"].nunique())
    return df


def ensure_monthly_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """Assure une fréquence mensuelle régulière par barrage (reindex)."""
    all_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="MS")
    barrages = df["barrage"].unique()
    full_index = pd.MultiIndex.from_product([barrages, all_dates], names=["barrage", "date"])
    df = df.set_index(["barrage", "date"]).reindex(full_index).reset_index()
    # Ensure barrage is a column, not index
    if "barrage" not in df.columns:
        df = df.reset_index()
    return df


def interpolate_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Interpole les valeurs manquantes par barrage (max 3 mois consécutifs)."""
    initial_nan = df["niveau_pct"].isna().sum()

    result_dfs = []
    for barrage, group in df.groupby("barrage"):
        g = group.copy()
        g = g.set_index("date")
        nan_groups = g["niveau_pct"].isna().astype(int).groupby((~g["niveau_pct"].isna()).cumsum()).sum()
        long_nan_groups = nan_groups[nan_groups > MAX_CONSECUTIVE_NAN].index

        mask_long_nan = g["niveau_pct"].isna() & (
            (~g["niveau_pct"].isna()).cumsum().isin(long_nan_groups)
        )

        g["niveau_pct"] = g["niveau_pct"].interpolate(method="time", limit=MAX_CONSECUTIVE_NAN)
        g.loc[mask_long_nan, "niveau_pct"] = np.nan
        g = g.reset_index()
        g["barrage"] = barrage
        result_dfs.append(g)

    df = pd.concat(result_dfs, ignore_index=True)

    final_nan = df["niveau_pct"].isna().sum()
    interpolated = int(initial_nan - final_nan)
    logger.info("Interpolation : %d valeurs interpolées, %d NaN restants (gaps > %d mois)",
                interpolated, final_nan, MAX_CONSECUTIVE_NAN)
    return df, interpolated


def reject_out_of_bounds(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Rejette les valeurs < 0 ou > 100 en les mettant à NaN."""
    mask = df["niveau_pct"].notna() & ((df["niveau_pct"] < 0) | (df["niveau_pct"] > 100))
    count = int(mask.sum())
    if count > 0:
        df.loc[mask, "niveau_pct"] = np.nan
        logger.warning("Rejet bornes : %d valeurs hors [0, 100] mises à NaN", count)
    else:
        logger.info("Rejet bornes : aucune valeur hors [0, 100]")
    return df, count


def flag_aberrations(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Marque les sauts anormaux (> 30 points/mois) par barrage."""
    df = df.copy()
    # Ensure barrage is a column
    if "barrage" not in df.columns and df.index.name == "barrage":
        df = df.reset_index()
    df["aberration_flag"] = False

    for barrage, group in df.groupby("barrage"):
        group = group.sort_values("date")
        diff = group["niveau_pct"].diff().abs()
        mask = diff > MAX_JUMP_PCT
        flagged = int(mask.sum())
        if flagged > 0:
            df.loc[group.index[mask], "aberration_flag"] = True
            logger.warning("Aberrations : %d sauts > %d points/mois pour barrage %s",
                          flagged, MAX_JUMP_PCT, barrage)
    total_flagged = int(df["aberration_flag"].sum())
    logger.info("Aberrations totales : %d flags", total_flagged)
    return df, total_flagged


def join_static_features(df: pd.DataFrame) -> pd.DataFrame:
    """Joint les caractéristiques statiques des barrages."""
    dams = load_dam_features()
    dams_subset = dams[["nom", "bassin_versant", "latitude", "longitude", "capacite_max_m3",
                        "hauteur_m", "annee_mise_en_service"]].drop_duplicates()

    df = df.merge(dams_subset, left_on="barrage", right_on="nom", how="left", validate="many_to_one")

    missing = df["bassin_versant"].isna().sum()
    if missing > 0:
        logger.warning("Jointure : %d lignes sans correspondance dans dam_features", missing)
    else:
        logger.info("Jointure : %d lignes, toutes avec caractéristiques statiques", len(df))
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Valide le schéma avec pandera."""
    try:
        SCHEMA.validate(df, lazy=True)
        logger.info("Validation pandera : OK")
    except pa.errors.SchemaErrors as e:
        logger.error("Échec validation pandera :\n%s", e.failure_cases.to_string())
        raise


def save_clean_data(df: pd.DataFrame, source: Literal["real", "synthetic"]) -> Path:
    """Sauvegarde les données nettoyées selon la source."""
    if source == "real":
        output_path = DATA_PROCESSED / "barrages_clean.parquet"
    else:
        output_path = DATA_PROCESSED / "synthetic" / "barrages_clean_synthetic.parquet"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Sauvegardé : %s (%.2f MB)", output_path, output_path.stat().st_size / 1e6)
    return output_path


def main() -> int:
    """Pipeline de nettoyage principal."""
    logger.info("=== Début nettoyage (source=%s) ===", DATA_SOURCE)

    df = load_raw_data(DATA_SOURCE)

    logger.info("--- Étape 1: Fréquence mensuelle ---")
    df = ensure_monthly_frequency(df)

    logger.info("--- Étape 2: Interpolation manquants ---")
    df, n_interp = interpolate_missing(df)

    logger.info("--- Étape 3: Rejet hors bornes ---")
    df, n_rejected = reject_out_of_bounds(df)

    logger.info("--- Étape 4: Détection aberrations ---")
    df, n_flags = flag_aberrations(df)

    logger.info("--- Étape 5: Jointure caractéristiques statiques ---")
    df = join_static_features(df)

    logger.info("--- Étape 6: Validation pandera ---")
    validate_schema(df)

    logger.info("--- Étape 7: Sauvegarde ---")
    save_clean_data(df, DATA_SOURCE)

    logger.info("=== Fin nettoyage ===")
    logger.info("Résumé : interpolées=%d, rejetées=%d, flags=%d, lignes_finales=%d",
                n_interp, n_rejected, n_flags, len(df))
    return 0


if __name__ == "__main__":
    exit(main())