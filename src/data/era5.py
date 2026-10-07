"""Ingestion données ERA5 (précipitations, température) via CDS API."""

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr
from dotenv import load_dotenv

from src.config import DATA_PROCESSED, DATA_RAW
from src.data.dam_features import load_dam_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

ERA5_RAW_DIR = DATA_RAW / "era5"
ERA5_RAW_DIR.mkdir(parents=True, exist_ok=True)

CDS_API_URL = os.getenv("CDSAPI_URL", "https://cds.climate.copernicus.eu/api")
CDS_API_KEY = os.getenv("CDSAPI_KEY", "")

MOROCCO_BBOX = [36, -17, 21, 1]  # [N, W, S, E]
START_YEAR = 2000
END_YEAR = datetime.now().year


def get_cds_client() -> cdsapi.Client:
    """Crée le client CDS API."""
    if not CDS_API_KEY:
        raise ValueError(
            "CDSAPI_KEY non trouvé dans .env.\n"
            "1. Créez un compte gratuit sur https://cds.climate.copernicus.eu/\n"
            "2. Allez dans votre profil > API key\n"
            "3. Ajoutez dans .env : CDSAPI_KEY=UID:KEY"
        )
    return cdsapi.Client(url=CDS_API_URL, key=CDS_API_KEY)


def download_era5_monthly(
    client: cdsapi.Client,
    variable: str,
    year: int,
    output_path: Path,
    max_retries: int = 3,
) -> bool:
    """Télécharge une variable ERA5 mensuelle pour une année."""
    request = {
        "product_type": "monthly_averaged_reanalysis",
        "variable": variable,
        "year": str(year),
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": "00:00",
        "area": MOROCCO_BBOX,
        "format": "netcdf",
    }

    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Téléchargement %s %d (tentative %d/%d)", variable, year, attempt, max_retries)
            client.retrieve("reanalysis-era5-single-levels-monthly-means", request, str(output_path))
            logger.info("Succès : %s (%.2f MB)", output_path.name, output_path.stat().st_size / 1e6)
            return True
        except Exception as e:
            logger.warning("Échec tentative %d : %s", attempt, e)
            if attempt < max_retries:
                wait_time = 2 ** attempt
                logger.info("Attente %ds avant retry...", wait_time)
                time.sleep(wait_time)
            else:
                logger.error("Échec définitif pour %s %d", variable, year)
                if output_path.exists():
                    output_path.unlink()
                return False
    return False


def download_all_era5() -> list[Path]:
    """Télécharge précipitation et température pour toutes les années."""
    client = get_cds_client()
    downloaded = []

    variables = {
        "total_precipitation": "precip",
        "2m_temperature": "temp",
    }

    for var_name, var_short in variables.items():
        for year in range(START_YEAR, END_YEAR + 1):
            output_path = ERA5_RAW_DIR / f"era5_{var_short}_{year}.nc"
            if output_path.exists():
                logger.info("Déjà présent : %s", output_path.name)
                downloaded.append(output_path)
                continue

            if download_era5_monthly(client, var_name, year, output_path):
                downloaded.append(output_path)

    return downloaded


def aggregate_by_basin(era5_files: list[Path]) -> pd.DataFrame:
    """Agrège ERA5 par bassin versant (moyenne des points de grille)."""
    dams = load_dam_features()

    basins = dams[["nom", "bassin_versant", "latitude", "longitude"]].drop_duplicates()
    logger.info("Agrégation pour %d barrages dans %d bassins",
                len(basins), basins["bassin_versant"].nunique())

    all_data = []

    for var_short in ["precip", "temp"]:
        var_files = [f for f in era5_files if f"_{var_short}_" in f.name]
        if not var_files:
            logger.warning("Aucun fichier pour %s", var_short)
            continue

        ds_list = []
        for f in sorted(var_files):
            try:
                ds = xr.open_dataset(f)
                ds_list.append(ds)
            except Exception as e:
                logger.error("Erreur lecture %s : %s", f, e)

        if not ds_list:
            continue

        combined = xr.concat(ds_list, dim="time")
        combined = combined.sortby("time")

        var_data = combined[list(combined.data_vars)[0]]

        for _, dam in basins.iterrows():
            lat, lon = dam["latitude"], dam["longitude"]
            try:
                point_data = var_data.sel(latitude=lat, longitude=lon, method="nearest")
                df_point = point_data.to_dataframe().reset_index()
                df_point["barrage"] = dam["nom"]
                df_point["bassin_versant"] = dam["bassin_versant"]
                df_point = df_point.rename(columns={df_point.columns[-2]: var_short})
                all_data.append(df_point[["time", "barrage", "bassin_versant", var_short]])
            except Exception as e:
                logger.error("Erreur extraction point %s : %s", dam["nom"], e)

    if not all_data:
        return pd.DataFrame()

    precip_df = [d for d in all_data if "precip" in d.columns]
    temp_df = [d for d in all_data if "temp" in d.columns]

    if precip_df:
        precip = pd.concat(precip_df, ignore_index=True)
    else:
        precip = pd.DataFrame()

    if temp_df:
        temp = pd.concat(temp_df, ignore_index=True)
    else:
        temp = pd.DataFrame()

    if precip.empty or temp.empty:
        logger.warning("Données incomplètes : precip=%d, temp=%d", len(precip), len(temp))
        return pd.DataFrame()

    merged = precip.merge(temp, on=["time", "barrage", "bassin_versant"])
    merged = merged.rename(columns={"time": "date"})
    merged["date"] = pd.to_datetime(merged["date"])

    if "precip" in merged.columns:
        merged["precip_mm"] = merged["precip"] * 1000
    if "temp" in merged.columns:
        merged["temp_c"] = merged["temp"] - 273.15

    merged = merged[["date", "barrage", "bassin_versant", "precip_mm", "temp_c"]]
    merged = merged.sort_values(["barrage", "date"]).reset_index(drop=True)

    logger.info("Agrégation terminée : %d lignes, %d barrages", len(merged), merged["barrage"].nunique())
    return merged


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute features glissantes : précip cumulee 3/6 mois, temp moy 3 mois."""
    df = df.copy()
    df = df.sort_values(["barrage", "date"]).reset_index(drop=True)

    for barrage, group in df.groupby("barrage"):
        idx = group.index

        df.loc[idx, "precip_3m"] = group["precip_mm"].rolling(3, min_periods=1).sum().values
        df.loc[idx, "precip_6m"] = group["precip_mm"].rolling(6, min_periods=1).sum().values
        df.loc[idx, "temp_3m_avg"] = group["temp_c"].rolling(3, min_periods=1).mean().values

    return df


def join_with_dam_data(era5_df: pd.DataFrame) -> pd.DataFrame:
    """Joint les features ERA5 au dataset des barrages nettoyé."""
    dam_path = DATA_PROCESSED / "barrages_clean.parquet"
    if not dam_path.exists():
        logger.error("Fichier barrages_clean.parquet non trouvé. Lancez d'abord src.data.clean")
        return pd.DataFrame()

    dams = pd.read_parquet(dam_path)
    dams["date"] = pd.to_datetime(dams["date"])

    merged = dams.merge(
        era5_df,
        on=["date", "barrage"],
        how="left",
        validate="many_to_one",
    )

    missing_precip = merged["precip_mm"].isna().sum()
    missing_temp = merged["temp_c"].isna().sum()
    logger.info("Jointure : %d lignes, precip manquants=%d, temp manquants=%d",
                len(merged), missing_precip, missing_temp)

    return merged


def main() -> int:
    """Pipeline complet ERA5."""
    logger.info("=== Début ingestion ERA5 ===")

    if not CDS_API_KEY:
        logger.error("CDSAPI_KEY manquant dans .env")
        logger.info("Créez un compte gratuit sur https://cds.climate.copernicus.eu/")
        logger.info("Puis ajoutez dans .env : CDSAPI_KEY=VOTRE_UID:VOTRE_KEY")
        return 1

    era5_files = download_all_era5()
    if not era5_files:
        logger.error("Aucun fichier ERA5 téléchargé")
        return 1

    era5_agg = aggregate_by_basin(era5_files)
    if era5_agg.empty:
        logger.error("Agrégation échouée")
        return 1

    era5_feat = add_rolling_features(era5_agg)

    final_df = join_with_dam_data(era5_feat)
    if final_df.empty:
        logger.error("Jointure échouée")
        return 1

    output_path = DATA_PROCESSED / "dataset_complet.parquet"
    final_df.to_parquet(output_path, index=False)
    logger.info("Sauvegardé : %s (%.2f MB)", output_path, output_path.stat().st_size / 1e6)
    logger.info("=== Fin ingestion ERA5 ===")
    return 0


if __name__ == "__main__":
    exit(main())