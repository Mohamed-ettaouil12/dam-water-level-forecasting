"""Ingestion données ERA5 (précipitations, température) via CDS API avec fallback simulé."""

import logging
import os
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import cdsapi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from dotenv import load_dotenv

from src.config import DATA_PROCESSED, DATA_RAW

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

MOROCCO_BBOX = [36, -17, 21, 1]
START_YEAR = 2000
END_YEAR = datetime.now().year

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_basins() -> pd.DataFrame:
    """Charge les bassins versants depuis configs/barrages.yaml."""
    yaml_path = CONFIGS_DIR / "barrages.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    df = pd.DataFrame(data["barrages"])
    basins = df[["nom", "bassin_versant", "latitude", "longitude"]].drop_duplicates()
    return basins


def get_cds_client() -> cdsapi.Client:
    """Crée le client CDS API."""
    if not CDS_API_KEY or CDS_API_KEY == "colle_ta_cle_ici":
        raise ValueError("CDSAPI_KEY non trouvé ou invalide dans .env")
    return cdsapi.Client(url=CDS_API_URL, key=CDS_API_KEY)


def download_era5_monthly(
    client: cdsapi.Client,
    variable: str,
    year: int,
    output_path: Path,
    max_retries: int = 3,
) -> bool:
    """Télécharge une variable ERA5 mensuelle pour une année avec retry exponentiel."""
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
    """Agrège ERA5 par bassin versant (moyenne des points de grille proches)."""
    basins = load_basins()
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


def generate_simulated_weather() -> pd.DataFrame:
    """Génère météo simulée réaliste pour le Maroc (fallback sans clé CDS)."""
    basins = load_basins()
    unique_basins = basins["bassin_versant"].unique()
    logger.warning("=" * 60)
    logger.warning("MODE DE SECOURS ACTIVÉ : Génération météo simulée")
    logger.warning("Raison : CDSAPI_KEY absente ou invalide dans .env")
    logger.warning("Le projet continue normalement avec des données synthétiques")
    logger.warning("=" * 60)

    np.random.seed(42)
    dates = pd.date_range(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31", freq="MS")

    all_rows = []

    for bassin in unique_basins:
        basin_dams = basins[basins["bassin_versant"] == bassin]
        n_years = len(dates) // 12

        for _, dam in basin_dams.iterrows():
            lat = dam["latitude"]

            year_trend = np.linspace(0, 0.5, n_years)
            year_trend = np.repeat(year_trend, 12)[:len(dates)]

            for i, date in enumerate(dates):
                month = date.month
                year_idx = i // 12

                is_wet_season = month in [11, 12, 1, 2, 3]
                is_dry_season = month in [6, 7, 8, 9]

                if is_wet_season:
                    base_precip = np.random.gamma(shape=2.0, scale=15.0)
                    if year_idx > 0 and np.random.random() < 0.15:
                        drought_factor = np.random.uniform(0.2, 0.5)
                        base_precip *= drought_factor
                elif is_dry_season:
                    base_precip = np.random.exponential(scale=2.0)
                else:
                    base_precip = np.random.gamma(shape=1.5, scale=8.0)

                base_precip += np.random.normal(0, 3.0)
                precip_mm = max(0, base_precip)

                seasonal_temp = 10 + 15 * np.sin(2 * np.pi * (month - 3) / 12)
                lat_adjustment = (lat - 31) * 0.5
                temp_c = seasonal_temp + lat_adjustment + year_trend[i] + np.random.normal(0, 2.0)

                all_rows.append({
                    "date": date,
                    "bassin": bassin,
                    "precip_mm": round(precip_mm, 1),
                    "temp_c": round(temp_c, 1),
                    "source": "simulated_weather"
                })

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["bassin", "date"]).reset_index(drop=True)
    logger.info("Météo simulée générée : %d lignes, %d bassins", len(df), df["bassin"].nunique())
    return df


def export_weather_csv(df: pd.DataFrame, source: str) -> Path:
    """Exporte vers data/processed/weather.csv avec les colonnes requises."""
    output_df = df[["date", "bassin", "precip_mm", "temp_c", "source"]].copy()
    output_path = DATA_PROCESSED / "weather.csv"
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    logger.info("Exporté : %s (%.2f MB)", output_path, output_path.stat().st_size / 1e6)
    return output_path


def plot_preview(df: pd.DataFrame) -> None:
    """Affiche un graphique rapide matplotlib de la pluie par bassin."""
    try:
        fig, ax = plt.subplots(figsize=(12, 6))

        for bassin in df["bassin"].unique():
            basin_data = df[df["bassin"] == bassin].sort_values("date")
            ax.plot(basin_data["date"], basin_data["precip_mm"], label=bassin, alpha=0.7, linewidth=0.8)

        ax.set_title("Précipitations mensuelles par bassin (aperçu)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Précipitations (mm)")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = DATA_PROCESSED / "weather_preview.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info("Graphique sauvegardé : %s", plot_path)
    except Exception as e:
        logger.warning("Impossible de générer le graphique : %s", e)


def main() -> int:
    """Pipeline principal : essaie ERA5 réel, fallback simulé si échec."""
    logger.info("=== Début ingestion météo ===")

    use_real_era5 = False
    era5_df = pd.DataFrame()

    if CDS_API_KEY and CDS_API_KEY != "colle_ta_cle_ici":
        try:
            era5_files = download_all_era5()
            if era5_files:
                era5_agg = aggregate_by_basin(era5_files)
                if not era5_agg.empty:
                    era5_agg["source"] = "real_era5"
                    era5_agg = era5_agg.rename(columns={"bassin_versant": "bassin"})
                    era5_df = era5_agg[["date", "bassin", "precip_mm", "temp_c", "source"]]
                    use_real_era5 = True
        except Exception as e:
            logger.error("Erreur CDS API : %s", e)

    if not use_real_era5:
        era5_df = generate_simulated_weather()
        source_label = "simulée (pas de clé CDS — le projet continue)"
    else:
        source_label = "ERA5 réelle"

    if era5_df.empty:
        logger.error("Aucune donnée météo générée")
        return 1

    export_weather_csv(era5_df, source_label.split()[0])

    logger.info("\n" + "=" * 50)
    logger.info("RÉSUMÉ")
    logger.info("=" * 50)
    logger.info("Météo : %s", source_label)
    logger.info("Lignes : %d", len(era5_df))
    logger.info("Bassins : %d", era5_df["bassin"].nunique())
    logger.info("Période : %s à %s", era5_df["date"].min().date(), era5_df["date"].max().date())
    logger.info("Précip. moy : %.1f mm/mois", era5_df["precip_mm"].mean())
    logger.info("Temp. moy : %.1f °C", era5_df["temp_c"].mean())
    logger.info("=" * 50)

    print("\nAperçu (5 premières lignes) :")
    print(era5_df.head().to_string(index=False))

    print("\nAperçu (5 dernières lignes) :")
    print(era5_df.tail().to_string(index=False))

    plot_preview(era5_df)

    logger.info("=== Fin ingestion météo ===")
    return 0


if __name__ == "__main__":
    exit(main())