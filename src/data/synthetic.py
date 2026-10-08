"""Générateur synthétique physique de séries de niveaux de barrages piloté par météo réelle."""

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.config import DATA_PROCESSED, DATA_RAW

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)

SYNTHETIC_DIR = DATA_PROCESSED / "synthetic"
SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"

WEATHER_CSV = DATA_PROCESSED / "weather.csv"

SEED = 42
N_SERIES = 100
N_YEARS = 25

DAM_PROFILES = {
    "grand": {"capacity_m3": 2_650_000_000, "height_m": 102, "n_series": 25, "basins": ["Oum Er-Rbia"],
              "irrigation_m3": {1: 2, 2: 3, 3: 8, 4: 15, 5: 22, 6: 28, 7: 30, 8: 28, 9: 15, 10: 6, 11: 2, 12: 1},
              "drinking_m3": 1.5, "baseflow_mult": 1.0},
    "moyen_grand": {"capacity_m3": 1_200_000_000, "height_m": 80, "n_series": 15, "basins": ["Inaouen", "Moulouya"],
                    "irrigation_m3": {1: 1, 2: 2, 3: 5, 4: 10, 5: 15, 6: 20, 7: 22, 8: 20, 9: 10, 10: 4, 11: 1, 12: 1},
                    "drinking_m3": 0.8, "baseflow_mult": 1.0},
    "moyen": {"capacity_m3": 600_000_000, "height_m": 55, "n_series": 15, "basins": ["Draa"],
              "irrigation_m3": {1: 0.5, 2: 1, 3: 3, 4: 6, 5: 10, 6: 14, 7: 15, 8: 14, 9: 6, 10: 2, 11: 0.5, 12: 0.5},
              "drinking_m3": 0.4, "baseflow_mult": 1.0},
    "petit_moyen": {"capacity_m3": 100_000_000, "height_m": 35, "n_series": 10, "basins": ["Tensift"],
                    "irrigation_m3": {1: 0.1, 2: 0.2, 3: 0.5, 4: 1.0, 5: 1.8, 6: 2.5, 7: 2.8, 8: 2.5, 9: 1.0, 10: 0.4, 11: 0.1, 12: 0.1},
                    "drinking_m3": 0.1, "baseflow_mult": 1.0},
    "petit": {"capacity_m3": 30_000_000, "height_m": 25, "n_series": 20, "basins": ["Tensift"],
              "irrigation_m3": {1: 0.06, 2: 0.1, 3: 0.4, 4: 0.8, 5: 1.3, 6: 1.8, 7: 2.0, 8: 1.8, 9: 0.8, 10: 0.25, 11: 0.06, 12: 0.06},
              "drinking_m3": 0.06, "baseflow_mult": 1.0},
}

RUNOFF_COEFF_BASE = {
    "Oum Er-Rbia": 40.0,
    "Inaouen": 18.0,
    "Moulouya": 20.0,
    "Draa": 9.0,
    "Tensift": 6.0,
}

BASEFLOW_BASE = {
    "Oum Er-Rbia": 50.0,
    "Inaouen": 40.0,
    "Moulouya": 30.0,
    "Draa": 18.0,
    "Tensift": 10.0,
}

DRINKING_WATER_MM3 = 0.8
CRISIS_RELEASE_PROB = 0.01
CRISIS_RELEASE_MAGNITUDE = (2, 5)

EVAP_COEFF = 0.0005


def load_basins() -> pd.DataFrame:
    """Charge les bassins versants depuis configs/barrages.yaml."""
    yaml_path = CONFIGS_DIR / "barrages.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    df = pd.DataFrame(data["barrages"])
    return df[["nom", "bassin_versant", "latitude", "longitude", "capacite_max_m3"]].drop_duplicates()


def load_weather() -> pd.DataFrame:
    """Charge la météo (réelle ou simulée) depuis data/processed/weather.csv et agrège par bassin/mois."""
    if not WEATHER_CSV.exists():
        raise FileNotFoundError(f"Fichier météo non trouvé : {WEATHER_CSV}. Lancez d'abord src.data.era5")
    df = pd.read_csv(WEATHER_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"bassin": "bassin_versant"})

    df = df.groupby(["bassin_versant", "date"], as_index=False).agg({
        "precip_mm": "mean",
        "temp_c": "mean",
        "source": "first"
    })
    df = df.sort_values(["bassin_versant", "date"]).reset_index(drop=True)

    logger.info("Météo chargée : %d lignes, %d bassins, source=%s",
                len(df), df["bassin_versant"].nunique(), df["source"].iloc[0])
    return df


def get_surface_area(level_pct: float, capacity_m3: float, height_m: float) -> float:
    """Estime la surface du plan d'eau (km²) en fonction du niveau."""
    if level_pct <= 0:
        return 0.0
    full_surface = (capacity_m3 / 1e6) / (height_m / 1000) * 0.6
    return full_surface * (level_pct / 100) ** 0.7


def evaporation_mm(temp_c: float, surface_km2: float) -> float:
    """Calcule l'évaporation en mm/mois (formule simplifiée Hargreaves)."""
    if surface_km2 <= 0:
        return 0.0
    t_max = temp_c + 5
    t_min = temp_c - 5
    ra = 15.0
    et0 = 0.0023 * (temp_c + 17.8) * (t_max - t_min) ** 0.5 * ra
    return et0 * 30 * EVAP_COEFF


def simulate_single_series(
    weather_df: pd.DataFrame,
    dam_profile: dict,
    basin: str,
    series_id: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simule une série unique de niveau de barrage."""

    capacity = dam_profile["capacity_m3"]
    height = dam_profile["height_m"]

    basin_weather = weather_df[weather_df["bassin_versant"] == basin].copy()
    if basin_weather.empty:
        raise ValueError(f"Aucune météo pour le bassin {basin}")

    basin_weather = basin_weather.sort_values("date").reset_index(drop=True)

    n_months = len(basin_weather)
    dates = pd.to_datetime(basin_weather["date"])
    precip = basin_weather["precip_mm"].values
    temp = basin_weather["temp_c"].values

    runoff_coeff = RUNOFF_COEFF_BASE[basin] * rng.uniform(0.85, 1.15)
    baseflow = BASEFLOW_BASE[basin] * dam_profile.get("baseflow_mult", 1.0) * rng.uniform(0.8, 1.2)

    irrigation_profile = dam_profile.get("irrigation_m3", {})
    drinking_m3 = dam_profile.get("drinking_m3", 0.5)

    drought_years = rng.choice(range(N_YEARS + 5), size=rng.integers(1, 4), replace=False)
    drought_active = np.zeros(n_months + 60, dtype=bool)
    for dy in drought_years:
        start = dy * 12
        end = min(start + rng.integers(24, 48), n_months + 60)
        drought_active[start:end] = True

    S = capacity * rng.uniform(0.45, 0.65)
    niveau_pct_series = np.zeros(n_months + 60)

    for t in range(n_months + 60):
        month = dates[t % n_months].month

        p = precip[t % n_months]
        if drought_active[t]:
            p *= rng.uniform(0.4, 0.6)

        noise_factor = rng.lognormal(mean=0, sigma=0.12)
        apports = runoff_coeff * p * noise_factor + baseflow

        surface_km2 = get_surface_area(S / capacity * 100, capacity, height)
        evap = evaporation_mm(temp[t % n_months], surface_km2) * surface_km2 * 1e6

        irrigation = irrigation_profile.get(month, 0) * 1e6 * rng.uniform(0.9, 1.1)
        drinking = drinking_m3 * 1e6 * rng.uniform(0.95, 1.05)

        crisis = 0.0
        if rng.random() < CRISIS_RELEASE_PROB:
            crisis = rng.uniform(*CRISIS_RELEASE_MAGNITUDE) * 1e6

        prelevements = irrigation + drinking + crisis

        S_new = S + apports * 1e6 - evap - prelevements
        S_new = max(0, min(capacity, S_new))

        niveau_pct = (S_new / capacity) * 100
        niveau_pct_series[t] = niveau_pct

        S = S_new

    niveau_pct_series = niveau_pct_series[60:]

    df = pd.DataFrame({
        "serie_id": series_id,
        "date": dates,
        "barrage": f"{dam_profile.get('nom', 'generic')}_{basin}_{series_id}",
        "niveau_pct": np.round(niveau_pct_series, 2),
        "scenario": f"{dam_profile.get('type', 'generic')}_{basin}",
        "source": "synthetic"
    })

    return df


def generate_all_series() -> tuple[pd.DataFrame, dict]:
    """Génère toutes les séries synthétiques."""
    np.random.seed(SEED)

    basins_df = load_basins()
    weather_df = load_weather()

    unique_basins = basins_df["bassin_versant"].unique()
    logger.info("Génération pour bassins : %s", list(unique_basins))

    all_series = []
    metadata = {
        "generation_date": datetime.now().isoformat(),
        "seed": SEED,
        "n_series": N_SERIES,
        "n_years": N_YEARS,
        "weather_source": weather_df["source"].iloc[0],
        "series": {}
    }

    series_id = 0
    for profile_name, profile in DAM_PROFILES.items():
        profile["type"] = profile_name
        for basin in profile["basins"]:
            if basin not in unique_basins:
                logger.warning("Bassin %s absent des données météo, ignoré", basin)
                continue

            n_series = profile["n_series"]
            logger.info("Génération %d séries pour %s (%s)", n_series, profile_name, basin)

            for i in range(n_series):
                rng = np.random.Generator(np.random.PCG64(SEED + series_id))
                series_df = simulate_single_series(
                    weather_df, profile, basin, series_id, rng
                )
                all_series.append(series_df)

                metadata["series"][str(series_id)] = {
                    "profile": profile_name,
                    "basin": basin,
                    "capacity_m3": profile["capacity_m3"],
                    "height_m": profile["height_m"],
                    "runoff_coeff_base": RUNOFF_COEFF_BASE[basin],
                    "baseflow_base": BASEFLOW_BASE[basin],
                    "seed": int(SEED + series_id),
                    "n_months": len(series_df),
                    "mean_level": float(series_df["niveau_pct"].mean()),
                    "min_level": float(series_df["niveau_pct"].min()),
                    "max_level": float(series_df["niveau_pct"].max()),
                }
                series_id += 1

    final_df = pd.concat(all_series, ignore_index=True)
    final_df = final_df.sort_values(["serie_id", "date"]).reset_index(drop=True)

    metadata["total_series_generated"] = series_id
    metadata["total_rows"] = len(final_df)

    logger.info("Génération terminée : %d séries, %d lignes", series_id, len(final_df))
    return final_df, metadata


def save_outputs(df: pd.DataFrame, metadata: dict) -> tuple[Path, Path]:
    """Sauvegarde les sorties."""
    parquet_path = SYNTHETIC_DIR / "synthetic_dams.parquet"
    json_path = SYNTHETIC_DIR / "metadata.json"

    df.to_parquet(parquet_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("Sauvegardé : %s (%.2f MB)", parquet_path, parquet_path.stat().st_size / 1e6)
    logger.info("Sauvegardé : %s", json_path)
    return parquet_path, json_path


def plot_representative_series(df: pd.DataFrame) -> Path:
    """Génère un graphique de 5 séries représentatives."""
    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)

    profiles = df["scenario"].unique()
    selected = np.random.choice(profiles, size=min(5, len(profiles)), replace=False)

    for idx, scenario in enumerate(selected):
        ax = axes[idx]
        scenario_data = df[df["scenario"] == scenario]
        series_ids = scenario_data["serie_id"].unique()[:3]

        for sid in series_ids:
            s_data = scenario_data[scenario_data["serie_id"] == sid].sort_values("date")
            ax.plot(s_data["date"], s_data["niveau_pct"], alpha=0.7, linewidth=0.8)

        ax.set_ylabel("Niveau (%)")
        ax.set_title(f"Scénario : {scenario} (3 séries)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

    axes[-1].set_xlabel("Date")
    fig.suptitle("Séries synthétiques représentatives - Niveaux de barrages", fontsize=14)
    plt.tight_layout()

    plot_path = SYNTHETIC_DIR / "synthetic_preview.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Graphique sauvegardé : %s", plot_path)
    return plot_path


def print_summary(df: pd.DataFrame) -> None:
    """Affiche un résumé des statistiques."""
    print("\n" + "=" * 60)
    print("RÉSUMÉ GÉNÉRATION SYNTHÉTIQUE")
    print("=" * 60)
    print(f"Séries générées : {df['serie_id'].nunique()}")
    print(f"Lignes totales : {len(df)}")
    print(f"Période : {df['date'].min().date()} à {df['date'].max().date()}")
    print(f"Niveau moyen : {df['niveau_pct'].mean():.1f}%")
    print(f"Niveau min/max : {df['niveau_pct'].min():.1f}% / {df['niveau_pct'].max():.1f}%")
    print(f"Scénarios : {df['scenario'].nunique()}")
    print(f"Source météo : {df['source'].iloc[0]}")
    print("\nPar scénario :")
    for scenario in sorted(df["scenario"].unique()):
        sdf = df[df["scenario"] == scenario]
        print(f"  {scenario}: {sdf['serie_id'].nunique()} séries, "
              f"niveau moy={sdf['niveau_pct'].mean():.1f}%, "
              f"min={sdf['niveau_pct'].min():.1f}%, max={sdf['niveau_pct'].max():.1f}%")
    print("=" * 60)


def main() -> int:
    """Pipeline principal de génération synthétique."""
    logger.info("=== Début génération synthétique ===")

    if not WEATHER_CSV.exists():
        logger.error("Fichier météo manquant. Lancez : python -m src.data.era5")
        return 1

    df, metadata = generate_all_series()
    save_outputs(df, metadata)
    plot_representative_series(df)
    print_summary(df)

    logger.info("=== Fin génération synthétique ===")
    return 0


if __name__ == "__main__":
    exit(main())