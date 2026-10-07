"""Tests unitaires pour le nettoyage des données barrages."""

import numpy as np
import pandas as pd
import pytest

from src.data.clean import (
    CleanSchema,
    clip_out_of_bounds,
    detect_jumps,
    interpolate_missing,
    join_static_features,
)


def make_test_df():
    """Crée un DataFrame de test synthétique."""
    dates = pd.date_range("2024-01-01", periods=12, freq="MS")
    data = {
        "date": list(dates) * 2,
        "barrage": ["DamA"] * 12 + ["DamB"] * 12,
        "niveau_pct": [
            50, 52, np.nan, np.nan, 55, 56, np.nan, np.nan, np.nan, np.nan, 60, 61,
            30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
        ],
        "reserve_m3": [1e6] * 24,
    }
    return pd.DataFrame(data)


def test_interpolate_missing_max_3_months():
    """Interpolation jusqu'à 3 mois consécutifs, au-delà reste NaN."""
    df = make_test_df()
    df, count = interpolate_missing(df)

    damA = df[df["barrage"] == "DamA"]["niveau_pct"].values
    assert count == 2
    assert not np.isnan(damA[2])
    assert not np.isnan(damA[3])
    assert np.isnan(damA[6])
    assert np.isnan(damA[7])
    assert np.isnan(damA[8])
    assert np.isnan(damA[9])


def test_clip_out_of_bounds():
    """Valeurs < 0 ou > 100 deviennent NaN."""
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5, freq="MS"),
        "barrage": ["DamA"] * 5,
        "niveau_pct": [-5, 50, 105, 99, 0],
        "reserve_m3": [1e6] * 5,
    })
    df, count = clip_out_of_bounds(df)
    assert count == 2
    assert np.isnan(df.iloc[0]["niveau_pct"])
    assert np.isnan(df.iloc[2]["niveau_pct"])
    assert df.iloc[1]["niveau_pct"] == 50
    assert df.iloc[3]["niveau_pct"] == 99
    assert df.iloc[4]["niveau_pct"] == 0


def test_detect_jumps():
    """Flag True si saut > 30 points en 1 mois."""
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=6, freq="MS"),
        "barrage": ["DamA"] * 6,
        "niveau_pct": [50, 52, 85, 87, 55, 56],
        "reserve_m3": [1e6] * 6,
    })
    df = detect_jumps(df)
    flags = df["aberration_flag"].values
    assert flags[2] == True
    assert flags[4] == True
    assert flags[0] == False
    assert flags[1] == False
    assert flags[3] == False
    assert flags[5] == False


def test_join_static_features():
    """Jointure avec caractéristiques statiques (mock)."""
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=3, freq="MS"),
        "barrage": ["Al Massira"] * 3,
        "niveau_pct": [50, 51, 52],
        "reserve_m3": [1e6] * 3,
    })
    df = join_static_features(df)
    assert "capacite_max_m3" in df.columns
    assert "hauteur_m" in df.columns
    assert "annee_mise_en_service" in df.columns
    assert df["capacite_max_m3"].iloc[0] == 2650000000
    assert df["hauteur_m"].iloc[0] == 102
    assert df["annee_mise_en_service"].iloc[0] == 1979


def test_clean_schema_validation():
    """Validation pandera du schéma nettoyé."""
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        "barrage": ["Test", "Test"],
        "niveau_pct": [50.0, 51.0],
        "reserve_m3": [1e6, 1.02e6],
        "aberration_flag": [False, False],
        "capacite_max_m3": [1e9, 1e9],
        "hauteur_m": [50.0, 50.0],
        "annee_mise_en_service": [2000, 2000],
        "bassin_versant": ["Test", "Test"],
        "latitude": [30.0, 30.0],
        "longitude": [-5.0, -5.0],
    })
    validated = CleanSchema.validate(df)
    assert len(validated) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])