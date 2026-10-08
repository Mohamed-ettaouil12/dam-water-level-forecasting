"""Tests unitaires pour le nettoyage des données."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.clean import (
    ensure_monthly_frequency,
    interpolate_missing,
    reject_out_of_bounds,
    flag_aberrations,
    join_static_features,
    validate_schema,
    SCHEMA,
    MAX_CONSECUTIVE_NAN,
    MAX_JUMP_PCT,
)


@pytest.fixture
def sample_df_real():
    """Données réelles de test avec trous, bornes, aberrations."""
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    data = {
        "date": list(dates) * 2,
        "barrage": ["Al Massira"] * 24 + ["Bin el Ouidane"] * 24,
        "niveau_pct": [
            50, 52, np.nan, np.nan, 55, 56, 58, 60, 62, 65, 68, 70,
            72, 75, 78, 80, 82, 85, 88, 90, 92, 95, 98, 100,
            30, 32, 35, 38, 40, 42, 45, 48, 50, 52, 55, 58,
            60, 62, 65, 68, 70, 72, 75, 78, 80, 82, 85, 88,
        ],
        "source": ["real"] * 48,
    }
    df = pd.DataFrame(data)
    df.loc[5, "niveau_pct"] = -5
    df.loc[10, "niveau_pct"] = 105
    df.loc[15, "niveau_pct"] = 100
    df.loc[16, "niveau_pct"] = 60
    return df


@pytest.fixture
def sample_df_synthetic():
    """Données synthétiques de test."""
    dates = pd.date_range("2020-01-01", periods=12, freq="MS")
    data = {
        "date": list(dates) * 2,
        "barrage": ["Al Massira"] * 12 + ["Bin el Ouidane"] * 12,
        "niveau_pct": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 95] * 2,
        "source": ["synthetic"] * 24,
    }
    return pd.DataFrame(data)


def test_ensure_monthly_frequency(sample_df_real):
    """Test que la fréquence mensuelle est assurée."""
    df = ensure_monthly_frequency(sample_df_real)
    assert "date" in df.columns
    assert "barrage" in df.columns
    for barrage in df["barrage"].unique():
        bdf = df[df["barrage"] == barrage].sort_values("date")
        diffs = bdf["date"].diff().dropna()
        valid_diffs = [
            pd.Timedelta(days=31), pd.Timedelta(days=30), 
            pd.Timedelta(days=28), pd.Timedelta(days=29)
        ]
        assert all(d in valid_diffs for d in diffs)


def test_interpolate_missing(sample_df_real):
    """Test interpolation des manquants (max 3 mois)."""
    df = ensure_monthly_frequency(sample_df_real)
    df, n_interp = interpolate_missing(df)
    assert n_interp > 0
    assert df["niveau_pct"].isna().sum() >= 0
    for barrage in df["barrage"].unique():
        bdf = df[df["barrage"] == barrage].sort_values("date")
        nan_streaks = bdf["niveau_pct"].isna().astype(int)
        max_streak = 0
        current = 0
        for v in nan_streaks:
            if v == 1:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        assert max_streak <= MAX_CONSECUTIVE_NAN, f"Gap > {MAX_CONSECUTIVE_NAN} mois non géré"


def test_reject_out_of_bounds(sample_df_real):
    """Test rejet valeurs < 0 ou > 100."""
    df = ensure_monthly_frequency(sample_df_real)
    df, n_rejected = reject_out_of_bounds(df)
    assert n_rejected == 2
    valid_mask = df["niveau_pct"].notna()
    assert (df.loc[valid_mask, "niveau_pct"] >= -1).all()
    assert (df.loc[valid_mask, "niveau_pct"] <= 101).all()


def test_flag_aberrations(sample_df_real):
    """Test détection sauts > 30 points/mois."""
    df = ensure_monthly_frequency(sample_df_real)
    df, _ = interpolate_missing(df)
    df, _ = reject_out_of_bounds(df)
    df, n_flags = flag_aberrations(df)
    assert "aberration_flag" in df.columns
    assert n_flags >= 1
    assert df["aberration_flag"].dtype == bool


def test_join_static_features(sample_df_synthetic):
    """Test jointure avec caractéristiques statiques."""
    df = join_static_features(sample_df_synthetic)
    expected_cols = ["bassin_versant", "latitude", "longitude", "capacite_max_m3", "hauteur_m", "annee_mise_en_service"]
    for col in expected_cols:
        assert col in df.columns, f"Colonne manquante: {col}"
    assert df["bassin_versant"].notna().all()


def test_validate_schema_passes(sample_df_synthetic):
    """Test validation pandera passe pour données valides."""
    df = join_static_features(sample_df_synthetic)
    df = ensure_monthly_frequency(df)
    df, _ = interpolate_missing(df)
    df, _ = reject_out_of_bounds(df)
    df, _ = flag_aberrations(df)
    validate_schema(df)


def test_validate_schema_fails_on_bad_data(sample_df_synthetic):
    """Test validation pandera échoue sur données invalides."""
    df = sample_df_synthetic.copy()
    df.loc[0, "niveau_pct"] = 150
    with pytest.raises(Exception):
        validate_schema(df)


def test_source_column_preserved(sample_df_real):
    """Test que la colonne source est préservée."""
    df = ensure_monthly_frequency(sample_df_real)
    df, _ = interpolate_missing(df)
    df, _ = reject_out_of_bounds(df)
    df, _ = flag_aberrations(df)
    df = join_static_features(df)
    assert "source" in df.columns
    assert set(df["source"].unique()) == {"real"}


def test_no_nan_in_critical_columns_after_clean(sample_df_real):
    """Test qu'il n'y a pas de NaN dans les colonnes critiques après nettoyage complet."""
    df = ensure_monthly_frequency(sample_df_real)
    df, _ = interpolate_missing(df)
    df, _ = reject_out_of_bounds(df)
    df, _ = flag_aberrations(df)
    df = join_static_features(df)
    assert not df["date"].isna().any()
    assert not df["barrage"].isna().any()
    assert not df["source"].isna().any()


def test_aberration_flag_only_on_same_barrage():
    """Test que les flags d'aberration sont par barrage."""
    dates = pd.date_range("2020-01-01", periods=6, freq="MS")
    data = {
        "date": list(dates) * 2,
        "barrage": ["Al Massira"] * 6 + ["Bin el Ouidane"] * 6,
        "niveau_pct": [50, 52, 55, 90, 92, 95, 30, 32, 35, 38, 40, 42],
        "source": ["real"] * 12,
    }
    df = pd.DataFrame(data)
    df, _ = flag_aberrations(df)
    assert df[df["barrage"] == "Al Massira"]["aberration_flag"].any()
    assert not df[df["barrage"] == "Bin el Ouidane"]["aberration_flag"].any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])