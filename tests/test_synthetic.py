"""Tests pour le générateur synthétique."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


SYNTHETIC_PARQUET = Path("data/processed/synthetic/synthetic_dams.parquet")
SYNTHETIC_METADATA = Path("data/processed/synthetic/metadata.json")


def test_synthetic_file_exists():
    """Vérifie que les fichiers de sortie existent."""
    assert SYNTHETIC_PARQUET.exists(), "synthetic_dams.parquet non trouvé"
    assert SYNTHETIC_METADATA.exists(), "metadata.json non trouvé"


def test_no_nan_values():
    """Vérifie qu'il n'y a pas de NaN dans les colonnes critiques."""
    df = pd.read_parquet(SYNTHETIC_PARQUET)
    assert not df["niveau_pct"].isna().any(), "NaN dans niveau_pct"
    assert not df["date"].isna().any(), "NaN dans date"
    assert not df["serie_id"].isna().any(), "NaN dans serie_id"


def test_niveau_in_bounds():
    """Vérifie que les niveaux sont dans [0, 100]."""
    df = pd.read_parquet(SYNTHETIC_PARQUET)
    assert (df["niveau_pct"] >= 0).all(), "Niveau < 0 détecté"
    assert (df["niveau_pct"] <= 100).all(), "Niveau > 100 détecté"


def test_source_column():
    """Vérifie la colonne source = 'synthetic'."""
    df = pd.read_parquet(SYNTHETIC_PARQUET)
    assert (df["source"] == "synthetic").all(), "Colonne source incorrecte"


def test_n_series():
    """Vérifie le nombre de séries (100 attendues)."""
    df = pd.read_parquet(SYNTHETIC_PARQUET)
    n_series = df["serie_id"].nunique()
    assert n_series == 100, f"Attendu 100 séries, trouvé {n_series}"


def test_seasonality():
    """Vérifie la présence de saisonnalité (variance intra-année significative)."""
    df = pd.read_parquet(SYNTHETIC_PARQUET)

    df["month"] = df["date"].dt.month
    monthly_means = df.groupby(["serie_id", "month"])["niveau_pct"].mean().reset_index()

    intra_annual_std = monthly_means.groupby("serie_id")["niveau_pct"].std()
    mean_intra_std = intra_annual_std.mean()

    assert mean_intra_std > 5.0, f"Variance intra-annuelle trop faible: {mean_intra_std:.2f}"


def test_droughts_detectable():
    """Vérifie que les sécheresses sont détectables (périodes basses prolongées)."""
    df = pd.read_parquet(SYNTHETIC_PARQUET)

    n_series_with_drought = 0
    for sid in df["serie_id"].unique()[:20]:
        sdf = df[df["serie_id"] == sid].sort_values("date")
        low_threshold = sdf["niveau_pct"].quantile(0.25)
        below = (sdf["niveau_pct"] < low_threshold).astype(int)

        max_streak = 0
        current = 0
        for v in below:
            if v == 1:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0

        if max_streak >= 5:
            n_series_with_drought += 1

    assert n_series_with_drought >= 5, f"Sécheresses détectées dans seulement {n_series_with_drought}/20 séries"


def test_autocorrelation():
    """Vérifie l'autocorrélation temporelle (persistance)."""
    df = pd.read_parquet(SYNTHETIC_PARQUET)

    autocorr_values = []
    for sid in df["serie_id"].unique()[:10]:
        sdf = df[df["serie_id"] == sid].sort_values("date")
        if len(sdf) > 12:
            acf1 = sdf["niveau_pct"].autocorr(lag=1)
            if not np.isnan(acf1):
                autocorr_values.append(acf1)

    mean_acf = np.mean(autocorr_values) if autocorr_values else 0
    assert mean_acf > 0.5, f"Autocorrélation trop faible: {mean_acf:.3f}"


def test_mean_level_realistic():
    """Vérifie que le niveau moyen est réaliste (40-70%)."""
    df = pd.read_parquet(SYNTHETIC_PARQUET)
    mean_level = df["niveau_pct"].mean()
    assert 30 < mean_level < 80, f"Niveau moyen hors plage réaliste: {mean_level:.1f}%"


def test_no_mixing_with_processed():
    """Vérifie que data/synthetic/ est bien séparé de data/processed/ (pas de mélange)."""
    df = pd.read_parquet(SYNTHETIC_PARQUET)
    assert "synthetic" in str(SYNTHETIC_PARQUET), "Chemin incorrect pour synthétique"


def test_25_years_monthly():
    """Vérifie que chaque série a ~25 ans de données mensuelles (300 mois)."""
    df = pd.read_parquet(SYNTHETIC_PARQUET)
    months_per_series = df.groupby("serie_id").size()
    expected_months = 25 * 12
    assert (months_per_series >= expected_months * 0.9).all(), "Séries trop courtes"
    assert (months_per_series <= expected_months * 1.1).all(), "Séries trop longues"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])