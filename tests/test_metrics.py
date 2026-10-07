"""Tests unitaires pour les métriques d'évaluation."""

import numpy as np
import pytest

from src.evaluation.metrics import (
    coverage,
    diebold_mariano,
    mae,
    mase,
    rmse,
)


def test_mae_perfect():
    """MAE = 0 pour prédiction parfaite."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert mae(y_true, y_pred) == 0.0


def test_mae_known():
    """MAE avec valeurs connues."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    assert abs(mae(y_true, y_pred) - 2.333333) < 1e-6


def test_mae_with_nan():
    """MAE ignore les NaN."""
    y_true = np.array([1.0, np.nan, 3.0, 4.0])
    y_pred = np.array([1.5, 2.5, 3.5, np.nan])
    assert abs(mae(y_true, y_pred) - 0.5) < 1e-6


def test_rmse_perfect():
    """RMSE = 0 pour prédiction parfaite."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert rmse(y_true, y_pred) == 0.0


def test_rmse_known():
    """RMSE avec valeurs connues."""
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 1.0, 1.0])
    assert rmse(y_true, y_pred) == 1.0


def test_mase_perfect():
    """MASE = 0 pour modèle parfait (dans la limite du bruit)."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([10.0, 20.0, 30.0, 40.0])
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
    assert mase(y_true, y_pred, y_train, seasonality=12) == 0.0


def test_mase_known():
    """MASE avec valeurs connues."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
    m = mase(y_true, y_pred, y_train, seasonality=12)
    assert m > 0


def test_coverage_perfect():
    """Coverage = 1.0 si toutes les valeurs dans l'intervalle."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_lower = np.array([9.0, 19.0, 29.0, 39.0, 49.0])
    y_upper = np.array([11.0, 21.0, 31.0, 41.0, 51.0])
    assert coverage(y_true, y_lower, y_upper) == 1.0


def test_coverage_90():
    """Coverage = 0.9 si 9 sur 10 dans l'intervalle."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
    y_lower = np.array([9.0] * 10)
    y_upper = np.array([95.0] * 10)
    assert abs(coverage(y_true, y_lower, y_upper) - 0.9) < 1e-6


def test_coverage_with_nan():
    """Coverage ignore les NaN."""
    y_true = np.array([10.0, np.nan, 30.0, 40.0])
    y_lower = np.array([9.0, 19.0, 29.0, 39.0])
    y_upper = np.array([11.0, 21.0, 31.0, 41.0])
    assert coverage(y_true, y_lower, y_upper) == 1.0


def test_diebold_mariano_equal():
    """DM stat ≈ 0 pour modèles identiques (variance nulle -> NaN)."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred1 = np.array([11.0, 19.0, 31.0, 39.0, 51.0])
    y_pred2 = np.array([11.0, 19.0, 31.0, 39.0, 51.0])
    dm, p = diebold_mariano(y_true, y_pred1, y_pred2)
    assert np.isnan(dm)
    assert np.isnan(p)


def test_diebold_mariano_different():
    """DM détecte une différence significative."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
    y_pred1 = y_true + 0.1
    y_pred2 = y_true + 5.0
    dm, p = diebold_mariano(y_true, y_pred1, y_pred2)
    assert dm < 0
    assert p < 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])