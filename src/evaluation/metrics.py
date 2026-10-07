"""Métriques d'évaluation pour la prévision des barrages."""

import logging
import numpy as np
import pandas as pd
from typing import Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not mask.any():
        return np.nan
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, seasonality: int = 12) -> float:
    """
    Mean Absolute Scaled Error.
    Baseline = valeur du même mois de l'année précédente (naïve saisonnière).
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not mask.any():
        return np.nan

    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_train) < seasonality + 1:
        logger.warning("y_train trop court pour saisonnalité %d, utilise saisonnalité 1", seasonality)
        seasonality = 1

    naive_forecast = np.roll(y_train, seasonality)[seasonality:]
    naive_true = y_train[seasonality:]

    mae_naive = np.mean(np.abs(naive_true - naive_forecast))
    if mae_naive == 0:
        return np.nan

    mae_model = np.mean(np.abs(y_true_clean - y_pred_clean))
    return float(mae_model / mae_naive)


def coverage(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """Pourcentage de valeurs réelles dans [lower, upper]."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_lower) & ~np.isnan(y_upper)
    if not mask.any():
        return np.nan
    in_interval = (y_true[mask] >= y_lower[mask]) & (y_true[mask] <= y_upper[mask])
    return float(np.mean(in_interval))


def diebold_mariano(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    loss: str = "square",
    h: int = 1,
) -> Tuple[float, float]:
    """
    Test de Diebold-Mariano pour comparer deux modèles.
    Retourne (statistique DM, p-value).
    H0: les deux modèles ont la même précision prévisionnelle.
    """
    from scipy import stats

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred1) & ~np.isnan(y_pred2)
    if not mask.any():
        return np.nan, np.nan

    y_t = y_true[mask]
    y_1 = y_pred1[mask]
    y_2 = y_pred2[mask]

    if loss == "square":
        d = (y_t - y_1) ** 2 - (y_t - y_2) ** 2
    elif loss == "abs":
        d = np.abs(y_t - y_1) - np.abs(y_t - y_2)
    else:
        raise ValueError(f"Loss inconnue: {loss}")

    n = len(d)
    if n < 2:
        return np.nan, np.nan

    d_mean = np.mean(d)
    gamma_0 = np.sum((d - d_mean) ** 2) / n

    if h > 1:
        gamma_h = 0
        for k in range(1, h):
            if k < n:
                gamma_h += np.sum((d[k:] - d_mean) * (d[:-k] - d_mean)) / n
        var_d = (gamma_0 + 2 * gamma_h) / n
    else:
        var_d = gamma_0 / n

    if var_d <= 0:
        return np.nan, np.nan

    dm_stat = d_mean / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return float(dm_stat), float(p_value)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lower: np.ndarray = None,
    y_upper: np.ndarray = None,
    y_train: np.ndarray = None,
    seasonality: int = 12,
) -> dict:
    """Évalue un ensemble de prédictions avec toutes les métriques."""
    results = {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
    }

    if y_train is not None:
        results["MASE"] = mase(y_true, y_pred, y_train, seasonality)

    if y_lower is not None and y_upper is not None:
        results["Coverage"] = coverage(y_true, y_lower, y_upper)

    return results


if __name__ == "__main__":
    y_true = np.array([10, 20, 30, 40, 50])
    y_pred = np.array([11, 19, 31, 39, 51])
    y_train = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70])

    print("MAE:", mae(y_true, y_pred))
    print("RMSE:", rmse(y_true, y_pred))
    print("MASE:", mase(y_true, y_pred, y_train, seasonality=12))

    y_lower = y_pred - 2
    y_upper = y_pred + 2
    print("Coverage:", coverage(y_true, y_lower, y_upper))

    y_pred2 = np.array([12, 18, 32, 38, 52])
    dm, p = diebold_mariano(y_true, y_pred, y_pred2)
    print(f"Diebold-Mariano: stat={dm:.4f}, p={p:.4f}")