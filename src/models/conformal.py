"""Conformal prediction utilities."""

import logging
import numpy as np

from src.config import SEED

logger = logging.getLogger(__name__)

np.random.seed(SEED)


def conformal_intervals(y_calib_residuals, y_pred, coverage=0.90):
    """Compute conformal prediction intervals.

    Args:
        y_calib_residuals: Calibration residuals (1D array)
        y_pred: Predictions (1D array)
        coverage: Desired coverage level (default 0.90)

    Returns:
        Tuple of (lower_bounds, upper_bounds)
    """
    q = np.quantile(np.abs(y_calib_residuals), coverage)
    return y_pred - q, y_pred + q


def compute_coverage(y_true, low, high):
    """Compute empirical coverage of prediction intervals.

    Args:
        y_true: True values (1D array)
        low: Lower bounds (1D array)
        high: Upper bounds (1D array)

    Returns:
        Coverage as float in [0, 1]
    """
    in_interval = (y_true >= low) & (y_true <= high)
    return float(np.mean(in_interval))


def compute_crps(y_true, y_pred, y_pred_high, y_pred_low):
    """Compute CRPS using Gaussian approximation.

    Args:
        y_true: True values (1D array)
        y_pred: Predictions (1D array)
        y_pred_high: Upper bounds (1D array)
        y_pred_low: Lower bounds (1D array)

    Returns:
        CRPS score (float)
    """
    from properscoring import crps_gaussian

    sigma = (y_pred_high - y_pred_low) / 2 / 1.645
    scores = crps_gaussian(y_true, mu=y_pred, sig=sigma)
    return float(np.mean(scores))