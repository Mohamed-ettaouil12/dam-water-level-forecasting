"""Tests for conformal prediction utilities."""

import numpy as np
import pytest

from src.models.conformal import conformal_intervals, compute_coverage, compute_crps


def test_conformal_quantile_exact():
    """Test that quantile is exact for constructed data."""
    # 10 values, 90% quantile -> index 8 (0-indexed) = 9th value = 9.0
    residuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    y_pred = np.array([50.0])

    low, high = conformal_intervals(residuals, y_pred, coverage=0.90)
    q = high[0] - y_pred[0]

    # With linear interpolation (default), 0.9 quantile of 10 values is at position 8.1
    # = 9.0 + 0.1 * (10.0 - 9.0) = 9.1
    np.testing.assert_allclose(q, 9.1, rtol=1e-10)


def test_coverage_09_on_constructed():
    """Test coverage is ~0.9 on constructed example."""
    np.random.seed(42)
    n = 1000
    residuals = np.random.normal(0, 1, n)
    y_pred = np.zeros(n)

    low, high = conformal_intervals(residuals, y_pred, coverage=0.90)

    # Generate test data with same distribution
    y_true = np.random.normal(0, 1, n)
    cov = compute_coverage(y_true, low, high)

    # Should be close to 0.90 (within sampling error)
    assert 0.85 <= cov <= 0.95


def test_crps_finite():
    """Test CRPS returns a finite value."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    y_pred_low = np.array([0.5, 1.3, 2.5])
    y_pred_high = np.array([1.7, 2.5, 3.9])

    crps = compute_crps(y_true, y_pred, y_pred_high, y_pred_low)

    assert np.isfinite(crps)
    assert crps >= 0.0