import numpy as np

from src.training.metrics import (
    bootstrap_confidence_intervals,
    metrics_from_probabilities,
    select_operating_threshold,
)


def test_threshold_metrics_and_bootstrap_are_bounded():
    targets = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.3, 0.7, 0.9])
    threshold = select_operating_threshold(targets, probabilities)
    metrics = metrics_from_probabilities(targets, probabilities, threshold)
    intervals = bootstrap_confidence_intervals(
        targets, probabilities, threshold, n_bootstrap=20, seed=1
    )
    assert 0.0 <= threshold <= 1.0
    assert metrics["tp"] + metrics["tn"] == 4
    assert all(0.0 <= bounds["lower"] <= bounds["upper"] <= 1.0 for bounds in intervals.values())

