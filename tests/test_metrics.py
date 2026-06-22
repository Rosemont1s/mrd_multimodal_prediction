import numpy as np

from src.training.metrics import (
    bootstrap_confidence_intervals,
    calibration_curve_data,
    decision_curve_data,
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
    assert all(
        0.0 <= bounds["lower"] <= bounds["upper"] <= 1.0
        for bounds in intervals.values()
    )


def test_target_sensitivity_threshold_preserves_requested_sensitivity():
    targets = np.array([0, 0, 0, 1, 1, 1, 1])
    probabilities = np.array([0.05, 0.20, 0.60, 0.40, 0.55, 0.80, 0.95])
    threshold = select_operating_threshold(
        targets,
        probabilities,
        strategy="target_sensitivity",
        target_sensitivity=0.75,
    )
    metrics = metrics_from_probabilities(targets, probabilities, threshold)

    assert threshold == 0.55
    assert metrics["sensitivity"] >= 0.75
    assert metrics["tests_avoided_rate"] == 3 / 7
    assert metrics["mrd_positive_miss_rate"] == 0.25


def test_calibration_and_decision_curve_outputs_are_well_formed():
    targets = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])

    calibration = calibration_curve_data(targets, probabilities, n_bins=2)
    decision_curve = decision_curve_data(
        targets, probabilities, thresholds=np.array([0.5])
    )

    assert sum(row["patients"] for row in calibration) == 4
    assert decision_curve[0]["threshold"] == 0.5
    assert "model_net_benefit" in decision_curve[0]
