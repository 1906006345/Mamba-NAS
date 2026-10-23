import numpy as np

from mamba_nas.convergence_metrics import (
    epsilon_indicator,
    front_metrics,
    hv_auc,
    igd_plus,
    nondominated_points,
    target_crossing,
)


def test_reference_front_removes_dominated_points():
    points = np.asarray([[0.1, 0.8], [0.2, 0.9], [0.3, 0.7]])
    front = nondominated_points(points)
    assert len(front) == 2
    assert [0.2, 0.9] not in front.tolist()


def test_hv_auc_and_target_crossing():
    assert hv_auc([1, 2, 3], [0.0, 0.5, 1.0], budget=3) == 1 / 3
    assert target_crossing([1, 2, 3], [0.1, 0.8, 1.0], 0.75) == 2


def test_igd_plus_and_epsilon_are_zero_for_reference_front():
    reference = np.asarray([[0.1, 0.8], [0.3, 0.4]])
    metrics = front_metrics(reference, reference)
    assert metrics["igd_plus"] == 0.0
    assert metrics["epsilon_indicator"] == 0.0
    assert igd_plus(reference, reference) == 0.0
    assert epsilon_indicator(reference, reference) == 0.0
