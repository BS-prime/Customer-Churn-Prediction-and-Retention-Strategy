import pytest

from src.components.threshold_optimization import (
    calculate_cost,
    search_optimal_threshold,
)


def test_calculate_cost_uses_false_positive_and_false_negative_costs():
    y_true = [0, 1, 1, 0]
    y_pred = [1, 0, 1, 0]

    cost = calculate_cost(y_true, y_pred, fp_cost=1.0, fn_cost=10.0)

    assert cost == 11.0


def test_search_optimal_threshold_picks_lowest_cost_threshold():
    y_true = [0, 1, 1, 0]
    y_prob = [0.8, 0.7, 0.6, 0.1]

    threshold_info = search_optimal_threshold(
        y_true=y_true,
        y_prob=y_prob,
        fp_cost=1.0,
        fn_cost=10.0,
    )

    assert threshold_info["min_cost"] == 1.0
    assert threshold_info["best_threshold"] == pytest.approx(0.11)
    assert threshold_info["cost_fp"] == 1.0
    assert threshold_info["cost_fn"] == 10.0
