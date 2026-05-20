import numpy as np


def retention_profit(
        prob,
        offer_cost=20,
        monthly_revenue=70,
        retention_success=0.3,
        retained_months=12
) -> float | int:
    expected_gain = prob * retention_success * monthly_revenue * retained_months

    profit = expected_gain - offer_cost
    return profit


def optimal_targeting(probs) -> np.ndarray:
    profits = [retention_profit(p) for p in probs]
    return np.array(profits)
