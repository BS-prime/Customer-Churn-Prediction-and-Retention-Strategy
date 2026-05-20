from src.retention_strategy.retention import retention_profit


def test_retention_profit_can_be_positive_for_high_enough_probability():
    assert retention_profit(0.1) > 0


def test_retention_profit_can_be_negative_for_zero_probability():
    assert retention_profit(0.0) == -20
