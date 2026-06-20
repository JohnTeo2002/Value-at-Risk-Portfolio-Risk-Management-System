import pandas as pd

from var_risk.backtesting import (
    rolling_var_backtest,
    evaluate_backtest,
    kupiec_pof_test,
    christoffersen_independence_test,
)


def test_kupiec_test_runs_and_bounds():
    lr, p = kupiec_pof_test(1000, 50, 0.95)
    assert lr >= 0
    assert 0 <= p <= 1


def test_kupiec_edge_cases_zero_and_all_breaches():
    lr0, p0 = kupiec_pof_test(500, 0, 0.95)
    assert lr0 >= 0 and 0 <= p0 <= 1
    lr1, p1 = kupiec_pof_test(500, 500, 0.95)
    assert lr1 >= 0 and 0 <= p1 <= 1


def test_rolling_backtest_produces_valid_dataframe(synthetic_portfolio):
    p = synthetic_portfolio
    bt = rolling_var_backtest(p.portfolio_returns, 0.95, window=200, method="parametric")
    assert {"actual_return", "var_forecast", "breach"}.issubset(bt.columns)
    assert len(bt) == len(p.portfolio_returns) - 200
    assert bt["breach"].dtype == bool


def test_evaluate_backtest_runs(synthetic_portfolio):
    p = synthetic_portfolio
    bt = rolling_var_backtest(p.portfolio_returns, 0.95, window=200, method="historical")
    result = evaluate_backtest(bt, 0.95)
    assert result.n_obs == len(bt)
    assert result.n_breaches == int(bt["breach"].sum())
    assert 0 <= result.kupiec_p_value <= 1
    assert 0 <= result.christoffersen_p_value <= 1


def test_christoffersen_test_on_independent_breaches():
    # Manually constructed, evenly-spaced breach pattern (no clustering)
    breaches = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 10)
    lr, p_value = christoffersen_independence_test(breaches)
    assert lr >= 0
    assert 0 <= p_value <= 1


def test_christoffersen_test_on_clustered_breaches():
    # All breaches grouped together -- should show strong dependence
    breaches = pd.Series([0] * 80 + [1] * 20)
    lr, p_value = christoffersen_independence_test(breaches)
    assert lr >= 0