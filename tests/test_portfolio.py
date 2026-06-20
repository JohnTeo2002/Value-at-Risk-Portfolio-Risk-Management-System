import numpy as np
import pandas as pd
import pytest

from var_risk.portfolio import Portfolio
from var_risk.data_loader import generate_synthetic_prices


def test_portfolio_weights_must_sum_to_one():
    prices = generate_synthetic_prices(["A", "B"], n_days=100, seed=1)
    with pytest.raises(ValueError):
        Portfolio(prices=prices, weights={"A": 0.5, "B": 0.6})


def test_portfolio_rejects_unknown_ticker():
    prices = generate_synthetic_prices(["A", "B"], n_days=100, seed=1)
    with pytest.raises(ValueError):
        Portfolio(prices=prices, weights={"A": 0.5, "Z": 0.5})


def test_portfolio_returns_shape(synthetic_portfolio):
    p = synthetic_portfolio
    assert len(p.portfolio_returns) == len(p.prices) - 1


def test_portfolio_mean_and_std_are_finite(synthetic_portfolio):
    p = synthetic_portfolio
    assert np.isfinite(p.portfolio_mean)
    assert p.portfolio_std > 0


def test_portfolio_summary_keys(synthetic_portfolio):
    summary = synthetic_portfolio.summary()
    expected_keys = {
        "n_assets", "tickers", "weights", "notional", "observations",
        "daily_mean_return", "daily_volatility", "annualized_volatility", "annualized_return",
    }
    assert expected_keys.issubset(summary.keys())


def test_asset_beta_against_benchmark(synthetic_portfolio_with_benchmark):
    portfolio, benchmark_returns = synthetic_portfolio_with_benchmark
    betas = portfolio.asset_beta(benchmark_returns)
    assert set(betas.index) == set(portfolio.tickers)
    assert betas.notna().all()