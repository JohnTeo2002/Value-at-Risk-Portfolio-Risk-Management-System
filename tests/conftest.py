import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from var_risk.data_loader import generate_synthetic_prices
from var_risk.portfolio import Portfolio


@pytest.fixture
def synthetic_portfolio():
    """A small, reproducible synthetic portfolio for unit tests."""
    tickers = ["A", "B", "C"]
    prices = generate_synthetic_prices(tickers, n_days=800, seed=7)
    weights = {"A": 0.5, "B": 0.3, "C": 0.2}
    return Portfolio(prices=prices, weights=weights, notional=1_000_000)


@pytest.fixture
def synthetic_portfolio_with_benchmark():
    """Synthetic portfolio plus a separate benchmark series for beta tests."""
    tickers = ["A", "B", "C", "BENCH"]
    prices = generate_synthetic_prices(tickers, n_days=800, seed=11)
    benchmark_returns = prices["BENCH"].pct_change().dropna()
    weights = {"A": 0.5, "B": 0.3, "C": 0.2}
    portfolio = Portfolio(prices=prices[["A", "B", "C"]], weights=weights, notional=1_000_000)
    return portfolio, benchmark_returns