"""
Utilities for retrieving historical price data used by the risk engine.

Two modes are supported:

1. Live data via `yfinance` (download_price_data)
    Requires internet
    access. Results are optionally cached to disk as CSV so repeated runs
    don't re-hit the network.

2. Synthetic data (generate_synthetic_prices) 
    Correlated, fat-tailed
    GBM-style price simulator used for offline development, unit testing,
    and CI pipelines where network access isn't available or desirable.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd


def download_price_data(
    tickers: Sequence[str],
    start: str,
    end: str,
    cache_dir: Optional[str] = "data_cache",
) -> pd.DataFrame:
    """Download adjusted-close prices for a list of tickers via yfinance.

    Parameters
    ----------
    tickers : sequence of str
        Ticker symbols, e.g. ["AAPL", "MSFT", "GOOGL"].
    start, end : str
        Date strings in "YYYY-MM-DD" format.
    cache_dir : str or None
        If provided, cached CSVs are read from / written to this directory
        so repeated runs don't re-download the same data.

    Returns
    -------
    pd.DataFrame
        Date-indexed DataFrame of prices, one column per ticker.
    """
    import yfinance as yf  # imported lazily: the rest of the package works
    # without yfinance/internet access (e.g. for testing) as long as this
    # function is never called.

    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        key = "_".join(sorted(tickers)) + f"_{start}_{end}.csv"
        cache_path = os.path.join(cache_dir, key)
        if os.path.exists(cache_path):
            return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    raw = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        # yfinance collapses the column MultiIndex for a single ticker
        prices = raw[["Close"]]
        prices.columns = list(tickers)

    prices = prices.dropna(how="all").ffill().dropna()

    if cache_path:
        prices.to_csv(cache_path)

    return prices


def generate_synthetic_prices(
    tickers: Sequence[str],
    n_days: int = 1500,
    start_price: float = 100.0,
    annual_vol: float = 0.25,
    annual_drift: float = 0.07,
    fat_tail_prob: float = 0.012,
    fat_tail_scale: float = 6.0,
    correlation: float = 0.35,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """Generate correlated synthetic daily price paths for offline use.

    Each asset follows a GBM-style daily-return process with a shared
    correlation structure. A small probability of much larger, still
    correlated shocks is mixed in on top of the base process, producing
    occasional "crash" days with fat tails and volatility clustering --
    useful for exercising the fat-tail / backtesting logic without needing
    real market data.

    Parameters
    ----------
    tickers : sequence of str
        Labels for the synthetic assets (e.g. ["AAPL", "MSFT", "SPY"]).
    n_days : int
        Number of business days to simulate.
    start_price : float
        Starting price for every asset.
    annual_vol, annual_drift : float
        Annualized base volatility and drift, converted to daily figures.
    fat_tail_prob : float
        Daily probability of a market-wide fat-tail shock day.
    fat_tail_scale : float
        Multiplier applied to volatility on a shock day.
    correlation : float
        Pairwise correlation assumed between all asset pairs.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Business-day-indexed DataFrame of simulated prices.
    """
    rng = np.random.default_rng(seed)
    n_assets = len(tickers)

    corr = np.full((n_assets, n_assets), correlation)
    np.fill_diagonal(corr, 1.0)
    chol = np.linalg.cholesky(corr)

    daily_vol = annual_vol / np.sqrt(252)
    daily_drift = annual_drift / 252

    base_z = rng.standard_normal((n_days, n_assets)) @ chol.T
    returns = daily_drift + daily_vol * base_z

    # Inject correlated fat-tail "crash" shocks on a random subset of days.
    shock_days = rng.random((n_days, 1)) < fat_tail_prob
    shock_z = rng.standard_normal((n_days, n_assets)) @ chol.T
    returns = returns + shock_days * (shock_z * daily_vol * fat_tail_scale)

    log_prices = np.log(start_price) + np.cumsum(returns, axis=0)
    prices = np.exp(log_prices)

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    return pd.DataFrame(prices, index=dates, columns=list(tickers))