"""
Portfolio construction: asset weighting, return calculation, and the
mean / covariance estimates that feed every downstream risk model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class Portfolio:
    """A weighted basket of equities with associated return statistics.

    Parameters
    ----------
    prices : pd.DataFrame
        Date-indexed price history, one column per ticker.
    weights : dict[str, float]
        Portfolio weight for each ticker. Must sum to ~1.0.
    notional : float
        Total dollar value of the portfolio (used to convert returns into
        dollar VaR / ES figures).
    """

    prices: pd.DataFrame
    weights: Dict[str, float]
    notional: float = 1_000_000.0

    asset_returns: pd.DataFrame = field(init=False, repr=False)
    weight_vector: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        tickers = list(self.prices.columns)
        unknown = set(self.weights) - set(tickers)
        if unknown:
            raise ValueError(f"Weights given for tickers not in price data: {unknown}")

        weight_sum = sum(self.weights.values())
        if not np.isclose(weight_sum, 1.0, atol=1e-3):
            raise ValueError(f"Portfolio weights must sum to 1.0 (got {weight_sum:.4f})")

        self.weight_vector = np.array([self.weights.get(t, 0.0) for t in tickers])
        self.asset_returns = self.prices.sort_index().pct_change().dropna(how="all")
        self.asset_returns = self.asset_returns.fillna(0.0)

    # -- basic accessors -------------------------------------------------

    @property
    def tickers(self) -> List[str]:
        return list(self.prices.columns)

    @property
    def portfolio_returns(self) -> pd.Series:
        """Daily simple returns of the portfolio (weights assumed static)."""
        return self.asset_returns.dot(self.weight_vector).rename("portfolio_return")

    @property
    def mean_returns(self) -> pd.Series:
        return self.asset_returns.mean()

    @property
    def cov_matrix(self) -> pd.DataFrame:
        return self.asset_returns.cov()

    @property
    def portfolio_mean(self) -> float:
        """Daily expected portfolio return, mu_p = w^T * mu."""
        return float(self.weight_vector @ self.mean_returns.values)

    @property
    def portfolio_std(self) -> float:
        """Daily portfolio volatility, sigma_p = sqrt(w^T * Sigma * w)."""
        cov = self.cov_matrix.values
        return float(np.sqrt(self.weight_vector @ cov @ self.weight_vector.T))

    # -- helpers used by stress testing -----------------------------------

    def asset_beta(self, benchmark_returns: pd.Series) -> pd.Series:
        """Estimate each asset's beta against a benchmark return series.

        Used by the stress-testing module to translate a benchmark-level
        crash scenario (e.g. "S&P 500 fell 34% during COVID") into an
        asset-specific shock via a single-factor (CAPM-style) model.
        """
        aligned = self.asset_returns.join(benchmark_returns.rename("__bench__"), how="inner").dropna()
        bench_var = aligned["__bench__"].var()
        betas = {}
        for ticker in self.tickers:
            cov = aligned[[ticker, "__bench__"]].cov().iloc[0, 1]
            betas[ticker] = cov / bench_var if bench_var > 0 else 1.0
        return pd.Series(betas)

    def summary(self) -> dict:
        """A compact dict summary of portfolio composition and risk stats."""
        return {
            "n_assets": len(self.tickers),
            "tickers": self.tickers,
            "weights": self.weights,
            "notional": self.notional,
            "observations": len(self.asset_returns),
            "daily_mean_return": self.portfolio_mean,
            "daily_volatility": self.portfolio_std,
            "annualized_volatility": self.portfolio_std * np.sqrt(252),
            "annualized_return": self.portfolio_mean * 252,
        }