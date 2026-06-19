"""
Three industry-standard Value at Risk (VaR) methodologies:

1. Parametric / Variance-Covariance (Delta-Normal)
   Assumes portfolio returns are normally distributed; VaR is derived
   analytically from the portfolio mean and standard deviation.

2. Historical Simulation
   Makes no distributional assumption -- VaR is read directly off the
   empirical quantile of actual historical portfolio returns, preserving
   whatever skew / fat tails are present in the data.

3. Monte Carlo Simulation
   Simulates thousands of correlated synthetic return paths (under either
   a Normal or fat-tailed Student-t assumption) and reads VaR off the
   simulated distribution. This is the most flexible method and is the
   one used here to explicitly quantify how much standard Normal models
   underestimate risk relative to a fat-tailed alternative.

All three return a common `VaRResult` so they can be compared apples-to-apples.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class VaRResult:
    """Container for a single VaR estimate."""

    method: str
    confidence: float
    horizon_days: int
    var_return: float            # as a fraction of portfolio value (positive = loss)
    var_value: float             # in dollar terms
    extra: Optional[Dict] = field(default=None, repr=False)

    def __repr__(self) -> str:
        return (
            f"{self.method} | {self.confidence:.0%} conf | {self.horizon_days}d horizon "
            f"-> {self.var_return:.4%}  (${self.var_value:,.0f})"
        )


def _z_score(confidence: float) -> float:
    """Standard normal quantile for the left tail at the given confidence."""
    return stats.norm.ppf(1 - confidence)


# ---------------------------------------------------------------------------
# 1. Parametric (Variance-Covariance / Delta-Normal) VaR
# ---------------------------------------------------------------------------

def parametric_var(
    mean: float,
    std: float,
    confidence: float,
    horizon_days: int,
    portfolio_value: float,
) -> VaRResult:
    """Variance-Covariance (Delta-Normal) VaR.

    VaR = -(mu*h + z_alpha * sigma * sqrt(h)) * V

    where mu, sigma are DAILY mean/std, z_alpha = Phi^-1(1-confidence)
    (negative for confidence > 50%), h is the horizon in days, and the
    square-root-of-time rule scales daily volatility to the horizon.

    Parameters
    ----------
    mean, std : float
        Daily portfolio mean return and standard deviation.
    confidence : float
        Confidence level, e.g. 0.95 or 0.99.
    horizon_days : int
        Risk horizon in trading days (e.g. 1 or 10).
    portfolio_value : float
        Notional portfolio value used to convert the return into dollars.
    """
    z = _z_score(confidence)
    mu_h = mean * horizon_days
    sigma_h = std * np.sqrt(horizon_days)
    var_return = -(mu_h + z * sigma_h)

    return VaRResult(
        method="Parametric (Variance-Covariance)",
        confidence=confidence,
        horizon_days=horizon_days,
        var_return=var_return,
        var_value=var_return * portfolio_value,
        extra={"z_score": z, "mu_h": mu_h, "sigma_h": sigma_h},
    )


# ---------------------------------------------------------------------------
# 2. Historical Simulation VaR
# ---------------------------------------------------------------------------

def historical_var(
    returns: pd.Series,
    confidence: float,
    horizon_days: int,
    portfolio_value: float,
    scaling: str = "overlapping",
) -> VaRResult:
    """Historical Simulation VaR using the empirical return distribution.

    Parameters
    ----------
    returns : pd.Series
        Historical daily portfolio returns.
    confidence : float
        Confidence level, e.g. 0.95 or 0.99.
    horizon_days : int
        Risk horizon in trading days.
    portfolio_value : float
        Notional portfolio value.
    scaling : {"overlapping", "sqrt_time"}
        "overlapping" builds true rolling h-day compounded returns from the
        historical sample (preferred -- preserves the actual h-day return
        distribution). Falls back to "sqrt_time" automatically if there is
        not enough history. "sqrt_time" instead scales the 1-day empirical
        quantile by sqrt(h), matching the convention used by the parametric
        method (useful for direct method comparison, but a cruder
        approximation for a non-Normal sample).
    """
    if horizon_days == 1 or scaling == "sqrt_time":
        pct = (1 - confidence) * 100
        q1 = np.percentile(returns, pct)
        var_return = -q1 * (np.sqrt(horizon_days) if horizon_days > 1 else 1.0)
        method_label = f"Historical Simulation (sqrt-time scaled, {horizon_days}d)"
    else:
        if len(returns) < horizon_days + 30:
            raise ValueError(
                f"Not enough history ({len(returns)} obs) to build reliable "
                f"{horizon_days}-day overlapping returns; pass scaling='sqrt_time' instead."
            )
        h_day_returns = (
            returns.rolling(horizon_days)
            .apply(lambda r: np.prod(1 + r) - 1, raw=True)
            .dropna()
        )
        pct = (1 - confidence) * 100
        q_h = np.percentile(h_day_returns, pct)
        var_return = -q_h
        method_label = f"Historical Simulation (overlapping {horizon_days}d windows)"

    return VaRResult(
        method=method_label,
        confidence=confidence,
        horizon_days=horizon_days,
        var_return=var_return,
        var_value=var_return * portfolio_value,
    )


# ---------------------------------------------------------------------------
# 3. Monte Carlo VaR
# ---------------------------------------------------------------------------

def monte_carlo_var(
    mean_vector: np.ndarray,
    cov_matrix: np.ndarray,
    weights: np.ndarray,
    confidence: float,
    horizon_days: int,
    portfolio_value: float,
    n_simulations: int = 50_000,
    distribution: str = "normal",
    t_dof: float = 5.0,
    seed: Optional[int] = None,
) -> VaRResult:
    """Monte Carlo VaR via simulated correlated multi-day asset returns.

    Simulates `n_simulations` independent `horizon_days`-day paths for every
    asset, preserving the historical correlation structure (via a Cholesky
    decomposition of the covariance matrix), compounds each path to a
    single h-day portfolio return, and reads VaR off the resulting
    simulated distribution.

    Two innovation distributions are supported:

    * "normal" -- standard multivariate Normal innovations. Should produce
      VaR estimates close to the closed-form parametric method, serving as
      a sanity check on the simulation engine.
    * "t"      -- correlated multivariate Student-t innovations (built via
      a normal/chi-square mixture), which have heavier tails than the
      Normal for a given variance. Comparing "t" vs. "normal" VaR at the
      same confidence directly quantifies the "fat-tail" risk that a
      pure Normal/parametric model would miss.

    Parameters
    ----------
    mean_vector : np.ndarray, shape (n_assets,)
        Daily mean return per asset.
    cov_matrix : np.ndarray, shape (n_assets, n_assets)
        Daily covariance matrix of asset returns.
    weights : np.ndarray, shape (n_assets,)
        Portfolio weights (sum to 1).
    confidence : float
        Confidence level, e.g. 0.95 or 0.99.
    horizon_days : int
        Number of days to simulate forward and compound.
    portfolio_value : float
        Notional portfolio value.
    n_simulations : int
        Number of Monte Carlo paths.
    distribution : {"normal", "t"}
        Innovation distribution (see above).
    t_dof : float
        Degrees of freedom for the Student-t innovations (lower = fatter
        tails). Typically estimated from data via `fit_student_t`.
    seed : int or None
        Random seed for reproducibility.
    """
    if distribution not in ("normal", "t"):
        raise ValueError("distribution must be 'normal' or 't'")

    rng = np.random.default_rng(seed)
    n_assets = len(weights)
    # tiny ridge for numerical stability on near-singular covariance matrices
    chol = np.linalg.cholesky(cov_matrix + 1e-12 * np.eye(n_assets))

    if distribution == "normal":
        z = rng.standard_normal((n_simulations, horizon_days, n_assets))
        daily_asset_returns = mean_vector + z @ chol.T
    else:
        z = rng.standard_normal((n_simulations, horizon_days, n_assets)) @ chol.T
        chi2 = rng.chisquare(t_dof, size=(n_simulations, horizon_days, 1))
        # scale so the simulated variance matches cov_matrix despite the
        # heavier Student-t tails
        scale = np.sqrt(t_dof / chi2) * np.sqrt((t_dof - 2) / t_dof)
        daily_asset_returns = mean_vector + z * scale

    path_portfolio_returns = daily_asset_returns @ weights         # (n_sim, horizon_days)
    horizon_returns = np.prod(1 + path_portfolio_returns, axis=1) - 1

    pct = (1 - confidence) * 100
    q = np.percentile(horizon_returns, pct)
    var_return = -q

    return VaRResult(
        method=f"Monte Carlo ({'Student-t, dof=%.1f' % t_dof if distribution == 't' else 'Normal'})",
        confidence=confidence,
        horizon_days=horizon_days,
        var_return=var_return,
        var_value=var_return * portfolio_value,
        extra={"simulated_returns": horizon_returns, "n_simulations": n_simulations},
    )


def fit_student_t(returns: pd.Series):
    """Maximum-likelihood fit of a Student-t distribution to a return series.

    Returns
    -------
    (dof, loc, scale) : tuple of float
        Degrees of freedom, location, and scale of the fitted Student-t.
        A low `dof` (roughly < 10) indicates significantly fatter-than-Normal
        tails -- direct quantitative evidence for "fat-tail risk."
    """
    dof, loc, scale = stats.t.fit(returns)
    return dof, loc, scale