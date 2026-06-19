"""
Conditional volatility models used to capture volatility clustering --
the well-documented tendency of large price moves to be followed by more
large price moves (and calm periods to be followed by calm periods),
which a single unconditional (sample) standard deviation cannot capture.

Two estimators are provided:

* `ewma_volatility`  -- RiskMetrics-style exponentially weighted moving
  average. Cheap, no fitting required, reacts quickly to new shocks.
* `garch_volatility` -- a GARCH(1,1) model (via the `arch` package) fit
  by maximum likelihood, which explicitly models the mean-reverting,
  clustering behavior of variance and produces a forward volatility
  forecast for use in next-period VaR.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def ewma_volatility(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """RiskMetrics-style EWMA volatility: sigma_t^2 = lam*sigma_{t-1}^2 + (1-lam)*r_{t-1}^2.

    Parameters
    ----------
    returns : pd.Series
        Daily return series.
    lam : float
        Decay factor. RiskMetrics' standard value for daily data is 0.94.

    Returns
    -------
    pd.Series
        Estimated conditional daily volatility (standard deviation), same
        index as `returns`.
    """
    squared = returns.pow(2)
    variance = squared.ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(variance).rename("ewma_volatility")


def garch_volatility(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    forecast_horizon: int = 1,
    dist: str = "t",
) -> Tuple[pd.Series, np.ndarray, object]:
    """Fit a GARCH(p, q) model and return in-sample vol + an n-step forecast.

    Using a Student-t innovation distribution (the default) additionally
    captures the fat-tailed nature of daily equity returns on top of the
    clustering captured by the GARCH recursion itself.

    Parameters
    ----------
    returns : pd.Series
        Daily return series (decimal, not percent).
    p, q : int
        GARCH lag orders.
    forecast_horizon : int
        Number of days ahead to forecast conditional volatility.
    dist : str
        Innovation distribution passed to `arch_model` ("t" or "normal").

    Returns
    -------
    (conditional_vol, forecast_vol, fitted_model)
        conditional_vol : in-sample fitted daily volatility (decimal scale)
        forecast_vol    : array of length `forecast_horizon`, forecast vol
        fitted_model    : the underlying fitted `arch` results object,
                          useful for diagnostics (e.g. fitted_model.summary())

    Raises
    ------
    ImportError
        If the optional `arch` dependency is not installed.
    """
    try:
        from arch import arch_model
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "GARCH volatility modeling requires the 'arch' package. "
            "Install it with: pip install arch"
        ) from exc

    # `arch` is numerically better-behaved on percent-scale returns.
    pct_returns = returns * 100
    model = arch_model(pct_returns, vol="GARCH", p=p, q=q, dist=dist, mean="Constant")
    fitted = model.fit(disp="off")

    conditional_vol = (fitted.conditional_volatility / 100).rename("garch_volatility")
    forecast = fitted.forecast(horizon=forecast_horizon, reindex=False)
    forecast_vol = np.sqrt(forecast.variance.values[-1]) / 100

    return conditional_vol, forecast_vol, fitted