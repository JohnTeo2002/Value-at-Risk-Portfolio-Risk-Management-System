"""
Expected Shortfall (ES), also known as Conditional VaR (CVaR): the
expected loss given that the loss already exceeds the VaR threshold.

Where VaR answers "what's the loss at the (1-confidence) cutoff?",
ES answers "given we're past that cutoff, how bad does it get on
average?" -- a more complete picture of tail risk, and the metric
favored by Basel III/FRTB for regulatory capital calculations precisely
because VaR ignores everything beyond the threshold itself.
"""
from __future__ import annotations

from typing import Dict, Union

import numpy as np
import pandas as pd
from scipy import stats


def parametric_es(
    mean: float,
    std: float,
    confidence: float,
    horizon_days: int,
    portfolio_value: float,
) -> Dict:
    """Closed-form Expected Shortfall under a Normal distribution assumption.

    ES = -mu*h + sigma*sqrt(h) * phi(z) / (1 - confidence)

    where z = Phi^-1(1 - confidence) and phi is the standard Normal pdf.
    This is the analytic tail expectation of a Normal distribution beyond
    the VaR quantile.
    """
    alpha = 1 - confidence
    z = stats.norm.ppf(alpha)
    mu_h = mean * horizon_days
    sigma_h = std * np.sqrt(horizon_days)
    es_return = -mu_h + sigma_h * stats.norm.pdf(z) / alpha

    return {
        "method": "Parametric (Normal)",
        "confidence": confidence,
        "horizon_days": horizon_days,
        "es_return": es_return,
        "es_value": es_return * portfolio_value,
    }


def empirical_es(
    returns: Union[pd.Series, np.ndarray],
    confidence: float,
    portfolio_value: float,
    method_name: str = "Historical",
) -> Dict:
    """Expected Shortfall from an empirical (historical or simulated) sample.

    Defined as the average return among the worst (1-confidence) fraction
    of observations -- i.e. the mean loss conditional on breaching VaR.
    Works identically whether `returns` come from real historical data or
    from a Monte Carlo simulation; just pass in the relevant sample.

    Parameters
    ----------
    returns : array-like
        Historical or simulated returns (already scaled to the desired
        horizon, e.g. compounded h-day returns).
    confidence : float
        Confidence level, e.g. 0.95 or 0.99.
    portfolio_value : float
        Notional portfolio value.
    method_name : str
        Label describing the source of `returns` (e.g. "Historical",
        "Monte Carlo (Normal)", "Monte Carlo (Student-t)").
    """
    arr = np.asarray(returns)
    pct = (1 - confidence) * 100
    threshold = np.percentile(arr, pct)
    tail = arr[arr <= threshold]
    es_return = -tail.mean() if len(tail) else np.nan

    return {
        "method": method_name,
        "confidence": confidence,
        "es_return": es_return,
        "es_value": es_return * portfolio_value,
        "var_threshold_return": -threshold,
        "tail_observations": int(len(tail)),
    }