"""
Statistical validation of VaR models against realized outcomes.

Implements:

* `rolling_var_backtest`           -- rolling, strictly out-of-sample 1-day
  VaR forecasts, flagged against the actual realized return each day.
* `kupiec_pof_test`                -- Kupiec (1995) Proportion-of-Failures
  likelihood-ratio test for unconditional coverage (does the breach rate
  match the model's stated confidence level?).
* `christoffersen_independence_test` -- Christoffersen (1998) test for
  whether breaches cluster in time (a sign the model failed to capture
  volatility clustering) rather than occurring independently as a
  correctly-specified model would imply.
* `evaluate_backtest`              -- combines both into a single report,
  including the joint conditional-coverage test (Kupiec + Christoffersen).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class BacktestResult:
    n_obs: int
    n_breaches: int
    breach_rate: float
    expected_rate: float
    kupiec_lr: float
    kupiec_p_value: float
    kupiec_reject: bool
    christoffersen_lr_ind: Optional[float] = None
    christoffersen_p_value: Optional[float] = None
    conditional_lr: Optional[float] = None
    conditional_p_value: Optional[float] = None

    def __repr__(self) -> str:
        verdict = "REJECTED (model miscalibrated)" if self.kupiec_reject else "not rejected (model OK)"
        return (
            f"Backtest: {self.n_breaches}/{self.n_obs} breaches "
            f"({self.breach_rate:.2%} vs expected {self.expected_rate:.2%}) | "
            f"Kupiec p={self.kupiec_p_value:.3f} -> {verdict}"
        )


def rolling_var_backtest(
    returns: pd.Series,
    confidence: float,
    window: int = 250,
    method: str = "parametric",
) -> pd.DataFrame:
    """Produce day-by-day, strictly out-of-sample 1-day VaR forecasts.

    For each day t (t >= window), VaR is estimated using only
    returns[t-window : t] -- i.e. information available *before* day t --
    then compared against the actual realized return on day t. This
    avoids look-ahead bias, which is essential for a backtest to be
    meaningful.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns.
    confidence : float
        Confidence level being backtested, e.g. 0.95 or 0.99.
    window : int
        Trailing lookback window (in trading days) used to estimate each
        day's VaR forecast.
    method : {"parametric", "historical"}
        Which VaR methodology to backtest.

    Returns
    -------
    pd.DataFrame
        Indexed by date, with columns: actual_return, var_forecast (as a
        positive-loss fraction), and breach (bool).
    """
    if method not in ("parametric", "historical"):
        raise ValueError("method must be 'parametric' or 'historical'")

    z = stats.norm.ppf(1 - confidence)
    records = []

    for t in range(window, len(returns)):
        hist = returns.iloc[t - window : t]
        actual = returns.iloc[t]

        if method == "parametric":
            mu, sigma = hist.mean(), hist.std()
            var_return = -(mu + z * sigma)
        else:
            var_return = -np.percentile(hist, (1 - confidence) * 100)

        breach = bool(actual < -var_return)
        records.append(
            {
                "date": returns.index[t],
                "actual_return": actual,
                "var_forecast": var_return,
                "breach": breach,
            }
        )

    return pd.DataFrame(records).set_index("date")


def kupiec_pof_test(n_obs: int, n_breaches: int, confidence: float) -> Tuple[float, float]:
    """Kupiec (1995) Proportion-of-Failures likelihood-ratio test.

    H0: the true breach probability equals the model's stated tail
    probability p = 1 - confidence. Under H0, the LR statistic is
    asymptotically chi-square distributed with 1 degree of freedom.

    Returns
    -------
    (lr_statistic, p_value)
    """
    p = 1 - confidence
    x, n = n_breaches, n_obs

    if n == 0:
        return 0.0, 1.0
    if x == 0:
        lr = -2 * n * np.log(1 - p)
    elif x == n:
        lr = -2 * n * np.log(p)
    else:
        p_hat = x / n
        lr = -2 * (
            (n - x) * np.log(1 - p) + x * np.log(p)
            - ((n - x) * np.log(1 - p_hat) + x * np.log(p_hat))
        )

    lr = max(lr, 0.0)  # guard against tiny negative values from floating-point error
    p_value = 1 - stats.chi2.cdf(lr, df=1)
    return float(lr), float(p_value)


def christoffersen_independence_test(breaches: pd.Series) -> Tuple[float, float]:
    """Christoffersen (1998) test for independence of consecutive VaR breaches.

    Tests whether breaches cluster together in time (consistent with
    volatility clustering / regime shifts the model failed to capture)
    versus occurring independently, as a correctly specified model implies.
    The LR statistic is asymptotically chi-square distributed with 1
    degree of freedom under the null of independence.

    Returns
    -------
    (lr_statistic, p_value)
    """
    b = breaches.astype(int).values
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(b)):
        prev, curr = b[i - 1], b[i]
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        else:
            n11 += 1

    n0, n1 = n00 + n01, n10 + n11
    pi01 = n01 / n0 if n0 else 0.0
    pi11 = n11 / n1 if n1 else 0.0
    pi = (n01 + n11) / (n0 + n1) if (n0 + n1) else 0.0

    def _safe_log(x: float) -> float:
        return np.log(x) if x > 0 else 0.0

    ll_unrestricted = (
        n00 * _safe_log(1 - pi01) + n01 * _safe_log(pi01)
        + n10 * _safe_log(1 - pi11) + n11 * _safe_log(pi11)
    )
    ll_restricted = n0 * _safe_log(1 - pi) + n1 * _safe_log(pi)

    lr_ind = -2 * (ll_restricted - ll_unrestricted)
    p_value = 1 - stats.chi2.cdf(lr_ind, df=1)
    return float(lr_ind), float(p_value)


def evaluate_backtest(backtest_df: pd.DataFrame, confidence: float) -> BacktestResult:
    """Run the full backtest evaluation: Kupiec + Christoffersen + joint test.

    Parameters
    ----------
    backtest_df : pd.DataFrame
        Output of `rolling_var_backtest`.
    confidence : float
        Confidence level that was backtested.
    """
    n_obs = len(backtest_df)
    n_breaches = int(backtest_df["breach"].sum())
    breach_rate = n_breaches / n_obs if n_obs else float("nan")
    expected_rate = 1 - confidence

    lr_kupiec, p_kupiec = kupiec_pof_test(n_obs, n_breaches, confidence)
    lr_ind, p_ind = christoffersen_independence_test(backtest_df["breach"])

    lr_cc = lr_kupiec + lr_ind
    p_cc = 1 - stats.chi2.cdf(lr_cc, df=2)

    return BacktestResult(
        n_obs=n_obs,
        n_breaches=n_breaches,
        breach_rate=breach_rate,
        expected_rate=expected_rate,
        kupiec_lr=lr_kupiec,
        kupiec_p_value=p_kupiec,
        kupiec_reject=p_kupiec < 0.05,
        christoffersen_lr_ind=lr_ind,
        christoffersen_p_value=p_ind,
        conditional_lr=lr_cc,
        conditional_p_value=p_cc,
    )