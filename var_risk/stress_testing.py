"""
Historical scenario stress testing: estimate how the current portfolio
would have fared under famous historical market crashes, independent of
whether the portfolio's own tickers actually existed/traded through that
period.

Methodology
-----------
Each asset's return under a stress scenario is approximated with a
single-factor (CAPM-style) shock:

    asset_stress_return_i = beta_i * benchmark_stress_return

where `beta_i` is the asset's historically estimated beta to a broad
market benchmark (see `Portfolio.asset_beta`), and `benchmark_stress_return`
is the benchmark's known return during the historical crash window. This
lets the framework stress-test *any* portfolio of tickers -- including
recently-listed stocks with no price history reaching back to, say, 2008 --
against well-known historical shocks.

IMPORTANT DATA DISCLAIMER
--------------------------
The benchmark shock figures in `HISTORICAL_SCENARIOS` below are
*approximate, illustrative* S&P 500 peak-to-trough (or single-day)
returns compiled from well-known public market history. They are provided
to demonstrate the stress-testing methodology end-to-end, not as an
authoritative data source. For production risk management, replace these
with verified figures from a licensed market data vendor.
"""
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

HISTORICAL_SCENARIOS: Dict[str, Dict] = {
    "Black Monday (Oct 1987)": {
        "window": "1987-10-14 to 1987-10-19",
        "benchmark_return": -0.205,
    },
    "Asian Financial Crisis (1997)": {
        "window": "Oct 1997",
        "benchmark_return": -0.070,
    },
    "Dot-com Crash (2000-2002)": {
        "window": "Mar 2000 to Oct 2002",
        "benchmark_return": -0.490,
    },
    "9/11 Aftermath (2001)": {
        "window": "Sep 2001",
        "benchmark_return": -0.116,
    },
    "Global Financial Crisis (2008)": {
        "window": "Sep 2008 to Nov 2008",
        "benchmark_return": -0.410,
    },
    "Flash Crash (May 2010)": {
        "window": "2010-05-06",
        "benchmark_return": -0.090,
    },
    "US Debt Ceiling Crisis (2011)": {
        "window": "Jul 2011 to Aug 2011",
        "benchmark_return": -0.165,
    },
    "China Black Monday (2015)": {
        "window": "Aug 2015",
        "benchmark_return": -0.110,
    },
    "Q4 2018 Selloff": {
        "window": "Oct 2018 to Dec 2018",
        "benchmark_return": -0.198,
    },
    "COVID-19 Crash (2020)": {
        "window": "Feb 2020 to Mar 2020",
        "benchmark_return": -0.339,
    },
    "2022 Bear Market": {
        "window": "Jan 2022 to Oct 2022",
        "benchmark_return": -0.254,
    },
}


def run_stress_test(
    weights: Dict[str, float],
    betas: pd.Series,
    portfolio_value: float,
    scenarios: Optional[Dict[str, Dict]] = None,
) -> pd.DataFrame:
    """Estimate portfolio loss under each historical stress scenario.

    Parameters
    ----------
    weights : dict[str, float]
        Portfolio weights by ticker.
    betas : pd.Series
        Beta of each ticker vs. the chosen benchmark (see
        `Portfolio.asset_beta`).
    portfolio_value : float
        Notional portfolio value.
    scenarios : dict, optional
        Override the default `HISTORICAL_SCENARIOS` table.

    Returns
    -------
    pd.DataFrame
        One row per scenario: benchmark shock, beta-adjusted portfolio
        shock return, and estimated dollar loss, sorted worst-to-least-bad.
    """
    scenarios = scenarios or HISTORICAL_SCENARIOS
    rows = []
    for name, info in scenarios.items():
        bench_ret = info["benchmark_return"]
        port_ret = sum(weights[t] * betas.get(t, 1.0) * bench_ret for t in weights)
        rows.append(
            {
                "scenario": name,
                "window": info["window"],
                "benchmark_shock": bench_ret,
                "portfolio_shock_return": port_ret,
                "estimated_loss": -port_ret * portfolio_value,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("estimated_loss", ascending=False)
        .reset_index(drop=True)
    )


def compare_to_var(stress_df: pd.DataFrame, var_values: Dict[str, float]) -> pd.DataFrame:
    """Flag stress scenarios whose estimated loss exceeds each VaR estimate.

    This directly answers the question the stress test is meant to ask:
    "which historical crashes were worse than what our VaR model said was
    a 1-in-20 (or 1-in-100) event?" -- i.e. quantifying fat-tail risk that
    standard VaR underestimates.

    Parameters
    ----------
    stress_df : pd.DataFrame
        Output of `run_stress_test`.
    var_values : dict[str, float]
        Mapping of a descriptive VaR label (e.g. "Parametric 99% 1d") to
        its dollar VaR value.
    """
    out = stress_df.copy()
    for label, var_value in var_values.items():
        out[f"exceeds_{label}"] = out["estimated_loss"] > var_value
    return out