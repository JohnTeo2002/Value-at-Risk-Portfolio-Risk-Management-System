"""
Matplotlib plotting helpers for the risk engine. Every function returns
a `matplotlib.figure.Figure` so callers can display it inline (e.g. in a
notebook) or save it to disk.
"""
from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["figure.dpi"] = 110


def plot_return_distribution(
    returns: pd.Series,
    var_levels: Dict[str, float],
    es_levels: Optional[Dict[str, float]] = None,
    title: str = "Portfolio Return Distribution vs. VaR / ES",
):
    """Histogram of returns with VaR (dashed) and ES (dotted) lines overlaid.

    Parameters
    ----------
    returns : pd.Series
        Historical (or simulated) return sample.
    var_levels : dict[str, float]
        Mapping of method label -> VaR as a return fraction (positive = loss).
    es_levels : dict[str, float], optional
        Mapping of method label -> ES as a return fraction.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(returns, bins=80, density=True, alpha=0.55, color="#3b6fa0", label="Empirical returns")

    n_lines = max(len(var_levels), 1)
    colors = plt.cm.tab10(np.linspace(0, 1, n_lines))

    for (label, var_ret), color in zip(var_levels.items(), colors):
        ax.axvline(-var_ret, color=color, linestyle="--", linewidth=1.8, label=f"{label} VaR")

    if es_levels:
        for (label, es_ret), color in zip(es_levels.items(), colors):
            ax.axvline(-es_ret, color=color, linestyle=":", linewidth=1.8, label=f"{label} ES")

    ax.set_title(title)
    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def plot_var_method_comparison(results_df: pd.DataFrame, title: str = "VaR by Method, Confidence & Horizon"):
    """Grouped bar chart comparing dollar VaR across methods.

    Parameters
    ----------
    results_df : pd.DataFrame
        Rows = confidence/horizon combos, columns = methods, values = $ VaR.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    results_df.plot(kind="bar", ax=ax, width=0.78)
    ax.set_title(title)
    ax.set_ylabel("VaR ($)")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_backtest(backtest_df: pd.DataFrame, title: str = "VaR Backtest: Forecast vs. Realized Returns"):
    """Line chart of realized returns vs. the rolling -VaR threshold, with breaches marked."""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(backtest_df.index, backtest_df["actual_return"], color="#444444", linewidth=0.8, label="Realized return")
    ax.plot(backtest_df.index, -backtest_df["var_forecast"], color="#d62728", linewidth=1.3, label="-VaR threshold")

    breaches = backtest_df[backtest_df["breach"]]
    ax.scatter(breaches.index, breaches["actual_return"], color="red", s=30, zorder=5, label="Breach")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(title)
    ax.set_ylabel("Daily return")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_stress_scenarios(stress_df: pd.DataFrame, title: str = "Stress Test: Estimated Loss by Historical Scenario"):
    """Horizontal bar chart of estimated dollar loss per stress scenario."""
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ordered = stress_df.sort_values("estimated_loss")
    ax.barh(ordered["scenario"], ordered["estimated_loss"], color="#a83232")
    ax.set_title(title)
    ax.set_xlabel("Estimated loss ($)")
    fig.tight_layout()
    return fig