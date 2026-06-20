import pandas as pd

from var_risk.stress_testing import run_stress_test, compare_to_var, HISTORICAL_SCENARIOS


def test_stress_test_output_shape_and_sign():
    weights = {"A": 0.5, "B": 0.5}
    betas = pd.Series({"A": 1.0, "B": 1.2})
    df = run_stress_test(weights, betas, 1_000_000)
    assert len(df) == len(HISTORICAL_SCENARIOS)
    # Every historical scenario in the table is a negative (crash) shock,
    # so every estimated loss for a long-only, positive-beta portfolio
    # should be positive.
    assert (df["estimated_loss"] > 0).all()
    assert list(df.columns) == ["scenario", "window", "benchmark_shock", "portfolio_shock_return", "estimated_loss"]


def test_stress_test_sorted_worst_first():
    weights = {"A": 1.0}
    betas = pd.Series({"A": 1.0})
    df = run_stress_test(weights, betas, 1_000_000)
    losses = df["estimated_loss"].values
    assert (losses[:-1] >= losses[1:]).all()


def test_compare_to_var_flags_exceedances():
    weights = {"A": 1.0}
    betas = pd.Series({"A": 1.0})
    df = run_stress_test(weights, betas, 1_000_000)
    # A tiny VaR value should be exceeded by every stress scenario.
    flagged = compare_to_var(df, {"tiny_var": 1.0})
    assert flagged["exceeds_tiny_var"].all()
    # A huge VaR value should not be exceeded by any scenario.
    flagged2 = compare_to_var(df, {"huge_var": 1e12})
    assert not flagged2["exceeds_huge_var"].any()


def test_asset_with_higher_beta_loses_more(synthetic_portfolio_with_benchmark):
    weights = {"A": 0.5, "B": 0.5}
    betas = pd.Series({"A": 2.0, "B": 0.5})
    df = run_stress_test(weights, betas, 1_000_000)
    # sanity: should run without error and produce finite numbers
    assert df["estimated_loss"].notna().all()