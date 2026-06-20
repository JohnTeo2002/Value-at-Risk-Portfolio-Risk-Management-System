#!/usr/bin/env python
"""
End-to-end pipeline: load data -> build portfolio -> compute VaR (3 methods)
at 95%/99% confidence and 1-day/10-day horizons -> compute Expected
Shortfall -> run historical stress tests -> backtest the models -> save a
full set of plots and CSV summaries.

Usage
-----
    # Live data (requires internet access for yfinance)
    python scripts/run_analysis.py \\
        --tickers AAPL MSFT GOOGL AMZN JPM \\
        --weights 0.25 0.25 0.20 0.15 0.15 \\
        --start 2018-01-01 --end 2024-12-31

    # Offline / demo mode using synthetic correlated price data
    python scripts/run_analysis.py --offline

Run `python scripts/run_analysis.py --help` for all options.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from var_risk.data_loader import download_price_data, generate_synthetic_prices
from var_risk.portfolio import Portfolio
from var_risk.var_models import parametric_var, historical_var, monte_carlo_var, fit_student_t
from var_risk.expected_shortfall import parametric_es, empirical_es
from var_risk.stress_testing import run_stress_test, compare_to_var
from var_risk.backtesting import rolling_var_backtest, evaluate_backtest
from var_risk.visualization import (
    plot_return_distribution,
    plot_var_method_comparison,
    plot_backtest,
    plot_stress_scenarios,
)


def parse_args():
    p = argparse.ArgumentParser(description="Run the full VaR / ES / stress-test / backtest pipeline.")
    p.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"])
    p.add_argument("--weights", nargs="+", type=float, default=[0.25, 0.25, 0.20, 0.15, 0.15])
    p.add_argument("--benchmark", default="SPY", help="Benchmark ticker used for beta / stress testing.")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--notional", type=float, default=1_000_000.0)
    p.add_argument("--confidences", nargs="+", type=float, default=[0.95, 0.99])
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 10])
    p.add_argument("--n-sims", type=int, default=50_000)
    p.add_argument("--backtest-window", type=int, default=250)
    p.add_argument("--offline", action="store_true", help="Use synthetic data; no internet/yfinance required.")
    p.add_argument("--outdir", default="outputs")
    return p.parse_args()


def load_data(args) -> pd.DataFrame:
    all_symbols = args.tickers + [args.benchmark]
    if args.offline:
        print("[offline mode] generating synthetic correlated price data ...")
        return generate_synthetic_prices(all_symbols, n_days=1500, seed=42)

    try:
        print(f"Downloading live price data for {all_symbols} ({args.start} to {args.end}) ...")
        return download_price_data(all_symbols, args.start, args.end)
    except Exception as exc:
        print(f"[warning] live data download failed ({exc!r}); falling back to synthetic data.")
        return generate_synthetic_prices(all_symbols, n_days=1500, seed=42)


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    weights = dict(zip(args.tickers, args.weights))

    prices = load_data(args)
    benchmark_returns = prices[args.benchmark].pct_change().dropna()
    asset_prices = prices[args.tickers]

    portfolio = Portfolio(prices=asset_prices, weights=weights, notional=args.notional)
    print("\nPortfolio summary:")
    for k, v in portfolio.summary().items():
        print(f"  {k}: {v}")

    # ---------------------------------------------------------------- VaR
    dof, _, _ = fit_student_t(portfolio.portfolio_returns)
    print(f"\nFitted Student-t degrees of freedom on portfolio returns: {dof:.2f} "
          f"({'fat tails detected' if dof < 15 else 'close to Normal'})")

    all_results = {}
    for conf in args.confidences:
        for h in args.horizons:
            label = f"{int(conf * 100)}%_{h}d"
            use_overlapping = h > 1 and len(portfolio.portfolio_returns) > h + 250

            pvar = parametric_var(portfolio.portfolio_mean, portfolio.portfolio_std, conf, h, portfolio.notional)
            hvar = historical_var(
                portfolio.portfolio_returns, conf, h, portfolio.notional,
                scaling="overlapping" if use_overlapping else "sqrt_time",
            )
            mvar_n = monte_carlo_var(
                portfolio.mean_returns.values, portfolio.cov_matrix.values, portfolio.weight_vector,
                conf, h, portfolio.notional, n_simulations=args.n_sims, distribution="normal", seed=1,
            )
            mvar_t = monte_carlo_var(
                portfolio.mean_returns.values, portfolio.cov_matrix.values, portfolio.weight_vector,
                conf, h, portfolio.notional, n_simulations=args.n_sims, distribution="t", t_dof=dof, seed=1,
            )

            all_results[label] = {
                "Parametric": pvar,
                "Historical": hvar,
                "Monte Carlo (Normal)": mvar_n,
                "Monte Carlo (Student-t)": mvar_t,
            }

            print(f"\n--- {conf:.0%} confidence / {h}-day horizon ---")
            for name, r in all_results[label].items():
                print(f"  {name:<24s}: {r}")

    # -------------------------------------------------------- Exp. Shortfall
    print("\nExpected Shortfall (1-day horizon):")
    es_summary = {}
    for conf in args.confidences:
        p_es = parametric_es(portfolio.portfolio_mean, portfolio.portfolio_std, conf, 1, portfolio.notional)
        h_es = empirical_es(portfolio.portfolio_returns.values, conf, portfolio.notional, "Historical")
        mc_returns = all_results[f"{int(conf * 100)}%_1d"]["Monte Carlo (Student-t)"].extra["simulated_returns"]
        mc_es = empirical_es(mc_returns, conf, portfolio.notional, "Monte Carlo (Student-t)")
        es_summary[conf] = {"Parametric": p_es, "Historical": h_es, "Monte Carlo (Student-t)": mc_es}
        print(f"  {conf:.0%} -> Parametric=${p_es['es_value']:,.0f}  "
              f"Historical=${h_es['es_value']:,.0f}  "
              f"MonteCarlo(t)=${mc_es['es_value']:,.0f}")

    # ------------------------------------------------------------ Stress test
    betas = portfolio.asset_beta(benchmark_returns)
    print("\nEstimated asset betas vs. benchmark:\n", betas.to_string())

    stress_df = run_stress_test(weights, betas, portfolio.notional)
    var_99_1d = {name: r.var_value for name, r in all_results["99%_1d"].items()}
    stress_df = compare_to_var(stress_df, var_99_1d)
    print("\nStress Test Results (vs. 99% 1-day VaR estimates):\n", stress_df.to_string(index=False))

    # ------------------------------------------------------------- Backtest
    print("\nBacktesting (rolling, out-of-sample):")
    backtest_summaries = {}
    cached_backtests = {}
    for conf in args.confidences:
        for method in ("parametric", "historical"):
            bt_df = rolling_var_backtest(portfolio.portfolio_returns, conf, window=args.backtest_window, method=method)
            result = evaluate_backtest(bt_df, conf)
            key = f"{method}_{int(conf * 100)}"
            backtest_summaries[key] = result
            cached_backtests[key] = bt_df
            print(f"  [{method:<10s} {conf:.0%}] {result}")

    # ----------------------------------------------------------------- Plots
    var_table = pd.DataFrame(
        {label: {name: r.var_value for name, r in methods.items()} for label, methods in all_results.items()}
    ).T

    fig1 = plot_var_method_comparison(var_table)
    fig1.savefig(os.path.join(args.outdir, "var_comparison.png"), dpi=150)

    first_conf_label = f"{int(args.confidences[0] * 100)}%_1d"
    fig2 = plot_return_distribution(
        portfolio.portfolio_returns,
        var_levels={name: r.var_return for name, r in all_results[first_conf_label].items()},
    )
    fig2.savefig(os.path.join(args.outdir, "return_distribution.png"), dpi=150)

    fig3 = plot_backtest(cached_backtests[f"parametric_{int(args.confidences[0] * 100)}"])
    fig3.savefig(os.path.join(args.outdir, "backtest.png"), dpi=150)

    fig4 = plot_stress_scenarios(stress_df)
    fig4.savefig(os.path.join(args.outdir, "stress_test.png"), dpi=150)

    var_table.to_csv(os.path.join(args.outdir, "var_summary.csv"))
    stress_df.to_csv(os.path.join(args.outdir, "stress_test_results.csv"), index=False)

    print(f"\nDone. All plots and CSV summaries saved to ./{args.outdir}/")


if __name__ == "__main__":
    main()