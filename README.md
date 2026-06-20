# Value at Risk (VaR) Portfolio Risk Management System

A production-ready Python system for quantifying and stress-testing portfolio risk using multiple Value at Risk methodologies, Expected Shortfall (CVaR), and historical backtesting. Designed to identify fat-tail risk where normal distributions underestimate losses.

## Features

- **Three VaR Methodologies**
  - **Parametric (Variance-Covariance)**: Fast, closed-form estimate assuming normal returns
  - **Historical Simulation**: Non-parametric method using actual historical returns
  - **Monte Carlo Simulation**: Flexible approach supporting normal and fat-tailed (Student-t) distributions

- **Risk Metrics**
  - 1-day and 10-day Value at Risk at 95% and 99% confidence levels
  - Expected Shortfall (Conditional Value at Risk) for tail-risk quantification
  - Portfolio-level statistics (mean return, volatility, correlation matrix)

- **Volatility Modeling**
  - Exponential Weighted Moving Average (EWMA) with RiskMetrics decay (λ=0.94)
  - GARCH(1,1) with Student-t innovations for volatility clustering

- **Stress Testing**
  - 11 historical market crash scenarios (Black Monday, GFC 2008, COVID-19, etc.)
  - Beta-adjusted CAPM-style portfolio shocks per scenario
  - Identifies which historical crashes exceeded VaR estimates (fat-tail underestimation)

- **Backtesting & Validation**
  - Rolling-window out-of-sample VaR forecasts vs. realized returns
  - Kupiec POF (Proportion of Failures) test with likelihood-ratio statistic
  - Christoffersen independence test for breach clustering
  - Joint conditional coverage assessment

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/var-portfolio-risk-system.git
cd var-portfolio-risk-system

# Install dependencies
pip install -r requirements.txt
```

**Requirements**: Python 3.8+

### Dependencies
- `numpy`, `pandas`, `scipy` — core numerical and statistical computing
- `matplotlib`, `seaborn` — visualization
- `yfinance` — download live historical market data
- `arch` — GARCH modeling for volatility clustering
- `pytest`, `tabulate` — testing and CLI output formatting

## Quick Start

### Run Full Analysis (Live Data)
```bash
# Analyze a 5-stock portfolio (AAPL, MSFT, GOOGL, AMZN, JPM) with default settings
python scripts/run_analysis.py

# Custom tickers and weights
python scripts/run_analysis.py \
  --tickers AAPL MSFT TSLA \
  --weights 0.4 0.4 0.2 \
  --notional 500000 \
  --confidences 0.95 0.99 \
  --horizons 1 10
```

### Run Offline (Synthetic Data, No Internet Required)
```bash
# Useful for CI/CD or testing without internet access
python scripts/run_analysis.py --offline
```

### CLI Options
```
--tickers              Stock tickers (default: AAPL MSFT GOOGL AMZN JPM)
--weights             Portfolio weights (default: 0.25 0.25 0.20 0.15 0.15)
--benchmark           Benchmark ticker for beta/stress testing (default: SPY)
--start               Data start date (default: 2018-01-01)
--end                 Data end date (default: 2024-12-31)
--notional            Portfolio notional value in USD (default: 1,000,000)
--confidences         Confidence levels for VaR (default: 0.95 0.99)
--horizons            Forecast horizons in days (default: 1 10)
--n-sims              Monte Carlo simulations (default: 50,000)
--backtest-window     Rolling backtest window in days (default: 250)
--offline             Use synthetic data instead of yfinance
--outdir              Output directory for results (default: outputs)
```

## Methodology

### Parametric VaR
Assumes portfolio returns follow a normal distribution:

$$\text{VaR}_{\alpha} = \mu - z_{\alpha} \cdot \sigma$$

where $z_{\alpha}$ is the critical z-score, $\mu$ is mean return, and $\sigma$ is volatility. Multi-day VaR scales via $\sqrt{H}$ (square-root-of-time rule).

### Historical Simulation VaR
Non-parametric approach using actual historical returns:

1. Compute overlapping H-day portfolio returns (preferred) or scale 1-day VaR by $\sqrt{H}$
2. Sort returns in ascending order
3. Extract the $\alpha$-quantile (e.g., 5th percentile for 95% confidence)

### Monte Carlo VaR
Simulate correlated multi-day price paths:

1. Compute covariance matrix from historical returns
2. Cholesky decomposition for correlation structure
3. Generate N simulations of H-day returns:
   - **Normal**: $r_t = \mu + L \cdot z_t$ where $z_t \sim \mathcal{N}(0,1)$
   - **Student-t**: Heavy-tailed innovations fit to actual returns (captures fat tails)
4. Extract $\alpha$-quantile of simulated losses

### Expected Shortfall (CVaR)
Average loss in the tail beyond VaR:

$$\text{ES}_{\alpha} = \mathbb{E}[L \mid L > \text{VaR}_{\alpha}]$$

Always exceeds VaR at the same confidence level; captures tail severity.

### Fat-Tail Risk Detection
Fits Student-t distribution to portfolio returns and compares degrees-of-freedom (dof):
- **dof > 30**: Near-normal tails
- **dof 4–10**: Moderate fat tails
- **dof < 4**: Severe fat tails (e.g., 2008 crisis, COVID-19)

Monte Carlo with Student-t produces higher VaR estimates, revealing risk underestimation in parametric models during crisis periods.

## Sample Output

The `examples/sample_output/` directory contains results from a sample run:

### VaR Estimates
![VaR Comparison Across Methods](examples/sample_output/var_comparison.png)

Portfolio: AAPL 25%, MSFT 25%, GOOGL 20%, AMZN 15%, JPM 15% | Notional: $1,000,000 | Period: 2018–2024

| Confidence | Horizon | Parametric | Historical | Monte Carlo (Normal) | Monte Carlo (Student-t) |
|-----------|---------|-----------|-----------|---------------------|----------------------|
| 95% | 1 day | $24,396 | $18,486 | $24,421 | $22,793 |
| 95% | 10 days | $77,058 | $73,584 | $75,219 | $73,995 |
| 99% | 1 day | $34,508 | $30,920 | $34,493 | $39,223 |
| 99% | 10 days | $109,037 | $152,388 | $104,566 | $109,975 |

### Return Distribution & Fat Tails
![Return Distribution](examples/sample_output/return_distribution.png)

Histogram of daily portfolio returns showing actual distribution (blue) vs. fitted normal (red). Student-t fit (dof=4.19) better captures observed tail events.

### Stress Testing
![Stress Test Scenarios](examples/sample_output/stress_test.png)

Historical crash scenarios and estimated portfolio losses. Top 3 worst scenarios:
1. **Dot-com Crash (2000–2002)**: −$187,765 loss (exceeds all VaR methods → fat-tail underestimation)
2. **Global Financial Crisis (2008)**: −$157,109 loss
3. **COVID-19 (2020)**: −$129,903 loss

### Backtesting
![Rolling VaR Backtest](examples/sample_output/backtest.png)

Out-of-sample rolling 1-day VaR estimates vs. realized returns over 5+ years. Red dots = breaches (realized loss exceeded forecast). Clustering of breaches during volatile periods validates volatility-clustering concept (GARCH motivation).

## Project Structure

```
var-portfolio-risk-system/
├── README.md                    # This file
├── LICENSE                      # MIT license
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git exclude patterns
│
├── var_risk/                    # Core library
│   ├── __init__.py
│   ├── data_loader.py          # yfinance + synthetic data generation
│   ├── portfolio.py            # Portfolio class: weights, returns, covariance
│   ├── volatility.py           # EWMA, GARCH volatility models
│   ├── var_models.py           # Parametric, Historical, Monte Carlo VaR
│   ├── expected_shortfall.py   # CVaR/ES computation
│   ├── stress_testing.py       # 11 historical crash scenarios
│   ├── backtesting.py          # Rolling VaR forecast vs. realized returns
│   └── visualization.py        # Matplotlib plotting functions
│
├── scripts/
│   └── run_analysis.py         # CLI entry point (argparse)
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # pytest fixtures
│   ├── test_portfolio.py       # Portfolio class tests
│   ├── test_var_models.py      # VaR method tests (accuracy, scaling)
│   ├── test_expected_shortfall.py
│   ├── test_backtesting.py     # Kupiec/Christoffersen tests
│   └── test_stress_testing.py
│
├── examples/
│   └── sample_output/          # Example results
│       ├── var_summary.csv
│       ├── stress_test_results.csv
│       ├── var_comparison.png
│       ├── return_distribution.png
│       ├── backtest.png
│       └── stress_test.png
│
└── .github/workflows/
    └── tests.yml               # GitHub Actions CI/CD
```

## Testing

Run the full test suite (29 tests):

```bash
pytest tests/ -v
```

Tests cover:
- Portfolio construction and return calculations
- VaR accuracy across all three methods
- Monte Carlo convergence (normal vs. Student-t)
- Expected Shortfall tail-averaging logic
- Stress scenario application and beta adjustments
- Kupiec POF and Christoffersen independence tests
- Rolling backtest breach detection

All tests pass with both live and synthetic data.

## Data & Limitations

### Data Sources
- **Live**: `yfinance` downloads daily OHLCV data from Yahoo Finance
- **Offline**: Synthetic GBM price paths with injected fat-tail shock days (for testing without internet)

### Important Disclaimers

1. **Stress Test Scenarios**: The 11 historical crash scenarios in `var_risk/stress_testing.py` are illustrative approximations (e.g., −20.5% for Black Monday 1987). For production risk management, replace with verified vendor data (e.g., Bloomberg, FactSet) or official exchange records.

2. **Normal Market Assumption**: Historical data used for VaR/ES assumes past volatility and correlations persist. During regime shifts or unprecedented shocks (e.g., 2008, COVID-19), models may underestimate risk.

3. **Fat-Tail Detection**: Student-t fitting improves tail estimates but is not a silver bullet. Extreme-value theory (EVT) and advanced copula models may be needed for tail dependencies in multi-asset portfolios.

4. **Backtesting Limitations**: A few years of backtest data has limited statistical power. Kupiec POF test may not reject poor models if breach count falls within ~10% confidence band.

5. **Regulatory Compliance**: This system is for educational and research purposes. For regulatory VaR (Basel III, Dodd-Frank), consult official guidance and model validation frameworks.

## Extensions & Future Work

- Implicit funding cost modeling and liquidity adjustments
- Copula-based multi-asset tail dependence
- Extreme value theory (EVT) for non-parametric tail quantiles
- Jump-diffusion and regime-switching models
- Real-time dashboard with WebSocket data feeds
- Integration with risk database backends

## License

MIT License — see [LICENSE](LICENSE) for details.

## References

- **Dowd, K.** (2007). *Measuring Market Risk*. John Wiley & Sons.
- **Jorion, P.** (2006). *Value at Risk: The New Benchmark for Managing Financial Risk*. McGraw-Hill.
- **Kupiec, P.** (1995). "Techniques for Verifying the Accuracy of Risk Measurement Models." *Journal of Derivatives*, 3(2), 73–84.
- **Christoffersen, P.** (2011). *Elements of Financial Risk Management*. Academic Press.
- **RiskMetrics** (1996). *Technical Document*. J.P. Morgan & Reuters.

---

**Questions or contributions?** Open an issue or PR on GitHub.
