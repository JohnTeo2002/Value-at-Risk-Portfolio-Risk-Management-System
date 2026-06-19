"""
A multi-method portfolio risk engine implementing Value at Risk (VaR),
Expected Shortfall (CVaR), historical stress testing, and statistical
backtesting for diversified equity portfolios.

Modules
-------
data_loader        : Price data acquisition (live via yfinance, or synthetic)
portfolio           : Portfolio construction, returns, covariance
volatility          : EWMA / GARCH(1,1) conditional volatility (vol clustering)
var_models          : Parametric, Historical Simulation, Monte Carlo VaR
expected_shortfall  : Expected Shortfall / Conditional VaR
stress_testing      : Historical crash-scenario stress testing
backtesting         : Rolling backtests + Kupiec / Christoffersen tests
visualization       : Plotting helpers for all of the above
"""

from .var_risk.portfolio import Portfolio
from .var_risk.var_models import VaRResult, parametric_var, historical_var, monte_carlo_var, fit_student_t
from .var_risk.expected_shortfall import parametric_es, empirical_es

__all__ = [
    "Portfolio",
    "VaRResult",
    "parametric_var",
    "historical_var",
    "monte_carlo_var",
    "fit_student_t",
    "parametric_es",
    "empirical_es",
]

__version__ = "1.0.0"