import numpy as np

from var_risk.var_models import (
    parametric_var,
    historical_var,
    monte_carlo_var,
    fit_student_t,
)


def test_parametric_var_is_positive(synthetic_portfolio):
    p = synthetic_portfolio
    result = parametric_var(p.portfolio_mean, p.portfolio_std, 0.95, 1, p.notional)
    assert result.var_value > 0
    assert result.var_return > 0


def test_var_increases_with_confidence(synthetic_portfolio):
    p = synthetic_portfolio
    v95 = parametric_var(p.portfolio_mean, p.portfolio_std, 0.95, 1, p.notional)
    v99 = parametric_var(p.portfolio_mean, p.portfolio_std, 0.99, 1, p.notional)
    assert v99.var_value > v95.var_value


def test_parametric_var_scales_with_sqrt_time(synthetic_portfolio):
    p = synthetic_portfolio
    v1 = parametric_var(p.portfolio_mean, p.portfolio_std, 0.95, 1, p.notional)
    v10 = parametric_var(p.portfolio_mean, p.portfolio_std, 0.95, 10, p.notional)
    assert v10.var_value > v1.var_value
    np.testing.assert_allclose(v10.extra["sigma_h"], v1.extra["sigma_h"] * np.sqrt(10))


def test_historical_var_matches_empirical_quantile(synthetic_portfolio):
    p = synthetic_portfolio
    result = historical_var(p.portfolio_returns, 0.95, 1, p.notional)
    expected_return = -np.percentile(p.portfolio_returns, 5)
    np.testing.assert_allclose(result.var_value, expected_return * p.notional, rtol=1e-6)


def test_historical_var_overlapping_horizon_runs(synthetic_portfolio):
    p = synthetic_portfolio
    result = historical_var(p.portfolio_returns, 0.95, 10, p.notional, scaling="overlapping")
    assert result.var_value > 0


def test_monte_carlo_normal_close_to_parametric(synthetic_portfolio):
    p = synthetic_portfolio
    mc = monte_carlo_var(
        p.mean_returns.values, p.cov_matrix.values, p.weight_vector,
        0.95, 1, p.notional, n_simulations=20_000, distribution="normal", seed=1,
    )
    pvar = parametric_var(p.portfolio_mean, p.portfolio_std, 0.95, 1, p.notional)
    # Same distributional family (Normal) -> Monte Carlo should be close to
    # the closed-form parametric result, within simulation noise.
    assert abs(mc.var_value - pvar.var_value) / pvar.var_value < 0.25


def test_monte_carlo_student_t_runs(synthetic_portfolio):
    p = synthetic_portfolio
    dof, _, _ = fit_student_t(p.portfolio_returns)
    mc = monte_carlo_var(
        p.mean_returns.values, p.cov_matrix.values, p.weight_vector,
        0.99, 1, p.notional, n_simulations=20_000, distribution="t", t_dof=dof, seed=2,
    )
    assert mc.var_value > 0
    assert "simulated_returns" in mc.extra


def test_monte_carlo_invalid_distribution_raises(synthetic_portfolio):
    p = synthetic_portfolio
    try:
        monte_carlo_var(
            p.mean_returns.values, p.cov_matrix.values, p.weight_vector,
            0.95, 1, p.notional, distribution="bogus",
        )
        assert False, "expected a ValueError"
    except ValueError:
        pass


def test_fit_student_t_runs(synthetic_portfolio):
    p = synthetic_portfolio
    dof, loc, scale = fit_student_t(p.portfolio_returns)
    assert dof > 0
    assert scale > 0