"""
Unit tests for risk metrics calculation module.

Tests ensure that all risk metrics are calculated correctly
and handle edge cases appropriately.
"""

import pytest
import pandas as pd
import numpy as np
from src.risk_metrics import (
    value_at_risk, 
    conditional_var, 
    beta, 
    maximum_drawdown,
    annualized_volatility,
    sharpe_ratio,
    calculate_returns
)


def test_value_at_risk_basic():
    """Test VaR calculation with simple return series."""
    returns = pd.Series([-0.02, -0.01, 0.0, 0.01, 0.02])
    var_95 = value_at_risk(returns, confidence=0.95)
    
    # 5th percentile of [-0.02, -0.01, 0.0, 0.01, 0.02] = -0.02
    assert var_95 == -0.02
    assert isinstance(var_95, float)


def test_value_at_risk_different_confidence():
    """Test VaR with different confidence levels."""
    returns = pd.Series([-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04])
    
    var_90 = value_at_risk(returns, confidence=0.90)
    var_95 = value_at_risk(returns, confidence=0.95)
    var_99 = value_at_risk(returns, confidence=0.99)
    
    # Higher confidence should yield more negative (conservative) VaR
    assert var_90 >= var_95 >= var_99  # Less negative to more negative


def test_conditional_var_calculation():
    """Test CVaR (Expected Shortfall) calculation."""
    returns = pd.Series([-0.05, -0.04, -0.01, 0.02, 0.03])
    cvar_95 = conditional_var(returns, confidence=0.95)
    
    # VaR at 95% = -0.04 (2nd smallest)
    # CVaR = average of returns <= -0.04 = (-0.05 + -0.04)/2 = -0.045
    assert cvar_95 == -0.045
    

def test_conditional_var_more_conservative_than_var():
    """Test that CVaR is always more conservative (lower) than VaR."""
    returns = pd.Series(np.random.normal(0, 0.01, 1000))
    
    var = value_at_risk(returns, confidence=0.95)
    cvar = conditional_var(returns, confidence=0.95)
    
    # CVaR should be less than or equal to VaR (more negative)
    assert cvar <= var


def test_beta_perfect_correlation():
    """Test Beta when asset perfectly correlates with market."""
    asset = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
    market = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
    
    assert beta(asset, market) == 1.0


def test_beta_twice_market_volatility():
    """Test Beta when asset is twice as volatile as market."""
    asset = pd.Series([0.02, 0.04, 0.06, 0.08, 0.10])
    market = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
    
    # Asset returns are exactly double market returns
    assert beta(asset, market) == 2.0


def test_beta_negative_correlation():
    """Test Beta when asset moves opposite to market."""
    asset = pd.Series([-0.01, -0.02, -0.03, -0.04, -0.05])
    market = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
    
    assert beta(asset, market) == -1.0


def test_maximum_drawdown_simple_case():
    """Test maximum drawdown calculation with simple price series."""
    prices = pd.Series([100, 110, 105, 95, 120])
    result = maximum_drawdown(prices)
    
    # Peak at 110, trough at 95: drawdown = (95-110)/110 = -13.64%
    assert round(result['max_drawdown'], 4) == -0.1364
    assert result['start_date'] == 1  # Index of 110
    assert result['end_date'] == 3     # Index of 95


def test_maximum_drawdown_no_drawdown():
    """Test when prices only go up (no drawdown)."""
    prices = pd.Series([100, 110, 120, 130, 140])
    result = maximum_drawdown(prices)
    
    assert result['max_drawdown'] == 0.0


def test_maximum_drawdown_multiple_drawdowns():
    """Test that function finds the maximum drawdown among several."""
    prices = pd.Series([100, 120, 110, 130, 100, 90, 110])
    result = maximum_drawdown(prices)
    
    # Largest drawdown: from 130 to 90 = (90-130)/130 = -30.77%
    assert round(result['max_drawdown'], 4) == -0.3077


def test_annualized_volatility():
    """Test annualized volatility calculation."""
    # Constant daily returns of 1%
    returns = pd.Series([0.01] * 252)
    vol = annualized_volatility(returns)
    
    # Standard deviation of constant returns is 0
    assert vol == 0.0


def test_sharpe_ratio_positive():
    """Test Sharpe ratio with positive excess returns."""
    returns = pd.Series([0.001] * 252)  # Consistent positive returns
    sr = sharpe_ratio(returns, risk_free_rate=0.02)
    
    # Should be positive
    assert sr > 0


def test_sharpe_ratio_negative():
    """Test Sharpe ratio with negative excess returns."""
    returns = pd.Series([-0.001] * 252)  # Consistent negative returns
    sr = sharpe_ratio(returns, risk_free_rate=0.02)
    
    # Should be negative (underperforms risk-free rate)
    assert sr < 0


def test_calculate_returns():
    """Test log return calculation."""
    prices = pd.Series([100, 101, 102, 101, 100])
    returns = calculate_returns(prices)
    
    # Log returns should sum to total log return
    total_log_return = np.log(100/100)  # 0 from start to end
    assert round(returns.sum(), 10) == round(total_log_return, 10)


def test_calculate_returns_length():
    """Test that returns have one fewer observation than prices."""
    prices = pd.Series([100, 101, 102, 103, 104])
    returns = calculate_returns(prices)
    
    assert len(returns) == len(prices) - 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])