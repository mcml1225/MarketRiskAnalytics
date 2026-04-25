"""
Core risk metrics calculation module.

Implements standard market risk measures including:
- Value at Risk (VaR)
- Expected Shortfall / Conditional VaR (CVaR)
- Beta (systematic risk)
- Maximum Drawdown (MDD)
- Volatility
- Sharpe Ratio

All metrics are calculated using industry-standard methodologies.
"""

import numpy as np
import pandas as pd
from scipy import stats
from .config import TRADING_DAYS_PER_YEAR, RISK_FREE_RATE


def calculate_returns(prices):
    """
    Calculate daily log returns from price data.
    
    Parameters:
    -----------
    prices : pd.DataFrame or pd.Series
        Price data with dates as index
    
    Returns:
    --------
    pd.DataFrame or pd.Series
        Daily log returns, first row is NaN (dropped automatically)
    """
    return np.log(prices / prices.shift(1)).dropna()


def annualized_volatility(returns, periods_per_year=TRADING_DAYS_PER_YEAR):
    """
    Calculate annualized volatility from daily returns.
    
    Parameters:
    -----------
    returns : pd.Series
        Daily returns series
    periods_per_year : int, optional
        Number of trading periods in a year (default: 252)
    
    Returns:
    --------
    float
        Annualized volatility as a decimal (e.g., 0.25 = 25%)
    """
    return returns.std() * np.sqrt(periods_per_year)


def value_at_risk(returns, confidence=0.95, method='historical'):
    """
    Calculate Value at Risk (VaR) using historical simulation.
    
    Parameters:
    -----------
    returns : pd.Series
        Daily returns series
    confidence : float, optional
        Confidence level between 0 and 1 (default: 0.95)
    method : str, optional
        Calculation method, only 'historical' implemented
    
    Returns:
    --------
    float
        VaR as a negative decimal (e.g., -0.02 = -2%)
    """
    if method == 'historical':
        var = np.percentile(returns, (1 - confidence) * 100)
        return var
    else:
        raise ValueError("Only 'historical' method is supported.")


def conditional_var(returns, confidence=0.95):
    """
    Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
    
    Parameters:
    -----------
    returns : pd.Series
        Daily returns series
    confidence : float, optional
        Confidence level between 0 and 1 (default: 0.95)
    
    Returns:
    --------
    float
        CVaR as a negative decimal (e.g., -0.03 = -3%)
    """
    var = value_at_risk(returns, confidence)
    tail_losses = returns[returns <= var]
    cvar = tail_losses.mean()
    return cvar


def beta(asset_returns, market_returns):
    """
    Calculate Beta, a measure of systematic risk.
    
    Parameters:
    -----------
    asset_returns : pd.Series
        Asset daily returns (single column/Series)
    market_returns : pd.Series
        Market benchmark daily returns (single column/Series)
    
    Returns:
    --------
    float
        Beta coefficient
    """
    # Ensure both are Series, not DataFrames
    if isinstance(asset_returns, pd.DataFrame):
        asset_returns = asset_returns.iloc[:, 0]
    if isinstance(market_returns, pd.DataFrame):
        market_returns = market_returns.iloc[:, 0]
    
    # Calculate covariance and variance
    covariance = np.cov(asset_returns, market_returns)[0, 1]
    market_variance = np.var(market_returns)
    
    if market_variance == 0:
        return 0.0
    
    return covariance / market_variance


def maximum_drawdown(prices):
    """
    Calculate Maximum Drawdown (MDD) and its characteristics.
    
    Parameters:
    -----------
    prices : pd.Series
        Price series with dates as index
    
    Returns:
    --------
    dict
        Dictionary containing max_drawdown, start_date, end_date, duration_days
    """
    cumulative_max = prices.expanding().max()
    drawdown = (prices - cumulative_max) / cumulative_max
    max_dd = drawdown.min()
    
    end_idx = drawdown.idxmin()
    start_idx = (prices[:end_idx] == cumulative_max[:end_idx].max()).idxmax()
    
    duration = None
    if hasattr(end_idx, 'days') and hasattr(start_idx, 'days'):
        duration = (end_idx - start_idx).days
    elif hasattr(end_idx, 'date') and hasattr(start_idx, 'date'):
        duration = (end_idx - start_idx).days
    
    return {
        'max_drawdown': max_dd,
        'start_date': start_idx,
        'end_date': end_idx,
        'duration_days': duration
    }


def sharpe_ratio(returns, risk_free_rate=RISK_FREE_RATE, periods_per_year=TRADING_DAYS_PER_YEAR):
    """
    Calculate annualized Sharpe Ratio.
    
    Parameters:
    -----------
    returns : pd.Series
        Daily returns series
    risk_free_rate : float, optional
        Annualized risk-free rate (default: 0.02 = 2%)
    periods_per_year : int, optional
        Number of trading periods in a year (default: 252)
    
    Returns:
    --------
    float
        Annualized Sharpe ratio
    """
    daily_rf = risk_free_rate / periods_per_year
    excess_returns = returns - daily_rf
    
    if returns.std() == 0:
        return 0.0
    
    annualized_sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()
    return annualized_sharpe


def summarize_risk_metrics(prices, benchmark_prices, confidence=0.95, risk_free_rate=RISK_FREE_RATE):
    """
    Generate a comprehensive summary DataFrame of all risk metrics.
    
    Parameters:
    -----------
    prices : pd.DataFrame
        Asset prices with columns as tickers
    benchmark_prices : pd.Series or pd.DataFrame
        Benchmark index prices
    confidence : float, optional
        Confidence level for VaR and CVaR (default: 0.95)
    risk_free_rate : float, optional
        Annualized risk-free rate (default: 0.02)
    
    Returns:
    --------
    pd.DataFrame
        Formatted DataFrame with metrics
    """
    # Calculate returns for assets
    asset_returns = calculate_returns(prices)
    
    # Ensure benchmark is a Series and calculate its returns
    if isinstance(benchmark_prices, pd.DataFrame):
        benchmark_series = benchmark_prices.iloc[:, 0]
    else:
        benchmark_series = benchmark_prices
    
    benchmark_returns = calculate_returns(benchmark_series)
    
    results = []
    
    for ticker in prices.columns:
        # Extract returns for this asset (as Series)
        rets = asset_returns[ticker].dropna()
        
        # Align dates with benchmark for Beta calculation
        common_dates = rets.index.intersection(benchmark_returns.index)
        rets_aligned = rets[common_dates]
        bench_aligned = benchmark_returns[common_dates]
        
        # Calculate all metrics
        vol = annualized_volatility(rets)
        var = value_at_risk(rets, confidence)
        cvar = conditional_var(rets, confidence)
        b = beta(rets_aligned, bench_aligned)
        mdd = maximum_drawdown(prices[ticker])['max_drawdown']
        sr = sharpe_ratio(rets, risk_free_rate)
        
        # Store formatted results
        results.append({
            'Ticker': ticker,
            'Volatility (Ann.)': f"{vol:.2%}",
            f'VaR ({int(confidence*100)}%)': f"{var:.2%}",
            f'CVaR ({int(confidence*100)}%)': f"{cvar:.2%}",
            'Beta': f"{b:.2f}",
            'Max Drawdown': f"{mdd:.2%}",
            'Sharpe Ratio': f"{sr:.2f}"
        })
    
    return pd.DataFrame(results)
