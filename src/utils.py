"""
Utility functions for data manipulation and analysis.
"""

import pandas as pd
import numpy as np
from .config import TRADING_DAYS_PER_YEAR


def rolling_volatility(returns, window=21):
    """
    Compute rolling annualized volatility.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Returns data with dates as index and tickers as columns
    window : int
        Rolling window size in days
    
    Returns:
    --------
    pd.DataFrame
        Rolling annualized volatility
    """
    if returns.empty or len(returns) < window:
        return pd.DataFrame()
    
    # Calculate rolling standard deviation
    rolling_std = returns.rolling(window=window).std()
    
    # Annualize
    rolling_vol = rolling_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    return rolling_vol


def align_dataframes(df1, df2):
    """Align two DataFrames on their index."""
    combined = pd.concat([df1, df2], axis=1, join='inner')
    midpoint = len(df1.columns)
    return combined.iloc[:, :midpoint], combined.iloc[:, midpoint:]


def format_percent(value, decimals=2):
    """Format decimal as percentage."""
    return f"{value:.{decimals}%}"


def clean_prices(prices):
    """
    Clean price data by removing outliers and invalid values.
    """
    cleaned = prices.copy()
    
    for col in cleaned.columns:
        # Remove values that are clearly wrong (> $10,000 for single stocks)
        cleaned[col] = cleaned[col].where(cleaned[col] < 10000)
        cleaned[col] = cleaned[col].where(cleaned[col] > 0)
    
    return cleaned