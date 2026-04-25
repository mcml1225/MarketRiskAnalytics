"""
Data fetching module for market data - CORRECTED VERSION
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from .config import START_DATE, END_DATE, BENCHMARK_TICKER


def get_stock_data(tickers, start=START_DATE, end=END_DATE):
    """
    Download adjusted closing prices for a list of stock tickers.
    """
    # Ensure tickers is a list
    if isinstance(tickers, str):
        tickers = [tickers]
    
    # Download data
    data = yf.download(
        tickers, 
        start=start, 
        end=end, 
        progress=False,
        auto_adjust=True
    )
    
    # Extract only Adjusted Close or Close prices
    if len(tickers) == 1:
        # Single ticker - returns DataFrame with 'Close' column
        if 'Close' in data.columns:
            result = pd.DataFrame(data['Close'])
            result.columns = tickers
        else:
            result = pd.DataFrame(data)
            if result.shape[1] == 1:
                result.columns = tickers
    else:
        # Multiple tickers - extract 'Close' from MultiIndex
        if 'Close' in data.columns.get_level_values(0):
            result = data['Close']
        else:
            result = data
    
    # Ensure we have a DataFrame with proper column names
    if isinstance(result, pd.Series):
        result = pd.DataFrame(result)
    
    # Remove any rows with all NaN values
    result = result.dropna(how='all')
    
    # Ensure numeric data
    result = result.astype(float)
    
    return result


def get_benchmark_data(benchmark=BENCHMARK_TICKER, start=START_DATE, end=END_DATE):
    """
    Download benchmark index data.
    """
    bench = yf.download(benchmark, start=start, end=end, progress=False, auto_adjust=True)
    
    if 'Close' in bench.columns:
        return bench['Close']
    elif len(bench.columns) == 1:
        return bench.iloc[:, 0]
    else:
        return bench


def get_all_data(tickers, benchmark=BENCHMARK_TICKER, include_macro=False):
    """
    Return asset prices and benchmark.
    """
    print(f"[INFO] Downloading data for: {tickers}")
    
    # Download asset prices
    prices = get_stock_data(tickers, START_DATE, END_DATE)
    
    # Download benchmark
    bench = get_benchmark_data(benchmark, START_DATE, END_DATE)
    
    # Align data on dates
    common_dates = prices.index.intersection(bench.index)
    
    if len(common_dates) == 0:
        print(f"[ERROR] No common dates found between assets and benchmark")
        return pd.DataFrame(), pd.Series(), None
    
    prices = prices.loc[common_dates]
    bench = bench.loc[common_dates]
    
    # Remove any remaining NaN values
    prices = prices.dropna()
    bench = bench.dropna()
    
    print(f"[INFO] Data range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"[INFO] Shape: {prices.shape}")
    
    return prices, bench, None