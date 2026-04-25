"""
Configuration settings for the Market Risk Analytics project.

This module contains all configurable parameters including default tickers,
date ranges, risk parameters, and calculation settings.
"""

# Default stock tickers for demonstration (US large caps)
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

# Default benchmark index (S&P 500)
BENCHMARK_TICKER = "^GSPC"

# Default date range (last 3 years from today)
import datetime as dt
END_DATE = dt.date.today()
START_DATE = END_DATE - dt.timedelta(days=3*365)

# Value at Risk (VaR) confidence level
# 0.95 means 95% confidence, 5% significance level
VAR_CONFIDENCE = 0.95

# Risk-free rate for Sharpe ratio calculation (annualized)
# Typically based on 10-year Treasury yield
RISK_FREE_RATE = 0.02  # 2% annualized

# Trading days per year for annualization calculations
TRADING_DAYS_PER_YEAR = 252

# Rolling window for volatility calculation (trading days)
ROLLING_VOLATILITY_WINDOW = 21  # Approximately 1 month