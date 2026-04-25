"""
Market Risk Analytics Dashboard - CORRECTED VERSION
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.data_fetcher import get_all_data
from src.risk_metrics import calculate_returns, summarize_risk_metrics
from src.utils import rolling_volatility
from src.config import (
    DEFAULT_TICKERS, 
    BENCHMARK_TICKER, 
    VAR_CONFIDENCE, 
    START_DATE, 
    END_DATE,
    RISK_FREE_RATE
)

# Page config
st.set_page_config(page_title="Market Risk Analytics", layout="wide")

# Title
st.title("📊 Market Risk Analytics Dashboard")
st.markdown("Analyze **Value at Risk (VaR)**, **CVaR**, **Beta**, and **Maximum Drawdown** for any stock portfolio.")

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.header("📈 Configuration")

tickers_input = st.sidebar.text_input(
    "Stock Tickers",
    value=",".join(DEFAULT_TICKERS),
    help="Enter tickers separated by commas (e.g., AAPL, MSFT, GOOGL)"
)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

benchmark = st.sidebar.text_input("Benchmark", value=BENCHMARK_TICKER)

confidence = st.sidebar.slider(
    "VaR Confidence Level",
    0.90, 0.99, VAR_CONFIDENCE, 0.01,
    help="Higher confidence = more conservative risk estimate"
)

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", START_DATE)
with col2:
    end_date = st.date_input("End Date", END_DATE)

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.success("Cache cleared!")

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data(ttl=3600)
def load_data(tickers, benchmark, start, end):
    import src.config as config
    config.START_DATE = start
    config.END_DATE = end
    return get_all_data(tickers, benchmark)

if not tickers:
    st.warning("⚠️ Please enter at least one ticker")
    st.stop()

with st.spinner("Loading market data..."):
    try:
        prices, benchmark_prices, _ = load_data(tickers, benchmark, start_date, end_date)
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

if prices.empty:
    st.error("No data found. Check ticker symbols.")
    st.stop()

# Calculate returns
returns = calculate_returns(prices)
benchmark_returns = calculate_returns(benchmark_prices)

# ============================================================================
# RISK METRICS TABLE
# ============================================================================

st.subheader("📋 Risk Metrics Summary")

metrics_df = summarize_risk_metrics(prices, benchmark_prices, confidence, RISK_FREE_RATE)
st.dataframe(metrics_df, use_container_width=True, hide_index=True)

# ============================================================================
# PRICE CHART - CORRECTED
# ============================================================================

st.subheader("📈 Historical Prices")

# Validate data before plotting
if not prices.empty and len(prices) > 0:
    # Clean data - remove any absurd values
    prices_clean = prices.copy()
    for col in prices_clean.columns:
        # Remove outliers (values > 10,000 might be errors)
        prices_clean[col] = prices_clean[col].where(prices_clean[col] < 10000)
    
    fig_price = go.Figure()
    
    for ticker in tickers:
        if ticker in prices_clean.columns:
            fig_price.add_trace(go.Scatter(
                x=prices_clean.index,
                y=prices_clean[ticker],
                mode='lines',
                name=ticker,
                line=dict(width=2)
            ))
    
    fig_price.update_layout(
        title="Adjusted Close Prices",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode='x unified',
        height=500
    )
    st.plotly_chart(fig_price, use_container_width=True)
else:
    st.warning("No price data available")

# ============================================================================
# ROLLING VOLATILITY - IMPROVED (LINE CHART INSTEAD OF HEATMAP)
# ============================================================================

st.subheader("🌊 Rolling 21-Day Volatility (Annualized)")

# Calculate rolling volatility
roll_vol = rolling_volatility(returns, window=21)

if not roll_vol.empty:
    # Use line chart instead of heatmap for better readability
    fig_vol = go.Figure()
    
    for ticker in tickers:
        if ticker in roll_vol.columns:
            fig_vol.add_trace(go.Scatter(
                x=roll_vol.index,
                y=roll_vol[ticker],
                mode='lines',
                name=ticker,
                line=dict(width=2)
            ))
    
    fig_vol.update_layout(
        title="Rolling Annualized Volatility (21-day window)",
        xaxis_title="Date",
        yaxis_title="Volatility (Annualized)",
        hovermode='x unified',
        height=450
    )
    st.plotly_chart(fig_vol, use_container_width=True)
    
    st.caption("""
    🔍 **Interpretation**: Higher values indicate increased market turbulence. 
    Spikes in volatility often correspond to market stress periods.
    """)
else:
    st.info("Insufficient data for rolling volatility calculation")

# ============================================================================
# CORRELATION MATRIX - IMPROVED (TREEMAP INSTEAD OF LARGE MATRIX)
# ============================================================================

st.subheader("🔗 Asset Correlations")

if len(tickers) > 1 and not returns.empty:
    # Calculate correlation matrix
    corr_matrix = returns.corr()
    
    # Option 1: Compact heatmap with better sizing
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect='auto',
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        title="Return Correlations"
    )
    fig_corr.update_layout(
        height=400,
        width=600,
        autosize=True
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Option 2: Bar chart showing average correlation (alternative view)
    st.subheader("Average Correlation with Portfolio")
    
    # Calculate average correlation for each asset
    avg_corr = corr_matrix.mean().sort_values(ascending=False)
    
    fig_avg_corr = px.bar(
        x=avg_corr.index,
        y=avg_corr.values,
        title="Average Correlation with Other Assets",
        labels={"x": "Asset", "y": "Average Correlation"},
        color=avg_corr.values,
        color_continuous_scale='RdBu_r',
        range_color=[-1, 1]
    )
    fig_avg_corr.update_layout(height=400)
    st.plotly_chart(fig_avg_corr, use_container_width=True)
    
    st.caption("""
    🔍 **Interpretation**: 
    - High positive correlation (>0.7): Assets move together (low diversification benefit)
    - Low or negative correlation: Better diversification potential
    """)

# ============================================================================
# CUMULATIVE RETURNS
# ============================================================================

st.subheader("📊 Cumulative Returns")

if not returns.empty:
    cum_returns = (1 + returns).cumprod()
    
    fig_cum = go.Figure()
    for ticker in tickers:
        if ticker in cum_returns.columns:
            fig_cum.add_trace(go.Scatter(
                x=cum_returns.index,
                y=cum_returns[ticker],
                mode='lines',
                name=ticker,
                line=dict(width=2)
            ))
    
    fig_cum.update_layout(
        title="Cumulative Returns (Indexed to 1.0)",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        hovermode='x unified',
        height=450
    )
    st.plotly_chart(fig_cum, use_container_width=True)

# ============================================================================
# RETURN DISTRIBUTION
# ============================================================================

st.subheader("📉 Return Distribution Analysis")

if len(tickers) > 0 and not returns.empty:
    selected = st.selectbox("Select asset", tickers)
    
    col_dist, col_stats = st.columns(2)
    
    with col_dist:
        fig_hist = px.histogram(
            returns[selected],
            nbins=50,
            title=f"{selected} - Daily Returns Distribution",
            labels={"value": "Daily Return", "count": "Frequency"}
        )
        fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
        fig_hist.add_vline(x=returns[selected].mean(), line_dash="dot", line_color="green")
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col_stats:
        # Summary statistics
        rets = returns[selected].dropna()
        stats_data = {
            "Mean Daily Return": f"{rets.mean():.4%}",
            "Std Daily Return": f"{rets.std():.4%}",
            "Skewness": f"{rets.skew():.3f}",
            "Kurtosis": f"{rets.kurtosis():.3f}",
            "Min Return": f"{rets.min():.4%}",
            "Max Return": f"{rets.max():.4%}"
        }
        
        st.markdown("### Summary Statistics")
        for key, value in stats_data.items():
            st.metric(key, value)

# ============================================================================
# DOWNLOAD BUTTONS
# ============================================================================

st.subheader("💾 Export Data")

col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    st.download_button(
        "Download Prices (CSV)",
        prices.to_csv().encode('utf-8'),
        f"prices_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv"
    )

with col_exp2:
    st.download_button(
        "Download Returns (CSV)",
        returns.to_csv().encode('utf-8'),
        f"returns_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv"
    )

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
**Data Source**: Yahoo Finance | **Built with**: Streamlit, Plotly, Pandas

⚠️ *For educational purposes only. Not financial advice.*
""")