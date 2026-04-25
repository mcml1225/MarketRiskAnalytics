# 📊 Market Risk Analytics

[![CI](https://github.com/mcml1225/MarketRiskAnalytics/actions/workflows/ci.yml/badge.svg)](https://github.com/mcml1225/MarketRiskAnalytics/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A professional Python toolkit for market risk analysis featuring **Value at Risk (VaR)**, **Expected Shortfall (CVaR)**, **Beta**, **Maximum Drawdown**, and interactive **Streamlit** dashboard.

## ✨ Features

- 📈 **Real-time data** from Yahoo Finance
- 📊 **Comprehensive risk metrics**:
  - Value at Risk (VaR) - Historical simulation
  - Conditional VaR (CVaR) / Expected Shortfall
  - Beta (Systematic risk)
  - Maximum Drawdown (MDD)
  - Annualized Volatility
  - Sharpe Ratio
- 🎨 **Interactive Streamlit dashboard**:
  - Dynamic ticker selection
  - Adjustable date ranges and confidence levels
  - Price charts and cumulative returns
  - Rolling volatility heatmaps
  - Correlation matrices
  - Return distribution analysis (histograms + Q-Q plots)
- 💾 **Data export** capabilities (CSV format)
- 🧪 **Comprehensive unit tests**

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation


#### Clone the repository
git clone https://github.com/mcml1225/MarketRiskAnalytics.git
cd MarketRiskAnalytics

#### Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

#### Install dependencies
pip install -r requirements.txt

Running the Dashboard

streamlit run app.py
Then open your browser to http://localhost:8501

### 📖 Usage Examples
Basic Analysis
python
from src.data_fetcher import get_all_data
from src.risk_metrics import summarize_risk_metrics

#### Fetch data for Apple and Microsoft
tickers = ["AAPL", "MSFT"]
prices, benchmark, _ = get_all_data(tickers)

#### Calculate risk metrics
metrics = summarize_risk_metrics(prices, benchmark)
print(metrics)
Command Line Testing

#### Run all tests
pytest tests/ -v

#### Run with coverage report
pytest tests/ --cov=src --cov-report=html
📊 Dashboard Preview
The dashboard provides:

Sidebar controls for tickers, date ranges, and risk parameters

Risk metrics table with all key indicators

Interactive price charts with hover details

Rolling volatility heatmap to identify risk periods

Correlation matrix for portfolio diversification analysis

Distribution analysis with histograms and Q-Q plots

One-click data export to CSV

📁 Project Structure

MarketRiskAnalytics/
├── .github/workflows/    # CI/CD configuration
├── src/
│   ├── __init__.py
│   ├── config.py         # Configuration parameters
│   ├── data_fetcher.py   # Yahoo Finance integration
│   ├── risk_metrics.py   # Core calculations
│   └── utils.py          # Helper functions
├── tests/                # Unit tests
├── data/                 # Data cache (gitignored)
├── notebooks/            # Jupyter notebooks
├── app.py               # Streamlit dashboard
├── requirements.txt     # Dependencies
└── README.md           # Documentation
🔍 Understanding the Metrics
Metric	Formula	Interpretation
VaR	Percentile(returns, 1-α)	Maximum loss at confidence α
CVaR	E[returns | returns ≤ VaR]	Average loss beyond VaR
Beta	Cov(asset, market) / Var(market)	Market sensitivity
Max Drawdown	min((P_t - P_peak)/P_peak)	Worst peak-to-trough loss
Sharpe Ratio	(R_p - R_f) / σ_p	Risk-adjusted return
🧪 Running Tests

#### Install testing dependencies
pip install pytest pytest-cov

#### Run all tests
pytest tests/ -v

#### Generate coverage report
pytest tests/ --cov=src --cov-report=html
🤝 Contributing
Fork the repository

Create a feature branch (git checkout -b feature/amazing-feature)

Commit changes (git commit -m 'Add amazing feature')

Push to branch (git push origin feature/amazing-feature)

Open a Pull Request

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Yahoo Finance for free market data

Streamlit for the amazing dashboard framework

Plotly for interactive visualizations
