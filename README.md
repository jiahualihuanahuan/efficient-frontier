# 📈 Efficient Frontier & Portfolio Optimizer

This Streamlit web application computes Markowitz's **Efficient Frontier** using historical market data. It allows users to build a custom portfolio, analyzes individual asset risk-return profiles, and calculates optimal asset allocations based on various performance metrics and personal risk tolerance.

## ✨ Features

*   **Interactive Asset Selection:** Choose from a curated list of asset presets (S&P 500, Bitcoin, Gold, Bonds, etc.) or input your own custom ticker symbols.
*   **Live Data Fetching:** Automatically downloads historical Total Return data (inclusive of dividends and splits) via Yahoo Finance (`yfinance`).
*   **Multiple Optimization Strategies:** Calculates optimal weights for:
    *   Maximum Sharpe Ratio
    *   Maximum Sortino Ratio
    *   Maximum Treynor Ratio
    *   Minimum Calmar Ratio (optimized for drawdown)
    *   Minimum Volatility
    *   Target Volatility (custom risk cap)
*   **Monte Carlo Simulation:** Generates 2,500 simulated portfolios to visualize the risk/return landscape.
*   **Interactive Visualizations:** Features interactive Plotly charts for the Efficient Frontier and an asset correlation heatmap.
*   **Customizable Parameters:** Adjust the risk-free rate, target maximum volatility, and historical date range.

## 🛠 Prerequisites

Make sure you have Python 3.7+ installed. You will need to install the following dependencies:

```bash
pip install streamlit yfinance numpy pandas scipy plotly
