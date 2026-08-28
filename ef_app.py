import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import scipy.optimize as sco
import plotly.graph_objects as go
from datetime import datetime

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Efficient Frontier Portfolio Optimizer",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Efficient Frontier & Portfolio Optimizer")
st.markdown("""
This app computes Markowitz's **Efficient Frontier** using historical market data. 
It identifies key optimal portfolios, individual asset risk-return profiles, and calculates an optimal asset mix based on your personal risk tolerance.
*Note: Returns are calculated as Total Return, inclusive of dividends and splits.*

**Ratio guide:**
- **Sharpe Ratio:** excess return earned per unit of total volatility. Higher values indicate better risk-adjusted performance.
- **Sortino Ratio:** excess return earned per unit of downside volatility, focusing only on returns below the risk-free target. Higher values are better.
- **Treynor Ratio:** excess return earned per unit of market risk, measured by beta against SPY when available. Higher values indicate better performance relative to market exposure.
- **Calmar Ratio:** annualized return divided by the portfolio's maximum historical drawdown. Higher values indicate stronger returns with smaller peak-to-trough losses.
""")

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.header("Portfolio Parameters")

default_tickers = "SPY, BTC-USD, GC=F, QQQ, DIA"
asset_presets = {
    "US Large-Cap Stocks": "SPY",
    "Global Stocks": "VT",
    "Developed-Market Stocks": "VEA",
    "Emerging-Market Stocks": "VWO",
    "US Small-Cap Stocks": "IWM",
    "Technology Stocks": "QQQ",
    "US Bonds": "BND",
    "Long-Term Treasuries": "TLT",
    "Inflation-Protected Bonds": "TIP",
    "US REITs": "VNQ",
    "Gold": "GLD",
    "Commodities": "DBC",
    "Bitcoin": "BTC-USD",
    "Treasury Bills": "SGOV"
}

if "selected_assets" not in st.session_state:
    st.session_state.selected_assets = default_tickers.split(", ")

st.sidebar.caption("Toggle asset classes to build a starting portfolio")
preset_columns = st.sidebar.columns(2)
for index, (asset_name, ticker) in enumerate(asset_presets.items()):
    is_selected = ticker in st.session_state.selected_assets
    button_label = f"{'✓ ' if is_selected else ''}{asset_name}"
    if preset_columns[index % 2].button(button_label, key=f"asset_preset_{ticker}", use_container_width=True):
        if is_selected:
            st.session_state.selected_assets.remove(ticker)
        else:
            st.session_state.selected_assets.append(ticker)
        st.session_state.ticker_input = ", ".join(st.session_state.selected_assets)

if "ticker_input" not in st.session_state:
    st.session_state.ticker_input = ", ".join(st.session_state.selected_assets)

ticker_input = st.sidebar.text_input("Enter Tickers (comma separated):", key="ticker_input")
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

date_option = st.sidebar.radio("Date Range Option:", ("Max Available", "Custom Date Range"))

if date_option == "Custom Date Range":
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", value=datetime.today())
else:
    start_date = None
    end_date = None

risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=2.0, step=0.1) / 100.0

max_vol_limit = st.sidebar.slider(
    "Target Max Volatility / Risk Cap (%)", 
    min_value=5.0, 
    max_value=60.0, 
    value=18.0, 
    step=0.5
) / 100.0

# ---------------------------------------------------------
# Data Fetching & Processing
# ---------------------------------------------------------
@st.cache_data
def load_data(symbols, start, end, is_max):
    # auto_adjust=True bakes dividends and splits into the 'Close' price 
    # to accurately reflect Total Return.
    if is_max:
        df = yf.download(symbols, period="max", auto_adjust=True)
    else:
        df = yf.download(symbols, start=start, end=end, auto_adjust=True)
    
    # yfinance returns MultiIndex columns for multiple tickers. 
    # We just want the 'Close' prices (which are now adjusted for dividends).
    if isinstance(df.columns, pd.MultiIndex):
        return df["Close"]
    else:
        # Fallback if only one ticker manages to download
        return df[["Close"]]

if len(tickers) < 2:
    st.warning("Please enter at least 2 valid tickers to compute portfolio optimizations.")
    st.stop()

with st.spinner("Fetching market data..."):
    data = load_data(tickers, start_date, end_date, date_option == "Max Available")

# Clean data
data = data.dropna(axis=1, how="all").dropna()

if data.empty or data.shape[1] < 2:
    st.error("Insufficient data downloaded. Please verify the ticker symbols and selected date range.")
    st.stop()

valid_tickers = list(data.columns)

# Calculate Annualized Total Returns and Covariance
returns = data.pct_change().dropna()
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252
num_assets = len(valid_tickers)
market_ticker = "SPY" if "SPY" in valid_tickers else valid_tickers[0]
market_returns = returns[market_ticker]

# ---------------------------------------------------------
# Optimization Helper Functions
# ---------------------------------------------------------
def portfolio_performance(weights, mean_returns, cov_matrix):
    perf_return = np.sum(mean_returns * weights)
    perf_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return perf_return, perf_volatility

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_ret, p_vol = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_vol

def portfolio_sortino_ratio(weights, returns, mean_returns, risk_free_rate):
    portfolio_returns = returns @ weights
    daily_target = (1 + risk_free_rate) ** (1 / 252) - 1
    downside_returns = np.minimum(portfolio_returns - daily_target, 0)
    downside_deviation = np.sqrt(np.mean(downside_returns ** 2) * 252)
    if downside_deviation == 0:
        return 0.0
    portfolio_return = np.sum(mean_returns * weights)
    return (portfolio_return - risk_free_rate) / downside_deviation

def neg_sortino_ratio(weights, returns, mean_returns, risk_free_rate):
    return -portfolio_sortino_ratio(weights, returns, mean_returns, risk_free_rate)

def portfolio_treynor_ratio(weights, returns, market_returns, mean_returns, risk_free_rate):
    portfolio_returns = returns @ weights
    beta = np.cov(portfolio_returns, market_returns)[0, 1] / np.var(market_returns)
    if beta <= 0:
        return 0.0
    portfolio_return = np.sum(mean_returns * weights)
    return (portfolio_return - risk_free_rate) / beta

def neg_treynor_ratio(weights, returns, market_returns, mean_returns, risk_free_rate):
    return -portfolio_treynor_ratio(weights, returns, market_returns, mean_returns, risk_free_rate)

def portfolio_calmar_ratio(weights, returns, mean_returns):
    portfolio_returns = returns @ weights
    cumulative_returns = (1 + portfolio_returns).cumprod()
    drawdowns = cumulative_returns / cumulative_returns.cummax() - 1
    max_drawdown = abs(drawdowns.min())
    if max_drawdown == 0:
        return 0.0
    portfolio_return = np.sum(mean_returns * weights)
    return portfolio_return / max_drawdown

def neg_calmar_ratio(weights, returns, mean_returns):
    return -portfolio_calmar_ratio(weights, returns, mean_returns)

def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]

def portfolio_return(weights, mean_returns, cov_matrix):
    return portfolio_performance(weights, mean_returns, cov_matrix)[0]

bounds = tuple((0, 1) for _ in range(num_assets))
init_guess = num_assets * [1.0 / num_assets,]
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# 1. Max Sharpe Ratio
opt_sharpe = sco.minimize(
    neg_sharpe_ratio, init_guess, 
    args=(mean_returns, cov_matrix, risk_free_rate),
    method='SLSQP', bounds=bounds, constraints=constraints
)
weights_sharpe = opt_sharpe.x
ret_sharpe, vol_sharpe = portfolio_performance(weights_sharpe, mean_returns, cov_matrix)
sharpe_max = (ret_sharpe - risk_free_rate) / vol_sharpe

# 2. Max Sortino Ratio
opt_sortino = sco.minimize(
    neg_sortino_ratio, init_guess,
    args=(returns, mean_returns, risk_free_rate),
    method='SLSQP', bounds=bounds, constraints=constraints
)
weights_sortino = opt_sortino.x
ret_sortino, vol_sortino = portfolio_performance(weights_sortino, mean_returns, cov_matrix)
sortino_max = portfolio_sortino_ratio(weights_sortino, returns, mean_returns, risk_free_rate)

# 3. Max Treynor Ratio
opt_treynor = sco.minimize(
    neg_treynor_ratio, init_guess,
    args=(returns, market_returns, mean_returns, risk_free_rate),
    method='SLSQP', bounds=bounds, constraints=constraints
)
weights_treynor = opt_treynor.x
ret_treynor, vol_treynor = portfolio_performance(weights_treynor, mean_returns, cov_matrix)
treynor_max = portfolio_treynor_ratio(weights_treynor, returns, market_returns, mean_returns, risk_free_rate)

# 4. Max Calmar Ratio
opt_calmar = sco.minimize(
    neg_calmar_ratio, init_guess,
    args=(returns, mean_returns),
    method='SLSQP', bounds=bounds, constraints=constraints
)
weights_calmar = opt_calmar.x
ret_calmar, vol_calmar = portfolio_performance(weights_calmar, mean_returns, cov_matrix)
calmar_max = portfolio_calmar_ratio(weights_calmar, returns, mean_returns)

# 5. Minimum Volatility
opt_min_vol = sco.minimize(
    portfolio_volatility, init_guess, 
    args=(mean_returns, cov_matrix),
    method='SLSQP', bounds=bounds, constraints=constraints
)
weights_min_vol = opt_min_vol.x
ret_min_vol, vol_min_vol = portfolio_performance(weights_min_vol, mean_returns, cov_matrix)
sharpe_min_vol = (ret_min_vol - risk_free_rate) / vol_min_vol

# 6. Target Max Volatility Constraint
if max_vol_limit < vol_min_vol:
    st.sidebar.warning(f"Target volatility is below minimum achievable ({vol_min_vol:.1%}). Defaulting to Min Volatility.")
    weights_target = weights_min_vol
    ret_target, vol_target = ret_min_vol, vol_min_vol
else:
    vol_constraint = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                      {'type': 'ineq', 'fun': lambda x: max_vol_limit - portfolio_volatility(x, mean_returns, cov_matrix)})
    
    opt_target = sco.minimize(
        lambda x: -portfolio_return(x, mean_returns, cov_matrix), init_guess,
        method='SLSQP', bounds=bounds, constraints=vol_constraint
    )
    weights_target = opt_target.x
    ret_target, vol_target = portfolio_performance(weights_target, mean_returns, cov_matrix)

sharpe_target = (ret_target - risk_free_rate) / vol_target

# ---------------------------------------------------------
# Efficient Frontier Curve
# ---------------------------------------------------------
target_returns = np.linspace(mean_returns.min(), mean_returns.max() * 1.1, 50)
efficient_volatilities = []

for t_ret in target_returns:
    c = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
         {'type': 'eq', 'fun': lambda x: portfolio_return(x, mean_returns, cov_matrix) - t_ret})
    res = sco.minimize(portfolio_volatility, init_guess, args=(mean_returns, cov_matrix), method='SLSQP', bounds=bounds, constraints=c)
    if res.success:
        efficient_volatilities.append(res.fun)
    else:
        efficient_volatilities.append(np.nan)

# ---------------------------------------------------------
# Monte Carlo Sim
# ---------------------------------------------------------
num_simulations = 2500
sim_weights = np.random.dirichlet(np.ones(num_assets), size=num_simulations)
sim_returns = np.dot(sim_weights, mean_returns)
sim_vols = np.sqrt(np.einsum('ij,jk,ik->i', sim_weights, cov_matrix, sim_weights))
sim_sharpe = (sim_returns - risk_free_rate) / sim_vols

ind_vols = np.sqrt(np.diag(cov_matrix))
ind_rets = mean_returns.values

# ---------------------------------------------------------
# UI Layout & Plot
# ---------------------------------------------------------
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Max Sharpe Return", f"{ret_sharpe:.2%}", f"Sharpe: {sharpe_max:.2f}")
col2.metric("Max Sortino Return", f"{ret_sortino:.2%}", f"Sortino: {sortino_max:.2f}")
col3.metric("Max Treynor Return", f"{ret_treynor:.2%}", f"Treynor: {treynor_max:.2f}")
col4.metric("Max Calmar Return", f"{ret_calmar:.2%}", f"Calmar: {calmar_max:.2f}")
col5.metric("Min Volatility Return", f"{ret_min_vol:.2%}", f"Vol: {vol_min_vol:.2%}")
col6.metric("Sweet Spot Return", f"{ret_target:.2%}", f"Vol Cap: {max_vol_limit:.1%}")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=sim_vols, y=sim_returns,
    mode='markers',
    marker=dict(size=5, color=sim_sharpe, colorscale='Viridis', showscale=True, colorbar=dict(title="Sharpe Ratio")),
    name='Simulated Portfolios',
    hoverinfo='text',
    text=[f"Return: {r:.2%}<br>Vol: {v:.2%}<br>Sharpe: {s:.2f}" for r, v, s in zip(sim_returns, sim_vols, sim_sharpe)]
))

fig.add_trace(go.Scatter(
    x=efficient_volatilities, y=target_returns,
    mode='lines',
    line=dict(color='deepskyblue', width=3, dash='dash'),
    name='Efficient Frontier'
))

fig.add_trace(go.Scatter(
    x=ind_vols, y=ind_rets,
    mode='markers+text',
    text=valid_tickers,
    textposition="middle right",
    marker=dict(size=12, color='orange', symbol='diamond'),
    name='Individual Assets'
))

fig.add_trace(go.Scatter(
    x=[vol_sharpe], y=[ret_sharpe],
    mode='markers',
    marker=dict(size=16, color='red', symbol='star'),
    name='Max Sharpe Ratio'
))

fig.add_trace(go.Scatter(
    x=[vol_sortino], y=[ret_sortino],
    mode='markers',
    marker=dict(size=16, color='cyan', symbol='star-diamond'),
    name='Max Sortino Ratio'
))

fig.add_trace(go.Scatter(
    x=[vol_treynor], y=[ret_treynor],
    mode='markers',
    marker=dict(size=16, color='yellow', symbol='hexagram'),
    name='Max Treynor Ratio'
))

fig.add_trace(go.Scatter(
    x=[vol_calmar], y=[ret_calmar],
    mode='markers',
    marker=dict(size=16, color='lightskyblue', symbol='hexagon2'),
    name='Max Calmar Ratio'
))

fig.add_trace(go.Scatter(
    x=[vol_min_vol], y=[ret_min_vol],
    mode='markers',
    marker=dict(size=14, color='green', symbol='square'),
    name='Min Volatility'
))

fig.add_trace(go.Scatter(
    x=[vol_target], y=[ret_target],
    mode='markers',
    marker=dict(size=16, color='magenta', symbol='cross'),
    name=f'Target Vol Spot ({max_vol_limit:.1%})'
))

fig.update_layout(
    title="Efficient Frontier (Total Return)",
    xaxis_title="Annualized Volatility (Risk)",
    yaxis_title="Annualized Expected Return",
    template="plotly_dark",
    height=600,
    legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("📊 Optimal Portfolio Allocations")
allocation_df = pd.DataFrame({
    'Ticker': valid_tickers,
    'Max Sharpe (%)': (weights_sharpe * 100).round(2),
    'Max Sortino (%)': (weights_sortino * 100).round(2),
    'Max Treynor (%)': (weights_treynor * 100).round(2),
    'Max Calmar (%)': (weights_calmar * 100).round(2),
    'Min Volatility (%)': (weights_min_vol * 100).round(2),
    f'Target Volatility ({max_vol_limit:.1%}) (%)': (weights_target * 100).round(2)
})

st.dataframe(allocation_df.set_index('Ticker'), use_container_width=True)

st.subheader("Asset Return Correlation")
correlation_fig = go.Figure(data=go.Heatmap(
    z=returns.corr().values,
    x=valid_tickers,
    y=valid_tickers,
    zmin=-1,
    zmax=1,
    colorscale="RdBu",
    text=returns.corr().round(2).values,
    texttemplate="%{text}",
    textfont=dict(size=12),
    colorbar=dict(title="Correlation")
))

correlation_fig.update_layout(
    title="Correlation of Daily Total Returns",
    xaxis_title="Asset",
    yaxis_title="Asset",
    template="plotly_dark",
    height=600
)

st.plotly_chart(correlation_fig, use_container_width=True)
