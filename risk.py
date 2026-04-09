import streamlit as st
from streamlit_gsheets import GSheetsConnection
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import statsmodels.api as sm
import datetime

# ==========================================
# 1. Dashboard Configuration & UI Setup
# ==========================================
st.set_page_config(page_title="Blue Eagle Capital Risk & Factor Dashboard", layout="wide")
st.title("Blue Eagle Capital: Portfolio Risk & Attribution Dashboard")
st.markdown("### Quarterly Risk Diagnostics, Attribution, and Stress Testing")

# ==========================================
# 2. Portfolio Data Ingestion
# ==========================================
st.sidebar.header("Portfolio Parameters")

conn = st.connection("gsheets", type=GSheetsConnection)
sheet_url = "https://docs.google.com/spreadsheets/d/11Fpu0mz2JevRHzqrvgR-TrC78qbALWAS1Yp8C2gEi4U/edit#gid=418083832"

try:
    with st.spinner("Fetching active portfolio data..."):
        portfolio_data = conn.read(spreadsheet=sheet_url, ttl="10m")
        portfolio_data.columns = portfolio_data.columns.str.strip()
        
        df = portfolio_data[['Date', 'Ticker', 'Value', '% Return']].copy()
        df = df.dropna(subset=['Ticker'])
        df = df[~df['Ticker'].astype(str).str.contains('Total', case=False, na=False)]
        
        df['Value'] = df['Value'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.replace('(', '-', regex=False).str.replace(')', '', regex=False)
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        
        df['% Return'] = df['% Return'].astype(str).str.replace('%', '', regex=False).str.replace('(', '-', regex=False).str.replace(')', '', regex=False)
        df['% Return'] = pd.to_numeric(df['% Return'], errors='coerce') / 100
        
        df = df[df['Value'] > 0]
        
        cash_df = df[df['Ticker'].astype(str).str.contains('Cash', case=False, na=False)]
        cash_balance = cash_df['Value'].sum() if not cash_df.empty else 0.0
        
        equity_df = df[~df['Ticker'].astype(str).str.contains('Cash', case=False, na=False)].copy()
        equity_value = equity_df['Value'].sum()
        total_portfolio_value = equity_value + cash_balance
        
        equity_df['Active_Weight'] = equity_df['Value'] / equity_value
        
        tickers = equity_df['Ticker'].astype(str).str.strip().tolist()
        weights = np.array(equity_df['Active_Weight'].tolist())
        entry_dates = equity_df['Date'].astype(str).tolist()
        actual_returns = np.array(equity_df['% Return'].fillna(0.0).tolist())
        
        st.sidebar.success("Active portfolio data loaded.")
        st.sidebar.write(f"**Gross Exposure:** ${equity_value:,.2f}")
        
except Exception as e:
    st.sidebar.error("Failed to connect to data source.")
    st.stop()

confidence_level = st.sidebar.slider("Confidence Level", 0.90, 0.99, 0.95)
lookback_years = st.sidebar.slider("Regression Lookback (Years)", 1, 15, 5)

end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=365 * lookback_years)

# ==========================================
# 3. Market Data & Factor Proxy Construction
# ==========================================
# Exactly 9 Risk Factors defined for the forward-looking model
mag7_components = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA']
standard_proxies = {
    'Market (Beta)': 'SPY',
    'Size (Small Cap)': 'IWM',
    'Value': 'IWD',
    'Growth': 'IWF',
    'Momentum': 'MTUM',
    'Quality': 'QUAL',
    'Oil Prices': 'USO',
    'Interest Rates (7-10y)': 'IEF'
}

@st.cache_data
def load_deep_history(port_tickers, factor_dict, mag7_tickers):
    all_symbols = list(set(port_tickers + list(factor_dict.values()) + mag7_tickers))
    data = yf.download(all_symbols, start="2007-01-01", end=datetime.date.today())['Close']
    data = data.ffill()
    data.index = data.index.tz_localize(None)
    
    returns_df = np.log(data / data.shift(1))
    returns_df['Mag_7_Proxy'] = returns_df[mag7_tickers].mean(axis=1)
    
    return returns_df

with st.spinner("Compiling historical market and factor data..."):
    full_returns = load_deep_history(tickers, standard_proxies, mag7_components)

regression_returns = full_returns.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)].copy()
regression_returns = regression_returns.dropna() 

factor_proxies = standard_proxies.copy()
factor_proxies['Mag 7 Exposure'] = 'Mag_7_Proxy'

port_component_returns = regression_returns[tickers]
portfolio_returns = port_component_returns.dot(weights)

st.divider()

# ==========================================
# 4. Return Attribution 
# ==========================================
st.header("1. Return Attribution")

# Simplified to just show Weights and Return Contribution (Removed Euler MCR/CCV math)
contrib_df = pd.DataFrame({
    'Portfolio Weight': weights,
    'Total Return Contribution': actual_returns * weights
}, index=tickers).sort_values(by='Total Return Contribution', ascending=False)

col1, col2 = st.columns(2)
with col1:
    st.dataframe(contrib_df.style.format({
        'Portfolio Weight': '{:.2%}',
        'Total Return Contribution': '{:.2%}'
    }), use_container_width=True)

with col2:
    st.bar_chart(contrib_df['Total Return Contribution'])

st.divider()

# ==========================================
# 5. Factor Risk Decomposition
# ==========================================
st.header("2. Factor Risk Decomposition")

factor_returns = regression_returns[list(factor_proxies.values())]
factor_returns.columns = list(factor_proxies.keys())

# Using HC3 Robust Standard Errors to fix Heteroskedasticity
X = sm.add_constant(factor_returns)
model = sm.OLS(portfolio_returns, X).fit(cov_type='HC3')

factor_results = pd.DataFrame({
    'Factor Sensitivities (Beta)': model.params[1:],
    'P-Value': model.pvalues[1:]
})

significant_factors = factor_results[factor_results['P-Value'] < 0.05]

col3, col4 = st.columns([2, 1])
with col3:
    st.bar_chart(factor_results['Factor Sensitivities (Beta)'])
with col4:
    st.write("**Statistically Significant Exposures (p < 0.05):**")
    st.dataframe(significant_factors['Factor Sensitivities (Beta)'].apply(lambda x: f"{x:.2f}"))
    st.write(f"**Adjusted R-Squared:** {model.rsquared_adj:.2f}")

st.divider()

# ==========================================
# 6. Portfolio Risk Analytics
# ==========================================
st.header("3. Portfolio Risk Analytics")

port_volatility = np.std(portfolio_returns) * np.sqrt(252)
alpha = 1 - confidence_level
historical_var = np.percentile(portfolio_returns, alpha * 100)
cvar = portfolio_returns[portfolio_returns <= historical_var].mean()

m1, m2, m3 = st.columns(3)
m1.metric("Annualized Volatility", f"{port_volatility*100:.2f}%")
m2.metric(f"Empirical VaR ({confidence_level*100:.0f}%)", f"{historical_var*100:.2f}%")
m3.metric("Expected Shortfall (CVaR)", f"{cvar*100:.2f}%")

st.divider()

# ==========================================
# 7. Historical Scenario Analysis
# ==========================================
st.header("4. Historical Scenario Analysis")
st.markdown("Constant-weight simulation mapping current portfolio composition against severe historical market drawdowns.")

stress_periods = {
    "Global Financial Crisis": ("2007-10-09", "2009-03-09"),
    "COVID-19 Crash": ("2020-02-19", "2020-03-23"),
    "2022 Rate Selloff": ("2022-01-03", "2022-10-12")
}

deep_port_returns = full_returns[tickers].fillna(0).dot(weights)

stress_results = []
for event, (start, end) in stress_periods.items():
    try:
        event_returns = deep_port_returns.loc[start:end]
        event_spy_returns = full_returns['SPY'].loc[start:end]
        
        port_drawdown = np.exp(event_returns.sum()) - 1
        spy_drawdown = np.exp(event_spy_returns.sum()) - 1
        
        stress_results.append({
            "Scenario": event,
            "Date Range": f"{start} to {end}",
            "Portfolio Max Drawdown": port_drawdown,
            "Benchmark Max Drawdown": spy_drawdown,
            "Active Drawdown Difference": port_drawdown - spy_drawdown
        })
    except Exception as e:
        continue

stress_df = pd.DataFrame(stress_results)
st.dataframe(stress_df.style.format({
    "Portfolio Max Drawdown": "{:.2%}",
    "Benchmark Max Drawdown": "{:.2%}",
    "Active Drawdown Difference": "{:.2%}"
}), use_container_width=True)

st.divider()

# ==========================================
# 8. Forward-Looking Investment Pitch
# ==========================================
st.header("5. Forward-Looking Investment Pitch")
st.markdown("### Quarterly Factor Outlook & Allocation Recommendations")
st.write("Based on our 9-factor regression, we have isolated the top 4 most meaningful drivers of our active portfolio risk. By assessing our current exposure (Beta) against the recent 63-day macroeconomic momentum of these factors, we propose the following forward-looking adjustments:")

# 1. Identify the top 4 most meaningful risk factors by absolute beta loading
top_4_factors = factor_results.sort_values(by='Factor Sensitivities (Beta)', key=abs, ascending=False).head(4)

# 2. Calculate the fundamental quarterly momentum (last 63 trading days) for these 4 factors
recent_quarter_returns = factor_returns[top_4_factors.index].tail(63).sum()

