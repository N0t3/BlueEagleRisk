import os
import streamlit as st
from streamlit_gsheets import GSheetsConnection
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# ==========================================
# BRAND CONFIGURATION
# ==========================================
COLORS = {
    'NAVY'  : '#002855',
    'SLATE' : '#64748b',
    'RED'   : '#d97706',
    'GRID'  : '#e2e8f0',
    'BG'    : '#ffffff',
}

plt.rcParams['font.family']      = 'sans-serif'
plt.rcParams['font.sans-serif']  = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.facecolor']   = COLORS['BG']
plt.rcParams['figure.facecolor'] = COLORS['BG']
plt.rcParams['axes.edgecolor']   = COLORS['GRID']
plt.rcParams['axes.labelcolor']  = COLORS['NAVY']
plt.rcParams['xtick.color']      = COLORS['SLATE']
plt.rcParams['ytick.color']      = COLORS['SLATE']
plt.rcParams['text.color']       = COLORS['NAVY']
plt.rcParams['grid.color']       = COLORS['GRID']
plt.rcParams['grid.linewidth']   = 0.6

DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    'bec_diverging', [COLORS['RED'], '#f5f5f5', COLORS['NAVY']]
)

# Logo — header only, not on individual charts
LOGO_FILENAME  = "Blue Circle Icon.png"
possible_paths = [
    LOGO_FILENAME,
    os.path.join(os.path.expanduser("~"), "Downloads", LOGO_FILENAME),
    os.path.join("C:", "Users", "Public", "Downloads", LOGO_FILENAME),
]
LOGO_PATH = None
for p in possible_paths:
    if os.path.exists(p):
        LOGO_PATH = p
        break


def styled_fig(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(COLORS['BG'])
    ax.set_facecolor(COLORS['BG'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['GRID'])
    ax.spines['bottom'].set_color(COLORS['GRID'])
    return fig, ax


# ==========================================
# ORTHOGONALIZATION — Gram-Schmidt / Sequential OLS
# ==========================================
def orthogonalize_factors(raw_factor_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sequentially orthogonalize factors via OLS residuals (Gram-Schmidt).
    """
    cols      = raw_factor_df.columns.tolist()
    ortho_df  = pd.DataFrame(index=raw_factor_df.index)
    ortho_df[cols[0]] = raw_factor_df[cols[0]]   # Market anchor — unchanged

    for i in range(1, len(cols)):
        y       = raw_factor_df[cols[i]]
        X_prior = sm.add_constant(ortho_df.iloc[:, :i])
        resid   = sm.OLS(y, X_prior).fit().resid
        ortho_df[cols[i]] = resid                # pure residual factor

    return ortho_df


# ==========================================
# 1. Dashboard Header
# ==========================================
st.set_page_config(
    page_title="Blue Eagle Capital Risk & Factor Dashboard", layout="wide"
)

hdr_col, logo_col = st.columns([5, 1])
with hdr_col:
    st.title("Blue Eagle Capital: Portfolio Risk & Attribution Dashboard")
    st.markdown("Quarterly Risk Diagnostics and Attribution")
with logo_col:
    if LOGO_PATH:
        st.image(LOGO_PATH, width=80)

# ==========================================
# 2. Portfolio Data Ingestion
# ==========================================
st.sidebar.header("Portfolio Parameters")
conn      = st.connection("gsheets", type=GSheetsConnection)
sheet_url = (
    "https://docs.google.com/spreadsheets/d/"
    "11Fpu0mz2JevRHzqrvgR-TrC78qbALWAS1Yp8C2gEi4U/edit#gid=418083832"
)

try:
    with st.spinner("Fetching active portfolio data..."):
        portfolio_data         = conn.read(spreadsheet=sheet_url, ttl="10m")
        portfolio_data.columns = portfolio_data.columns.str.strip()

        df = portfolio_data[['Date', 'Ticker', 'Value', '% Return']].copy()
        df = df.dropna(subset=['Ticker'])
        df = df[~df['Ticker'].astype(str).str.contains('Total', case=False, na=False)]

        df['Value'] = (df['Value'].astype(str)
                       .str.replace('$', '', regex=False)
                       .str.replace(',', '', regex=False)
                       .str.replace('(', '-', regex=False)
                       .str.replace(')', '',  regex=False))
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

        df['% Return'] = (df['% Return'].astype(str)
                          .str.replace('%', '', regex=False)
                          .str.replace('(', '-', regex=False)
                          .str.replace(')', '',  regex=False))
        df['% Return'] = pd.to_numeric(df['% Return'], errors='coerce') / 100
        df             = df[df['Value'] > 0]

        cash_df      = df[df['Ticker'].astype(str).str.contains('Cash', case=False, na=False)]
        cash_balance = cash_df['Value'].sum() if not cash_df.empty else 0.0

        equity_df    = df[~df['Ticker'].astype(str).str.contains('Cash', case=False, na=False)].copy()
        equity_value = equity_df['Value'].sum()
        equity_df['Active_Weight'] = equity_df['Value'] / equity_value

        tickers        = equity_df['Ticker'].astype(str).str.strip().tolist()
        weights        = np.array(equity_df['Active_Weight'].tolist())
        actual_returns = np.array(equity_df['% Return'].fillna(0.0).tolist())

        st.sidebar.success("Active portfolio data loaded.")
        st.sidebar.write(f"**Gross Exposure:** ${equity_value:,.2f}")

except Exception as e:
    st.sidebar.error(f"Failed to connect to data source. Error: {e}")
    st.stop()

confidence_level = st.sidebar.slider("Confidence Level", 0.90, 0.99, 0.95)
lookback_years   = st.sidebar.slider("Regression Lookback (Years)", 1, 15, 5)
rf_rate          = st.sidebar.number_input("Risk-Free Rate (%)", value=5.0, step=0.25) / 100

end_date   = datetime.date.today()
start_date = end_date - datetime.timedelta(days=365 * lookback_years)

# ==========================================
# 3. Market Data & Factor Proxy Construction
# ==========================================
mag7_components = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA']

standard_proxies = {
    'Market (Beta)'          : 'SPY',   
    'Size (Small Cap)'       : 'IWM',   
    'Value'                  : 'IWD',   
    'Growth'                 : 'IWF',   
    'Momentum'               : 'MTUM',  
    'Quality'                : 'QUAL',  
    'Oil Prices'             : 'USO',   
    'Interest Rates (7-10y)' : 'IEF',   
    'Gold'                   : 'GLD',   
    'US Dollar'              : 'UUP',   
}

@st.cache_data
def load_deep_history(port_tickers, factor_dict, mag7_tickers):
    all_symbols = list(set(port_tickers + list(factor_dict.values()) + mag7_tickers))
    data        = yf.download(all_symbols, start="2007-01-01",
                              end=datetime.date.today())['Close']
    data        = data.ffill()
    data.index  = data.index.tz_localize(None)
    returns_df  = np.log(data / data.shift(1))
    returns_df['Mag_7_Proxy'] = returns_df[list(mag7_tickers)].mean(axis=1)
    return returns_df

with st.spinner("Compiling historical market and factor data..."):
    full_returns = load_deep_history(tickers, standard_proxies, mag7_components)

regression_returns = full_returns.loc[
    pd.to_datetime(start_date):pd.to_datetime(end_date)
].copy().dropna()

factor_proxies                   = standard_proxies.copy()
factor_proxies['Mag 7 Exposure'] = 'Mag_7_Proxy'
factor_names                     = list(factor_proxies.keys())

port_component_returns = regression_returns[tickers]
portfolio_returns      = port_component_returns.dot(weights)

# Raw factor returns
raw_factor_returns         = regression_returns[list(factor_proxies.values())]
raw_factor_returns.columns = factor_names

# Orthogonalized factor returns
with st.spinner("Orthogonalizing factors..."):
    ortho_factor_returns = orthogonalize_factors(raw_factor_returns)

st.divider()

# ==========================================
# 4. Return Attribution
# ==========================================
st.header("1. Return Attribution")

contrib_df = pd.DataFrame({
    'Portfolio Weight'         : weights,
    'Total Return Contribution': actual_returns * weights,
}, index=tickers).sort_values(by='Total Return Contribution', ascending=False)

col1, col2 = st.columns(2)
with col1:
    st.dataframe(contrib_df.style.format({
        'Portfolio Weight'         : '{:.2%}',
        'Total Return Contribution': '{:.2%}',
    }), use_container_width=True)

with col2:
    bar_c = [COLORS['NAVY'] if v >= 0 else COLORS['RED']
             for v in contrib_df['Total Return Contribution']]
    fig_attr, ax_attr = styled_fig((8, max(4, len(tickers) * 0.45)))
    ax_attr.barh(contrib_df.index,
                 contrib_df['Total Return Contribution'] * 100,
                 color=bar_c, edgecolor=COLORS['BG'], linewidth=0.5)
    ax_attr.axvline(0, color=COLORS['SLATE'], linewidth=0.8, linestyle='--')
    ax_attr.set_xlabel("Return Contribution (%)", fontsize=9)
    ax_attr.set_title("Return Contribution by Holding",
                      fontsize=11, fontweight='bold', color=COLORS['NAVY'], pad=10)
    ax_attr.tick_params(labelsize=8)
    plt.tight_layout()
    st.pyplot(fig_attr)
    plt.close()

st.divider()

# ==========================================
# 5. Factor Risk Decomposition
# ==========================================
st.header("2. Factor Risk Decomposition")


# ---- OLS on ORTHOGONALIZED factors ----
X_ortho  = ortho_factor_returns.copy()
X_const  = sm.add_constant(X_ortho)

port_model = sm.OLS(portfolio_returns, X_const).fit(cov_type='HC3')

factor_results = pd.DataFrame({
    'Factor Sensitivities (Beta)': port_model.params[1:],
    'P-Value'                    : port_model.pvalues[1:],
})
significant_factors = factor_results[factor_results['P-Value'] < 0.05]

# ---- VIF on orthogonalized factors ----
vif_values = [variance_inflation_factor(X_const.values, i + 1)
              for i in range(X_ortho.shape[1])]

vif_df = pd.DataFrame({
    'Factor'  : factor_names,
    'VIF (Raw Factors)'   : [variance_inflation_factor(
                                sm.add_constant(raw_factor_returns).values, i + 1)
                             for i in range(raw_factor_returns.shape[1])],
    'VIF (Orthogonalized)': vif_values,
}).sort_values('VIF (Raw Factors)', ascending=False).reset_index(drop=True)

vif_df['Status (Raw)']   = vif_df['VIF (Raw Factors)'].apply(
    lambda x: 'High (>10)' if x > 10 else ('Moderate (5-10)' if x > 5 else 'Low (<5)')
)
vif_df['Status (Ortho)'] = vif_df['VIF (Orthogonalized)'].apply(
    lambda x: 'High (>10)' if x > 10 else ('Moderate (5-10)' if x > 5 else 'Low (<5)')
)

# ---- PCA shrinkage on orthogonalized factor covariance ----
K         = X_ortho.shape[1]
F_cov_raw = X_ortho.cov().values

pca_full     = PCA().fit(X_ortho)
cum_exp_var  = pca_full.explained_variance_ratio_.cumsum()
n_components = int(np.searchsorted(cum_exp_var, 0.95)) + 1
n_components = max(1, min(n_components, K))

pca_shrunk   = PCA(n_components=n_components).fit(X_ortho)
V            = pca_shrunk.components_
lambdas      = pca_shrunk.explained_variance_

F_cov_factor = (V.T * lambdas) @ V
F_cov_resid  = np.diag(np.diag(F_cov_raw - F_cov_factor))
F_cov_shrunk = F_cov_factor + F_cov_resid

# ---- Per-stock betas + specific variances ----
B                  = np.zeros((len(tickers), K))
specific_variances = np.zeros(len(tickers))

with st.spinner("Running per-stock factor regressions..."):
    for i, ticker in enumerate(tickers):
        if ticker in regression_returns.columns:
            m_i                   = sm.OLS(regression_returns[ticker], X_const).fit()
            B[i, :]               = m_i.params[1:]
            specific_variances[i] = m_i.resid.var()

# ---- Risk decomposition ----
port_beta        = B.T @ weights
factor_var       = float(port_beta @ F_cov_shrunk @ port_beta) * 252
idio_var         = float((weights ** 2) @ specific_variances)  * 252
total_var        = float(portfolio_returns.var()) * 252

factor_vol       = np.sqrt(max(factor_var, 0))
idio_vol         = np.sqrt(max(idio_var,   0))
total_vol_decomp = np.sqrt(max(total_var,  0))

pct_factor   = factor_var / total_var if total_var > 0 else 0
pct_idio     = idio_var   / total_var if total_var > 0 else 0
pct_residual = max(0.0, 1.0 - pct_factor - pct_idio)

# ---- Display: betas ----
col3, col4 = st.columns([2, 1])
with col3:
    st.subheader("Portfolio Factor Beta Exposures (Pure / Orthogonalized)")
    beta_vals  = factor_results['Factor Sensitivities (Beta)']
    bar_c_beta = [COLORS['NAVY'] if v >= 0 else COLORS['RED'] for v in beta_vals]
    fig_beta, ax_beta = styled_fig((9, 5))
    ax_beta.barh(beta_vals.index, beta_vals.values,
                 color=bar_c_beta, edgecolor=COLORS['BG'], linewidth=0.5)
    ax_beta.axvline(0, color=COLORS['SLATE'], linewidth=0.8, linestyle='--')
    ax_beta.set_xlabel("Beta", fontsize=9)
    ax_beta.set_title(
        "Pure Factor Sensitivities — Gram-Schmidt Orthogonalized",
        fontsize=11, fontweight='bold', color=COLORS['NAVY'], pad=10
    )
    ax_beta.tick_params(labelsize=8)
    plt.tight_layout()
    st.pyplot(fig_beta)
    plt.close()

with col4:
    st.write("**Significant Exposures (p < 0.05)**")
    st.dataframe(
        significant_factors['Factor Sensitivities (Beta)'].apply(lambda x: f"{x:.2f}"),
        use_container_width=True,
    )
    st.write(f"**Adj. R-Squared:** {port_model.rsquared_adj:.2f}")

# ---- VIF before / after comparison (In an expander) ----
with st.expander("Multicollinearity: VIF Before and After Orthogonalization"):
    c_vif1, c_vif2 = st.columns([3, 1])
    with c_vif1:
        st.dataframe(
            vif_df.style
                  .format({'VIF (Raw Factors)': '{:.2f}', 'VIF (Orthogonalized)': '{:.2f}'})
                  .background_gradient(subset=['VIF (Raw Factors)'],    cmap='YlOrRd')
                  .background_gradient(subset=['VIF (Orthogonalized)'], cmap='YlGn_r'),
            use_container_width=True,
        )
    with c_vif2:
        st.markdown("""
    **VIF Reference**
    | Range  | Assessment         |
    |--------|--------------------|
    | < 5    | Low — no issue     |
    | 5–10   | Moderate — monitor |
    | > 10   | High — review      |

    After orthogonalization, all VIFs
    collapse toward 1.0 by
    construction — confirming each
    factor is independent.
    """)

# ---- Factor vs. Idiosyncratic Risk Decomposition ----
st.subheader("Factor vs. Idiosyncratic Risk Decomposition")

rd1, rd2, rd3, rd4 = st.columns(4)
rd1.metric("Total Portfolio Vol",  f"{total_vol_decomp*100:.2f}%")
rd2.metric("Factor Risk (Vol)",    f"{factor_vol*100:.2f}%",
           delta=f"{pct_factor*100:.1f}% of total")
rd3.metric("Idiosyncratic Risk",   f"{idio_vol*100:.2f}%",
           delta=f"{pct_idio*100:.1f}% of total")
idio_label = "Stock-picker driven" if pct_idio > 0.35 else "Factor / market driven"
rd4.metric("Risk Character", idio_label,
           delta=f"Specific ratio: {pct_idio*100:.1f}%")

# Scaled stacked horizontal bar decomposition
fig_rd, ax_rd = styled_fig((9, 2.2))

bar_height = 0.42
segments = [
    (pct_factor,   COLORS['NAVY'], f"Factor Risk\n{pct_factor*100:.1f}%"),
    (pct_idio,     COLORS['RED'],  f"Idiosyncratic\n{pct_idio*100:.1f}%"),
    (pct_residual, COLORS['GRID'], f"Residual\n{pct_residual*100:.1f}%"),
]

left = 0.0
for pct, color, label in segments:
    if pct > 0.001:
        ax_rd.barh(0, pct, left=left, height=bar_height,
                   color=color, edgecolor=COLORS['BG'], linewidth=1.0)
        if pct > 0.05:
            ax_rd.text(left + pct / 2, 0, label,
                       ha='center', va='center', fontsize=8.5,
                       fontweight='bold',
                       color=COLORS['BG'] if color != COLORS['GRID'] else COLORS['SLATE'])
        left += pct

ax_rd.set_xlim(0, 1)
ax_rd.set_ylim(-0.6, 0.6)
ax_rd.set_xlabel("Proportion of Total Variance", fontsize=9, color=COLORS['SLATE'])
ax_rd.set_title(
    f"Portfolio Variance Decomposition  —  Total Annualised Vol: {total_vol_decomp*100:.2f}%",
    fontsize=11, fontweight='bold', color=COLORS['NAVY'], pad=10
)
ax_rd.set_yticks([])
ax_rd.spines['left'].set_visible(False)
ax_rd.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
ax_rd.tick_params(axis='x', labelsize=8)

for boundary, label in [
    (pct_factor,              f"{factor_vol*100:.1f}%\nvol"),
    (pct_factor + pct_idio,   f"{idio_vol*100:.1f}%\nvol"),
]:
    ax_rd.axvline(boundary, color=COLORS['SLATE'], linewidth=0.7,
                  linestyle=':', ymin=0.05, ymax=0.95)

plt.tight_layout()
st.pyplot(fig_rd)
plt.close()

st.divider()

# ==========================================
# 6. Portfolio Risk Analytics
# ==========================================
st.header("3. Portfolio Risk Analytics")

port_volatility = float(np.std(portfolio_returns)) * np.sqrt(252)
alpha           = 1 - confidence_level
historical_var  = float(np.percentile(portfolio_returns, alpha * 100))
cvar            = float(portfolio_returns[portfolio_returns <= historical_var].mean())
annual_return   = float(portfolio_returns.mean()) * 252

sharpe       = (annual_return - rf_rate) / port_volatility if port_volatility > 0 else np.nan
downside     = portfolio_returns[portfolio_returns < 0]
downside_std = float(np.std(downside)) * np.sqrt(252) if len(downside) > 1 else np.nan
sortino      = (annual_return - rf_rate) / downside_std if (downside_std and downside_std > 0) else np.nan

cum_ret      = (1 + portfolio_returns).cumprod()
rolling_max  = cum_ret.cummax()
drawdown_s   = (cum_ret - rolling_max) / rolling_max
max_drawdown = float(drawdown_s.min())
calmar       = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

m1, m2, m3 = st.columns(3)
m4, m5, m6 = st.columns(3)
m7, m8, m9 = st.columns(3)

m1.metric("Annualized Return",         f"{annual_return*100:.2f}%")
m2.metric("Annualized Volatility",      f"{port_volatility*100:.2f}%")
m3.metric(f"Hist. VaR ({confidence_level*100:.0f}%)", f"{historical_var*100:.2f}%")
m4.metric("CVaR / Expected Shortfall", f"{cvar*100:.2f}%")
m5.metric("Sharpe Ratio",              f"{sharpe:.2f}",
          help=f"(Return - Rf) / Vol  |  Rf = {rf_rate*100:.2f}%")
m6.metric("Sortino Ratio",             f"{sortino:.2f}",
          help="(Return - Rf) / Downside Vol — penalises negative returns only")

m7.metric("Max Drawdown", f"{max_drawdown*100:.2f}%")
m8.metric("Calmar Ratio", f"{calmar:.2f}", help="Annualised Return / |Max Drawdown|")

st.divider()

# ==========================================
# 7. Factor Exposure Quilt Diagram
# ==========================================
st.header("4. Factor Exposure Quilt — Current Holdings")
st.markdown(
    "Each cell shows a holding's OLS beta to the corresponding *orthogonalized* factor. "
    "Every beta is a pure, independent effect — market overlap has been removed from all style factors."
)

quilt_df = pd.DataFrame(B, index=tickers, columns=factor_names)

fig_quilt, ax_quilt = plt.subplots(
    figsize=(max(16, len(factor_names) * 1.3), max(6, len(tickers) * 0.7))
)
fig_quilt.patch.set_facecolor(COLORS['BG'])

sns.heatmap(
    quilt_df,
    annot=True, fmt=".2f",
    cmap=DIVERGING_CMAP,
    center=0,
    linewidths=0.6, linecolor=COLORS['GRID'],
    ax=ax_quilt,
    cbar_kws={'label': 'Factor Beta', 'shrink': 0.65},
    annot_kws={"size": 8, "weight": "bold", "color": COLORS['BG']},
)
ax_quilt.set_title(
    "Factor Exposure Quilt: Per-Stock Pure Beta Loadings (Orthogonalized Factors)",
    fontsize=13, fontweight='bold', color=COLORS['NAVY'], pad=14,
)
ax_quilt.set_xlabel("Risk Factors",       fontsize=10, color=COLORS['SLATE'])
ax_quilt.set_ylabel("Portfolio Holdings", fontsize=10, color=COLORS['SLATE'])
ax_quilt.tick_params(axis='x', rotation=35, labelsize=9, colors=COLORS['SLATE'])
ax_quilt.tick_params(axis='y', rotation=0,  labelsize=9, colors=COLORS['SLATE'])
cbar1 = ax_quilt.collections[0].colorbar
cbar1.ax.tick_params(labelsize=8, colors=COLORS['SLATE'])
cbar1.set_label('Factor Beta', color=COLORS['SLATE'], fontsize=9)
plt.tight_layout()
st.pyplot(fig_quilt)
plt.close()

# ---- Factor correlation matrix — (In an expander) ----
with st.expander("Factor Correlation Matrix — Post-Orthogonalization (Target: Near-Diagonal)"):
    std_diag    = np.sqrt(np.maximum(np.diag(F_cov_shrunk), 1e-12))
    F_corr_norm = F_cov_shrunk / np.outer(std_diag, std_diag)
    np.fill_diagonal(F_corr_norm, 1.0)
    F_corr_df   = pd.DataFrame(F_corr_norm, index=factor_names, columns=factor_names)

    fig_corr, ax_corr = plt.subplots(figsize=(11, 8))
    fig_corr.patch.set_facecolor(COLORS['BG'])

    sns.heatmap(
        F_corr_df,
        annot=True, fmt=".2f",
        cmap=DIVERGING_CMAP,
        center=0, vmin=-1, vmax=1,
        linewidths=0.5, linecolor=COLORS['GRID'],
        ax=ax_corr,
        annot_kws={"size": 8, "weight": "bold", "color": COLORS['BG']},
        square=True,
    )
    ax_corr.set_title(
        "Post-Orthogonalization Factor Correlation Matrix\n"
        "Off-diagonal values near zero confirm successful factor independence",
        fontsize=12, fontweight='bold', color=COLORS['NAVY'],
    )
    ax_corr.tick_params(axis='x', rotation=35, labelsize=8, colors=COLORS['SLATE'])
    ax_corr.tick_params(axis='y', rotation=0,  labelsize=8, colors=COLORS['SLATE'])
    ax_corr.collections[0].colorbar.ax.tick_params(labelsize=8, colors=COLORS['SLATE'])
    plt.tight_layout()
    st.pyplot(fig_corr)
    plt.close()

st.divider()

# ==========================================
# 8. Historical Scenario Analysis
# ==========================================
st.header("5. Historical Scenario Analysis")
st.markdown("Constant-weight simulation of the current portfolio through severe historical drawdowns.")

stress_periods = {
    "Global Financial Crisis": ("2007-10-09", "2009-03-09"),
    "COVID-19 Crash"         : ("2020-02-19", "2020-03-23"),
    "2022 Rate Selloff"      : ("2022-01-03", "2022-10-12"),
}

deep_port_returns = full_returns[tickers].fillna(0).dot(weights)
stress_results    = []

for event, (s, e) in stress_periods.items():
    try:
        ev_port = deep_port_returns.loc[s:e]
        ev_spy  = full_returns['SPY'].loc[s:e]
        port_dd = np.exp(ev_port.sum()) - 1
        spy_dd  = np.exp(ev_spy.sum())  - 1
        stress_results.append({
            "Scenario"                  : event,
            "Date Range"                : f"{s} to {e}",
            "Portfolio Drawdown"        : port_dd,
            "Benchmark Drawdown"        : spy_dd,
            "Active Drawdown Difference": port_dd - spy_dd,
        })
    except Exception:
        continue

stress_df = pd.DataFrame(stress_results)
if not stress_df.empty:
    st.dataframe(
        stress_df.style.format({
            "Portfolio Drawdown"        : "{:.2%}",
            "Benchmark Drawdown"        : "{:.2%}",
            "Active Drawdown Difference": "{:.2%}",
        }),
        use_container_width=True,
    )

st.divider()

# ==========================================
# 9. Forward-Looking Investment Pitch
# ==========================================
st.header("6. Forward-Looking Investment Pitch")
st.markdown("Quarterly Factor Outlook and Allocation Recommendations")

top_4_factors = factor_results.sort_values(
    by='Factor Sensitivities (Beta)', key=abs, ascending=False
).head(4)
recent_quarter_returns = ortho_factor_returns[top_4_factors.index].tail(63).sum()

st.write("**Top 4 Factor Exposures by Absolute Beta:**")
st.dataframe(top_4_factors.style.format({
    'Factor Sensitivities (Beta)': '{:.2f}',
    'P-Value'                    : '{:.4f}',
}), use_container_width=True)

st.write("**Trailing Quarterly Momentum — Orthogonalized Factors (63-day log return):**")
qtr_colors = [COLORS['NAVY'] if v >= 0 else COLORS['RED']
              for v in recent_quarter_returns.values]
fig_qtr, ax_qtr = styled_fig((8, 4))
ax_qtr.bar(recent_quarter_returns.index, recent_quarter_returns.values * 100,
           color=qtr_colors, edgecolor=COLORS['BG'], linewidth=0.5)
ax_qtr.axhline(0, color=COLORS['SLATE'], linewidth=0.8, linestyle='--')
ax_qtr.set_ylabel("63-Day Log Return (%)", fontsize=9)
ax_qtr.set_title("Trailing Quarterly Factor Momentum (Pure Factor Returns)",
                 fontsize=11, fontweight='bold', color=COLORS['NAVY'], pad=10)
ax_qtr.tick_params(axis='x', rotation=20, labelsize=9)
ax_qtr.tick_params(axis='y', labelsize=8)
plt.tight_layout()
st.pyplot(fig_qtr)
plt.close()