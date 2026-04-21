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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
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
    'bec_diverging_soft',
    [
        (0.00, COLORS['RED']),
        (0.47, '#f3c892'),
        (0.50, '#ffffff'),
        (0.53, '#c7d4e5'),
        (1.00, COLORS['NAVY'])
    ],
    N=256
)

# ==========================================
# LOGO CONFIGURATION (Safe Cloud Loading)
# ==========================================
LOGO_FILENAME = "Blue Circle Icon.png"
LOGO_PATH     = None

try:
    if os.path.exists(LOGO_FILENAME):
        LOGO_PATH = LOGO_FILENAME
except Exception:
    LOGO_PATH = None


def styled_fig(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(COLORS['BG'])
    ax.set_facecolor(COLORS['BG'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['GRID'])
    ax.spines['bottom'].set_color(COLORS['GRID'])
    return fig, ax


def _yahoo_symbol(sym: str) -> str:
    """Match Yahoo Finance listing style (uppercase; class shares use '-')."""
    s = str(sym).strip().upper()
    return s.replace(".", "-") if s else s


def _window_factor_col(name: str) -> str:
    """Column name in price/returns frame (ETFs normalized; synthetic factors unchanged)."""
    return name if name == "Mag_7_Proxy" else _yahoo_symbol(name)


def _extract_close_prices(raw: pd.DataFrame, symbols: list) -> pd.DataFrame:
    """
    Normalize yfinance download output to a single DataFrame of close prices
    with one column per symbol (handles MultiIndex vs flat columns).
    """
    syms = [_yahoo_symbol(s) for s in symbols]
    if raw is None or raw.empty:
        return pd.DataFrame(columns=syms)

    if isinstance(raw.columns, pd.MultiIndex):
        top       = raw.columns.get_level_values(0).unique()
        price_key = ("Close" if "Close" in top
                     else ("Adj Close" if "Adj Close" in top else top[0]))
        panel = raw[price_key].copy()
        if isinstance(panel, pd.Series):
            panel = panel.to_frame(name=syms[0])
        panel.columns = [_yahoo_symbol(c) for c in panel.columns]
    else:
        if "Close" not in raw.columns:
            return pd.DataFrame(columns=syms)
        col   = raw["Close"]
        panel = col.to_frame(name=syms[0]) if isinstance(col, pd.Series) else col.copy()
        if panel.shape[1] == 1 and syms:
            panel.columns = [syms[0]]
        else:
            panel.columns = [_yahoo_symbol(c) for c in panel.columns]

    return panel.reindex(columns=syms)


# ==========================================
# ORTHOGONALIZATION — Gram-Schmidt / Sequential OLS
# ==========================================
def orthogonalize_factors(raw_factor_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sequentially orthogonalize factors via OLS residuals (Gram-Schmidt).
    """
    df = (
        raw_factor_df.dropna(how="any")
        .apply(pd.to_numeric, errors="coerce")
        .dropna(how="any")
    )
    cols = df.columns.tolist()
    if not cols or df.shape[0] < len(cols) + 1:
        raise ValueError(
            f"Insufficient data for factor orthogonalization "
            f"(rows={df.shape[0]}, factors={len(cols)})."
        )

    ortho_df          = pd.DataFrame(index=df.index)
    ortho_df[cols[0]] = df[cols[0]]  # Market anchor — unchanged

    for i in range(1, len(cols)):
        y       = df[cols[i]]
        X_prior = sm.add_constant(ortho_df.iloc[:, :i])
        resid   = sm.OLS(y, X_prior).fit().resid
        ortho_df[cols[i]] = resid

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
# 2. Portfolio Data Ingestion (Public CSV Bypass)
# ==========================================
st.sidebar.header("Portfolio Parameters")

sheet_id = "11Fpu0mz2JevRHzqrvgR-TrC78qbALWAS1Yp8C2gEi4U"
tab_gid  = "418083832"
csv_url  = (
    f"https://docs.google.com/spreadsheets/d/{sheet_id}"
    f"/export?format=csv&gid={tab_gid}"
)

try:
    with st.spinner("Fetching active portfolio data..."):
        portfolio_data         = pd.read_csv(csv_url)
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

        tickers        = [
            _yahoo_symbol(t) for t in equity_df['Ticker'].astype(str).str.strip().tolist()
        ]
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
    raw_syms    = port_tickers + list(factor_dict.values()) + list(mag7_tickers)
    all_symbols = list(dict.fromkeys(_yahoo_symbol(s) for s in raw_syms))

    raw = pd.DataFrame()
    for attempt in range(3):
        raw = yf.download(
            all_symbols,
            start="2007-01-01",
            end=datetime.date.today(),
            progress=False,
            threads=False,
        )
        if raw is not None and not raw.empty:
            break

    data = _extract_close_prices(raw, all_symbols)
    if data.empty or data.notna().to_numpy().sum() == 0:
        return pd.DataFrame()

    data = data.sort_index()
    data = data[~data.index.duplicated(keep="last")]
    data = data.ffill()
    data.index = data.index.tz_localize(None)

    returns_df        = np.log(data / data.shift(1))
    mag7_cols         = [_yahoo_symbol(m) for m in mag7_tickers]
    returns_df["Mag_7_Proxy"] = (
        returns_df.reindex(columns=mag7_cols)[mag7_cols].mean(axis=1)
    )
    return returns_df


with st.spinner("Compiling historical market and factor data..."):
    full_returns = load_deep_history(tickers, standard_proxies, mag7_components)

if full_returns is None or full_returns.empty:
    st.error(
        "Yahoo Finance returned no usable price data (common on cloud when rate-limited). "
        "Wait a minute and reload, or pin a stable `yfinance` version in requirements.txt."
    )
    st.stop()

factor_proxies               = standard_proxies.copy()
factor_proxies['Mag 7 Exposure'] = 'Mag_7_Proxy'
factor_names                 = list(factor_proxies.keys())
factor_cols                  = list(factor_proxies.values())

_reg_start      = pd.to_datetime(start_date)
_reg_end        = pd.to_datetime(end_date)
window          = full_returns.sort_index().loc[_reg_start:_reg_end].copy()
factor_cols_use = [_window_factor_col(c) for c in factor_cols]

missing_factors = [c for c, w in zip(factor_cols, factor_cols_use)
                   if w not in window.columns]
if missing_factors:
    st.error(
        "One or more factor series are missing from the Yahoo download. "
        f"Missing: {missing_factors}. "
        f"Sample columns: {list(window.columns)[:30]}..."
    )
    st.stop()

ts_tickers = [t for t in tickers
              if t in window.columns and window[t].notna().any()]
missing_px = [t for t in tickers if t not in ts_tickers]

if missing_px:
    st.warning(
        "No usable Yahoo price history in this window for: "
        + ", ".join(missing_px)
        + ". Those holdings are excluded from time-series regressions; "
        "weights are renormalized over symbols with data."
    )
if not ts_tickers:
    st.error("None of the portfolio tickers have usable price data in this lookback window.")
    st.stop()

subset_cols = factor_cols_use + ts_tickers
try:
    regression_returns = window.dropna(subset=subset_cols, how="any").copy()
except KeyError as e:
    st.error(
        "Market data is missing required factor or ticker columns. "
        f"Details: {e}"
    )
    st.stop()

if regression_returns.shape[0] == 0:
    na_days = window[subset_cols].isna().sum().sort_values(ascending=False)
    st.error(
        "No overlapping days with complete factor and portfolio returns. "
        "Below: days with missing data per series in the lookback (largest first). "
        "Fix tickers on the sheet (Yahoo symbols, e.g. BRK-B not BRK.B) or widen the window."
    )
    st.dataframe(na_days.to_frame("missing_days"), use_container_width=True)
    st.stop()

min_obs = max(30, len(factor_names) + 5)
if regression_returns.shape[0] < min_obs:
    st.error(
        f"Need at least {min_obs} trading days with complete data for factor models; "
        f"found {regression_returns.shape[0]}."
    )
    st.stop()

w_ts                   = np.array([weights[tickers.index(t)] for t in ts_tickers])
w_ts                   = w_ts / w_ts.sum()
port_component_returns = regression_returns[ts_tickers]
portfolio_returns      = port_component_returns.dot(w_ts)

raw_factor_returns         = regression_returns[factor_cols_use].copy()
raw_factor_returns.columns = factor_names

with st.spinner("Orthogonalizing factors..."):
    try:
        ortho_factor_returns = orthogonalize_factors(raw_factor_returns)
    except ValueError as e:
        st.error(str(e))
        st.stop()

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
X_ortho    = ortho_factor_returns.copy()
X_const    = sm.add_constant(X_ortho)
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
    'Factor'              : factor_names,
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
        vif_display = vif_df.copy()
        vif_display['VIF (Raw Factors)'] = vif_display['VIF (Raw Factors)'].round(2)
        vif_display['VIF (Orthogonalized)'] = vif_display['VIF (Orthogonalized)'].round(2)
        st.dataframe(
            vif_display.style
                .background_gradient(subset=['VIF (Raw Factors)'], cmap='YlOrRd')
                .apply(lambda s: ['background-color: #d9f99d'] * len(s), subset=['VIF (Orthogonalized)']),
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

fig_rd, ax_rd = styled_fig((9, 2.2))
bar_height = 0.42
segments   = [
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
                       ha='center', va='center', fontsize=8.5, fontweight='bold',
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
for boundary in [pct_factor, pct_factor + pct_idio]:
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
sortino      = ((annual_return - rf_rate) / downside_std
                if (downside_std and downside_std > 0) else np.nan)

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
m7.metric("Max Drawdown",              f"{max_drawdown*100:.2f}%")
m8.metric("Calmar Ratio",              f"{calmar:.2f}",
          help="Annualised Return / |Max Drawdown|")

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
abs_max = np.nanpercentile(np.abs(quilt_df.values), 95)
abs_max = max(abs_max, 0.25)
norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

sns.heatmap(
    quilt_df,
    annot=True, fmt=".2f",
    cmap=DIVERGING_CMAP,
    norm=norm,
    linewidths=0.8, linecolor=COLORS['GRID'],
    ax=ax_quilt,
    cbar_kws={'label': 'Factor Beta', 'shrink': 0.75},
    annot_kws={"size": 8, "weight": "bold", "color": COLORS['BG']},
)

for text, val in zip(ax_quilt.texts, quilt_df.to_numpy().ravel()):
    text.set_color('white' if abs(val) >= abs_max * 0.45 else '#0f172a')
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
# 9. Manager Evaluation — Benchmark-Relative Metrics & Risk Scorecard
#    Benchmark: ACWI (iShares MSCI ACWI ETF — closest free proxy for MSCI ACWI IMI)
#    Portfolio classification: Concentrated / High-Conviction
# ==========================================
st.header("6. Manager Evaluation — Benchmark-Relative Risk Assessment")
st.caption(
    "Benchmark: ACWI (iShares MSCI ACWI ETF) — used as a practical proxy for MSCI ACWI IMI. "
    "Scoring thresholds are calibrated for a concentrated, high-conviction portfolio."
)


@st.cache_data
def load_benchmark(start: str, end: str) -> pd.Series:
    raw  = yf.download("ACWI", start=start, end=end, progress=False, threads=False)
    data = _extract_close_prices(raw, ["ACWI"])
    if data.empty:
        return pd.Series(dtype=float)
    data = data.sort_index().ffill()
    data.index = data.index.tz_localize(None)
    return np.log(data["ACWI"] / data["ACWI"].shift(1)).dropna()


with st.spinner("Fetching benchmark (ACWI) data..."):
    bench_daily = load_benchmark(
        str(start_date - datetime.timedelta(days=10)), str(end_date)
    )

if bench_daily.empty:
    st.error("Could not load ACWI benchmark data from Yahoo Finance. Try reloading.")
    st.stop()

# Align daily returns
common_idx    = portfolio_returns.index.intersection(bench_daily.index)
port_aligned  = portfolio_returns.reindex(common_idx)
bench_aligned = bench_daily.reindex(common_idx)

# Monthly returns (sum of log returns within month)
port_monthly  = port_aligned.resample("ME").sum()
bench_monthly = bench_aligned.resample("ME").sum()
common_months = port_monthly.index.intersection(bench_monthly.index)
port_m        = port_monthly.reindex(common_months)
bench_m       = bench_monthly.reindex(common_months)
alpha_m       = port_m - bench_m

# ---- Core 5 Metric Calculations ----

# 1. Tracking Error
daily_excess   = port_aligned - bench_aligned
tracking_error = float(daily_excess.std() * np.sqrt(252))

# 2. Hit Rate
hit_rate = float((alpha_m > 0).mean()) if len(alpha_m) > 0 else np.nan

# 3. Slugging
pos_alpha = alpha_m[alpha_m > 0]
neg_alpha = alpha_m[alpha_m < 0]
slugging  = (float(pos_alpha.mean()) / abs(float(neg_alpha.mean()))
             if len(pos_alpha) > 0 and len(neg_alpha) > 0 else np.nan)

# 4. Up Capture
up_months  = bench_m[bench_m > 0]
up_capture = (float(port_m.reindex(up_months.index).mean()) / float(up_months.mean())
              if len(up_months) > 0 else np.nan)

# 5. Down Capture
down_months  = bench_m[bench_m < 0]
down_capture = (float(port_m.reindex(down_months.index).mean()) / float(down_months.mean())
                if len(down_months) > 0 else np.nan)

# ---- Benchmark-Relative Metrics Table ----
st.subheader("Benchmark-Relative Metrics")

n_months     = len(common_months)
metrics_data = {
    "Metric"     : ["Tracking Error", "Hit Rate", "Slugging Ratio",
                    "Up Capture", "Down Capture"],
    "Value"      : [
        f"{tracking_error*100:.2f}%",
        f"{hit_rate*100:.1f}%"        if not np.isnan(hit_rate)     else "N/A",
        f"{slugging:.2f}x"            if not np.isnan(slugging)     else "N/A",
        f"{up_capture*100:.1f}%"      if not np.isnan(up_capture)   else "N/A",
        f"{down_capture*100:.1f}%"    if not np.isnan(down_capture) else "N/A",
    ],
    "Definition" : [
        "Annualised vol of daily excess returns vs ACWI",
        "% of months portfolio outperforms ACWI",
        "Avg win alpha / avg loss alpha — how much bigger are wins than losses",
        "Avg portfolio return in up-benchmark months / avg benchmark return",
        "Avg portfolio return in down-benchmark months / avg benchmark return",
    ],
    "Concentrated Benchmark": [
        "4–15% expected",
        ">50% acceptable; >55% strong",
        ">1.5x strong; >1.75x very strong",
        ">100% ideal",
        "<100% ideal; <90% strong",
    ],
}
st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)
st.caption(f"Calculated over {n_months} months of overlapping data (ACWI proxy for MSCI ACWI IMI).")

# ---- Scoring Functions (Concentrated / High-Conviction thresholds) ----

def score_tracking_error(te: float) -> float:
    te_pct = te * 100
    if   te_pct >= 6  and te_pct <= 15: return 9.0
    elif te_pct >= 4  and te_pct <  6:  return 7.0
    elif te_pct > 15  and te_pct <= 20: return 6.0
    elif te_pct >= 2  and te_pct <  4:  return 4.0
    elif te_pct > 20:                   return 3.0
    else:                               return 2.0

def score_hit_rate(hr: float) -> float:
    hr_pct = hr * 100
    if   hr_pct >= 55: return 9.5
    elif hr_pct >= 50: return 7.5
    elif hr_pct >= 45: return 5.5
    elif hr_pct >= 40: return 3.5
    else:              return 2.0

def score_slugging(sl: float) -> float:
    if   sl >= 2.00: return 10.0
    elif sl >= 1.75: return 8.5
    elif sl >= 1.50: return 7.0
    elif sl >= 1.25: return 5.0
    elif sl >= 1.00: return 3.0
    else:            return 1.5

def score_up_capture(uc: float) -> float:
    uc_pct = uc * 100
    if   uc_pct >= 105: return 10.0
    elif uc_pct >= 100: return 8.5
    elif uc_pct >= 90:  return 6.5
    elif uc_pct >= 80:  return 4.0
    else:               return 2.0

def score_down_capture(dc: float) -> float:
    dc_pct = dc * 100
    if   dc_pct <= 80:  return 10.0
    elif dc_pct <= 90:  return 8.5
    elif dc_pct <= 100: return 7.0
    elif dc_pct <= 110: return 4.5
    else:               return 2.0

def score_idio_pct(idio: float) -> float:
    idio_pct = idio * 100
    if   idio_pct >= 70: return 10.0
    elif idio_pct >= 60: return 8.0
    elif idio_pct >= 50: return 6.0
    elif idio_pct >= 40: return 4.0
    else:                return 2.0

# ---- Compute component scores ----
s_te   = score_tracking_error(tracking_error) if not np.isnan(tracking_error) else 5.0
s_hr   = score_hit_rate(hit_rate)             if not np.isnan(hit_rate)        else 5.0
s_sl   = score_slugging(slugging)             if not np.isnan(slugging)        else 5.0
s_uc   = score_up_capture(up_capture)         if not np.isnan(up_capture)      else 5.0
s_dc   = score_down_capture(down_capture)     if not np.isnan(down_capture)    else 5.0
s_idio = score_idio_pct(pct_idio)

# Weights — calibrated for concentrated / high-conviction portfolio
SCORE_WEIGHTS = {
    "Idiosyncratic Risk %": 0.25,
    "Slugging Ratio"      : 0.25,
    "Up Capture"          : 0.15,
    "Down Capture"        : 0.15,
    "Hit Rate"            : 0.10,
    "Tracking Error"      : 0.10,
}
component_scores = {
    "Idiosyncratic Risk %": s_idio,
    "Slugging Ratio"      : s_sl,
    "Up Capture"          : s_uc,
    "Down Capture"        : s_dc,
    "Hit Rate"            : s_hr,
    "Tracking Error"      : s_te,
}
overall_score = sum(
    component_scores[k] * SCORE_WEIGHTS[k] for k in SCORE_WEIGHTS
)

# ---- Scorecard Display ----
st.subheader("Risk Scorecard — Concentrated Manager Evaluation")

score_col, breakdown_col = st.columns([1, 2])

with score_col:
    if   overall_score >= 8.0: band, band_color = "Strong",   COLORS['NAVY']
    elif overall_score >= 6.0: band, band_color = "Adequate", COLORS['SLATE']
    elif overall_score >= 4.0: band, band_color = "Weak",     COLORS['RED']
    else:                      band, band_color = "Poor",      COLORS['RED']

    fig_gauge, ax_gauge = plt.subplots(figsize=(3.8, 3.8))
    fig_gauge.patch.set_facecolor(COLORS['BG'])
    ax_gauge.set_facecolor(COLORS['BG'])

    theta = np.linspace(0, 2 * np.pi, 300)
    ax_gauge.plot(np.cos(theta), np.sin(theta),
                  color=COLORS['GRID'], linewidth=14, solid_capstyle='round')

    arc_frac  = overall_score / 10.0
    arc_theta = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi * arc_frac, 300)
    ax_gauge.plot(np.cos(arc_theta), np.sin(arc_theta),
                  color=band_color, linewidth=14, solid_capstyle='round')

    ax_gauge.text(0, 0.12, f"{overall_score:.1f}",
                  ha='center', va='center', fontsize=32,
                  fontweight='bold', color=COLORS['NAVY'])
    ax_gauge.text(0, -0.28, "/ 10",
                  ha='center', va='center', fontsize=13, color=COLORS['SLATE'])
    ax_gauge.text(0, -0.62, band,
                  ha='center', va='center', fontsize=12,
                  fontweight='bold', color=band_color)

    ax_gauge.set_xlim(-1.35, 1.35)
    ax_gauge.set_ylim(-1.35, 1.35)
    ax_gauge.set_aspect('equal')
    ax_gauge.axis('off')
    ax_gauge.set_title("Overall Score", fontsize=11,
                       fontweight='bold', color=COLORS['NAVY'], pad=8)
    plt.tight_layout()
    st.pyplot(fig_gauge)
    plt.close()

with breakdown_col:
    labels     = list(component_scores.keys())
    scores     = list(component_scores.values())
    wt_labels  = [SCORE_WEIGHTS[l] for l in labels]
    bar_colors = [COLORS['NAVY'] if s >= 7.0
                  else (COLORS['SLATE'] if s >= 5.0 else COLORS['RED'])
                  for s in scores]

    # Map each component to its actual underlying metric value
    actual_values = {
        "Idiosyncratic Risk %": f"{pct_idio*100:.1f}%",
        "Slugging Ratio"      : f"{slugging:.2f}x"           if not np.isnan(slugging)       else "N/A",
        "Up Capture"          : f"{up_capture*100:.1f}%"     if not np.isnan(up_capture)     else "N/A",
        "Down Capture"        : f"{down_capture*100:.1f}%"   if not np.isnan(down_capture)   else "N/A",
        "Hit Rate"            : f"{hit_rate*100:.1f}%"       if not np.isnan(hit_rate)       else "N/A",
        "Tracking Error"      : f"{tracking_error*100:.2f}%" if not np.isnan(tracking_error) else "N/A",
    }
    display_labels = [f"{lbl}\n({actual_values.get(lbl, '')})" for lbl in labels]

    fig_comp, ax_comp = styled_fig((7, 4.5))
    y_pos = np.arange(len(labels))

    ax_comp.barh(y_pos, [10] * len(labels), height=0.55,
                 color=COLORS['GRID'], edgecolor=COLORS['BG'])
    ax_comp.barh(y_pos, scores, height=0.55,
                 color=bar_colors, edgecolor=COLORS['BG'])

    for i, (s, w) in enumerate(zip(scores, wt_labels)):
        ax_comp.text(s + 0.2, i, f"{s:.1f}  (wt: {w*100:.0f}%)",
                     va='center', fontsize=8.5, color=COLORS['NAVY'])

    ax_comp.set_yticks(y_pos)
    ax_comp.set_yticklabels(display_labels, fontsize=9)
    ax_comp.set_xlim(0, 13)
    ax_comp.set_xlabel("Component Score (out of 10)", fontsize=9)
    ax_comp.set_title("Component Score Breakdown",
                      fontsize=11, fontweight='bold', color=COLORS['NAVY'], pad=10)

    for xval, lbl in [(7.0, "Strong"), (5.0, "Adequate")]:
        ax_comp.axvline(xval, color=COLORS['SLATE'], linewidth=0.8,
                        linestyle='--', alpha=0.6)
        ax_comp.text(xval + 0.1, len(labels) - 0.35, lbl,
                     fontsize=7.5, color=COLORS['SLATE'], va='top')

    ax_comp.tick_params(axis='x', labelsize=8)
    ax_comp.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig_comp)
    plt.close()

# ---- Narrative summary ----
st.subheader("Summary Assessment")

narratives = []

if s_idio >= 8:
    narratives.append(
        f"Idiosyncratic risk is {pct_idio*100:.1f}% of total variance — above the 60-70% target "
        "for concentrated managers, confirming returns are driven by stock selection rather than factor exposure."
    )
elif s_idio >= 6:
    narratives.append(
        f"Idiosyncratic risk is {pct_idio*100:.1f}% of total variance — within an acceptable range "
        "for concentrated managers, though closer to the 65-70% target would be preferred."
    )
else:
    narratives.append(
        f"Idiosyncratic risk is {pct_idio*100:.1f}% of total variance — below the 60-70% threshold "
        "expected for concentrated managers. A larger share of risk is explained by factor exposure "
        "rather than stock selection."
    )

if not np.isnan(slugging):
    if s_sl >= 8:
        narratives.append(
            f"Slugging ratio of {slugging:.2f}x is strong — average winning months outperform "
            "losing months by a wide margin, consistent with a high-conviction approach."
        )
    elif s_sl >= 6:
        narratives.append(
            f"Slugging ratio of {slugging:.2f}x is moderate — wins are larger than losses "
            "but not by enough to fully compensate for a lower hit rate."
        )
    else:
        narratives.append(
            f"Slugging ratio of {slugging:.2f}x is below target. For a concentrated manager "
            "with a lower hit rate, this needs to be above 1.5x to justify the approach."
        )

if not np.isnan(up_capture) and not np.isnan(down_capture):
    uc_pct = up_capture   * 100
    dc_pct = down_capture * 100
    if uc_pct >= 100 and dc_pct <= 100:
        narratives.append(
            f"Capture ratio profile is favourable: Up Capture of {uc_pct:.1f}% and "
            f"Down Capture of {dc_pct:.1f}% — capturing more upside than downside vs ACWI."
        )
    elif uc_pct < 100 and dc_pct > 100:
        narratives.append(
            f"Capture profile is unfavourable: Up Capture of {uc_pct:.1f}% and Down Capture "
            f"of {dc_pct:.1f}% — the portfolio gives back more on the downside than it gains "
            "on the upside. This is the combination flagged as a red flag by the risk manager."
        )
    else:
        narratives.append(
            f"Up Capture is {uc_pct:.1f}% and Down Capture is {dc_pct:.1f}% vs ACWI. "
            "Monitoring this relationship over time is important."
        )

if not np.isnan(hit_rate):
    hr_pct = hit_rate * 100
    if hr_pct >= 55:
        narratives.append(
            f"Hit Rate of {hr_pct:.1f}% is above the 55% threshold — the portfolio outperforms "
            "the benchmark in the majority of months."
        )
    elif hr_pct >= 45:
        narratives.append(
            f"Hit Rate of {hr_pct:.1f}% is acceptable for a concentrated strategy, provided "
            "the slugging ratio compensates for the months of underperformance."
        )
    else:
        narratives.append(
            f"Hit Rate of {hr_pct:.1f}% is below 45% — the portfolio underperforms ACWI "
            "in more than half of months. This is only acceptable if the slugging ratio is very high."
        )

for para in narratives:
    st.markdown(f"- {para}")