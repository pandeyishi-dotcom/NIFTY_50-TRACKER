# app.py
"""
NIFTY 50 Live Dashboard — Robust (updated)
Features:
 - Auto-rotating fallback: Yahoo (bulk + fast_info) -> AlphaVantage -> NSE LTP
 - AlphaVantage: GLOBAL_QUOTE, TIME_SERIES_DAILY_ADJUSTED, COMPANY_OVERVIEW (if key present)
 - Market-cap calculation separated from display
 - Market-cap sorting (largest first) for table
 - Rank by % Change (nullable Int64)
 - Historical fallback for sparklines and 90d: yfinance -> AlphaVantage
 - Sector detection improved (Yahoo fast_info -> AlphaVantage Overview -> NSE)
 - Friendly market cap display (crores) / N/A
"""

import os
import io
import math
import time
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="NIFTY 50 — Robust Dashboard", layout="wide")
st.title("📈 NIFTY 50 Live Dashboard — Robust (final fixes)")

# Put your Alpha Vantage key in env var or Streamlit secrets
ALPHAVANTAGE_API_KEY = (
    os.environ.get("ALPHAVANTAGE_API_KEY")
    or (st.secrets["ALPHAVANTAGE_API_KEY"] if "ALPHAVANTAGE_API_KEY" in st.secrets else None)
)

# Rotation order
FALLBACK_ORDER = ["yahoo", "alphavantage", "nse"]

DEBUG_IMG_PATH = "/mnt/data/62096619-831d-4050-9274-626569f125b9.png"  # optional screenshot path

NIFTY50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "BHARTIARTL.NS", "HINDUNILVR.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS",
    "SBIN.NS", "BAJFINANCE.NS", "AXISBANK.NS", "ASIANPAINT.NS", "HCLTECH.NS",
    "SUNPHARMA.NS", "MARUTI.NS", "ULTRACEMCO.NS", "POWERGRID.NS", "NTPC.NS",
    "TITAN.NS", "TECHM.NS", "ONGC.NS", "JSWSTEEL.NS", "TATASTEEL.NS",
    "NESTLEIND.NS", "WIPRO.NS", "GRASIM.NS", "COALINDIA.NS", "ADANIENT.NS",
    "ADANIPORTS.NS", "EICHERMOT.NS", "BRITANNIA.NS", "HDFCLIFE.NS", "BAJAJFINSV.NS",
    "HEROMOTOCO.NS", "DRREDDY.NS", "CIPLA.NS", "BPCL.NS", "SHREECEM.NS",
    "INDUSINDBK.NS", "SBILIFE.NS", "TATACONSUM.NS", "HINDALCO.NS", "BAJAJ-AUTO.NS",
    "DIVISLAB.NS", "UPL.NS", "APOLLOHOSP.NS", "M&M.NS", "TATAMOTORS.NS"
]

# ---------------------------
# Helpers
# ---------------------------
def safe_json(resp):
    try:
        return resp.json()
    except Exception:
        return None

def format_market_cap_display(mc):
    """mc: rupees (numeric) or NaN/None -> return friendly string or N/A"""
    try:
        if mc is None or (isinstance(mc, float) and math.isnan(mc)):
            return "N/A"
        mc = float(mc)
        crores = mc / 1e7
        if crores >= 10000:
            return f"{crores/10000:,.2f} Lakh Cr"
        if crores >= 1:
            return f"{crores:,.2f} Cr"
        return f"{mc:,.0f} ₹"
    except Exception:
        return "N/A"

# ---------------------------
# Data source helpers
# ---------------------------
def get_alpha_vantage_quote(symbol, api_key, timeout=6):
    """AlphaVantage GLOBAL_QUOTE for symbol without .NS"""
    if not api_key:
        return None, None, None, None, "alphavantage_no_key"
    base = "https://www.alphavantage.co/query"
    params = {"function": "GLOBAL_QUOTE", "symbol": symbol + ".NS", "apikey": api_key}
    try:
        r = requests.get(base, params=params, timeout=timeout)
        data = safe_json(r)
        if not data or "Global Quote" not in data or not data["Global Quote"]:
            return None, None, None, None, "alphavantage_no_data"
        g = data["Global Quote"]
        price = float(g.get("05. price")) if g.get("05. price") else None
        prev_close = float(g.get("08. previous close")) if g.get("08. previous close") else None
        pct_change = None
        if g.get("10. change percent"):
            try:
                pct_change = float(g.get("10. change percent").replace("%", ""))
            except Exception:
                pct_change = None
        return price, prev_close, pct_change, None, "alphavantage_ok"
    except Exception:
        return None, None, None, None, "alphavantage_error"

def get_alpha_vantage_overview(symbol, api_key, timeout=6):
    """AlphaVantage COMPANY_OVERVIEW for sector and market cap"""
    if not api_key:
        return None, None, "alphavantage_no_key"
    base = "https://www.alphavantage.co/query"
    params = {"function": "OVERVIEW", "symbol": symbol + ".NS", "apikey": api_key}
    try:
        r = requests.get(base, params=params, timeout=timeout)
        data = safe_json(r)
        if not data:
            return None, None, "alphavantage_no_data"
        sector = data.get("Sector")
        mcap = None
        try:
            mcap_str = data.get("MarketCapitalization")
            if mcap_str:
                mcap = float(mcap_str)
        except Exception:
            mcap = None
        return sector, mcap, "alphavantage_overview_ok"
    except Exception:
        return None, None, "alphavantage_error"

def get_nse_ltp(symbol, timeout=6):
    """NSE public quote-equity endpoint (best-effort)"""
    url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
        "Accept": "application/json, text/javascript, */*; q=0.01",
    }
    session = requests.Session()
    try:
        session.get("https://www.nseindia.com", headers=headers, timeout=timeout)
        r = session.get(url, headers=headers, timeout=timeout)
        data = safe_json(r)
        if not data:
            return None, None, None, None, "nse_no_data"
        price_info = data.get("priceInfo") or {}
        last_price = price_info.get("lastPrice") or price_info.get("last")
        prev_close = price_info.get("previousClose")
        pchange = price_info.get("pChange")
        # some NSE responses include 'sector' in securityInfo (rare)
        secinfo = data.get("securityInfo", {}) or {}
        sector_nse = secinfo.get("industry") or secinfo.get("industryType")
        return last_price, prev_close, pchange, None, "nse_ok"
    except Exception:
        return None, None, None, None, "nse_error"

def get_yahoo_bulk_prices(ticker_list, period_days=2):
    try:
        prices = yf.download(
            tickers=ticker_list,
            period=f"{period_days}d",
            interval="1d",
            group_by="ticker",
            threads=True,
            progress=False,
            auto_adjust=False
        )
        return prices, "yahoo_download_ok"
    except Exception:
        return pd.DataFrame(), "yahoo_download_error"

def get_yahoo_fast_info(ticker):
    try:
        fi = yf.Ticker(ticker).fast_info or {}
        mc = fi.get("market_cap") or fi.get("marketCap") or None
        sector = fi.get("sector") or None
        shortName = fi.get("shortName") or None
        return mc, sector, shortName, "yahoo_fast_ok"
    except Exception:
        return None, None, None, "yahoo_fast_error"

# ---------------------------
# Historical fallback: yfinance -> AlphaVantage TIME_SERIES_DAILY_ADJUSTED
# ---------------------------
def fetch_history_alpha(symbol, api_key, start_date, end_date):
    """Fetch daily adjusted series from AlphaVantage and return DataFrame with Close index"""
    if not api_key:
        return pd.DataFrame()
    base = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol + ".NS",
        "outputsize": "full",
        "apikey": api_key
    }
    try:
        r = requests.get(base, params=params, timeout=10)
        data = safe_json(r)
        if not data:
            return pd.DataFrame()
        ts = data.get("Time Series (Daily)") or {}
        if not ts:
            return pd.DataFrame()
        rows = []
        for date_str, values in ts.items():
            dt = pd.to_datetime(date_str)
            if dt.date() < start_date or dt.date() > end_date:
                continue
            close = float(values.get("5. adjusted close") or values.get("4. close"))
            rows.append({"Date": dt, "Close": close})
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).set_index("Date").sort_index()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def fetch_history_for_tickers(tickers, start_date, end_date):
    histories = {}
    for t in tickers:
        # try yfinance first
        try:
            hist = yf.download(t, start=start_date, end=end_date + timedelta(days=1), progress=False, threads=False)
            if not hist.empty:
                histories[t] = hist[["Close"]].rename(columns={"Close": "Close"})
                continue
        except Exception:
            pass
        # fallback to AlphaVantage per-ticker if API key present
        symbol = t.replace(".NS", "")
        hist_av = fetch_history_alpha(symbol, ALPHAVANTAGE_API_KEY, start_date, end_date)
        histories[t] = hist_av if not hist_av.empty else pd.DataFrame()
    return histories

# ---------------------------
# Main loader with rotating fallback, improved sector & marketcap handling
# ---------------------------
@st.cache_data(ttl=300)
def load_live_data_with_fallback(ticker_list, fallback_order=None):
    fallback_order = fallback_order or FALLBACK_ORDER
    prices_df, yahoo_status = get_yahoo_bulk_prices(ticker_list, period_days=2)
    rows = []
    for t in ticker_list:
        symbol = t.replace(".NS", "")
        price = None
        prev_close = None
        pct_change = None
        market_cap = None
        company = symbol
        sector = None
        source = "missing"

        # Yahoo bulk + fast_info
        if "yahoo" in fallback_order:
            try:
                if isinstance(prices_df, pd.DataFrame) and not prices_df.empty:
                    if t in prices_df.columns.get_level_values(0):
                        try:
                            closes = prices_df[t]["Close"].dropna()
                            if len(closes) >= 1:
                                price = float(closes.iloc[-1])
                            if len(closes) >= 2:
                                prev_close = float(closes.iloc[-2])
                        except Exception:
                            pass
                    else:
                        if "Close" in prices_df.columns:
                            cs = prices_df["Close"].dropna()
                            if len(cs) >= 1:
                                price = float(cs.iloc[-1])
                            if len(cs) >= 2:
                                prev_close = float(cs.iloc[-2])
                mc, sec, sname, status = get_yahoo_fast_info(t)
                if mc:
                    market_cap = mc
                if sec:
                    sector = sec
                if sname:
                    company = sname
                if price is not None:
                    if prev_close is not None and prev_close != 0:
                        pct_change = ((price - prev_close) / prev_close) * 100
                    source = "yahoo"
            except Exception:
                pass

        # AlphaVantage quote & overview fallback
        if (price is None or (isinstance(price, float) and math.isnan(price))) and "alphavantage" in fallback_order:
            p, pc, pchg, mc_av, status_av = get_alpha_vantage_quote(symbol, ALPHAVANTAGE_API_KEY)
            if p is not None:
                price = p
                prev_close = pc
                pct_change = pchg
                source = "alphavantage"
            # overview for sector & market cap
            sec_av, mc_overview, status_over = get_alpha_vantage_overview(symbol, ALPHAVANTAGE_API_KEY)
            if sec_av:
                sector = sec_av
            if mc_overview:
                market_cap = mc_overview

        # NSE fallback
        if (price is None or (isinstance(price, float) and math.isnan(price))) and "nse" in fallback_order:
            p_nse, prev_nse, pchg_nse, mc_nse, status_nse = get_nse_ltp(symbol)
            if p_nse is not None:
                price = p_nse
                prev_close = prev_nse
                pct_change = pchg_nse
                source = "nse"
            # sector fallback from NSE not always present

        # final attempt to populate market cap / sector / name from any source
        if market_cap is None:
            try:
                mc2, sec2, sname3, _ = get_yahoo_fast_info(t)
                if mc2:
                    market_cap = mc2
                if sec2 and sector is None:
                    sector = sec2
                if sname3 and company == symbol:
                    company = sname3
            except Exception:
                pass

        rows.append({
            "Ticker": t,
            "Company": company,
            "Sector": sector if sector else None,
            "Price": price,
            "Prev Close": prev_close,
            "% Change": pct_change,
            "Market Cap": market_cap,   # keep original for display (may be NaN)
            "Source": source
        })

    df = pd.DataFrame(rows)

    # sanitize numeric
    df["Market Cap"] = pd.to_numeric(df["Market Cap"], errors="coerce")
    df["% Change"] = pd.to_numeric(df["% Change"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Prev Close"] = pd.to_numeric(df["Prev Close"], errors="coerce")

    # Market Cap Calc (for weighting) - do not overwrite user display Market Cap
    if df["Market Cap"].notna().sum() >= 1:
        med = df.loc[df["Market Cap"].notna(), "Market Cap"].median()
        df["Market Cap Calc"] = df["Market Cap"].fillna(med)
        df["Market Cap Display Available"] = True
    else:
        # set calc to equal synthetic caps but keep display as NaN (so UI shows N/A)
        df["Market Cap Calc"] = 1.0
        df["Market Cap Display Available"] = False

    # compute % Change where possible
    mask_compute = df["% Change"].isna() & df["Price"].notna() & df["Prev Close"].notna() & (df["Prev Close"] != 0)
    df.loc[mask_compute, "% Change"] = ((df.loc[mask_compute, "Price"] - df.loc[mask_compute, "Prev Close"]) / df.loc[mask_compute, "Prev Close"]) * 100

    # compute Stake (%) using Market Cap Calc
    total_cap = df["Market Cap Calc"].sum()
    if total_cap == 0:
        df["Stake (%)"] = 0.0
    else:
        df["Stake (%)"] = (df["Market Cap Calc"] / total_cap) * 100

    df["Weighted Impact"] = (df["% Change"].fillna(0) * df["Stake (%)"]) / 100
    df["Trend"] = np.where(df["% Change"] > 0, "Gainer", "Loser")
    df["Color"] = df["% Change"].apply(lambda x: "green" if x > 0 else "red")
    df["Abs Change"] = df["% Change"].abs().fillna(0)

    return df

# ---------------------------
# UI controls
# ---------------------------
st.sidebar.header("Filters & Controls")
with st.sidebar:
    if st.checkbox("Show debug screenshot (dev)", value=False):
        try:
            st.image(DEBUG_IMG_PATH, caption="Debug screenshot", use_column_width=True)
        except Exception:
            st.info("Debug image not found: " + DEBUG_IMG_PATH)

    st.info(f"Fallback order: {', '.join(FALLBACK_ORDER)} — AlphaVantage key present: {'Yes' if ALPHAVANTAGE_API_KEY else 'No'}")
    search = st.text_input("Search company / ticker (substring)", "")
    top_n = st.slider("Show top N movers (by abs % change)", min_value=5, max_value=50, value=10, step=1)
    marketcap_min = st.number_input("Minimum market cap (crore INR) — approx", value=0.0, step=100.0)
    refresh_now = st.button("Refresh live data")

# ---------------------------
# Fetch & prepare data
# ---------------------------
with st.spinner("Fetching live NIFTY 50 data (robust fallback)..."):
    df_live = load_live_data_with_fallback(NIFTY50_TICKERS, fallback_order=FALLBACK_ORDER)

if df_live is None or df_live.empty:
    st.error("Couldn't fetch live data from any source. Check network/API keys.")
    st.stop()

# sector options (include Unknown as option)
sectors = ["All"] + sorted(df_live["Sector"].fillna("Unknown").unique().tolist())
sector_filter = st.sidebar.selectbox("Sector filter", options=sectors, index=0)

df = df_live.copy()
if search:
    mask = df["Company"].str.contains(search, case=False, na=False) | df["Ticker"].str.contains(search, case=False, na=False)
    df = df.loc[mask]
if sector_filter != "All":
    sel = None if sector_filter == "Unknown" else sector_filter
    if sel is None:
        df = df.loc[df["Sector"].isna()]
    else:
        df = df.loc[df["Sector"] == sel]
if marketcap_min and marketcap_min > 0:
    min_val = marketcap_min * 1e7
    df = df.loc[df["Market Cap Calc"] >= min_val]

# safe numeric coercion & rank
df["% Change"] = pd.to_numeric(df["% Change"], errors="coerce")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Prev Close"] = pd.to_numeric(df["Prev Close"], errors="coerce")
df["Rank"] = df["% Change"].rank(ascending=False, method="first")
df["Rank"] = df["Rank"].astype("Int64")

# Default display order: by Market Cap Calc descending (largest caps first)
df = df.sort_values(by="Market Cap Calc", ascending=False).reset_index(drop=True)

last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"Data sources: Yahoo / AlphaVantage / NSE (rotating fallback). Last updated: {last_updated}")

# ---------------------------
# Summary metrics
# ---------------------------
total_positive = df.loc[df["% Change"] > 0, "% Change"].sum()
total_negative = df.loc[df["% Change"] < 0, "% Change"].sum()
overall_impact = df["Weighted Impact"].sum()

c1, c2, c3, c4 = st.columns([1.5, 1.5, 1.5, 1])
c1.metric("Total Gainers Impact", f"{total_positive:.2f} %", delta=f"{total_positive:.2f}")
c2.metric("Total Losers Impact", f"{total_negative:.2f} %", delta=f"{total_negative:.2f}")
c3.metric("Overall Weighted Impact", f"{overall_impact:.2f} %", delta=f"{overall_impact:.2f}")
c4.metric("Companies Shown", f"{len(df)}", delta=f"{len(df)}")

st.write("---")

# ---------------------------
# Top gainers / losers — formatted output
# ---------------------------
top_gainers = df.sort_values(by="% Change", ascending=False).head(5)
top_losers = df.sort_values(by="% Change", ascending=True).head(5)

gcol1, gcol2 = st.columns(2)
with gcol1:
    st.subheader("Top 5 Gainers")
    gain_lines = []
    for _, r in top_gainers.iterrows():
        pct = r["% Change"] if pd.notna(r["% Change"]) else 0.0
        mc_display = format_market_cap_display(r.get("Market Cap"))
        gain_lines.append(f"**{r['Company']}** ({r['Ticker']}) — `{pct:.2f}%` — Source: {r.get('Source','-')} — MarketCap: **{mc_display}**")
    st.markdown("<br>".join(gain_lines), unsafe_allow_html=True)

with gcol2:
    st.subheader("Top 5 Losers")
    loser_lines = []
    for _, r in top_losers.iterrows():
        pct = r["% Change"] if pd.notna(r["% Change"]) else 0.0
        mc_display = format_market_cap_display(r.get("Market Cap"))
        loser_lines.append(f"**{r['Company']}** ({r['Ticker']}) — `{pct:.2f}%` — Source: {r.get('Source','-')} — MarketCap: **{mc_display}**")
    st.markdown("<br>".join(loser_lines), unsafe_allow_html=True)

st.write("---")

# ---------------------------
# Table: show Market Cap Display (human-friendly), sort by Market Cap Calc desc
# ---------------------------
table_df = df[["Rank", "Ticker", "Company", "Sector", "Stake (%)", "Price", "% Change", "Weighted Impact", "Market Cap", "Market Cap Calc", "Source"]].copy()
table_df["% Change"] = table_df["% Change"].round(2)
table_df["Stake (%)"] = table_df["Stake (%)"].round(2)
table_df["Weighted Impact"] = table_df["Weighted Impact"].round(4)
table_df["Market Cap Display"] = table_df["Market Cap"].apply(format_market_cap_display)

# hide internal calc in UI but keep exposed for CSV if user wants
display_table = table_df.drop(columns=["Market Cap Calc"]).rename(columns={"Market Cap Display": "Market Cap"})
# Download CSV (filtered)
csv_buf = io.StringIO()
display_table.to_csv(csv_buf, index=False)
st.download_button("Download filtered table as CSV", data=csv_buf.getvalue().encode(), file_name="nifty50_filtered.csv", mime="text/csv")

st.subheader("🏦 Company Performance (sorted by Market Cap)")
st.dataframe(display_table, use_container_width=True)

# ---------------------------
# Mini sparklines (use fetch_history_for_tickers fallback)
# ---------------------------
def make_sparkline_from_hist(hist_df):
    if hist_df is None or hist_df.empty:
        return None
    fig = go.Figure(go.Scatter(x=hist_df.index, y=hist_df["Close"], mode="lines", line=dict(width=1)))
    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=60, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig

with st.expander("Show mini historical charts for displayed companies (last 30 days)"):
    display_tickers = display_table["Ticker"].tolist()[:12]
    end = datetime.now().date()
    start = end - timedelta(days=30)
    histories = fetch_history_for_tickers(display_tickers, start, end)
    cols = st.columns(3)
    for i, t in enumerate(display_tickers):
        row = display_table.loc[display_table["Ticker"] == t].iloc[0]
        col = cols[i % 3]
        with col:
            st.write(f"**{row['Company']}** — {t} (Source: {row['Source']})")
            hist = histories.get(t)
            fig_sp = make_sparkline_from_hist(hist)
            if fig_sp:
                st.plotly_chart(fig_sp, use_container_width=True)
            else:
                st.info("Mini history not available")

st.write("---")

# ---------------------------
# Charts: % change bar (unchanged), sector impact, stake pie, gainers vs losers
# ---------------------------
fig1 = px.bar(df.sort_values(by="% Change", ascending=False), x="Company", y="% Change", color="Trend",
              text="Rank", title="% Change by Company")
fig1.update_traces(textposition="outside")
st.plotly_chart(fig1, use_container_width=True)

# Sector weighted impact: show meaningful chart only if >1 sector exists
sector_impact = df.groupby(df["Sector"].fillna("Unknown"))["Weighted Impact"].sum().reset_index().rename(columns={"Sector":"Sector", "Weighted Impact":"Weighted Impact"})
if sector_impact.shape[0] == 0:
    st.info("No sector data available to plot weighted sector impact.")
else:
    if sector_impact.shape[0] == 1:
        # If only Unknown sector present, show a small notice + bar with clearer color
        st.warning("Most tickers are missing sector metadata; sector breakdown unavailable.")
        fig2 = px.bar(sector_impact, x="Sector", y="Weighted Impact", title="Weighted Sector Impact (single category)", color="Weighted Impact", color_continuous_scale=px.colors.sequential.Blues)
    else:
        fig2 = px.bar(sector_impact.sort_values(by="Weighted Impact"), x="Sector", y="Weighted Impact",
                      color="Weighted Impact", title="Weighted Sector Impact", color_continuous_scale=px.colors.diverging.RdYlGn)
    st.plotly_chart(fig2, use_container_width=True)

fig3 = px.pie(df, values="Stake (%)", names="Company", title="Company Stake Distribution in NIFTY 50")
st.plotly_chart(fig3, use_container_width=True)

trend_count = df["Trend"].value_counts().reset_index()
trend_count.columns = ["Trend", "Count"]
fig4 = px.pie(trend_count, names="Trend", values="Count", title="Gainers vs Losers Ratio", color="Trend",
              color_discrete_map={"Gainer": "green", "Loser": "red"})
st.plotly_chart(fig4, use_container_width=True)

# Sector heatmap
with st.expander("Sector vs Weighted Impact heatmap"):
    pivot = sector_impact.copy()
    if pivot.shape[0] <= 1:
        st.info("Not enough sector variety to build a heatmap.")
    else:
        heat = px.imshow(pivot[["Weighted Impact"]].T, labels=dict(x="Sector", y="Metric", color="Weighted Impact"),
                         x=pivot["Sector"].tolist(), y=["Weighted Impact"], aspect="auto", title="Sector Heatmap")
        st.plotly_chart(heat, use_container_width=True)

st.write("---")

# ---------------------------
# Historical comparison panel (use alpha fallback for history)
# ---------------------------
st.subheader("Historical Comparison — Multi Company")
default_end = datetime.now().date()
default_start = default_end - timedelta(days=365)
col_a, col_b, col_c = st.columns([2,2,1])
with col_a:
    selected_companies = st.multiselect("Select companies (tickers) for historical comparison", options=df["Ticker"].tolist(), default=df["Ticker"].tolist()[:3])
with col_b:
    start_date = st.date_input("Start date", value=default_start)
    end_date = st.date_input("End date", value=default_end)
with col_c:
    normalize = st.checkbox("Normalize (index to 100)", value=True)

if selected_companies:
    histories = fetch_history_for_tickers(selected_companies, start_date, end_date)
    fig_hist = go.Figure()
    any_data = False
    for t in selected_companies:
        hist = histories.get(t)
        if hist is None or hist.empty:
            continue
        any_data = True
        y = hist["Close"]
        if normalize:
            y = (y / y.iloc[0]) * 100
        fig_hist.add_trace(go.Scatter(x=hist.index, y=y, name=t))
    fig_hist.update_layout(title="Historical Close (Normalized)" if normalize else "Historical Close", xaxis_title="Date", yaxis_title="Indexed / Price")
    if any_data:
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("No historical data found for the selected tickers & date range.")

st.write("---")

# ---------------------------
# Company detail panel
# ---------------------------
st.subheader("Company Detail")
ticker_choice = st.selectbox("Choose a single ticker to inspect", options=df["Ticker"].tolist())
if ticker_choice:
    row = df.loc[df["Ticker"] == ticker_choice].iloc[0]
    mc_display = format_market_cap_display(row["Market Cap"])
    st.write(f"**{row['Company']}** ({row['Ticker']}) — Sector: {row['Sector'] if pd.notna(row['Sector']) else 'Unknown'}")
    st.write(f"Price: {row['Price'] if pd.notna(row['Price']) else 'N/A'}, Prev Close: {row['Prev Close'] if pd.notna(row['Prev Close']) else 'N/A'}, % Change: {row['% Change'] if pd.notna(row['% Change']) else 0:.2f}%")
    st.write(f"Market Cap: {mc_display}, Stake: {row['Stake (%)']:.2f}%, Data source: {row['Source']}")
    # attempt 90-day history
    hist90 = fetch_history_for_tickers([ticker_choice], datetime.now().date()-timedelta(days=90), datetime.now().date()).get(ticker_choice)
    if hist90 is not None and not hist90.empty:
        fig_o = px.line(hist90, y="Close", title=f"{row['Ticker']} — Last 90 days Close")
        st.plotly_chart(fig_o, use_container_width=True)
        st.download_button("Download company historical (90d) CSV", data=hist90.to_csv().encode(), file_name=f"{ticker_choice}_90d.csv")
    else:
        st.info("90-day history not available for this ticker (tried yfinance + AlphaVantage).")

st.write("---")
st.caption("Fixes applied: market-cap display, sorting by market-cap, improved sector detection, history fallbacks (yfinance->AlphaVantage), nullable rank, and cleaner top movers.")

# ---------------------------
# Notes & next steps
# ---------------------------
st.markdown(
    """
    **Notes**
    - Alpha Vantage improves sector & historical retrieval but is rate-limited (free: 5 calls/min). Keep ALPHAVANTAGE_API_KEY set for best results.
    - NSE public endpoints can be flaky; if you see persistent 'nse' as source with missing metadata, prefer using AlphaVantage or a paid feed.
    - If you'd like, I can:
        - Add AlphaVantage rate-limit queuing (token bucket) so fallbacks do not hit limits.
        - Implement a Redis cache + background worker to refresh historical & overview data.
        - Add a mapping CSV for sectors (manual override) so you always get correct sector assignments.
    """
)
