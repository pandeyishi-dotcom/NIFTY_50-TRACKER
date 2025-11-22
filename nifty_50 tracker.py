# app.py
"""
NIFTY 50 Live Dashboard — Production-grade fallback
Sources: Yahoo (yfinance bulk + fast_info) -> AlphaVantage (GLOBAL_QUOTE) -> NSE LTP
Features:
 - Rotating fallback per-ticker
 - Cache-safe loaders
 - NaN-safe numeric coercion
 - Nullable integer Rank (Int64) to avoid IntCastingNaNError
 - Filters, downloads, sparklines, historical comparison, sector heatmap, top movers
Instructions:
 - Set ALPHAVANTAGE_API_KEY via environment variable or Streamlit secrets:
    export ALPHAVANTAGE_API_KEY="your_key_here"
   or in .streamlit/secrets.toml: ALPHAVANTAGE_API_KEY = "your_key_here"
 - Install requirements from requirements.txt included earlier.
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

# ---------------------------
# Config & constants
# ---------------------------
st.set_page_config(page_title="NIFTY 50 — Robust Dashboard", layout="wide")
st.title("📈 NIFTY 50 Live Dashboard — Yahoo / AlphaVantage / NSE Fallback (Production-grade)")

# Provide your Alpha Vantage key via env var or Streamlit secrets
ALPHAVANTAGE_API_KEY = (
    os.environ.get("ALPHAVANTAGE_API_KEY")
    or (st.secrets["ALPHAVANTAGE_API_KEY"] if "ALPHAVANTAGE_API_KEY" in st.secrets else None)
)

# Rotation order: you can reorder if needed
FALLBACK_ORDER = ["yahoo", "alphavantage", "nse"]

# Debug image path (if you uploaded a screenshot)
DEBUG_IMG_PATH = "/mnt/data/8e8d363f-d093-44cb-9a70-bc21849fe486.png"

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
# Utilities
# ---------------------------
def safe_json(resp):
    try:
        return resp.json()
    except Exception:
        return None

# Alpha Vantage GLOBAL_QUOTE (fallback)
def get_alpha_vantage_quote(symbol, api_key, timeout=6):
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

# NSE LTP (best-effort public endpoint)
def get_nse_ltp(symbol, timeout=6):
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
        return last_price, prev_close, pchange, None, "nse_ok"
    except Exception:
        return None, None, None, None, "nse_error"

# Yahoo bulk price fetch
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
# Main loader with rotating fallback
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
        sector = "Unknown"
        source = "missing"

        # ---------- Yahoo (bulk + fast_info) ----------
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
                # yahoo fast_info
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

        # ---------- Alpha Vantage fallback ----------
        if (price is None or (isinstance(price, float) and math.isnan(price))) and "alphavantage" in fallback_order:
            p, pc, pchg, mc_av, status_av = get_alpha_vantage_quote(symbol, ALPHAVANTAGE_API_KEY)
            if p is not None:
                price = p
                prev_close = pc
                pct_change = pchg
                if mc_av:
                    market_cap = mc_av
                # try to get company name
                try:
                    _, _, sname2, _ = get_yahoo_fast_info(t)
                    if sname2:
                        company = sname2
                except Exception:
                    pass
                source = "alphavantage"

        # ---------- NSE fallback ----------
        if (price is None or (isinstance(price, float) and math.isnan(price))) and "nse" in fallback_order:
            p_nse, prev_nse, pchg_nse, mc_nse, status_nse = get_nse_ltp(symbol)
            if p_nse is not None:
                price = p_nse
                prev_close = prev_nse
                pct_change = pchg_nse
                if mc_nse:
                    market_cap = mc_nse
                # attempt company name via yahoo fast_info
                try:
                    _, _, sname2, _ = get_yahoo_fast_info(t)
                    if sname2:
                        company = sname2
                except Exception:
                    pass
                source = "nse"

        # final attempt to populate market cap / sector / company from yahoo fast_info
        if market_cap is None:
            try:
                mc2, sec2, sname3, _ = get_yahoo_fast_info(t)
                if mc2:
                    market_cap = mc2
                if sec2 and sector == "Unknown":
                    sector = sec2
                if sname3 and company == symbol:
                    company = sname3
            except Exception:
                pass

        rows.append({
            "Ticker": t,
            "Company": company,
            "Sector": sector if sector else "Unknown",
            "Price": price,
            "Prev Close": prev_close,
            "% Change": pct_change,
            "Market Cap": market_cap,
            "Source": source
        })

    df = pd.DataFrame(rows)

    # ---------------------------
    # Sanitize numeric columns & fill small gaps
    # ---------------------------
    df["Market Cap"] = pd.to_numeric(df["Market Cap"], errors="coerce")
    df["% Change"] = pd.to_numeric(df["% Change"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Prev Close"] = pd.to_numeric(df["Prev Close"], errors="coerce")

    # If there are some rows missing market cap, fill using median of available caps
    if df["Market Cap"].notna().sum() >= 1:
        med = df.loc[df["Market Cap"].notna(), "Market Cap"].median()
        df["Market Cap"] = df["Market Cap"].fillna(med)
    else:
        # no market cap data at all: use equal synthetic cap so stake becomes equal
        df["Market Cap"] = 1.0

    # Compute % Change where possible (Price and Prev Close available but % Change missing)
    mask_compute = df["% Change"].isna() & df["Price"].notna() & df["Prev Close"].notna() & (df["Prev Close"] != 0)
    df.loc[mask_compute, "% Change"] = ((df.loc[mask_compute, "Price"] - df.loc[mask_compute, "Prev Close"]) / df.loc[mask_compute, "Prev Close"]) * 100

    # Compute stake and weighted impact (guard divide by zero)
    total_cap = df["Market Cap"].sum()
    if total_cap == 0:
        df["Stake (%)"] = 0.0
    else:
        df["Stake (%)"] = (df["Market Cap"] / total_cap) * 100

    df["Weighted Impact"] = (df["% Change"].fillna(0) * df["Stake (%)"]) / 100
    df["Trend"] = np.where(df["% Change"] > 0, "Gainer", "Loser")
    df["Color"] = df["% Change"].apply(lambda x: "green" if x > 0 else "red")
    df["Abs Change"] = df["% Change"].abs().fillna(0)

    return df

# ---------------------------
# Historical fetcher (sparklines & comparison)
# ---------------------------
@st.cache_data(ttl=600)
def fetch_history_for_tickers(tickers, start_date, end_date):
    histories = {}
    for t in tickers:
        try:
            hist = yf.download(t, start=start_date, end=end_date + timedelta(days=1), progress=False, threads=False)
            if not hist.empty:
                histories[t] = hist[["Close"]].rename(columns={"Close": "Close"})
            else:
                histories[t] = pd.DataFrame()
        except Exception:
            histories[t] = pd.DataFrame()
    return histories

# ---------------------------
# UI: Sidebar controls
# ---------------------------
st.sidebar.header("Filters & Controls")
with st.sidebar:
    if st.checkbox("Show debug screenshot (dev)", value=False):
        try:
            st.image(DEBUG_IMG_PATH, caption="Debug screenshot", use_column_width=True)
        except Exception:
            st.info("Debug image not found at: " + DEBUG_IMG_PATH)

    st.info(f"Data fallback order: {', '.join(FALLBACK_ORDER)}. AlphaVantage key present: {'Yes' if ALPHAVANTAGE_API_KEY else 'No'}")
    search = st.text_input("Search company / ticker (substring)", "")
    top_n = st.slider("Show top N movers by absolute % change", min_value=5, max_value=50, value=10, step=1)
    marketcap_min = st.number_input("Minimum market cap (crore INR) — approx", value=0.0, step=100.0)
    refresh_now = st.button("Refresh live data")

# ---------------------------
# Load data
# ---------------------------
with st.spinner("Fetching live NIFTY 50 data using fallback system..."):
    df_live = load_live_data_with_fallback(NIFTY50_TICKERS, fallback_order=FALLBACK_ORDER)

if df_live is None or df_live.empty:
    st.error("No live data could be fetched from any source. Please check your network or APIs.")
    st.stop()

# ---------------------------
# Apply filters
# ---------------------------
sectors = ["All"] + sorted(df_live["Sector"].fillna("Unknown").unique().tolist())
sector_filter = st.sidebar.selectbox("Sector filter", options=sectors, index=0)

df = df_live.copy()
if search:
    mask = df["Company"].str.contains(search, case=False, na=False) | df["Ticker"].str.contains(search, case=False, na=False)
    df = df.loc[mask]
if sector_filter != "All":
    df = df.loc[df["Sector"] == sector_filter]
if marketcap_min and marketcap_min > 0:
    min_val = marketcap_min * 1e7  # crore -> rupees
    df = df.loc[df["Market Cap"] >= min_val]

# ---------------------------
# SAFE RANK + numeric coercion (drop-in)
# ---------------------------
# ensure numeric types
df["% Change"] = pd.to_numeric(df["% Change"], errors="coerce")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Prev Close"] = pd.to_numeric(df["Prev Close"], errors="coerce")

# compute rank (float) then convert to pandas nullable integer Int64
df["Rank"] = df["% Change"].rank(ascending=False, method="first")
df["Rank"] = df["Rank"].astype("Int64")

# ---------------------------
# Summary metrics & UI
# ---------------------------
df = df.sort_values(by="% Change", ascending=False).reset_index(drop=True)
last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

total_positive = df.loc[df["% Change"] > 0, "% Change"].sum()
total_negative = df.loc[df["% Change"] < 0, "% Change"].sum()
overall_impact = df["Weighted Impact"].sum()

c1, c2, c3, c4 = st.columns([1.5, 1.5, 1.5, 1])
c1.metric("Total Gainers Impact", f"{total_positive:.2f} %", delta=f"{total_positive:.2f}")
c2.metric("Total Losers Impact", f"{total_negative:.2f} %", delta=f"{total_negative:.2f}")
c3.metric("Overall Weighted Impact", f"{overall_impact:.2f} %", delta=f"{overall_impact:.2f}")
c4.metric("Companies Shown", f"{len(df)}", delta=f"{len(df)}")

st.caption(f"Data sources: Yahoo / AlphaVantage / NSE (auto-rotating fallback). Last updated: {last_updated}")

st.write("---")

# ---------------------------
# Top movers and table
# ---------------------------
top_gainers = df.sort_values(by="% Change", ascending=False).head(5)
top_losers = df.sort_values(by="% Change", ascending=True).head(5)

g1, g2 = st.columns(2)
with g1:
    st.subheader("Top 5 Gainers")
    for _, r in top_gainers.iterrows():
        st.write(f"**{r['Company']}** ({r['Ticker']}) — {r['% Change']:.2f}% — Source: {r['Source']} — MarketCap: {int(r['Market Cap']):,}")
with g2:
    st.subheader("Top 5 Losers")
    for _, r in top_losers.iterrows():
        st.write(f"**{r['Company']}** ({r['Ticker']}) — {r['% Change']:.2f}% — Source: {r['Source']} — MarketCap: {int(r['Market Cap']):,}")

st.write("---")

table_df = df[["Rank", "Ticker", "Company", "Sector", "Stake (%)", "Price", "% Change", "Weighted Impact", "Market Cap", "Source", "Abs Change"]].copy()
table_df["% Change"] = table_df["% Change"].round(2)
table_df["Stake (%)"] = table_df["Stake (%)"].round(2)
table_df["Weighted Impact"] = table_df["Weighted Impact"].round(4)
table_display = table_df.sort_values(by="Abs Change", ascending=False).head(top_n)

# Download CSV for filtered
csv_buf = io.StringIO()
table_display.to_csv(csv_buf, index=False)
st.download_button("Download filtered table as CSV", data=csv_buf.getvalue().encode(), file_name="nifty50_filtered.csv", mime="text/csv")

st.subheader("🏦 Company Performance (filtered)")
st.dataframe(table_display.drop(columns=["Abs Change"]), use_container_width=True)

# ---------------------------
# Sparklines
# ---------------------------
def make_sparkline(ticker):
    end = datetime.now().date()
    start = end - timedelta(days=30)
    try:
        hist = yf.download(ticker, start=start, end=end + timedelta(days=1), progress=False)
        if hist.empty:
            return None
        fig = go.Figure(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', line=dict(width=1)))
        fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=60, xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig
    except Exception:
        return None

with st.expander("Show mini historical charts for displayed companies (last 30 days)"):
    cols = st.columns(3)
    for i, (_, row) in enumerate(table_display.head(12).iterrows()):
        col = cols[i % 3]
        with col:
            st.write(f"**{row['Company']}** — {row['Ticker']} (Source: {row['Source']})")
            fig_sp = make_sparkline(row["Ticker"])
            if fig_sp:
                st.plotly_chart(fig_sp, use_container_width=True)
            else:
                st.info("Mini history not available")

st.write("---")

# ---------------------------
# Charts & heatmap
# ---------------------------
fig1 = px.bar(df.sort_values(by="% Change", ascending=False), x="Company", y="% Change", color="Trend",
              text="Rank", title="% Change by Company")
fig1.update_traces(textposition="outside")
st.plotly_chart(fig1, use_container_width=True)

sector_impact = df.groupby("Sector")["Weighted Impact"].sum().reset_index()
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

with st.expander("Sector vs Weighted Impact heatmap"):
    pivot = df.pivot_table(index="Sector", values="Weighted Impact", aggfunc="sum").reset_index()
    if not pivot.empty:
        heat = px.imshow(pivot[["Weighted Impact"]].T, labels=dict(x="Sector", y="Metric", color="Weighted Impact"),
                         x=pivot["Sector"].tolist(), y=["Weighted Impact"], aspect="auto", title="Sector Heatmap")
        st.plotly_chart(heat, use_container_width=True)
    else:
        st.info("Not enough data to build heatmap for selected filters.")

st.write("---")

# ---------------------------
# Historical comparison panel
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
    for t in selected_companies:
        hist = histories.get(t)
        if hist is None or hist.empty:
            continue
        y = hist["Close"]
        if normalize:
            y = (y / y.iloc[0]) * 100
        fig_hist.add_trace(go.Scatter(x=hist.index, y=y, name=t))
    fig_hist.update_layout(title="Historical Close (Normalized)" if normalize else "Historical Close", xaxis_title="Date", yaxis_title="Indexed / Price")
    if fig_hist.data:
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
    st.write(f"**{row['Company']}** ({row['Ticker']}) — Sector: {row['Sector']}")
    st.write(f"Price: {row['Price']}, Prev Close: {row['Prev Close']}, % Change: {row['% Change']:.2f}%")
    st.write(f"Market Cap: {int(row['Market Cap']):,}, Stake: {row['Stake (%)']:.2f}%, Data source: {row['Source']}")
    hist90 = fetch_history_for_tickers([ticker_choice], datetime.now().date()-timedelta(days=90), datetime.now().date()).get(ticker_choice)
    if hist90 is not None and not hist90.empty:
        fig_o = px.line(hist90, y="Close", title=f"{row['Ticker']} — Last 90 days Close")
        st.plotly_chart(fig_o, use_container_width=True)
        st.download_button("Download company historical (90d) CSV", data=hist90.to_csv().encode(), file_name=f"{ticker_choice}_90d.csv")
    else:
        st.info("90-day history not available.")

st.write("---")
st.caption("Production-grade fallback: Yahoo (bulk + fast_info) → Alpha Vantage → NSE LTP. Cache TTLs help reduce rate-limit issues. For enterprise reliability, use a paid market-data feed.")

# ---------------------------
# Notes
# ---------------------------
st.markdown(
    """
    **Notes & next steps**
    - Alpha Vantage free tier has strict rate limits (5 calls/min). This app uses Alpha Vantage only as a per-ticker fallback.
    - NSE public endpoints may require cookie/headers and can be flaky for heavy use.
    - For a production environment consider:
        - Redis + background refresh worker
        - Rotating API keys or a paid multi-provider strategy
        - An official licensed feed for NSE
    - Want me to add rate-limit queuing for Alpha Vantage (token bucket) or a Redis refresh worker? I can implement either next.
    """
)
