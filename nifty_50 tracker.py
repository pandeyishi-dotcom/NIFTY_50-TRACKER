# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import time

st.set_page_config(page_title="NIFTY 50 Live Dashboard", layout="wide")
st.title("📈 NIFTY 50 Live Performance Dashboard (Fixed & Robust)")

# Optional: display the screenshot you uploaded while debugging (replace path if different)
# Developer-provided image path:
DEBUG_IMG_PATH = "/mnt/data/8e8d363f-d093-44cb-9a70-bc21849fe486.png"

# -------------------------
# NIFTY 50 tickers
# -------------------------
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

# -------------------------
# Robust data loader
# - use yf.download in bulk for price history (fast)
# - use Ticker(...).fast_info for lightweight meta data (market cap)
# - avoid .info completely
# -------------------------
@st.cache_data(ttl=300)
def load_live_data(ticker_list):
    # We'll fetch 2 days of close data, so we can compute today's vs previous close.
    # Bulk call reduces requests and avoids rate-limits.
    try:
        prices = yf.download(
            tickers=ticker_list,
            period="2d",
            interval="1d",
            group_by="ticker",
            threads=True,
            progress=False,
            auto_adjust=False
        )
    except Exception:
        prices = pd.DataFrame()

    rows = []
    for t in ticker_list:
        try:
            # Extract closes robustly depending on yf.download return format
            today_close = None
            prev_close = None
            if isinstance(prices, pd.DataFrame) and not prices.empty:
                # Case: yf.download with multiple tickers returns a multiindex-tracked DataFrame
                # If group_by="ticker", we can index with [t]
                if t in prices.columns.get_level_values(0):
                    # columns like (Ticker, 'Close')
                    try:
                        close_series = prices[t]["Close"].dropna()
                        if len(close_series) >= 1:
                            today_close = close_series.iloc[-1]
                        if len(close_series) >= 2:
                            prev_close = close_series.iloc[-2]
                    except Exception:
                        pass
                else:
                    # alternative layout: single-level columns when one ticker or edge cases
                    # try to find any "Close" column
                    if "Close" in prices.columns:
                        cs = prices["Close"].dropna()
                        if len(cs) >= 1:
                            today_close = cs.iloc[-1]
                        if len(cs) >= 2:
                            prev_close = cs.iloc[-2]

            # Fetch lightweight meta info (fast_info)
            try:
                fi = yf.Ticker(t).fast_info or {}
            except Exception:
                fi = {}

            # company short name fallback
            company = fi.get("shortName") or t.replace(".NS", "")
            # different versions: some fast_info use 'market_cap', others use 'market_cap' key
            market_cap = fi.get("market_cap") or fi.get("marketCap") or None
            sector = fi.get("sector") or "Unknown"

            pct_change = None
            if today_close is not None and prev_close is not None and prev_close != 0:
                pct_change = ((today_close - prev_close) / prev_close) * 100

            rows.append({
                "Ticker": t,
                "Company": company,
                "Sector": sector,
                "Price": today_close,
                "Prev Close": prev_close,
                "% Change": pct_change,
                "Market Cap": market_cap
            })
        except Exception:
            # Fail-safe: include a row with None values (so UI can show missing data gracefully)
            rows.append({
                "Ticker": t,
                "Company": t.replace(".NS", ""),
                "Sector": "Unknown",
                "Price": None,
                "Prev Close": None,
                "% Change": None,
                "Market Cap": None
            })

    df = pd.DataFrame(rows)

    # Filter out entries missing crucial fields (we keep as much as possible)
    df = df.dropna(subset=["% Change", "Market Cap"], how="any")
    if df.empty:
        return df

    # Some market caps might be strings or very large; ensure numeric
    df["Market Cap"] = pd.to_numeric(df["Market Cap"], errors="coerce")
    df = df.dropna(subset=["Market Cap", "% Change"])

    total_cap = df["Market Cap"].sum()
    df["Stake (%)"] = (df["Market Cap"] / total_cap) * 100
    df["Weighted Impact"] = (df["% Change"] * df["Stake (%)"]) / 100
    df["Trend"] = np.where(df["% Change"] > 0, "Gainer", "Loser")
    df["Color"] = df["% Change"].apply(lambda x: "green" if x > 0 else "red")
    df["Abs Change"] = df["% Change"].abs()

    return df

# -------------------------
# Historical fetcher for charts
# -------------------------
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

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Filters & Controls")
with st.sidebar:
    # Debug image toggle
    if st.checkbox("Show debug screenshot (dev)", value=False):
        try:
            st.image(DEBUG_IMG_PATH, caption="Debug screenshot", use_column_width=True)
        except Exception:
            st.info("Debug image not found at: " + DEBUG_IMG_PATH)

    search = st.text_input("Search company / ticker (substring)", "")
    top_n = st.slider("Show top N movers by absolute % change", min_value=5, max_value=50, value=10, step=1)
    marketcap_min = st.number_input("Minimum market cap (crore INR) — approx", value=0.0, step=100.0)
    refresh_now = st.button("Refresh live data")

# -------------------------
# Load data
# -------------------------
with st.spinner("Fetching live NIFTY 50 data (robust mode)..."):
    df_live = load_live_data(NIFTY50_TICKERS)

if df_live is None or df_live.empty:
    st.error("No live data available right now. Yahoo Finance may be rate-limiting or data missing.")
    st.stop()

# populate sector filter
sectors = ["All"] + sorted(df_live["Sector"].fillna("Unknown").unique().tolist())
sector_filter = st.sidebar.selectbox("Sector filter", options=sectors, index=0)

# apply search and sector & marketcap filters
df = df_live.copy()
if search:
    mask = df["Company"].str.contains(search, case=False, na=False) | df["Ticker"].str.contains(search, case=False, na=False)
    df = df.loc[mask]
if sector_filter != "All":
    df = df.loc[df["Sector"] == sector_filter]
if marketcap_min and marketcap_min > 0:
    min_val = marketcap_min * 1e7  # crore -> rupees (approx)
    df = df.loc[df["Market Cap"] >= min_val]

# Ranking / sorting
df = df.sort_values(by="% Change", ascending=False).reset_index(drop=True)
df["Rank"] = df["% Change"].rank(ascending=False, method="first").astype(int)

# last updated info
last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"Data source: Yahoo Finance (`yfinance` bulk + fast_info). Last updated: {last_updated} (cache TTL 5min)")

# Summary metrics
total_positive = df.loc[df["% Change"] > 0, "% Change"].sum()
total_negative = df.loc[df["% Change"] < 0, "% Change"].sum()
overall_impact = df["Weighted Impact"].sum()

col1, col2, col3, col4 = st.columns([1.5,1.5,1.5,1])
col1.metric("Total Gainers Impact", f"{total_positive:.2f} %", delta=f"{total_positive:.2f}")
col2.metric("Total Losers Impact", f"{total_negative:.2f} %", delta=f"{total_negative:.2f}")
col3.metric("Overall Weighted Impact", f"{overall_impact:.2f} %", delta=f"{overall_impact:.2f}")
col4.metric("Companies Shown", f"{len(df)}", delta=f"{len(df)}")

st.write("---")

# Top gainers / losers cards
top_gainers = df.sort_values(by="% Change", ascending=False).head(5)
top_losers = df.sort_values(by="% Change", ascending=True).head(5)

gcol1, gcol2 = st.columns(2)
with gcol1:
    st.subheader("Top 5 Gainers")
    for _, r in top_gainers.iterrows():
        st.write(f"**{r['Company']}** ({r['Ticker']}) — {r['% Change']:.2f}%  — MarketCap: {int(r['Market Cap']):,}")
with gcol2:
    st.subheader("Top 5 Losers")
    for _, r in top_losers.iterrows():
        st.write(f"**{r['Company']}** ({r['Ticker']}) — {r['% Change']:.2f}%  — MarketCap: {int(r['Market Cap']):,}")

st.write("---")

# Prepare table for display
table_df = df[["Rank", "Ticker", "Company", "Sector", "Stake (%)", "Price", "% Change", "Weighted Impact", "Market Cap", "Abs Change"]].copy()
table_df["% Change"] = table_df["% Change"].round(2)
table_df["Stake (%)"] = table_df["Stake (%)"].round(2)
table_df["Weighted Impact"] = table_df["Weighted Impact"].round(4)

# Limit to top_n movers by abs change
table_display = table_df.sort_values(by="Abs Change", ascending=False).head(top_n)

# Download button for filtered table
csv_buf = io.StringIO()
table_display.to_csv(csv_buf, index=False)
csv_bytes = csv_buf.getvalue().encode()
st.download_button("Download filtered table as CSV", data=csv_bytes, file_name="nifty50_filtered.csv", mime="text/csv")

st.subheader("🏦 Company Performance (filtered)")
st.dataframe(table_display.drop(columns=["Abs Change"]), use_container_width=True)

# Mini sparklines
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
            st.write(f"**{row['Company']}** — {row['Ticker']}")
            fig_sp = make_sparkline(row["Ticker"])
            if fig_sp:
                st.plotly_chart(fig_sp, use_container_width=True)
            else:
                st.info("Mini history not available")

st.write("---")

# Charts: % change by company, sector weighted impact, stake pie, gainers vs losers
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

# Sector heatmap
with st.expander("Sector vs Weighted Impact heatmap"):
    pivot = df.pivot_table(index="Sector", values="Weighted Impact", aggfunc="sum").reset_index()
    if not pivot.empty:
        heat = px.imshow(pivot[["Weighted Impact"]].T, labels=dict(x="Sector", y="Metric", color="Weighted Impact"),
                         x=pivot["Sector"].tolist(), y=["Weighted Impact"], aspect="auto", title="Sector Heatmap")
        st.plotly_chart(heat, use_container_width=True)
    else:
        st.info("Not enough data to build heatmap for selected filters.")

st.write("---")

# Historical comparison panel
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

# Company detail panel
st.subheader("Company Detail")
ticker_choice = st.selectbox("Choose a single ticker to inspect", options=df["Ticker"].tolist())
if ticker_choice:
    row = df.loc[df["Ticker"] == ticker_choice].iloc[0]
    st.write(f"**{row['Company']}** ({row['Ticker']}) — Sector: {row['Sector']}")
    st.write(f"Price: {row['Price']}, Prev Close: {row['Prev Close']}, % Change: {row['% Change']:.2f}%")
    st.write(f"Market Cap: {int(row['Market Cap']):,}, Stake: {row['Stake (%)']:.2f}%")
    hist90 = fetch_history_for_tickers([ticker_choice], datetime.now().date()-timedelta(days=90), datetime.now().date()).get(ticker_choice)
    if hist90 is not None and not hist90.empty:
        fig_o = px.line(hist90, y="Close", title=f"{row['Ticker']} — Last 90 days Close")
        st.plotly_chart(fig_o, use_container_width=True)
        st.download_button("Download company historical (90d) CSV", data=hist90.to_csv().encode(), file_name=f"{ticker_choice}_90d.csv")
    else:
        st.info("90-day history not available.")

st.write("---")
st.caption("Fixed loader: uses bulk yf.download for price data and fast_info for market cap. Avoids .info and Yahoo rate-limits. Cache TTLs reduce redundant calls.")

# Pro tips
with st.container():
    st.markdown(
        """
        **Pro tips**
        - If you still see missing rows, try waiting a few minutes; yfinance and Yahoo can hit transient rate limits.
        - For production use, consider a paid data provider or an official NSE feed to avoid third-party limits.
        - If you want, I can add: intraday streaming, Redis cache, or a fallback API (NSE / AlphaVantage / TwelveData).
        """
    )
