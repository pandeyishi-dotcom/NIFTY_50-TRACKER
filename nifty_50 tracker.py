import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io

st.set_page_config(page_title="NIFTY 50 Live Dashboard", layout="wide")
st.title("📈 NIFTY 50 Live Performance Dashboard (Enhanced)")

# -------------------------
# Ticker list (same as yours)
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
# Caching helpers
# -------------------------
@st.cache_data(ttl=300)
def load_live_data(ticker_list):
    # Use yf.Tickers for bulk info, but wrap in try/except to avoid single-ticker failures breaking everything
    tickers_obj = yf.Tickers(" ".join(ticker_list))
    rows = []
    for t in ticker_list:
        try:
            info = tickers_obj.tickers[t].info or {}
        except Exception:
            info = {}
        company = info.get("shortName", t.replace(".NS", ""))
        price = info.get("regularMarketPrice")
        prev_close = info.get("regularMarketPreviousClose")
        market_cap = info.get("marketCap")
        sector = info.get("sector", "Unknown")
        pct_change = None
        if price is not None and prev_close is not None and prev_close != 0:
            pct_change = ((price - prev_close) / prev_close) * 100
        rows.append({
            "Ticker": t,
            "Company": company,
            "Sector": sector,
            "Price": price,
            "Prev Close": prev_close,
            "% Change": pct_change,
            "Market Cap": market_cap
        })
    df = pd.DataFrame(rows)
    # drop rows missing the essentials, but keep as much as possible
    df = df.dropna(subset=["% Change", "Market Cap"], how="any")
    if df.empty:
        return df
    total_cap = df["Market Cap"].sum()
    df["Stake (%)"] = (df["Market Cap"] / total_cap) * 100
    # additional derived metrics
    df["Weighted Impact"] = (df["% Change"] * df["Stake (%)"]) / 100
    df["Trend"] = np.where(df["% Change"] > 0, "Gainer", "Loser")
    df["Color"] = df["% Change"].apply(lambda x: "green" if x > 0 else "red")
    return df

@st.cache_data(ttl=600)
def fetch_history_for_tickers(tickers, start_date, end_date):
    # returns a dict of dataframes keyed by ticker
    histories = {}
    for t in tickers:
        try:
            df_hist = yf.download(t, start=start_date, end=end_date, progress=False)
            # keep only close & date index
            if not df_hist.empty:
                histories[t] = df_hist[["Close"]].rename(columns={"Close": "Close"})
        except Exception as e:
            histories[t] = pd.DataFrame()
    return histories

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Filters & Controls")
with st.sidebar:
    search = st.text_input("Search company / ticker (substring)", "")
    sector_filter = st.selectbox("Sector filter", options=["All"], index=0)
    # we'll fill sector choices after loading data
    top_n = st.slider("Show top N movers by absolute % change", min_value=5, max_value=50, value=10, step=1)
    marketcap_min = st.number_input("Minimum market cap (crore INR) — approx", value=0.0, step=100.0)
    refresh_now = st.button("Refresh live data")

# Fetch live data (spinner)
with st.spinner("Fetching live NIFTY 50 data..."):
    df_live = load_live_data(NIFTY50_TICKERS)

if df_live.empty:
    st.error("No live data available right now. Yahoo Finance may be rate-limiting. Try again in a moment.")
    st.stop()

# populate sector filter options using live data
sectors = ["All"] + sorted(df_live["Sector"].fillna("Unknown").unique().tolist())
if sector_filter == "All" and len(sectors) > 1:
    # replace the default sector_filter with the first actual option if user hasn't chosen
    sector_filter = st.sidebar.selectbox("Sector filter", options=sectors, index=0)
else:
    sector_filter = st.sidebar.selectbox("Sector filter", options=sectors, index=sectors.index(sector_filter) if sector_filter in sectors else 0)

# apply search and sector & marketcap filters
df = df_live.copy()
if search:
    mask = df["Company"].str.contains(search, case=False, na=False) | df["Ticker"].str.contains(search, case=False, na=False)
    df = df.loc[mask]
if sector_filter != "All":
    df = df.loc[df["Sector"] == sector_filter]
if marketcap_min and marketcap_min > 0:
    # user enters crores; convert to absolute rupees (crore -> *1e7) but since marketCap is in absolute rupees, adapt:
    min_val = marketcap_min * 1e7
    df = df.loc[df["Market Cap"] >= min_val]

# Ranking / sorting
df = df.sort_values(by="% Change", ascending=False).reset_index(drop=True)
df["Rank"] = df["% Change"].rank(ascending=False, method="first").astype(int)
df["Abs Change"] = df["% Change"].abs()

# last updated info
last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"Data source: Yahoo Finance (`yfinance`). Last updated: {last_updated} (auto-cache TTL 5min)")

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
top_gainers = df.head(5)
top_losers = df.tail(5).sort_values(by="% Change").head(5)

gcol1, gcol2 = st.columns(2)
with gcol1:
    st.subheader("Top 5 Gainers")
    for _, r in top_gainers.iterrows():
        st.write(f"**{r['Company']}** ({r['Ticker']}) — {r['% Change']:.2f}%  — MarketCap: {r['Market Cap']:,}")
with gcol2:
    st.subheader("Top 5 Losers")
    for _, r in top_losers.iterrows():
        st.write(f"**{r['Company']}** ({r['Ticker']}) — {r['% Change']:.2f}%  — MarketCap: {r['Market Cap']:,}")

st.write("---")

# Add sparkline column for last 30 days
def make_sparkline(ticker):
    # small inline chart using plotly
    end = datetime.now().date()
    start = end - timedelta(days=30)
    try:
        hist = yf.download(ticker, start=start, end=end+timedelta(days=1), progress=False)
        if hist.empty:
            return None
        fig = go.Figure(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', line=dict(width=1)))
        fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=60, xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig
    except Exception:
        return None

# Prepare table for display
table_df = df[["Rank", "Ticker", "Company", "Sector", "Stake (%)", "Price", "% Change", "Weighted Impact", "Market Cap"]].copy()
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

# Dataframe + expanders for mini charts per selected companies
st.subheader("🏦 Company Performance (filtered)")
st.dataframe(table_display, use_container_width=True)

# Provide expanders with mini-sparklines for the top 10 displayed companies
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

# Charts: % change by company (color-coded), sector weighted impact, company stake pie, gainers vs losers
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

# Sector heatmap (pivot)
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
    # plot combined
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

# Company detail expander for user-selected single ticker
st.subheader("Company Detail")
ticker_choice = st.selectbox("Choose a single ticker to inspect", options=df["Ticker"].tolist())
if ticker_choice:
    row = df.loc[df["Ticker"] == ticker_choice].iloc[0]
    st.write(f"**{row['Company']}** ({row['Ticker']}) — Sector: {row['Sector']}")
    st.write(f"Price: {row['Price']}, Prev Close: {row['Prev Close']}, % Change: {row['% Change']:.2f}%")
    st.write(f"Market Cap: {row['Market Cap']:,}, Stake: {row['Stake (%)']:.2f}%")
    # fetch 90-day history and show OHLC
    hist90 = fetch_history_for_tickers([ticker_choice], datetime.now().date()-timedelta(days=90), datetime.now().date()).get(ticker_choice)
    if hist90 is not None and not hist90.empty:
        fig_o = px.line(hist90, y="Close", title=f"{row['Ticker']} — Last 90 days Close")
        st.plotly_chart(fig_o, use_container_width=True)
    else:
        st.info("90-day history not available.")

    with st.expander("Show company info & quick actions"):
        st.write("**Quick actions**")
        st.download_button("Download company historical (90d) CSV", data=hist90.to_csv().encode() if hist90 is not None else "", file_name=f"{ticker_choice}_90d.csv")
        st.write("**Notes / News**")
        st.info("News scraping / sentiment integration can be added (requires an external news API).")

st.write("---")
st.caption("Enhanced features: filters, download, historical comparison, sparklines, top movers & sector heatmap. Data via yfinance. Cache TTLs reduce redundant calls.")

# final tip panel
with st.container():
    st.markdown(
        """
        **Pro tips**
        - Use the sidebar search + sector filter to quickly narrow to sector-specific movers.
        - Use the historical comparison to compute relative performance (normalize).
        - If you plan heavy use, consider an API key / paid data provider to avoid yfinance rate limits.
        """
    )
