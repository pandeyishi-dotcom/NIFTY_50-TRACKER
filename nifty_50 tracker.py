# app.py — Robust NIFTY50 dashboard (sync, production-safe)
import os
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

st.set_page_config(page_title="NIFTY50 Robust Dashboard", layout="wide")
st.title("📈 NIFTY 50 — Robust Dashboard (fixed)")

# ------------ Config ------------
NIFTY50_TICKERS = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "BHARTIARTL.NS","HINDUNILVR.NS","ITC.NS","KOTAKBANK.NS","LT.NS",
    "SBIN.NS","BAJFINANCE.NS","AXISBANK.NS","ASIANPAINT.NS","HCLTECH.NS",
    "SUNPHARMA.NS","MARUTI.NS","ULTRACEMCO.NS","POWERGRID.NS","NTPC.NS",
    "TITAN.NS","TECHM.NS","ONGC.NS","JSWSTEEL.NS","TATASTEEL.NS",
    "NESTLEIND.NS","WIPRO.NS","GRASIM.NS","COALINDIA.NS","ADANIENT.NS",
    "ADANIPORTS.NS","EICHERMOT.NS","BRITANNIA.NS","HDFCLIFE.NS","BAJAJFINSV.NS",
    "HEROMOTOCO.NS","DRREDDY.NS","CIPLA.NS","BPCL.NS","SHREECEM.NS",
    "INDUSINDBK.NS","SBILIFE.NS","TATACONSUM.NS","HINDALCO.NS","BAJAJ-AUTO.NS",
    "DIVISLAB.NS","UPL.NS","APOLLOHOSP.NS","M&M.NS","TATAMOTORS.NS"
]

ALPHA_KEY = os.environ.get("ALPHAVANTAGE_API_KEY") or (st.secrets.get("ALPHAVANTAGE_API_KEY") if "ALPHAVANTAGE_API_KEY" in st.secrets else None)

# ------------ Helpers ------------
def safe_json(resp):
    try:
        return resp.json()
    except Exception:
        return None

def format_mcap_display(mc):
    """Display-friendly market cap string (crores) or 'N/A' when missing."""
    try:
        if mc is None or (isinstance(mc, float) and math.isnan(mc)):
            return "N/A"
        mc = float(mc)
        crores = mc / 1e7
        return f"{crores:,.2f} Cr"
    except Exception:
        return "N/A"

def _cell(v):
    if v is None:
        return None
    if isinstance(v, (list, dict, tuple, set, pd.Series, pd.DataFrame)):
        return str(v)
    try:
        if hasattr(v, "item"):
            return v.item()
    except Exception:
        pass
    return v

def sanitize_for_streamlit(df):
    """
    Safely convert dataframe to Arrow-friendly scalars.
    Handles duplicate column names (turns multi-col into flattened names).
    """
    cleaned = {}
    for col in df.columns:
        series = df[col]
        # if duplicate names, pandas returns a DataFrame slice
        if isinstance(series, pd.DataFrame):
            for i, subcol in enumerate(series.columns):
                new_name = f"{col}_{i}"
                cleaned[new_name] = series[subcol].apply(_cell)
        else:
            cleaned[col] = series.apply(_cell)
    return pd.DataFrame(cleaned)

# ------------ Data-source helpers (sync) ------------
def alpha_global_quote(symbol):
    """AlphaVantage GLOBAL_QUOTE (symbol without .NS)"""
    if not ALPHA_KEY:
        return None, None, None
    url = "https://www.alphavantage.co/query"
    params = {"function": "GLOBAL_QUOTE", "symbol": symbol + ".NS", "apikey": ALPHA_KEY}
    try:
        r = requests.get(url, params=params, timeout=8)
        data = safe_json(r)
        g = data.get("Global Quote", {}) if data else {}
        if not g:
            return None, None, None
        price = g.get("05. price")
        prev = g.get("08. previous close")
        pct = g.get("10. change percent")
        try:
            price = float(price) if price else None
            prev = float(prev) if prev else None
            pct = float(pct.replace("%", "")) if pct else None
        except Exception:
            price, prev, pct = None, None, None
        return price, prev, pct
    except Exception:
        return None, None, None

def alpha_overview(symbol):
    """AlphaVantage OVERVIEW for Sector and MarketCapitalization"""
    if not ALPHA_KEY:
        return None, None
    url = "https://www.alphavantage.co/query"
    params = {"function": "OVERVIEW", "symbol": symbol + ".NS", "apikey": ALPHA_KEY}
    try:
        r = requests.get(url, params=params, timeout=8)
        data = safe_json(r)
        if not data:
            return None, None
        sector = data.get("Sector")
        mcap = data.get("MarketCapitalization")
        try:
            mcap = float(mcap) if mcap else None
        except Exception:
            mcap = None
        return sector, mcap
    except Exception:
        return None, None

def nse_quote(symbol):
    """Best-effort NSE quote (public endpoint). May fail depending on NSE protections."""
    url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
        "Accept": "application/json, text/javascript, */*; q=0.01",
    }
    s = requests.Session()
    try:
        s.get("https://www.nseindia.com", headers=headers, timeout=5)
        r = s.get(url, headers=headers, timeout=8)
        data = safe_json(r)
        if not data:
            return None, None, None, None
        price_info = data.get("priceInfo") or {}
        last = price_info.get("lastPrice") or price_info.get("last")
        prev = price_info.get("previousClose")
        pchange = price_info.get("pChange")
        secinfo = data.get("securityInfo") or {}
        sector = secinfo.get("industry") or secinfo.get("industryType")
        return last, prev, pchange, sector
    except Exception:
        return None, None, None, None

# ------------ Load & build robust dataframe ------------
@st.cache_data(ttl=150)
def load_live_data(tickers):
    rows = []
    # 1) bulk price: yfinance (fast)
    try:
        prices = yf.download(tickers=tickers, period="2d", interval="1d", group_by="ticker", threads=True, progress=False, auto_adjust=False)
    except Exception:
        prices = pd.DataFrame()

    for t in tickers:
        sym = t.replace(".NS", "")
        company = sym
        price = None
        prev = None
        pct = None
        market_cap = None
        sector = None
        source = None

        # Attempt 1: yfinance bulk price
        try:
            if isinstance(prices, pd.DataFrame) and t in prices.columns.get_level_values(0):
                closes = prices[t]["Close"].dropna()
                if len(closes) >= 1:
                    price = float(closes.iloc[-1])
                if len(closes) >= 2:
                    prev = float(closes.iloc[-2])
        except Exception:
            pass

        # Attempt 2: yahoo fast_info for metadata
        try:
            fi = yf.Ticker(t).fast_info
            if fi:
                # try common keys
                market_cap = fi.get("market_cap") or fi.get("marketCap") or market_cap
                # some tickers may provide sector in different casing
                sector = fi.get("sector") or fi.get("Sector") or sector
                company = fi.get("shortName") or fi.get("longName") or company
                if price is not None:
                    source = "yahoo"
        except Exception:
            pass

        # Compute pct if we have prev
        if price is not None and prev is not None and prev != 0:
            try:
                pct = ((price - prev) / prev) * 100
            except Exception:
                pct = None

        # Alpha fallback for price
        if price is None:
            p, pc, pchg = alpha_global_quote(sym)
            if p is not None:
                price, prev, pct = p, pc, pchg
                source = source or "alpha"

        # NSE fallback for price & sector
        if price is None or sector is None:
            n_price, n_prev, n_pchg, n_sector = nse_quote(sym)
            if n_price is not None and price is None:
                price, prev, pct = n_price, n_prev, n_pchg
                source = source or "nse"
            if n_sector and sector is None:
                sector = n_sector

        # Alpha overview for sector & market cap (if missing)
        if (sector is None or market_cap is None) and ALPHA_KEY:
            sec_av, mcap_av = alpha_overview(sym)
            if sec_av and sector is None:
                sector = sec_av
            if mcap_av and market_cap is None:
                market_cap = mcap_av

        rows.append({
            "Ticker": t,
            "Company": company,
            "Sector": sector if sector else None,
            "Price": price,
            "Prev Close": prev,
            "% Change": pct,
            "Market Cap": market_cap,
            "Source": source or "unknown"
        })
    return pd.DataFrame(rows)

# ------------ Main app flow ------------
with st.spinner("Fetching live data..."):
    df = load_live_data(NIFTY50_TICKERS)

if df is None or df.empty:
    st.error("Failed to load any data. Check network / API keys. (No rows returned).")
    st.stop()

# Numeric coercion
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Prev Close"] = pd.to_numeric(df["Prev Close"], errors="coerce")
df["% Change"] = pd.to_numeric(df["% Change"], errors="coerce")
df["Market Cap"] = pd.to_numeric(df["Market Cap"], errors="coerce")

# Recompute missing % Change where possible
mask = df["% Change"].isna() & df["Price"].notna() & df["Prev Close"].notna() & (df["Prev Close"] != 0)
df.loc[mask, "% Change"] = ((df.loc[mask, "Price"] - df.loc[mask, "Prev Close"]) / df.loc[mask, "Prev Close"]) * 100

# Market Cap Calc: use median to fill missing for weighting logic only
if df["Market Cap"].notna().sum() >= 1:
    med = df.loc[df["Market Cap"].notna(), "Market Cap"].median()
    df["Market Cap Calc"] = df["Market Cap"].fillna(med)
else:
    # no market cap available at all — use synthetic equal calc but do not surface it
    df["Market Cap Calc"] = 1.0

# Stake & weighted impact
total_calc = df["Market Cap Calc"].sum() if df["Market Cap Calc"].sum() != 0 else 1.0
df["Stake (%)"] = (df["Market Cap Calc"] / total_calc) * 100
df["Weighted Impact"] = (df["% Change"].fillna(0) * df["Stake (%)"]) / 100

# Trend + color
df["Trend"] = np.where(df["% Change"] > 0, "Gainer", "Loser")
df["Color"] = df["% Change"].apply(lambda x: "green" if x > 0 else "red")

# Rank (nullable int) — safe cast
ranks = df["% Change"].rank(ascending=False, method="first")
try:
    df["Rank"] = ranks.astype("Int64")
except Exception:
    # safe fallback: convert to integers (0 if missing)
    df["Rank"] = ranks.fillna(0).astype(int)

# Default sort: largest market-cap first (use Market Cap Calc)
df = df.sort_values(by="Market Cap Calc", ascending=False).reset_index(drop=True)

# Sidebar filters
st.sidebar.header("Filters & Controls")
search = st.sidebar.text_input("Search company / ticker (substring)", value="")
sector_options = ["All"] + sorted(df["Sector"].fillna("Unknown").unique().tolist())
sector_sel = st.sidebar.selectbox("Sector filter", sector_options, index=0)
min_mcap_crore = st.sidebar.number_input("Minimum market cap (crore INR)", value=0.0, step=100.0)
refresh = st.sidebar.button("Refresh data")

# Apply filters
display_df = df.copy()
if search:
    display_df = display_df[display_df["Company"].str.contains(search, case=False, na=False) | display_df["Ticker"].str.contains(search, case=False, na=False)]
if sector_sel != "All":
    if sector_sel == "Unknown":
        display_df = display_df[display_df["Sector"].isna()]
    else:
        display_df = display_df[display_df["Sector"] == sector_sel]
if min_mcap_crore > 0:
    display_df = display_df[display_df["Market Cap Calc"] >= (min_mcap_crore * 1e7)]

# Top-line metrics
total_positive = display_df.loc[display_df["% Change"] > 0, "% Change"].sum()
total_negative = display_df.loc[display_df["% Change"] < 0, "% Change"].sum()
overall_impact = display_df["Weighted Impact"].sum()

c1, c2, c3 = st.columns(3)
c1.metric("Total Gainers Impact", f"{total_positive:.2f} %")
c2.metric("Total Losers Impact", f"{total_negative:.2f} %")
c3.metric("Overall Weighted Impact", f"{overall_impact:.2f} %")

st.write("---")

# Top gainers/losers
top_gainers = display_df.sort_values("% Change", ascending=False).head(5)
top_losers = display_df.sort_values("% Change", ascending=True).head(5)
gcol, lcol = st.columns(2)
with gcol:
    st.subheader("Top 5 Gainers")
    for _, r in top_gainers.iterrows():
        pct = r["% Change"] if pd.notna(r["% Change"]) else 0.0
        st.markdown(f"**{r['Company']}** ({r['Ticker']}) — `{pct:.2f}%` — MarketCap: **{format_mcap_display(r['Market Cap'])}** — Source: {r['Source']}")
with lcol:
    st.subheader("Top 5 Losers")
    for _, r in top_losers.iterrows():
        pct = r["% Change"] if pd.notna(r["% Change"]) else 0.0
        st.markdown(f"**{r['Company']}** ({r['Ticker']}) — `{pct:.2f}%` — MarketCap: **{format_mcap_display(r['Market Cap'])}** — Source: {r['Source']}")

st.write("---")

# Table for UI (display Market Cap as friendly string; do NOT show Market Cap Calc)
table = display_df[["Rank","Ticker","Company","Sector","Stake (%)","Price","% Change","Weighted Impact","Market Cap","Source"]].copy()
table["Stake (%)"] = table["Stake (%)"].round(2)
table["% Change"] = table["% Change"].round(2)
table["Weighted Impact"] = table["Weighted Impact"].round(4)
table["Market Cap Display"] = table["Market Cap"].apply(format_mcap_display)
display_table = table.drop(columns=["Market Cap"]).rename(columns={"Market Cap Display":"Market Cap"})

# Sanitize and show
safe_table = sanitize_for_streamlit(display_table)
st.subheader("Company Performance")
st.dataframe(safe_table, use_container_width=True)
st.download_button("Download filtered table (CSV)", data=safe_table.to_csv(index=False).encode(), file_name="nifty50_filtered.csv", mime="text/csv")

st.write("---")

# Mini sparklines — show last 30 days for first 12 displayed tickers
with st.expander("Mini historical charts for displayed companies (last 30 days)"):
    tickers_to_show = display_table["Ticker"].tolist()[:12]
    end = datetime.now().date()
    start = end - timedelta(days=30)
    # fetch histories with yfinance; fall back to Alpha if needed
    for i, t in enumerate(tickers_to_show):
        company_row = display_table[display_table["Ticker"] == t].iloc[0]
        col = st.columns(3)[i % 3]
        with col:
            st.write(f"**{company_row['Company']}** — {t} (Source: {company_row['Source']})")
            try:
                hist = yf.download(t, start=start, end=end + timedelta(days=1), progress=False)
                if hist is not None and not hist.empty:
                    fig = go.Figure(go.Scatter(x=hist.index, y=hist["Close"], mode="lines", line=dict(width=1)))
                    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=80, xaxis=dict(visible=False), yaxis=dict(visible=False))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Mini history not available")
            except Exception:
                st.info("Mini history not available")

st.write("---")

# Charts
fig_pct = px.bar(display_df.sort_values("% Change", ascending=False), x="Company", y="% Change", color="Trend", title="% Change by Company")
fig_pct.update_traces(texttemplate="%{y:.2f}", textposition="outside")
st.plotly_chart(fig_pct, use_container_width=True)

# Sector impact chart — only if we have multiple sectors
sector_impact = df.groupby(df["Sector"].fillna("Unknown"))["Weighted Impact"].sum().reset_index()
if len(sector_impact) <= 1:
    st.warning("Sector metadata limited or absent. Sector breakdown may be unavailable.")
else:
    st.subheader("Weighted Sector Impact")
    st.plotly_chart(px.bar(sector_impact.sort_values("Weighted Impact"), x="Sector", y="Weighted Impact", color="Weighted Impact", title="Weighted Sector Impact", color_continuous_scale=px.colors.diverging.RdYlGn), use_container_width=True)

# Company stake distribution (limit to top 20 to avoid crowded pie)
stake_df = df.sort_values("Market Cap Calc", ascending=False).head(20)
st.plotly_chart(px.pie(stake_df, values="Stake (%)", names="Company", title="Company Stake Distribution (Top 20)"), use_container_width=True)

# Gainers vs losers pie (counts)
trend_count = display_df["Trend"].value_counts().rename_axis("Trend").reset_index(name="Count")
if trend_count["Count"].sum() == 0:
    st.info("No gainers/losers to display.")
else:
    st.plotly_chart(px.pie(trend_count, names="Trend", values="Count", title="Gainers vs Losers Ratio", color_discrete_map={"Gainer":"green","Loser":"red"}), use_container_width=True)

st.write("---")

# Company detail panel
st.subheader("Company Detail")
choice = st.selectbox("Choose a single ticker to inspect", options=df["Ticker"].tolist())
selected = df[df["Ticker"] == choice] if choice else pd.DataFrame()
if not selected.empty:
    r = selected.iloc[0]
    st.markdown(f"**{r['Company']}** ({r['Ticker']}) — Sector: {r['Sector'] if pd.notna(r['Sector']) else 'Unknown'}")
    st.markdown(f"Price: **{r['Price'] if pd.notna(r['Price']) else 'N/A'}** | Prev Close: **{r['Prev Close'] if pd.notna(r['Prev Close']) else 'N/A'}** | % Change: **{r['% Change']:.2f}%**" if pd.notna(r['% Change']) else "N/A")
    st.markdown(f"Market Cap: **{format_mcap_display(r['Market Cap'])}** — Stake: **{r['Stake (%)']:.2f}%** — Source: {r['Source']}")
    # 90-day history fallback
    try:
        hist90 = yf.download(r["Ticker"], period="3mo", progress=False)
        if hist90 is not None and not hist90.empty:
            st.plotly_chart(px.line(hist90, y="Close", title=f"{r['Ticker']} — Last 90 days"), use_container_width=True)
            st.download_button("Download 90d CSV", data=hist90.to_csv().encode(), file_name=f"{r['Ticker']}_90d.csv", mime="text/csv")
        else:
            st.info("90-day history not available (yfinance).")
            # optional AlphaVantage fallback omitted to avoid rate-limit
    except Exception:
        st.info("90-day history not available (error fetching).")
else:
    st.info("No company selected or data missing.")

st.write("---")
st.caption("Robust update: Market-cap placeholders hidden, safer fallbacks, sanitized tables. If you still see many N/A values, set ALPHAVANTAGE_API_KEY in Streamlit secrets for better metadata.")

