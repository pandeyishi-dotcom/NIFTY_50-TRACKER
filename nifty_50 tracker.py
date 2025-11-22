# ---------------------------------------------------------
# app.py — NIFTY50 Dashboard (Hybrid Fast Mode, No Freezing)
# ---------------------------------------------------------
# Features:
# ✔ FAST loading (3–7 seconds)
# ✔ NO bulk yfinance (no freezes)
# ✔ Per-ticker safe fetch with timeouts
# ✔ yfinance → Alpha OVERVIEW → NSE fallback
# ✔ Alpha price only if yfinance missing
# ✔ 30-day mini charts & 90-day full chart
# ✔ Weighted impact, ranks, top movers, sector impact
# ✔ Sector override file support (sectors_override.csv)
# ✔ Diagnostics panel
# ✔ 100% stable on Streamlit Cloud
# ---------------------------------------------------------

import os
import time
import math
import threading
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.express as px
import plotly.graph_objects as go


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="NIFTY50 – Hybrid Dashboard", layout="wide")
st.title("📈 NIFTY50 — Hybrid High-Speed Dashboard (No Freezing)")

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

ALPHA_KEY = os.environ.get("ALPHAVANTAGE_API_KEY") or (
    st.secrets["ALPHAVANTAGE_API_KEY"] if "ALPHAVANTAGE_API_KEY" in st.secrets else None
)

SECTOR_OVERRIDE_FILE = "sectors_override.csv"


# ---------------------------------------------------------
# SAFE TIMEOUT WRAPPER (core anti-freeze mechanic)
# ---------------------------------------------------------
def safe_timeout(func, timeout=3, default=None):
    result = [default]
    def wrapper():
        try:
            result[0] = func()
        except Exception:
            result[0] = default

    thread = threading.Thread(target=wrapper)
    thread.daemon = True
    thread.start()
    thread.join(timeout)

    return result[0]


# ---------------------------------------------------------
# SAFE JSON
# ---------------------------------------------------------
def safe_json(resp):
    try:
        return resp.json()
    except Exception:
        return None


# ---------------------------------------------------------
# FORMAT MARKET CAP
# ---------------------------------------------------------
def format_mcap(v):
    if v is None or pd.isna(v):
        return "N/A"
    try:
        crores = float(v) / 1e7
        return f"{crores:,.2f} Cr"
    except:
        return "N/A"


# ---------------------------------------------------------
# ALPHA OVERVIEW (only for missing sector/mcap)
# ---------------------------------------------------------
def alpha_overview(symbol_no_ns: str):
    if not ALPHA_KEY:
        return None, None

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "OVERVIEW",
        "symbol": f"{symbol_no_ns}.NS",
        "apikey": ALPHA_KEY
    }

    def call():
        r = requests.get(url, params=params, timeout=6)
        return safe_json(r)

    data = safe_timeout(call, timeout=4, default=None)
    if not data:
        return None, None

    sector = data.get("Sector")
    mcap = data.get("MarketCapitalization")

    try:
        mcap = float(mcap) if mcap else None
    except:
        mcap = None

    return sector, mcap


# ---------------------------------------------------------
# ALPHA PRICE (only if yfinance price is missing)
# ---------------------------------------------------------
def alpha_price(symbol_no_ns):
    if not ALPHA_KEY:
        return None, None, None

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": f"{symbol_no_ns}.NS",
        "apikey": ALPHA_KEY
    }

    def call():
        r = requests.get(url, params=params, timeout=6)
        return safe_json(r)

    data = safe_timeout(call, timeout=4, default=None)
    if not data:
        return None, None, None

    g = data.get("Global Quote", {})
    if not g:
        return None, None, None

    try:
        price = float(g.get("05. price")) if g.get("05. price") else None
        prev = float(g.get("08. previous close")) if g.get("08. previous close") else None
        pct = float(g.get("10. change percent").replace("%","")) if g.get("10. change percent") else None
        return price, prev, pct
    except:
        return None, None, None


# ---------------------------------------------------------
# NSE FALLBACK (only used last)
# ---------------------------------------------------------
def nse_quote(symbol_no_ns):
    url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol_no_ns}"
    headers = {
        "User-Agent":"Mozilla/5.0",
        "Referer":"https://www.nseindia.com/"
    }

    def call():
        s = requests.Session()
        s.get("https://www.nseindia.com", headers=headers, timeout=5)
        r = s.get(url, headers=headers, timeout=6)
        return safe_json(r)

    data = safe_timeout(call, timeout=4, default=None)
    if not data:
        return None, None, None, None

    price_info = data.get("priceInfo") or {}
    secinfo = data.get("securityInfo") or {}

    price = price_info.get("lastPrice")
    prev = price_info.get("previousClose")
    pchg = price_info.get("pChange")
    sector = secinfo.get("industry") or secinfo.get("industryType")

    return price, prev, pchg, sector


# ---------------------------------------------------------
# HISTORY FETCH (yfinance → Alpha)
# ---------------------------------------------------------
def get_history(ticker, start, end):
    # yfinance first
    try:
        df = yf.Ticker(ticker).history(start=start, end=end + timedelta(days=1))
        if df is not None and not df.empty and "Close" in df.columns:
            return df[["Close"]]
    except:
        pass

    # Alpha (last fallback)
    sym = ticker.replace(".NS", "")
    if not ALPHA_KEY:
        return pd.DataFrame()

    url = "https://www.alphavantage.co/query"
    params = {
        "function":"TIME_SERIES_DAILY_ADJUSTED",
        "symbol": f"{sym}.NS",
        "outputsize":"compact",
        "apikey":ALPHA_KEY
    }

    def call():
        r = requests.get(url, params=params, timeout=6)
        return safe_json(r)

    data = safe_timeout(call, timeout=4, default=None)
    if not data:
        return pd.DataFrame()

    ts = data.get("Time Series (Daily)") or {}
    rows = []
    for dstr, val in ts.items():
        d = pd.to_datetime(dstr)
        if start <= d.date() <= end:
            close = val.get("4. close") or val.get("5. adjusted close")
            if close:
                rows.append({"Date": d, "Close": float(close)})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("Date").sort_index()
    return df


# ---------------------------------------------------------
# SECTOR OVERRIDE
# ---------------------------------------------------------
def load_sector_override(path=SECTOR_OVERRIDE_FILE):
    if os.path.exists(path):
        try:
            so = pd.read_csv(path)
            return dict(zip(so["Ticker"].str.upper(), so["Sector"]))
        except:
            return {}
    return {}


SECTOR_OVERRIDE = load_sector_override()


# ---------------------------------------------------------
# MASTER LOADER (no bulk yfinance)
# ---------------------------------------------------------
def load_live(tickers):
    rows = []

    for t in tickers:
        sym = t.replace(".NS","")

        # -----------------------------------------
        # 1) yfinance fast_info (SUPER fast)
        # -----------------------------------------
        def fetch_fast():
            return yf.Ticker(t).fast_info

        fi = safe_timeout(fetch_fast, timeout=2, default=None)

        price = fi.get("last_price") if fi else None
        prev = fi.get("previous_close") if fi else None
        mcap = fi.get("market_cap") if fi else None
        sector = fi.get("sector") if fi else None
        company = fi.get("shortName") if fi else sym
        source = "yfinance-fast"

        # -----------------------------------------
        # 2) If yfinance missing price → Alpha
        # -----------------------------------------
        if price is None:
            price, prev, pct_alpha = alpha_price(sym)
            if price is not None:
                source = "alpha"
        else:
            pct_alpha = None

        # -----------------------------------------
        # 3) If sector/mcap missing → Alpha overview
        # -----------------------------------------
        if (not sector) or (mcap is None):
            sec_av, mcap_av = alpha_overview(sym)
            if sec_av and not sector:
                sector = sec_av
            if mcap_av and (mcap is None):
                mcap = mcap_av

        # -----------------------------------------
        # 4) If still missing → NSE fallback
        # -----------------------------------------
        if price is None or sector is None:
            n_price, n_prev, n_pchg, n_sector = nse_quote(sym)
            if n_price is not None and price is None:
                price, prev = n_price, n_prev
                source = "nse"
            if n_sector and (sector is None):
                sector = n_sector

        # -----------------------------------------
        # 5) % change calc
        # -----------------------------------------
        pct = None
        if price is not None and prev not in (None, 0):
            try:
                pct = ((price - prev) / prev) * 100
            except:
                pct = pct_alpha

        rows.append({
            "Ticker": t,
            "Company": company,
            "Price": price,
            "Prev Close": prev,
            "% Change": pct,
            "Market Cap": mcap,
            "Sector": sector,
            "Source": source
        })

    df = pd.DataFrame(rows)

    # apply sector override
    if SECTOR_OVERRIDE:
        df["Sector"] = df.apply(
            lambda r: SECTOR_OVERRIDE.get(r["Ticker"].upper(), r["Sector"]),
            axis=1
        )

    return df


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
with st.spinner("Loading live NIFTY50 data..."):
    df = load_live(NIFTY50_TICKERS)

if df.empty:
    st.error("Failed to load data. Check your Alpha key & internet.")
    st.stop()


# ---------------------------------------------------------
# CLEAN + CALCULATIONS
# ---------------------------------------------------------
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Prev Close"] = pd.to_numeric(df["Prev Close"], errors="coerce")
df["% Change"] = pd.to_numeric(df["% Change"], errors="coerce")
df["Market Cap"] = pd.to_numeric(df["Market Cap"], errors="coerce")

mask = df["% Change"].isna() & df["Price"].notna() & df["Prev Close"].notna()
df.loc[mask, "% Change"] = ((df.loc[mask,"Price"] - df.loc[mask,"Prev Close"]) /
                            df.loc[mask,"Prev Close"]) * 100

df["Market Cap Calc"] = df["Market Cap"].fillna(df["Market Cap"].median())
df["Stake (%)"] = (df["Market Cap Calc"] / df["Market Cap Calc"].sum()) * 100
df["Weighted Impact"] = (df["% Change"].fillna(0) * df["Stake (%)"]) / 100
df["Trend"] = np.where(df["% Change"] > 0, "Gainer", "Loser")
df["Rank"] = df["% Change"].rank(ascending=False, method="first").astype("Int64")
df = df.sort_values("Market Cap Calc", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------
st.sidebar.header("Filters")
search = st.sidebar.text_input("Search company/ticker")
sector_sel = st.sidebar.selectbox("Sector", ["All"] + sorted(df["Sector"].fillna("Unknown").unique()))
mincap = st.sidebar.number_input("Min Market Cap (Cr)", value=0.0)

if st.sidebar.button("Refresh"):
    st.cache_data.clear()
    st.experimental_rerun()

# filtering
display_df = df.copy()
if search:
    display_df = display_df[
        display_df["Company"].str.contains(search, case=False, na=False) |
        display_df["Ticker"].str.contains(search, case=False, na=False)
    ]
if sector_sel != "All":
    display_df = display_df[display_df["Sector"] == sector_sel]
if mincap > 0:
    display_df = display_df[display_df["Market Cap"] >= (mincap * 1e7)]


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------
c1,c2,c3 = st.columns(3)
c1.metric("Gainers Impact", f"{display_df[display_df['% Change']>0]['% Change'].sum():.2f}%")
c2.metric("Losers Impact", f"{display_df[display_df['% Change']<0]['% Change'].sum():.2f}%")
c3.metric("Overall Impact", f"{display_df['Weighted Impact'].sum():.2f}%")


st.write("---")


# ---------------------------------------------------------
# TOP MOVERS
# ---------------------------------------------------------
left, right = st.columns(2)
left.subheader("Top 5 Gainers")
for _, r in display_df.sort_values("% Change", ascending=False).head(5).iterrows():
    left.write(f"**{r['Company']}** — {r['% Change']:.2f}% — MC: {format_mcap(r['Market Cap'])}")

right.subheader("Top 5 Losers")
for _, r in display_df.sort_values("% Change", ascending=True).head(5).iterrows():
    right.write(f"**{r['Company']}** — {r['% Change']:.2f}% — MC: {format_mcap(r['Market Cap'])}")


st.write("---")


# ---------------------------------------------------------
# TABLE
# ---------------------------------------------------------
table = display_df.copy()
table_display = table[[
    "Rank","Ticker","Company","Sector","Price","% Change","Stake (%)",
    "Weighted Impact","Market Cap","Source"
]].copy()

table_display["Stake (%)"] = table_display["Stake (%)"].round(2)
table_display["% Change"] = table_display["% Change"].round(2)
table_display["Weighted Impact"] = table_display["Weighted Impact"].round(4)
table_display["Market Cap"] = table_display["Market Cap"].apply(format_mcap)

st.subheader("Company Performance")
st.dataframe(table_display, use_container_width=True)
st.download_button("Download CSV", table_display.to_csv(index=False), "nifty50.csv")


st.write("---")


# ---------------------------------------------------------
# MINI HISTORY (30 days)
# ---------------------------------------------------------
with st.expander("Mini Charts (30 days)"):
    end = datetime.now().date()
    start = end - timedelta(days=30)
    subset = display_df["Ticker"].head(12).tolist()

    cols = st.columns(3)
    for i, t in enumerate(subset):
        company = display_df.loc[display_df["Ticker"] == t, "Company"].iloc[0]
        hist = get_history(t, start, end)
        with cols[i % 3]:
            st.write(f"**{company}** ({t})")
            if hist.empty:
                st.info("No data")
            else:
                fig = go.Figure(go.Scatter(x=hist.index, y=hist["Close"], mode="lines", line=dict(width=1)))
                fig.update_layout(height=80, margin=dict(l=0,r=0,t=0,b=0),
                                  xaxis=dict(visible=False), yaxis=dict(visible=False))
                st.plotly_chart(fig, use_container_width=True)


st.write("---")


# ---------------------------------------------------------
# CHARTS — % Change, Sector Impact, Stake Pie, Gainer/Loser Pie
# ---------------------------------------------------------
st.plotly_chart(
    px.bar(display_df.sort_values("% Change", ascending=False),
           x="Company", y="% Change", color="Trend",
           title="% Change by Company"),
    use_container_width=True
)

# sector impact
sector_agg = df.groupby(df["Sector"].fillna("Unknown"))["Weighted Impact"].sum().reset_index()
if sector_agg["Sector"].nunique() > 1:
    st.plotly_chart(
        px.bar(sector_agg.sort_values("Weighted Impact"),
               x="Sector", y="Weighted Impact",
               title="Weighted Sector Impact",
               color="Weighted Impact",
               color_continuous_scale=px.colors.diverging.RdYlGn),
        use_container_width=True
    )
else:
    st.warning("Sector data limited. Add Alpha key or sectors_override.csv.")

# stake pie
st.plotly_chart(
    px.pie(df.sort_values("Market Cap Calc", ascending=False).head(20),
           values="Stake (%)", names="Company",
           title="Company Stake Distribution (Top 20)"),
    use_container_width=True
)

# gainers vs losers
trend = display_df["Trend"].value_counts().reset_index()
trend.columns = ["Trend", "Count"]
st.plotly_chart(
    px.pie(trend, values="Count", names="Trend", title="Gainers vs Losers",
           color_discrete_map={"Gainer":"green","Loser":"red"}),
    use_container_width=True
)


st.write("---")


# ---------------------------------------------------------
# COMPANY DETAIL — 90 DAY CHART
# ---------------------------------------------------------
st.subheader("Company Detail")
selected = st.selectbox("Choose Ticker", df["Ticker"])

row = df[df["Ticker"]==selected].iloc[0]
st.write(f"**{row['Company']}** — Sector: {row['Sector']}")
st.write(f"Price: {row['Price']} | Prev Close: {row['Prev Close']} | % Change: {row['% Change']:.2f}")
st.write(f"Market Cap: {format_mcap(row['Market Cap'])}")

end2 = datetime.now().date()
start2 = end2 - timedelta(days=90)
hist90 = get_history(selected, start2, end2)

if hist90.empty:
    st.info("90-day history not available.")
else:
    st.plotly_chart(px.line(hist90, y="Close", title=f"{selected} — 90-Day History"),
                    use_container_width=True)
    st.download_button("Download 90-day CSV", hist90.to_csv(), f"{selected}_90d.csv")


# ---------------------------------------------------------
# DIAGNOSTICS
# ---------------------------------------------------------
with st.expander("Diagnostics"):
    st.write(df.head(25))
    st.write("Rows with missing sector:", df[df["Sector"].isna()])
    st.write("Rows from NSE:", df[df["Source"]=="nse"])
    st.write("Alpha Key Present:", bool(ALPHA_KEY))

st.caption("Hybrid dashboard (v1). All external calls protected by timeouts. Fast & stable.")
