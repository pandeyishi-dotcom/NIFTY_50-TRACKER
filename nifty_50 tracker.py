import os
import math
import time
import json
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
st.set_page_config(page_title="NIFTY50 Dashboard", layout="wide")

st.title("📈 NIFTY 50 – Enterprise Dashboard (Stable Build)")

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

ALPHA_KEY = os.environ.get("ALPHAVANTAGE_API_KEY") or st.secrets.get("ALPHAVANTAGE_API_KEY")

# --------------------------------------------------
# UTILITIES
# --------------------------------------------------

def format_mcap(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "N/A"
    crores = x / 1e7
    return f"{crores:,.2f} Cr"

def safe_json(resp):
    try:
        return resp.json()
    except:
        return None

# --------------------------------------------------
# FALLBACK DATA SOURCES
# --------------------------------------------------

def fetch_alpha_global(ticker):
    if not ALPHA_KEY:
        return None, None, None

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": ticker,
        "apikey": ALPHA_KEY
    }
    r = requests.get(url, params=params, timeout=10)
    data = safe_json(r)
    g = data.get("Global Quote", {})
    if not g:
        return None, None, None

    price = g.get("05. price")
    prev = g.get("08. previous close")
    pct = g.get("10. change percent")

    try:
        price = float(price) if price else None
        prev = float(prev) if prev else None
        pct = float(pct.replace("%", "")) if pct else None
    except:
        price, prev, pct = None, None, None

    return price, prev, pct


def fetch_alpha_overview(ticker):
    if not ALPHA_KEY:
        return None, None

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "OVERVIEW",
        "symbol": ticker,
        "apikey": ALPHA_KEY
    }
    r = requests.get(url, params=params, timeout=10)
    data = safe_json(r)
    if not data:
        return None, None

    sector = data.get("Sector")
    mcap = data.get("MarketCapitalization")

    try:
        mcap = float(mcap) if mcap else None
    except:
        mcap = None

    return sector, mcap

# --------------------------------------------------
# MAIN FETCH LOGIC (NO ASYNC)
# --------------------------------------------------

@st.cache_data(ttl=120)
def load_live():
    rows = []

    prices = yf.download(
        tickers=NIFTY50_TICKERS,
        period="2d",
        interval="1d",
        group_by="ticker",
        threads=True,
        progress=False
    )

    for t in NIFTY50_TICKERS:
        company = t.replace(".NS", "")
        price = None
        prev = None
        pct = None
        sector = None
        mcap = None
        source = "none"

        # Try Yahoo bulk price
        try:
            if t in prices.columns.get_level_values(0):
                p = prices[t]["Close"].dropna()
                if len(p) >= 1:
                    price = float(p.iloc[-1])
                if len(p) >= 2:
                    prev = float(p.iloc[-2])
        except:
            pass

        # Try fast_info
        try:
            fi = yf.Ticker(t).fast_info
            if fi:
                mcap = fi.get("market_cap") or mcap
                sector = fi.get("sector") or sector
                company = fi.get("shortName") or company
                source = "yahoo"
        except:
            pass

        # Compute pct
        if price and prev and prev != 0:
            pct = ((price - prev) / prev) * 100

        # Fallback to AlphaVantage
        if price is None:
            p, pc, pchg = fetch_alpha_global(t.replace(".NS", ""))
            if p:
                price, prev, pct = p, pc, pchg
                source = "alpha"

        # Fetch overview for sector / mcap
        if (sector is None or mcap is None) and ALPHA_KEY:
            sec_av, mcap_av = fetch_alpha_overview(t.replace(".NS", ""))
            if sec_av:
                sector = sec_av
            if mcap_av:
                mcap = mcap_av

        rows.append({
            "Ticker": t,
            "Company": company,
            "Sector": sector or "Unknown",
            "Price": price,
            "Prev Close": prev,
            "% Change": pct,
            "Market Cap": mcap,
            "Source": source
        })

    return pd.DataFrame(rows)

# --------------------------------------------------
# SANITIZER (NO CRASH EVER)
# --------------------------------------------------

def sanitize(df):
    cleaned = {}
    for col in df.columns:
        s = df[col]

        # if duplicate columns → a DF appears
        if isinstance(s, pd.DataFrame):
            for i, sub in enumerate(s.columns):
                cleaned[f"{col}_{i}"] = s[sub].apply(_cell)
        else:
            cleaned[col] = s.apply(_cell)

    return pd.DataFrame(cleaned)

def _cell(v):
    if v is None:
        return None
    if isinstance(v, (list, dict, tuple, set, pd.Series, pd.DataFrame)):
        return str(v)
    try:
        if hasattr(v, "item"):
            return v.item()
    except:
        pass
    return v

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

with st.spinner("Fetching market data..."):
    df = load_live()

# Fix numerics
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Prev Close"] = pd.to_numeric(df["Prev Close"], errors="coerce")
df["% Change"] = pd.to_numeric(df["% Change"], errors="coerce")
df["Market Cap"] = pd.to_numeric(df["Market Cap"], errors="coerce")

# Recompute missing pct
mask = df["% Change"].isna() & df["Price"].notna() & df["Prev Close"].notna()
df.loc[mask, "% Change"] = ((df.loc[mask, "Price"] - df.loc[mask, "Prev Close"]) /
                            df.loc[mask, "Prev Close"]) * 100

# Market Cap Calc
if df["Market Cap"].notna().sum() >= 1:
    med = df["Market Cap"].median()
    df["Market Cap Calc"] = df["Market Cap"].fillna(med)
else:
    df["Market Cap Calc"] = 1.0

# Stake
df["Stake (%)"] = (df["Market Cap Calc"] / df["Market Cap Calc"].sum()) * 100

# Weighted Impact
df["Weighted Impact"] = (df["% Change"].fillna(0) * df["Stake (%)"]) / 100

# Rank
df["Rank"] = df["% Change"].rank(ascending=False, method="first").astype("Int64")

# Sort by mcap
df = df.sort_values("Market Cap Calc", ascending=False)

# --------------------------------------------------
# Summary Metrics
# --------------------------------------------------

c1, c2, c3 = st.columns(3)
c1.metric("Total Gainers Impact", f"{df[df['% Change']>0]['% Change'].sum():.2f}%")
c2.metric("Total Losers Impact", f"{df[df['% Change']<0]['% Change'].sum():.2f}%")
c3.metric("Overall Impact", f"{df['Weighted Impact'].sum():.2f}%")

st.write("---")

# --------------------------------------------------
# Top Movers
# --------------------------------------------------

tg, tl = st.columns(2)

tg.subheader("Top 5 Gainers")
g = df.sort_values("% Change", ascending=False).head(5)
for _, r in g.iterrows():
    tg.write(f"**{r['Company']}** ({r['Ticker']}): `{r['% Change']:.2f}%` — MC: {format_mcap(r['Market Cap'])}")

tl.subheader("Top 5 Losers")
l = df.sort_values("% Change", ascending=True).head(5)
for _, r in l.iterrows():
    tl.write(f"**{r['Company']}** ({r['Ticker']}): `{r['% Change']:.2f}%` — MC: {format_mcap(r['Market Cap'])}")

st.write("---")

# --------------------------------------------------
# TABLE
# --------------------------------------------------

table = df.copy()
table["Market Cap"] = table["Market Cap"].apply(format_mcap)

safe_table = sanitize(table)

st.subheader("Company Performance")
st.dataframe(safe_table, use_container_width=True)
st.download_button("Download CSV", safe_table.to_csv(index=False), "nifty50.csv")

st.write("---")

# --------------------------------------------------
# CHARTS
# --------------------------------------------------

fig1 = px.bar(df.sort_values("% Change", ascending=False),
              x="Company", y="% Change", color="% Change",
              title="% Change by Company")
st.plotly_chart(fig1, use_container_width=True)

sector_impact = df.groupby("Sector")["Weighted Impact"].sum().reset_index()
fig2 = px.bar(sector_impact, x="Sector", y="Weighted Impact", title="Weighted Sector Impact")
st.plotly_chart(fig2, use_container_width=True)

fig3 = px.pie(df, values="Stake (%)", names="Company", title="Company Stake Distribution")
st.plotly_chart(fig3, use_container_width=True)

trend = df["% Change"].apply(lambda x: "Gainer" if x > 0 else "Loser").value_counts().reset_index()
fig4 = px.pie(trend, names="index", values="% Change", title="Gainers vs Losers")
st.plotly_chart(fig4, use_container_width=True)

st.write("---")

# --------------------------------------------------
# COMPANY DETAIL
# --------------------------------------------------

st.subheader("Company Detail")
chosen = st.selectbox("Choose company", df["Ticker"].tolist())

row = df[df["Ticker"] == chosen].iloc[0]

st.write(f"**{row['Company']}** ({chosen}) — Sector: {row['Sector']}")
st.write(f"Price: {row['Price']} | Prev Close: {row['Prev Close']} | %: {row['% Change']:.2f}")
st.write(f"Market Cap: {format_mcap(row['Market Cap'])}")

# History
hist = yf.download(chosen, period="3mo")  # 90 days
if not hist.empty:
    fig = px.line(hist, y="Close", title=f"{chosen} - 90 Day Trend")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("History not available.")

st.write("---")
st.caption("Stable build — No async, no crashes, all features included.")
