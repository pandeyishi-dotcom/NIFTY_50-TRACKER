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
    try:
        crores = float(x) / 1e7
        return f"{crores:,.2f} Cr"
    except Exception:
        return "N/A"

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
    try:
        r = requests.get(url, params=params, timeout=10)
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
        except:
            price, prev, pct = None, None, None

        return price, prev, pct
    except Exception:
        return None, None, None


def fetch_alpha_overview(ticker):
    if not ALPHA_KEY:
        return None, None

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "OVERVIEW",
        "symbol": ticker,
        "apikey": ALPHA_KEY
    }
    try:
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
    except Exception:
        return None, None

# --------------------------------------------------
# MAIN FETCH LOGIC (NO ASYNC)
# --------------------------------------------------

@st.cache_data(ttl=120)
def load_live():
    rows = []

    # bulk download prices; wrap in try so whole app doesn't crash
    try:
        prices = yf.download(
            tickers=NIFTY50_TICKERS,
            period="2d",
            interval="1d",
            group_by="ticker",
            threads=True,
            progress=False
        )
    except Exception:
        prices = pd.DataFrame()

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
            if isinstance(prices, pd.DataFrame) and t in prices.columns.get_level_values(0):
                p = prices[t]["Close"].dropna()
                if len(p) >= 1:
                    price = float(p.iloc[-1])
                if len(p) >= 2:
                    prev = float(p.iloc[-2])
        except Exception:
            pass

        # Try fast_info
        try:
            fi = yf.Ticker(t).fast_info
            if fi:
                mcap = fi.get("market_cap") or mcap
                sector = fi.get("sector") or sector
                company = fi.get("shortName") or company
                if price is not None:
                    source = "yahoo"
        except Exception:
            pass

        # Compute pct
        try:
            if price is not None and prev is not None and prev != 0:
                pct = ((price - prev) / prev) * 100
        except Exception:
            pct = None

        # Fallback to AlphaVantage price if missing
        if price is None:
            p, pc, pchg = fetch_alpha_global(t.replace(".NS", ""))
            if p is not None:
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
            "Sector": sector if sector else None,
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

def sanitize(df):
    cleaned = {}
    for col in df.columns:
        s = df[col]
        # if duplicate columns → a DF appears
        if isinstance(s, pd.DataFrame):
            for i, sub in enumerate(s.columns):
                name = f"{col}_{i}"
                cleaned[name] = s[sub].apply(_cell)
        else:
            cleaned[col] = s.apply(_cell)
    return pd.DataFrame(cleaned)

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

# Recompute missing pct where possible
mask = df["% Change"].isna() & df["Price"].notna() & df["Prev Close"].notna() & (df["Prev Close"] != 0)
df.loc[mask, "% Change"] = ((df.loc[mask, "Price"] - df.loc[mask, "Prev Close"]) /
                            df.loc[mask, "Prev Close"]) * 100

# Market Cap Calc
if df["Market Cap"].notna().sum() >= 1:
    med = df["Market Cap"].median()
    df["Market Cap Calc"] = df["Market Cap"].fillna(med)
else:
    df["Market Cap Calc"] = 1.0

# Stake
total_cap_calc = df["Market Cap Calc"].sum() if df["Market Cap Calc"].sum() != 0 else 1.0
df["Stake (%)"] = (df["Market Cap Calc"] / total_cap_calc) * 100

# Weighted Impact
df["Weighted Impact"] = (df["% Change"].fillna(0) * df["Stake (%)"]) / 100

# Rank
df["Rank"] = df["% Change"].rank(ascending=False, method="first").astype("Int64")

# Sort by mcap
df = df.sort_values("Market Cap Calc", ascending=False).reset_index(drop=True)

# --------------------------------------------------
# Summary Metrics
# --------------------------------------------------

c1, c2, c3 = st.columns(3)
try:
    c1.metric("Total Gainers Impact", f"{df[df['% Change']>0]['% Change'].sum():.2f}%")
except Exception:
    c1.metric("Total Gainers Impact", "N/A")
try:
    c2.metric("Total Losers Impact", f"{df[df['% Change']<0]['% Change'].sum():.2f}%")
except Exception:
    c2.metric("Total Losers Impact", "N/A")
try:
    c3.metric("Overall Impact", f"{df['Weighted Impact'].sum():.2f}%")
except Exception:
    c3.metric("Overall Impact", "N/A")

st.write("---")

# --------------------------------------------------
# Top Movers
# --------------------------------------------------

tg, tl = st.columns(2)

tg.subheader("Top 5 Gainers")
g = df.sort_values("% Change", ascending=False).head(5)
for _, r in g.iterrows():
    pct = r["% Change"] if pd.notna(r["% Change"]) else 0.0
    tg.write(f"**{r['Company']}** ({r['Ticker']}): `{pct:.2f}%` — MC: {format_mcap(r['Market Cap'])}")

tl.subheader("Top 5 Losers")
l = df.sort_values("% Change", ascending=True).head(5)
for _, r in l.iterrows():
    pct = r["% Change"] if pd.notna(r["% Change"]) else 0.0
    tl.write(f"**{r['Company']}** ({r['Ticker']}): `{pct:.2f}%` — MC: {format_mcap(r['Market Cap'])}")

st.write("---")

# --------------------------------------------------
# TABLE
# --------------------------------------------------

table = df.copy()
table["Market Cap"] = table["Market Cap"].apply(format_mcap)

safe_table = sanitize(table)

st.subheader("Company Performance")
st.dataframe(safe_table, use_container_width=True)
st.download_button("Download CSV", safe_table.to_csv(index=False).encode(), "nifty50.csv", mime="text/csv")

st.write("---")

# --------------------------------------------------
# CHARTS
# --------------------------------------------------

fig1 = px.bar(df.sort_values("% Change", ascending=False),
              x="Company", y="% Change", color="% Change",
              title="% Change by Company")
st.plotly_chart(fig1, use_container_width=True)

sector_impact = df.groupby(df["Sector"].fillna("Unknown"))["Weighted Impact"].sum().reset_index()
fig2 = px.bar(sector_impact, x="Sector", y="Weighted Impact", title="Weighted Sector Impact")
st.plotly_chart(fig2, use_container_width=True)

fig3 = px.pie(df, values="Stake (%)", names="Company", title="Company Stake Distribution")
st.plotly_chart(fig3, use_container_width=True)

# ---------- Corrected Gainers vs Losers pie ----------
trend_count = df["% Change"].apply(lambda x: "Gainer" if pd.notna(x) and x > 0 else "Loser").value_counts().reset_index()
trend_count.columns = ["Trend", "Count"]

if trend_count["Count"].sum() == 0:
    st.info("No gainers/losers data to show.")
else:
    fig4 = px.pie(
        trend_count,
        names="Trend",
        values="Count",
        title="Gainers vs Losers Ratio",
        color="Trend",
        color_discrete_map={"Gainer": "green", "Loser": "red"}
    )
    st.plotly_chart(fig4, use_container_width=True)

st.write("---")

# --------------------------------------------------
# COMPANY DETAIL
# --------------------------------------------------

st.subheader("Company Detail")
chosen = st.selectbox("Choose company", df["Ticker"].tolist())

row = df[df["Ticker"] == chosen].iloc[0] if not df[df["Ticker"] == chosen].empty else None

if row is not None:
    st.write(f"**{row.get('Company','-')}** ({chosen}) — Sector: {row.get('Sector') if pd.notna(row.get('Sector')) else 'Unknown'}")
    price_display = f"{row['Price']}" if pd.notna(row['Price']) else "N/A"
    prev_display = f"{row['Prev Close']}" if pd.notna(row['Prev Close']) else "N/A"
    pct_display = f"{row['% Change']:.2f}%" if pd.notna(row['% Change']) else "N/A"
    st.write(f"Price: {price_display} | Prev Close: {prev_display} | %: {pct_display}")
    st.write(f"Market Cap: {format_mcap(row['Market Cap'])}")
else:
    st.info("Selected company data not available.")

# History
try:
    hist = yf.download(chosen, period="3mo")  # 90 days
    if not hist.empty:
        fig = px.line(hist, y="Close", title=f"{chosen} - 90 Day Trend")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("History not available.")
except Exception:
    st.info("History not available (error fetching historical data).")

st.write("---")
st.caption("Stable build — No async, no crashes, all features included.")
