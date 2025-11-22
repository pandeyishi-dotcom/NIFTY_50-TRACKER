# app.py — Enterprise-grade NIFTY50 dashboard (Option C)
"""
Requirements (pip):
 streamlit
 pandas
 numpy
 yfinance
 plotly
 requests
 aiohttp
 aioredis
 python-dotenv

Before running:
 - export ALPHAVANTAGE_API_KEY="your_key"
 - export REDIS_URL="redis://:password@host:port/0"   (optional, recommended)
 - optional: put sector overrides in sectors_override.csv with columns: Ticker, Sector

Notes:
 - This file uses asyncio + aiohttp for background fetch concurrency (AlphaVantage + NSE).
 - AlphaVantage token bucket ensures we respect 5 calls/min free tier.
 - If Redis is available, metadata & histories are cached there; otherwise a simple in-memory cache is used.
"""

import os
import io
import math
import time
import json
import asyncio
from datetime import datetime, timedelta
from functools import partial

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import requests
import aiohttp

# Optional: Redis async client
try:
    import aioredis
except Exception:
    aioredis = None

# Load .env locally if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="NIFTY 50 — Enterprise Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("🚀 NIFTY 50 — Enterprise-grade Dashboard (Optimized)")

ALPHAVANTAGE_API_KEY = (
    os.environ.get("ALPHAVANTAGE_API_KEY")
    or (st.secrets["ALPHAVANTAGE_API_KEY"] if "ALPHAVANTAGE_API_KEY" in st.secrets else None)
)

REDIS_URL = os.environ.get("REDIS_URL") or None  # e.g. redis://:passwd@host:6379/0

# Fallback order can be changed: 'yahoo', 'alphavantage', 'nse'
FALLBACK_ORDER = ["yahoo", "alphavantage", "nse"]

# developer screenshot (optional)
DEBUG_IMG_PATH = "/mnt/data/97d8d668-f6ab-412a-969b-5eaac6b33716.png"

# core tickers
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
# Utilities & caches
# ---------------------------

# Async Redis wrapper if available
redis = None
if REDIS_URL and aioredis:
    try:
        # create connection lazily in async functions, keep URL here
        REDIS_ENABLED = True
    except Exception:
        REDIS_ENABLED = False
else:
    REDIS_ENABLED = False

# In-memory caches as fallback (thread-safe primitives not required for single-process Streamlit)
_mem_cache = {}

def mem_set(key, value, ttl_seconds=300):
    expire = time.time() + ttl_seconds
    _mem_cache[key] = (value, expire)

def mem_get(key):
    v = _mem_cache.get(key)
    if not v:
        return None
    val, expire = v
    if time.time() > expire:
        _mem_cache.pop(key, None)
        return None
    return val

# small helper to format market cap
def format_market_cap_display(mc):
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

def safe_json(resp):
    try:
        return resp.json()
    except Exception:
        return None

# Sanitize DataFrame for Streamlit/Arrow
def sanitize_for_streamlit(df_in):
    df = df_in.copy()
    df = df.where(pd.notnull(df), None)
    for col in df.columns:
        col_dtype = str(df[col].dtype)
        if col_dtype.startswith("Int"):
            df[col] = df[col].apply(lambda x: int(x) if (x is not None and not (isinstance(x, float) and math.isnan(x))) else None)
        elif pd.api.types.is_float_dtype(df[col].dtype):
            df[col] = df[col].apply(lambda x: float(x) if x is not None else None)
        elif df[col].dtype == object:
            def safe_obj(v):
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
            df[col] = df[col].apply(safe_obj)
    return df

# ---------------------------
# Async helpers for AlphaVantage & NSE (rate-limited)
# ---------------------------
# Token bucket for AlphaVantage free tier: 5 requests per minute (adjustable)
class AlphaVantageRateLimiter:
    def __init__(self, calls_per_minute=5):
        self.capacity = calls_per_minute
        self.tokens = calls_per_minute
        self.fill_interval = 60.0 / calls_per_minute
        self.last_time = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_time
            # refill
            refill = int(elapsed / self.fill_interval)
            if refill > 0:
                self.tokens = min(self.capacity, self.tokens + refill)
                self.last_time = now
            if self.tokens > 0:
                self.tokens -= 1
                return True
            # wait until next token
            await asyncio.sleep(self.fill_interval)
            # after sleep, consume one
            self.tokens = max(0, self.tokens - 1)
            self.last_time = time.monotonic()
            return True

# create a global limiter
AV_LIMITER = AlphaVantageRateLimiter(calls_per_minute=5)

async def alphavantage_global_quote(session, symbol, api_key, timeout=10):
    """Async global quote (price). symbol without .NS"""
    if not api_key:
        return None, None, None, None, "alphavantage_no_key"
    await AV_LIMITER.acquire()
    base = "https://www.alphavantage.co/query"
    params = {"function": "GLOBAL_QUOTE", "symbol": symbol + ".NS", "apikey": api_key}
    try:
        async with session.get(base, params=params, timeout=timeout) as r:
            data = await r.json()
            g = data.get("Global Quote") if data else {}
            if not g:
                return None, None, None, None, "alphavantage_no_data"
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

async def alphavantage_overview(session, symbol, api_key, timeout=10):
    """Async company overview for sector & market cap"""
    if not api_key:
        return None, None, "alphavantage_no_key"
    await AV_LIMITER.acquire()
    base = "https://www.alphavantage.co/query"
    params = {"function": "OVERVIEW", "symbol": symbol + ".NS", "apikey": api_key}
    try:
        async with session.get(base, params=params, timeout=timeout) as r:
            data = await r.json()
            sector = data.get("Sector")
            mcap = None
            if data.get("MarketCapitalization"):
                try:
                    mcap = float(data.get("MarketCapitalization"))
                except Exception:
                    mcap = None
            return sector, mcap, "alphavantage_overview_ok"
    except Exception:
        return None, None, "alphavantage_error"

async def alphavantage_time_series(session, symbol, api_key, start_date, end_date, timeout=15):
    """Async TIME_SERIES_DAILY_ADJUSTED (full). Returns pandas DataFrame indexed by Date with Close column."""
    if not api_key:
        return pd.DataFrame()
    # use limiter
    await AV_LIMITER.acquire()
    base = "https://www.alphavantage.co/query"
    params = {"function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": symbol + ".NS", "outputsize": "full", "apikey": api_key}
    try:
        async with session.get(base, params=params, timeout=timeout) as r:
            data = await r.json()
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

# NSE async call (best-effort)
async def nse_quote(session, symbol, timeout=8):
    url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
        "Accept": "application/json, text/javascript, */*; q=0.01",
    }
    try:
        # sometimes NSE needs initial cookie; do a quick GET to base first
        async with session.get("https://www.nseindia.com", headers=headers, timeout=timeout) as init_r:
            await init_r.text()
        async with session.get(url, headers=headers, timeout=timeout) as r:
            data = await r.json()
            price_info = data.get("priceInfo") or {}
            last_price = price_info.get("lastPrice") or price_info.get("last")
            prev_close = price_info.get("previousClose")
            pchange = price_info.get("pChange")
            secinfo = data.get("securityInfo") or {}
            sector_nse = secinfo.get("industry") or secinfo.get("industryType")
            return last_price, prev_close, pchange, None, sector_nse, "nse_ok"
    except Exception:
        return None, None, None, None, None, "nse_error"

# ---------------------------
# Caching helpers (async if Redis available)
# ---------------------------
async def get_redis():
    if not REDIS_ENABLED:
        return None
    global redis
    if redis:
        return redis
    try:
        redis = await aioredis.from_url(REDIS_URL)
        return redis
    except Exception:
        return None

async def cache_get(key):
    # try redis first
    if REDIS_ENABLED and aioredis:
        r = await get_redis()
        if r:
            try:
                val = await r.get(key)
                if val:
                    return json.loads(val)
            except Exception:
                pass
    return mem_get(key)

async def cache_set(key, value, ttl=300):
    if REDIS_ENABLED and aioredis:
        r = await get_redis()
        if r:
            try:
                await r.set(key, json.dumps(value), ex=ttl)
                return True
            except Exception:
                pass
    mem_set(key, value, ttl)
    return True

# ---------------------------
# High-level loader that uses concurrency
# ---------------------------
# We'll use yfinance bulk for price wholesale (fast), async for per-ticker metadata & fallbacks.
async def fetch_metadata_concurrent(tickers, fallback_order, av_key):
    """
    returns DataFrame with columns:
    Ticker, Company, Sector, Price, Prev Close, % Change, Market Cap (display), Market Cap Calc, Source
    """
    # 1) bulk prices via yfinance (synchronous) — use thread executor to avoid blocking loop
    loop = asyncio.get_event_loop()
    prices = await loop.run_in_executor(None, partial(yf.download,
                                                      " ".join(tickers),
                                                      period="2d",
                                                      interval="1d",
                                                      group_by="ticker",
                                                      threads=True,
                                                      progress=False,
                                                      auto_adjust=False))

    # prepare list of tasks for metadata fetches (AlphaVantage & NSE)
    async with aiohttp.ClientSession() as session:
        tasks = []
        for t in tickers:
            tasks.append(fetch_single_ticker(session, t, prices, fallback_order, av_key))
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # results is list of dicts or exceptions
    rows = []
    for res in results:
        if isinstance(res, Exception):
            # fallback safe row
            rows.append({
                "Ticker": "UNKNOWN",
                "Company": None,
                "Sector": None,
                "Price": None,
                "Prev Close": None,
                "% Change": None,
                "Market Cap": None,
                "Source": "error"
            })
        else:
            rows.append(res)
    df = pd.DataFrame(rows)
    return df

async def fetch_single_ticker(session, ticker, prices_df, fallback_order, av_key):
    """
    For each ticker:
     - try to get price & prev_close from prices_df (yfinance bulk)
     - try yahoo fast_info (sync in executor) for market cap & name
     - if price missing, try AlphaVantage (global quote)
     - if still missing, try NSE
     - try AlphaVantage Overview for sector & market cap (rate-limited)
    """
    symbol = ticker.replace(".NS", "")
    price = None
    prev_close = None
    pct_change = None
    market_cap = None
    company = symbol
    sector = None
    source = "missing"

    # try yfinance bulk price
    try:
        if isinstance(prices_df, pd.DataFrame) and not prices_df.empty:
            # group_by ticker -> multiindex columns
            try:
                if ticker in prices_df.columns.get_level_values(0):
                    s = prices_df[ticker]["Close"].dropna()
                    if len(s) >= 1:
                        price = float(s.iloc[-1])
                    if len(s) >= 2:
                        prev_close = float(s.iloc[-2])
                else:
                    if "Close" in prices_df.columns:
                        s = prices_df["Close"].dropna()
                        if len(s) >= 1:
                            price = float(s.iloc[-1])
                        if len(s) >= 2:
                            prev_close = float(s.iloc[-2])
            except Exception:
                pass
    except Exception:
        pass

    # try yahoo fast_info for meta (use thread executor)
    try:
        loop = asyncio.get_event_loop()
        mc, sec, sname, status = await loop.run_in_executor(None, partial(get_yahoo_fast_info_sync, ticker))
        if mc:
            market_cap = mc
        if sec:
            sector = sec
        if sname:
            company = sname
    except Exception:
        pass

    # compute pct if possible
    if price is not None and prev_close is not None and prev_close != 0:
        try:
            pct_change = ((price - prev_close) / prev_close) * 100
            source = source if source != "missing" else "yahoo"
        except Exception:
            pct_change = None

    # If price missing, iterate fallbacks
    if (price is None or (isinstance(price, float) and math.isnan(price))) and "alphavantage" in fallback_order:
        p, pc, pchg, mc_av, status_av = await alphavantage_global_quote(session, symbol, av_key)
        if p is not None:
            price = p
            prev_close = pc
            pct_change = pchg
            source = "alphavantage"
    if (price is None or (isinstance(price, float) and math.isnan(price))) and "nse" in fallback_order:
        p_nse, prev_nse, pchg_nse, mc_nse, sector_nse, status_nse = await nse_quote(session, symbol)
        if p_nse is not None:
            price = p_nse
            prev_close = prev_nse
            pct_change = pchg_nse
            if sector_nse:
                sector = sector_nse
            source = "nse"

    # get alpha overview for missing market cap/sector if available
    if (market_cap is None or sector is None) and av_key:
        sec_av, mc_overview, status_over = await alphavantage_overview(session, symbol, av_key)
        if sec_av and sector is None:
            sector = sec_av
        if mc_overview and market_cap is None:
            market_cap = mc_overview

    # ensure numeric coercion
    try:
        market_cap = float(market_cap) if market_cap is not None else None
    except Exception:
        market_cap = None
    try:
        price = float(price) if price is not None else None
    except Exception:
        price = None
    try:
        prev_close = float(prev_close) if prev_close is not None else None
    except Exception:
        prev_close = None
    try:
        pct_change = float(pct_change) if pct_change is not None else None
    except Exception:
        pct_change = None

    return {
        "Ticker": ticker,
        "Company": company,
        "Sector": sector,
        "Price": price,
        "Prev Close": prev_close,
        "% Change": pct_change,
        "Market Cap": market_cap,
        "Source": source
    }

# helper: sync wrapper for yahoo fast_info (to be used inside executor)
def get_yahoo_fast_info_sync(ticker):
    try:
        fi = yf.Ticker(ticker).fast_info or {}
        mc = fi.get("market_cap") or fi.get("marketCap") or None
        sector = fi.get("sector") or None
        shortName = fi.get("shortName") or None
        return mc, sector, shortName, "yahoo_fast_ok"
    except Exception:
        return None, None, None, "yahoo_fast_error"

# ---------------------------
# High-level wrapper to call fetch_metadata_concurrent from sync code
# ---------------------------
def load_live_data_optimized(ticker_list, fallback_order=None, av_key=None):
    # first try cache (simple key)
    cache_key = f"live_meta:{','.join(sorted(ticker_list))}:{','.join(fallback_order)}"
    cached = mem_get(cache_key)
    if cached:
        return pd.DataFrame(cached)
    # run async fetch
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        df = loop.run_until_complete(fetch_metadata_concurrent(ticker_list, fallback_order or FALLBACK_ORDER, av_key))
    finally:
        try:
            loop.close()
        except Exception:
            pass
    # store in mem cache
    mem_set(cache_key, df.to_dict(orient="records"), ttl_seconds=120)
    return df

# ---------------------------
# Historical (sync wrapper uses async alphavantage fallback)
# ---------------------------
def fetch_history_for_tickers_optimized(tickers, start_date, end_date):
    histories = {}
    # try yfinance in bulk for speed: we will call per-ticker if yfinance empty
    for t in tickers:
        try:
            hist = yf.download(t, start=start_date, end=end_date + timedelta(days=1), progress=False, threads=False)
            if not hist.empty:
                histories[t] = hist[["Close"]].rename(columns={"Close": "Close"})
                continue
        except Exception:
            pass
        # fallback to async alphavantage (run per ticker)
        if ALPHAVANTAGE_API_KEY:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                with aiohttp.ClientSession() as session:
                    # use helper that runs once; simpler to call synchronous wrapper
                    df = loop.run_until_complete(alphavantage_time_series(session, t.replace(".NS", ""), ALPHAVANTAGE_API_KEY, start_date, end_date))
                    histories[t] = df if df is not None else pd.DataFrame()
            except Exception:
                histories[t] = pd.DataFrame()
            finally:
                try:
                    loop.close()
                except Exception:
                    pass
        else:
            histories[t] = pd.DataFrame()
    return histories

# ---------------------------
# Sector override loader (optional CSV mapping)
# ---------------------------
def load_sector_overrides(path="sectors_override.csv"):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            # expect columns: Ticker, Sector
            mapping = dict(zip(df["Ticker"].astype(str).str.upper(), df["Sector"]))
            return mapping
        except Exception:
            return {}
    return {}

SECTOR_OVERRIDE = load_sector_overrides()

# ---------------------------
# Streamlit UI & app logic
# ---------------------------

# Sidebar
st.sidebar.header("Controls")
with st.sidebar:
    if st.checkbox("Show debug screenshot (dev)", value=False):
        try:
            st.image(DEBUG_IMG_PATH, caption="Debug image", use_column_width=True)
        except Exception:
            st.info("No debug image found.")
    st.write("AlphaVantage key present:", bool(ALPHAVANTAGE_API_KEY))
    st.write("Redis enabled:", bool(REDIS_ENABLED))
    search = st.text_input("Search company / ticker", value="")
    top_n = st.slider("Top N movers (abs % change)", min_value=5, max_value=50, value=10)
    marketcap_min = st.number_input("Minimum market cap (crore INR)", value=0.0, step=100.0)
    refresh = st.button("Refresh data (bypass cache)")

# Load live data (optimized)
with st.spinner("Fetching optimized live data..."):
    df_live = load_live_data_optimized(NIFTY50_TICKERS, fallback_order=FALLBACK_ORDER, av_key=ALPHAVANTAGE_API_KEY)

if df_live is None or df_live.empty:
    st.error("No live data available. Check APIs/network.")
    st.stop()

# Convert to DataFrame and apply overrides
df = df_live.copy()
# apply sector override if available
if SECTOR_OVERRIDE:
    df["Ticker_upper"] = df["Ticker"].astype(str).str.upper()
    df["Sector"] = df.apply(lambda r: SECTOR_OVERRIDE.get(r["Ticker_upper"], r["Sector"]), axis=1)
    df = df.drop(columns=["Ticker_upper"])

# sanitize numerics
df["Market Cap"] = pd.to_numeric(df["Market Cap"], errors="coerce")
df["% Change"] = pd.to_numeric(df["% Change"], errors="coerce")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Prev Close"] = pd.to_numeric(df["Prev Close"], errors="coerce")

# Market Cap Calc (for weighting)
if df["Market Cap"].notna().sum() >= 1:
    med = df.loc[df["Market Cap"].notna(), "Market Cap"].median()
    df["Market Cap Calc"] = df["Market Cap"].fillna(med)
else:
    df["Market Cap Calc"] = 1.0

# compute % Change when missing but price & prev close present
mask_compute = df["% Change"].isna() & df["Price"].notna() & df["Prev Close"].notna() & (df["Prev Close"] != 0)
df.loc[mask_compute, "% Change"] = ((df.loc[mask_compute, "Price"] - df.loc[mask_compute, "Prev Close"]) / df.loc[mask_compute, "Prev Close"]) * 100

# Stake & weighted impact
total_cap = df["Market Cap Calc"].sum()
df["Stake (%)"] = (df["Market Cap Calc"] / total_cap) * 100 if total_cap != 0 else 0.0
df["Weighted Impact"] = (df["% Change"].fillna(0) * df["Stake (%)"]) / 100
df["Trend"] = np.where(df["% Change"] > 0, "Gainer", "Loser")
df["Color"] = df["% Change"].apply(lambda x: "green" if x > 0 else "red")
df["Abs Change"] = df["% Change"].abs().fillna(0)

# Rank with nullable Int type
df["Rank"] = df["% Change"].rank(ascending=False, method="first")
df["Rank"] = df["Rank"].astype("Int64")

# Default sorting by Market Cap Calc desc
df = df.sort_values(by="Market Cap Calc", ascending=False).reset_index(drop=True)

# Filters
if search:
    mask = df["Company"].str.contains(search, case=False, na=False) | df["Ticker"].str.contains(search, case=False, na=False)
    df = df.loc[mask]
# Sector filter (include Unknown)
sectors = ["All"] + sorted(df["Sector"].fillna("Unknown").unique().tolist())
sector_sel = st.sidebar.selectbox("Sector filter", options=sectors, index=0)
if sector_sel != "All":
    if sector_sel == "Unknown":
        df = df.loc[df["Sector"].isna()]
    else:
        df = df.loc[df["Sector"] == sector_sel]

if marketcap_min and marketcap_min > 0:
    df = df.loc[df["Market Cap Calc"] >= (marketcap_min * 1e7)]

# Summary metrics
total_positive = df.loc[df["% Change"] > 0, "% Change"].sum()
total_negative = df.loc[df["% Change"] < 0, "% Change"].sum()
overall_impact = df["Weighted Impact"].sum()

c1, c2, c3, c4 = st.columns([1.5, 1.5, 1.5, 1])
c1.metric("Total Gainers Impact", f"{total_positive:.2f} %")
c2.metric("Total Losers Impact", f"{total_negative:.2f} %")
c3.metric("Overall Weighted Impact", f"{overall_impact:.2f} %")
c4.metric("Companies Shown", f"{len(df)}")

st.write("---")

# Top movers (formatted)
top_gainers = df.sort_values(by="% Change", ascending=False).head(5)
top_losers = df.sort_values(by="% Change", ascending=True).head(5)
g1, g2 = st.columns(2)
with g1:
    st.subheader("Top 5 Gainers")
    lines = []
    for _, r in top_gainers.iterrows():
        pct = r["% Change"] if pd.notna(r["% Change"]) else 0.0
        mc_disp = format_market_cap_display(r["Market Cap"])
        lines.append(f"**{r['Company']}** ({r['Ticker']}) — `{pct:.2f}%` — Source: {r['Source']} — MarketCap: **{mc_disp}**")
    st.markdown("<br>".join(lines), unsafe_allow_html=True)
with g2:
    st.subheader("Top 5 Losers")
    lines = []
    for _, r in top_losers.iterrows():
        pct = r["% Change"] if pd.notna(r["% Change"]) else 0.0
        mc_disp = format_market_cap_display(r["Market Cap"])
        lines.append(f"**{r['Company']}** ({r['Ticker']}) — `{pct:.2f}%` — Source: {r['Source']} — MarketCap: **{mc_disp}**")
    st.markdown("<br>".join(lines), unsafe_allow_html=True)

st.write("---")

# Table with sanitized data for display
table_df = df[["Rank", "Ticker", "Company", "Sector", "Stake (%)", "Price", "% Change", "Weighted Impact", "Market Cap", "Market Cap Calc", "Source"]].copy()
table_df["Market Cap Display"] = table_df["Market Cap"].apply(format_market_cap_display)
display_table = table_df.drop(columns=["Market Cap Calc"]).rename(columns={"Market Cap Display": "Market Cap"})

# Sanitize for Streamlit / Arrow
try:
    display_table_clean = sanitize_for_streamlit(display_table)
    st.download_button("Download filtered table as CSV", data=display_table_clean.to_csv(index=False).encode(), file_name="nifty_filtered.csv", mime="text/csv")
    st.subheader("Company Performance (sorted by market cap)")
    st.dataframe(display_table_clean, use_container_width=True)
except Exception as e:
    st.error("Table rendering failed — showing debug info.")
    st.write("Error:", e)
    dtype_report = pd.DataFrame({
        "column": display_table.columns.astype(str),
        "dtype": [str(display_table[col].dtype) for col in display_table.columns],
        "has_nested": [display_table[col].apply(lambda x: isinstance(x, (list, dict, tuple, set, pd.Series, pd.DataFrame))).any() for col in display_table.columns]
    })
    st.dataframe(dtype_report, use_container_width=True)
    st.table(display_table.head(10))

st.write("---")

# Mini sparklines (yfinance -> AlphaVantage fallback via async)
def make_sparkline_fig_from_series(series):
    fig = go.Figure(go.Scatter(x=series.index, y=series.values, mode="lines", line=dict(width=1)))
    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=60, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig

with st.expander("Mini historical charts (last 30 days)"):
    view_tickers = display_table["Ticker"].tolist()[:12]
    end = datetime.now().date()
    start = end - timedelta(days=30)
    histories = fetch_history_for_tickers_optimized(view_tickers, start, end)
    cols = st.columns(3)
    for i, t in enumerate(view_tickers):
        row = display_table.loc[display_table["Ticker"] == t].iloc[0]
        col = cols[i % 3]
        with col:
            st.write(f"**{row['Company']}** — {t} (Source: {row['Source']})")
            hist = histories.get(t)
            if hist is not None and not hist.empty:
                st.plotly_chart(make_sparkline_fig_from_series(hist["Close"]), use_container_width=True)
            else:
                st.info("Mini history not available")

st.write("---")

# Charts
fig1 = px.bar(df.sort_values(by="% Change", ascending=False), x="Company", y="% Change", color="Trend", text="Rank", title="% Change by Company")
fig1.update_traces(textposition="outside")
st.plotly_chart(fig1, use_container_width=True)

# sector impact
sector_impact = df.groupby(df["Sector"].fillna("Unknown"))["Weighted Impact"].sum().reset_index().rename(columns={"Sector":"Sector","Weighted Impact":"Weighted Impact"})
if sector_impact.shape[0] <= 1:
    st.warning("Sector metadata is limited; sector breakdown may be unavailable.")
else:
    fig2 = px.bar(sector_impact.sort_values(by="Weighted Impact"), x="Sector", y="Weighted Impact", color="Weighted Impact", title="Weighted Sector Impact", color_continuous_scale=px.colors.diverging.RdYlGn)
    st.plotly_chart(fig2, use_container_width=True)

fig3 = px.pie(df, values="Stake (%)", names="Company", title="Company Stake Distribution in NIFTY 50")
st.plotly_chart(fig3, use_container_width=True)

trend_count = df["Trend"].value_counts().reset_index()
trend_count.columns = ["Trend","Count"]
fig4 = px.pie(trend_count, names="Trend", values="Count", title="Gainers vs Losers Ratio", color="Trend", color_discrete_map={"Gainer": "green", "Loser":"red"})
st.plotly_chart(fig4, use_container_width=True)

with st.expander("Sector vs Weighted Impact heatmap"):
    if sector_impact.shape[0] <= 1:
        st.info("Not enough sector diversity to build heatmap.")
    else:
        heat = px.imshow(sector_impact[["Weighted Impact"]].T, labels=dict(x="Sector", y="Metric", color="Weighted Impact"), x=sector_impact["Sector"].tolist(), y=["Weighted Impact"], aspect="auto", title="Sector Heatmap")
        st.plotly_chart(heat, use_container_width=True)

st.write("---")

# Historical comparison (multi-company)
st.subheader("Historical Comparison — Multi Company")
col_a, col_b, col_c = st.columns([2,2,1])
with col_a:
    selected_companies = st.multiselect("Select tickers for comparison", options=df["Ticker"].tolist(), default=df["Ticker"].tolist()[:3])
with col_b:
    start_date = st.date_input("Start date", value=datetime.now().date() - timedelta(days=365))
    end_date = st.date_input("End date", value=datetime.now().date())
with col_c:
    normalize = st.checkbox("Normalize (index to 100)", value=True)

if selected_companies:
    histories = fetch_history_for_tickers_optimized(selected_companies, start_date, end_date)
    fig_hist = go.Figure()
    any_data = False
    for t in selected_companies:
        h = histories.get(t)
        if h is None or h.empty:
            continue
        any_data = True
        y = h["Close"]
        if normalize:
            y = (y / y.iloc[0]) * 100
        fig_hist.add_trace(go.Scatter(x=h.index, y=y, name=t))
    if any_data:
        fig_hist.update_layout(title="Historical Close (Normalized)" if normalize else "Historical Close", xaxis_title="Date", yaxis_title="Indexed / Price")
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("No historical data found for the selected tickers & date range (tried yfinance + AlphaVantage).")

st.write("---")

# Company Detail
st.subheader("Company Detail")
ticker_choice = st.selectbox("Choose a ticker", options=df["Ticker"].tolist())
if ticker_choice:
    row = df.loc[df["Ticker"] == ticker_choice].iloc[0]
    mc_disp = format_market_cap_display(row["Market Cap"])
    st.write(f"**{row['Company']}** ({row['Ticker']}) — Sector: {row['Sector'] if pd.notna(row['Sector']) else 'Unknown'}")
    st.write(f"Price: {row['Price'] if pd.notna(row['Price']) else 'N/A'}, Prev Close: {row['Prev Close'] if pd.notna(row['Prev Close']) else 'N/A'}, % Change: {row['% Change'] if pd.notna(row['% Change']) else 0:.2f}%")
    st.write(f"Market Cap: {mc_disp}, Stake: {row['Stake (%)']:.2f}%, Data source: {row['Source']}")
    # 90d history
    hist90 = fetch_history_for_tickers_optimized([ticker_choice], datetime.now().date()-timedelta(days=90), datetime.now().date()).get(ticker_choice)
    if hist90 is not None and not hist90.empty:
        st.plotly_chart(px.line(hist90, y="Close", title=f"{row['Ticker']} — Last 90 days Close"), use_container_width=True)
        st.download_button("Download 90d CSV", data=hist90.to_csv().encode(), file_name=f"{ticker_choice}_90d.csv", mime="text/csv")
    else:
        st.info("90-day history not available (tried yfinance + AlphaVantage).")

st.write("---")
st.caption("Enterprise-grade: async AlphaVantage with token-bucket, Redis optional, market-cap-driven sorting and robust fallbacks. Want Redis & background refresh next?")

# EOF
