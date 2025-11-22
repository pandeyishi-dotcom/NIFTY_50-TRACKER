# app_debug.py — NIFTY50 dashboard with aggressive debugging and watchdogs
# Drop in place of app.py for fast diagnosis.
import os
import time
import math
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.express as px
import plotly.graph_objects as go

# -------------------------
# Setup
# -------------------------
st.set_page_config(page_title="NIFTY50 Debug Dashboard", layout="wide")
st.title("🐞 NIFTY50 — Debug Dashboard")

# Logging to file (so you can download it)
LOG_PATH = "debug_log.txt"
logging.basicConfig(filename=LOG_PATH, level=logging.DEBUG, filemode="w",
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("nifty_debug")

# Controls
st.sidebar.header("Debug controls")
MAX_TOTAL_SECONDS = st.sidebar.number_input("Max total load seconds", value=30, min_value=5, max_value=300, step=5)
PER_TICKER_TIMEOUT = st.sidebar.number_input("Per-ticker soft timeout (s)", value=6, min_value=1, max_value=60, step=1)
USE_ALPHA = st.sidebar.checkbox("Use AlphaVantage (enable if key present)", value=False)
ALPHA_KEY = os.environ.get("ALPHAVANTAGE_API_KEY") or (st.secrets.get("ALPHAVANTAGE_API_KEY") if "ALPHAVANTAGE_API_KEY" in st.secrets else None)
if USE_ALPHA and not ALPHA_KEY:
    st.sidebar.error("Alpha enabled but ALPHAVANTAGE_API_KEY not found in env/secrets")

st.sidebar.markdown("**Dev tips:**\n- Toggle Alpha to test speed.\n- Add `sectors_override.csv` to skip sector lookups.")

# A compact NIFTY subset to speed debugging by default
NIFTY_TEST = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "BHARTIARTL.NS","HINDUNILVR.NS","ITC.NS","KOTAKBANK.NS","LT.NS"
]

TICKERS = st.sidebar.multiselect("Tickers to load (top 50 shown by default) — keep small for debug",
                                 options=NIFTY_TEST, default=NIFTY_TEST)

if not TICKERS:
    st.warning("Select at least one ticker in the sidebar to run debug loader.")
    st.stop()

# -------------------------
# Helpers (safe)
# -------------------------
def safe_json(resp):
    try:
        return resp.json()
    except Exception:
        return None

def log_and_write(s, level="info"):
    now = datetime.utcnow().isoformat()
    entry = f"{now}  {s}"
    if level == "error":
        logger.error(s)
    else:
        logger.info(s)
    # stream to UI too
    st.write(s)

# Minimal alpha helpers (polite)
class AlphaLimiter:
    def __init__(self, calls_per_min=5):
        self.interval = 60.0 / max(1, calls_per_min)
        self.last = 0.0
    def wait(self):
        now = time.time()
        diff = now - self.last
        if diff < self.interval:
            time.sleep(self.interval - diff)
        self.last = time.time()

alpha_limiter = AlphaLimiter(5)

def alpha_global_quote(sym_no_ns: str):
    if not ALPHA_KEY:
        return None, None, None
    alpha_limiter.wait()
    url = "https://www.alphavantage.co/query"
    params = {"function":"GLOBAL_QUOTE","symbol":f"{sym_no_ns}.NS","apikey":ALPHA_KEY}
    try:
        r = requests.get(url, params=params, timeout=8)
        if r.status_code != 200:
            return None, None, None
        data = safe_json(r)
        g = data.get("Global Quote", {}) if data else {}
        p = g.get("05. price"); pc = g.get("08. previous close"); pct = g.get("10. change percent")
        try:
            return (float(p) if p else None, float(pc) if pc else None, float(pct.replace("%","")) if pct else None)
        except Exception:
            return None, None, None
    except Exception:
        return None, None, None

def nse_quote(sym_no_ns: str):
    url = f"https://www.nseindia.com/api/quote-equity?symbol={sym_no_ns}"
    headers = {"User-Agent":"Mozilla/5.0","Referer":"https://www.nseindia.com/"}
    s = requests.Session()
    try:
        s.get("https://www.nseindia.com", headers=headers, timeout=4)
        r = s.get(url, headers=headers, timeout=6)
        if r.status_code != 200:
            return None, None, None, None
        data = safe_json(r)
        if not data:
            return None, None, None, None
        pi = data.get("priceInfo") or {}
        secinfo = data.get("securityInfo") or {}
        last = pi.get("lastPrice") or pi.get("last")
        prev = pi.get("previousClose")
        pchg = pi.get("pChange")
        sector = secinfo.get("industry") or secinfo.get("industryType")
        return last, prev, pchg, sector
    except Exception as ex:
        logger.debug(f"nse_quote error for {sym_no_ns}: {ex}")
        return None, None, None, None

# -------------------------
# Debug loader with watchdog
# -------------------------
def load_debug(tickers, max_total_seconds=30, per_ticker_timeout=6, use_alpha=False):
    rows = []
    start_all = time.time()
    # try bulk download but with threads=False to avoid some hangs
    try:
        bulk = yf.download(tickers=tickers, period="2d", interval="1d", group_by="ticker", threads=False, progress=False, auto_adjust=False)
        log_and_write("Bulk yfinance attempt completed (may be empty).")
    except Exception as e:
        bulk = pd.DataFrame()
        log_and_write(f"Bulk yfinance failed: {e}", level="error")

    total = len(tickers)
    progress = st.progress(0)
    for i, t in enumerate(tickers):
        ticker_start = time.time()
        elapsed = time.time() - start_all
        if elapsed > max_total_seconds:
            log_and_write(f"Watchdog: total time exceeded {max_total_seconds}s — stopping early and returning partial data.", level="error")
            break

        st.write(f"---\nFetching {t} ({i+1}/{total}) — elapsed {int(elapsed)}s")
        logger.info(f"Starting {t}")
        company = t.replace(".NS","")
        price = prev = pct = None
        sector = None
        mcap = None
        source = "none"

        # bulk quick read
        try:
            if isinstance(bulk, pd.DataFrame) and t in bulk.columns.get_level_values(0):
                closes = bulk[t]["Close"].dropna()
                if len(closes) >= 1:
                    price = float(closes.iloc[-1])
                if len(closes) >= 2:
                    prev = float(closes.iloc[-2])
                source = source or "yahoo-bulk"
                log_and_write(f"{t}: got price from yahoo-bulk -> {price}")
        except Exception as e:
            log_and_write(f"{t}: bulk parse failure: {e}", level="error")

        # yfinance fast_info
        try:
            finfo = yf.Ticker(t).fast_info
            if finfo:
                mcap = finfo.get("market_cap") or finfo.get("marketCap") or None
                sector = finfo.get("sector") or finfo.get("Sector") or sector
                company = finfo.get("shortName") or company
                source = source or "yahoo-fast"
                log_and_write(f"{t}: fast_info ok (sector:{sector}, mcap:{mcap})")
        except Exception as e:
            log_and_write(f"{t}: fast_info failed: {e}")

        # compute pct quickly if possible
        try:
            if price is not None and prev is not None and prev != 0:
                pct = ((price - prev) / prev) * 100
        except Exception:
            pct = None

        # alpha fallback for price (optional)
        if price is None and use_alpha:
            try:
                p, pc, pchg = alpha_global_quote(t.replace(".NS",""))
                if p is not None:
                    price, prev, pct = p, pc, pchg
                    source = source or "alpha"
                    log_and_write(f"{t}: alpha price fetched -> {price}")
            except Exception as e:
                log_and_write(f"{t}: alpha price error: {e}")

        # nse fallback
        if price is None:
            try:
                n_price, n_prev, n_pchg, n_sector = nse_quote(t.replace(".NS",""))
                if n_price is not None:
                    price, prev, pct = n_price, n_prev, n_pchg
                    source = source or "nse"
                    log_and_write(f"{t}: nse price fetched -> {price}")
                if n_sector and not sector:
                    sector = n_sector
                    log_and_write(f"{t}: nse sector -> {sector}")
            except Exception as e:
                log_and_write(f"{t}: nse error: {e}")

        rows.append({"Ticker": t, "Company": company, "Sector": sector, "Price": price,
                     "Prev Close": prev, "% Change": pct, "Market Cap": mcap, "Source": source})

        # progress
        progress.progress(int(((i+1)/total) * 100))
        ticker_elapsed = time.time() - ticker_start
        if ticker_elapsed > per_ticker_timeout:
            log_and_write(f"{t}: took {int(ticker_elapsed)}s (> {per_ticker_timeout}s). Continuing...", level="error")

    return pd.DataFrame(rows)

# -------------------------
# Run debug load and show results
# -------------------------
with st.spinner("Debug: loading data with watchdog..."):
    df = load_debug(TICKERS, max_total_seconds=MAX_TOTAL_SECONDS, per_ticker_timeout=PER_TICKER_TIMEOUT, use_alpha=USE_ALPHA)

if df is None or df.empty:
    st.error("No rows returned by debug loader. Check network / DNS.")
    st.stop()

# show a compact diagnostics table
st.subheader("Debug diagnostics (first rows)")
diag = df[["Ticker","Company","Source","Price","Prev Close","Market Cap","Sector","% Change"]].copy()
st.dataframe(diag.fillna("N/A"), use_container_width=True)

# write helpful hints based on results
n_missing_price = df["Price"].isna().sum()
n_from_nse = (df["Source"] == "nse").sum()
n_from_yahoo = (df["Source"].str.contains("yahoo", na=False)).sum()
st.info(f"Tickers with missing price: {n_missing_price}. From NSE: {n_from_nse}. From Yahoo: {n_from_yahoo}.")
log_and_write(f"Summary: missing_price={n_missing_price}, from_nse={n_from_nse}, from_yahoo={n_from_yahoo}")

# provide log download
if os.path.exists(LOG_PATH):
    with open(LOG_PATH, "r") as f:
        txt = f.read()
    st.download_button("Download debug log", txt, file_name=LOG_PATH)

st.write("---")
st.caption("This debug run uses limited tickers and shorter watchdogs to isolate failures. Paste the diagnostic table or download the log and share it here; I'll parse it and produce a final production patch.")
