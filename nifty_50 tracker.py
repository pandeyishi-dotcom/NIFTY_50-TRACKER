# app.py — NIFTY50 Option-C Hybrid (final, non-freezing)
# Hybrid sector resolution: local CSV -> Alpha overview (only when missing)
# Data fallback order (per-ticker): yfinance (price) -> Alpha price -> NSE
# No yfinance .info / .fast_info usage. All external calls are timeout-protected.

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

# -------------------------
# Config
# -------------------------
st.set_page_config(page_title="NIFTY50 — Hybrid (Option C)", layout="wide")
st.title("📈 NIFTY50 — Hybrid (Option C) Stable Dashboard")

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

SECTOR_OVERRIDE_FILE = "sectors_override.csv"

# -------------------------
# Safe secrets
# -------------------------
def get_alpha_key():
    k = os.environ.get("ALPHAVANTAGE_API_KEY")
    if k:
        return k
    try:
        if hasattr(st, "secrets") and st.secrets is not None:
            return st.secrets.get("ALPHAVANTAGE_API_KEY")
    except Exception:
        return None
    return None

ALPHA_KEY = get_alpha_key()

# -------------------------
# Utilities
# -------------------------
def safe_json(resp):
    try:
        return resp.json()
    except Exception:
        return None

def format_mcap(mc):
    if mc is None or pd.isna(mc):
        return "N/A"
    try:
        return f"{float(mc)/1e7:,.2f} Cr"
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

def sanitize(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = {}
    for col in df.columns:
        s = df[col]
        if isinstance(s, pd.DataFrame):
            for i, sub in enumerate(s.columns):
                cleaned[f"{col}_{i}"] = s[sub].apply(_cell)
        else:
            cleaned[col] = s.apply(_cell)
    return pd.DataFrame(cleaned)

# Thread-safe timeout helper (prevents blocking the main thread)
def safe_timeout(fn, timeout=3, default=None):
    res = [default]
    def worker():
        try:
            res[0] = fn()
        except Exception:
            res[0] = default
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout)
    return res[0]

# -------------------------
# Alpha helpers (rate-limited)
# -------------------------
class AlphaLimiter:
    def __init__(self, calls_per_min=5):
        self.interval = 60.0 / max(1, calls_per_min)
        self._last = 0.0
    def wait(self):
        now = time.time()
        diff = now - self._last
        if diff < self.interval:
            time.sleep(self.interval - diff)
        self._last = time.time()

alpha_limiter = AlphaLimiter(calls_per_min=5)

def request_with_retries(url, params=None, timeout=8, retries=1):
    for attempt in range(retries+1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
        except Exception:
            pass
        if attempt < retries:
            time.sleep(0.6 + attempt)
    return None

def alpha_overview(sym_no_ns: str) -> Tuple[Optional[str], Optional[float]]:
    if not ALPHA_KEY:
        return None, None
    alpha_limiter.wait()
    url = "https://www.alphavantage.co/query"
    params = {"function":"OVERVIEW","symbol":f"{sym_no_ns}.NS","apikey":ALPHA_KEY}
    def call():
        r = request_with_retries(url, params=params, timeout=8, retries=1)
        return safe_json(r) if r else None
    data = safe_timeout(call, timeout=5, default=None)
    if not data:
        return None, None
    sec = data.get("Sector")
    mcap = data.get("MarketCapitalization")
    try:
        mcap = float(mcap) if mcap else None
    except Exception:
        mcap = None
    return sec, mcap

def alpha_global_price(sym_no_ns: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not ALPHA_KEY:
        return None, None, None
    alpha_limiter.wait()
    url = "https://www.alphavantage.co/query"
    params = {"function":"GLOBAL_QUOTE","symbol":f"{sym_no_ns}.NS","apikey":ALPHA_KEY}
    def call():
        r = request_with_retries(url, params=params, timeout=8, retries=1)
        return safe_json(r) if r else None
    data = safe_timeout(call, timeout=5, default=None)
    if not data:
        return None, None, None
    g = data.get("Global Quote", {}) or {}
    try:
        p = float(g.get("05. price")) if g.get("05. price") else None
        pc = float(g.get("08. previous close")) if g.get("08. previous close") else None
        pct = float(g.get("10. change percent").replace("%","")) if g.get("10. change percent") else None
        return p, pc, pct
    except Exception:
        return None, None, None

def alpha_time_series(sym_no_ns: str, start_date, end_date) -> pd.DataFrame:
    if not ALPHA_KEY:
        return pd.DataFrame()
    alpha_limiter.wait()
    url = "https://www.alphavantage.co/query"
    params = {"function":"TIME_SERIES_DAILY_ADJUSTED","symbol":f"{sym_no_ns}.NS","outputsize":"compact","apikey":ALPHA_KEY}
    def call():
        r = request_with_retries(url, params=params, timeout=8, retries=1)
        return safe_json(r) if r else None
    data = safe_timeout(call, timeout=6, default=None)
    if not data:
        return pd.DataFrame()
    ts = data.get("Time Series (Daily)") or {}
    rows = []
    for dstr, vals in ts.items():
        d = pd.to_datetime(dstr)
        if d.date() < start_date or d.date() > end_date:
            continue
        close = vals.get("5. adjusted close") or vals.get("4. close")
        if close is None:
            continue
        rows.append({"Date": d, "Close": float(close)})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("Date").sort_index()

# -------------------------
# NSE fallback (last resort)
# -------------------------
def nse_quote(sym_no_ns: str):
    url = f"https://www.nseindia.com/api/quote-equity?symbol={sym_no_ns}"
    headers = {
        "User-Agent":"Mozilla/5.0",
        "Accept-Language":"en-US,en;q=0.9",
        "Referer":"https://www.nseindia.com/",
        "Accept":"application/json, text/javascript, */*; q=0.01",
    }
    def call():
        try:
            s = requests.Session()
            s.get("https://www.nseindia.com", headers=headers, timeout=4)
            r = s.get(url, headers=headers, timeout=6)
            return safe_json(r) if r else None
        except Exception:
            return None
    data = safe_timeout(call, timeout=6, default=None)
    if not data:
        return None, None, None, None
    price_info = data.get("priceInfo") or {}
    secinfo = data.get("securityInfo") or {}
    last = price_info.get("lastPrice")
    prev = price_info.get("previousClose")
    pchg = price_info.get("pChange")
    sector = secinfo.get("industry") or secinfo.get("industryType")
    return last, prev, pchg, sector

# -------------------------
# History helper (yfinance -> alpha)
# -------------------------
def get_history_for_ticker(ticker: str, start_date, end_date) -> pd.DataFrame:
    def yf_call():
        try:
            # Use Ticker.history but wrapped in timeout
            t = yf.Ticker(ticker)
            h = t.history(start=start_date, end=end_date + timedelta(days=1), interval="1d", actions=False)
            if h is not None and not h.empty and "Close" in h.columns:
                return h[["Close"]]
        except Exception:
            return pd.DataFrame()
        return pd.DataFrame()
    res = safe_timeout(yf_call, timeout=4, default=pd.DataFrame())
    if res is not None and not res.empty:
        return res
    # fallback to alpha
    sym = ticker.replace(".NS", "")
    return alpha_time_series(sym, start_date, end_date)

# -------------------------
# Sector override loader (local CSV)
# -------------------------
def load_sector_override(path=SECTOR_OVERRIDE_FILE) -> Dict[str, str]:
    if os.path.exists(path):
        try:
            so = pd.read_csv(path)
            if {"Ticker","Sector"}.issubset(set(so.columns)):
                return dict(zip(so["Ticker"].astype(str).str.upper(), so["Sector"]))
        except Exception:
            pass
    return {}

SECTOR_OVERRIDE = load_sector_override()

# -------------------------
# Master loader (per-ticker, cached)
# -------------------------
@st.cache_data(ttl=120)
def load_live(tickers):
    rows = []
    for t in tickers:
        sym = t.replace(".NS", "")
        company = sym
        price = prev = pct = None
        mcap = None
        sector = None
        source = "none"

        # 1) yfinance: small per-ticker history (fast, protected)
        def yf_price_call():
            try:
                df = yf.Ticker(t).history(period="2d", interval="1d")
                if df is not None and not df.empty and "Close" in df.columns:
                    closes = df["Close"].dropna()
                    if len(closes) >= 1:
                        p = float(closes.iloc[-1])
                        pr = float(closes.iloc[-2]) if len(closes) >= 2 else None
                        return p, pr
            except Exception:
                return None
            return None
        pr = safe_timeout(yf_price_call, timeout=3, default=None)
        if pr:
            price, prev = pr
            source = "yfinance"

        # 2) If sector present in local override, use it
        t_up = t.upper()
        if t_up in SECTOR_OVERRIDE:
            sector = SECTOR_OVERRIDE[t_up]

        # 3) Alpha price fallback (if price missing)
        if price is None and ALPHA_KEY:
            p, pc, pchg = alpha_global_price(sym)
            if p is not None:
                price, prev, pct = p, pc, pchg
                source = source or "alpha-price"

        # 4) NSE fallback (last)
        if (price is None or sector is None):
            n_price, n_prev, n_pchg, n_sector = nse_quote(sym)
            if n_price is not None and price is None:
                price, prev, pct = n_price, n_prev, n_pchg
                source = source or "nse"
            if n_sector and sector is None:
                sector = n_sector

        # 5) Alpha overview only for missing sector or missing market cap
        if (sector is None or mcap is None) and ALPHA_KEY:
            sec_av, mcap_av = alpha_overview(sym)
            if sec_av and sector is None:
                sector = sec_av
            if mcap_av and mcap is None:
                mcap = mcap_av

        # compute pct if missing
        if pct is None and price is not None and prev not in (None, 0):
            try:
                pct = ((float(price) - float(prev)) / float(prev)) * 100
            except Exception:
                pct = None

        rows.append({
            "Ticker": t,
            "Company": company,
            "Sector": sector if sector else None,
            "Price": price,
            "Prev Close": prev,
            "% Change": pct,
            "Market Cap": mcap,
            "Source": source or "unknown"
        })

    df = pd.DataFrame(rows)

    # apply sector override map (ensures override wins)
    if SECTOR_OVERRIDE:
        df["Ticker_u"] = df["Ticker"].astype(str).str.upper()
        df["Sector"] = df.apply(lambda r: SECTOR_OVERRIDE.get(r["Ticker_u"], r["Sector"]), axis=1)
        df = df.drop(columns=["Ticker_u"], errors="ignore")

    return df

# -------------------------
# Run loader
# -------------------------
with st.spinner("Loading market data (hybrid Option C)..."):
    df = load_live(NIFTY50_TICKERS)

if df is None or df.empty:
    st.error("No data loaded. Check network or API keys. App will not continue.")
    st.stop()

# -------------------------
# Clean + compute
# -------------------------
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Prev Close"] = pd.to_numeric(df["Prev Close"], errors="coerce")
df["% Change"] = pd.to_numeric(df["% Change"], errors="coerce")
df["Market Cap"] = pd.to_numeric(df["Market Cap"], errors="coerce")

mask = df["% Change"].isna() & df["Price"].notna() & df["Prev Close"].notna() & (df["Prev Close"] != 0)
df.loc[mask, "% Change"] = ((df.loc[mask,"Price"] - df.loc[mask,"Prev Close"]) / df.loc[mask,"Prev Close"]) * 100

# Market cap fallback using median
if df["Market Cap"].notna().sum() >= 1:
    med = df.loc[df["Market Cap"].notna(), "Market Cap"].median()
    df["Market Cap Calc"] = df["Market Cap"].fillna(med)
else:
    df["Market Cap Calc"] = 1.0

total_calc = df["Market Cap Calc"].sum() if df["Market Cap Calc"].sum() != 0 else 1.0
df["Stake (%)"] = (df["Market Cap Calc"] / total_calc) * 100
df["Weighted Impact"] = (df["% Change"].fillna(0) * df["Stake (%)"]) / 100
df["Trend"] = np.where(df["% Change"] > 0, "Gainer", "Loser")
r_raw = df["% Change"].rank(ascending=False, method="first")
try:
    df["Rank"] = r_raw.astype("Int64")
except Exception:
    df["Rank"] = r_raw.fillna(0).astype(int)
df = df.sort_values("Market Cap Calc", ascending=False).reset_index(drop=True)

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Controls")
search = st.sidebar.text_input("Search company/ticker")
sectors = ["All"] + sorted(df["Sector"].fillna("Unknown").unique().tolist())
sector_sel = st.sidebar.selectbox("Sector", sectors, index=0)
min_mcap_crore = st.sidebar.number_input("Min Market Cap (Cr)", value=0.0, step=100.0)
if st.sidebar.button("Clear cache & refresh"):
    st.cache_data.clear()
    st.experimental_rerun()

display_df = df.copy()
if search:
    display_df = display_df[
        display_df["Company"].str.contains(search, case=False, na=False) |
        display_df["Ticker"].str.contains(search, case=False, na=False)
    ]
if sector_sel != "All":
    if sector_sel == "Unknown":
        display_df = display_df[display_df["Sector"].isna()]
    else:
        display_df = display_df[display_df["Sector"] == sector_sel]
if min_mcap_crore > 0:
    display_df = display_df[display_df["Market Cap"] >= (min_mcap_crore * 1e7)]

# -------------------------
# Metrics & top movers
# -------------------------
tot_pos = display_df.loc[display_df["% Change"]>0,"% Change"].sum()
tot_neg = display_df.loc[display_df["% Change"]<0,"% Change"].sum()
overall = display_df["Weighted Impact"].sum()
c1, c2, c3 = st.columns(3)
c1.metric("Total Gainers Impact", f"{tot_pos:.2f} %")
c2.metric("Total Losers Impact", f"{tot_neg:.2f} %")
c3.metric("Overall Weighted Impact", f"{overall:.2f} %")
st.write("---")

left, right = st.columns(2)
with left:
    st.subheader("Top 5 Gainers")
    for _, r in display_df.sort_values("% Change", ascending=False).head(5).iterrows():
        st.markdown(f"**{r['Company']}** ({r['Ticker']}) — `{(r['% Change'] if pd.notna(r['% Change']) else 0):.2f}%` — MC: **{format_mcap(r['Market Cap'])}** — Source: {r['Source']}")
with right:
    st.subheader("Top 5 Losers")
    for _, r in display_df.sort_values("% Change", ascending=True).head(5).iterrows():
        st.markdown(f"**{r['Company']}** ({r['Ticker']}) — `{(r['% Change'] if pd.notna(r['% Change']) else 0):.2f}%` — MC: **{format_mcap(r['Market Cap'])}** — Source: {r['Source']}")

st.write("---")

# -------------------------
# Table & download
# -------------------------
table = display_df[["Rank","Ticker","Company","Sector","Stake (%)","Price","% Change","Weighted Impact","Market Cap","Source"]].copy()
table["Stake (%)"] = table["Stake (%)"].round(2)
table["% Change"] = table["% Change"].round(2)
table["Weighted Impact"] = table["Weighted Impact"].round(4)
table["Market Cap"] = table["Market Cap"].apply(format_mcap)
st.subheader("Company Performance")
st.dataframe(sanitize(table), use_container_width=True)
st.download_button("Download CSV", table.to_csv(index=False), "nifty50.csv")

st.write("---")

# -------------------------
# Mini historical charts (30d)
# -------------------------
with st.expander("Mini historical charts (last 30 days)"):
    tickers_small = table["Ticker"].tolist()[:12]
    end = datetime.now().date()
    start = end - timedelta(days=30)
    cols = st.columns(3)
    for i, t in enumerate(tickers_small):
        company = table.loc[table["Ticker"]==t,"Company"].iloc[0]
        src = table.loc[table["Ticker"]==t,"Source"].iloc[0]
        with cols[i%3]:
            st.write(f"**{company}** — {t} (Source: {src})")
            hist = get_history_for_ticker(t, start, end)
            if hist is None or hist.empty:
                st.info("Mini history not available")
            else:
                fig = go.Figure(go.Scatter(x=hist.index, y=hist["Close"], mode="lines", line=dict(width=1)))
                fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=80, xaxis=dict(visible=False), yaxis=dict(visible=False))
                st.plotly_chart(fig, use_container_width=True)

st.write("---")

# -------------------------
# Charts
# -------------------------
st.plotly_chart(px.bar(display_df.sort_values("% Change", ascending=False), x="Company", y="% Change", color="Trend", title="% Change by Company"), use_container_width=True)

sector_agg = df.groupby(df["Sector"].fillna("Unknown"))["Weighted Impact"].sum().reset_index()
if sector_agg["Sector"].nunique() > 1:
    st.plotly_chart(px.bar(sector_agg.sort_values("Weighted Impact"), x="Sector", y="Weighted Impact", color="Weighted Impact", color_continuous_scale=px.colors.diverging.RdYlGn, title="Weighted Sector Impact"), use_container_width=True)
else:
    st.warning("Sector metadata limited or absent. Add ALPHAVANTAGE_API_KEY to Streamlit secrets or provide sectors_override.csv")

st.plotly_chart(px.pie(df.sort_values("Market Cap Calc", ascending=False).head(20), values="Stake (%)", names="Company", title="Company Stake Distribution (Top 20)"), use_container_width=True)

trend = display_df["Trend"].value_counts().reset_index()
trend.columns = ["Trend","Count"]
st.plotly_chart(px.pie(trend, values="Count", names="Trend", title="Gainers vs Losers", color_discrete_map={"Gainer":"green","Loser":"red"}), use_container_width=True)

st.write("---")

# -------------------------
# Company detail & 90d
# -------------------------
st.subheader("Company Detail")
choice = st.selectbox("Choose ticker", df["Ticker"].tolist())
if choice:
    row = df[df["Ticker"]==choice].iloc[0]
    st.markdown(f"**{row['Company']}** ({row['Ticker']}) — Sector: {row['Sector'] if pd.notna(row['Sector']) else 'Unknown'}")
    if pd.notna(row['% Change']):
        st.markdown(f"Price: **{row['Price'] if pd.notna(row['Price']) else 'N/A'}** | Prev Close: **{row['Prev Close'] if pd.notna(row['Prev Close']) else 'N/A'}** | %: **{row['% Change']:.2f}%**")
    else:
        st.markdown(f"Price: **{row['Price'] if pd.notna(row['Price']) else 'N/A'}** | Prev Close: **{row['Prev Close'] if pd.notna(row['Prev Close']) else 'N/A'}** | %: N/A")
    st.markdown(f"Market Cap: **{format_mcap(row['Market Cap'])}** — Stake: **{row['Stake (%)']:.2f}%** — Source: {row['Source']}")

    start90 = datetime.now().date() - timedelta(days=90)
    hist90 = get_history_for_ticker(choice, start90, datetime.now().date())
    if hist90 is None or hist90.empty:
        st.info("90-day history not available.")
    else:
        st.plotly_chart(px.line(hist90, y="Close", title=f"{choice} — 90-Day History"), use_container_width=True)
        st.download_button("Download 90d CSV", hist90.to_csv(), f"{choice}_90d.csv")

st.write("---")

# -------------------------
# Diagnostics
# -------------------------
with st.expander("Data-source diagnostics (sample)"):
    diag = df[["Ticker","Company","Source","Price","Prev Close","Market Cap","Sector","% Change"]].copy()
    st.dataframe(sanitize(diag), use_container_width=True)
    st.write("Alpha key present:", bool(ALPHA_KEY))
    st.write("Rows missing sector:", int(df["Sector"].isna().sum()))
    st.write("Rows using NSE as source:", int((df["Source"]=="nse").sum()))

st.caption("Hybrid Option C — per-ticker safe fetches, local sector override + Alpha fallback, NSE as last resort.")
