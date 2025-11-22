# app.py — NIFTY50 production-recovery dashboard (drop-in)
# Restores all features: Yahoo/Alpha/NSE fallback, history fallbacks, sector override, diagnostics.
# Requirements listed separately in requirements.txt.

import os
import time
import math
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------
# Basic config
# ---------------------------
st.set_page_config(page_title="NIFTY50 — Restored Dashboard", layout="wide")
st.title("📈 NIFTY50 — Restored Stable Dashboard")

# NIFTY50 list
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
SECTOR_OVERRIDE_FILE = "sectors_override.csv"

# ---------------------------
# Utilities
# ---------------------------
def safe_json(resp):
    try:
        return resp.json()
    except Exception:
        return None

def format_mcap(mc):
    if mc is None or (isinstance(mc, float) and math.isnan(mc)):
        return "N/A"
    try:
        crores = float(mc) / 1e7
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

# ---------------------------
# Alpha polite limiter & helpers
# ---------------------------
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

def request_with_retries(url, params=None, timeout=10, retries=2):
    for attempt in range(retries+1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
        except Exception:
            pass
        if attempt < retries:
            time.sleep(1 + attempt)
    return None

def alpha_global_quote(sym_no_ns: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not ALPHA_KEY:
        return None, None, None
    alpha_limiter.wait()
    url = "https://www.alphavantage.co/query"
    params = {"function":"GLOBAL_QUOTE","symbol":f"{sym_no_ns}.NS","apikey":ALPHA_KEY}
    r = request_with_retries(url, params=params, timeout=10, retries=2)
    if not r:
        return None, None, None
    data = safe_json(r)
    g = data.get("Global Quote", {}) if data else {}
    if not g:
        return None, None, None
    p = g.get("05. price"); pc = g.get("08. previous close"); pct = g.get("10. change percent")
    try:
        p = float(p) if p else None
        pc = float(pc) if pc else None
        pct = float(pct.replace("%","")) if pct else None
    except Exception:
        p, pc, pct = None, None, None
    return p, pc, pct

def alpha_overview(sym_no_ns: str) -> Tuple[Optional[str], Optional[float]]:
    if not ALPHA_KEY:
        return None, None
    alpha_limiter.wait()
    url = "https://www.alphavantage.co/query"
    params = {"function":"OVERVIEW","symbol":f"{sym_no_ns}.NS","apikey":ALPHA_KEY}
    r = request_with_retries(url, params=params, timeout=10, retries=2)
    if not r:
        return None, None
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

def alpha_time_series(sym_no_ns: str, start_date, end_date) -> pd.DataFrame:
    if not ALPHA_KEY:
        return pd.DataFrame()
    alpha_limiter.wait()
    url = "https://www.alphavantage.co/query"
    params = {"function":"TIME_SERIES_DAILY_ADJUSTED","symbol":f"{sym_no_ns}.NS","outputsize":"full","apikey":ALPHA_KEY}
    r = request_with_retries(url, params=params, timeout=12, retries=2)
    if not r:
        return pd.DataFrame()
    data = safe_json(r)
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
    df = pd.DataFrame(rows).set_index("Date").sort_index()
    return df

# ---------------------------
# NSE fallback (best-effort)
# ---------------------------
def nse_quote(sym_no_ns: str):
    url = f"https://www.nseindia.com/api/quote-equity?symbol={sym_no_ns}"
    headers = {
        "User-Agent":"Mozilla/5.0",
        "Accept-Language":"en-US,en;q=0.9",
        "Referer":"https://www.nseindia.com/",
        "Accept":"application/json, text/javascript, */*; q=0.01",
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
        pchg = price_info.get("pChange")
        secinfo = data.get("securityInfo") or {}
        sector = secinfo.get("industry") or secinfo.get("industryType")
        return last, prev, pchg, sector
    except Exception:
        return None, None, None, None

# ---------------------------
# History helper
# ---------------------------
def get_history_for_ticker(ticker: str, start_date, end_date) -> pd.DataFrame:
    try:
        df = yf.download(ticker, start=start_date, end=end_date + timedelta(days=1), progress=False)
        if df is not None and not df.empty and "Close" in df.columns:
            return df[["Close"]]
    except Exception:
        pass
    try:
        t = yf.Ticker(ticker)
        h = t.history(start=start_date, end=end_date + timedelta(days=1), interval="1d", actions=False)
        if h is not None and not h.empty and "Close" in h.columns:
            return h[["Close"]]
    except Exception:
        pass
    if ALPHA_KEY:
        sym = ticker.replace(".NS", "")
        df_av = alpha_time_series(sym, start_date, end_date)
        if df_av is not None and not df_av.empty:
            return df_av
    return pd.DataFrame()

# ---------------------------
# Sector override loader
# ---------------------------
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

# ---------------------------
# Master loader (cached)
# ---------------------------
@st.cache_data(ttl=180)
def load_live(tickers):
    rows = []
    try:
        bulk = yf.download(tickers=tickers, period="2d", interval="1d", group_by="ticker", threads=True, progress=False, auto_adjust=False)
    except Exception:
        bulk = pd.DataFrame()

    for t in tickers:
        sym = t.replace(".NS","")
        company = sym
        price = prev = pct = None
        mcap = None; sector = None; source = None

        try:
            if isinstance(bulk, pd.DataFrame) and t in bulk.columns.get_level_values(0):
                closes = bulk[t]["Close"].dropna()
                if len(closes) >= 1:
                    price = float(closes.iloc[-1])
                if len(closes) >= 2:
                    prev = float(closes.iloc[-2])
        except Exception:
            pass

        try:
            fi = yf.Ticker(t).fast_info
            if fi:
                mcap = fi.get("market_cap") or fi.get("marketCap") or mcap
                sector = fi.get("sector") or fi.get("Sector") or sector
                company = fi.get("shortName") or fi.get("longName") or company
                if price is not None:
                    source = "yahoo"
        except Exception:
            pass

        if price is not None and prev is not None and prev != 0:
            try:
                pct = ((price - prev) / prev) * 100
            except Exception:
                pct = None

        if price is None:
            p, pc, pchg = alpha_global_quote(sym)
            if p is not None:
                price, prev, pct = p, pc, pchg
                source = source or "alpha"

        if (price is None) or (sector is None):
            n_price, n_prev, n_pchg, n_sector = nse_quote(sym)
            if n_price is not None and price is None:
                price, prev, pct = n_price, n_prev, n_pchg
                source = source or "nse"
            if n_sector and sector is None:
                sector = n_sector

        if (sector is None or mcap is None) and ALPHA_KEY:
            sec_av, mcap_av = alpha_overview(sym)
            if sec_av and sector is None:
                sector = sec_av
            if mcap_av and mcap is None:
                mcap = mcap_av

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

    if SECTOR_OVERRIDE:
        df["Ticker_u"] = df["Ticker"].astype(str).str.upper()
        df["Sector"] = df.apply(lambda r: SECTOR_OVERRIDE.get(r["Ticker_u"], r["Sector"]), axis=1)
        df = df.drop(columns=["Ticker_u"])

    return df

# ---------------------------
# Run loader
# ---------------------------
with st.spinner("Loading market data..."):
    df = load_live(NIFTY50_TICKERS)

if df is None or df.empty:
    st.error("No data loaded. Check network, yfinance, and ALPHAVANTAGE_API_KEY (optional). See /mnt/data/error.pdf for earlier logs.")
    st.stop()

# numeric coercion & fixes
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Prev Close"] = pd.to_numeric(df["Prev Close"], errors="coerce")
df["% Change"] = pd.to_numeric(df["% Change"], errors="coerce")
df["Market Cap"] = pd.to_numeric(df["Market Cap"], errors="coerce")
mask = df["% Change"].isna() & df["Price"].notna() & df["Prev Close"].notna() & (df["Prev Close"] != 0)
df.loc[mask, "% Change"] = ((df.loc[mask,"Price"] - df.loc[mask,"Prev Close"]) / df.loc[mask,"Prev Close"]) * 100

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

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("Controls")
search = st.sidebar.text_input("Search company/ticker", value="")
sectors = ["All"] + sorted(df["Sector"].fillna("Unknown").unique().tolist())
sector_sel = st.sidebar.selectbox("Sector", sectors, index=0)
min_mcap_crore = st.sidebar.number_input("Min market cap (crore INR)", value=0.0, step=100.0)
if st.sidebar.button("Clear cache & refresh"):
    st.cache_data.clear()
    st.experimental_rerun()

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

# ---------------------------
# Metrics & top movers
# ---------------------------
tot_pos = display_df.loc[display_df["% Change"]>0,"% Change"].sum()
tot_neg = display_df.loc[display_df["% Change"]<0,"% Change"].sum()
overall = display_df["Weighted Impact"].sum()
c1,c2,c3 = st.columns(3)
c1.metric("Total Gainers Impact", f"{tot_pos:.2f} %")
c2.metric("Total Losers Impact", f"{tot_neg:.2f} %")
c3.metric("Overall Weighted Impact", f"{overall:.2f} %")
st.write("---")

left,right = st.columns(2)
with left:
    st.subheader("Top 5 Gainers")
    for _, r in display_df.sort_values("% Change", ascending=False).head(5).iterrows():
        st.markdown(f"**{r['Company']}** ({r['Ticker']}) — `{(r['% Change'] if pd.notna(r['% Change']) else 0):.2f}%` — MC: **{format_mcap(r['Market Cap'])}** — Source: {r['Source']}")
with right:
    st.subheader("Top 5 Losers")
    for _, r in display_df.sort_values("% Change", ascending=True).head(5).iterrows():
        st.markdown(f"**{r['Company']}** ({r['Ticker']}) — `{(r['% Change'] if pd.notna(r['% Change']) else 0):.2f}%` — MC: **{format_mcap(r['Market Cap'])}** — Source: {r['Source']}")

st.write("---")

# ---------------------------
# Table & downloads
# ---------------------------
table = display_df[["Rank","Ticker","Company","Sector","Stake (%)","Price","% Change","Weighted Impact","Market Cap","Source"]].copy()
table["Stake (%)"] = table["Stake (%)"].round(2)
table["% Change"] = table["% Change"].round(2)
table["Weighted Impact"] = table["Weighted Impact"].round(4)
table["Market Cap"] = table["Market Cap"].apply(format_mcap)
safe_table = sanitize(table)
st.subheader("Company Performance")
st.dataframe(safe_table, use_container_width=True)
st.download_button("Download CSV", data=safe_table.to_csv(index=False).encode(), file_name="nifty50.csv", mime="text/csv")
st.write("---")

# ---------------------------
# Mini history (30d)
# ---------------------------
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
            if hist is not None and not hist.empty:
                fig = go.Figure(go.Scatter(x=hist.index, y=hist["Close"], mode="lines", line=dict(width=1)))
                fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=80, xaxis=dict(visible=False), yaxis=dict(visible=False))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Mini history not available")

st.write("---")

# ---------------------------
# Charts (pct, sector impact, stake, gainers/losers)
# ---------------------------
fig_pct = px.bar(display_df.sort_values("% Change", ascending=False), x="Company", y="% Change", color="Trend", title="% Change by Company")
fig_pct.update_traces(texttemplate="%{y:.2f}", textposition="outside")
st.plotly_chart(fig_pct, use_container_width=True)

sector_impact = df.groupby(df["Sector"].fillna("Unknown"))["Weighted Impact"].sum().reset_index()
if len(sector_impact) <= 1:
    st.warning("Sector metadata limited or absent. Add ALPHAVANTAGE_API_KEY to Streamlit secrets or drop sectors_override.csv.")
else:
    st.subheader("Weighted Sector Impact")
    st.plotly_chart(px.bar(sector_impact.sort_values("Weighted Impact"), x="Sector", y="Weighted Impact", title="Weighted Sector Impact", color="Weighted Impact", color_continuous_scale=px.colors.diverging.RdYlGn), use_container_width=True)

st.plotly_chart(px.pie(df.sort_values("Market Cap Calc", ascending=False).head(20), values="Stake (%)", names="Company", title="Company Stake Distribution (Top 20)"), use_container_width=True)

trend_count = display_df["Trend"].value_counts().rename_axis("Trend").reset_index(name="Count")
if trend_count["Count"].sum() == 0:
    st.info("No gainers/losers to display.")
else:
    st.plotly_chart(px.pie(trend_count, names="Trend", values="Count", title="Gainers vs Losers", color_discrete_map={"Gainer":"green","Loser":"red"}), use_container_width=True)

st.write("---")

# ---------------------------
# Company detail + 90d
# ---------------------------
st.subheader("Company Detail")
choice = st.selectbox("Choose ticker", options=df["Ticker"].tolist())
selected = df[df["Ticker"]==choice].iloc[0] if choice else None
if selected is not None:
    st.markdown(f"**{selected['Company']}** ({selected['Ticker']}) — Sector: {selected['Sector'] if pd.notna(selected['Sector']) else 'Unknown'}")
    st.markdown(f"Price: **{selected['Price'] if pd.notna(selected['Price']) else 'N/A'}** | Prev Close: **{selected['Prev Close'] if pd.notna(selected['Prev Close']) else 'N/A'}** | %: **{selected['% Change']:.2f}%**" if pd.notna(selected['% Change']) else "N/A")
    st.markdown(f"Market Cap: **{format_mcap(selected['Market Cap'])}** — Stake: **{selected['Stake (%)']:.2f}%** — Source: {selected['Source']}")
    try:
        hist90 = get_history_for_ticker(selected["Ticker"], datetime.now().date()-timedelta(days=90), datetime.now().date())
        if hist90 is not None and not hist90.empty:
            st.plotly_chart(px.line(hist90, y="Close", title=f"{selected['Ticker']} — Last 90 days"), use_container_width=True)
            st.download_button("Download 90d CSV", data=hist90.to_csv().encode(), file_name=f"{selected['Ticker']}_90d.csv", mime="text/csv")
        else:
            st.info("90-day history not available.")
    except Exception:
        st.info("90-day history not available (error).")
else:
    st.info("No ticker selected or data missing.")

# ---------------------------
# Diagnostics
# ---------------------------
with st.expander("Data-source diagnostics (first 200 rows)"):
    diag = df[["Ticker","Company","Source","Price","Prev Close","Market Cap","Sector","% Change"]].copy()
    st.dataframe(sanitize(diag), use_container_width=True)

st.caption("If many rows show Source=nse and histories missing, add ALPHAVANTAGE_API_KEY to Streamlit secrets (recommended) and/or add sectors_override.csv. You uploaded previous logs at /mnt/data/error.pdf.")
