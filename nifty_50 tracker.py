import sys, subprocess

# Auto-install missing dependencies
for pkg in ["streamlit", "yfinance", "pandas", "numpy", "requests"]:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests

st.set_page_config(page_title="Nifty 50 Live Tracker", layout="wide")

st.title("📊 Nifty 50 Live Market Tracker")

st.markdown("""
This Streamlit app tracks live Nifty 50 stock data — including price, returns, risk (volatility proxy),
EPS, sector, and simulates index impact for Nifty moves of ±100, 200, 250, 300, and 500 points.
""")

# -----------------------
# Define Nifty 50 symbols
# -----------------------
nifty_symbols = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS","HINDUNILVR.NS",
    "ITC.NS","SBIN.NS","BHARTIARTL.NS","BAJFINANCE.NS","LT.NS","KOTAKBANK.NS","AXISBANK.NS",
    "ASIANPAINT.NS","MARUTI.NS","SUNPHARMA.NS","HCLTECH.NS","TITAN.NS","ULTRACEMCO.NS",
    "NESTLEIND.NS","POWERGRID.NS","ADANIENT.NS","ADANIPORTS.NS","COALINDIA.NS","ONGC.NS",
    "NTPC.NS","WIPRO.NS","JSWSTEEL.NS","TECHM.NS","HDFCLIFE.NS","SBILIFE.NS","BAJAJFINSV.NS",
    "TATAMOTORS.NS","GRASIM.NS","INDUSINDBK.NS","HINDALCO.NS","BRITANNIA.NS","EICHERMOT.NS",
    "DRREDDY.NS","DIVISLAB.NS","HEROMOTOCO.NS","CIPLA.NS","TATASTEEL.NS","UPL.NS","BPCL.NS",
    "SHREECEM.NS","BAJAJ-AUTO.NS","APOLLOHOSP.NS","M&M.NS","SBICARD.NS","LTIM.NS"
]

# -------------------------
# Static sector classification (for demo)
# -------------------------
sector_map = {
    "RELIANCE.NS": "Energy & Petrochemicals",
    "TCS.NS": "IT Services", "INFY.NS": "IT Services", "HDFCBANK.NS": "Banking",
    "ICICIBANK.NS": "Banking", "HINDUNILVR.NS": "FMCG", "ITC.NS": "FMCG", "SBIN.NS": "Banking",
    "BHARTIARTL.NS": "Telecom", "BAJFINANCE.NS": "NBFC", "LT.NS": "Infrastructure",
    "KOTAKBANK.NS": "Banking", "AXISBANK.NS": "Banking", "ASIANPAINT.NS": "FMCG",
    "MARUTI.NS": "Automobile", "SUNPHARMA.NS": "Pharmaceuticals", "HCLTECH.NS": "IT Services",
    "TITAN.NS": "Consumer Durables", "ULTRACEMCO.NS": "Cement", "NESTLEIND.NS": "FMCG",
    "POWERGRID.NS": "Power", "ADANIENT.NS": "Conglomerate", "ADANIPORTS.NS": "Ports & Logistics",
    "COALINDIA.NS": "Mining", "ONGC.NS": "Energy", "NTPC.NS": "Power", "WIPRO.NS": "IT Services",
    "JSWSTEEL.NS": "Metals", "TECHM.NS": "IT Services", "HDFCLIFE.NS": "Insurance",
    "SBILIFE.NS": "Insurance", "BAJAJFINSV.NS": "NBFC", "TATAMOTORS.NS": "Automobile",
    "GRASIM.NS": "Cement", "INDUSINDBK.NS": "Banking", "HINDALCO.NS": "Metals",
    "BRITANNIA.NS": "FMCG", "EICHERMOT.NS": "Automobile", "DRREDDY.NS": "Pharmaceuticals",
    "DIVISLAB.NS": "Pharmaceuticals", "HEROMOTOCO.NS": "Automobile", "CIPLA.NS": "Pharmaceuticals",
    "TATASTEEL.NS": "Metals", "UPL.NS": "Agro Chemicals", "BPCL.NS": "Energy",
    "SHREECEM.NS": "Cement", "BAJAJ-AUTO.NS": "Automobile", "APOLLOHOSP.NS": "Healthcare",
    "M&M.NS": "Automobile", "SBICARD.NS": "NBFC", "LTIM.NS": "IT Services"
}

# ------------------------
# Fetch Live Data
# ------------------------
st.info("Fetching live data from Yahoo Finance... (may take 10–15 seconds)")

data = yf.download(tickers=nifty_symbols, period="1d", interval="1m", group_by='ticker', threads=True)

# ------------------------
# Process Data
# ------------------------
rows = []
for symbol in nifty_symbols:
    try:
        df = data[symbol]
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
        volatility = df['Close'].pct_change().std() * np.sqrt(252)
        volume = latest['Volume']
        eps = np.random.uniform(20, 120)  # placeholder (simulate EPS)
        weight = np.random.uniform(0.5, 5)  # dummy weight %
        risk = volatility * 100
        sector = sector_map.get(symbol, "Unknown")

        rows.append({
            "Symbol": symbol.replace(".NS", ""),
            "Price": round(latest['Close'], 2),
            "% Change": round(change, 2),
            "Volume": int(volume),
            "Risk (Volatility%)": round(risk, 2),
            "Weight (%)": round(weight, 2),
            "EPS": round(eps, 2),
            "Sector": sector,
            "Status": "Gainer" if change > 0 else "Loser"
        })
    except Exception:
        continue

df_final = pd.DataFrame(rows)

# ------------------------
# Nifty Impact Simulation
# ------------------------
nifty_move_points = [100, 200, 250, 300, 500]
if "Price" in df_final.columns:
    nifty_price = df_final["Price"].mean() * len(df_final) / 50  # rough proxy
else:
    st.warning("⚠️ 'Price' column missing — using default NIFTY index value for simulation.")
    nifty_price = 22000  # fallback value (you can adjust to live NIFTY)

impact_df = pd.DataFrame({
    "Nifty Move (pts)": nifty_move_points,
    "Impact (%)": [round((m / nifty_price) * 100, 2) for m in nifty_move_points]
})

# ------------------------
# Display
# ------------------------
st.subheader("📈 Live Nifty 50 Stocks Data")
st.dataframe(df_final.sort_values("% Change", ascending=False), use_container_width=True)

st.subheader("📊 Simulated Nifty Impact Scenarios")
st.table(impact_df)

st.subheader("🏭 Sector-wise Exposure")
sector_view = df_final.groupby("Sector")["Weight (%)"].sum().reset_index().sort_values("Weight (%)", ascending=False)
st.bar_chart(sector_view.set_index("Sector"))

st.caption("Data from Yahoo Finance. EPS & Weight are simulated placeholders for demonstration.")
