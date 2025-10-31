# nifty_50_tracker.py
# Streamlit App – Nifty 50 Live Market Tracker

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time

st.set_page_config(page_title="📊 Nifty 50 Live Market Tracker", layout="wide")

st.title("📊 Nifty 50 Live Market Tracker")
st.markdown(
    """
    This Streamlit app tracks live **Nifty 50** stock data — including price, efficiency, risk (volatility proxy), EPS,
    sector holdings, and simulates index impact for ±100, 200, 250, 300, and 500 points.
    
    *Fetching live data from Yahoo Finance… (may take 10–15 seconds)*
    """
)

# ------------------------------------------------------
# Nifty 50 symbols and sectors (shortened, editable)
# ------------------------------------------------------
nifty_symbols = {
    "RELIANCE.NS": "Energy",
    "TCS.NS": "IT",
    "INFY.NS": "IT",
    "HDFCBANK.NS": "Banking",
    "ICICIBANK.NS": "Banking",
    "SBIN.NS": "Banking",
    "HINDUNILVR.NS": "FMCG",
    "ITC.NS": "FMCG",
    "BHARTIARTL.NS": "Telecom",
    "KOTAKBANK.NS": "Banking",
    "LT.NS": "Infrastructure",
    "HCLTECH.NS": "IT",
    "ASIANPAINT.NS": "Consumer",
    "AXISBANK.NS": "Banking",
    "MARUTI.NS": "Auto",
    "SUNPHARMA.NS": "Pharma",
    "BAJFINANCE.NS": "Financial",
    "WIPRO.NS": "IT",
    "ULTRACEMCO.NS": "Cement",
    "TITAN.NS": "Consumer",
    "ONGC.NS": "Energy",
    "NTPC.NS": "Energy",
    "POWERGRID.NS": "Energy",
    "COALINDIA.NS": "Energy",
    "HDFCLIFE.NS": "Insurance",
    "TATAMOTORS.NS": "Auto",
    "GRASIM.NS": "Cement",
    "TATASTEEL.NS": "Metals",
    "JSWSTEEL.NS": "Metals",
    "ADANIPORTS.NS": "Infrastructure",
    "TECHM.NS": "IT",
    "DRREDDY.NS": "Pharma",
    "BRITANNIA.NS": "FMCG",
    "CIPLA.NS": "Pharma",
    "DIVISLAB.NS": "Pharma",
    "HERO.NS": "Auto",
    "EICHERMOT.NS": "Auto",
    "BAJAJFINSV.NS": "Financial",
    "HINDALCO.NS": "Metals",
    "ADANIENT.NS": "Infrastructure",
    "TATACONSUM.NS": "FMCG",
    "BPCL.NS": "Energy",
    "INDUSINDBK.NS": "Banking",
    "SBILIFE.NS": "Insurance",
    "APOLLOHOSP.NS": "Healthcare",
    "M&M.NS": "Auto",
    "UPL.NS": "Chemicals",
    "NESTLEIND.NS": "FMCG",
    "BAJAJ-AUTO.NS": "Auto",
    "TATAPOWER.NS": "Energy",
    "LTIM.NS": "IT",
}

# ------------------------------------------------------
# Fetch Live Data
# ------------------------------------------------------
progress = st.progress(0)
data_list = []

for i, (symbol, sector) in enumerate(nifty_symbols.items()):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        df = ticker.history(period="5d")

        price = info.get("regularMarketPrice") or df["Close"].iloc[-1]
        prev_close = info.get("previousClose") or df["Close"].iloc[-2]
        pct_change = ((price - prev_close) / prev_close) * 100 if prev_close else 0
        eps = info.get("trailingEps", np.nan)
        vol = df["Close"].pct_change().std() * np.sqrt(252)  # annualized volatility proxy

        data_list.append(
            {
                "Symbol": symbol.replace(".NS", ""),
                "Price": round(price, 2),
                "% Change": round(pct_change, 2),
                "Risk": round(vol * 100, 2),
                "EPS": eps,
                "Sector": sector,
                "Volume": info.get("volume", np.nan),
            }
        )
    except Exception as e:
        data_list.append(
            {
                "Symbol": symbol.replace(".NS", ""),
                "Price": np.nan,
                "% Change": np.nan,
                "Risk": np.nan,
                "EPS": np.nan,
                "Sector": sector,
                "Volume": np.nan,
            }
        )

    progress.progress((i + 1) / len(nifty_symbols))
    time.sleep(0.1)

df_final = pd.DataFrame(data_list)

# ------------------------------------------------------
# Clean up missing columns safely
# ------------------------------------------------------
expected_cols = ["Price", "% Change", "Risk", "EPS", "Sector", "Volume"]
for col in expected_cols:
    if col not in df_final.columns:
        df_final[col] = None

# ------------------------------------------------------
# Estimate Index Impact Simulation
# ------------------------------------------------------
if "Price" in df_final.columns and df_final["Price"].notna().any():
    nifty_price = df_final["Price"].mean() * len(df_final) / 50
else:
    st.warning("⚠️ 'Price' column missing — using default NIFTY value for simulation.")
    nifty_price = 22000  # fallback index value

impact_levels = [100, 200, 250, 300, 500]
sim_data = []

for change in impact_levels:
    up = nifty_price + change
    down = nifty_price - change
    sim_data.append({"Change": f"+{change}", "Estimated Nifty": round(up, 2), "Impact": "Gainer"})
    sim_data.append({"Change": f"-{change}", "Estimated Nifty": round(down, 2), "Impact": "Loser"})

df_sim = pd.DataFrame(sim_data)

# ------------------------------------------------------
# Display Data
# ------------------------------------------------------
st.subheader("📈 Live Nifty 50 Stocks Data")

if "% Change" in df_final.columns and df_final["% Change"].notna().any():
    st.dataframe(df_final.sort_values("% Change", ascending=False), use_container_width=True)
else:
    st.warning("⚠️ '% Change' data unavailable — showing unsorted table.")
    st.dataframe(df_final, use_container_width=True)

st.subheader("📊 Nifty Index Impact Simulation")
st.dataframe(df_sim, use_container_width=True)

# ------------------------------------------------------
# Sector-wise Exposure
# ------------------------------------------------------
st.subheader("🏭 Sector-wise Holdings")
sector_summary = (
    df_final.groupby("Sector")
    .agg({"Symbol": "count", "Price": "mean"})
    .rename(columns={"Symbol": "Companies", "Price": "Avg Price"})
    .reset_index()
)
st.dataframe(sector_summary, use_container_width=True)

st.success("✅ App ready. Data refreshes each time you reload the page.")
