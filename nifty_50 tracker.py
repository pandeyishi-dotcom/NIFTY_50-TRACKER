# nifty_50_tracker_v2.py
# Streamlit App – Advanced Nifty 50 Live Tracker

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
import plotly.express as px

st.set_page_config(page_title="📊 Nifty 50 Live Market Dashboard", layout="wide")

st.title("📊 Nifty 50 Live Market Dashboard")
st.markdown(
    """
    **Live Nifty 50 data with sector analytics, stock ranking, gainers/losers, and impact simulation.**
    *Fetching from Yahoo Finance — may take 10–15 seconds.*
    """
)

# -------------------------------------------------------------------
# Nifty 50 tickers and sectors
# -------------------------------------------------------------------
nifty_symbols = {
    "RELIANCE.NS": "Energy", "TCS.NS": "IT", "INFY.NS": "IT", "HDFCBANK.NS": "Banking",
    "ICICIBANK.NS": "Banking", "SBIN.NS": "Banking", "HINDUNILVR.NS": "FMCG",
    "ITC.NS": "FMCG", "BHARTIARTL.NS": "Telecom", "KOTAKBANK.NS": "Banking",
    "LT.NS": "Infrastructure", "HCLTECH.NS": "IT", "ASIANPAINT.NS": "Consumer",
    "AXISBANK.NS": "Banking", "MARUTI.NS": "Auto", "SUNPHARMA.NS": "Pharma",
    "BAJFINANCE.NS": "Financial", "WIPRO.NS": "IT", "ULTRACEMCO.NS": "Cement",
    "TITAN.NS": "Consumer", "ONGC.NS": "Energy", "NTPC.NS": "Energy", "POWERGRID.NS": "Energy",
    "COALINDIA.NS": "Energy", "HDFCLIFE.NS": "Insurance", "TATAMOTORS.NS": "Auto",
    "GRASIM.NS": "Cement", "TATASTEEL.NS": "Metals", "JSWSTEEL.NS": "Metals",
    "ADANIPORTS.NS": "Infrastructure", "TECHM.NS": "IT", "DRREDDY.NS": "Pharma",
    "BRITANNIA.NS": "FMCG", "CIPLA.NS": "Pharma", "DIVISLAB.NS": "Pharma",
    "HERO.NS": "Auto", "EICHERMOT.NS": "Auto", "BAJAJFINSV.NS": "Financial",
    "HINDALCO.NS": "Metals", "ADANIENT.NS": "Infrastructure", "TATACONSUM.NS": "FMCG",
    "BPCL.NS": "Energy", "INDUSINDBK.NS": "Banking", "SBILIFE.NS": "Insurance",
    "APOLLOHOSP.NS": "Healthcare", "M&M.NS": "Auto", "UPL.NS": "Chemicals",
    "NESTLEIND.NS": "FMCG", "BAJAJ-AUTO.NS": "Auto", "TATAPOWER.NS": "Energy",
    "LTIM.NS": "IT"
}

# -------------------------------------------------------------------
# Fetch data
# -------------------------------------------------------------------
progress = st.progress(0)
data_list = []

for i, (symbol, sector) in enumerate(nifty_symbols.items()):
    try:
        t = yf.Ticker(symbol)
        info = t.info
        df = t.history(period="5d")

        price = info.get("regularMarketPrice") or df["Close"].iloc[-1]
        prev_close = info.get("previousClose") or df["Close"].iloc[-2]
        pct_change = ((price - prev_close) / prev_close) * 100 if prev_close else 0
        eps = info.get("trailingEps", np.nan)
        vol = df["Close"].pct_change().std() * np.sqrt(252) * 100  # annualized volatility %

        data_list.append({
            "Symbol": symbol.replace(".NS", ""),
            "Price": round(price, 2),
            "% Change": round(pct_change, 2),
            "Risk (Volatility %)": round(vol, 2),
            "EPS": eps,
            "Sector": sector,
            "Volume": info.get("volume", np.nan)
        })
    except Exception:
        data_list.append({
            "Symbol": symbol.replace(".NS", ""),
            "Price": np.nan, "% Change": np.nan, "Risk (Volatility %)": np.nan,
            "EPS": np.nan, "Sector": sector, "Volume": np.nan
        })
    progress.progress((i + 1) / len(nifty_symbols))
    time.sleep(0.1)

df = pd.DataFrame(data_list)

# -------------------------------------------------------------------
# Cleaning + enrichments
# -------------------------------------------------------------------
df["Rank"] = df["% Change"].rank(ascending=False, method="first").astype(int)
df["Status"] = np.where(df["% Change"] >= 0, "Gainer", "Loser")

# sector stake %
sector_totals = df.groupby("Sector")["Price"].sum()
df["Sector Stake %"] = df.apply(
    lambda r: (r["Price"] / sector_totals[r["Sector"]]) * 100 if r["Price"] and not np.isnan(r["Price"]) else np.nan,
    axis=1
)

# Impact simulation per stock (±100–500 pts)
impact_points = [100, 200, 250, 300, 500]
impact_records = []
for _, row in df.iterrows():
    for p in impact_points:
        for direction, sign in [("+", 1), ("−", -1)]:
            new_price = row["Price"] * (1 + sign * p / 22000)  # rough sensitivity
            impact_records.append({
                "Symbol": row["Symbol"], "Sector": row["Sector"], "Change": f"{direction}{p}",
                "Simulated Price": round(new_price, 2),
                "Impact": "Gainer" if sign == 1 else "Loser"
            })
impact_df = pd.DataFrame(impact_records)

# -------------------------------------------------------------------
# Display main table
# -------------------------------------------------------------------
st.subheader("📈 Live Stocks Data")
colored = df.style.map(
    lambda v: "color: green;" if isinstance(v, (int, float)) and v > 0 else
              ("color: red;" if isinstance(v, (int, float)) and v < 0 else None),
    subset=["% Change"]
)
st.dataframe(colored, use_container_width=True)

# -------------------------------------------------------------------
# Charts section
# -------------------------------------------------------------------
st.subheader("📊 Top Gainers and Losers")
top_gainers = df.nlargest(10, "% Change")
top_losers = df.nsmallest(10, "% Change")

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(px.bar(top_gainers, x="Symbol", y="% Change", color="Sector",
                           title="Top 10 Gainers", color_discrete_sequence=px.colors.qualitative.Dark24), use_container_width=True)
with col2:
    st.plotly_chart(px.bar(top_losers, x="Symbol", y="% Change", color="Sector",
                           title="Top 10 Losers", color_discrete_sequence=px.colors.qualitative.Dark24), use_container_width=True)

# Sector pie
st.subheader("🏭 Sector Exposure (Avg Price Weight)")
sector_summary = df.groupby("Sector").agg({"Price": "sum"}).reset_index()
fig_sector = px.pie(sector_summary, names="Sector", values="Price", hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel)
st.plotly_chart(fig_sector, use_container_width=True)

# Impact simulation table
st.subheader("📉 Impact Simulation for Each Stock")
st.dataframe(impact_df, use_container_width=True)

st.success("✅ Dashboard ready — reload for live updates.")
