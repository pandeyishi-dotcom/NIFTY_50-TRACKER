import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px

st.set_page_config(page_title="NIFTY 50 Live Dashboard", layout="wide")
st.title("📈 NIFTY 50 Live Performance Dashboard")

# NIFTY 50 stock list with Yahoo Finance tickers
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

@st.cache_data(ttl=600)
def load_live_data():
    tickers = yf.Tickers(" ".join(NIFTY50_TICKERS))
    data_rows = []
    for t in NIFTY50_TICKERS:
        info = tickers.tickers[t].info
        company = info.get("shortName", t.replace(".NS", ""))
        price = info.get("regularMarketPrice")
        prev_close = info.get("regularMarketPreviousClose")
        market_cap = info.get("marketCap")
        sector = info.get("sector", "Unknown")
        pct_change = None
        if price and prev_close and prev_close != 0:
            pct_change = ((price - prev_close) / prev_close) * 100
        data_rows.append({
            "Company": company,
            "Sector": sector,
            "Price": price,
            "Prev Close": prev_close,
            "% Change": pct_change,
            "Market Cap": market_cap
        })
    df = pd.DataFrame(data_rows)
    df = df.dropna(subset=["% Change", "Market Cap"])
    total_cap = df["Market Cap"].sum()
    df["Stake (%)"] = (df["Market Cap"] / total_cap) * 100
    return df

with st.spinner("Fetching live NIFTY 50 data..."):
    df = load_live_data()

# Rank & trend
df["Rank"] = df["% Change"].rank(ascending=False, method="first").astype(int)
df["Trend"] = np.where(df["% Change"] > 0, "Gainer", "Loser")
df["Weighted Impact"] = (df["% Change"] * df["Stake (%)"]) / 100
df["Color"] = df["% Change"].apply(lambda x: "green" if x > 0 else "red")
df = df.sort_values(by="Rank")

# Metrics
total_positive = df.loc[df["% Change"] > 0, "% Change"].sum()
total_negative = df.loc[df["% Change"] < 0, "% Change"].sum()
overall_impact = df["Weighted Impact"].sum()

col1, col2, col3 = st.columns(3)
col1.metric("Total Gainers Impact", f"{total_positive:.2f} %", delta=f"{total_positive:.2f}")
col2.metric("Total Losers Impact", f"{total_negative:.2f} %", delta=f"{total_negative:.2f}")
col3.metric("Overall Weighted Impact", f"{overall_impact:.2f} %", delta=f"{overall_impact:.2f}")

st.write("---")

# Table
st.subheader("🏦 Company Performance")
st.dataframe(
    df[["Rank", "Company", "Sector", "Stake (%)", "Price", "% Change", "Trend", "Weighted Impact"]],
    use_container_width=True
)

# Charts
fig1 = px.bar(df, x="Company", y="% Change", color="Trend",
              color_discrete_map={"Gainer": "green", "Loser": "red"},
              text="Rank", title="% Change by Company")
fig1.update_traces(textposition="outside")
st.plotly_chart(fig1, use_container_width=True)

sector_impact = df.groupby("Sector")["Weighted Impact"].sum().reset_index()
fig2 = px.bar(sector_impact, x="Sector", y="Weighted Impact",
              color="Weighted Impact", color_continuous_scale=["red", "green"],
              title="Weighted Sector Impact")
st.plotly_chart(fig2, use_container_width=True)

fig3 = px.pie(df, values="Stake (%)", names="Company",
              title="Company Stake Distribution in NIFTY 50")
st.plotly_chart(fig3, use_container_width=True)

trend_count = df["Trend"].value_counts().reset_index()
trend_count.columns = ["Trend", "Count"]
fig4 = px.pie(trend_count, names="Trend", values="Count", color="Trend",
              color_discrete_map={"Gainer": "green", "Loser": "red"},
              title="Gainers vs Losers Ratio")
st.plotly_chart(fig4, use_container_width=True)

st.write("---")
st.caption("Live data via Yahoo Finance (`yfinance`). Updated every 10 minutes.")
