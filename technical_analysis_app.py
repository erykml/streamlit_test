import streamlit as st
import yfinance as yf
from datetime import timedelta, date
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import talib
import requests
from io import StringIO
from plotly.subplots import make_subplots

# --- Streamlit settings ---
st.set_page_config(page_title="Technical Analysis App", layout="wide")


# --- Data functions ---
@st.cache_data
def get_sp500_components():
    """Get S&P 500 tickers and their company names."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text
    df = pd.read_html(StringIO(html))[0]
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    tickers = df["Symbol"].tolist()
    tickers_companies_dict = dict(zip(df["Symbol"], df["Security"]))
    return tickers, tickers_companies_dict


@st.cache_data(ttl=600)
def load_data(symbol, start, end):
    """Download historical price data."""
    df = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )
    return df.dropna()


# --- Helper: RSI plot ---
def plot_rsi(df, rsi_col="RSI", rsi_period=14, upper=70, lower=30):
    """Create RSI plot."""
    fig = px.line(df, x=df.index, y=rsi_col, title=f"RSI ({rsi_period})")
    fig.add_hline(y=upper, line_dash="dash", line_color="red")
    fig.add_hline(y=lower, line_dash="dash", line_color="green")
    fig.update_layout(
        yaxis=dict(range=[0, 100]),
        height=400,
        showlegend=False,
    )
    return fig


# --- Sidebar inputs ---
st.sidebar.header("Configuration")

available_tickers, tickers_companies_dict = get_sp500_components()
ticker = st.sidebar.selectbox(
    "Ticker", available_tickers, format_func=tickers_companies_dict.get
)
today = date.today()
one_year_ago = today - timedelta(days=365)
start_date = st.sidebar.date_input("Start date", one_year_ago)
end_date = st.sidebar.date_input("End date", today)

if start_date > end_date:
    st.sidebar.error("The end date must fall after the start date")

# --- Technical Indicators ---
st.sidebar.header("Technical Indicators")

show_volume = st.sidebar.checkbox("Show Volume", value=True)

exp_sma = st.sidebar.expander("Simple Moving Averages (SMA)")
add_sma1 = exp_sma.checkbox("Add SMA 1", value=True)
sma1_period = exp_sma.number_input("SMA 1 Period", 1, 200, 20)
add_sma2 = exp_sma.checkbox("Add SMA 2")
sma2_period = exp_sma.number_input("SMA 2 Period", 1, 200, 50)

exp_rsi = st.sidebar.expander("Relative Strength Index (RSI)")
add_rsi = exp_rsi.checkbox("Add RSI")
rsi_period = exp_rsi.number_input("RSI Periods", 1, 50, 14)
rsi_upper = exp_rsi.number_input("RSI Upper", 50, 90, 70)
rsi_lower = exp_rsi.number_input("RSI Lower", 10, 50, 30)

# --- Main content ---
st.title("📊 Technical Analysis App")
st.markdown(
    """
### User manual
* you can select any of the companies that is a component of the S&P index
* you can select the time period of your interest
* you can download the selected data as a CSV file
* you can add the following Technical Indicators to the plot: Simple Moving 
Average, Relative Strength Index
* you can experiment with different parameters of the indicators
"""
)

# --- Load and process data ---
df = load_data(ticker, start_date, end_date)
if add_sma1:
    df[f"SMA_{sma1_period}"] = talib.SMA(df["Close"], timeperiod=sma1_period)
if add_sma2:
    df[f"SMA_{sma2_period}"] = talib.SMA(df["Close"], timeperiod=sma2_period)
if add_rsi:
    df["RSI"] = talib.RSI(df["Close"], timeperiod=rsi_period)

# --- Data preview ---
with st.expander("📁 Data Preview"):
    available_cols = df.columns.tolist()
    columns_to_show = st.multiselect("Columns", available_cols, default=available_cols)
    st.dataframe(df[columns_to_show])
    csv = df[columns_to_show].to_csv().encode("utf-8")
    st.download_button("Download CSV", csv, f"{ticker}_data.csv", "text/csv")

# --- Plots ---
price_fig = make_subplots(
    rows=2 if show_volume else 1,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.7, 0.3] if show_volume else [1.0],
    subplot_titles=(
        (f"{tickers_companies_dict.get(ticker, ticker)} Price", "Volume")
        if show_volume
        else (f"{tickers_companies_dict.get(ticker, ticker)} Price",)
    ),
)

# Price Trace (Candlestick)
price_fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price",
    ),
    row=1,
    col=1,
)

# SMA Traces
if add_sma1:
    price_fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[f"SMA_{sma1_period}"],
            mode="lines",
            name=f"SMA {sma1_period}",
            line=dict(color="blue", width=1.5),
        ),
        row=1,
        col=1,
    )
if add_sma2:
    price_fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[f"SMA_{sma2_period}"],
            mode="lines",
            name=f"SMA {sma2_period}",
            line=dict(color="orange", width=1.5, dash="dot"),
        ),
        row=1,
        col=1,
    )

# Volume Trace
if show_volume:
    price_fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color="silver"),
        row=2,
        col=1,
    )

# Formatting
price_fig.update_layout(
    xaxis_rangeslider_visible=False,
    height=700,
)

st.plotly_chart(price_fig, width="stretch")

# RSI chart
if add_rsi:
    rsi_fig = plot_rsi(
        df, rsi_col="RSI", rsi_period=rsi_period, upper=rsi_upper, lower=rsi_lower
    )
    st.plotly_chart(rsi_fig, width="stretch")
