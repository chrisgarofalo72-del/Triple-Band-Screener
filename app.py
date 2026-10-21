import io
import re
from datetime import datetime
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Triple EMA Band | NYSE + Nasdaq", layout="wide")
st.title("Triple EMA Band Stock Scanner")
st.warning("Experimental reconstruction, not the presenter's verified formula. Uses EMA(high), EMA(close), EMA(low). No results are trading advice.")
st.caption("Yahoo Finance delayed/adjusted daily price data; NYSE/Nasdaq symbol directories from Nasdaq Trader. Run after market close for completed daily bars. Large scans can be slow or rate-limited.")

DIRECTORY = "https://www.nasdaqtrader.com/dynamic/SymDir/"
DEFAULT = "AAPL MSFT NVDA GOOGL AMZN META TSLA AMD PLTR SOUN IONQ RGTI QBTS ORIC APD CRM AVGO INTC COIN NFLX MU SMCI AXON ARM"

@st.cache_data(ttl=86400, show_spinner=False)
def exchange_tickers():
    def get_table(filename):
        url = DIRECTORY + filename
        response = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text), sep="|", dtype=str)
        return df.fillna("")

    nas = get_table("nasdaqlisted.txt")
    oth = get_table("otherlisted.txt")
    nas = nas[(nas.get("Test Issue", "N") == "N") & (nas.get("ETF", "N") == "N")]
    oth = oth[(oth.get("Test Issue", "N") == "N") & (oth.get("ETF", "N") == "N") & (oth["Exchange"] == "N")]
    nas_symbols = nas["Symbol"].astype(str).tolist()
    nyse_symbols = oth["ACT Symbol"].astype(str).tolist()
    return sorted(set(nas_symbols)), sorted(set(nyse_symbols))

with st.sidebar:
    st.header("Scan universe")
    universe = st.selectbox("Choose stock universe", ["My tickers", "Nasdaq only", "NYSE only", "Nasdaq + NYSE"])
    manual = st.text_area("My tickers (for custom scan)", DEFAULT, height=140)
    scope = st.radio("Scan size", ["First 250 (test)", "First 1,000", "ALL eligible symbols"], horizontal=False)
    length = st.number_input("EMA band length", min_value=2, max_value=200, value=20)
    lookback = st.number_input("Signals in last N sessions", min_value=1, max_value=30, value=3)
    signal_type = st.selectbox("Signal", ["Close crosses ABOVE upper EMA", "Close crosses ABOVE center EMA", "Close crosses ABOVE lower EMA", "Close crosses BELOW lower EMA"])
    require_sma = st.checkbox("Require close above rising SMA 150", value=True)
    min_price = st.number_input("Minimum share price ($)", min_value=0.0, value=5.0)
    min_vol = st.number_input("Minimum 20-day average volume", min_value=0, value=500000, step=100000)
    latest_only = st.checkbox("Still on signal side of band", value=True)
    batch_size = st.select_slider("Symbols per download batch", options=[20, 30, 40, 50], value=30)

try:
    if universe == "My tickers":
        symbols = list(dict.fromkeys(x.strip().upper() for x in re.split(r"[\s,;]+", manual) if x.strip()))
    else:
        nas, nyse = exchange_tickers()
        if universe == "Nasdaq only": symbols = nas
        elif universe == "NYSE only": symbols = nyse
        else: symbols = sorted(set(nas + nyse))
except Exception as exc:
    st.error("Could not download the exchange's symbol list. Please try again later or select My tickers.")
    st.caption(str(exc))
    symbols = []

eligible_count = len(symbols)
if scope == "First 250 (test)": symbols = symbols[:250]
elif scope == "First 1,000": symbols = symbols[:1000]
st.info(f"Selected {len(symbols):,} of {eligible_count:,} symbols in {universe}. Exchange ticker counts exclude ETFs and test issues; some symbols may have insufficient price history or Yahoo Finance data.")

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_batch(symbols_tuple):
    tickers = list(symbols_tuple)
    raw = yf.download(tickers=tickers, period="1y", interval="1d", auto_adjust=True, progress=False, group_by="ticker", threads=True, timeout=30)
    if raw is None or raw.empty:
        return {}
    outputs = {}
    for ticker in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if ticker in raw.columns.get_level_values(0):
                    frame = raw[ticker].copy()
                else:
                    continue
            elif len(tickers) == 1:
                frame = raw.copy()
            else:
                continue
            cols = ["Open", "High", "Low", "Close", "Volume"]
            if not set(cols).issubset(frame.columns): continue
            frame = frame[cols].dropna().copy()
            if len(frame) >= 155: outputs[ticker] = frame
        except Exception:
            pass
    return outputs

def analyze(ticker, d):
    try:
        for col, source in [("Top", "High"), ("Middle", "Close"), ("Bottom", "Low")]:
            d[col] = d[source].ewm(span=int(length), adjust=False).mean()
        d["SMA150"] = d["Close"].rolling(150).mean()
        d["AvgVol20"] = d["Volume"].rolling(20).mean()
        target = {"Close crosses ABOVE upper EMA": "Top", "Close crosses ABOVE center EMA": "Middle", "Close crosses ABOVE lower EMA": "Bottom", "Close crosses BELOW lower EMA": "Bottom"}[signal_type]
        if "BELOW" in signal_type:
            sig = (d.Close.shift(1) >= d[target].shift(1)) & (d.Close < d[target])
        else:
            sig = (d.Close.shift(1) <= d[target].shift(1)) & (d.Close > d[target])
        recent = d.index[sig.fillna(False) & d.index.isin(d.index[-int(lookback):])]
        if not len(recent): return None
        last = d.iloc[-1]
        prev = d.iloc[-6]
        if last.Close < min_price or last.AvgVol20 < min_vol: return None
        if require_sma and not (last.Close > last.SMA150 and last.SMA150 > prev.SMA150): return None
        if latest_only and (not (last.Close < last[target]) if "BELOW" in signal_type else not (last.Close > last[target])): return None
        hit = recent[-1]
        return {"Ticker": ticker, "Signal date": str(hit.date()), "Sessions ago": int(len(d) - 1 - d.index.get_loc(hit)), "Close": round(float(last.Close),2), "Band top": round(float(last.Top),2), "Band center": round(float(last.Middle),2), "Band bottom": round(float(last.Bottom),2), "SMA150": round(float(last.SMA150),2), "Avg volume (20d)": int(last.AvgVol20), "Above top %": round(float(100 * (last.Close / last.Top - 1)), 2)}
    except Exception:
        return None

if st.button("Run scan", type="primary", disabled=not symbols):
    matches = []
    missing = []
    tested = 0
    progress = st.progress(0, text="Preparing scan...")
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        try:
            data = fetch_batch(tuple(batch))
        except Exception:
            data = {}
        for ticker in batch:
            if ticker not in data:
                missing.append(ticker)
                continue
            tested += 1
            result = analyze(ticker, data[ticker].copy())
            if result is not None: matches.append(result)
        progress.progress(min((i+len(batch))/len(symbols), 1.0), text=f"Checked {i+len(batch):,}/{len(symbols):,} symbols · {tested:,} with usable data · {len(matches)} matches")
    progress.empty()
    st.session_state["last_scan"] = {"results": pd.DataFrame(matches), "missing": missing, "tested": tested, "selected": len(symbols), "universe": universe}

if "last_scan" in st.session_state:
    run = st.session_state["last_scan"]
    st.subheader("Scan results")
    st.write(f"**{len(run['results']):,} matches** · {run['tested']:,} evaluated with usable price data · {len(run['missing']):,} missing/insufficient-data symbols (out of {run['selected']:,} selected).")
    st.caption("If stocks had missing data, this was not a complete market scan. Increasing the universe selection changes future scans, not the saved result displayed below.")
    if not run["results"].empty:
        out = run["results"].sort_values(["Sessions ago", "Ticker"])
        st.dataframe(out, use_container_width=True, hide_index=True)
        st.download_button("Download matches (CSV)", out.to_csv(index=False), file_name="triple_band_matches.csv", mime="text/csv")
        selected = st.selectbox("Plot a matched stock", out["Ticker"].tolist())
        d = fetch_batch((selected,)).get(selected)
        if d is not None:
            for col, source in [("Top", "High"), ("Middle", "Close"), ("Bottom", "Low")]:
                d[col] = d[source].ewm(span=int(length), adjust=False).mean()
            d["SMA150"] = d.Close.rolling(150).mean()
            fig = go.Figure(go.Candlestick(x=d.index, open=d.Open, high=d.High, low=d.Low, close=d.Close, name="Price"))
            for col, color in [("Top", "#3b82f6"), ("Middle", "#2563eb"), ("Bottom", "#93c5fd"), ("SMA150", "#ef4444")]:
                fig.add_trace(go.Scatter(x=d.index, y=d[col], name=col, line=dict(color=color)))
            fig.update_layout(template="plotly_dark", height=540, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No matches were found in the evaluated stocks using these settings.")
    if run["missing"]:
        with st.expander(f"Show {len(run['missing']):,} skipped/missing ticker symbols"):
            st.code(", ".join(run["missing"]))
            st.download_button("Download skipped tickers", "Ticker\n" + "\n".join(run["missing"]), file_name="triple_band_skipped.csv", mime="text/csv")

st.divider()
st.caption("A signal is identified when a completed daily close crosses the chosen EMA between consecutive sessions. Exact entry/exit rules in the Facebook video remain unverified. Thousands of downloads may be throttled by Yahoo Finance or exceed Streamlit Cloud resource/time limits; start with 250, then 1,000 before ALL.")
