# app.py
import os
import re
import time
from io import StringIO
from typing import Dict

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ---------------- Security / hosting notes ----------------
# - Set APP_PASSWORD env/secret to require a password. Leave unset to keep it public.
# - Do NOT print or log API keys. We keep keys in session_state only.
# - Cache is memory-only for 24h and keyed by (symbol, api_key). No disk persistence.

# ---------------- Config defaults ----------------
DEFAULT_SHORT = 20
DEFAULT_MID = 50
DEFAULT_LONG = 300
DEFAULT_MOMENTUM = 5
DEFAULT_GAP_SHORT = 1.0
DEFAULT_GAP_LONG  = 3.0
RATE_LIMIT_SECONDS_ALL = 20   # spacing between symbols when fetching ALL
SESSION_FETCH_THROTTLE = 10   # min seconds between fetch button presses per user/session

st.set_page_config(page_title="Stock Analyzer + Forecasts", layout="wide")
st.title("📈 Stock Analyzer (MA20 / MA50 / MA300) + 🔮 Forecasts")

# ---------------- Optional Password Gate ----------------
APP_PASSWORD = os.environ.get("APP_PASSWORD") or st.secrets.get("APP_PASSWORD", "")
if APP_PASSWORD:
    if st.session_state.get("_auth_ok") is not True:
        with st.sidebar:
            st.subheader("Authentication")
            pw = st.text_input("Password", type="password")
            if st.button("Enter"):
                if pw == APP_PASSWORD:
                    st.session_state["_auth_ok"] = True
                else:
                    st.error("Wrong password.")
        if st.session_state.get("_auth_ok") is not True:
            st.stop()

# ---------------- Session state ----------------
if "tickers" not in st.session_state:
    st.session_state.tickers: list[str] = []
if "data_store" not in st.session_state:
    st.session_state.data_store: Dict[str, pd.DataFrame] = {}
if "analysis_store" not in st.session_state:
    st.session_state.analysis_store: Dict[str, dict] = {}
if "selected_symbol" not in st.session_state:
    st.session_state.selected_symbol = None
if "last_fetch_ts" not in st.session_state:
    st.session_state.last_fetch_ts = 0.0

# Soft quota tracker (optional UX only; not logged/persisted)
import datetime as _dt
if "quota" not in st.session_state:
    st.session_state.quota = {"date": _dt.date.today(), "count": 0, "limit": 25}
elif st.session_state.quota["date"] != _dt.date.today():
    st.session_state.quota.update({"date": _dt.date.today(), "count": 0})

# ---------------- Helpers: validation & throttling ----------------
TICKER_RE = re.compile(r"^[A-Z.\-]{1,10}$")

def sanitize_ticker(raw: str) -> str | None:
    s = (raw or "").upper().strip()
    return s if TICKER_RE.match(s) else None

def throttle_ok() -> bool:
    now = time.time()
    if now - st.session_state.last_fetch_ts < SESSION_FETCH_THROTTLE:
        wait = int(SESSION_FETCH_THROTTLE - (now - st.session_state.last_fetch_ts))
        st.warning(f"Please wait ~{max(wait,1)}s before another fetch.")
        return False
    st.session_state.last_fetch_ts = now
    return True

# ---------------- Data fetching (robust + cached in-memory) ----------------
# IMPORTANT: cache_data persist=False ensures nothing is written to disk.
@st.cache_data(show_spinner=False, ttl=60*60*24, persist=False)
def fetch_symbol_csv(symbol: str, api_key: str, retries: int = 3, base_wait: int = 20) -> pd.DataFrame:
    """
    Fetch Alpha Vantage TIME_SERIES_DAILY as CSV and return a tidy DataFrame.
    Detects rate-limit/error responses, retries with backoff, and only parses valid CSV.
    The cache key includes 'api_key', so different users' keys don't share results.
    """
    import time as _t
    symbol = symbol.upper().strip()
    url = (
        "https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY&symbol={symbol}"
        f"&outputsize=full&datatype=csv&apikey={api_key}"
    )

    last_err = None
    for attempt in range(1, retries + 1):
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        text = r.text.strip()

        # CSV sanity
        first_line = text.splitlines()[0].lower() if text else ""
        looks_like_csv = first_line.startswith("timestamp")
        low = text.lower()
        throttled = ("thank you for using alpha vantage" in low) or ("error message" in low) or ("note" in low)

        if looks_like_csv and not throttled:
            try:
                df = pd.read_csv(StringIO(text), parse_dates=["timestamp"])
                if df.empty or "close" not in df.columns:
                    last_err = ValueError(f"{symbol}: unexpected CSV format.")
                else:
                    df = df.sort_values("timestamp").set_index("timestamp")
                    return df
            except Exception as e:
                last_err = e

        # Not CSV / throttled → backoff
        last_err = RuntimeError(
            f"{symbol}: API not ready (attempt {attempt}/{retries}). "
            f"{text[:160].replace('\\n',' ')}"
        )
        if attempt < retries:
            wait = base_wait * attempt  # 20s, 40s, 60s...
            _t.sleep(wait)

    raise last_err or RuntimeError(f"{symbol}: Unknown fetch error.")

# ---------------- Analysis ----------------
def analyze_symbol(
    df: pd.DataFrame,
    short_window: int,
    mid_window: int,
    long_window: int,
    momentum_window: int,
    gap_threshold_pct: float,
    long_gap_threshold_pct: float
) -> dict:
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column.")
    need = max(short_window, mid_window, long_window, momentum_window) + 1
    if len(df) < need:
        raise ValueError(f"Need at least {need} rows for indicators.")

    tmp = df.copy()
    tmp["ma_short"] = tmp["close"].rolling(short_window, min_periods=short_window).mean()
    tmp["ma_mid"]   = tmp["close"].rolling(mid_window,   min_periods=mid_window).mean()
    tmp["ma_long"]  = tmp["close"].rolling(long_window,  min_periods=long_window).mean()

    last = tmp.iloc[-1]
    prev = tmp.iloc[-momentum_window - 1]

    momentum    = float(last["close"] - prev["close"])
    short_slope = float(last["ma_short"] - tmp["ma_short"].iloc[-momentum_window - 1])
    long_slope  = float(last["ma_long"]  - tmp["ma_long"].iloc[-momentum_window - 1])

    gap_short = float(last["close"] - last["ma_short"])
    gap_short_pct = float(gap_short / last["ma_short"] * 100) if pd.notna(last["ma_short"]) else float("nan")

    gap_long = float(last["close"] - last["ma_long"])
    gap_long_pct = float(gap_long / last["ma_long"] * 100) if pd.notna(last["ma_long"]) else float("nan")

    bull_stack = last["close"] > last["ma_short"] > last["ma_mid"] > last["ma_long"]
    bear_stack = last["close"] < last["ma_short"] < last["ma_mid"] < last["ma_long"]

    if momentum > 0 and short_slope > 0:
        direction = "UP"
    elif momentum < 0 and short_slope < 0:
        direction = "DOWN"
    else:
        direction = "SIDEWAYS"

    cond_buy = bull_stack and short_slope > 0 and long_slope >= 0 and gap_short_pct > gap_threshold_pct
    cond_sell = bear_stack and short_slope < 0 and long_slope <= 0 and gap_short_pct < -gap_threshold_pct

    if cond_buy:
        signal = "BUY"
        reason = "Bullish stack (P>MA_s>MA_m>MA_l), MAs rising, price above short MA."
    elif cond_sell:
        signal = "SELL"
        reason = "Bearish stack (P<MA_s<MA_m<MA_l), MAs falling, price below short MA."
    else:
        signal = "HOLD"
        if pd.notna(gap_long_pct) and gap_long_pct >= long_gap_threshold_pct and long_slope > 0:
            reason = "Uptrend; price extended above long MA—risk of pullback."
        elif pd.notna(gap_long_pct) and gap_long_pct <= -long_gap_threshold_pct and long_slope < 0:
            reason = "Downtrend; price extended below long MA—watch for bear rallies."
        else:
            reason = "Mixed/neutral; wait for clearer alignment or pullback."

    return {
        "as_of": tmp.index[-1].date().isoformat(),
        "close": float(last["close"]),
        "ma_short": float(last["ma_short"]),
        "ma_mid": float(last["ma_mid"]),
        "ma_long": float(last["ma_long"]),
        "short_window": short_window,
        "mid_window": mid_window,
        "long_window": long_window,
        "momentum_window": momentum_window,
        "momentum": float(momentum),
        "short_slope": float(short_slope),
        "long_slope": float(long_slope),
        "gap_short": float(gap_short),
        "gap_short_pct": float(gap_short_pct),
        "gap_long": float(gap_long),
        "gap_long_pct": float(gap_long_pct),
        "direction": direction,
        "signal": signal,
        "reason": reason,
    }

# ---------------- Forecasting ----------------
def ensure_business_freq(series: pd.Series) -> pd.Series:
    s = series.asfreq("B")
    return s.ffill()

def forecast_naive(series: pd.Series, horizon: int) -> pd.Series:
    last = series.iloc[-1]
    idx = pd.bdate_range(series.index[-1] + pd.offsets.BDay(), periods=horizon)
    return pd.Series([last]*horizon, index=idx, name="forecast")

def forecast_sma(series: pd.Series, horizon: int, window: int = 20) -> pd.Series:
    sma = series.rolling(window, min_periods=window).mean().iloc[-1]
    idx = pd.bdate_range(series.index[-1] + pd.offsets.BDay(), periods=horizon)
    return pd.Series([float(sma)]*horizon, index=idx, name="forecast")

def forecast_holt(series: pd.Series, horizon: int, damped: bool = False) -> pd.Series:
    model = ExponentialSmoothing(
        series, trend="add", seasonal=None, damped_trend=damped, initialization_method="heuristic"
    ).fit(optimized=True)
    idx = pd.bdate_range(series.index[-1] + pd.offsets.BDay(), periods=horizon)
    fc = model.forecast(horizon)
    fc.index = idx
    return fc.rename("forecast")

def walk_forward_backtest(series: pd.Series, model: str, horizon: int, train_min: int = 300, sma_window: int = 20):
    y = series.copy()
    preds, actuals = [], []
    steps = min(252, max(1, len(y) - train_min - horizon))
    for i in range(steps):
        end = len(y) - (steps - i)
        train = y.iloc[:end]
        if len(train) < train_min:
            continue
        if model == "Naive":
            fc = forecast_naive(train, horizon)
        elif model == "SMA":
            fc = forecast_sma(train, horizon, window=sma_window)
        else:
            fc = forecast_holt(train, horizon)
        test_idx = fc.index
        if test_idx[-1] > y.index[-1]:
            continue
        actual = y.reindex(test_idx)
        mask = actual.notna()
        preds.extend(fc[mask].values)
        actuals.extend(actual[mask].values)
    if not preds:
        return np.nan, np.nan
    err = np.array(preds) - np.array(actuals)
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    return mae, rmse

def is_flat(series: pd.Series, tol: float = 1e-9) -> bool:
    if len(series) == 0:
        return True
    return (np.nanmax(series.values) - np.nanmin(series.values)) < tol

# ---------------- Sidebar UI ----------------
with st.sidebar:
    st.header("Settings / Privacy")
    # Users bring their own key; keep it session-only, masked
    api_key = st.text_input("Alpha Vantage API Key (kept in your session)", type="password")
    st.caption(f"API usage today (best-effort): {st.session_state.quota['count']}/{st.session_state.quota['limit']} calls")

    st.subheader("Indicators")
    short_w = st.number_input("Short MA window", 5, 200, DEFAULT_SHORT, step=1)
    mid_w   = st.number_input("Mid MA window",   10, 300, DEFAULT_MID, step=1)
    long_w  = st.number_input("Long MA window",  50, 600, DEFAULT_LONG, step=1)
    mom_w   = st.number_input("Momentum window", 1,  60,  DEFAULT_MOMENTUM, step=1)
    gap_s   = st.number_input("Gap% vs short MA (confirm)", 0.0, 10.0, DEFAULT_GAP_SHORT, step=0.1)
    gap_l   = st.number_input("Gap% vs long MA (stretched)", 0.0, 20.0, DEFAULT_GAP_LONG,  step=0.5)

    st.divider()
    st.subheader("Manage tickers")
    new_ticker_raw = st.text_input("Add ticker (e.g., AMD)").upper().strip()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("➕ Add"):
            sym = sanitize_ticker(new_ticker_raw)
            if not sym:
                st.warning("Ticker must be A–Z . - (max 10 chars).")
            elif sym in st.session_state.tickers:
                st.warning(f"{sym} is already in the list.")
            else:
                st.session_state.tickers.append(sym)
    with col_b:
        rm_choice = st.multiselect("Remove tickers", st.session_state.tickers, [])
        if st.button("🗑️ Remove selected"):
            st.session_state.tickers = [t for t in st.session_state.tickers if t not in rm_choice]
            for sym in rm_choice:
                st.session_state.data_store.pop(sym, None)
                st.session_state.analysis_store.pop(sym, None)
                if st.session_state.selected_symbol == sym:
                    st.session_state.selected_symbol = None

    st.divider()
    st.subheader("Fetching")
    fetch_all_btn = st.button("⬇️ Fetch ALL tickers")
    fetch_one_choice = st.selectbox("Ticker to fetch", st.session_state.tickers or ["—"], index=0)
    fetch_one_btn = st.button("⬇️ Fetch SELECTED")

    st.divider()
    analyze_btn = st.button("🧠 Analyze (all fetched)")

    st.divider()
    st.subheader("Forecasting")
    model_choice = st.selectbox("Model", ["Naive", "SMA", "Holt (trend)"])
    horizon = st.number_input("Horizon (business days)", 1, 60, 10, step=1)
    sma_k = st.number_input("SMA window (for SMA model)", 5, 100, 20, step=1)
    damped = st.checkbox("Damped trend (Holt)", value=False)
    bt = st.checkbox("Run backtest (rolling)", value=True)
    forecast_btn = st.button("🔮 Forecast")

# ---------------- Actions: Fetch ALL ----------------
if fetch_all_btn:
    if not api_key:
        st.error("Please enter your Alpha Vantage API key.")
    elif not st.session_state.tickers:
        st.warning("Add at least one ticker.")
    elif not throttle_ok():
        pass
    else:
        with st.spinner("Fetching data for ALL tickers..."):
            for i, sym in enumerate(st.session_state.tickers, start=1):
                try:
                    df = fetch_symbol_csv(sym, api_key, retries=3, base_wait=RATE_LIMIT_SECONDS_ALL)
                    st.session_state.data_store[sym] = df
                    st.success(f"[{i}/{len(st.session_state.tickers)}] {sym}: {len(df)} rows.")
                except Exception as e:
                    st.error(f"{sym}: {e}")
                if i < len(st.session_state.tickers):
                    time.sleep(RATE_LIMIT_SECONDS_ALL)
        options_now = sorted(st.session_state.data_store.keys())
        if options_now and (st.session_state.selected_symbol not in options_now):
            st.session_state.selected_symbol = options_now[0]

# ---------------- Actions: Fetch SELECTED ----------------
if fetch_one_btn:
    if not api_key:
        st.error("Please enter your Alpha Vantage API key.")
    elif not st.session_state.tickers:
        st.warning("Add at least one ticker.")
    elif not throttle_ok():
        pass
    else:
        sym = fetch_one_choice if fetch_one_choice != "—" else None
        if not sym:
            st.warning("Choose a ticker to fetch.")
        else:
            with st.spinner(f"Fetching data for {sym}..."):
                try:
                    df = fetch_symbol_csv(sym, api_key, retries=3, base_wait=RATE_LIMIT_SECONDS_ALL)
                    st.session_state.data_store[sym] = df
                    st.success(f"{sym}: {len(df)} rows.")
                    options_now = sorted(st.session_state.data_store.keys())
                    if options_now and (st.session_state.selected_symbol not in options_now):
                        st.session_state.selected_symbol = options_now[0]
                except Exception as e:
                    st.error(f"{sym}: {e}")

# ---------------- Actions: Analyze ALL fetched ----------------
def analyze_all():
    if not st.session_state.data_store:
        st.warning("Fetch data first.")
        return
    with st.spinner("Running analysis..."):
        for sym, df in st.session_state.data_store.items():
            try:
                res = analyze_symbol(
                    df,
                    short_window=short_w,
                    mid_window=mid_w,
                    long_window=long_w,
                    momentum_window=mom_w,
                    gap_threshold_pct=gap_s,
                    long_gap_threshold_pct=gap_l,
                )
                st.session_state.analysis_store[sym] = res
            except Exception as e:
                st.error(f"{sym}: analysis failed — {e}")
    st.success("Analysis complete.")

if analyze_btn:
    analyze_all()

# ---------------- Main content ----------------
if st.session_state.tickers:
    st.subheader("Tickers")
    st.write(" • ".join(st.session_state.tickers))

left, right = st.columns([2, 1])

with left:
    if st.session_state.data_store:
        options = sorted(st.session_state.data_store.keys())
        if options:
            if (st.session_state.selected_symbol is None) or (st.session_state.selected_symbol not in options):
                st.session_state.selected_symbol = options[0]

            st.selectbox("Select symbol", options, key="selected_symbol")
            sym = st.session_state.selected_symbol

            df = st.session_state.data_store[sym].copy()
            df[f"MA{short_w}"] = df["close"].rolling(short_w, min_periods=short_w).mean()
            df[f"MA{mid_w}"]   = df["close"].rolling(mid_w,   min_periods=mid_w).mean()
            df[f"MA{long_w}"]  = df["close"].rolling(long_w,  min_periods=long_w).mean()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df["close"], mode="lines", name="Close"))
            fig.add_trace(go.Scatter(x=df.index, y=df[f"MA{short_w}"], mode="lines", name=f"MA{short_w}"))
            fig.add_trace(go.Scatter(x=df.index, y=df[f"MA{mid_w}"],   mode="lines", name=f"MA{mid_w}"))
            fig.add_trace(go.Scatter(x=df.index, y=df[f"MA{long_w}"],  mode="lines", name=f"MA{long_w}"))
            fig.update_layout(height=500, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)

            st.caption("Tip: adjust windows in the sidebar, then click **Analyze** for updated signals.")

            # Forecast action
            if forecast_btn:
                close = ensure_business_freq(st.session_state.data_store[sym]["close"])
                with st.spinner("Forecasting..."):
                    if model_choice == "Naive":
                        fc = forecast_naive(close, horizon); model_tag = "Naive"
                    elif model_choice == "SMA":
                        fc = forecast_sma(close, horizon, window=sma_k); model_tag = "SMA"
                    else:
                        fc = forecast_holt(close, horizon, damped=damped); model_tag = "Holt"

                    if bt:
                        mae, rmse = walk_forward_backtest(
                            close, model=model_tag, horizon=horizon, train_min=max(300, DEFAULT_LONG), sma_window=sma_k
                        )
                    else:
                        mae, rmse = (np.nan, np.nan)

                fig2 = go.Figure()
                hist = close.last("365D")
                fig2.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines", name=f"{sym} Close"))
                fig2.add_trace(go.Scatter(x=fc.index, y=fc.values, mode="lines", name=f"{model_choice} forecast"))
                fig2.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
                st.plotly_chart(fig2, use_container_width=True)

                colA, colB = st.columns(2)
                colA.metric("MAE (backtest)", "—" if np.isnan(mae) else f"{mae:.2f}")
                colB.metric("RMSE (backtest)", "—" if np.isnan(rmse) else f"{rmse:.2f}")
                if is_flat(fc):
                    st.info("Forecast looks flat. Try turning OFF damping, increasing horizon, or using Holt.")
                st.caption("Backtest uses rolling-origin over recent data. Forecasts are educational only.")

with right:
    st.subheader("Latest analysis")
    sel = st.session_state.selected_symbol
    if sel and sel in st.session_state.analysis_store:
        res = st.session_state.analysis_store[sel]

        col1, col2, col3 = st.columns(3)
        col1.metric("Close", f"{res['close']:.2f}")
        col2.metric(f"MA{res['short_window']}", f"{res['ma_short']:.2f}")
        col3.metric(f"MA{res['long_window']}", f"{res['ma_long']:.2f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Gap vs short MA", f"{res['gap_short']:.2f}", f"{res['gap_short_pct']:.2f}%")
        col5.metric("Gap vs long MA", f"{res['gap_long']:.2f}", f"{res['gap_long_pct']:.2f}%")
        col6.metric("Direction", res["direction"])

        sig = res["signal"]
        icon = {"BUY": "✅", "SELL": "🛑", "HOLD": "⏸️"}.get(sig, "ℹ️")
        st.markdown(f"### {icon} Signal: **{sig}**")
        st.write(res["reason"])
        st.caption(f"As of {res['as_of']}. Educational only — not financial advice.")
    else:
        st.caption("Run **Analyze** to see metrics for the selected symbol.")

# No download/logging endpoints; no persistence to disk.
