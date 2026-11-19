"""NSX Volatility Forecaster: Streamlit front end for the Honours project.

Run:  streamlit run app.py
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nsxlib import lstm as L
from nsxlib import models as M
from nsxlib.backtest import backtest
from nsxlib.data import DataError, clean_prices, read_table, simulate_prices, suggest_columns

DEFAULT = Path(__file__).parent / "data" / "FTSE_NSX_INDEX_1.csv"
st.set_page_config(page_title="NSX Volatility Forecaster", layout="wide")

st.title("NSX Volatility Forecaster")
st.caption("Working prototype for the Honours project: Volatility Forecasting on a Frontier Market: "
           "GARCH, EGARCH and LSTM on the Namibian Stock Exchange Index"
           "Festus T. Amutenya, UNAM"
           "A RESERACH TOOL, NOT INVESTMENT ADVICE")

# ---------------------------------------------------------------- data
with st.sidebar:
    st.header("1. Data")
    _opts = ["Upload my own file", "Simulated example"] + (["NSX example (bundled)"] if DEFAULT.exists() else [])
    source = st.radio("Data source", _opts, index=len(_opts) - 1 if DEFAULT.exists() else 1)
    up = None
    if source.startswith("Upload"):
        up = st.file_uploader("CSV or Excel file with dates and prices (or returns)", type=["csv", "txt", "xlsx", "xls"])
        st.caption("Any column names, separators and date formats. Daily, weekly or monthly. "
                   "The app finds the columns; you can correct them below.")
    elif source.startswith("Simulated"):
        sim_n = st.slider("Number of days", 500, 4000, 1500, 250)
        sim_seed = st.number_input("Random seed", 1, 9999, 1)


@st.cache_data(show_spinner=False)
def _table(raw: bytes | None, name: str, which: str, n: int, seed: int):
    if which == "up":
        return read_table(io.BytesIO(raw), name)
    if which == "sim":
        return simulate_prices(n, seed)
    return read_table(str(DEFAULT))


try:
    if source.startswith("Upload"):
        if up is None:
            st.info("Upload a CSV or Excel file in the sidebar, or choose an example.")
            st.stop()
        table = _table(up.getvalue(), up.name, "up", 0, 0)
    elif source.startswith("Simulated"):
        table = _table(None, "", "sim", int(sim_n), int(sim_seed))
    else:
        table = _table(None, "", "nsx", 0, 0)
    guess = suggest_columns(table)
    with st.sidebar:
        st.subheader("2. Columns")
        cols = list(table.columns)
        dsel = st.selectbox("Date column", ["(none)"] + cols,
                            index=(cols.index(guess["date"]) + 1) if guess["date"] in cols else 0)
        pool = guess["numeric"] or cols
        vsel = st.selectbox("Value column", pool, index=pool.index(guess["value"]) if guess["value"] in pool else 0)
        ksel = st.selectbox("Values are", ["Detect automatically", "Prices / index levels",
                                           "Returns in percent", "Returns as decimals (0.012 = 1.2%)"])
    kind = {"Detect automatically": "auto", "Prices / index levels": "prices", "Returns in percent": "returns",
            "Returns as decimals (0.012 = 1.2%)": "returns_decimal"}[ksel]
    df, rep = clean_prices(table, dsel, vsel, kind)
except DataError as e:
    st.error(f"Data problem: {e}")
    st.stop()
except Exception as e:  # last-resort guard so the demo never shows a stack trace
    st.error(f"Unexpected problem reading the file: {e}")
    st.stop()

PPY, UNIT = rep.periods_per_year, rep.unit
n_obs = len(df)
st.session_state.setdefault("bt_key", None)
if st.session_state["bt_key"] != (source, getattr(up, "name", None), dsel, vsel, kind, n_obs):
    st.session_state["bt_key"] = (source, getattr(up, "name", None), dsel, vsel, kind, n_obs)
    st.session_state.pop("bt", None)

with st.expander(f"Data checks: {n_obs:,} {rep.freq_label.split(' ')[0]} returns, "
                 f"{df['Date'].iloc[0]:%d %b %Y} to {df['Date'].iloc[-1]:%d %b %Y}", expanded=bool(rep.warnings)):
    for line in rep.summary_lines():
        st.write("• " + line)
    d = df["Return"].describe()
    st.write(f"Mean {d['mean']:.3f}%, SD {d['std']:.3f}%, min {d['min']:.2f}%, max {d['max']:.2f}%")
    st.dataframe(table.head(8), width="stretch")

tab1, tab2, tab3 = st.tabs(["Forecast next period", "Backtest (compare models)", "About / method"])
ann_label = f"Annualised (%), {PPY} per year"

# ---------------------------------------------------------------- forecast
with tab1:
    c1, c2, c3 = st.columns(3)
    expanding = c1.checkbox("Use all data (expanding window)", value=False)
    default_w = int(min(1000, n_obs))
    wsize = c1.number_input(f"Window ({UNIT}s)", M.MIN_FIT_OBS, n_obs, max(M.MIN_FIT_OBS, default_w), 50,
                            disabled=expanding)
    model_choice = c2.multiselect("Models", ["GARCH", "EGARCH", "EWMA"], default=["GARCH", "EGARCH", "EWMA"])
    lam = c3.slider("EWMA lambda", 0.80, 0.99, 0.94, 0.01)
    y = df["Return"].to_numpy()
    sub = y if expanding else y[-int(wsize):]
    if not model_choice:
        st.info("Select at least one model.")
    else:
        rows, warns = [], []
        for k in model_choice:
            try:
                if k == "EWMA":
                    v = M.ewma_next(sub, lam)
                    rows.append({"Model": k, f"Next-{UNIT} volatility (%)": v ** 0.5,
                                 ann_label: M.annualise(v ** 0.5, PPY), "Status": "OK"})
                else:
                    f = M.fit(sub, k)
                    rows.append({"Model": k, f"Next-{UNIT} volatility (%)": f.next_vol,
                                 ann_label: M.annualise(f.next_vol, PPY),
                                 "Status": "OK" if f.stable else "Check warnings"})
                    warns += [f"{k}: {w}" for w in f.warnings]
            except Exception as e:
                rows.append({"Model": k, f"Next-{UNIT} volatility (%)": np.nan, ann_label: np.nan,
                             "Status": f"Failed: {e}"})
        res = pd.DataFrame(rows)
        st.dataframe(res.style.format({f"Next-{UNIT} volatility (%)": "{:.3f}", ann_label: "{:.1f}"}),
                     hide_index=True, width="stretch")
        for w in warns:
            st.warning(w)
        st.download_button("Download forecast (CSV)", res.to_csv(index=False).encode(), "forecast.csv", "text/csv")
        recent = df.tail(min(250, n_obs))
        fig = go.Figure(go.Bar(x=recent["Date"], y=recent["Return"], name="Return (%)"))
        fig.update_layout(title=f"Last {len(recent)} returns", height=300, margin=dict(t=40, b=10))
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- backtest
with tab2:
    max_test = max(50, n_obs - M.MIN_FIT_OBS)
    c1, c2, c3, c4 = st.columns(4)
    n_test = c1.number_input(f"Test {UNIT}s", 50, max_test, min(500, max(50, n_obs // 3), max_test), 50)
    win_mode = c2.selectbox("Window", ["fixed length", "expanding"])
    win_len = c2.number_input(f"Window length ({UNIT}s)", M.MIN_FIT_OBS, max(M.MIN_FIT_OBS, n_obs), 
                              int(min(1000, max(M.MIN_FIT_OBS, n_obs - int(n_test) - 5))), 50,
                              disabled=win_mode == "expanding")
    every = c3.number_input(f"Refit every ({UNIT}s)", 5, 500, 100, 5)
    use_lstm = c4.checkbox("Include LSTM (slower)", value=False, disabled=not L.AVAILABLE,
                           help=None if L.AVAILABLE else "TensorFlow not installed")
    if st.button("Run backtest", type="primary"):
        window = "expanding" if win_mode == "expanding" else int(win_len)
        bar = st.progress(0.0, "Fitting models...")
        try:
            extra = None
            if use_lstm:
                bar.progress(0.05, "Training LSTM (about a minute)...")
                extra = {"LSTM": L.lstm_forecast(df["Return"].to_numpy(), int(n_test), epochs=20)}
            out = backtest(df, int(n_test), window, int(every), extra=extra,
                           progress=lambda p: bar.progress(min(0.1 + 0.9 * p, 1.0)))
            bar.empty()
            st.session_state["bt"] = out
        except (ValueError, RuntimeError) as e:
            bar.empty()
            st.error(str(e))
    out = st.session_state.get("bt")
    if out is not None:
        st.subheader("Accuracy (lower is better)")
        st.dataframe(out.table.style.format({"QLIKE": "{:.3f}", "MSE": "{:.2f}"}), hide_index=True,
                     width="stretch")
        st.caption("QLIKE and MSE are measured against the squared return, as in the report.")
        st.subheader("Diebold-Mariano tests against GARCH (QLIKE)")
        st.dataframe(out.dm.style.format({"DM statistic": "{:.2f}", "p-value": "{:.3f}"}), hide_index=True,
                     width="stretch")
        for n in out.notes:
            st.warning(n)
        fc = out.forecasts
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fc["Date"], y=np.sqrt(fc["r2"]), name="|return|", opacity=0.3, mode="lines"))
        for c in [c for c in fc.columns if c not in ("Date", "r2")]:
            fig.add_trace(go.Scatter(x=fc["Date"], y=np.sqrt(fc[c]), name=c, mode="lines"))
        fig.update_layout(title="Forecast volatility (%) vs absolute return", height=420, margin=dict(t=40))
        st.plotly_chart(fig, width="stretch")
        st.download_button("Download forecasts (CSV)", fc.to_csv(index=False).encode(), "backtest.csv", "text/csv")

# ---------------------------------------------------------------- about
with tab3:
    st.markdown("""
**Models.** GARCH(1,1) and EGARCH(1,1,1) estimated by maximum likelihood from several starting values;
EWMA with lambda 0.94; 60-day rolling and expanding variance as benchmarks; optional LSTM on the previous
60 squared returns.

**Evaluation.** One-step-ahead variance forecasts compared with the squared return using QLIKE and MSE;
Diebold-Mariano test (Newey-West, Harvey-Leybourne-Newbold correction).

**Built-in safeguard from the study.** A short, calm estimation window can give EGARCH a negative alpha and an
unstable forecast. The app warns when the window is short, when no ARCH effect is present, or when alpha is negative.
Try *Last 500 days* in the backtest to reproduce the failure, then switch to 1,000 days or expanding.

**Limits.** One series at a time, any frequency (daily, weekly, monthly), at least 251 observations; the squared return is a noisy proxy for true volatility; results describe the
sample used and are not a trading recommendation.
""")
