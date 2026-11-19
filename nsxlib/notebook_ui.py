"""Notebook front end (Google Colab / Jupyter) for the same library the Streamlit app uses.

Call launch() in a notebook cell. The window shows: data source, column choices, data checks, next-period
forecast, and a backtest. Everything is computed by nsxlib, so results match the Streamlit app.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from . import models as M
from .backtest import backtest
from .data import DataError, clean_prices, read_table, simulate_prices, suggest_columns


def forecast_table(df, rep, window="1000", models=("GARCH", "EGARCH", "EWMA"), lam=0.94):
    """Next-period volatility for each model. Returns (DataFrame, list of warning strings)."""
    y = df["Return"].to_numpy()
    sub = y if str(window) == "expanding" else y[-int(window):]
    rows, warns = [], []
    for k in models:
        try:
            if k == "EWMA":
                v = M.ewma_next(sub, lam)
                rows.append((k, v ** 0.5, M.annualise(v ** 0.5, rep.periods_per_year), "OK"))
            else:
                f = M.fit(sub, k)
                rows.append((k, f.next_vol, M.annualise(f.next_vol, rep.periods_per_year),
                             "OK" if f.stable else "Check warnings"))
                warns += [f"{k}: {w}" for w in f.warnings]
        except Exception as e:  # keep the notebook alive
            rows.append((k, np.nan, np.nan, f"Failed: {e}"))
    cols = ["Model", f"Next-{rep.unit} volatility (%)", f"Annualised (%), {rep.periods_per_year}/yr", "Status"]
    return pd.DataFrame(rows, columns=cols), warns


def launch(example_path: str | None = None):
    import ipywidgets as w
    from IPython.display import display
    import plotly.graph_objects as go

    state = {"table": None, "df": None, "rep": None}
    out = w.Output()
    source = w.ToggleButtons(options=["Upload a file", "Simulated example"] + (["NSX example"] if example_path else []),
                             description="Data:")
    up = w.FileUpload(accept=".csv,.txt,.xlsx,.xls", multiple=False, description="Choose file")
    n_sim = w.IntSlider(value=1500, min=500, max=4000, step=250, description="Days:")
    load_btn = w.Button(description="1. Load data", button_style="primary")
    date_dd = w.Dropdown(description="Date col:")
    val_dd = w.Dropdown(description="Value col:")
    kind_dd = w.Dropdown(options=[("Detect automatically", "auto"), ("Prices / index levels", "prices"),
                                  ("Returns in percent", "returns"), ("Returns as decimals", "returns_decimal")],
                         description="Values are:")
    check_btn = w.Button(description="2. Check data", button_style="primary")
    window = w.IntText(value=1000, description="Window:")
    expanding = w.Checkbox(value=False, description="Use all data (expanding)")
    fc_btn = w.Button(description="3. Forecast next period", button_style="success")
    n_test = w.IntText(value=500, description="Test obs:")
    refit = w.IntText(value=100, description="Refit every:")
    bt_btn = w.Button(description="4. Run backtest", button_style="success")

    def say(*items):
        with out:
            for it in items:
                display(it)

    def on_load(_):
        out.clear_output()
        try:
            if source.value == "Upload a file":
                if not up.value:
                    say("Choose a file first."); return
                item = up.value[0] if isinstance(up.value, (tuple, list)) else list(up.value.values())[0]
                content = item["content"] if isinstance(item, dict) else item.content
                name = item["name"] if isinstance(item, dict) else item.name
                state["table"] = read_table(io.BytesIO(bytes(content)), name)
            elif source.value == "Simulated example":
                state["table"] = simulate_prices(n_sim.value)
            else:
                state["table"] = read_table(example_path)
            g = suggest_columns(state["table"])
            cols = list(state["table"].columns)
            date_dd.options = ["(none)"] + cols
            date_dd.value = g["date"] if g["date"] in cols else "(none)"
            val_dd.options = g["numeric"] or cols
            val_dd.value = g["value"] if g["value"] in val_dd.options else val_dd.options[0]
            say(f"Loaded {len(state['table']):,} rows. Columns: {', '.join(map(str, cols))}. "
                "Check the two column choices above, then press 'Check data'.", state["table"].head(5))
        except DataError as e:
            say(f"Data problem: {e}")
        except Exception as e:
            say(f"Could not read the file: {e}")

    def on_check(_):
        out.clear_output()
        if state["table"] is None:
            say("Load data first."); return
        try:
            df, rep = clean_prices(state["table"], date_dd.value, val_dd.value, kind_dd.value)
        except DataError as e:
            say(f"Data problem: {e}"); return
        state["df"], state["rep"] = df, rep
        window.value = int(min(1000, len(df)))
        n_test.value = int(min(500, max(50, len(df) // 3)))
        say("\n".join("• " + ln for ln in rep.summary_lines()))

    def on_forecast(_):
        out.clear_output()
        if state["df"] is None:
            say("Run 'Check data' first."); return
        res, warns = forecast_table(state["df"], state["rep"], "expanding" if expanding.value else window.value)
        say(res.round(3), *["WARNING: " + x for x in warns])
        recent = state["df"].tail(250)
        fig = go.Figure(go.Bar(x=recent["Date"], y=recent["Return"]))
        fig.update_layout(title="Recent returns (%)", height=280, margin=dict(t=40, b=10))
        say(fig)

    def on_backtest(_):
        out.clear_output()
        if state["df"] is None:
            say("Run 'Check data' first."); return
        try:
            res = backtest(state["df"], int(n_test.value), "expanding" if expanding.value else int(window.value),
                           int(refit.value))
        except (ValueError, RuntimeError) as e:
            say(f"Cannot run the backtest: {e}"); return
        say("Accuracy against the squared return (lower is better):", res.table.round(3),
            "Diebold-Mariano tests:", res.dm.round(3), *["WARNING: " + n for n in res.notes])
        fc = res.forecasts
        fig = go.Figure([go.Scatter(x=fc["Date"], y=np.sqrt(fc["r2"]), name="|return|", opacity=0.3)])
        for c in [c for c in fc.columns if c not in ("Date", "r2")]:
            fig.add_trace(go.Scatter(x=fc["Date"], y=np.sqrt(fc[c]), name=c))
        fig.update_layout(title="Forecast volatility (%) vs absolute return", height=380, margin=dict(t=40))
        say(fig)

    load_btn.on_click(on_load); check_btn.on_click(on_check)
    fc_btn.on_click(on_forecast); bt_btn.on_click(on_backtest)
    display(w.VBox([w.HTML("<h3>NSX Volatility Forecaster</h3>"), source, up, n_sim, load_btn,
                    w.HBox([date_dd, val_dd, kind_dd]), check_btn,
                    w.HBox([window, expanding, fc_btn]), w.HBox([n_test, refit, bt_btn]), out]))
    return {"state": state, "on_load": on_load, "on_check": on_check, "on_forecast": on_forecast,
            "on_backtest": on_backtest, "widgets": dict(source=source, up=up)}
