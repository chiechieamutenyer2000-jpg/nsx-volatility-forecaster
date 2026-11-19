"""Out-of-sample evaluation: rolling refits, QLIKE/MSE losses and the Diebold-Mariano test."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from . import models as M

EPS = 1e-8


def qlike(r2, h):
    h = np.maximum(np.asarray(h, float), EPS)
    return np.log(h) + np.asarray(r2, float) / h


def mse(r2, h):
    return (np.asarray(r2, float) - np.asarray(h, float)) ** 2


def dm_test(loss_a, loss_b, h: int = 1) -> tuple[float, float]:
    """Diebold-Mariano with Newey-West variance and Harvey-Leybourne-Newbold correction.

    Positive statistic: model B has the lower average loss (B is better).
    Returns (statistic, two-sided p-value).
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    if n < 20 or np.allclose(d, d[0]):
        return float("nan"), float("nan")
    dbar = d.mean()
    lag = int(math.floor(n ** (1 / 3)))
    dc = d - dbar
    var = np.dot(dc, dc) / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * np.dot(dc[k:], dc[:-k]) / n
    if var <= 0:
        return float("nan"), float("nan")
    stat = dbar / math.sqrt(var / n)
    hln = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat *= hln
    p = 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))
    return float(stat), float(p)


@dataclass
class BacktestResult:
    forecasts: pd.DataFrame       # Date, r2, one column per model (variance forecasts)
    table: pd.DataFrame           # per-model QLIKE, MSE, rank
    dm: pd.DataFrame              # pairwise vs the reference model
    notes: list


def backtest(df: pd.DataFrame, n_test: int = 500, window: int | str = 1000, refit_every: int = 100,
             kinds=("GARCH", "EGARCH"), extra: dict | None = None, progress=None) -> BacktestResult:
    """Forecast the last n_test days one step ahead.

    window: number of past days used for each fit, or 'expanding'.
    extra: optional {name: array of variance forecasts aligned to the test days} (e.g. LSTM).
    """
    y = df["Return"].to_numpy(float)
    N = len(y)
    if n_test < 50:
        raise ValueError("Use at least 50 test days.")
    if isinstance(window, int) and window < M.MIN_FIT_OBS:
        raise ValueError(f"Window must be at least {M.MIN_FIT_OBS} days.")
    first = N - n_test
    min_hist = window if isinstance(window, int) else 500
    if first < min_hist:
        raise ValueError(f"Not enough history: {first} days before the test period, but {min_hist} are needed. "
                         "Reduce the test length or the window.")
    if refit_every < 1:
        raise ValueError("Refit interval must be at least 1 day.")
    notes, out = [], {k: np.full(n_test, np.nan) for k in kinds}
    nstarts = list(range(0, n_test, refit_every))
    for bi, s0 in enumerate(nstarts):
        t0 = first + s0
        t1 = min(first + s0 + refit_every, N)
        lo = 0 if window == "expanding" else t0 - window
        for k in kinds:
            try:
                f = M.fit(y[lo:t0], k)
                serious = [w for w in f.warnings if "negative" in w or "beta is at or above" in w
                           or "No clear ARCH" in w]
                if serious:
                    notes.append(f"{k}: unreliable fit in block starting {df['Date'].iloc[t0]:%Y-%m-%d} "
                                 f"({serious[0]})")
                hv = M.filtered_variance(k, f.params, y[lo:t1])
                out[k][s0:t1 - first] = hv[-(t1 - t0):]
            except Exception as exc:
                notes.append(f"{k}: fit failed in block starting {df['Date'].iloc[t0]:%Y-%m-%d} ({exc}); "
                             "EWMA used for that block.")
        if progress:
            progress((bi + 1) / len(nstarts))
    ew = M.ewma_series(y)[first:]
    r2 = y[first:] ** 2
    fc = pd.DataFrame({"Date": df["Date"].iloc[first:].to_numpy(), "r2": r2})
    for k in kinds:
        v = out[k]
        bad = ~np.isfinite(v) | (v <= 0)
        v[bad] = ew[bad]                      # documented fallback
        fc[k] = v
    fc["EWMA"] = ew
    fc["Rolling60"] = [np.var(y[max(0, t - 60):t], ddof=1) for t in range(first, N)]
    fc["Expanding"] = [np.var(y[:t], ddof=1) for t in range(first, N)]
    for name, arr in (extra or {}).items():
        arr = np.asarray(arr, float)
        if len(arr) != n_test:
            raise ValueError(f"Forecasts for {name} have the wrong length.")
        fc[name] = arr
    cols = [c for c in fc.columns if c not in ("Date", "r2")]
    rows = []
    for c in cols:
        rows.append({"Model": c, "QLIKE": float(np.mean(qlike(fc["r2"], fc[c]))),
                     "MSE": float(np.mean(mse(fc["r2"], fc[c])))})
    tab = pd.DataFrame(rows).sort_values("QLIKE").reset_index(drop=True)
    tab["Rank (QLIKE)"] = np.arange(1, len(tab) + 1)
    ref = "GARCH" if "GARCH" in cols else cols[0]
    dmrows = []
    for c in cols:
        if c == ref:
            continue
        st, p = dm_test(qlike(fc["r2"], fc[ref]), qlike(fc["r2"], fc[c]))
        dmrows.append({"Compared with": ref, "Model": c, "DM statistic": st, "p-value": p,
                       "Reading": ("not significant" if not (p < 0.05) else
                                   (f"{c} better" if st > 0 else f"{ref} better"))})
    return BacktestResult(fc, tab, pd.DataFrame(dmrows), notes)
