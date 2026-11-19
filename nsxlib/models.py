"""GARCH(1,1), EGARCH(1,1,1), EWMA and rolling-variance forecasts, with stability diagnostics.

The next-day variance is computed from the model's own recursion (the filtered path), not from the
`arch` package's forecast() routine, because the project found the two can disagree for fragile fits.
"""
from __future__ import annotations

import math
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")
from dataclasses import dataclass, field

import numpy as np
from arch import arch_model
from statsmodels.stats.diagnostic import het_arch

SQ2PI = math.sqrt(2 / math.pi)
MIN_FIT_OBS = 250


@dataclass
class FitResult:
    kind: str
    params: dict
    converged: bool
    loglik: float
    n_obs: int
    arch_lm_p: float
    next_var: float                      # forecast of tomorrow's conditional variance (%^2)
    last_var: float                      # filtered variance for the last observation
    last_resid: float
    warnings: list = field(default_factory=list)

    @property
    def next_vol(self) -> float:
        return math.sqrt(self.next_var)

    @property
    def stable(self) -> bool:
        return not self.warnings


def _build(y, kind):
    if kind == "EGARCH":
        return arch_model(y, mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist="normal", rescale=False)
    return arch_model(y, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False)


def _starts(y, kind):
    mu, v = float(np.mean(y)), float(np.var(y))
    if kind == "GARCH":
        return [[mu, v * (1 - a - b), a, b] for a, b in [(0.05, 0.90), (0.10, 0.85), (0.15, 0.80), (0.03, 0.95)]]
    return [[mu, (1 - b) * math.log(v), a, g, b] for a in (0.05, 0.15, 0.30) for g in (-0.10, 0.0)
            for b in (0.80, 0.90, 0.97)]


def next_variance(kind: str, p: dict, y_last: float, h_last: float) -> float:
    """One step of the variance recursion from (y_last, h_last)."""
    e = y_last - p["mu"]
    if kind == "GARCH":
        return p["omega"] + p["alpha"] * e * e + p["beta"] * h_last
    z = e / math.sqrt(h_last)
    ln_h = p["omega"] + p["alpha"] * (abs(z) - SQ2PI) + p["gamma"] * z + p["beta"] * math.log(h_last)
    return math.exp(min(ln_h, 50.0))


def fit(y, kind: str) -> FitResult:
    """Fit by maximum likelihood from several starting values and keep the best converged solution."""
    y = np.asarray(y, dtype=float)
    if kind not in ("GARCH", "EGARCH"):
        raise ValueError("kind must be 'GARCH' or 'EGARCH'")
    if len(y) < MIN_FIT_OBS:
        raise ValueError(f"At least {MIN_FIT_OBS} observations are needed to fit {kind}; got {len(y)}.")
    if not np.all(np.isfinite(y)):
        raise ValueError("The return series contains missing or infinite values.")
    am = _build(y, kind)
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for sv in _starts(y, kind):
            try:
                r = am.fit(starting_values=np.array(sv), disp="off", show_warning=False,
                           options={"maxiter": 1000})
            except Exception:
                continue
            ok = r.convergence_flag == 0 and np.isfinite(r.loglikelihood)
            if ok and (best is None or r.loglikelihood > best.loglikelihood):
                best = r
    if best is None:
        raise RuntimeError(f"{kind} estimation did not converge from any starting value. "
                           "Try a longer estimation window.")
    names = list(best.params.index)
    p = {"mu": float(best.params["mu"]), "omega": float(best.params["omega"]),
         "alpha": float(best.params["alpha[1]"]), "beta": float(best.params["beta[1]"])}
    if kind == "EGARCH":
        p["gamma"] = float(best.params["gamma[1]"])
    h_last = float(np.asarray(best.conditional_volatility)[-1] ** 2)
    nv = next_variance(kind, p, float(y[-1]), h_last)
    lm = float(het_arch(y - np.mean(y), nlags=10)[1])
    w = diagnose(kind, p, lm, len(y))
    if not np.isfinite(nv) or nv <= 0:
        raise RuntimeError(f"{kind} produced an invalid forecast; try a different window.")
    return FitResult(kind, p, True, float(best.loglikelihood), len(y), lm, nv, h_last,
                     float(y[-1] - p["mu"]), w)


def diagnose(kind: str, p: dict, arch_lm_p: float, n: int) -> list[str]:
    """Plain-language warnings, taken from the project's findings about EGARCH instability."""
    w = []
    if n < 750:
        w.append(f"Short estimation window ({n} days). The study found windows of 500 days can be unreliable; "
                 "1,000 days or more is safer.")
    if arch_lm_p > 0.10:
        w.append(f"No clear ARCH effect in this window (ARCH-LM p = {arch_lm_p:.2f}). Volatility models "
                 "are poorly identified when the window is calm.")
    if kind == "GARCH":
        if p["alpha"] + p["beta"] >= 1:
            w.append("alpha + beta is at or above 1 (non-stationary variance).")
    else:
        if p["alpha"] < 0:
            w.append("EGARCH alpha is negative: the degenerate solution described in the study. "
                     "Treat this forecast as unreliable and use a longer window.")
        if p["beta"] >= 1:
            w.append("EGARCH beta is at or above 1: the variance recursion can explode.")
    return w


def ewma_series(y, lam: float = 0.94) -> np.ndarray:
    """Variance forecast for day t made with data before t (RiskMetrics)."""
    y = np.asarray(y, dtype=float)
    out = np.empty(len(y)); s2 = float(np.var(y[:min(50, len(y))])); out[0] = s2
    for t in range(1, len(y)):
        s2 = lam * s2 + (1 - lam) * y[t - 1] ** 2
        out[t] = s2
    return out


def ewma_next(y, lam: float = 0.94) -> float:
    s = ewma_series(y, lam)
    return lam * s[-1] + (1 - lam) * float(y[-1]) ** 2


def rolling_next(y, w: int) -> float:
    y = np.asarray(y, dtype=float)
    return float(np.var(y[-w:], ddof=1))


def annualise(daily_vol_pct: float, days: int = 252) -> float:
    return daily_vol_pct * math.sqrt(days)


def filtered_variance(kind: str, p: dict, y) -> np.ndarray:
    """Variance forecast for every day in y given fixed parameters p (uses only earlier data)."""
    y = np.asarray(y, dtype=float)
    am = _build(y, kind)
    names = ["mu", "omega", "alpha", "gamma", "beta"] if kind == "EGARCH" else ["mu", "omega", "alpha", "beta"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = am.fix(np.array([p[n] for n in names]))
    return np.asarray(r.conditional_volatility) ** 2
