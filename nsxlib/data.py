"""Data loading and validation for any price or return series.

The study used one file (FTSE/NSX Overall Index, Investing.com export), but the tool must work on other data too.
So this module (1) reads CSV or Excel files with different separators, encodings, number formats and date
formats, (2) finds the date column and the value column by itself (the user can override), (3) accepts prices or
returns, (4) detects the observation frequency so that results are annualised correctly, and (5) checks the
series and reports every change in plain words. The rest of the system only ever sees a clean, sorted
series of log returns in percent.
"""
from __future__ import annotations

import csv
import io
import math
import re
from dataclasses import dataclass, field
from typing import IO, Union

import numpy as np
import pandas as pd

MIN_RETURNS = 250          # fewest returns the models can be fitted on (same as models.MIN_FIT_OBS)
MAX_ABS_RETURN = 20.0      # a daily move above 20 % that reverses next day is treated as a data error
SPIKE_FLAG = 5.0           # |return| above this that reverses next period is flagged (not changed)

DATE_NAMES = ("date", "datetime", "time", "timestamp", "day", "trading date", "trade date", "period", "month")
PRICE_NAMES = ("adj close", "adj. close", "adjusted close", "close", "closing price", "close price", "price",
               "last", "last price", "level", "index", "value", "settle", "settlement", "nav")
RETURN_HINTS = ("return", "ret", "log ret", "logret")
NOT_A_PRICE = ("vol", "open", "high", "low", "change", "chg", "%", "turnover", "volume", "bid", "ask")


class DataError(ValueError):
    """The uploaded data cannot be used. The message is written for the end user."""


@dataclass
class ValidationReport:
    rows_read: int = 0
    rows_used: int = 0
    dropped_unreadable_date: int = 0
    dropped_bad_price: int = 0
    dropped_duplicates: int = 0
    corrected_prices: list = field(default_factory=list)   # (date, old, new)
    flagged_spikes: list = field(default_factory=list)     # dates
    warnings: list = field(default_factory=list)
    date_col: str | None = None
    value_col: str | None = None
    kind: str = "prices"                  # "prices" or "returns"
    freq_label: str = "daily"
    periods_per_year: int = 252
    dates_synthetic: bool = False
    notes: list = field(default_factory=list)

    @property
    def unit(self) -> str:
        return {"daily": "day", "weekly": "week", "monthly": "month", "quarterly": "quarter",
                "annual": "year"}.get(self.freq_label.split(" ")[0], "period")

    def summary_lines(self) -> list[str]:
        what = "returns" if self.kind == "returns" else "prices"
        out = [f"Used column '{self.value_col}' as {what}"
               + (f" and '{self.date_col}' as the date." if self.date_col else " (no date column found)."),
               f"Frequency detected: {self.freq_label} ({self.periods_per_year} periods a year, used for annualising).",
               f"Rows read: {self.rows_read}; rows used: {self.rows_used}."]
        if self.dropped_unreadable_date:
            out.append(f"Removed {self.dropped_unreadable_date} rows with an unreadable date.")
        if self.dropped_bad_price:
            out.append(f"Removed {self.dropped_bad_price} rows with a missing or non-numeric value"
                       + ("" if self.kind == "returns" else " or a non-positive price") + ".")
        if self.dropped_duplicates:
            out.append(f"Removed {self.dropped_duplicates} duplicate dates (last entry kept).")
        for d, old, new in self.corrected_prices:
            out.append(f"Corrected price on {d:%Y-%m-%d}: {old:,.1f} replaced by {new:,.1f} (mean of neighbours).")
        if self.flagged_spikes:
            out.append(f"Flagged {len(self.flagged_spikes)} spike-and-reversal observations (kept unchanged).")
        out.extend(self.notes)
        out.extend(self.warnings)
        return out


# ------------------------------------------------------------------ reading files
def _read_bytes(source) -> tuple[bytes, str]:
    if isinstance(source, (str,)):
        with open(source, "rb") as fh:
            return fh.read(), source.lower()
    name = getattr(source, "name", "") or ""
    data = source.read()
    if isinstance(data, str):
        data = data.encode("utf-8")
    return data, str(name).lower()


def _find_separator(text: str) -> str:
    """Pick the delimiter whose count is the same on every line (so decimal commas do not fool it)."""
    lines = [ln for ln in text.splitlines()[:40] if ln.strip()]
    for cand in (";", "\t", "|", ","):
        counts = {ln.count(cand) for ln in lines}
        if len(counts) == 1 and counts != {0}:
            return cand
    try:
        return csv.Sniffer().sniff(text[:8192], delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def read_table(source: Union[str, IO], filename: str | None = None) -> pd.DataFrame:
    """Read a CSV/TSV/semicolon file or an Excel sheet into a table of text (or numbers for Excel)."""
    data, name = _read_bytes(source)
    name = (filename or name or "").lower()
    if not data.strip():
        raise DataError("The file is empty.")
    if data[:2] == b"PK" or name.endswith((".xlsx", ".xlsm", ".xls")):
        try:
            raw = pd.read_excel(io.BytesIO(data), sheet_name=0)
        except Exception as exc:
            raise DataError("The Excel file could not be read. Save it as .xlsx or .csv and try again "
                            f"({exc.__class__.__name__}).") from None
        raw.columns = [str(c).strip() for c in raw.columns]
        return raw
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeError:
            continue
    sep = _find_separator(text)
    try:
        raw = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, skip_blank_lines=True)
    except pd.errors.EmptyDataError:
        raise DataError("The file is empty.") from None
    except pd.errors.ParserError as exc:
        raise DataError("The file could not be read as a table. Check that it is a CSV or Excel file with "
                        f"one header row ({exc.__class__.__name__}).") from None
    raw.columns = [str(c).strip() for c in raw.columns]
    return raw


# ------------------------------------------------------------------ parsing numbers and dates
_NONNUM = re.compile(r"[^0-9,.\-+eE()]")
_THOUS_COMMA = re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$")
_THOUS_DOT = re.compile(r"^[+-]?\d{1,3}(\.\d{3}){2,}(,\d+)?$")


_DATEISH = re.compile(r"^\s*\d{1,4}[-./]\d{1,2}[-./]\d{1,4}([ T].*)?$|[/:]|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", re.I)


def _one_number(x) -> float:
    if x is None:
        return math.nan
    raw = str(x).strip()
    if _DATEISH.search(raw):
        return math.nan                      # dates are not numbers ("17/01/2014" must not become 17012014)
    s = _NONNUM.sub("", raw)
    if not s:
        return math.nan
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        s = s.replace(",", "") if _THOUS_COMMA.match(s) else s.replace(",", ".")
    elif _THOUS_DOT.match(s):
        s = s.replace(".", "")
    try:
        v = float(s)
    except ValueError:
        return math.nan
    return -v if neg else v


def to_number(s: pd.Series) -> pd.Series:
    """Convert text such as '1,234.5', '1.234,5', '12%' or '(3.1)' to numbers; unreadable cells become NaN."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    return pd.Series([_one_number(v) for v in s], index=s.index, dtype=float)


def parse_dates(s: pd.Series) -> pd.Series:
    """Parse dates in ISO, day-first or month-first form (chosen by which reads more rows), or Excel serials."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce")
    if pd.api.types.is_numeric_dtype(s):
        v = pd.to_numeric(s, errors="coerce")
        if v.between(20000, 80000).mean() > 0.9:
            return pd.to_datetime(v, unit="D", origin="1899-12-30", errors="coerce")
        return pd.Series(pd.NaT, index=s.index)
    x = s.astype(str).str.strip()

    def parse(dayfirst: bool) -> pd.Series:
        try:
            return pd.to_datetime(x, errors="coerce", dayfirst=dayfirst, format="mixed")
        except (TypeError, ValueError):
            return pd.to_datetime(x, errors="coerce", dayfirst=dayfirst)

    a, b = parse(False), parse(True)
    na, nb = a.notna().sum(), b.notna().sum()
    if na != nb:
        return a if na > nb else b
    if (a.fillna(pd.Timestamp(0)) == b.fillna(pd.Timestamp(0))).all():
        return a                                   # unambiguous (e.g. ISO dates)

    def order_score(d: pd.Series) -> float:        # share of steps that go forward (or all backward)
        step = d.dropna().diff().dropna().dt.days
        if len(step) == 0:
            return 0.0
        return max((step >= 0).mean(), (step <= 0).mean())

    return a if order_score(a) >= order_score(b) else b


# ------------------------------------------------------------------ column detection
def suggest_columns(raw: pd.DataFrame) -> dict:
    """Guess which column holds the dates and which holds the values; list every usable numeric column."""
    names = {str(c).strip().lower(): c for c in raw.columns}
    date_col = None
    for n in DATE_NAMES:
        if n in names and parse_dates(raw[names[n]]).notna().mean() >= 0.8:
            date_col = names[n]
            break
    if date_col is None:
        best = 0.0
        for c in raw.columns:
            if pd.api.types.is_numeric_dtype(raw[c]) and not pd.api.types.is_datetime64_any_dtype(raw[c]):
                continue
            if not pd.api.types.is_datetime64_any_dtype(raw[c]) and to_number(raw[c]).notna().mean() >= 0.95:
                continue                      # a column of plain numbers (e.g. "99,20") is a value, not a date
            rate = parse_dates(raw[c]).notna().mean()
            if rate > max(best, 0.8):
                date_col, best = c, rate
    numeric = []
    for c in raw.columns:
        if c == date_col:
            continue
        ok = to_number(raw[c]).notna().mean()
        if ok >= 0.8:
            numeric.append(c)
    value_col = None
    lower = {c: str(c).strip().lower() for c in numeric}
    for n in PRICE_NAMES:
        hit = [c for c in numeric if lower[c] == n]
        if hit:
            value_col = hit[0]
            break
    if value_col is None:
        for c in numeric:
            if any(h in lower[c] for h in RETURN_HINTS):
                value_col = c
                break
    if value_col is None:
        plain = [c for c in numeric if not any(h in lower[c] for h in NOT_A_PRICE)]
        value_col = (plain or numeric or [None])[0]
    return {"date": date_col, "value": value_col, "numeric": numeric}


def _guess_kind(name: str, v: pd.Series) -> str:
    n = str(name).strip().lower()
    if any(h in n for h in RETURN_HINTS):
        return "returns"
    nonpos = (v.dropna() <= 0).mean() if v.notna().any() else 0
    return "returns" if nonpos > 0.01 else "prices"


# ------------------------------------------------------------------ frequency
def detect_frequency(dates: pd.Series) -> tuple[str, int]:
    gaps = dates.diff().dt.days.dropna()
    if gaps.empty:
        return "daily", 252
    med = float(gaps.median())
    if med <= 4:
        seven_day = med <= 1 and (gaps == 1).mean() > 0.9
        return ("daily (7-day market)", 365) if seven_day else ("daily", 252)
    if med <= 9:
        return "weekly", 52
    if 20 <= med <= 40:
        return "monthly", 12
    if 80 <= med <= 100:
        return "quarterly", 4
    if 350 <= med <= 380:
        return "annual", 1
    ppy = max(1, int(round(365.25 / med)))
    return f"irregular (about every {med:.0f} days)", ppy


# ------------------------------------------------------------------ main entry points
def load_prices(source: Union[str, IO], filename: str | None = None, date_col: str | None = None,
                value_col: str | None = None, kind: str = "auto") -> tuple[pd.DataFrame, ValidationReport]:
    """Read a CSV/Excel file of prices or returns and return (Date, Price, Return) plus a report.

    kind: 'auto', 'prices', 'returns' (percent) or 'returns_decimal' (0.012 means 1.2 %).
    Raises DataError with a readable message for any unusable input.
    """
    return clean_prices(read_table(source, filename), date_col, value_col, kind)


def clean_prices(raw: pd.DataFrame, date_col: str | None = None, value_col: str | None = None,
                 kind: str = "auto") -> tuple[pd.DataFrame, ValidationReport]:
    rep = ValidationReport(rows_read=len(raw))
    if raw is None or raw.empty:
        raise DataError("The file has no rows.")
    sug = suggest_columns(raw)
    date_col = date_col if date_col not in (None, "(none)") else sug["date"]
    value_col = value_col or sug["value"]
    if value_col is None or value_col not in raw.columns:
        raise DataError("No numeric column with prices or returns was found. "
                        f"Columns found: {', '.join(map(str, raw.columns))}.")
    if date_col is not None and date_col not in raw.columns:
        raise DataError(f"The date column '{date_col}' is not in the file.")
    rep.date_col, rep.value_col = date_col, value_col

    vals = to_number(raw[value_col])
    if kind == "auto":
        kind = _guess_kind(value_col, vals)
    rep.kind = "returns" if kind.startswith("returns") else "prices"

    if date_col is None:
        dates = pd.Series(pd.bdate_range("2000-01-03", periods=len(raw)), index=raw.index)
        rep.dates_synthetic = True
        rep.warnings.append("No date column was found, so the rows are treated as consecutive daily observations "
                            "in file order. Dates shown are placeholders.")
    else:
        dates = parse_dates(raw[date_col])
    df = pd.DataFrame({"Date": dates, "Value": vals})
    n = len(df)
    df = df.dropna(subset=["Date"]); rep.dropped_unreadable_date = n - len(df)
    n = len(df)
    ok = np.isfinite(df["Value"]) if rep.kind == "returns" else (np.isfinite(df["Value"]) & (df["Value"] > 0))
    df = df[ok]; rep.dropped_bad_price = n - len(df)
    n = len(df)
    if not rep.dates_synthetic:
        df = df.sort_values("Date")
    df = df.drop_duplicates("Date", keep="last"); rep.dropped_duplicates = n - len(df)
    df = df.reset_index(drop=True)
    if len(df) < MIN_RETURNS + (0 if rep.kind == "returns" else 1):
        raise DataError(f"Only {len(df)} usable observations were found; at least {MIN_RETURNS + 1} are needed "
                        "to fit the models reliably.")

    rep.freq_label, rep.periods_per_year = detect_frequency(df["Date"]) if not rep.dates_synthetic else ("daily", 252)
    big = MAX_ABS_RETURN * math.sqrt(max(1.0, 252 / rep.periods_per_year))   # looser limit for weekly/monthly data

    if rep.kind == "prices":
        for _ in range(5):                       # a few passes handle adjacent cases
            r = 100 * np.log(df["Value"] / df["Value"].shift(1))
            bad = np.where(r.abs().to_numpy() > big)[0]
            if len(bad) == 0:
                break
            i = int(bad[0])
            if i + 1 < len(df) and i >= 1 and r.iloc[i] * r.iloc[i + 1] < 0 \
                    and abs(r.iloc[i] + r.iloc[i + 1]) < 0.25 * abs(r.iloc[i]):
                old = float(df.loc[i, "Value"])
                new = float((df.loc[i - 1, "Value"] + df.loc[i + 1, "Value"]) / 2)
                rep.corrected_prices.append((df.loc[i, "Date"], old, new))
                df.loc[i, "Value"] = new
            else:
                break                            # a move that does not reverse is reported below, not changed
        df["Return"] = 100 * np.log(df["Value"] / df["Value"].shift(1))
        df["Price"] = df["Value"]
        df = df.dropna(subset=["Return"]).reset_index(drop=True)
    else:
        r = df["Value"].to_numpy(float)
        if kind == "returns_decimal" or (kind == "returns" and np.nanstd(r) < 0.15 and np.nanmax(np.abs(r)) < 1.0):
            df["Value"] = df["Value"] * 100
            rep.notes.append("Returns looked like decimals (0.012 = 1.2 %), so they were multiplied by 100.")
        df["Return"] = df["Value"]
        df["Price"] = 100 * np.exp(df["Return"].cumsum() / 100)
        rep.notes.append("Input was returns, so the price path in charts is rebuilt from them (start = 100). "
                         "Returns are treated as log returns in percent.")
    df = df[["Date", "Price", "Return"]].reset_index(drop=True)

    r = df["Return"].to_numpy()
    if np.std(r) == 0:
        raise DataError("The series is constant; volatility cannot be modelled.")
    spike = np.where((np.abs(r[:-1]) > SPIKE_FLAG) & (r[:-1] * r[1:] < 0) &
                     (np.abs(r[:-1] + r[1:]) < 0.5 * np.abs(r[:-1])))[0]
    rep.flagged_spikes = [df.loc[i, "Date"] for i in spike]
    huge = np.where(np.abs(r) > big)[0]
    if len(huge):
        dts = ", ".join(f"{df.loc[i, 'Date']:%Y-%m-%d}" for i in huge[:3])
        rep.warnings.append(f"{len(huge)} move(s) above {big:.0f} % that do not reverse (for example {dts}). "
                            "They were kept. They may be stock splits, rebasing or data errors; check the file.")
    if len(df) < 750:
        rep.warnings.append(f"Only {len(df)} observations. Short series give unreliable GARCH and EGARCH estimates.")
    if not rep.dates_synthetic:
        gaps = df["Date"].diff().dt.days.dropna()
        limit = 10 if rep.periods_per_year >= 252 else 400 / max(rep.periods_per_year, 1)
        if (gaps > limit).any():
            rep.warnings.append(f"The series has {(gaps > limit).sum()} gap(s) much longer than usual.")
    rep.rows_used = len(df)
    return df, rep


def simulate_prices(n: int = 1500, seed: int = 1, freq: str = "B", start: str = "2018-01-01",
                    start_level: float = 100.0) -> pd.DataFrame:
    """Simulated price series with GARCH-style volatility clustering, for demonstrations and tests."""
    rng = np.random.default_rng(seed)
    h, out = 1.0, []
    for _ in range(n):
        e = math.sqrt(h) * rng.standard_normal()
        out.append(e)
        h = 0.05 + 0.1 * e * e + 0.85 * h
    price = start_level * np.exp(np.cumsum(np.array(out) / 100 * (1.0 if freq == "B" else 2.0)))
    return pd.DataFrame({"Date": pd.date_range(start, periods=n, freq=freq).strftime("%Y-%m-%d"),
                         "Close": np.round(price, 2)})
