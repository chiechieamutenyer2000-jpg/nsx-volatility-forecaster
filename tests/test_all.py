import io
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from nsxlib.data import load_prices, clean_prices, DataError
from nsxlib import models as M
from nsxlib.backtest import backtest, dm_test, qlike

CSV = Path(__file__).parents[1] / "data" / "FTSE_NSX_INDEX_1.csv"


@pytest.fixture(scope="module")
def df():
    return load_prices(str(CSV))[0]


def synth(n=800, seed=0):
    rng = np.random.default_rng(seed)
    h, out = 1.0, []
    for _ in range(n):
        e = np.sqrt(h) * rng.standard_normal()
        out.append(e)
        h = 0.05 + 0.1 * e * e + 0.85 * h
    return np.array(out)


def csv_from(prices, dates=None):
    d = dates if dates is not None else pd.bdate_range("2015-01-01", periods=len(prices)).strftime("%m/%d/%Y")
    return io.StringIO(pd.DataFrame({"Date": d, "Price": prices}).to_csv(index=False))


# ---- data
def test_real_data_loads(df):
    assert len(df) == 3151 and df["Return"].isna().sum() == 0

def test_known_bad_prices_corrected():
    _, rep = load_prices(str(CSV))
    assert len(rep.corrected_prices) == 2

def test_no_numeric_column():
    with pytest.raises(DataError, match="No numeric column"):
        load_prices(io.StringIO("a,b\nx,y\nz,w\n"))

def test_empty_file():
    with pytest.raises(DataError):
        load_prices(io.StringIO(""))

def test_too_few_rows():
    with pytest.raises(DataError, match="at least"):
        load_prices(csv_from(np.linspace(100, 110, 50)))

def test_constant_series():
    with pytest.raises(DataError, match="constant"):
        load_prices(csv_from(np.full(400, 100.0)))

def test_garbage_and_duplicates_cleaned():
    p = 100 * np.exp(np.cumsum(0.01 * np.random.default_rng(1).standard_normal(400)))
    dates = list(pd.bdate_range("2015-01-01", periods=400).strftime("%m/%d/%Y"))
    d = pd.DataFrame({"Date": dates, "Price": p.astype(object)})
    d.loc[5, "Price"] = "abc"; d.loc[6, "Date"] = "not a date"; d.loc[8, "Date"] = d.loc[7, "Date"]
    _, rep = clean_prices(d)
    assert rep.dropped_bad_price == 1 and rep.dropped_unreadable_date == 1 and rep.dropped_duplicates == 1

def test_unreversed_jump_is_warning_not_error():
    p = 100 * np.exp(np.cumsum(0.01 * np.random.default_rng(2).standard_normal(400)))
    p[200:] *= 3                     # permanent jump, e.g. a rebasing: kept, but warned about
    out, rep = load_prices(csv_from(p))
    assert len(out) == 399 and any("do not reverse" in w for w in rep.warnings)

def test_unsorted_input_sorted():
    p = 100 * np.exp(np.cumsum(0.01 * np.random.default_rng(3).standard_normal(400)))
    d = pd.DataFrame({"Date": pd.bdate_range("2015-01-01", periods=400), "Price": p}).iloc[::-1]
    out, _ = clean_prices(d.assign(Date=d["Date"].dt.strftime("%m/%d/%Y")))
    assert out["Date"].is_monotonic_increasing


# ---- models
def test_garch_recovers_parameters():
    f = M.fit(synth(3000), "GARCH")
    assert 0.04 < f.params["alpha"] < 0.2 and 0.7 < f.params["beta"] < 0.95 and f.next_var > 0

def test_egarch_runs_and_positive(df):
    f = M.fit(df["Return"].to_numpy()[-1000:], "EGARCH")
    assert f.next_var > 0 and np.isfinite(f.next_var)

def test_too_short_series_rejected():
    with pytest.raises(ValueError, match="At least"):
        M.fit(synth(100), "GARCH")

def test_nan_rejected():
    y = synth(400); y[10] = np.nan
    with pytest.raises(ValueError, match="missing"):
        M.fit(y, "GARCH")

def test_bad_kind():
    with pytest.raises(ValueError):
        M.fit(synth(400), "ARCH")

def test_short_window_warns(df):
    f = M.fit(df["Return"].to_numpy()[-500:], "EGARCH")
    assert any("500" in w for w in f.warnings)

def test_diagnose_flags_negative_alpha():
    w = M.diagnose("EGARCH", {"alpha": -0.1, "beta": 0.99}, 0.9, 500)
    assert any("negative" in x for x in w)

def test_ewma_known_value():
    y = np.array([1.0] * 60 + [2.0])
    assert M.ewma_next(y) > M.ewma_next(y[:-1])

def test_next_variance_matches_arch(df):
    y = df["Return"].to_numpy()[-1000:]
    f = M.fit(y, "GARCH")
    alt = f.params["omega"] + f.params["alpha"] * (y[-1] - f.params["mu"]) ** 2 + f.params["beta"] * f.last_var
    assert f.next_var == pytest.approx(alt)


# ---- backtest
def test_dm_symmetry_and_sign():
    rng = np.random.default_rng(0)
    a = rng.random(300) + 0.3; b = rng.random(300)
    s, p = dm_test(a, b)
    assert s > 0 and p < 0.05                # b has lower loss -> positive
    s2, _ = dm_test(b, a)
    assert s2 == pytest.approx(-s)

def test_dm_degenerate_returns_nan():
    s, p = dm_test(np.ones(100), np.ones(100))
    assert np.isnan(s)

def test_backtest_shapes(df):
    r = backtest(df, 200, 1000, 100)
    assert len(r.forecasts) == 200 and (r.forecasts.drop(columns=["Date", "r2"]) > 0).all().all()
    assert set(r.table["Model"]) >= {"GARCH", "EGARCH", "EWMA"}

def test_backtest_reproduces_report_numbers(df):
    """Design A, 500-day rolling window, refit every 250: values reported in the study."""
    r = backtest(df, 946, 500, 250).table.set_index("Model")
    assert r.loc["EWMA", "QLIKE"] == pytest.approx(1.422, abs=0.002)
    assert r.loc["GARCH", "QLIKE"] == pytest.approx(1.457, abs=0.002)
    assert r.loc["EGARCH", "QLIKE"] == pytest.approx(2.031, abs=0.005)   # the instability finding

def test_backtest_input_errors(df):
    with pytest.raises(ValueError):
        backtest(df, 10, 1000, 100)
    with pytest.raises(ValueError):
        backtest(df, 3000, 1000, 100)
    with pytest.raises(ValueError):
        backtest(df, 200, 100, 100)
    with pytest.raises(ValueError):
        backtest(df, 200, 1000, 0)

def test_qlike_is_lowest_at_truth():
    r2 = np.full(1000, 2.0)
    assert qlike(r2, np.full(1000, 2.0)).mean() < qlike(r2, np.full(1000, 1.0)).mean()


# ---- any data: formats, frequencies, kinds
def _prices(n=600, seed=5):
    return 100 * np.exp(np.cumsum(0.01 * np.random.default_rng(seed).standard_normal(n)))

def test_semicolon_european_decimals():
    p = _prices()
    d = pd.bdate_range("2016-01-01", periods=len(p)).strftime("%d.%m.%Y")
    def eu(x):
        return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    txt = "Datum;Schluss\n" + "\n".join(f"{a};{eu(b)}" for a, b in zip(d, p))
    out, rep = load_prices(io.StringIO(txt))
    assert len(out) == len(p) - 1 and rep.value_col == "Schluss" and rep.date_col == "Datum"
    assert abs(out["Price"].iloc[0] - round(p[1], 2)) < 0.01

def test_yahoo_style_picks_adj_close():
    p = _prices()
    d = pd.DataFrame({"Date": pd.bdate_range("2016-01-01", periods=len(p)).strftime("%Y-%m-%d"),
                      "Open": p * 0.99, "High": p * 1.01, "Low": p * 0.98, "Close": p * 1.0,
                      "Adj Close": p * 1.0, "Volume": np.arange(len(p)) + 1000})
    _, rep = load_prices(io.StringIO(d.to_csv(index=False)))
    assert rep.value_col == "Adj Close" and rep.kind == "prices"

def test_thousand_separators_and_percent_text():
    from nsxlib.data import to_number
    s = to_number(pd.Series(["1,234.5", "1.234,5", "12%", "(3.1)", "abc", "", "7,5"]))
    assert s.iloc[0] == 1234.5 and s.iloc[1] == 1234.5 and s.iloc[2] == 12 and s.iloc[3] == -3.1
    assert np.isnan(s.iloc[4]) and np.isnan(s.iloc[5]) and s.iloc[6] == 7.5

def test_weekly_and_monthly_frequency_detected():
    for freq, label, ppy in (("W", "weekly", 52), ("ME", "monthly", 12)):
        p = _prices(400)
        d = pd.DataFrame({"Date": pd.date_range("2000-01-01", periods=400, freq=freq), "Price": p})
        _, rep = clean_prices(d)
        assert rep.freq_label == label and rep.periods_per_year == ppy

def test_returns_only_input_percent_and_decimal():
    r = 0.01 * np.random.default_rng(7).standard_normal(500)
    dates = pd.bdate_range("2016-01-01", periods=500)
    out_dec, rep = clean_prices(pd.DataFrame({"Date": dates, "Return": r}))
    assert rep.kind == "returns" and abs(out_dec["Return"].std() - 1.0) < 0.1
    out_pct, _ = clean_prices(pd.DataFrame({"Date": dates, "Return": r * 100}))
    assert np.allclose(out_dec["Return"], out_pct["Return"])

def test_no_date_column_is_allowed_with_warning():
    out, rep = clean_prices(pd.DataFrame({"Level": _prices()}))
    assert rep.dates_synthetic and any("No date column" in w for w in rep.warnings) and len(out) == 599

def test_excel_file_read(tmp_path):
    p = _prices()
    f = tmp_path / "x.xlsx"
    pd.DataFrame({"Date": pd.bdate_range("2016-01-01", periods=len(p)), "Close": p}).to_excel(f, index=False)
    out, rep = load_prices(str(f))
    assert len(out) == len(p) - 1 and rep.value_col == "Close"

def test_dayfirst_month_first_resolved():
    from nsxlib.data import parse_dates
    dts = pd.bdate_range("2016-01-01", periods=300)
    assert parse_dates(pd.Series(dts.strftime("%d/%m/%Y"))).equals(pd.Series(dts))
    assert parse_dates(pd.Series(dts.strftime("%m/%d/%Y"))).equals(pd.Series(dts))

def test_simulated_data_runs_end_to_end():
    from nsxlib.data import simulate_prices
    out, _ = clean_prices(simulate_prices(900))
    r = backtest(out, 200, 500, 100)
    assert len(r.table) >= 3


def test_dates_are_never_read_as_numbers():
    from nsxlib.data import to_number
    s = to_number(pd.Series(["17/01/2014", "2014-01-17", "17.01.2014", "5 Jan 2014", "2014-01-17 00:00:00", "12:30"]))
    assert s.isna().all()

def test_weekly_file_with_unusual_column_names():
    d = pd.DataFrame({"Week ending": pd.date_range("2010-01-01", periods=400, freq="W-FRI").strftime("%d/%m/%Y"),
                      "Index level": _prices(400)})
    out, rep = clean_prices(d)
    assert rep.date_col == "Week ending" and rep.value_col == "Index level" and rep.freq_label == "weekly"
