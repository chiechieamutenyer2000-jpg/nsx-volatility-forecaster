# NSX Volatility Forecaster (prototype)

Working prototype for the Honours project *Volatility Forecasting on a Frontier Market: GARCH, EGARCH and LSTM on the
Namibian Stock Exchange Index* (Festus T. Amutenya, UNAM).

## Run it
```
pip install -r requirements.txt        # Python 3.10+
streamlit run app.py                   # opens http://localhost:8501
python -m pytest -q                    # 33 automated tests
```
Optional LSTM: `pip install tensorflow` (the app works without it; the LSTM checkbox is disabled).

## What it does
1. **Data (any series)**: the bundled FTSE/NSX file, a simulated series, or your own CSV/Excel file. Delimiter, encoding,
   number format, date order, column names, frequency (daily, weekly, monthly) and prices-or-returns are detected; you can
   correct the column choices. Isolated corrupted prices are fixed and reported (for the NSX file: 2021-02-15 and
   2024-12-24); unusable files get a plain-language message.
2. **Forecast next period**: GARCH(1,1), EGARCH(1,1,1), EWMA volatility with warnings (short window, no ARCH effect,
   negative alpha) taken from the study's EGARCH-instability finding. CSV download.
3. **Backtest**: rolling refits, QLIKE and MSE, Diebold-Mariano tests, forecast chart, optional LSTM.
4. **Check against the report**: Design A (946 test days, 500-day window, refit every 250) gives EWMA QLIKE 1.422,
   GARCH 1.457, EGARCH 2.031: the same numbers as in the report (checked by an automated test).

## Layout
`app.py` interface; `nsxlib/data.py` validation; `nsxlib/models.py` models and diagnostics; `nsxlib/backtest.py`
losses, DM test, backtest; `nsxlib/lstm.py` optional LSTM; `tests/test_all.py`.

## Limits
One series at a time, at least 251 observations; squared return is a noisy proxy; not investment advice. The LSTM here is a single fast run
(20 epochs, one seed, trained once), not the 3-seed retrained version in the report. Check Investing.com's terms
before redistributing the data file.

## Other ways to run it
* `NSX_Volatility_Forecaster.ipynb`: self-contained Google Colab notebook (Runtime > Run all).
* `DEPLOY_PERMANENT_LINK.md`: free permanent web address (Streamlit Community Cloud) that works when Colab is closed.
