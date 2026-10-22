"""
TradeWiseAI - Step 1/2: data pipeline
Downloads prices, computes indicators, builds labels, creates walk-forward splits.
pip install yfinance pandas numpy ta
"""
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands

# ---------------- CONFIG (edit these) ----------------
TICKERS = [  # start small: 10 liquid NIFTY 50 names
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "SBIN.NS", "ITC.NS", "LT.NS", "BHARTIARTL.NS", "AXISBANK.NS",
]
START, END = "2018-01-01", "2025-12-31"
HORIZON = 1          # predict direction over next N trading days
THRESHOLD = 0.002    # ignore moves smaller than +/-0.2% (noise)
# -----------------------------------------------------


def download(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"]
    df["ret_1d"] = c.pct_change()
    df["rsi_14"] = RSIIndicator(c, window=14).rsi()
    macd = MACD(c)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()
    bb = BollingerBands(c, window=20, window_dev=2)
    df["bb_pct"] = bb.bollinger_pband()      # position inside the bands
    df["bb_width"] = bb.bollinger_wband()
    df["sma_10"] = SMAIndicator(c, 10).sma_indicator() / c - 1
    df["sma_50"] = SMAIndicator(c, 50).sma_indicator() / c - 1
    df["ema_20"] = EMAIndicator(c, 20).ema_indicator() / c - 1
    df["vol_chg"] = df["Volume"].pct_change()
    return df


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    # Features at close of day t -> label is the return from t to t+HORIZON.
    fwd = df["Close"].shift(-HORIZON) / df["Close"] - 1
    df["fwd_ret"] = fwd
    df["label"] = np.where(fwd > THRESHOLD, 1, np.where(fwd < -THRESHOLD, 0, np.nan))
    return df


def build_dataset() -> pd.DataFrame:
    frames = []
    for t in TICKERS:
        df = add_labels(add_indicators(download(t)))
        df["ticker"] = t
        frames.append(df)
    data = pd.concat(frames)
    data = data.dropna(subset=["rsi_14", "sma_50", "fwd_ret"])  # warm-up + last rows
    return data.reset_index().rename(columns={"index": "Date", "Date": "Date"})


def walk_forward_splits(dates: pd.Series, train_years=3, val_months=6, test_months=6):
    """Yield (train_idx, val_idx, test_idx) as boolean masks. Chronological, no shuffling.
    Window slides forward by test_months each fold."""
    dates = pd.to_datetime(dates)
    start = dates.min()
    end = dates.max()
    cur = start + pd.DateOffset(years=train_years)
    while cur + pd.DateOffset(months=val_months + test_months) <= end:
        train = (dates >= start) & (dates < cur)  # expanding window; use a rolling start if preferred
        val = (dates >= cur) & (dates < cur + pd.DateOffset(months=val_months))
        test_start = cur + pd.DateOffset(months=val_months)
        test = (dates >= test_start) & (dates < test_start + pd.DateOffset(months=test_months))
        yield train.values, val.values, test.values
        cur += pd.DateOffset(months=test_months)


if __name__ == "__main__":
    data = build_dataset()
    data.to_csv("dataset_prices_indicators.csv", index=False)
    print(data.shape, data["ticker"].nunique(), "tickers")
    print("Label balance (labelled rows):")
    print(data["label"].value_counts(normalize=True, dropna=True))
    n = sum(1 for _ in walk_forward_splits(data["Date"]))
    print(f"{n} walk-forward folds")