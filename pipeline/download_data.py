"""Step 1: download daily adjusted close prices for all factors and assets.

Source: Yahoo Finance via the open-source `yfinance` library.
Output: data/prices.csv (one column per ticker, adjusted close).
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

from universe import ASSETS, DATA_END, DATA_START, FACTORS

OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

tickers = sorted(set(FACTORS) | set(ASSETS))
print(f"Downloading {len(tickers)} tickers {DATA_START} -> {DATA_END}")
raw = yf.download(tickers, start=DATA_START, end=DATA_END, auto_adjust=True, progress=False, threads=False)
prices = raw["Close"].sort_index()

missing = [t for t in tickers if t not in prices or prices[t].dropna().empty]
if missing:
    raise SystemExit(f"Failed to download: {missing}")

prices.to_csv(OUT / "prices.csv")
print(prices.notna().sum().sort_values().to_string())
print("Saved", OUT / "prices.csv", prices.shape)
