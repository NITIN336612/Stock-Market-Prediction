import yfinance as yf
import pandas as pd

ticker = "^NSEI"

df = yf.download(
    ticker,
    start="2015-01-01",
    end="2026-01-01",
    interval="1d",
    auto_adjust=False
)

print(df.head())
df.to_csv("DATA/RAW/NIFTY50_raw.csv")