
import yfinance as yf
import pandas as pd

from pathlib import Path
from datetime import datetime, timedelta

# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DIR = PROJECT_ROOT / "DATA" / "RAW"

OUTPUT_FILE = RAW_DIR / "NIFTY50_yfinance_full.csv"

TICKER = "^NSEI"

START_DATE = "2015-01-01"

# Yahoo Finance end date is exclusive
END_DATE = (
    datetime.now().date() + timedelta(days=1)
).strftime("%Y-%m-%d")


# ==========================================
# DOWNLOAD DATA
# ==========================================

def download_stock_data():

    print("=" * 55)
    print("NIFTY 50 HISTORICAL DATA UPDATER")
    print("=" * 55)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("\nDownloading NIFTY 50 data...")
    print("Ticker:", TICKER)
    print("Start:", START_DATE)
    print("End:", END_DATE)

    df = yf.download(
        TICKER,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=False,
        progress=False,
        multi_level_index=False
    )

    if df.empty:
        raise RuntimeError(
            "No data received. Check internet connection or Yahoo Finance."
        )

    # Handle possible multi-level columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Prepare date column
    df.index = pd.to_datetime(df.index)

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df = df.reset_index()

    # Standardize column names
    df.columns = [
        str(col).strip() for col in df.columns
    ]

    if "Datetime" in df.columns:
        df.rename(columns={"Datetime": "Date"}, inplace=True)

    required = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    # Keep available market columns
    columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume"
    ]

    df = df[
        [col for col in columns if col in df.columns]
    ].copy()

    # Clean data
    df["Date"] = pd.to_datetime(df["Date"])

    df = df.drop_duplicates(subset=["Date"])

    df = df.sort_values("Date")

    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    # Save new dataset separately
    df.to_csv(OUTPUT_FILE, index=False)

    # ======================================
    # SUMMARY
    # ======================================

    print("\nDOWNLOAD SUCCESSFUL!")

    print("Total records:", len(df))

    print(
        "First date:",
        df["Date"].min().date()
    )

    print(
        "Latest date:",
        df["Date"].max().date()
    )

    print(
        "Latest closing price:",
        df["Close"].iloc[-1]
    )

    print("\nSaved at:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 55)


if __name__ == "__main__":
    download_stock_data()
