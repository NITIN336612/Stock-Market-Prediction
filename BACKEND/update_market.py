"""
Incrementally update NIFTY 50 daily OHLCV data from Yahoo Finance.

Run from the BACKEND directory:
    python update_market_data.py

This updates:
  1. DATA/PROCESSED/NIFTY50_processed.csv
  2. BACKEND/stock_market.db (stock_data table)

It does not rebuild technical indicators or retrain ML models.
"""

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import sqlite3

import pandas as pd
import yfinance as yf

from database import DATABASE_PATH, initialize_database, reconcile_prediction_history


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STOCK_DATA_FILE = PROJECT_ROOT / "DATA" / "PROCESSED" / "NIFTY50_processed.csv"
TIMEZONE = ZoneInfo("Asia/Kolkata")
TICKER = "^NSEI"
REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def latest_database_date():
    with sqlite3.connect(DATABASE_PATH) as connection:
        row = connection.execute(
            "SELECT MAX(date) FROM stock_data"
        ).fetchone()
    return row[0] if row and row[0] else None


def download_new_sessions(start_date: str, end_date: str) -> pd.DataFrame:
    print(f"Downloading {TICKER} data from {start_date} to before {end_date}...")
    downloaded = yf.download(
        TICKER,
        start=start_date,
        end=end_date,            # Yahoo Finance end date is exclusive
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if downloaded.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    # yfinance may return MultiIndex columns even for a single ticker.
    if isinstance(downloaded.columns, pd.MultiIndex):
        downloaded.columns = downloaded.columns.get_level_values(0)

    downloaded = downloaded.reset_index()
    date_column = "Date" if "Date" in downloaded.columns else "Datetime"
    if date_column not in downloaded.columns:
        raise ValueError(f"Yahoo Finance response has no date column: {downloaded.columns.tolist()}")

    missing = set(REQUIRED_COLUMNS[1:]) - set(downloaded.columns)
    if missing:
        raise ValueError(f"Yahoo Finance response is missing columns: {sorted(missing)}")

    downloaded = downloaded[[
        date_column, "Open", "High", "Low", "Close", "Volume"
    ]].copy()
    downloaded = downloaded.rename(columns={date_column: "Date"})

    dates = pd.to_datetime(downloaded["Date"], errors="coerce")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    downloaded["Date"] = dates.dt.strftime("%Y-%m-%d")

    for column in ["Open", "High", "Low", "Close", "Volume"]:
        downloaded[column] = pd.to_numeric(downloaded[column], errors="coerce")

    downloaded = downloaded.dropna(subset=["Date", "Open", "High", "Low", "Close"])
    downloaded = downloaded.drop_duplicates(subset=["Date"], keep="last")
    downloaded = downloaded.sort_values("Date").reset_index(drop=True)
    return downloaded


def update_processed_csv(new_rows: pd.DataFrame) -> None:
    STOCK_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    if STOCK_DATA_FILE.exists():
        existing = pd.read_csv(STOCK_DATA_FILE)
        missing = set(REQUIRED_COLUMNS) - set(existing.columns)
        if missing:
            raise ValueError(f"Processed CSV is missing columns: {sorted(missing)}")
        existing = existing[REQUIRED_COLUMNS].copy()
    else:
        existing = pd.DataFrame(columns=REQUIRED_COLUMNS)

    combined = pd.concat([existing, new_rows[REQUIRED_COLUMNS]], ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    combined = combined.dropna(subset=["Date", "Close"])
    combined = combined.drop_duplicates(subset=["Date"], keep="last")
    combined = combined.sort_values("Date").reset_index(drop=True)

    # Write to a temporary file first, then replace the CSV atomically.
    temp_file = STOCK_DATA_FILE.with_suffix(".tmp")
    combined.to_csv(temp_file, index=False)
    temp_file.replace(STOCK_DATA_FILE)
    print(f"Processed CSV updated: {STOCK_DATA_FILE}")
    print(f"Total unique sessions in processed CSV: {len(combined)}")


def insert_new_database_rows(new_rows: pd.DataFrame) -> int:
    inserted = 0
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_stock_data_date
            ON stock_data(date)
        """)

        for row in new_rows.itertuples(index=False):
            cursor = connection.execute("""
                INSERT INTO stock_data (date, open, high, low, close, volume)
                SELECT ?, ?, ?, ?, ?, ?
                WHERE NOT EXISTS (
                    SELECT 1 FROM stock_data WHERE date = ?
                )
            """, (
                row.Date, row.Open, row.High, row.Low,
                row.Close, row.Volume, row.Date
            ))
            inserted += cursor.rowcount
    return inserted


def main():
    print("=" * 55)
    print("NIFTY 50 AUTOMATIC MARKET DATA UPDATER")
    print("=" * 55)

    initialize_database()

    latest_date = latest_database_date()
    if latest_date:
        start_date = (datetime.strptime(latest_date, "%Y-%m-%d") + timedelta(days=1)).date()
    else:
        start_date = datetime(2015, 1, 1).date()

    now = datetime.now(TIMEZONE)
    today = now.date()

    # Avoid importing today's still-forming daily candle if run before 4 PM IST.
    end_date = today + timedelta(days=1) if now.hour >= 16 else today

    if start_date >= end_date:
        print(f"Database is already current through {latest_date or 'no available session'}.")
        return

    new_rows = download_new_sessions(start_date.isoformat(), end_date.isoformat())

    if new_rows.empty:
        print("Yahoo Finance returned no new completed trading sessions.")
        print("This can happen on weekends, market holidays, or when data is not yet published.")
        return

    # Persist the CSV first. If the DB insert fails, rerunning safely deduplicates the CSV.
    update_processed_csv(new_rows)

    inserted = insert_new_database_rows(new_rows)
    print("New rows inserted into SQLite:", inserted)

    reconciled = reconcile_prediction_history()
    print("Historical forecasts reconciled:", reconciled)

    latest_after_update = latest_database_date()
    print("Latest market session now in database:", latest_after_update)
    print("Update completed.")


if __name__ == "__main__":
    main()
