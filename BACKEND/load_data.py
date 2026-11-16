import pandas as pd
import sqlite3
from pathlib import Path

from database import (
    DATABASE_PATH,
    initialize_database,
    reconcile_prediction_history,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STOCK_DATA_FILE = PROJECT_ROOT / "DATA" / "PROCESSED" / "NIFTY50_processed.csv"
EVALUATION_FILE = PROJECT_ROOT / "EVALUATION" / "model_comparison.csv"


def load_stock_data():
    print("\nLoading stock data...")
    if not STOCK_DATA_FILE.exists():
        raise FileNotFoundError(f"Stock dataset not found: {STOCK_DATA_FILE}")

    df = pd.read_csv(STOCK_DATA_FILE)
    required_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing stock columns: {missing}")

    df = df[required_columns].copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df = df.rename(columns={
        "Date": "date", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume"
    })

    with sqlite3.connect(DATABASE_PATH) as connection:
        df.to_sql("stock_data_new", connection, if_exists="replace", index=False)
        connection.execute("DROP TABLE IF EXISTS stock_data")
        connection.execute("ALTER TABLE stock_data_new RENAME TO stock_data")

    print("Stock data loaded successfully.")
    print("Rows inserted:", len(df))
    matched = reconcile_prediction_history()
    print("Historical forecasts reconciled:", matched)


def load_model_evaluation():
    print("\nLoading model evaluation results...")
    if not EVALUATION_FILE.exists():
        raise FileNotFoundError(f"Evaluation file not found: {EVALUATION_FILE}")

    df = pd.read_csv(EVALUATION_FILE)
    required_columns = [
        "Model", "Price_MAE", "Price_RMSE",
        "MAPE_Percent", "Price_R2", "Return_R2"
    ]
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing evaluation columns: {missing}")

    df = df[required_columns].copy()
    df.columns = [
        "model_name", "price_mae", "price_rmse",
        "mape", "price_r2", "return_r2"
    ]

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("DELETE FROM model_evaluation")
        df.to_sql("model_evaluation", connection, if_exists="append", index=False)

    print("Model evaluation loaded successfully.")
    print("Models inserted:", len(df))


if __name__ == "__main__":
    print("\n====================================")
    print("STOCK MARKET DATABASE LOADER")
    print("====================================")
    initialize_database()
    load_stock_data()
    load_model_evaluation()
    print("\nDATABASE LOADING COMPLETED!")
