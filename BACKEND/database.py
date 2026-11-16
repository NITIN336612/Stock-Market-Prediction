import sqlite3
from pathlib import Path

# Always use the database located beside database.py
BACKEND_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BACKEND_DIR / "stock_market.db"


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():

    connection = get_connection()
    cursor = connection.cursor()

    # Stock price data
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL
        )
    """)

    # Model predictions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            model_name TEXT NOT NULL,
            today_close REAL,
            predicted_next_close REAL,
            predicted_return REAL
        )
    """)

    # Model evaluation
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_evaluation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            price_mae REAL,
            price_rmse REAL,
            mape REAL,
            price_r2 REAL,
            return_r2 REAL
        )
    """)

    connection.commit()
    connection.close()

    print("Database initialized successfully.")
    print("Database location:", DATABASE_PATH)


if __name__ == "__main__":
    initialize_database()