import sqlite3
from contextlib import contextmanager
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BACKEND_DIR / "stock_market.db"


@contextmanager
def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
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
        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                model_name TEXT NOT NULL,
                today_close REAL,
                predicted_next_close REAL,
                predicted_return REAL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                forecast_date TEXT NOT NULL,
                target_date TEXT,
                model_name TEXT NOT NULL,
                base_close REAL NOT NULL,
                predicted_close REAL NOT NULL,
                predicted_return REAL NOT NULL,
                signal TEXT NOT NULL,
                actual_close REAL,
                actual_return REAL,
                absolute_error REAL,
                percentage_error REAL,
                direction_correct INTEGER,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(forecast_date, model_name)
            )
        """)
        connection.execute("""
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
    print("Database initialized successfully.")
    print("Database location:", DATABASE_PATH)


def save_forecasts(forecast):
    """Save each model's forecast once per forecast date; never overwrite history."""
    forecast_date = str(forecast["latest_date"])
    base_close = float(forecast["latest_close"])
    saved = 0
    with get_connection() as connection:
        for item in forecast["predictions"]:
            cursor = connection.execute("""
                INSERT OR IGNORE INTO prediction_history
                (forecast_date, model_name, base_close, predicted_close,
                 predicted_return, signal)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                forecast_date,
                item["model"],
                base_close,
                float(item["predicted_price"]),
                float(item["predicted_return"]),
                item["signal"]
            ))
            saved += cursor.rowcount
    return saved


def reconcile_prediction_history():
    """
    Attach each unresolved forecast to the first available stock session
    after its forecast date, then calculate out-of-sample errors.
    """
    updated = 0
    with get_connection() as connection:
        pending = connection.execute("""
            SELECT id, forecast_date, base_close, predicted_close, predicted_return
            FROM prediction_history
            WHERE actual_close IS NULL
        """).fetchall()

        for row in pending:
            actual = connection.execute("""
                SELECT date, close
                FROM stock_data
                WHERE date > ?
                ORDER BY date ASC
                LIMIT 1
            """, (row["forecast_date"],)).fetchone()

            if actual is None or actual["close"] is None or not row["base_close"]:
                continue

            actual_close = float(actual["close"])
            actual_return = actual_close / float(row["base_close"]) - 1
            absolute_error = abs(float(row["predicted_close"]) - actual_close)
            percentage_error = (absolute_error / actual_close * 100) if actual_close else None
            predicted_direction = 1 if float(row["predicted_return"]) > 0 else (
                -1 if float(row["predicted_return"]) < 0 else 0
            )
            actual_direction = 1 if actual_return > 0 else (-1 if actual_return < 0 else 0)
            direction_correct = int(predicted_direction == actual_direction)

            connection.execute("""
                UPDATE prediction_history
                SET target_date = ?,
                    actual_close = ?,
                    actual_return = ?,
                    absolute_error = ?,
                    percentage_error = ?,
                    direction_correct = ?
                WHERE id = ? AND actual_close IS NULL
            """, (
                actual["date"], actual_close, actual_return,
                absolute_error, percentage_error, direction_correct, row["id"]
            ))
            updated += 1
    return updated
