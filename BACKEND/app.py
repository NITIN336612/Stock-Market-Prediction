from pathlib import Path
import csv
from flask import Flask, jsonify, render_template, request
import sqlite3

from database import (
    DATABASE_PATH,
    get_connection,
    initialize_database,
    save_forecasts,
    reconcile_prediction_history,
)
from prediction_service import predict_next_day

app = Flask(__name__)


@app.route("/")
def dashboard():
    with get_connection() as connection:
        latest_stock = connection.execute(
            "SELECT * FROM stock_data ORDER BY date DESC LIMIT 1"
        ).fetchone()
        stock_count = connection.execute(
            "SELECT COUNT(*) AS count FROM stock_data"
        ).fetchone()["count"]
        models = connection.execute(
            "SELECT * FROM model_evaluation ORDER BY mape ASC"
        ).fetchall()
        best_model = connection.execute(
            "SELECT * FROM model_evaluation ORDER BY mape ASC LIMIT 1"
        ).fetchone()

    return render_template(
        "dashboard.html",
        latest_stock=dict(latest_stock) if latest_stock else None,
        stock_count=stock_count,
        models=[dict(model) for model in models],
        best_model=dict(best_model) if best_model else None,
    )


@app.route("/api/stocks")
def get_stocks():
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT * FROM stock_data ORDER BY date"
        ).fetchall()
    return jsonify([dict(row) for row in rows])


@app.route("/api/stocks/latest")
def get_latest_stock():
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM stock_data ORDER BY date DESC LIMIT 1"
        ).fetchone()
    if row is None:
        return jsonify({"error": "No stock data found"}), 404
    return jsonify(dict(row))


@app.route("/api/models")
def get_models():
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT * FROM model_evaluation ORDER BY mape ASC"
        ).fetchall()
    return jsonify([dict(row) for row in rows])


@app.route("/api/models/best")
def get_best_model():
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM model_evaluation ORDER BY mape ASC LIMIT 1"
        ).fetchone()
    if row is None:
        return jsonify({"error": "No model evaluation data found"}), 404
    return jsonify(dict(row))


@app.route("/api/status")
def database_status():
    with get_connection() as connection:
        stock_count = connection.execute(
            "SELECT COUNT(*) AS count FROM stock_data"
        ).fetchone()["count"]
        model_count = connection.execute(
            "SELECT COUNT(*) AS count FROM model_evaluation"
        ).fetchone()["count"]
        forecast_count = connection.execute(
            "SELECT COUNT(*) AS count FROM prediction_history"
        ).fetchone()["count"]
    return jsonify({
        "database": "connected",
        "stock_records": stock_count,
        "models": model_count,
        "saved_forecasts": forecast_count,
    })


@app.route("/api/predictions/next-day")
def next_day_prediction():
    try:
        result = predict_next_day()
        # UNIQUE(forecast_date, model_name) prevents refreshes creating duplicates.
        result["new_forecasts_saved"] = save_forecasts(result)
        return jsonify(result)
    except Exception as error:
        app.logger.exception("Next-day prediction failed")
        return jsonify({"error": str(error)}), 500


@app.route("/api/predictions/history")
def prediction_history():
    limit = min(max(int(request.args.get("limit", 100)), 1), 500)
    with get_connection() as connection:
        rows = connection.execute("""
            SELECT forecast_date, target_date, model_name, base_close,
                   predicted_close, predicted_return, signal, actual_close,
                   actual_return, absolute_error, percentage_error,
                   direction_correct, created_at
            FROM prediction_history
            ORDER BY forecast_date DESC, model_name
            LIMIT ?
        """, (limit,)).fetchall()
    return jsonify([dict(row) for row in rows])


@app.route("/api/predictions/performance")
def prediction_performance():
    with get_connection() as connection:
        rows = connection.execute("""
            SELECT model_name,
                   COUNT(*) AS total_forecasts,
                   SUM(CASE WHEN actual_close IS NOT NULL THEN 1 ELSE 0 END) AS evaluated_forecasts,
                   AVG(CASE WHEN actual_close IS NOT NULL THEN absolute_error END) AS mae,
                   AVG(CASE WHEN actual_close IS NOT NULL THEN percentage_error END) AS mape,
                   AVG(CASE WHEN actual_close IS NOT NULL THEN direction_correct * 1.0 END) * 100 AS directional_accuracy
            FROM prediction_history
            GROUP BY model_name
            ORDER BY model_name
        """).fetchall()
    return jsonify([dict(row) for row in rows])


# Historical test-set files are kept separate from live prediction history.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_DIR = PROJECT_ROOT / "EVALUATION"
TRAINING_PREDICTION_FILES = {
    "lstm": "lstm_predictions.csv",
    "random_forest": "random_forest_predictions.csv",
    "xgboost": "xgboost_predictions.csv",
    "naive_baseline": "naive_baseline_predictions.csv",
}


def read_csv_records(filename):
    file_path = EVALUATION_DIR / filename
    if not file_path.exists():
        return None
    with file_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        return list(csv.DictReader(csv_file))


@app.route("/api/training-results/<model_name>")
def training_results(model_name):
    filename = TRAINING_PREDICTION_FILES.get(model_name)
    if filename is None:
        return jsonify({"error": "Unknown training model"}), 404

    rows = read_csv_records(filename)
    if rows is None:
        return jsonify({"error": f"Missing evaluation file: {filename}"}), 404

    numeric_fields = [
        "Today_Close", "Actual_Next_Close", "Predicted_Next_Close",
        "Actual_Return", "Predicted_Return"
    ]
    for row in rows:
        for field in numeric_fields:
            row[field] = float(row[field]) if row.get(field) not in (None, "") else None
    return jsonify(rows)


@app.route("/api/training-comparison")
def training_comparison():
    rows = read_csv_records("model_comparison.csv")
    if rows is None:
        return jsonify({"error": "Missing model_comparison.csv"}), 404

    numeric_fields = [
        "Return_MAE", "Return_RMSE", "Return_R2",
        "Price_MAE", "Price_RMSE", "MAPE_Percent", "Price_R2"
    ]
    for row in rows:
        for field in numeric_fields:
            row[field] = float(row[field]) if row.get(field) not in (None, "") else None
    return jsonify(rows)


if __name__ == "__main__":
    initialize_database()
    reconcile_prediction_history()
    print("==========================================")
    print("INTELLIGENT STOCK MARKET ANALYTICS")
    print("==========================================")
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
