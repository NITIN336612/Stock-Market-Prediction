from flask import Flask, jsonify, render_template
import sqlite3
from prediction_service import predict_next_day

from database import initialize_database


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

DATABASE_PATH = "BACKEND/stock_market.db"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db_connection():

    connection = sqlite3.connect(DATABASE_PATH)

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/")
def dashboard():

    connection = get_db_connection()

    # Latest stock record
    latest_stock = connection.execute("""
        SELECT *
        FROM stock_data
        ORDER BY date DESC
        LIMIT 1
    """).fetchone()

    # Total records
    stock_count = connection.execute("""
        SELECT COUNT(*) AS count
        FROM stock_data
    """).fetchone()["count"]

    # Model evaluation
    models = connection.execute("""
        SELECT *
        FROM model_evaluation
        ORDER BY mape ASC
    """).fetchall()

    # Best model
    best_model = connection.execute("""
        SELECT *
        FROM model_evaluation
        ORDER BY mape ASC
        LIMIT 1
    """).fetchone()

    connection.close()

    return render_template(
        "dashboard.html",
        latest_stock=dict(latest_stock) if latest_stock else None,
        stock_count=stock_count,
        models=[dict(model) for model in models],
        best_model=dict(best_model) if best_model else None
    )


# ============================================================
# API — ALL STOCK DATA
# ============================================================

@app.route("/api/stocks")
def get_stocks():

    connection = get_db_connection()

    rows = connection.execute("""
        SELECT *
        FROM stock_data
        ORDER BY date
    """).fetchall()

    connection.close()

    return jsonify([dict(row) for row in rows])


# ============================================================
# API — LATEST STOCK
# ============================================================

@app.route("/api/stocks/latest")
def get_latest_stock():

    connection = get_db_connection()

    row = connection.execute("""
        SELECT *
        FROM stock_data
        ORDER BY date DESC
        LIMIT 1
    """).fetchone()

    connection.close()

    if row is None:
        return jsonify({
            "error": "No stock data found"
        }), 404

    return jsonify(dict(row))


# ============================================================
# API — MODEL EVALUATION
# ============================================================

@app.route("/api/models")
def get_models():

    connection = get_db_connection()

    rows = connection.execute("""
        SELECT *
        FROM model_evaluation
        ORDER BY mape ASC
    """).fetchall()

    connection.close()

    return jsonify([dict(row) for row in rows])


# ============================================================
# API — BEST MODEL
# ============================================================

@app.route("/api/models/best")
def get_best_model():

    connection = get_db_connection()

    row = connection.execute("""
        SELECT *
        FROM model_evaluation
        ORDER BY mape ASC
        LIMIT 1
    """).fetchone()

    connection.close()

    if row is None:
        return jsonify({
            "error": "No model evaluation data found"
        }), 404

    return jsonify(dict(row))


# ============================================================
# API — DATABASE STATUS
# ============================================================

@app.route("/api/status")
def database_status():

    connection = get_db_connection()

    stock_count = connection.execute("""
        SELECT COUNT(*) AS count
        FROM stock_data
    """).fetchone()["count"]

    model_count = connection.execute("""
        SELECT COUNT(*) AS count
        FROM model_evaluation
    """).fetchone()["count"]

    connection.close()

    return jsonify({
        "database": "connected",
        "stock_records": stock_count,
        "models": model_count
    })


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    initialize_database()

    print("==========================================")
    print("INTELLIGENT STOCK MARKET ANALYTICS")
    print("==========================================")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
    
# ============================================================
# API — NEXT TRADING DAY PREDICTION
# ============================================================

@app.route("/api/predictions/next-day")
def next_day_prediction():

    try:
        predictions = predict_next_day()
        return jsonify(predictions)

    except Exception as error:

        return jsonify({
            "error": str(error)
        }), 500