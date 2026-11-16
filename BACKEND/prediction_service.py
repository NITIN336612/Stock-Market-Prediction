import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_FILE = (
    PROJECT_ROOT / "DATA" / "FEATURES" / "NIFTY50_features.csv"
)

LATEST_FILE = (
    PROJECT_ROOT / "DATA" / "FEATURES" / "NIFTY50_latest_features.csv"
)

MODEL_DIR = PROJECT_ROOT / "MODELS"


# ============================================================
# MODEL FILES
# ============================================================

RF_PATH = MODEL_DIR / "random_forest_model.pkl"
XGB_PATH = MODEL_DIR / "xgboost_model.pkl"

LSTM_PATH = MODEL_DIR / "lstm_model.keras"
SCALER_PATH = MODEL_DIR / "lstm_scaler.pkl"


# ============================================================
# FEATURE CONFIGURATION
# ============================================================

FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "Daily_Return", "Volatility_20",
    "MA_7", "MA_20", "MA_50",
    "EMA_12", "EMA_26", "MACD", "MACD_Signal",
    "RSI", "Lag_1", "Lag_5", "Lag_10",
    "Volume_MA_20"
]

LOOKBACK = 20


# ============================================================
# LOAD SAVED MODELS
# ============================================================

rf_model = joblib.load(RF_PATH)

xgb_model = joblib.load(XGB_PATH)

lstm_scaler = joblib.load(SCALER_PATH)

lstm_model = tf.keras.models.load_model(LSTM_PATH)


# ============================================================
# PREDICTION HELPERS
# ============================================================

def get_signal(predicted_return):

    if predicted_return > 0.0005:
        return "Bullish"

    elif predicted_return < -0.0005:
        return "Bearish"

    return "Neutral"


def predict_next_day():

    # --------------------------------------------------------
    # LOAD FEATURE DATA
    # --------------------------------------------------------

    if not FEATURE_FILE.exists() or not LATEST_FILE.exists():
        raise FileNotFoundError("Feature files are missing.")

    historical = pd.read_csv(FEATURE_FILE)

    latest = pd.read_csv(LATEST_FILE)

    if historical.empty or latest.empty:
        raise ValueError("Feature dataset is empty.")

    # --------------------------------------------------------
    # VALIDATE FEATURES
    # --------------------------------------------------------

    missing = set(FEATURES) - set(latest.columns)

    if missing:
        raise ValueError(f"Missing features: {missing}")

    if len(historical) < LOOKBACK - 1:
        raise ValueError("Not enough historical rows for LSTM.")

    # --------------------------------------------------------
    # LATEST MARKET INFORMATION
    # --------------------------------------------------------

    latest_row = latest.iloc[-1]

    latest_close = float(latest_row["Close"])

    latest_date = str(latest_row["Date"])

    if not np.isfinite(latest_close) or latest_close <= 0:
        raise ValueError("Invalid latest closing price.")

    # --------------------------------------------------------
    # PREPARE MODEL INPUT
    # --------------------------------------------------------

    latest_features = latest[FEATURES].tail(1)

    historical_features = historical[FEATURES].tail(
        LOOKBACK - 1
    )

    if latest_features.isnull().any().any():
        raise ValueError("Latest feature row contains missing values.")

    if historical_features.isnull().any().any():
        raise ValueError("Historical features contain missing values.")

    # Random Forest and XGBoost
    X_latest = latest_features

    # LSTM: last 19 historical rows + latest row
    lstm_input = pd.concat(
        [historical_features, latest_features],
        ignore_index=True
    )

    scaled_input = lstm_scaler.transform(lstm_input)

    lstm_sequence = scaled_input.reshape(
        1,
        LOOKBACK,
        len(FEATURES)
    )

    # --------------------------------------------------------
    # GENERATE PREDICTIONS
    # --------------------------------------------------------

    rf_return = float(
        rf_model.predict(X_latest)[0]
    )

    xgb_return = float(
        xgb_model.predict(X_latest)[0]
    )

    lstm_return = float(
        lstm_model.predict(
            lstm_sequence,
            verbose=0
        )[0][0]
    )

    predictions = {
        "LSTM": lstm_return,
        "Random Forest": rf_return,
        "XGBoost": xgb_return
    }

    # --------------------------------------------------------
    # FORMAT RESULTS
    # --------------------------------------------------------

    results = []

    for model_name, predicted_return in predictions.items():

        predicted_price = (
            latest_close * (1 + predicted_return)
        )

        results.append({

            "model": model_name,

            "predicted_return": round(
                predicted_return, 6
            ),

            "predicted_change_percent": round(
                predicted_return * 100, 4
            ),

            "predicted_price": round(
                predicted_price, 2
            ),

            "expected_change": round(
                predicted_price - latest_close, 2
            ),

            "signal": get_signal(predicted_return)

        })

    # --------------------------------------------------------
    # RETURN RESPONSE
    # --------------------------------------------------------

    return {

        "latest_date": latest_date,

        "latest_close": round(latest_close, 2),

        "forecast_type": "Next trading session",

        "confidence_warning": (
            "Experimental forecast. Historical evaluation "
            "does not establish reliable predictive performance."
        ),

        "predictions": results

    }
