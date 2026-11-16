import pandas as pd
import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# FILE PATHS
# ============================================================

X_TRAIN_FILE = "DATA/FEATURES/TRAIN_TEST/X_train.csv"
X_TEST_FILE = "DATA/FEATURES/TRAIN_TEST/X_test.csv"
Y_TRAIN_FILE = "DATA/FEATURES/TRAIN_TEST/y_train.csv"
Y_TEST_FILE = "DATA/FEATURES/TRAIN_TEST/y_test.csv"

FEATURE_FILE = "DATA/FEATURES/NIFTY50_features.csv"

MODEL_FILE = "MODELS/random_forest_model.pkl"
PREDICTION_FILE = "EVALUATION/random_forest_predictions.csv"
GRAPH_FILE = "EVALUATION/random_forest_actual_vs_predicted.png"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading training and testing data...")

X_train = pd.read_csv(X_TRAIN_FILE)
X_test = pd.read_csv(X_TEST_FILE)
y_train = pd.read_csv(Y_TRAIN_FILE).squeeze()
y_test = pd.read_csv(Y_TEST_FILE).squeeze()

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# ============================================================
# CREATE RANDOM FOREST MODEL
# ============================================================

print("\nCreating Random Forest model...")

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    max_features="sqrt"
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("Training Random Forest...")

model.fit(X_train, y_train)

print("Training completed.")


# ============================================================
# PREDICT RETURNS
# ============================================================

print("\nGenerating return predictions...")

predicted_returns = model.predict(X_test)


# ============================================================
# RETURN METRICS
# ============================================================

return_mae = mean_absolute_error(y_test, predicted_returns)

return_rmse = np.sqrt(
    mean_squared_error(y_test, predicted_returns)
)

return_r2 = r2_score(y_test, predicted_returns)


# ============================================================
# CONVERT RETURNS TO PRICE
# ============================================================

features = pd.read_csv(FEATURE_FILE)

features["Date"] = pd.to_datetime(features["Date"])

features = features.sort_values("Date").reset_index(drop=True)


# Same 80/20 split used in prepare_data.py
split_index = int(len(features) * 0.80)

test_features = features.iloc[split_index:].copy()

today_close = test_features["Close"].values

actual_next_close = (
    today_close * (1 + y_test.values)
)

predicted_next_close = (
    today_close * (1 + predicted_returns)
)


# ============================================================
# PRICE METRICS
# ============================================================

price_mae = mean_absolute_error(
    actual_next_close,
    predicted_next_close
)

price_rmse = np.sqrt(
    mean_squared_error(
        actual_next_close,
        predicted_next_close
    )
)

price_mape = np.mean(
    np.abs(
        (actual_next_close - predicted_next_close)
        / actual_next_close
    )
) * 100

price_r2 = r2_score(
    actual_next_close,
    predicted_next_close
)


# ============================================================
# RESULTS
# ============================================================

print("\n==========================================")
print("RANDOM FOREST RESULTS")
print("==========================================")

print("\nRETURN PREDICTION")
print("------------------------------------------")
print(f"MAE  : {return_mae:.6f}")
print(f"RMSE : {return_rmse:.6f}")
print(f"R²   : {return_r2:.4f}")

print("\nPRICE PREDICTION")
print("------------------------------------------")
print(f"MAE  : {price_mae:.4f}")
print(f"RMSE : {price_rmse:.4f}")
print(f"MAPE : {price_mape:.4f}%")
print(f"R²   : {price_r2:.4f}")


# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving trained model...")

os.makedirs("MODELS", exist_ok=True)
os.makedirs("EVALUATION", exist_ok=True)

joblib.dump(model, MODEL_FILE)

print("Model saved to:")
print(MODEL_FILE)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({
    "Date": test_features["Date"].values,
    "Today_Close": today_close,
    "Actual_Next_Close": actual_next_close,
    "Predicted_Next_Close": predicted_next_close,
    "Actual_Return": y_test.values,
    "Predicted_Return": predicted_returns
})

prediction_df.to_csv(
    PREDICTION_FILE,
    index=False
)

print("\nPredictions saved to:")
print(PREDICTION_FILE)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(importance.head(10).to_string(index=False))


# ============================================================
# GRAPH
# ============================================================

print("\nCreating prediction graph...")

import matplotlib

# Use non-GUI backend to avoid Tkinter errors
matplotlib.use("Agg")

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.plot(
    prediction_df["Date"],
    prediction_df["Actual_Next_Close"],
    label="Actual Price"
)

plt.plot(
    prediction_df["Date"],
    prediction_df["Predicted_Next_Close"],
    label="Predicted Price"
)

plt.xlabel("Date")
plt.ylabel("NIFTY 50 Price")

plt.title(
    "Random Forest: Actual vs Predicted NIFTY 50 Price"
)

plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    GRAPH_FILE,
    dpi=300
)

plt.close()

print("Graph saved to:")
print(GRAPH_FILE)

print("\n==========================================")
print("RANDOM FOREST COMPLETED SUCCESSFULLY")
print("==========================================")