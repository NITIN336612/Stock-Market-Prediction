import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ============================================================
# SETTINGS
# ============================================================

FEATURE_FILE = "DATA/FEATURES/NIFTY50_features.csv"

MODEL_FILE = "MODELS/lstm_model.keras"
SCALER_FILE = "MODELS/lstm_scaler.pkl"

PREDICTION_FILE = "EVALUATION/lstm_predictions.csv"
GRAPH_FILE = "EVALUATION/lstm_actual_vs_predicted.png"

LOOKBACK = 20

FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Daily_Return",
    "Volatility_20",
    "MA_7",
    "MA_20",
    "MA_50",
    "EMA_12",
    "EMA_26",
    "MACD",
    "MACD_Signal",
    "RSI",
    "Lag_1",
    "Lag_5",
    "Lag_10",
    "Volume_MA_20"
]

TARGET = "Target_Return"

EPOCHS = 50
BATCH_SIZE = 32


# ============================================================
# LOAD DATA
# ============================================================

print("Loading feature dataset...")

df = pd.read_csv(FEATURE_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)

print("Dataset shape:", df.shape)

print(
    "Date range:",
    df["Date"].min(),
    "to",
    df["Date"].max()
)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index].copy()

test_df = df.iloc[split_index:].copy()

print("\n==========================================")
print("TRAIN / TEST SPLIT")
print("==========================================")

print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))

print(
    "Training period:",
    train_df["Date"].iloc[0],
    "to",
    train_df["Date"].iloc[-1]
)

print(
    "Testing period:",
    test_df["Date"].iloc[0],
    "to",
    test_df["Date"].iloc[-1]
)


# ============================================================
# FEATURE SCALING
# ============================================================

print("\nScaling features...")

scaler = StandardScaler()

# IMPORTANT:
# Fit scaler ONLY on training data.
# This prevents test-data leakage.

scaler.fit(train_df[FEATURES])

train_scaled = scaler.transform(
    train_df[FEATURES]
)

test_scaled = scaler.transform(
    test_df[FEATURES]
)

joblib.dump(
    scaler,
    SCALER_FILE
)

print("Scaler saved to:")
print(SCALER_FILE)


# ============================================================
# CREATE TRAINING SEQUENCES
# ============================================================

print("\nCreating LSTM sequences...")

X_train = []
y_train = []

train_targets = train_df[TARGET].values

for i in range(LOOKBACK, len(train_scaled)):

    X_train.append(
        train_scaled[i - LOOKBACK:i]
    )

    y_train.append(
        train_targets[i]
    )

X_train = np.array(X_train)

y_train = np.array(y_train)


# ============================================================
# CREATE TEST SEQUENCES
# ============================================================

X_test = []
y_test = []

test_targets = test_df[TARGET].values

test_dates = []
test_today_close = []

# To predict the first test day, we need the
# previous 20 observations. Those observations
# come from the end of the training period.

combined_features = pd.concat(
    [
        train_df[FEATURES].tail(LOOKBACK),
        test_df[FEATURES]
    ],
    ignore_index=True
)

combined_scaled = scaler.transform(
    combined_features
)

for i in range(
    LOOKBACK,
    len(combined_scaled)
):

    X_test.append(
        combined_scaled[i - LOOKBACK:i]
    )

    y_test.append(
        test_targets[i - LOOKBACK]
    )

    test_dates.append(
        test_df["Date"].iloc[i - LOOKBACK]
    )

    test_today_close.append(
        test_df["Close"].iloc[i - LOOKBACK]
    )

X_test = np.array(X_test)

y_test = np.array(y_test)

test_dates = np.array(test_dates)

test_today_close = np.array(
    test_today_close
)


# ============================================================
# DISPLAY SHAPES
# ============================================================

print("\n==========================================")
print("LSTM DATA")
print("==========================================")

print(
    "X_train shape:",
    X_train.shape
)

print(
    "y_train shape:",
    y_train.shape
)

print(
    "X_test shape:",
    X_test.shape
)

print(
    "y_test shape:",
    y_test.shape
)


# ============================================================
# BUILD LSTM MODEL
# ============================================================

print("\n==========================================")
print("BUILDING LSTM MODEL")
print("==========================================")

model = Sequential([
    Input(shape=(LOOKBACK, len(FEATURES))),

    LSTM(
        64,
        return_sequences=True
    ),

    Dropout(0.2),

    LSTM(
        32,
        return_sequences=False
    ),

    Dropout(0.2),

    Dense(16, activation="relu"),

    Dense(1)
])


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer="adam",
    loss="mse"
)

model.summary()


# ============================================================
# EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining LSTM...")

history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.10,
    shuffle=False,
    callbacks=[early_stopping],
    verbose=1
)

print("\nTraining completed.")


# ============================================================
# PREDICT RETURNS
# ============================================================

print("\nGenerating predictions...")

predicted_returns = model.predict(
    X_test,
    verbose=0
).flatten()


# ============================================================
# RETURN METRICS
# ============================================================

return_mae = mean_absolute_error(
    y_test,
    predicted_returns
)

return_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predicted_returns
    )
)

return_r2 = r2_score(
    y_test,
    predicted_returns
)


# ============================================================
# CONVERT RETURNS TO PRICE
# ============================================================

actual_next_close = (
    test_today_close *
    (1 + y_test)
)

predicted_next_close = (
    test_today_close *
    (1 + predicted_returns)
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
        (
            actual_next_close -
            predicted_next_close
        )
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
print("LSTM RESULTS")
print("==========================================")

print("\nRETURN PREDICTION")
print("------------------------------------------")

print(
    f"MAE  : {return_mae:.6f}"
)

print(
    f"RMSE : {return_rmse:.6f}"
)

print(
    f"R²   : {return_r2:.4f}"
)

print("\nPRICE PREDICTION")
print("------------------------------------------")

print(
    f"MAE  : {price_mae:.4f}"
)

print(
    f"RMSE : {price_rmse:.4f}"
)

print(
    f"MAPE : {price_mape:.4f}%"
)

print(
    f"R²   : {price_r2:.4f}"
)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    "MODELS",
    exist_ok=True
)

os.makedirs(
    "EVALUATION",
    exist_ok=True
)

model.save(
    MODEL_FILE
)

print("\nModel saved to:")
print(MODEL_FILE)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({

    "Date": test_dates,

    "Today_Close": test_today_close,

    "Actual_Next_Close": actual_next_close,

    "Predicted_Next_Close": predicted_next_close,

    "Actual_Return": y_test,

    "Predicted_Return": predicted_returns

})

prediction_df.to_csv(
    PREDICTION_FILE,
    index=False
)

print("\nPredictions saved to:")
print(PREDICTION_FILE)


# ============================================================
# PLOT
# ============================================================

print("\nCreating prediction graph...")

import matplotlib

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
    "LSTM: Actual vs Predicted NIFTY 50 Price"
)

plt.legend()

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    GRAPH_FILE,
    dpi=300
)

plt.close()

print("Graph saved to:")
print(GRAPH_FILE)


# ============================================================
# COMPLETION
# ============================================================

print("\n==========================================")
print("LSTM COMPLETED SUCCESSFULLY")
print("==========================================")