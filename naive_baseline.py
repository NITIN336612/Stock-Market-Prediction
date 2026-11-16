import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# SETTINGS
# ============================================================

FEATURE_FILE = "DATA/FEATURES/NIFTY50_features.csv"

PREDICTION_FILE = "EVALUATION/naive_baseline_predictions.csv"

# ============================================================
# LOAD DATA
# ============================================================

print("Loading feature dataset...")

df = pd.read_csv(FEATURE_FILE)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print("Dataset shape:", df.shape)

# ============================================================
# CHRONOLOGICAL 80:20 TEST SPLIT
# ============================================================

split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

print("\n==========================================")
print("NAIVE BASELINE")
print("==========================================")

print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))

print(
    "Testing period:",
    test_df["Date"].iloc[0],
    "to",
    test_df["Date"].iloc[-1]
)

# ============================================================
# NAIVE FORECAST
# ============================================================
# Baseline assumption:
# Tomorrow's closing price = today's closing price.
#
# Therefore:
# Predicted return = 0
# Predicted next close = today's close

today_close = test_df["Close"].values
actual_return = test_df["Target_Return"].values

predicted_return = np.zeros(len(test_df))

actual_next_close = today_close * (1 + actual_return)
predicted_next_close = today_close

# ============================================================
# RETURN METRICS
# ============================================================

return_mae = mean_absolute_error(
    actual_return,
    predicted_return
)

return_rmse = np.sqrt(
    mean_squared_error(
        actual_return,
        predicted_return
    )
)

return_r2 = r2_score(
    actual_return,
    predicted_return
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
# SAVE PREDICTIONS
# ============================================================

os.makedirs("EVALUATION", exist_ok=True)

prediction_df = pd.DataFrame({
    "Date": test_df["Date"],
    "Today_Close": today_close,
    "Actual_Next_Close": actual_next_close,
    "Predicted_Next_Close": predicted_next_close,
    "Actual_Return": actual_return,
    "Predicted_Return": predicted_return
})

prediction_df.to_csv(
    PREDICTION_FILE,
    index=False
)

print("\nPredictions saved to:")
print(PREDICTION_FILE)

print("\n==========================================")
print("NAIVE BASELINE COMPLETED SUCCESSFULLY")
print("==========================================")
