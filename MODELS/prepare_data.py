import pandas as pd
import os

from websockets import Close


# ============================================================
# 1. FILE PATH
# ============================================================

INPUT_FILE = "DATA/FEATURES/NIFTY50_features.csv"


# ============================================================
# 2. LOAD FEATURE DATA
# ============================================================

print("Loading feature dataset...")

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)


print("Dataset shape:", df.shape)
print("Date range:", df["Date"].min(), "to", df["Date"].max())


# ============================================================
# 3. SELECT FEATURES
# ============================================================

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
# TARGET = Close.shift(-1)

# ============================================================
# 4. CREATE X AND y
# ============================================================

X = df[FEATURES]

y = df[TARGET]


print("\nNumber of input features:", len(FEATURES))

print("Target:", TARGET)


# ============================================================
# 5. TIME-BASED TRAIN TEST SPLIT
# ============================================================

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]

X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]

y_test = y.iloc[split_index:]


# ============================================================
# 6. DISPLAY SPLIT INFORMATION
# ============================================================

print("\n==========================================")
print("TRAIN / TEST SPLIT")
print("==========================================")

print("Training samples:", len(X_train))

print("Testing samples:", len(X_test))

print("\nTraining period:")

print(
    df["Date"].iloc[0],
    "to",
    df["Date"].iloc[split_index - 1]
)

print("\nTesting period:")

print(
    df["Date"].iloc[split_index],
    "to",
    df["Date"].iloc[-1]
)


# ============================================================
# 7. SAVE SPLIT DATA
# ============================================================

os.makedirs("DATA/FEATURES/TRAIN_TEST", exist_ok=True)

X_train.to_csv(
    "DATA/FEATURES/TRAIN_TEST/X_train.csv",
    index=False
)

X_test.to_csv(
    "DATA/FEATURES/TRAIN_TEST/X_test.csv",
    index=False
)

y_train.to_csv(
    "DATA/FEATURES/TRAIN_TEST/y_train.csv",
    index=False
)

y_test.to_csv(
    "DATA/FEATURES/TRAIN_TEST/y_test.csv",
    index=False
)


print("\n==========================================")
print("DATA PREPARATION COMPLETED")
print("==========================================")

print("Training and testing datasets saved successfully.")