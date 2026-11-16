import pandas as pd
import numpy as np
import os

# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "DATA/FEATURES/NIFTY50_features.csv"

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


# ============================================================
# LOAD DATA
# ============================================================

print("Loading feature dataset...")

df = pd.read_csv(INPUT_FILE)

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
# SELECT FEATURES AND TARGET
# ============================================================

X = df[FEATURES].values
y = df[TARGET].values

dates = df["Date"].values
close_prices = df["Close"].values

print("\nNumber of features:", len(FEATURES))
print("Lookback window:", LOOKBACK)
print("Target:", TARGET)


# ============================================================
# CHRONOLOGICAL TRAIN/TEST SPLIT
# ============================================================

split_index = int(len(df) * 0.80)

print("\n==========================================")
print("CHRONOLOGICAL SPLIT")
print("==========================================")

print("Split index:", split_index)

print(
    "Training period:",
    df["Date"].iloc[0],
    "to",
    df["Date"].iloc[split_index - 1]
)

print(
    "Testing period:",
    df["Date"].iloc[split_index],
    "to",
    df["Date"].iloc[-1]
)


# ============================================================
# CREATE SEQUENCES
# ============================================================

X_sequences = []
y_sequences = []
sequence_dates = []
sequence_close = []

for i in range(LOOKBACK, len(df)):

    sequence = X[i - LOOKBACK:i]

    target = y[i]

    X_sequences.append(sequence)
    y_sequences.append(target)

    sequence_dates.append(dates[i])
    sequence_close.append(close_prices[i])


X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

sequence_dates = np.array(sequence_dates)
sequence_close = np.array(sequence_close)


# ============================================================
# SPLIT SEQUENCES
# ============================================================

# The target at index i belongs to the corresponding
# original dataframe row i.

train_mask = sequence_dates < dates[split_index]

test_mask = sequence_dates >= dates[split_index]

X_train_lstm = X_sequences[train_mask]
X_test_lstm = X_sequences[test_mask]

y_train_lstm = y_sequences[train_mask]
y_test_lstm = y_sequences[test_mask]

dates_train = sequence_dates[train_mask]
dates_test = sequence_dates[test_mask]

close_train = sequence_close[train_mask]
close_test = sequence_close[test_mask]


# ============================================================
# DISPLAY SHAPES
# ============================================================

print("\n==========================================")
print("LSTM SEQUENCE DATA")
print("==========================================")

print(
    "X_train shape:",
    X_train_lstm.shape
)

print(
    "X_test shape:",
    X_test_lstm.shape
)

print(
    "y_train shape:",
    y_train_lstm.shape
)

print(
    "y_test shape:",
    y_test_lstm.shape
)


# ============================================================
# SAVE DATA
# ============================================================

OUTPUT_DIR = "DATA/FEATURES/LSTM"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

np.save(
    f"{OUTPUT_DIR}/X_train_lstm.npy",
    X_train_lstm
)

np.save(
    f"{OUTPUT_DIR}/X_test_lstm.npy",
    X_test_lstm
)

np.save(
    f"{OUTPUT_DIR}/y_train_lstm.npy",
    y_train_lstm
)

np.save(
    f"{OUTPUT_DIR}/y_test_lstm.npy",
    y_test_lstm
)

np.save(
    f"{OUTPUT_DIR}/dates_test_lstm.npy",
    dates_test
)

np.save(
    f"{OUTPUT_DIR}/close_test_lstm.npy",
    close_test
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n==========================================")
print("LSTM DATA PREPARATION COMPLETED")
print("==========================================")

print("Files saved in:")
print(OUTPUT_DIR)