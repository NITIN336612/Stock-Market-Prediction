import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "DATA" / "RAW" / "NIFTY50_yfinance_full.csv"

PROCESSED_FILE = ROOT / "DATA" / "PROCESSED" / "NIFTY50_processed.csv"

FEATURE_FILE = ROOT / "DATA" / "FEATURES" / "NIFTY50_features.csv"

LATEST_FILE = ROOT / "DATA" / "FEATURES" / "NIFTY50_latest_features.csv"
print("Project root:", ROOT)
print("Input file:", INPUT_FILE)
print("Input exists:", INPUT_FILE.exists())

# ============================================================
# 2. LOAD RAW DATA
# ============================================================

print("\nLoading updated NIFTY 50 data...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

print("Original shape:", df.shape)


# ============================================================
# 3. CLEAN DATA
# ============================================================

df["Date"] = pd.to_datetime(df["Date"])

required_columns = [
    "Date", "Open", "High", "Low", "Close", "Volume"
]

df = df[required_columns].copy()

for column in required_columns[1:]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(
    subset=["Date", "Open", "High", "Low", "Close"]
)

df = df.drop_duplicates(subset=["Date"])

df = df.sort_values("Date").reset_index(drop=True)


# ============================================================
# 4. SAVE PROCESSED DATA
# ============================================================

PROCESSED_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(PROCESSED_FILE, index=False)

print("Processed data saved:", len(df))


# ============================================================
# 5. DAILY RETURNS
# ============================================================

print("Calculating daily returns...")

df["Daily_Return"] = df["Close"].pct_change()


# ============================================================
# 6. MOVING AVERAGES
# ============================================================

print("Calculating moving averages...")

df["MA_7"] = df["Close"].rolling(7).mean()

df["MA_20"] = df["Close"].rolling(20).mean()

df["MA_50"] = df["Close"].rolling(50).mean()


# ============================================================
# 7. EXPONENTIAL MOVING AVERAGES
# ============================================================

print("Calculating EMA...")

df["EMA_12"] = df["Close"].ewm(
    span=12,
    adjust=False
).mean()

df["EMA_26"] = df["Close"].ewm(
    span=26,
    adjust=False
).mean()


# ============================================================
# 8. MACD
# ============================================================

print("Calculating MACD...")

df["MACD"] = df["EMA_12"] - df["EMA_26"]

df["MACD_Signal"] = df["MACD"].ewm(
    span=9,
    adjust=False
).mean()


# ============================================================
# 9. RSI
# ============================================================

print("Calculating RSI...")

price_change = df["Close"].diff()

gain = price_change.clip(lower=0)

loss = -price_change.clip(upper=0)

average_gain = gain.rolling(14).mean()

average_loss = loss.rolling(14).mean()

relative_strength = average_gain / average_loss

df["RSI"] = 100 - (
    100 / (1 + relative_strength)
)


# ============================================================
# 10. VOLATILITY
# ============================================================

print("Calculating volatility...")

df["Volatility_20"] = (
    df["Daily_Return"]
    .rolling(20)
    .std()
)


# ============================================================
# 11. LAG FEATURES
# ============================================================

print("Creating lag features...")

df["Lag_1"] = df["Close"].shift(1)

df["Lag_5"] = df["Close"].shift(5)

df["Lag_10"] = df["Close"].shift(10)


# ============================================================
# 12. VOLUME FEATURES
# ============================================================

print("Calculating volume moving average...")

df["Volume_MA_20"] = (
    df["Volume"]
    .rolling(20)
    .mean()
)


# ============================================================
# 13. TARGET RETURN
# ============================================================

print("Creating next-day prediction target...")

df["Target_Return"] = (
    df["Close"].shift(-1) / df["Close"] - 1
)


# ============================================================
# 14. HANDLE INVALID VALUES
# ============================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

# Preserve the latest row separately.
# Historical training data requires a known target.

FEATURE_COLUMNS = [
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

latest = df.dropna(
    subset=FEATURE_COLUMNS
).tail(1).copy()

training_data = df.dropna().copy()


# ============================================================
# 15. SAVE DATASETS
# ============================================================

FEATURE_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

training_data.to_csv(
    FEATURE_FILE,
    index=False
)

latest.to_csv(
    LATEST_FILE,
    index=False
)


# ============================================================
# 16. SUMMARY
# ============================================================

print("\n========================================")
print("FEATURE ENGINEERING COMPLETED")
print("========================================")

print("Raw records:", len(pd.read_csv(INPUT_FILE)))

print("Processed records:", len(df))

print("Training records:", len(training_data))

print("Latest prediction record:", len(latest))

print("Latest date:", latest["Date"].iloc[0])

print("Latest close:", latest["Close"].iloc[0])

print("\nTraining dataset:", FEATURE_FILE)

print("Latest features:", LATEST_FILE)

print("\n========================================")