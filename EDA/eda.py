import pandas as pd

# Load raw dataset
df = pd.read_csv("DATA/RAW/NIFTY50_raw.csv", skiprows=[1, 2])

# Rename the first column
df.rename(columns={"Price": "Date"}, inplace=True)

# Convert Date from text to datetime
df["Date"] = pd.to_datetime(df["Date"])

print("First 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nDescriptive statistics:")
print(df.describe(include="number"))

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.plot(df["Date"], df["Close"])

plt.title("NIFTY 50 Closing Price (2015–2025)")
plt.xlabel("Date")
plt.ylabel("Closing Price")

plt.tight_layout()

plt.savefig("EDA/plots/closing_price.png", dpi=300)
plt.show()

plt.close()

# Trading volume analysis

plt.figure(figsize=(12, 6))

plt.plot(df["Date"], df["Volume"])

plt.title("NIFTY 50 Trading Volume (2015–2025)")
plt.xlabel("Date")
plt.ylabel("Volume")

plt.tight_layout()

plt.savefig("EDA/plots/trading_volume.png", dpi=300)
plt.show()

plt.close()

# Calculate daily returns
df["Daily_Return"] = df["Close"].pct_change()

print("\nDaily return statistics:")
print(df["Daily_Return"].describe())

# Daily return visualization

plt.figure(figsize=(12, 6))

plt.plot(df["Date"], df["Daily_Return"])

plt.title("NIFTY 50 Daily Returns (2015–2025)")
plt.xlabel("Date")
plt.ylabel("Daily Return")

plt.tight_layout()

plt.savefig("EDA/plots/daily_returns.png", dpi=300)
plt.show()

plt.close()

# Distribution of daily returns

plt.figure(figsize=(10, 6))

plt.hist(df["Daily_Return"].dropna(), bins=50)

plt.title("Distribution of NIFTY 50 Daily Returns")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig("EDA/plots/return_distribution.png", dpi=300)
plt.show()

plt.close()
# Calculate 20-day rolling volatility
df["Volatility_20"] = df["Daily_Return"].rolling(window=20).std()

print("\n20-day rolling volatility statistics:")
print(df["Volatility_20"].describe())

# 20-day rolling volatility visualization

plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["Volatility_20"])
plt.title("NIFTY 50 20-Day Rolling Volatility (2015–2025)")
plt.xlabel("Date")
plt.ylabel("20-Day Volatility")
plt.tight_layout()

plt.savefig("EDA/plots/rolling_volatility_20.png", dpi=300)
plt.show()
plt.close()

# Correlation analysis

correlation = df[
    ["Open", "High", "Low", "Close", "Adj Close", "Volume",
     "Daily_Return", "Volatility_20"]
].corr()

print("\nCorrelation matrix:")
print(correlation)

import seaborn as sns
plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)
plt.title("Correlation Heatmap of NIFTY 50 Variables")
plt.tight_layout()

plt.savefig("EDA/plots/correlation_heatmap.png", dpi=300)
plt.show()
plt.close()



# Identify extreme daily returns

print("\nLargest negative returns:")
print(
    df.nsmallest(5, "Daily_Return")
    [["Date", "Daily_Return"]]
)

print("\nLargest positive returns:")
print(
    df.nlargest(5, "Daily_Return")
    [["Date", "Daily_Return"]]
)

print("\nDate range:")
print("Start:", df["Date"].min())
print("End:", df["Date"].max())

print("\nDates sorted:", df["Date"].is_monotonic_increasing)

# Remove the first NaN return
df = df.dropna(subset=["Daily_Return"])

# Save processed dataset
df.to_csv("DATA/PROCESSED/NIFTY50_processed.csv", index=False)

print("\nProcessed dataset saved successfully.")
print("Final shape:", df.shape)