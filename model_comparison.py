import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# SETTINGS
# ============================================================

MODEL_FILES = {
    "Naive Baseline": "EVALUATION/naive_baseline_predictions.csv",
    "Random Forest": "EVALUATION/random_forest_predictions.csv",
    "XGBoost": "EVALUATION/xgboost_predictions.csv",
    "LSTM": "EVALUATION/lstm_predictions.csv"
}

OUTPUT_FILE = "EVALUATION/model_comparison.csv"
GRAPH_FILE = "EVALUATION/model_comparison.png"

# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(file_path):
    df = pd.read_csv(file_path)

    actual_return = df["Actual_Return"].values
    predicted_return = df["Predicted_Return"].values

    actual_price = df["Actual_Next_Close"].values
    predicted_price = df["Predicted_Next_Close"].values

    return {
        "Return_MAE": mean_absolute_error(
            actual_return,
            predicted_return
        ),
        "Return_RMSE": np.sqrt(
            mean_squared_error(
                actual_return,
                predicted_return
            )
        ),
        "Return_R2": r2_score(
            actual_return,
            predicted_return
        ),
        "Price_MAE": mean_absolute_error(
            actual_price,
            predicted_price
        ),
        "Price_RMSE": np.sqrt(
            mean_squared_error(
                actual_price,
                predicted_price
            )
        ),
        "MAPE_Percent": np.mean(
            np.abs(
                (actual_price - predicted_price)
                / actual_price
            )
        ) * 100,
        "Price_R2": r2_score(
            actual_price,
            predicted_price
        )
    }

# ============================================================
# CHECK FILES
# ============================================================

print("Checking prediction files...")

missing = []

for model, file_path in MODEL_FILES.items():
    if not os.path.exists(file_path):
        missing.append(f"{model}: {file_path}")

if missing:
    print("\nMissing prediction files:")
    for item in missing:
        print("-", item)

    print("\nRun the missing model before running comparison.")
    raise SystemExit(1)

# ============================================================
# CALCULATE COMPARISON
# ============================================================

results = []

for model, file_path in MODEL_FILES.items():

    print(f"\nEvaluating: {model}")

    metrics = calculate_metrics(file_path)

    results.append({
        "Model": model,
        **metrics
    })

comparison_df = pd.DataFrame(results)

# Round values for readable output
comparison_df = comparison_df.round({
    "Return_MAE": 6,
    "Return_RMSE": 6,
    "Return_R2": 4,
    "Price_MAE": 4,
    "Price_RMSE": 4,
    "MAPE_Percent": 4,
    "Price_R2": 4
})

# ============================================================
# SELECT BEST MODEL
# ============================================================

best_model = comparison_df.loc[
    comparison_df["MAPE_Percent"].idxmin(),
    "Model"
]

# ============================================================
# DISPLAY
# ============================================================

print("\n==========================================")
print("MODEL COMPARISON")
print("==========================================")

print(
    comparison_df[
        [
            "Model",
            "Price_MAE",
            "Price_RMSE",
            "MAPE_Percent",
            "Price_R2",
            "Return_R2"
        ]
    ].to_string(index=False)
)

print("\nBest model based on lowest price MAPE:")
print(best_model)

# ============================================================
# SAVE CSV
# ============================================================

os.makedirs("EVALUATION", exist_ok=True)

comparison_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nComparison saved to:")
print(OUTPUT_FILE)

# ============================================================
# CREATE COMPARISON GRAPH
# ============================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

models = comparison_df["Model"]
mape = comparison_df["MAPE_Percent"]

plt.figure(figsize=(10, 6))
plt.bar(models, mape)
plt.xlabel("Model")
plt.ylabel("MAPE (%)")
plt.title("Model Comparison: MAPE")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(GRAPH_FILE, dpi=300)
plt.close()

print("Comparison graph saved to:")
print(GRAPH_FILE)

print("\n==========================================")
print("MODEL COMPARISON COMPLETED SUCCESSFULLY")
print("==========================================")
