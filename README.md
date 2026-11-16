# Intelligent Stock Market Analytics

A Python project for exploring and forecasting the NIFTY 50 index. It downloads historical market data, builds technical-indicator features, trains and evaluates forecasting models, and serves market data and experimental next-session forecasts through a Flask dashboard.

> **Disclaimer:** Forecasts are experimental and are not financial advice or a guarantee of future market performance.

## Features

- Download NIFTY 50 daily history from Yahoo Finance (`^NSEI`).
- Clean market data and calculate returns, moving averages, RSI, MACD, volatility, and lag features.
- Train and compare a naive baseline, Random Forest, XGBoost, and LSTM.
- Explore model metrics, historical closing prices, and next-trading-session forecasts in a Flask dashboard.
- Access stock data, model evaluation, and forecast results through JSON endpoints.

## Requirements

- Python version compatible with the TensorFlow version pinned in `requirements.txt`
- Internet access to download market data

The project dependencies are pinned in `requirements.txt`.

## Setup

Run commands from the repository root.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents environment activation, use the virtual environment's interpreter directly (for example, `.venv\Scripts\python.exe`).

## Run the dashboard

The repository includes datasets, trained model files, evaluation results, and a SQLite database. With dependencies installed, start the application from the repository root:

```powershell
python BACKEND\app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in a browser. The dashboard loads its stock data and model metrics from `BACKEND/stock_market.db`; next-day forecasts also require the saved model artifacts in `MODELS/` and feature files in `DATA/FEATURES/`.

### API endpoints

| Endpoint | Description |
| --- | --- |
| `/api/stocks` | Historical stock records |
| `/api/stocks/latest` | Most recent stock record |
| `/api/models` | Model evaluation metrics |
| `/api/models/best` | Model with the lowest MAPE |
| `/api/status` | Database record counts and connection status |
| `/api/predictions/next-day` | Experimental forecasts from LSTM, Random Forest, and XGBoost |

## Rebuild the data and models (optional)

The following pipeline updates the existing artifacts. Training models can take significantly longer than starting the dashboard.

```powershell
# Download current NIFTY 50 history
python update_market_data.py

# Clean the data and generate indicators/features
python DATA\FEATURES\feacture_engineering.py

# Train/test split and run each model
python MODELS\prepare_data.py
python naive_baseline.py
python MODELS\random_forest.py
python MODELS\xgboost_model.py
python MODELS\lstm_model.py

# Compare model predictions and generate evaluation metrics/chart
python model_comparison.py

# Refresh the dashboard database from processed data and metrics
python BACKEND\load_data.py

# Start the dashboard
python BACKEND\app.py
```

The comparison script writes `EVALUATION/model_comparison.png`. To update the dashboard's chart image as well, copy it to `BACKEND/static/model_comparison.png` after running the comparison:

```powershell
Copy-Item EVALUATION\model_comparison.png BACKEND\static\model_comparison.png
```

`data_collection.py` is an earlier fixed-date data-download script; use `update_market_data.py` for the current historical-data update workflow.

## Repository layout

| Path | Contents |
| --- | --- |
| `BACKEND/` | Flask app, SQLite database helpers/loader, templates, and static assets |
| `DATA/RAW/` | Downloaded source market data |
| `DATA/PROCESSED/` | Cleaned market data |
| `DATA/FEATURES/` | Engineered features and train/test datasets |
| `EDA/` | Exploratory analysis script and plots |
| `MODELS/` | Model training scripts and saved model artifacts |
| `EVALUATION/` | Prediction outputs, metrics, and comparison charts |
| `requirements.txt` | Pinned Python dependencies |

## Notes

- Data is downloaded from Yahoo Finance and may be subject to its availability and terms.
- Model results describe historical test data; they do not establish reliable future performance.
- The feature-engineering script is named `feacture_engineering.py` in the repository.
