# NYC shelter population forecast

This project estimates how many people will be in New York City homeless
shelters on a selected date.

The website has one date field. For a date already covered by the census data,
it returns the recorded population. For a later date, it asks a trained PyTorch
neural network for an estimate and shows an approximate range around that
number.

The latest census record in this version is July 27, 2026. Forecasts are
available for the following 90 days. These numbers are meant for planning and
testing. They are not official census totals or staffing guarantees.

## What is included

- A simple website for choosing a date and viewing the result
- A FastAPI endpoint that the website uses to request estimates
- A PyTorch neural network trained on daily shelter census history
- Ridge and XGBoost models used to compare performance
- Data checks, saved model files, test reports, and charts
- Docker support for running the same website and API in a container

## Pipeline

```text
CSV file
   ↓
pandas: reads and sorts the daily census
   ↓
Pandera: validates columns, types, nonnegative counts, duplicates, and gaps
   ↓
NumPy: calculates calendar cycles, lags, momentum, and rolling features
   ↓
scikit-learn: builds the regularized Ridge baseline
   ↓
XGBoost: builds the boosted-tree comparison model
   ↓
PyTorch: builds and trains the serving neural network
   ↓
MLflow: records parameters, metrics, model files, and charts in local SQLite
   ↓
matplotlib: draws test predictions, forecasts, and training history
   ↓
FastAPI: serves health checks and prediction requests
   ↓
Docker: packages the API, model, and data snapshot
   ↓
Git/GitHub: saves the project history
```

No cloud account is required. MLflow uses `mlflow.db` and `mlartifacts/` in the
local project directory by default.

## Results

All models learn the next-day change and add it to the previous census count.
The latest 90 observations, April 29 through July 27, 2026, are untouched until
the final test.

| Model | MAE (people) | RMSE (people) | MAPE |
|---|---:|---:|---:|
| PyTorch neural network | **101.1** | **130.7** | **0.122%** |
| XGBoost | 109.7 | 144.9 | 0.132% |
| Previous-day naive | 118.9 | 154.1 | 0.144% |
| scikit-learn Ridge | 124.1 | 160.6 | 0.150% |
| Seven-day naive | 180.2 | 216.4 | 0.218% |

The PyTorch network reduced MAE by **15.0%** relative to the strongest naive
benchmark. It has hidden layers of 64 and 32 units, ReLU activations, 5%
dropout, AdamW optimization, and Huber loss. Chronological validation selected
176 training epochs.

## Quick start

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
shelter-forecast train
```

Training validates the data, evaluates every model, records three MLflow runs,
refits on all available history, saves model artifacts, and creates a recursive
14-day forecast.

Run quality checks:

```bash
python -m ruff check .
python -m pytest
```

Reuse the saved PyTorch model:

```bash
shelter-forecast forecast --horizon 30
```

## MLflow

Training uses a local SQLite backend, so no MLflow or Databricks account is
needed. Open the experiment UI after training:

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 127.0.0.1 \
  --port 5000
```

Then visit `http://127.0.0.1:5000`.

## FastAPI

Start the service:

```bash
shelter-forecast serve
```

Open `http://127.0.0.1:8000` to use the date estimator. The page defaults to
the current date and shows either the recorded census or a PyTorch forecast
with an approximate range. Interactive API documentation is available at
`http://127.0.0.1:8000/docs`.

Request one date directly:

```bash
curl "http://127.0.0.1:8000/api/estimate?target_date=2026-07-29"
```

Request a forecast from the bundled history:

```bash
curl -X POST http://127.0.0.1:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"horizon": 14}'
```

The request may also contain `observations`, a list of at least 56 consecutive
objects with `date` and `population`. Pandera validates supplied history before
inference. Forecast horizons are limited to 1 to 90 days.

The date estimator accepts dates from March 1, 2021 through 90 days after the
latest observation. Historical dates return the recorded value; later dates
return the recursive neural-network estimate.

## Docker

Build and run the same prediction service:

```bash
docker build -t shelter-forecasting:local .
docker run --rm -p 8000:8000 shelter-forecasting:local
```

Check the container:

```bash
curl http://127.0.0.1:8000/health
```

## Outputs

- `artifacts/pytorch_model.pt`: framework-native PyTorch weights used by the API
- `artifacts/neural_metadata.json`: feature scaling and serving metadata
- `artifacts/xgboost_model.json`: reusable boosted-tree model
- `reports/metrics.json`: validation/test metrics and MLflow run IDs
- `reports/test_predictions.csv`: actual values and every model prediction
- `reports/forecast.csv`: recursive PyTorch forecast and approximate intervals
- `reports/forecast.png`: holdout comparison and forecast chart
- `reports/neural_training.png`: PyTorch learning curves
- `mlflow.db` and `mlartifacts/`: local MLflow state, intentionally git-ignored

The interval uses PyTorch validation residual quantiles and widens with the
square root of the horizon. It is an approximate error band rather than a
calibrated probabilistic forecast.

## Data and leakage controls

The included snapshot has 1,975 complete daily observations from March 1, 2021
through July 27, 2026. It has no missing days, duplicate dates, negative counts,
or nulls.

Only information available before each prediction is used:

- population lags at 1, 2, 3, 7, 14, 21, 28, and 56 days;
- 7-, 14-, and 28-day rolling statistics calculated after a one-day shift;
- weekly and annual calendar cycles;
- time trend and lag momentum.

Same-day shelter-category fields are excluded because they would be unknown for
a future date and would leak information into the model.

## Limitations

- This is an experimental forecasting tool, not an operational staffing
  guarantee.
- Structural breaks, policy changes, extreme weather, migration, and capacity
  changes are not represented directly.
- Daily census history alone cannot explain why demand changes.
- Retrain after adding new observations and monitor performance against both
  naive and trained baselines.
- Production deployment should add authentication, request logging, monitoring,
  calibrated uncertainty, and external drivers with publication-time checks.
