# Shelter population forecasting

An end-to-end neural-network project for forecasting the daily number of people
in New York City homeless shelters. The repository includes a reproducible data
snapshot, leakage-safe feature engineering, chronological model selection,
baseline comparisons, tests, and a command-line workflow.

## What the model predicts

The target is **Total Individuals in Shelter**. The forecaster uses only
information that would exist at prediction time:

- prior population values at 1, 2, 3, 7, 14, 21, 28, and 56 days;
- rolling statistics calculated from prior days;
- weekly and annual calendar cycles; and
- a time trend.

The other same-day census columns are deliberately excluded. They are useful
descriptively, but their future values would not be known when making a real
forecast and would introduce leakage.

## Quick start

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
shelter-forecast train
```

The training command:

1. validates and sorts the daily data;
2. reserves the latest 90 days for a final test;
3. uses the preceding 90 days to choose among three MLP architectures;
4. compares the selected network with previous-day and seven-day baselines;
5. refits the selected architecture on all available history; and
6. produces a recursive 14-day forecast.

Run the quality checks with:

```bash
python -m ruff check .
python -m pytest
```

After training, reuse the saved model with:

```bash
shelter-forecast forecast --horizon 30
```

## Results

The selected network has one hidden layer with 32 units. On the untouched
90-day test period from April 29 through July 27, 2026:

| Model | MAE (people) | RMSE (people) | MAPE |
|---|---:|---:|---:|
| Neural network | **107.1** | **137.3** | **0.129%** |
| Previous-day baseline | 118.9 | 154.1 | 0.144% |
| Seven-day baseline | 180.2 | 216.4 | 0.218% |

The network reduced mean absolute error by 9.9% relative to the strongest
baseline. These scores describe one-day-ahead conditional forecasts, not the
full recursive 14-day path.

## Outputs

- `reports/metrics.json`: model selection, split dates, and final test metrics
- `reports/test_predictions.csv`: actual and predicted values in the holdout
- `reports/forecast.csv`: future point forecasts and approximate intervals
- `reports/forecast.png`: diagnostic and forecast chart
- `artifacts/shelter_mlp.joblib`: local reusable model bundle (git-ignored)

The interval is based on validation residual quantiles and widens with the
square root of the forecast horizon. It is an approximate error band, not a
calibrated probabilistic forecast.

## Data

The included snapshot, `data/raw/DHS_Homeless_Shelter_Census_20260728.csv`,
contains 1,975 complete daily observations from March 1, 2021 through July 27,
2026. It has no missing calendar days, duplicate dates, or null values.

The dataset is aggregate census data and contains no person-level records.

## Model and evaluation

The model is a scikit-learn `MLPRegressor`: a feed-forward multilayer
perceptron with ReLU activations. It learns the day-over-day population change
and adds that change to the previous census count. This residual formulation
gives the network a strong persistence anchor. Inputs and the change target are
standardized during training. Architecture selection and final evaluation are
separated in time so the reported test period is not used to select the
network.

The test metrics in `reports/metrics.json` are one-day-ahead conditional
forecasts: each test row uses the population history that would have been
observed before that date. Future forecasts are recursive, so uncertainty and
error can grow with the horizon.

## Limitations

- This is a decision-support prototype, not an operational staffing guarantee.
- Structural breaks, policy changes, extreme weather, migration, and capacity
  changes are not represented directly.
- Daily census history alone cannot explain why demand changes.
- Retrain after adding new observations and monitor performance against the
  included baselines.
- Before production use, add forecast monitoring, stronger uncertainty
  calibration, and relevant external drivers with publication-time checks.
