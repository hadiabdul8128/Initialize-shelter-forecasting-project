"""Model selection, honest chronological evaluation, and recursive forecasting."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shelter_forecasting.features import (
    FEATURE_COLUMNS,
    build_supervised_frame,
    next_feature_row,
)


@dataclass(frozen=True)
class Candidate:
    """A small, auditable neural-network search space."""

    name: str
    hidden_layer_sizes: tuple[int, ...]
    alpha: float


DEFAULT_CANDIDATES = (
    Candidate("compact", (32,), 0.001),
    Candidate("two_layer", (64, 32), 0.001),
    Candidate("regularized", (64, 32), 0.01),
)


def make_model(
    candidate: Candidate,
    *,
    random_state: int = 42,
    max_iter: int = 1_000,
) -> TransformedTargetRegressor:
    """Construct a scaled multilayer perceptron regressor."""
    network = MLPRegressor(
        hidden_layer_sizes=candidate.hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=candidate.alpha,
        batch_size=32,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=max_iter,
        shuffle=False,
        random_state=random_state,
        tol=1e-5,
        n_iter_no_change=50,
    )
    pipeline = Pipeline(
        [
            ("feature_scaler", StandardScaler()),
            ("network", network),
        ]
    )
    return TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Compute metrics in people and percentage units."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mape_percent": float(np.mean(np.abs((actual - predicted) / actual)) * 100),
    }


def train_and_evaluate(
    history: pd.DataFrame,
    *,
    test_days: int = 90,
    validation_days: int = 90,
    random_state: int = 42,
    max_iter: int = 1_000,
    candidates: tuple[Candidate, ...] = DEFAULT_CANDIDATES,
) -> dict[str, Any]:
    """Select a network chronologically, evaluate it, then refit on all data."""
    supervised = build_supervised_frame(history)
    minimum_rows = test_days + validation_days + 60
    if len(supervised) < minimum_rows:
        raise ValueError(
            f"Need at least {minimum_rows} supervised rows; found {len(supervised)}"
        )

    pretest = supervised.iloc[:-test_days]
    test = supervised.iloc[-test_days:]
    train = pretest.iloc[:-validation_days]
    validation = pretest.iloc[-validation_days:]

    x_train = train[FEATURE_COLUMNS]
    y_train_change = train["population"] - train["lag_1"]
    x_validation = validation[FEATURE_COLUMNS]
    y_validation = validation["population"]

    search_results: list[dict[str, Any]] = []
    validation_predictions: dict[str, np.ndarray] = {}
    for candidate in candidates:
        model = make_model(candidate, random_state=random_state, max_iter=max_iter)
        model.fit(x_train, y_train_change)
        predicted = validation["lag_1"].to_numpy() + model.predict(x_validation)
        validation_predictions[candidate.name] = predicted
        search_results.append(
            {
                **asdict(candidate),
                "hidden_layer_sizes": list(candidate.hidden_layer_sizes),
                "validation": regression_metrics(y_validation.to_numpy(), predicted),
            }
        )

    selected_result = min(search_results, key=lambda item: item["validation"]["mae"])
    selected = next(
        candidate for candidate in candidates if candidate.name == selected_result["name"]
    )
    residuals = y_validation.to_numpy() - validation_predictions[selected.name]
    residual_quantiles = {
        "lower": float(np.quantile(residuals, 0.025)),
        "upper": float(np.quantile(residuals, 0.975)),
    }

    evaluation_model = make_model(selected, random_state=random_state, max_iter=max_iter)
    evaluation_model.fit(
        pretest[FEATURE_COLUMNS],
        pretest["population"] - pretest["lag_1"],
    )
    test_prediction = test["lag_1"].to_numpy() + evaluation_model.predict(
        test[FEATURE_COLUMNS]
    )

    test_predictions = test[["date", "population"]].rename(
        columns={"population": "actual"}
    )
    test_predictions["neural_network"] = test_prediction
    test_predictions["naive_previous_day"] = test["lag_1"].to_numpy()
    test_predictions["seasonal_naive_7_day"] = test["lag_7"].to_numpy()

    test_metrics = {
        "neural_network": regression_metrics(test_predictions["actual"], test_prediction),
        "naive_previous_day": regression_metrics(
            test_predictions["actual"], test_predictions["naive_previous_day"]
        ),
        "seasonal_naive_7_day": regression_metrics(
            test_predictions["actual"], test_predictions["seasonal_naive_7_day"]
        ),
    }

    final_model = make_model(selected, random_state=random_state, max_iter=max_iter)
    final_model.fit(
        supervised[FEATURE_COLUMNS],
        supervised["population"] - supervised["lag_1"],
    )

    metadata = {
        "target": "Total Individuals in Shelter",
        "modeled_quantity": "day-over-day population change",
        "feature_count": len(FEATURE_COLUMNS),
        "selected_candidate": selected_result,
        "candidate_search": search_results,
        "test_metrics": test_metrics,
        "splits": {
            "training_start": str(train["date"].min().date()),
            "training_end": str(train["date"].max().date()),
            "validation_start": str(validation["date"].min().date()),
            "validation_end": str(validation["date"].max().date()),
            "test_start": str(test["date"].min().date()),
            "test_end": str(test["date"].max().date()),
            "test_days": test_days,
            "validation_days": validation_days,
        },
        "residual_interval": {
            "method": "2.5th and 97.5th percentiles of chronological validation residuals",
            **residual_quantiles,
        },
        "random_state": random_state,
        "trained_through": str(history["date"].max().date()),
    }

    return {
        "model": final_model,
        "metadata": metadata,
        "test_predictions": test_predictions.reset_index(drop=True),
        "residual_quantiles": residual_quantiles,
    }


def recursive_forecast(
    history: pd.DataFrame,
    model: TransformedTargetRegressor,
    *,
    horizon: int,
    residual_quantiles: dict[str, float],
) -> pd.DataFrame:
    """Forecast future days recursively using predicted values as new lags."""
    if horizon < 1:
        raise ValueError("Forecast horizon must be at least one day")

    working = history[["date", "population"]].copy().sort_values("date")
    rows: list[dict[str, Any]] = []
    for step in range(1, horizon + 1):
        forecast_date = pd.Timestamp(working["date"].max()) + pd.offsets.Day(1)
        features = next_feature_row(working, forecast_date)
        prediction = float(features.iloc[0]["lag_1"] + model.predict(features)[0])
        interval_scale = np.sqrt(step)
        rows.append(
            {
                "date": forecast_date,
                "forecast_population": prediction,
                "lower_95_approx": (
                    prediction + interval_scale * residual_quantiles["lower"]
                ),
                "upper_95_approx": (
                    prediction + interval_scale * residual_quantiles["upper"]
                ),
                "horizon_day": step,
            }
        )
        working = pd.concat(
            [
                working,
                pd.DataFrame({"date": [forecast_date], "population": [prediction]}),
            ],
            ignore_index=True,
        )

    forecast = pd.DataFrame(rows)
    count_columns = ["forecast_population", "lower_95_approx", "upper_95_approx"]
    forecast[count_columns] = forecast[count_columns].round().astype(int)
    return forecast
