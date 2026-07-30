"""Recursive forecasting shared by the CLI, FastAPI service, and tests."""

from typing import Any

import numpy as np
import pandas as pd

from shelter_forecasting.features import next_feature_row
from shelter_forecasting.neural import NeuralPredictor


def recursive_forecast(
    history: pd.DataFrame,
    predictor: NeuralPredictor,
    *,
    horizon: int,
    residual_quantiles: dict[str, float],
) -> pd.DataFrame:
    """Forecast future days recursively with the trained PyTorch network."""
    if horizon < 1:
        raise ValueError("Forecast horizon must be at least one day")

    working = history[["date", "population"]].copy().sort_values("date")
    rows: list[dict[str, Any]] = []
    for step in range(1, horizon + 1):
        forecast_date = pd.Timestamp(working["date"].max()) + pd.offsets.Day(1)
        features = next_feature_row(working, forecast_date)
        prediction = float(features.iloc[0]["lag_1"] + predictor.predict_change(features)[0])
        interval_scale = np.sqrt(step)
        rows.append(
            {
                "date": forecast_date,
                "forecast_population": prediction,
                "lower_95_approx": (prediction + interval_scale * residual_quantiles["lower"]),
                "upper_95_approx": (prediction + interval_scale * residual_quantiles["upper"]),
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
