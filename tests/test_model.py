import numpy as np
import pandas as pd

from shelter_forecasting.forecasting import recursive_forecast
from shelter_forecasting.model import train_and_evaluate


def synthetic_history(days: int = 300) -> pd.DataFrame:
    time = np.arange(days)
    population = 50_000 + 8 * time + 250 * np.sin(2 * np.pi * time / 7)
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=days, freq="D"),
            "population": population,
        }
    )


def test_all_model_families_and_recursive_forecast_smoke():
    history = synthetic_history()
    result = train_and_evaluate(
        history,
        test_days=30,
        validation_days=30,
        xgb_estimators=20,
        neural_epochs=40,
        neural_patience=8,
    )

    forecast = recursive_forecast(
        history,
        result["pytorch_predictor"],
        horizon=5,
        residual_quantiles=result["residual_quantiles"],
    )

    assert len(result["test_predictions"]) == 30
    assert set(result["metadata"]["test_metrics"]) == {
        "sklearn_ridge",
        "xgboost",
        "pytorch_neural_network",
        "naive_previous_day",
        "seasonal_naive_7_day",
    }
    assert len(forecast) == 5
    assert forecast["date"].min() == history["date"].max() + pd.offsets.Day(1)
    assert np.isfinite(forecast["forecast_population"]).all()
