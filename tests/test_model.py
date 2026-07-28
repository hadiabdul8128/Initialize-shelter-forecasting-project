import numpy as np
import pandas as pd
import pytest

from shelter_forecasting.model import Candidate, recursive_forecast, train_and_evaluate


def synthetic_history(days: int = 280) -> pd.DataFrame:
    time = np.arange(days)
    population = 50_000 + 8 * time + 250 * np.sin(2 * np.pi * time / 7)
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=days, freq="D"),
            "population": population,
        }
    )


@pytest.mark.filterwarnings(
    "ignore:Stochastic Optimizer.*:sklearn.exceptions.ConvergenceWarning"
)
def test_training_and_recursive_forecast_smoke():
    history = synthetic_history()
    candidates = (Candidate("test_network", (8,), 0.01),)
    result = train_and_evaluate(
        history,
        test_days=30,
        validation_days=30,
        max_iter=80,
        candidates=candidates,
    )

    forecast = recursive_forecast(
        history,
        result["model"],
        horizon=5,
        residual_quantiles=result["residual_quantiles"],
    )

    assert len(result["test_predictions"]) == 30
    assert result["metadata"]["selected_candidate"]["name"] == "test_network"
    assert len(forecast) == 5
    assert forecast["date"].min() == history["date"].max() + pd.offsets.Day(1)
    assert np.isfinite(forecast["forecast_population"]).all()
