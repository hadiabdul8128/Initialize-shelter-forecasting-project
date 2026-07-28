import numpy as np
import pandas as pd

from shelter_forecasting.features import build_supervised_frame, next_feature_row


def history_frame(days: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=days, freq="D"),
            "population": np.arange(1_000, 1_000 + days, dtype=float),
        }
    )


def test_features_use_only_prior_observations():
    history = history_frame()
    supervised = build_supervised_frame(history)
    first = supervised.iloc[0]
    source_index = history.index[history["date"] == first["date"]][0]

    assert first["lag_1"] == history.loc[source_index - 1, "population"]
    assert first["lag_56"] == history.loc[source_index - 56, "population"]
    assert first["rolling_mean_7"] == history.loc[
        source_index - 7 : source_index - 1, "population"
    ].mean()


def test_next_feature_row_uses_latest_value_as_lag_one():
    history = history_frame()
    forecast_date = history["date"].max() + pd.offsets.Day(1)

    row = next_feature_row(history, forecast_date)

    assert row.iloc[0]["lag_1"] == history.iloc[-1]["population"]
    assert row.iloc[0]["lag_7"] == history.iloc[-7]["population"]
