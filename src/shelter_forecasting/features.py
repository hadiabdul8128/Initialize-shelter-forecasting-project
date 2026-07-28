"""Leakage-safe time-series feature engineering."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

DEFAULT_LAGS = (1, 2, 3, 7, 14, 21, 28, 56)
DEFAULT_WINDOWS = (7, 14, 28)


def feature_columns(
    lags: Sequence[int] = DEFAULT_LAGS,
    windows: Sequence[int] = DEFAULT_WINDOWS,
) -> list[str]:
    """Return feature names in stable model-input order."""
    columns = [
        "trend_days",
        "day_of_week_sin",
        "day_of_week_cos",
        "day_of_year_sin",
        "day_of_year_cos",
    ]
    columns.extend(f"lag_{lag}" for lag in lags)
    for window in windows:
        columns.extend(
            [
                f"rolling_mean_{window}",
                f"rolling_std_{window}",
                f"rolling_min_{window}",
                f"rolling_max_{window}",
            ]
        )
    columns.extend(["momentum_7", "momentum_28"])
    return columns


FEATURE_COLUMNS = feature_columns()


def engineer_features(
    history: pd.DataFrame,
    lags: Sequence[int] = DEFAULT_LAGS,
    windows: Sequence[int] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Create features using observations strictly earlier than each target date."""
    required = {"date", "population"}
    if not required.issubset(history.columns):
        raise ValueError("Feature input must contain date and population columns")

    frame = history[["date", "population"]].copy().sort_values("date").reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"])
    origin = frame["date"].min()
    frame["trend_days"] = (frame["date"] - origin).dt.days.astype(float)

    day_of_week = frame["date"].dt.dayofweek
    frame["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    frame["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7)

    day_of_year = frame["date"].dt.dayofyear
    days_in_year = np.where(frame["date"].dt.is_leap_year, 366, 365)
    frame["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / days_in_year)
    frame["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / days_in_year)

    for lag in lags:
        frame[f"lag_{lag}"] = frame["population"].shift(lag)

    prior_population = frame["population"].shift(1)
    for window in windows:
        rolling = prior_population.rolling(window=window, min_periods=window)
        frame[f"rolling_mean_{window}"] = rolling.mean()
        frame[f"rolling_std_{window}"] = rolling.std(ddof=0)
        frame[f"rolling_min_{window}"] = rolling.min()
        frame[f"rolling_max_{window}"] = rolling.max()

    frame["momentum_7"] = frame["lag_1"] - frame["lag_7"]
    frame["momentum_28"] = frame["lag_1"] - frame["lag_28"]
    return frame


def build_supervised_frame(history: pd.DataFrame) -> pd.DataFrame:
    """Build complete feature/target rows for training or evaluation."""
    frame = engineer_features(history)
    return frame.dropna(subset=[*FEATURE_COLUMNS, "population"]).reset_index(drop=True)


def next_feature_row(history: pd.DataFrame, forecast_date: pd.Timestamp) -> pd.DataFrame:
    """Build one future feature row from observed or recursively predicted history."""
    forecast_date = pd.Timestamp(forecast_date)
    if forecast_date <= pd.Timestamp(history["date"].max()):
        raise ValueError("Forecast date must be later than the available history")

    future = pd.DataFrame({"date": [forecast_date], "population": [np.nan]})
    extended = pd.concat([history[["date", "population"]], future], ignore_index=True)
    row = engineer_features(extended).iloc[[-1]]
    if row[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Not enough history to build all forecasting features")
    return row[FEATURE_COLUMNS]
