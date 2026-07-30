"""Scikit-learn baseline, XGBoost, PyTorch evaluation, and forecasting."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from shelter_forecasting.features import FEATURE_COLUMNS, build_supervised_frame
from shelter_forecasting.neural import (
    DEFAULT_HIDDEN_SIZES,
    refit_neural_network,
    train_neural_network,
)


def make_sklearn_baseline(alpha: float = 10.0) -> Pipeline:
    """Build a regularized linear baseline for daily population change."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def make_xgboost_model(
    *,
    n_estimators: int = 400,
    random_state: int = 42,
) -> XGBRegressor:
    """Build a conservative boosted-tree model for a small daily dataset."""
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=8,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=8.0,
        random_state=random_state,
        n_jobs=1,
        tree_method="hist",
    )


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
    xgb_estimators: int = 400,
    neural_epochs: int = 500,
    neural_patience: int = 60,
) -> dict[str, Any]:
    """Train all three model families with chronological validation and testing."""
    supervised = build_supervised_frame(history)
    minimum_rows = test_days + validation_days + 60
    if len(supervised) < minimum_rows:
        raise ValueError(f"Need at least {minimum_rows} supervised rows; found {len(supervised)}")

    pretest = supervised.iloc[:-test_days]
    test = supervised.iloc[-test_days:]
    train = pretest.iloc[:-validation_days]
    validation = pretest.iloc[-validation_days:]

    x_train = train[FEATURE_COLUMNS]
    y_train_change = train["population"] - train["lag_1"]
    x_validation = validation[FEATURE_COLUMNS]
    y_validation_change = validation["population"] - validation["lag_1"]
    y_validation_level = validation["population"].to_numpy()

    ridge_selection = make_sklearn_baseline()
    ridge_selection.fit(x_train, y_train_change)
    ridge_validation = validation["lag_1"].to_numpy() + ridge_selection.predict(x_validation)

    xgboost_selection = make_xgboost_model(
        n_estimators=xgb_estimators,
        random_state=random_state,
    )
    xgboost_selection.fit(x_train, y_train_change)
    xgboost_validation = validation["lag_1"].to_numpy() + xgboost_selection.predict(x_validation)

    neural_selection = train_neural_network(
        x_train,
        y_train_change,
        x_validation,
        y_validation_change,
        epochs=neural_epochs,
        patience=neural_patience,
        random_state=random_state,
    )
    neural_validation = validation["lag_1"].to_numpy() + neural_selection.predict_change(
        x_validation
    )

    validation_metrics = {
        "sklearn_ridge": regression_metrics(y_validation_level, ridge_validation),
        "xgboost": regression_metrics(y_validation_level, xgboost_validation),
        "pytorch_neural_network": regression_metrics(y_validation_level, neural_validation),
    }
    residuals = y_validation_level - neural_validation
    residual_quantiles = {
        "lower": float(np.quantile(residuals, 0.025)),
        "upper": float(np.quantile(residuals, 0.975)),
    }

    x_pretest = pretest[FEATURE_COLUMNS]
    y_pretest_change = pretest["population"] - pretest["lag_1"]
    x_test = test[FEATURE_COLUMNS]

    ridge_evaluation = make_sklearn_baseline()
    ridge_evaluation.fit(x_pretest, y_pretest_change)
    ridge_test = test["lag_1"].to_numpy() + ridge_evaluation.predict(x_test)

    xgboost_evaluation = make_xgboost_model(
        n_estimators=xgb_estimators,
        random_state=random_state,
    )
    xgboost_evaluation.fit(x_pretest, y_pretest_change)
    xgboost_test = test["lag_1"].to_numpy() + xgboost_evaluation.predict(x_test)

    selected_epochs = neural_selection.best_epoch
    neural_evaluation = refit_neural_network(
        x_pretest,
        y_pretest_change,
        epochs=selected_epochs,
        random_state=random_state,
    )
    neural_test = test["lag_1"].to_numpy() + neural_evaluation.predict_change(x_test)

    test_predictions = test[["date", "population"]].rename(columns={"population": "actual"})
    test_predictions["sklearn_ridge"] = ridge_test
    test_predictions["xgboost"] = xgboost_test
    test_predictions["pytorch_neural_network"] = neural_test
    test_predictions["naive_previous_day"] = test["lag_1"].to_numpy()
    test_predictions["seasonal_naive_7_day"] = test["lag_7"].to_numpy()

    test_metrics = {
        column: regression_metrics(test_predictions["actual"], test_predictions[column])
        for column in [
            "sklearn_ridge",
            "xgboost",
            "pytorch_neural_network",
            "naive_previous_day",
            "seasonal_naive_7_day",
        ]
    }

    x_all = supervised[FEATURE_COLUMNS]
    y_all_change = supervised["population"] - supervised["lag_1"]
    ridge_final = make_sklearn_baseline()
    ridge_final.fit(x_all, y_all_change)
    xgboost_final = make_xgboost_model(
        n_estimators=xgb_estimators,
        random_state=random_state,
    )
    xgboost_final.fit(x_all, y_all_change)
    neural_final = refit_neural_network(
        x_all,
        y_all_change,
        epochs=selected_epochs,
        random_state=random_state,
    )

    metadata = {
        "target": "Total Individuals in Shelter",
        "modeled_quantity": "day-over-day population change",
        "primary_serving_model": "pytorch_neural_network",
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "model_settings": {
            "sklearn_ridge": {"alpha": 10.0},
            "xgboost": {
                "n_estimators": xgb_estimators,
                "learning_rate": 0.03,
                "max_depth": 3,
            },
            "pytorch_neural_network": {
                "hidden_sizes": list(DEFAULT_HIDDEN_SIZES),
                "dropout": 0.05,
                "optimizer": "AdamW",
                "loss": "HuberLoss",
                "selected_epochs": selected_epochs,
                "maximum_epochs": neural_epochs,
                "patience": neural_patience,
            },
        },
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
            "method": ("PyTorch validation residual percentiles, widened by square-root horizon"),
            **residual_quantiles,
        },
        "random_state": random_state,
        "trained_through": str(history["date"].max().date()),
    }

    return {
        "pytorch_predictor": neural_final,
        "ridge_model": ridge_final,
        "xgboost_model": xgboost_final,
        "metadata": metadata,
        "test_predictions": test_predictions.reset_index(drop=True),
        "residual_quantiles": residual_quantiles,
        "neural_training_history": neural_selection.training_history,
    }
