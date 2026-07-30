"""Persist model artifacts, evaluation tables, and matplotlib charts."""

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import pandas as pd

from shelter_forecasting.neural import save_neural_bundle

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def save_training_outputs(
    *,
    output_dir: Path,
    artifact_dir: Path,
    result: dict[str, Any],
    forecast: pd.DataFrame,
) -> dict[str, Path]:
    """Save metrics, predictions, model files, and diagnostic charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / "test_predictions.csv"
    forecast_path = output_dir / "forecast.csv"
    forecast_chart_path = output_dir / "forecast.png"
    training_chart_path = output_dir / "neural_training.png"
    training_history_path = output_dir / "neural_training_history.csv"

    metrics_path.write_text(
        json.dumps(result["metadata"], indent=2) + "\n",
        encoding="utf-8",
    )
    result["test_predictions"].to_csv(predictions_path, index=False)
    forecast.to_csv(forecast_path, index=False)
    result["neural_training_history"].to_csv(training_history_path, index=False)

    weights_path, neural_metadata_path = save_neural_bundle(
        result["pytorch_predictor"],
        artifact_dir=artifact_dir,
        residual_quantiles=result["residual_quantiles"],
        trained_through=result["metadata"]["trained_through"],
    )
    joblib.dump(result["ridge_model"], artifact_dir / "sklearn_ridge.joblib")
    result["xgboost_model"].save_model(artifact_dir / "xgboost_model.json")

    _save_forecast_chart(
        result["test_predictions"],
        forecast,
        forecast_chart_path,
    )
    _save_training_chart(result["neural_training_history"], training_chart_path)
    return {
        "metrics": metrics_path,
        "predictions": predictions_path,
        "forecast": forecast_path,
        "forecast_chart": forecast_chart_path,
        "training_chart": training_chart_path,
        "training_history": training_history_path,
        "pytorch_weights": weights_path,
        "neural_metadata": neural_metadata_path,
        "xgboost_model": artifact_dir / "xgboost_model.json",
    }


def _save_forecast_chart(
    test_predictions: pd.DataFrame,
    forecast: pd.DataFrame,
    destination: Path,
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

    axes[0].plot(
        test_predictions["date"],
        test_predictions["actual"],
        label="Actual",
        color="#172554",
        linewidth=2.2,
    )
    axes[0].plot(
        test_predictions["date"],
        test_predictions["pytorch_neural_network"],
        label="PyTorch neural network",
        color="#2563eb",
        linewidth=1.7,
    )
    axes[0].plot(
        test_predictions["date"],
        test_predictions["xgboost"],
        label="XGBoost",
        color="#16a34a",
        linewidth=1.3,
        alpha=0.9,
    )
    axes[0].plot(
        test_predictions["date"],
        test_predictions["sklearn_ridge"],
        label="scikit-learn Ridge",
        color="#9333ea",
        linewidth=1.2,
        alpha=0.85,
    )
    axes[0].plot(
        test_predictions["date"],
        test_predictions["naive_previous_day"],
        label="Previous-day naive",
        color="#94a3b8",
        linewidth=1,
        alpha=0.8,
    )
    axes[0].set_title("90-day chronological test window")
    axes[0].set_ylabel("People in shelter")
    axes[0].legend(frameon=False, ncol=3)
    axes[0].grid(alpha=0.2)

    axes[1].plot(
        forecast["date"],
        forecast["forecast_population"],
        marker="o",
        label="PyTorch forecast",
        color="#2563eb",
    )
    axes[1].fill_between(
        forecast["date"],
        forecast["lower_95_approx"],
        forecast["upper_95_approx"],
        color="#93c5fd",
        alpha=0.35,
        label="Approximate 95% residual interval",
    )
    axes[1].set_title(f"{len(forecast)}-day recursive forecast")
    axes[1].set_ylabel("People in shelter")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2)

    figure.suptitle("NYC homeless shelter census forecasting", fontsize=15)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def _save_training_chart(history: pd.DataFrame, destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    axis.plot(
        history["epoch"],
        history["train_huber_loss"],
        label="Training Huber loss",
        color="#2563eb",
    )
    if "validation_scaled_mae" in history:
        axis.plot(
            history["epoch"],
            history["validation_scaled_mae"],
            label="Validation scaled MAE",
            color="#f97316",
        )
    axis.set_title("PyTorch training history")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(destination, dpi=160)
    plt.close(figure)
