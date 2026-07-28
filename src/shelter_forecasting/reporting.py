"""Persist training outputs and human-readable diagnostic charts."""

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def save_training_outputs(
    *,
    output_dir: Path,
    artifact_dir: Path,
    result: dict[str, Any],
    forecast: pd.DataFrame,
) -> None:
    """Save metrics, predictions, forecasts, chart, and a reusable model bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "metrics.json").write_text(
        json.dumps(result["metadata"], indent=2) + "\n",
        encoding="utf-8",
    )
    result["test_predictions"].to_csv(output_dir / "test_predictions.csv", index=False)
    forecast.to_csv(output_dir / "forecast.csv", index=False)
    joblib.dump(
        {
            "model": result["model"],
            "metadata": result["metadata"],
            "residual_quantiles": result["residual_quantiles"],
        },
        artifact_dir / "shelter_mlp.joblib",
    )
    _save_chart(result["test_predictions"], forecast, output_dir / "forecast.png")


def _save_chart(
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
        linewidth=2,
    )
    axes[0].plot(
        test_predictions["date"],
        test_predictions["neural_network"],
        label="Neural network",
        color="#2563eb",
        linewidth=1.6,
    )
    axes[0].plot(
        test_predictions["date"],
        test_predictions["naive_previous_day"],
        label="Previous-day baseline",
        color="#94a3b8",
        linewidth=1,
        alpha=0.9,
    )
    axes[0].set_title("90-day chronological test window")
    axes[0].set_ylabel("People in shelter")
    axes[0].legend(frameon=False, ncol=3)
    axes[0].grid(alpha=0.2)

    axes[1].plot(
        forecast["date"],
        forecast["forecast_population"],
        marker="o",
        label="Forecast",
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
