"""Local or remote MLflow experiment tracking."""

import json
from pathlib import Path
from typing import Any

import mlflow


def record_mlflow_runs(
    *,
    metadata: dict[str, Any],
    paths: dict[str, Path],
    tracking_uri: str,
    experiment_name: str,
) -> dict[str, str]:
    """Record comparable sklearn, XGBoost, and PyTorch runs in MLflow."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    run_ids: dict[str, str] = {}

    for model_name in ["sklearn_ridge", "xgboost", "pytorch_neural_network"]:
        with mlflow.start_run(run_name=model_name) as run:
            run_ids[model_name] = run.info.run_id
            settings = metadata["model_settings"][model_name]
            mlflow.log_params(
                {
                    key: json.dumps(value) if isinstance(value, (dict, list)) else value
                    for key, value in settings.items()
                }
            )
            mlflow.log_params(
                {
                    "target": metadata["target"],
                    "feature_count": metadata["feature_count"],
                    "training_end": metadata["splits"]["training_end"],
                    "validation_end": metadata["splits"]["validation_end"],
                    "test_end": metadata["splits"]["test_end"],
                    "random_state": metadata["random_state"],
                }
            )
            validation = metadata["validation_metrics"][model_name]
            test = metadata["test_metrics"][model_name]
            mlflow.log_metrics(
                {
                    "validation_mae": validation["mae"],
                    "validation_rmse": validation["rmse"],
                    "validation_mape_percent": validation["mape_percent"],
                    "test_mae": test["mae"],
                    "test_rmse": test["rmse"],
                    "test_mape_percent": test["mape_percent"],
                }
            )
            mlflow.set_tags(
                {
                    "pipeline_stage": model_name,
                    "forecast_target": "daily_shelter_population",
                    "serving_model": str(model_name == metadata["primary_serving_model"]).lower(),
                }
            )
            if model_name == "pytorch_neural_network":
                for key in [
                    "metrics",
                    "forecast",
                    "predictions",
                    "forecast_chart",
                    "training_chart",
                    "training_history",
                    "pytorch_weights",
                    "neural_metadata",
                ]:
                    mlflow.log_artifact(str(paths[key]))
            elif model_name == "xgboost":
                mlflow.log_artifact(str(paths["xgboost_model"]))

    return run_ids
