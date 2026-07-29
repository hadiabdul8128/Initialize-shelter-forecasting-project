"""Command-line interface for training, tracking, forecasting, and serving."""

import argparse
import json
from pathlib import Path

from shelter_forecasting.data import load_census
from shelter_forecasting.forecasting import recursive_forecast
from shelter_forecasting.model import train_and_evaluate
from shelter_forecasting.neural import load_neural_bundle
from shelter_forecasting.reporting import save_training_outputs
from shelter_forecasting.tracking import record_mlflow_runs

DEFAULT_DATA = Path("data/raw/DHS_Homeless_Shelter_Census_20260728.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shelter-forecast",
        description="Train and serve a daily shelter population forecast.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train, evaluate, track, and forecast")
    train.add_argument("--data", type=Path, default=DEFAULT_DATA)
    train.add_argument("--reports-dir", type=Path, default=Path("reports"))
    train.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    train.add_argument("--horizon", type=int, default=14)
    train.add_argument("--test-days", type=int, default=90)
    train.add_argument("--validation-days", type=int, default=90)
    train.add_argument("--xgb-estimators", type=int, default=400)
    train.add_argument("--neural-epochs", type=int, default=500)
    train.add_argument("--neural-patience", type=int, default=60)
    train.add_argument("--random-state", type=int, default=42)
    train.add_argument(
        "--tracking-uri",
        help="MLflow URI; defaults to a local SQLite database",
    )
    train.add_argument(
        "--experiment-name",
        default="shelter-population-forecasting",
    )

    forecast = subparsers.add_parser("forecast", help="Forecast with saved PyTorch weights")
    forecast.add_argument("--data", type=Path, default=DEFAULT_DATA)
    forecast.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    forecast.add_argument("--horizon", type=int, default=14)
    forecast.add_argument("--output", type=Path, default=Path("reports/forecast.csv"))

    serve = subparsers.add_parser("serve", help="Run the FastAPI prediction service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def run_train(args: argparse.Namespace) -> None:
    history = load_census(args.data)
    result = train_and_evaluate(
        history,
        test_days=args.test_days,
        validation_days=args.validation_days,
        random_state=args.random_state,
        xgb_estimators=args.xgb_estimators,
        neural_epochs=args.neural_epochs,
        neural_patience=args.neural_patience,
    )
    forecast = recursive_forecast(
        history,
        result["pytorch_predictor"],
        horizon=args.horizon,
        residual_quantiles=result["residual_quantiles"],
    )
    paths = save_training_outputs(
        output_dir=args.reports_dir,
        artifact_dir=args.artifacts_dir,
        result=result,
        forecast=forecast,
    )

    tracking_uri = args.tracking_uri or "sqlite:///mlflow.db"
    run_ids = record_mlflow_runs(
        metadata=result["metadata"],
        paths=paths,
        tracking_uri=tracking_uri,
        experiment_name=args.experiment_name,
    )
    result["metadata"]["mlflow"] = {
        "tracking_uri": tracking_uri,
        "experiment_name": args.experiment_name,
        "run_ids": run_ids,
    }
    paths["metrics"].write_text(
        json.dumps(result["metadata"], indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {
        "test_metrics": result["metadata"]["test_metrics"],
        "pytorch_selected_epochs": result["metadata"]["model_settings"]["pytorch_neural_network"][
            "selected_epochs"
        ],
        "trained_through": result["metadata"]["trained_through"],
        "forecast_start": str(forecast["date"].min().date()),
        "forecast_end": str(forecast["date"].max().date()),
        "mlflow_tracking_uri": tracking_uri,
        "mlflow_run_ids": run_ids,
    }
    print(json.dumps(summary, indent=2))


def run_forecast(args: argparse.Namespace) -> None:
    history = load_census(args.data)
    predictor, metadata = load_neural_bundle(args.artifacts_dir)
    forecast = recursive_forecast(
        history,
        predictor,
        horizon=args.horizon,
        residual_quantiles=metadata["residual_quantiles"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(args.output, index=False)
    print(f"Wrote {len(forecast)} forecast rows to {args.output}")


def run_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run(
        "shelter_forecasting.api:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        run_train(args)
    elif args.command == "forecast":
        run_forecast(args)
    else:
        run_serve(args)


if __name__ == "__main__":
    main()
