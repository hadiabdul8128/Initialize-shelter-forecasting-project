"""Command-line interface for training and forecasting."""

import argparse
import json
from pathlib import Path

import joblib

from shelter_forecasting.data import load_census
from shelter_forecasting.model import recursive_forecast, train_and_evaluate
from shelter_forecasting.reporting import save_training_outputs

DEFAULT_DATA = Path("data/raw/DHS_Homeless_Shelter_Census_20260728.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shelter-forecast",
        description="Train and run a daily shelter population neural-network forecast.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train, evaluate, and forecast")
    train.add_argument("--data", type=Path, default=DEFAULT_DATA)
    train.add_argument("--reports-dir", type=Path, default=Path("reports"))
    train.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    train.add_argument("--horizon", type=int, default=14)
    train.add_argument("--test-days", type=int, default=90)
    train.add_argument("--validation-days", type=int, default=90)
    train.add_argument("--max-iter", type=int, default=1_000)
    train.add_argument("--random-state", type=int, default=42)

    forecast = subparsers.add_parser("forecast", help="Forecast with a saved model")
    forecast.add_argument("--data", type=Path, default=DEFAULT_DATA)
    forecast.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/shelter_mlp.joblib"),
    )
    forecast.add_argument("--horizon", type=int, default=14)
    forecast.add_argument("--output", type=Path, default=Path("reports/forecast.csv"))
    return parser


def run_train(args: argparse.Namespace) -> None:
    history = load_census(args.data)
    result = train_and_evaluate(
        history,
        test_days=args.test_days,
        validation_days=args.validation_days,
        random_state=args.random_state,
        max_iter=args.max_iter,
    )
    forecast = recursive_forecast(
        history,
        result["model"],
        horizon=args.horizon,
        residual_quantiles=result["residual_quantiles"],
    )
    save_training_outputs(
        output_dir=args.reports_dir,
        artifact_dir=args.artifacts_dir,
        result=result,
        forecast=forecast,
    )

    summary = {
        "selected_candidate": result["metadata"]["selected_candidate"]["name"],
        "test_metrics": result["metadata"]["test_metrics"],
        "trained_through": result["metadata"]["trained_through"],
        "forecast_start": str(forecast["date"].min().date()),
        "forecast_end": str(forecast["date"].max().date()),
    }
    print(json.dumps(summary, indent=2))


def run_forecast(args: argparse.Namespace) -> None:
    history = load_census(args.data)
    bundle = joblib.load(args.model)
    forecast = recursive_forecast(
        history,
        bundle["model"],
        horizon=args.horizon,
        residual_quantiles=bundle["residual_quantiles"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(args.output, index=False)
    print(f"Wrote {len(forecast)} forecast rows to {args.output}")


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        run_train(args)
    else:
        run_forecast(args)


if __name__ == "__main__":
    main()
