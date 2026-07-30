"""FastAPI service for the trained PyTorch shelter forecast."""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from shelter_forecasting.data import load_census, validate_history
from shelter_forecasting.forecasting import recursive_forecast
from shelter_forecasting.neural import NeuralPredictor, load_neural_bundle

DEFAULT_DATA_PATH = Path(
    os.getenv(
        "SHELTER_DATA_PATH",
        "data/raw/DHS_Homeless_Shelter_Census_20260728.csv",
    )
)
DEFAULT_MODEL_DIR = Path(os.getenv("SHELTER_MODEL_DIR", "artifacts"))
STATIC_DIR = Path(__file__).resolve().parent / "static"


class Observation(BaseModel):
    date: date
    population: float = Field(ge=0)


class ForecastRequest(BaseModel):
    horizon: int = Field(default=14, ge=1, le=90)
    observations: list[Observation] | None = None


class ForecastPoint(BaseModel):
    date: date
    forecast_population: int
    lower_95_approx: int
    upper_95_approx: int
    horizon_day: int


class ForecastResponse(BaseModel):
    model: str
    trained_through: date
    history_through: date
    forecasts: list[ForecastPoint]


class EstimateResponse(BaseModel):
    requested_date: date
    population: int
    lower_95_approx: int
    upper_95_approx: int
    source: Literal["observed", "pytorch_forecast"]
    days_after_last_observation: int
    last_observed_date: date
    model: str


def create_app(
    *,
    data_path: Path = DEFAULT_DATA_PATH,
    model_dir: Path = DEFAULT_MODEL_DIR,
) -> FastAPI:
    """Create an app with explicit paths for production and testability."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        predictor, model_metadata = load_neural_bundle(model_dir)
        app.state.predictor = predictor
        app.state.model_metadata = model_metadata
        app.state.default_history = load_census(data_path)
        yield

    app = FastAPI(
        title="Shelter Population Forecast API",
        version="0.2.0",
        description=("Recursive daily forecasts from the trained PyTorch neural network."),
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health(request: Request) -> dict[str, str | int]:
        history = request.app.state.default_history
        metadata = request.app.state.model_metadata
        last_date = history["date"].max()
        return {
            "status": "ok",
            "model": metadata["model_type"],
            "trained_through": metadata["trained_through"],
            "history_rows": len(history),
            "history_start": str(history["date"].min().date()),
            "last_observed_date": str(last_date.date()),
            "maximum_forecast_date": str((last_date + pd.offsets.Day(90)).date()),
        }

    @app.get("/api/estimate", response_model=EstimateResponse)
    def estimate(
        request: Request,
        target_date: Annotated[
            date,
            Query(description="Date to look up or forecast"),
        ],
    ) -> EstimateResponse:
        history = request.app.state.default_history
        metadata = request.app.state.model_metadata
        target_timestamp = pd.Timestamp(target_date)
        first_observed = pd.Timestamp(history["date"].min())
        last_observed = pd.Timestamp(history["date"].max())

        if target_timestamp < first_observed:
            raise HTTPException(
                status_code=422,
                detail=(f"Choose a date on or after {first_observed.date()}."),
            )

        if target_timestamp <= last_observed:
            population = int(history.loc[history["date"] == target_timestamp, "population"].iloc[0])
            return EstimateResponse(
                requested_date=target_date,
                population=population,
                lower_95_approx=population,
                upper_95_approx=population,
                source="observed",
                days_after_last_observation=0,
                last_observed_date=last_observed.date(),
                model=metadata["model_type"],
            )

        horizon = int((target_timestamp - last_observed).days)
        if horizon > 90:
            maximum = (last_observed + pd.offsets.Day(90)).date()
            raise HTTPException(
                status_code=422,
                detail=f"Choose a date on or before {maximum}.",
            )

        predictor: NeuralPredictor = request.app.state.predictor
        forecast = recursive_forecast(
            history,
            predictor,
            horizon=horizon,
            residual_quantiles=metadata["residual_quantiles"],
        ).iloc[-1]
        return EstimateResponse(
            requested_date=target_date,
            population=int(forecast["forecast_population"]),
            lower_95_approx=int(forecast["lower_95_approx"]),
            upper_95_approx=int(forecast["upper_95_approx"]),
            source="pytorch_forecast",
            days_after_last_observation=horizon,
            last_observed_date=last_observed.date(),
            model=metadata["model_type"],
        )

    @app.post("/forecast", response_model=ForecastResponse)
    def forecast(payload: ForecastRequest, request: Request) -> ForecastResponse:
        try:
            history = _request_history(
                payload,
                default=request.app.state.default_history,
            )
            predictor: NeuralPredictor = request.app.state.predictor
            metadata = request.app.state.model_metadata
            result = recursive_forecast(
                history,
                predictor,
                horizon=payload.horizon,
                residual_quantiles=metadata["residual_quantiles"],
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        points = [
            ForecastPoint(
                date=row.date.date(),
                forecast_population=int(row.forecast_population),
                lower_95_approx=int(row.lower_95_approx),
                upper_95_approx=int(row.upper_95_approx),
                horizon_day=int(row.horizon_day),
            )
            for row in result.itertuples(index=False)
        ]
        return ForecastResponse(
            model=metadata["model_type"],
            trained_through=date.fromisoformat(metadata["trained_through"]),
            history_through=history["date"].max().date(),
            forecasts=points,
        )

    return app


def _request_history(
    payload: ForecastRequest,
    *,
    default: pd.DataFrame,
) -> pd.DataFrame:
    if payload.observations is None:
        return default
    if len(payload.observations) < 56:
        raise ValueError("At least 56 consecutive daily observations are required")
    frame = pd.DataFrame(
        [
            {
                "date": pd.Timestamp(observation.date),
                "population": observation.population,
            }
            for observation in payload.observations
        ]
    )
    return validate_history(frame)


app = create_app()
