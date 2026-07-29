"""FastAPI service for the trained PyTorch shelter forecast."""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
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

    @app.get("/health")
    def health(request: Request) -> dict[str, str | int]:
        history = request.app.state.default_history
        metadata = request.app.state.model_metadata
        return {
            "status": "ok",
            "model": metadata["model_type"],
            "trained_through": metadata["trained_through"],
            "history_rows": len(history),
        }

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
