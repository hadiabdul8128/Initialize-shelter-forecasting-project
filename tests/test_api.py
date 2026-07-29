from pathlib import Path

from fastapi.testclient import TestClient

from shelter_forecasting.api import create_app

ROOT = Path(__file__).resolve().parents[1]


def test_health_and_forecast_endpoints():
    app = create_app(
        data_path=ROOT / "data/raw/DHS_Homeless_Shelter_Census_20260728.csv",
        model_dir=ROOT / "artifacts",
    )

    with TestClient(app) as client:
        home = client.get("/")
        health = client.get("/health")
        forecast = client.post("/forecast", json={"horizon": 3})

    assert home.status_code == 200
    assert "Estimate shelter population by date" in home.text
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert forecast.status_code == 200
    assert forecast.json()["model"] == "PyTorch ShelterMLP"
    assert len(forecast.json()["forecasts"]) == 3


def test_date_estimate_returns_observed_or_forecast_value():
    app = create_app(
        data_path=ROOT / "data/raw/DHS_Homeless_Shelter_Census_20260728.csv",
        model_dir=ROOT / "artifacts",
    )

    with TestClient(app) as client:
        observed = client.get(
            "/api/estimate",
            params={"target_date": "2026-07-27"},
        )
        forecast = client.get(
            "/api/estimate",
            params={"target_date": "2026-07-29"},
        )
        too_far = client.get(
            "/api/estimate",
            params={"target_date": "2026-10-26"},
        )

    assert observed.status_code == 200
    assert observed.json()["source"] == "observed"
    assert observed.json()["population"] == 82_561
    assert forecast.status_code == 200
    assert forecast.json()["source"] == "pytorch_forecast"
    assert forecast.json()["population"] == 82_503
    assert too_far.status_code == 422
