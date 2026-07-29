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
        health = client.get("/health")
        forecast = client.post("/forecast", json={"horizon": 3})

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert forecast.status_code == 200
    assert forecast.json()["model"] == "PyTorch ShelterMLP"
    assert len(forecast.json()["forecasts"]) == 3
