FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SHELTER_DATA_PATH=/app/data/raw/DHS_Homeless_Shelter_Census_20260728.csv \
    SHELTER_MODEL_DIR=/app/artifacts

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      "torch>=2.6,<3" \
    && python -m pip install --no-cache-dir .

COPY data ./data
COPY artifacts/pytorch_model.pt artifacts/neural_metadata.json ./artifacts/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "shelter_forecasting.api:app", "--host", "0.0.0.0", "--port", "8000"]
