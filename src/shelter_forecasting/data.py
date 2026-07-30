"""Pandas ingestion followed by Pandera schema and calendar validation."""

from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

SOURCE_COLUMNS = {
    "Date of Census": "date",
    "Total Individuals in Shelter": "population",
    "Total Single Adults in Shelter": "single_adults",
    "Families with Children in Shelter": "families_with_children",
    "Adult Families in Shelter": "adult_families",
}

SOURCE_SCHEMA = pa.DataFrameSchema(
    {
        "Date of Census": pa.Column(pa.DateTime, nullable=False, coerce=True),
        "Total Individuals in Shelter": pa.Column(
            int, checks=pa.Check.ge(0), nullable=False, coerce=True
        ),
        "Total Single Adults in Shelter": pa.Column(
            int, checks=pa.Check.ge(0), nullable=False, coerce=True
        ),
        "Families with Children in Shelter": pa.Column(
            int, checks=pa.Check.ge(0), nullable=False, coerce=True
        ),
        "Adult Families in Shelter": pa.Column(
            int, checks=pa.Check.ge(0), nullable=False, coerce=True
        ),
    },
    strict=False,
    coerce=True,
)

HISTORY_SCHEMA = pa.DataFrameSchema(
    {
        "date": pa.Column(pa.DateTime, nullable=False, coerce=True),
        "population": pa.Column(float, checks=pa.Check.ge(0), nullable=False, coerce=True),
    },
    strict=False,
    coerce=True,
)


def load_census(path: str | Path) -> pd.DataFrame:
    """Read the source CSV with pandas, then validate it with Pandera."""
    raw = pd.read_csv(Path(path))
    try:
        validated = SOURCE_SCHEMA.validate(raw, lazy=True)
    except SchemaErrors as error:
        raise ValueError(_schema_error_message("Census", error)) from error

    frame = validated[list(SOURCE_COLUMNS)].rename(columns=SOURCE_COLUMNS)
    return _validate_calendar(frame)


def validate_history(history: pd.DataFrame) -> pd.DataFrame:
    """Validate canonical date/population history, including API request data."""
    try:
        validated = HISTORY_SCHEMA.validate(history.copy(), lazy=True)
    except SchemaErrors as error:
        raise ValueError(_schema_error_message("History", error)) from error
    return _validate_calendar(validated[["date", "population"]])


def _validate_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame.empty:
        raise ValueError("Census history cannot be empty")
    if frame["date"].duplicated().any():
        duplicates = frame.loc[frame["date"].duplicated(), "date"].dt.date.tolist()
        raise ValueError(f"Duplicate census dates: {duplicates}")

    expected_dates = pd.date_range(frame["date"].min(), frame["date"].max(), freq="D")
    missing_dates = expected_dates.difference(frame["date"])
    if not missing_dates.empty:
        preview = ", ".join(str(value.date()) for value in missing_dates[:5])
        raise ValueError(f"Missing daily observations: {preview}")
    return frame


def _schema_error_message(name: str, error: SchemaErrors) -> str:
    failures = error.failure_cases[["column", "check", "failure_case"]].head(5)
    details = failures.to_dict(orient="records")
    return f"{name} schema validation failed: {details}"
