"""Data loading and validation."""

from pathlib import Path

import pandas as pd

SOURCE_COLUMNS = {
    "Date of Census": "date",
    "Total Individuals in Shelter": "population",
    "Total Single Adults in Shelter": "single_adults",
    "Families with Children in Shelter": "families_with_children",
    "Adult Families in Shelter": "adult_families",
}


def load_census(path: str | Path) -> pd.DataFrame:
    """Load the census CSV and return a validated, chronological data frame."""
    path = Path(path)
    frame = pd.read_csv(path)

    missing_columns = set(SOURCE_COLUMNS).difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    frame = frame[list(SOURCE_COLUMNS)].rename(columns=SOURCE_COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"], format="%m/%d/%Y", errors="raise")

    numeric_columns = [column for column in frame.columns if column != "date"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    frame = frame.sort_values("date").reset_index(drop=True)
    if frame["date"].duplicated().any():
        duplicates = frame.loc[frame["date"].duplicated(), "date"].dt.date.tolist()
        raise ValueError(f"Duplicate census dates: {duplicates}")

    expected_dates = pd.date_range(frame["date"].min(), frame["date"].max(), freq="D")
    missing_dates = expected_dates.difference(frame["date"])
    if not missing_dates.empty:
        preview = ", ".join(str(value.date()) for value in missing_dates[:5])
        raise ValueError(f"Missing daily observations: {preview}")

    if frame[numeric_columns].isna().any().any():
        raise ValueError("Census data contains missing numeric values")
    if (frame[numeric_columns] < 0).any().any():
        raise ValueError("Census counts cannot be negative")

    return frame
