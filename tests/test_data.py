import pandas as pd
import pytest

from shelter_forecasting.data import load_census


def source_frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date of Census": dates.strftime("%m/%d/%Y"),
            "Total Individuals in Shelter": range(100, 100 + len(dates)),
            "Total Single Adults in Shelter": 50,
            "Families with Children in Shelter": 20,
            "Adult Families in Shelter": 10,
        }
    )


def test_load_census_sorts_and_renames(tmp_path):
    frame = source_frame(pd.date_range("2026-01-01", periods=3, freq="D"))
    path = tmp_path / "census.csv"
    frame.iloc[::-1].to_csv(path, index=False)

    loaded = load_census(path)

    assert loaded.columns.tolist() == [
        "date",
        "population",
        "single_adults",
        "families_with_children",
        "adult_families",
    ]
    assert loaded["date"].is_monotonic_increasing
    assert loaded["population"].tolist() == [100, 101, 102]


def test_load_census_rejects_missing_day(tmp_path):
    dates = pd.DatetimeIndex(["2026-01-01", "2026-01-03"])
    path = tmp_path / "census.csv"
    source_frame(dates).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Missing daily observations"):
        load_census(path)


def test_pandera_rejects_negative_population(tmp_path):
    frame = source_frame(pd.date_range("2026-01-01", periods=3, freq="D"))
    frame.loc[1, "Total Individuals in Shelter"] = -1
    path = tmp_path / "census.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="schema validation failed"):
        load_census(path)
