from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


@dataclass(frozen=True)
class ResampleConfig:
    """
    Upsample 1H -> 15min by duplicating values within each hour.
    """
    datetime_col: str = "Time"
    freq: str = "15min"            # target resolution
    method: str = "ffill"          # forward-fill duplicates the last known hourly row
    sort: bool = True


def read_dataset(path: str | Path) -> pd.DataFrame:
    """
    Read CSV into a DataFrame. Adjust kwargs if your file is not comma-separated.
    """
    return pd.read_csv(path)


def parse_and_validate_datetime(df: pd.DataFrame, cfg: ResampleConfig) -> pd.DataFrame:
    """
    Parse datetime column and ensure it is usable for time-based indexing.
    """
    if cfg.datetime_col not in df.columns:
        raise KeyError(f"Missing datetime column: {cfg.datetime_col}")

    out = df.copy()
    out[cfg.datetime_col] = pd.to_datetime(out[cfg.datetime_col], utc=True, errors="raise")

    if cfg.sort:
        out = out.sort_values(cfg.datetime_col)

    return out


def upsample_grouped(
    df: pd.DataFrame,
    cfg: ResampleConfig,
    group_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Upsample to 15min resolution. If group_cols is provided, resampling is done
    independently per group (recommended for your dataset, e.g. per cnecEic/branchEic/...).

    Forward-fill ensures:
      at 01:00, 01:15, 01:30, 01:45 -> same values as at 01:00
    """
    group_cols = list(group_cols) if group_cols else []

    # Columns that remain unchanged by resampling (we'll reconstruct the time column)
    time_col = cfg.datetime_col

    def _resample_one(block: pd.DataFrame) -> pd.DataFrame:
        block = block.sort_values(time_col).set_index(time_col)

        # Resample all columns (numeric + object). ffill duplicates the previous row.
        block_15 = getattr(block.resample(cfg.freq), cfg.method)()

        # Put datetime back as a column
        block_15 = block_15.reset_index()
        return block_15

    if not group_cols:
        return _resample_one(df)

    # Grouped resampling (handles many constraints per hour without mixing them)
    res = (
        df.groupby(group_cols, dropna=False, sort=False, group_keys=False)
        .apply(_resample_one)
        .reset_index(drop=True)
    )
    return res


def write_dataset(df: pd.DataFrame, path: str | Path) -> None:
    df.to_csv(path, index=False)


def main(
    input_csv: str,
    output_csv: str,
) -> None:
    cfg = ResampleConfig(datetime_col="Time", freq="15min", method="ffill")

    df = read_dataset(input_csv)
    df = parse_and_validate_datetime(df, cfg)

    # Recommended grouping for your data to avoid mixing different constraints:
    # choose columns that define a unique "series" over time in your file.
    group_cols = [
        "CNEC Name",
        "RAM (MW)",
        "ptdf_BE",
        "ptdf_FR",
        "ptdf_DE",
        "ptdf_NL"
    ]

    df_15 = upsample_grouped(df, cfg, group_cols=group_cols)

    # Optional: if you don't want to upsample beyond each group's last timestamp,
    # ffill already does that (it won't create times after the last point for the group).

    write_dataset(df_15, output_csv)


if __name__ == "__main__":
    # Example:
    # main("input.csv", "output_15min.csv")
    main("data/clean/jao/shadow_prices/2025/jao_not_downsampled_2025.csv", "data/clean/jao/shadow_prices/2025/jao_clean_2025.csv")
    pass