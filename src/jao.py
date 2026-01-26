from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd
import requests
from tqdm import tqdm


# CORE Shadow Prices API endpoint
CORE_SHADOW_PRICES_URL = "https://publicationtool.jao.eu/core/api/data/shadowPrices"

TAKE = 50000                  # pagination size
REQUEST_SLEEP_SECONDS = 0.8   # avoid rate limiting


def day_window_utc(d: date):
    start = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start, end


def iso_utc(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def fetch_day(
    session: requests.Session,
    d: date,
    filter_obj: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:

    start, end = day_window_utc(d)
    rows: List[Dict[str, Any]] = []
    skip = 0

    while True:
        params = {
            "FromUtc": iso_utc(start),
            "ToUtc": iso_utc(end),
            "Skip": skip,
            "Take": TAKE,
        }

        if filter_obj is not None:
            params["Filter"] = json.dumps(filter_obj, separators=(",", ":"))

        r = session.get(CORE_SHADOW_PRICES_URL, params=params, timeout=60)

        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(10)
            r = session.get(CORE_SHADOW_PRICES_URL, params=params, timeout=60)

        r.raise_for_status()
        payload = r.json()
        data = payload.get("data", [])

        if not data:
            break

        rows.extend(data)

        if len(data) < TAKE:
            break

        skip += TAKE
        time.sleep(REQUEST_SLEEP_SECONDS)

    df = pd.DataFrame(rows)

    if "dateTimeUtc" in df.columns:
        df["dateTimeUtc"] = pd.to_datetime(df["dateTimeUtc"], utc=True)

    return df


def download_2025(out_dir: str = "data/jao/shadow_prices/2025"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    start = date(2025, 1, 1)
    end = date(2026, 1, 1)

    # Example optional filter:
    # filter_obj = {"NonRedundant": True}
    filter_obj = None

    with requests.Session() as session:
        d = start
        for _ in tqdm(range((end - start).days), desc="Downloading CORE Shadow Prices 2025"):
            out_file = out_path / f"shadowPrices_{d.isoformat()}.parquet"

            if out_file.exists():
                d += timedelta(days=1)
                continue

            try:
                df = fetch_day(session, d, filter_obj)
                df.to_parquet(out_file, index=False)

            except Exception as e:
                (out_path / f"shadowPrices_{d.isoformat()}.error.txt").write_text(str(e))

            time.sleep(REQUEST_SLEEP_SECONDS)
            d += timedelta(days=1)


if __name__ == "__main__":
    download_2025()