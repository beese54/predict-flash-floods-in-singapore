"""
F1 — Download NEA 5-minute rainfall data from data.gov.sg API.
Output: data/raw/rainfall/stations.json + data/raw/rainfall/<year>.parquet

API notes:
  - Endpoint: https://api-open.data.gov.sg/v2/real-time/api/rainfall
  - Date param format: YYYY-MM-DD (returns all 5-min readings for that date)
  - Results paginated: 25 snapshots per page, follow paginationToken until null
  - Each page: {stations, readings: [{timestamp, data: [{stationId, value}]}]}

Usage:
    python -m src.collect.nea_rainfall              # all years 2016-present
    python -m src.collect.nea_rainfall --year 2024  # single year
    python -m src.collect.nea_rainfall --start-year 2022  # 2022 onward only
"""
import argparse
import json
import logging
import shutil
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from src.utils import get_config, project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

API_URL = "https://api-open.data.gov.sg/v2/real-time/api/rainfall"
REQUEST_DELAY = 1.5   # seconds between page requests (avoids 429)
RETRY_DELAY  = 10.0  # seconds to wait after a 429


def _fetch_day(day: date, session: requests.Session) -> tuple[list[dict], list[dict]]:
    """Fetch all 5-min rainfall snapshots for a single calendar date.
    Returns (station_list, reading_rows).
    """
    station_meta: dict[str, dict] = {}
    reading_rows: list[dict] = []
    pagination_token = None
    page = 0

    while True:
        params: dict = {"date": day.strftime("%Y-%m-%d")}
        if pagination_token:
            params["paginationToken"] = pagination_token

        for attempt in range(4):
            try:
                resp = session.get(API_URL, params=params, timeout=20)
                if resp.status_code == 429:
                    log.warning(f"    429 rate-limit on {day} page {page}, waiting {RETRY_DELAY}s ...")
                    time.sleep(RETRY_DELAY)
                    continue
                if resp.status_code == 404:
                    # Date not available in API (too old or future)
                    return list(station_meta.values()), reading_rows
                resp.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                log.warning(f"    Request error {day} page {page} attempt {attempt}: {e}")
                time.sleep(REQUEST_DELAY * (attempt + 1))
        else:
            log.warning(f"  Skipping {day} page {page} after 4 failed attempts")
            break

        payload = resp.json().get("data", {})

        # Collect station metadata (same every page, but harmless to re-collect)
        for s in payload.get("stations", []):
            station_meta[s["id"]] = {
                "station_id": s["id"],
                "name": s.get("name", ""),
                "lat": s.get("location", {}).get("latitude"),
                "lon": s.get("location", {}).get("longitude"),
            }

        # Parse reading snapshots: each has a timestamp + list of {stationId, value}
        for snapshot in payload.get("readings", []):
            ts = snapshot.get("timestamp", "")
            for obs in snapshot.get("data", []):
                reading_rows.append({
                    "timestamp": ts,
                    "station_id": obs["stationId"],
                    "rainfall_mm": float(obs.get("value", 0.0)),
                })

        pagination_token = payload.get("paginationToken")
        page += 1

        if not pagination_token:
            break

        time.sleep(REQUEST_DELAY)

    return list(station_meta.values()), reading_rows


def download_year(year: int, out_dir: Path, config: dict) -> None:
    parquet_path = out_dir / f"{year}.parquet"
    if parquet_path.exists():
        log.info(f"  {parquet_path.name} already exists — skipping.")
        return

    ckpt_dir = out_dir / "tmp" / str(year)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cfg_start = config["data"]["start_date"]
    cfg_end   = config["data"]["end_date"]

    year_start = date(year, 1, 1)
    if year == int(cfg_start[:4]):
        year_start = date.fromisoformat(cfg_start)
    year_end = date(year, 12, 31)
    if year == int(cfg_end[:4]):
        year_end = date.fromisoformat(cfg_end)
    year_end = min(year_end, date.today())

    log.info(f"Downloading year {year}: {year_start} → {year_end} ...")

    all_stations: dict[str, dict] = {}
    stations_path = out_dir / "stations.json"

    if stations_path.exists():
        with open(stations_path) as f:
            for s in json.load(f):
                all_stations[s["station_id"]] = s

    current = year_start
    total_days = (year_end - year_start).days + 1
    done = 0

    with requests.Session() as session:
        while current <= year_end:
            day_ckpt = ckpt_dir / f"{current}.parquet"
            if not day_ckpt.exists():
                stations, rows = _fetch_day(current, session)
                for s in stations:
                    all_stations[s["station_id"]] = s
                pd.DataFrame(rows).to_parquet(day_ckpt, index=False)
                time.sleep(REQUEST_DELAY)
            done += 1
            if done % 30 == 0 or done == total_days:
                log.info(f"  {year}: {done}/{total_days} days done")
            current += timedelta(days=1)

    # Merge daily checkpoints into final parquet
    day_dfs = [pd.read_parquet(f) for f in sorted(ckpt_dir.glob("*.parquet"))]
    non_empty = [df for df in day_dfs if not df.empty]
    if non_empty:
        df = pd.concat(non_empty, ignore_index=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Singapore").dt.tz_localize(None)
        df = df.sort_values(["timestamp", "station_id"]).reset_index(drop=True)
        df.to_parquet(parquet_path, index=False)
        log.info(f"  Saved {len(df):,} rows → {parquet_path.name}")
    else:
        log.warning(f"  No data collected for year {year}")

    with open(stations_path, "w") as f:
        json.dump(list(all_stations.values()), f, indent=2)
    log.info(f"  Station metadata: {len(all_stations)} stations → stations.json")

    shutil.rmtree(ckpt_dir)
    log.info(f"  Checkpoints cleaned up for {year}")


def main(target_year: int | None = None, start_year: int | None = None) -> None:
    config  = get_config()
    out_dir = project_root() / "data" / "raw" / "rainfall"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_start_year = int(config["data"]["start_date"][:4])
    data_end_year   = int(config["data"]["end_date"][:4])

    if target_year:
        years = [target_year]
    elif start_year:
        years = list(range(start_year, data_end_year + 1))
    else:
        years = list(range(data_start_year, data_end_year + 1))

    for yr in years:
        download_year(yr, out_dir, config)

    log.info("NEA rainfall download complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year",       type=int, default=None, help="Download a single year")
    parser.add_argument("--start-year", type=int, default=None, help="Download from this year onward")
    args = parser.parse_args()
    main(args.year, args.start_year)
