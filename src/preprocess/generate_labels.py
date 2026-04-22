"""
E4 — Generate (grid_cell_id, timestamp, flood) label dataset.
Prediction target: did a flood occur within the NEXT 6 hours?
Input:  data/processed/flood_events.parquet + data/processed/singapore_grid.geojson
Output: data/processed/labels.parquet
"""
import logging
from datetime import timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from src.utils import get_config, project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def generate_labels(root: Path, config: dict) -> None:
    events_path = root / "data" / "processed" / "flood_events.parquet"
    grid_path = root / "data" / "processed" / "singapore_grid.geojson"

    events = pd.read_parquet(events_path)
    grid = gpd.read_file(grid_path)

    time_window = timedelta(hours=config["labels"]["time_window_hours"])
    pred_horizon = timedelta(hours=config["labels"]["prediction_horizon_hours"])
    flood_radius_km = config["labels"]["flood_radius_km"]

    # Separate FLOOD_RISK events (precursor — not positive labels) from actual floods
    if "message_type" in events.columns:
        flood_risk_events = events[events["message_type"] == "FLOOD_RISK"].copy()
        label_events = events[events["message_type"] != "FLOOD_RISK"].copy()
        if not flood_risk_events.empty:
            risk_out = root / "data" / "processed" / "flood_risk_events.parquet"
            flood_risk_events.to_parquet(risk_out, index=False)
            log.info(f"  {len(flood_risk_events)} FLOOD_RISK (precursor) events saved → {risk_out}")
    else:
        label_events = events

    # Build set of (grid_cell_id, event_datetime) positive labels
    # For each event location: mark the cell + all neighbours within flood_radius_km
    log.info(f"Building positive labels from {len(label_events)} event-location pairs ...")

    # Grid centroids for radius lookup
    grid["lat_c"] = grid["lat_centroid"]
    grid["lon_c"] = grid["lon_centroid"]

    positive_windows: list[dict] = []
    for _, ev in label_events.iterrows():
        # Use precise event_datetime when available (Telegram FLASH_FLOOD); else date at midnight
        if "event_datetime" in ev and pd.notna(ev["event_datetime"]):
            event_dt = pd.Timestamp(ev["event_datetime"])
            # Telegram datetimes are UTC-aware; convert to SGT then strip tz to match pipeline
            if event_dt.tzinfo is not None:
                event_dt = event_dt.tz_convert("Asia/Singapore").tz_localize(None)
        else:
            event_dt = pd.Timestamp(ev["date"])
        ev_lat, ev_lon = ev["lat"], ev["lon"]

        # Find all grid cells within flood_radius_km
        nearby_cells = []
        for _, cell in grid.iterrows():
            dist = _haversine_km(ev_lat, ev_lon, cell["lat_c"], cell["lon_c"])
            if dist <= flood_radius_km:
                nearby_cells.append(cell["grid_cell_id"])

        if not nearby_cells:
            # Fallback: use the directly assigned cell
            if pd.notna(ev.get("grid_cell_id")):
                nearby_cells = [ev["grid_cell_id"]]

        for cell_id in nearby_cells:
            positive_windows.append({
                "grid_cell_id": cell_id,
                "flood_start": event_dt - time_window,
                "flood_end": event_dt + time_window,
            })

    log.info(f"  {len(positive_windows)} positive cell-windows created")

    # Generate 5-min timestamp skeleton over the effective date range
    start = pd.Timestamp(config["data"]["start_date"])
    end = pd.Timestamp(config["data"]["end_date"])
    timestamps = pd.date_range(start=start, end=end, freq="5min")

    # For each positive window: mark all timestamps in [flood_start - pred_horizon, flood_end] as flood=1
    # (i.e., any timestamp T where flood occurs within T to T+pred_horizon)
    positive_set: set[tuple[str, pd.Timestamp]] = set()
    for pw in positive_windows:
        # T is labelled flood=1 if T+pred_horizon >= flood_start and T <= flood_end
        window_start = pw["flood_start"] - pred_horizon
        window_end = pw["flood_end"]
        ts_in_window = timestamps[(timestamps >= window_start) & (timestamps <= window_end)]
        for ts in ts_in_window:
            positive_set.add((pw["grid_cell_id"], ts))

    log.info(f"  {len(positive_set):,} (cell, timestamp) positive pairs")
    log.info("Building full label DataFrame (only storing positive rows + sample of negatives) ...")

    # To avoid an astronomically large dataset, we store:
    # 1. All positive (cell, timestamp) rows
    # 2. A random sample of negative rows (50× oversampling of positives)
    all_cells = grid["grid_cell_id"].tolist()

    pos_rows = [{"grid_cell_id": cid, "timestamp": ts, "flood": 1} for cid, ts in positive_set]

    # Sample negatives: random (cell, timestamp) combinations not in positive_set
    n_neg_sample = min(len(pos_rows) * 50, len(all_cells) * len(timestamps) // 100)
    rng = np.random.default_rng(42)
    neg_rows = []
    while len(neg_rows) < n_neg_sample:
        batch_cells = rng.choice(all_cells, size=min(10000, n_neg_sample * 2), replace=True)
        batch_ts = rng.choice(timestamps, size=len(batch_cells), replace=True)
        for cid, ts in zip(batch_cells, batch_ts):
            if (cid, ts) not in positive_set:
                neg_rows.append({"grid_cell_id": cid, "timestamp": ts, "flood": 0})
            if len(neg_rows) >= n_neg_sample:
                break

    labels_df = pd.DataFrame(pos_rows + neg_rows)
    labels_df["timestamp"] = pd.to_datetime(labels_df["timestamp"])
    labels_df = labels_df.sort_values(["grid_cell_id", "timestamp"]).reset_index(drop=True)

    pos_rate = labels_df["flood"].mean() * 100
    log.info(f"  Total rows: {len(labels_df):,} | Positive rate: {pos_rate:.3f}%")

    out_path = root / "data" / "processed" / "labels.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_parquet(out_path, index=False)
    log.info(f"Saved → {out_path}")


if __name__ == "__main__":
    cfg = get_config()
    generate_labels(project_root(), cfg)
