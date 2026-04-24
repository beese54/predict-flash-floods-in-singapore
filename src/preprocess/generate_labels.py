"""
E4 — Generate ordinal label dataset for flash flood prediction.

Label classes (flood_class):
  0 = normal          — no flood or risk expected in prediction horizon
  1 = flood_risk      — drain ≥90% capacity (FLOOD_RISK Telegram event) expected
  2 = flash_flood     — confirmed flash flood (FLASH_FLOOD Telegram / ST article) expected

Two prediction horizons are built:
  flood_class_30min   — event within next 30 minutes  (operational precision)
  flood_class_6h      — event within next 6 hours     (early warning)

For backward compatibility the binary `flood` column (class==2, 6h horizon) is also kept.

Input:
  data/processed/flood_events.parquet      — confirmed floods (class 2)
  data/processed/flood_risk_events.parquet — precursor risk events (class 1), if exists
  data/processed/st_flood_labels.csv       — manual ST date+time annotations
  data/processed/extracted_events.json     — used to map ST source_row_id → URL
  data/processed/singapore_grid.geojson
Output:
  data/processed/labels.parquet
"""
import json
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


def _load_st_time_map(root: Path) -> dict[int, tuple[str, str]]:
    """Return {source_row_id: (date_str, time_str)} from manually-annotated st_flood_labels.csv."""
    labels_path = root / "data" / "processed" / "st_flood_labels.csv"
    extracted_path = root / "data" / "processed" / "extracted_events.json"

    if not labels_path.exists():
        return {}

    labels_df = pd.read_csv(labels_path, dtype=str)
    url_to_time: dict[str, tuple[str, str]] = {}
    for _, row in labels_df.iterrows():
        if str(row.get("is_flood", "")).lower() != "true":
            continue
        date_v = row.get("flood_event_date", "")
        time_v = row.get("flood_event_time", "")
        if date_v and date_v not in ("", "nan", "None"):
            url_to_time[row["article_url"]] = (
                date_v,
                time_v if time_v not in ("", "nan", "None") else "",
            )

    if not extracted_path.exists() or not url_to_time:
        return {}

    with open(extracted_path, encoding="utf-8") as f:
        extracted = json.load(f)

    time_map: dict[int, tuple[str, str]] = {}
    for e in extracted:
        if e.get("source") == "straits_times":
            url = e.get("url", "")
            if url in url_to_time:
                time_map[int(e["source_row_id"])] = url_to_time[url]

    log.info(f"Loaded manual ST times for {len(time_map)} source_row_ids")
    return time_map


def _event_datetime(ev: pd.Series, st_time_map: dict[int, tuple[str, str]]) -> pd.Timestamp:
    """Resolve the best available datetime for an event row."""
    # ST events: use manually-annotated time if available
    if ev.get("source") == "straits_times":
        sid = ev.get("source_row_id")
        if sid is not None and int(sid) in st_time_map:
            date_str, time_str = st_time_map[int(sid)]
            if time_str and str(time_str) not in ("nan", "None", ""):
                return pd.Timestamp(f"{date_str} {time_str}")
            return pd.Timestamp(date_str)

    # Telegram events: precise event_datetime (UTC-aware → SGT → tz-naive)
    if "event_datetime" in ev.index and pd.notna(ev["event_datetime"]):
        ts = pd.Timestamp(ev["event_datetime"])
        if ts.tzinfo is not None:
            ts = ts.tz_convert("Asia/Singapore").tz_localize(None)
        return ts

    return pd.Timestamp(ev["date"])


def generate_labels(root: Path, config: dict) -> None:
    grid_path = root / "data" / "processed" / "singapore_grid.geojson"
    flash_path = root / "data" / "processed" / "flood_events.parquet"
    risk_path  = root / "data" / "processed" / "flood_risk_events.parquet"

    grid = gpd.read_file(grid_path)
    flash_events = pd.read_parquet(flash_path)

    risk_events = pd.DataFrame()
    if risk_path.exists():
        risk_events = pd.read_parquet(risk_path)
        log.info(f"Loaded {len(risk_events)} FLOOD_RISK events (class 1)")
    else:
        log.warning("flood_risk_events.parquet not found — running with class-2 labels only. "
                    "Run geocode_events.py first to generate FLOOD_RISK geocodes.")

    time_window  = timedelta(hours=config["labels"]["time_window_hours"])
    flood_radius = config["labels"]["flood_radius_km"]
    horizons     = config["labels"].get("prediction_horizons", [6.0])

    # Load manually-annotated ST event times
    st_time_map = _load_st_time_map(root)

    # Build grid centroid lookup
    grid["lat_c"] = grid["lat_centroid"]
    grid["lon_c"] = grid["lon_centroid"]

    def nearby_cells(lat: float, lon: float) -> list[str]:
        result = []
        for _, cell in grid.iterrows():
            if _haversine_km(lat, lon, cell["lat_c"], cell["lon_c"]) <= flood_radius:
                result.append(cell["grid_cell_id"])
        return result

    # Collect positive windows: each entry has (grid_cell_id, flood_start, flood_end, flood_class)
    positive_windows: list[dict] = []

    def add_windows(events_df: pd.DataFrame, flood_class: int) -> None:
        for _, ev in events_df.iterrows():
            if pd.isna(ev.get("lat")) or pd.isna(ev.get("lon")):
                continue
            event_dt = _event_datetime(ev, st_time_map)
            cells = nearby_cells(float(ev["lat"]), float(ev["lon"]))
            if not cells and pd.notna(ev.get("grid_cell_id")):
                cells = [ev["grid_cell_id"]]
            for cell_id in cells:
                positive_windows.append({
                    "grid_cell_id": cell_id,
                    "flood_start":  event_dt - time_window,
                    "flood_end":    event_dt + time_window,
                    "flood_class":  flood_class,
                })

    log.info(f"Building class-2 (FLASH_FLOOD) windows from {len(flash_events)} event-location rows ...")
    add_windows(flash_events, flood_class=2)

    if not risk_events.empty:
        log.info(f"Building class-1 (FLOOD_RISK) windows from {len(risk_events)} event-location rows ...")
        add_windows(risk_events, flood_class=1)

    log.info(f"Total positive windows before horizon expansion: {len(positive_windows)}")

    # Build 5-min timestamp grid
    start = pd.Timestamp(config["data"]["start_date"])
    end   = pd.Timestamp(config["data"]["end_date"])
    timestamps = pd.date_range(start=start, end=end, freq="5min")

    # For each horizon, build a dict: (cell_id, ts) → max flood_class
    horizon_results: dict[str, dict[tuple, int]] = {}
    for horizon_h in horizons:
        col = f"flood_class_{int(horizon_h * 60)}min" if horizon_h < 1 else f"flood_class_{int(horizon_h)}h"
        pred_horizon = timedelta(hours=horizon_h)
        class_dict: dict[tuple, int] = {}
        for pw in positive_windows:
            window_start = pw["flood_start"] - pred_horizon
            window_end   = pw["flood_end"]
            in_window = timestamps[(timestamps >= window_start) & (timestamps <= window_end)]
            for ts in in_window:
                key = (pw["grid_cell_id"], ts)
                class_dict[key] = max(class_dict.get(key, 0), pw["flood_class"])
        horizon_results[col] = class_dict
        log.info(f"  {col}: {len(class_dict):,} positive (cell, timestamp) pairs")

    # Determine primary col names
    col_30min = "flood_class_30min"
    col_6h    = "flood_class_6h"
    # Fallback if config has different horizons
    cols_available = list(horizon_results.keys())
    if col_30min not in horizon_results and cols_available:
        col_30min = cols_available[0]
    if col_6h not in horizon_results and len(cols_available) > 1:
        col_6h = cols_available[-1]
    elif col_6h not in horizon_results:
        col_6h = col_30min

    # Merge all horizon keys into a combined positive set
    all_positive_keys: set[tuple] = set()
    for d in horizon_results.values():
        all_positive_keys.update(d.keys())

    log.info(f"Total unique positive (cell, ts) across all horizons: {len(all_positive_keys):,}")

    # Build rows: all positives + sampled negatives
    all_cells = grid["grid_cell_id"].tolist()
    pos_rows = []
    for key, _ in horizon_results[col_6h].items():
        cell_id, ts = key
        row = {"grid_cell_id": cell_id, "timestamp": ts}
        for col, d in horizon_results.items():
            row[col] = d.get(key, 0)
        # Backward-compat binary flood column (class==2 at 6h)
        row["flood"] = 1 if horizon_results[col_6h].get(key, 0) == 2 else 0
        pos_rows.append(row)

    # Also include rows that are positive only in non-6h horizons
    keys_6h = set(horizon_results[col_6h].keys())
    for col, d in horizon_results.items():
        if col == col_6h:
            continue
        for key in d:
            if key not in keys_6h:
                cell_id, ts = key
                row = {"grid_cell_id": cell_id, "timestamp": ts}
                for c, dd in horizon_results.items():
                    row[c] = dd.get(key, 0)
                row["flood"] = 0
                pos_rows.append(row)

    n_neg = min(len(pos_rows) * 50, len(all_cells) * len(timestamps) // 100)
    rng = np.random.default_rng(42)
    neg_rows = []
    while len(neg_rows) < n_neg:
        batch_cells = rng.choice(all_cells, size=min(10_000, n_neg * 2), replace=True)
        batch_ts    = rng.choice(timestamps, size=len(batch_cells), replace=True)
        for cid, ts in zip(batch_cells, batch_ts):
            if (cid, ts) not in all_positive_keys:
                row = {"grid_cell_id": cid, "timestamp": ts, "flood": 0}
                for col in horizon_results:
                    row[col] = 0
                neg_rows.append(row)
            if len(neg_rows) >= n_neg:
                break

    labels_df = pd.DataFrame(pos_rows + neg_rows)
    labels_df["timestamp"] = pd.to_datetime(labels_df["timestamp"])
    labels_df = labels_df.sort_values(["grid_cell_id", "timestamp"]).reset_index(drop=True)

    # Fill any missing horizon columns with 0
    for col in horizon_results:
        if col not in labels_df.columns:
            labels_df[col] = 0

    pos_rate = labels_df["flood"].mean() * 100
    log.info(f"Total rows: {len(labels_df):,} | Binary flood positive rate: {pos_rate:.3f}%")
    for col in horizon_results:
        rate = (labels_df[col] > 0).mean() * 100
        log.info(f"  {col} any-positive rate: {rate:.3f}%")

    out_path = root / "data" / "processed" / "labels.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_parquet(out_path, index=False)
    log.info(f"Saved → {out_path}")


if __name__ == "__main__":
    cfg = get_config()
    generate_labels(project_root(), cfg)
