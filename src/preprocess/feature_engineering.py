"""
F2 — Feature engineering: IDW rainfall interpolation + rolling windows.
Input:  data/raw/rainfall/*.parquet + data/raw/rainfall/stations.json
        + data/processed/singapore_grid.geojson + data/processed/labels.parquet
Output: data/processed/features_{year}.parquet (per-year cache)
        data/processed/ml_dataset.parquet (merged with labels)

Incremental: already-cached per-year feature files are skipped on re-runs.
Re-run freely as new yearly rainfall parquets become available.
"""
import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.utils import get_config, project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── IDW interpolation helpers ─────────────────────────────────────────────────

def _build_weight_matrix(
    stations: pd.DataFrame,
    grid: gpd.GeoDataFrame,
    search_radius_km: float,
    min_stations: int,
) -> np.ndarray:
    """
    Precompute IDW weight matrix W of shape (n_cells, n_stations).
    W[i, j] = normalised 1/d² weight of station j for cell i.
    Cells with fewer than min_stations nearby stations get uniform island-mean weights.
    """
    s_lats = stations["lat"].values
    s_lons = stations["lon"].values
    n_stations = len(stations)
    n_cells = len(grid)

    W = np.zeros((n_cells, n_stations))

    for i, (_, cell) in enumerate(grid.iterrows()):
        c_lat, c_lon = cell["lat_centroid"], cell["lon_centroid"]
        dlat = np.radians(s_lats - c_lat)
        dlon = np.radians(s_lons - c_lon)
        cos_lat = np.cos(np.radians(c_lat))
        dist_km = 6371 * 2 * np.arcsin(
            np.sqrt(np.sin(dlat / 2) ** 2 + cos_lat * np.cos(np.radians(s_lats)) * np.sin(dlon / 2) ** 2)
        )
        mask = dist_km <= search_radius_km
        if mask.sum() < min_stations:
            # Fallback: uniform weight over all stations (island mean)
            W[i, :] = 1.0 / n_stations
        else:
            d = np.where(dist_km < 0.01, 0.01, dist_km)
            w = np.where(mask, 1.0 / d ** 2, 0.0)
            W[i, :] = w / w.sum()

    return W


def _idw_interpolate(rain_wide: pd.DataFrame, W: np.ndarray) -> pd.DataFrame:
    """
    Vectorised IDW: multiply rain matrix R (T×S) by W.T (S×C) → cell_rain (T×C).
    rain_wide: index=timestamp, columns=station_ids (aligned to W columns order)
    Returns DataFrame: index=timestamp, columns=grid_cell_ids
    """
    R = rain_wide.values  # (T, S)
    cell_rain = R @ W.T   # (T, C)
    return pd.DataFrame(cell_rain, index=rain_wide.index)


# ── Rolling feature computation ───────────────────────────────────────────────

def _rolling_features_chunked(
    cell_rain_wide: pd.DataFrame,  # index=timestamp (5-min), columns=cell_idx (int position)
    grid_cell_ids: list,
    windows_hours: list[float],
    chunk_size: int = 50,
):
    """
    Yield rolling feature DataFrames in chunks of `chunk_size` cells to avoid OOM
    on full-year data (105k timestamps x 1887 cells x 13 float64 cols ~ 15 GB).
    """
    n_cells = cell_rain_wide.shape[1]
    window_steps = {h: max(1, int(h * 60 / 5)) for h in windows_hours}
    steps_48h = int(48 * 60 / 5)
    steps_1h = int(1 * 60 / 5)
    steps_30min = int(0.5 * 60 / 5)

    chunk = []
    for col_idx in range(n_cells):
        series = cell_rain_wide.iloc[:, col_idx]
        cell_id = grid_cell_ids[col_idx]

        feat = pd.DataFrame(index=cell_rain_wide.index)
        feat["grid_cell_id"] = cell_id

        for h in windows_hours:
            label = f"rain_{int(h*60) if h < 1 else int(h)}{'min' if h < 1 else 'hr'}"
            feat[label] = series.rolling(window_steps[h], min_periods=1).sum()

        feat["rain_48hr"] = series.rolling(steps_48h, min_periods=1).sum()
        feat["max_intensity_1hr"] = series.rolling(steps_1h, min_periods=1).max()

        rolling_30 = series.rolling(steps_30min, min_periods=1).sum()
        feat["rain_delta_30min"] = rolling_30.diff(steps_30min).fillna(0)

        is_dry = (series == 0).astype(int)
        dry_streak = is_dry * (is_dry.groupby((is_dry != is_dry.shift()).cumsum()).cumcount() + 1)
        feat["dry_spell_hours"] = (dry_streak * 5 / 60).round(2)

        chunk.append(feat.reset_index().rename(columns={"index": "timestamp"}))

        if len(chunk) >= chunk_size:
            yield pd.concat(chunk, ignore_index=True)
            chunk = []

    if chunk:
        yield pd.concat(chunk, ignore_index=True)


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["timestamp"])
    df = df.copy()
    df["hour"] = dt.dt.hour
    df["month"] = dt.dt.month
    df["day_of_week"] = dt.dt.dayofweek
    df["is_wet_season"] = dt.dt.month.isin([11, 12, 1, 5, 6, 7]).astype(int)
    return df


# ── Per-year processing (incremental cache) ───────────────────────────────────

def _process_year(
    year: int,
    rain_path: Path,
    stations: pd.DataFrame,
    grid: gpd.GeoDataFrame,
    W: np.ndarray,
    grid_cell_ids: list,
    windows_hours: list[float],
    processed_dir: Path,
    force: bool = False,
) -> Path:
    cache_path = processed_dir / f"features_{year}.parquet"
    if cache_path.exists() and not force:
        log.info(f"  {year}: cache hit — skipping (delete features_{year}.parquet to reprocess)")
        return cache_path

    log.info(f"  {year}: loading rainfall ...")
    rain_long = pd.read_parquet(rain_path)
    rain_long["timestamp"] = pd.to_datetime(rain_long["timestamp"])

    # Pivot: (timestamp × station_id), fill missing with 0
    rain_wide = rain_long.pivot_table(
        index="timestamp", columns="station_id", values="rainfall_mm", aggfunc="first"
    )
    # Align columns to station order in W; add missing stations as 0
    all_station_ids = stations["station_id"].tolist()
    rain_wide = rain_wide.reindex(columns=all_station_ids, fill_value=0)
    rain_wide = rain_wide.resample("5min").mean().fillna(0)

    log.info(f"  {year}: IDW interpolation ({len(rain_wide)} timestamps × {len(grid_cell_ids)} cells) ...")
    cell_rain_wide = _idw_interpolate(rain_wide, W)

    log.info(f"  {year}: computing rolling features (chunked) ...")
    writer = None
    total_rows = 0
    for chunk_df in _rolling_features_chunked(cell_rain_wide, grid_cell_ids, windows_hours):
        chunk_df = _add_temporal_features(chunk_df)
        table = pa.Table.from_pandas(chunk_df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(cache_path), table.schema)
        writer.write_table(table)
        total_rows += len(chunk_df)
    if writer:
        writer.close()
    log.info(f"  {year}: saved → {cache_path.name} ({total_rows:,} rows)")
    return cache_path


# ── Main entry point ──────────────────────────────────────────────────────────

def build_dataset(root: Path, config: dict, force_years: list[int] | None = None) -> None:
    """
    Build ml_dataset.parquet incrementally.
    force_years: list of years to reprocess even if cache exists (e.g. [2018]).
    """
    rain_dir     = root / "data" / "raw" / "rainfall"
    stations_path = rain_dir / "stations.json"
    grid_path    = root / "data" / "processed" / "singapore_grid.geojson"
    labels_path  = root / "data" / "processed" / "labels.parquet"
    processed_dir = root / "data" / "processed"

    # ── Load static assets ────────────────────────────────────────────────────
    with open(stations_path) as f:
        stations_list = json.load(f)
    stations = pd.DataFrame(stations_list).dropna(subset=["lat", "lon"])
    stations["station_id"] = stations["station_id"].astype(str)

    grid = gpd.read_file(grid_path)
    grid_cell_ids = grid["grid_cell_id"].tolist()

    cfg_feat = config["features"]
    windows_hours = cfg_feat["rolling_windows_hours"]

    # ── Precompute IDW weight matrix (done once) ───────────────────────────────
    log.info(f"Precomputing IDW weights ({len(grid)} cells × {len(stations)} stations) ...")
    W = _build_weight_matrix(
        stations, grid,
        cfg_feat["idw_search_radius_km"],
        cfg_feat["idw_min_stations"],
    )

    # ── Process each year's rainfall (incremental) ────────────────────────────
    rain_parquets = sorted(rain_dir.glob("20[0-9][0-9].parquet"))
    if not rain_parquets:
        raise FileNotFoundError(f"No yearly rainfall parquets found in {rain_dir}")

    log.info(f"Found {len(rain_parquets)} yearly rainfall files: {[p.stem for p in rain_parquets]}")
    cached_paths = []
    for rp in rain_parquets:
        year = int(rp.stem)
        force = force_years is not None and year in force_years
        cache = _process_year(
            year, rp, stations, grid, W, grid_cell_ids, windows_hours, processed_dir, force=force
        )
        cached_paths.append(cache)

    # ── Merge cached year features with labels (year-by-year to avoid OOM) ──────
    log.info("Merging cached features with labels ...")
    labels = pd.read_parquet(labels_path)
    labels["timestamp"] = pd.to_datetime(labels["timestamp"])
    labels["_year"] = labels["timestamp"].dt.year

    grid_meta = grid[["grid_cell_id", "lat_centroid", "lon_centroid"]]
    merged_chunks = []

    for cache_path in cached_paths:
        year = int(cache_path.stem.split("_")[1])
        label_year = labels[labels["_year"] == year].copy()
        if label_year.empty:
            log.info(f"  {year}: no label rows, skipping")
            continue

        log.info(f"  {year}: matching {len(label_year):,} label rows against features ...")
        label_keys = label_year[["grid_cell_id", "timestamp"]]

        # Read feature file in batches, keep only rows that match a label key
        pf = pq.ParquetFile(cache_path)
        year_feat_chunks = []
        for batch in pf.iter_batches(batch_size=5_000_000):
            feat_chunk = batch.to_pandas()
            feat_chunk["timestamp"] = pd.to_datetime(feat_chunk["timestamp"])
            matched = feat_chunk.merge(label_keys, on=["grid_cell_id", "timestamp"], how="inner")
            if not matched.empty:
                year_feat_chunks.append(matched)

        if year_feat_chunks:
            feat_year = pd.concat(year_feat_chunks, ignore_index=True)
            chunk = label_year.merge(feat_year, on=["grid_cell_id", "timestamp"], how="left")
        else:
            chunk = label_year.copy()
        merged_chunks.append(chunk)
        log.info(f"  {year}: merged {len(chunk):,} rows")

    ml_df = pd.concat(merged_chunks, ignore_index=True).drop(columns=["_year"])

    # Merge grid centroid info
    ml_df = ml_df.merge(grid_meta, on="grid_cell_id", how="left")

    nan_rate = ml_df.isnull().mean()
    high_nan = nan_rate[nan_rate > 0.05]
    if not high_nan.empty:
        log.warning(f"Features with >5% NaN:\n{high_nan}")

    out_path = processed_dir / "ml_dataset.parquet"
    ml_df.to_parquet(out_path, index=False)
    log.info(f"ml_dataset shape: {ml_df.shape}")
    log.info(f"Positive rate: {ml_df['flood'].mean()*100:.3f}%")
    log.info(f"Feature columns: {[c for c in ml_df.columns if c not in ('grid_cell_id','timestamp','flood','lat_centroid','lon_centroid')]}")
    log.info(f"Saved → {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-year", type=int, nargs="*", help="Reprocess these years even if cached")
    args = parser.parse_args()
    cfg = get_config()
    build_dataset(project_root(), cfg, force_years=args.force_year)
