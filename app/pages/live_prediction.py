"""
Page: Live flash flood prediction using current NEA rainfall.
Fetches the last 48h of NEA 5-minute readings, computes rolling features,
and runs both v2 models to show current risk across Singapore.
"""
import json
import pickle
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess.feature_engineering import (
    _add_temporal_features,
    _build_weight_matrix,
    _idw_interpolate,
)

st.set_page_config(page_title="Live Prediction", layout="wide")
st.title("⚡ Live Flash Flood Prediction")

CLASS_COLORS = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
CLASS_NAMES  = {0: "Normal",  1: "Flood Risk", 2: "Flash Flood"}
NEA_API      = "https://api-open.data.gov.sg/v2/real-time/api/rainfall"
WINDOWS_HOURS = [0.5, 1, 3, 6, 12, 24]


# ── Static asset loaders (cached for the lifetime of the Streamlit session) ───

@st.cache_resource
def load_models_and_features():
    feat_path = ROOT / "models" / "feature_list.json"
    if not feat_path.exists():
        return None, None, None
    with open(feat_path) as f:
        feature_cols = json.load(f)
    models = {}
    for fname, label in [("lgbm_30min_v2.pkl", "30-min"), ("lgbm_6h_v2.pkl", "6-hour")]:
        mp = ROOT / "models" / fname
        if mp.exists():
            with open(mp, "rb") as f:
                models[label] = pickle.load(f)
    thresh_path = ROOT / "models" / "thresholds.json"
    thresholds  = {}
    if thresh_path.exists():
        with open(thresh_path) as f:
            raw = json.load(f)
        thresholds = {
            "30-min": raw.get("lgbm_30min_v2"),
            "6-hour": raw.get("lgbm_6h_v2"),
        }
    return models, feature_cols, thresholds


@st.cache_resource
def load_grid_and_weights():
    grid_path     = ROOT / "data" / "processed" / "singapore_grid.geojson"
    stations_path = ROOT / "data" / "raw" / "rainfall" / "stations.json"
    if not grid_path.exists() or not stations_path.exists():
        return None, None, None, None
    grid = gpd.read_file(grid_path)
    with open(stations_path) as f:
        stations_list = json.load(f)
    stations = (
        pd.DataFrame(stations_list)
        .dropna(subset=["lat", "lon"])
        .reset_index(drop=True)
    )
    stations["station_id"] = stations["station_id"].astype(str)
    W = _build_weight_matrix(stations, grid, search_radius_km=10, min_stations=2)
    return grid, stations, W, stations["station_id"].tolist()


# ── NEA API fetch ─────────────────────────────────────────────────────────────

def _fetch_day(day: date, session: requests.Session) -> tuple[dict, list]:
    station_meta: dict = {}
    reading_rows: list = []
    pagination_token   = None

    while True:
        params: dict = {"date": day.strftime("%Y-%m-%d")}
        if pagination_token:
            params["paginationToken"] = pagination_token

        resp = session.get(NEA_API, params=params, timeout=20)
        if resp.status_code == 404:
            break
        resp.raise_for_status()

        payload = resp.json().get("data", {})
        for s in payload.get("stations", []):
            station_meta[s["id"]] = {
                "station_id": s["id"],
                "lat": s.get("location", {}).get("latitude"),
                "lon": s.get("location", {}).get("longitude"),
            }
        for snap in payload.get("readings", []):
            ts = snap.get("timestamp", "")
            for obs in snap.get("data", []):
                reading_rows.append({
                    "timestamp":    ts,
                    "station_id":   obs["stationId"],
                    "rainfall_mm":  float(obs.get("value", 0.0)),
                })

        pagination_token = payload.get("paginationToken")
        if not pagination_token:
            break
        time.sleep(0.3)

    return station_meta, reading_rows


def fetch_live_rainfall(days_back: int = 2) -> tuple[pd.DataFrame, pd.Timestamp]:
    today     = date.today()
    all_rows: list = []

    with requests.Session() as session:
        for i in range(days_back - 1, -1, -1):
            d = today - timedelta(days=i)
            _, rows = _fetch_day(d, session)
            all_rows.extend(rows)
            if i > 0:
                time.sleep(0.5)

    if not all_rows:
        raise RuntimeError("No data returned from NEA API.")

    df = pd.DataFrame(all_rows)
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], utc=True)
        .dt.tz_convert("Asia/Singapore")
        .dt.tz_localize(None)
    )
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df, df["timestamp"].max()


# ── Rolling feature computation (vectorised) ──────────────────────────────────

def compute_live_features(
    rain_long: pd.DataFrame,
    station_ids: list,
    W: np.ndarray,
    grid: gpd.GeoDataFrame,
) -> pd.DataFrame:
    # Pivot → (T, S); align columns to W station order
    rain_wide = rain_long.pivot_table(
        index="timestamp", columns="station_id", values="rainfall_mm", aggfunc="first"
    )
    rain_wide = rain_wide.reindex(columns=station_ids, fill_value=0)
    rain_wide = rain_wide.resample("5min").mean().fillna(0)

    # IDW → cell rain matrix (T, C)
    cell_rain = _idw_interpolate(rain_wide, W)  # columns = integer 0..C-1
    latest_ts = cell_rain.index[-1]
    n_cells   = cell_rain.shape[1]

    window_steps = {h: max(1, int(h * 60 / 5)) for h in WINDOWS_HOURS}
    steps_48h   = int(48 * 60 / 5)
    steps_1h    = int(1 * 60 / 5)
    steps_30min = int(0.5 * 60 / 5)

    feat: dict = {
        "grid_cell_id": grid["grid_cell_id"].tolist(),
        "timestamp":    latest_ts,
    }

    # Rolling sums / max — vectorised over all cells at once
    for h in WINDOWS_HOURS:
        label = f"rain_{int(h*60) if h < 1 else int(h)}{'min' if h < 1 else 'hr'}"
        feat[label] = cell_rain.rolling(window_steps[h], min_periods=1).sum().iloc[-1].values

    feat["rain_48hr"]         = cell_rain.rolling(steps_48h, min_periods=1).sum().iloc[-1].values
    feat["max_intensity_1hr"] = cell_rain.rolling(steps_1h,  min_periods=1).max().iloc[-1].values

    rolling_30 = cell_rain.rolling(steps_30min, min_periods=1).sum()
    feat["rain_delta_30min"] = rolling_30.diff(steps_30min).fillna(0).iloc[-1].values

    # Dry spell: trailing consecutive-zero count × 5min / 60
    arr     = cell_rain.values          # (T, C)
    arr_rev = arr[::-1, :]              # reverse time axis
    last_nonzero = np.argmax(arr_rev != 0, axis=0)   # first non-zero from end
    has_rain     = (arr != 0).any(axis=0)
    dry_streaks  = np.where(has_rain, last_nonzero, arr.shape[0])
    feat["dry_spell_hours"] = (dry_streaks * 5 / 60).round(2)

    feat_df = pd.DataFrame(feat)
    feat_df = _add_temporal_features(feat_df)

    grid_meta = grid[["grid_cell_id", "lat_centroid", "lon_centroid"]]
    feat_df   = feat_df.merge(grid_meta, on="grid_cell_id", how="left")
    return feat_df


# ── Map builder ───────────────────────────────────────────────────────────────

def _build_map(snap_geo: gpd.GeoDataFrame) -> folium.Map:
    m = folium.Map(location=[1.3521, 103.8198], zoom_start=11, tiles="CartoDB positron")
    for _, row in snap_geo.iterrows():
        cls   = int(row["pred_class"])
        score = float(row["risk_score"])
        if score < 0.02 and cls == 0:
            continue
        opacity = min(0.15 + score * 0.7, 0.85)
        tooltip = (
            f"Cell: {row['grid_cell_id']}<br>"
            f"Predicted: {CLASS_NAMES[cls]}<br>"
            f"Normal: {row['p0']:.3f} | Flood Risk: {row['p1']:.3f} | Flash Flood: {row['p2']:.3f}"
        )
        folium.GeoJson(
            row["geometry"].__geo_interface__,
            style_function=lambda _, c=CLASS_COLORS[cls], o=opacity: {
                "fillColor": c, "color": "none", "fillOpacity": o,
            },
            tooltip=tooltip,
        ).add_to(m)
    return m


# ── Page layout ───────────────────────────────────────────────────────────────

models, feature_cols, thresholds = load_models_and_features()
grid, stations, W, station_ids  = load_grid_and_weights()

if not models or feature_cols is None:
    st.error("Models or `feature_list.json` not found. Run the training pipeline first.")
    st.stop()

if grid is None:
    st.error("Grid or stations file not found. Run the full pipeline first.")
    st.stop()

st.markdown(
    """
Fetch the latest NEA 5-minute rainfall readings and predict current flood risk across Singapore.
- **30-min model** — Is flooding imminent right now?
- **6-hour model** — Should resources be pre-positioned?
"""
)

if st.button("⚡ Fetch Live NEA Data & Predict", type="primary"):
    with st.spinner("Fetching last 48 hours of NEA rainfall (this may take ~30s)..."):
        try:
            rain_long, latest_ts = fetch_live_rainfall(days_back=2)
        except RuntimeError as e:
            st.error(f"NEA API unavailable: {e}")
            st.stop()

    freshness_mins = (pd.Timestamp.now() - latest_ts).total_seconds() / 60
    if freshness_mins > 30:
        st.warning(
            f"Latest NEA reading is {freshness_mins:.0f} minutes old — data may be delayed."
        )
    else:
        st.success(
            f"Fetched {len(rain_long):,} readings. "
            f"Latest: {latest_ts.strftime('%Y-%m-%d %H:%M')} SGT "
            f"({freshness_mins:.0f} min ago)"
        )

    with st.spinner("Computing rolling features and running models..."):
        feat_df = compute_live_features(rain_long, station_ids, W, grid)
        X = feat_df[feature_cols].values

        snap_geos: dict = {}
        for label, mdl in models.items():
            probs  = mdl.predict(X)   # (n, 3)
            thresh = thresholds.get(label)
            df = feat_df[["grid_cell_id"]].copy()
            if thresh:
                y_pred = np.zeros(len(probs), dtype=int)
                y_pred[probs[:, 1] >= thresh["flood_risk"]]  = 1
                y_pred[probs[:, 2] >= thresh["flash_flood"]] = 2
                df["pred_class"] = y_pred
            else:
                df["pred_class"] = np.argmax(probs, axis=1)
            df["p0"]          = probs[:, 0]
            df["p1"]          = probs[:, 1]
            df["p2"]          = probs[:, 2]
            df["risk_score"]  = probs[:, 1] + probs[:, 2]

            sg = grid.merge(df, on="grid_cell_id", how="left")
            sg["pred_class"] = sg["pred_class"].fillna(0).astype(int)
            sg["risk_score"] = sg["risk_score"].fillna(0.0)
            sg[["p0", "p1", "p2"]] = sg[["p0", "p1", "p2"]].fillna(0.0)
            snap_geos[label] = sg

    # ── Side-by-side maps ─────────────────────────────────────────────────────
    ts_label = latest_ts.strftime("%Y-%m-%d %H:%M") + " SGT"
    col_30, col_6h = st.columns(2)

    for col, label in zip([col_30, col_6h], ["30-min", "6-hour"]):
        if label not in snap_geos:
            col.warning(f"{label} model not available.")
            continue
        with col:
            st.subheader(f"{label} model — {ts_label}")
            m = _build_map(snap_geos[label])
            st_folium(m, width=None, height=500, key=f"map_{label}")

    # ── Top risk cells ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Top 10 Highest-Risk Cells")
    tabs = st.tabs(["30-min model", "6-hour model"])
    for tab, label in zip(tabs, ["30-min", "6-hour"]):
        if label not in snap_geos:
            tab.warning(f"{label} model not available.")
            continue
        with tab:
            sg = snap_geos[label]
            top = (
                sg.nlargest(10, "risk_score")
                [["grid_cell_id", "pred_class", "p0", "p1", "p2",
                  "lat_centroid", "lon_centroid"]]
                .copy()
            )
            top["pred_class"] = top["pred_class"].map(CLASS_NAMES)
            st.dataframe(
                top.rename(columns={
                    "pred_class": "Class", "p0": "P(Normal)",
                    "p1": "P(Risk)", "p2": "P(Flood)",
                    "lat_centroid": "Lat", "lon_centroid": "Lon",
                }),
                use_container_width=True,
            )

    st.markdown("---")
    st.markdown(
        "**Colour legend:** 🟢 Normal &nbsp;&nbsp; 🟡 Flood Risk (CCTV / drain sensors triggered) "
        "&nbsp;&nbsp; 🔴 Flash Flood (confirmed)"
    )

else:
    st.info(
        "Click the button above to fetch live NEA data and generate a current risk map. "
        "Each fetch downloads ~2 days of 5-minute readings (no API key required)."
    )
