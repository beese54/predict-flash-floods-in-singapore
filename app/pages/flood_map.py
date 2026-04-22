"""
Page: Interactive Singapore flood probability map.
Shows predicted flood probability per 1km grid cell.
"""
import json
import pickle
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

ROOT = Path(__file__).parent.parent.parent

st.set_page_config(page_title="Flood Probability Map", layout="wide")
st.title("🗺️ Singapore Flood Probability Map")


@st.cache_resource
def load_model_and_features():
    model_path = ROOT / "models" / "lgbm_flood_v1.pkl"
    feat_path = ROOT / "models" / "feature_list.json"
    if not model_path.exists() or not feat_path.exists():
        return None, None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(feat_path) as f:
        features = json.load(f)
    return model, features


@st.cache_data
def load_ml_dataset():
    ml_path = ROOT / "data" / "processed" / "ml_dataset.parquet"
    if not ml_path.exists():
        return None
    return pd.read_parquet(ml_path)


@st.cache_data
def load_grid():
    grid_path = ROOT / "data" / "processed" / "singapore_grid.geojson"
    if not grid_path.exists():
        return None
    return gpd.read_file(grid_path)


model, feature_cols = load_model_and_features()
ml_df = load_ml_dataset()
grid = load_grid()

if model is None or ml_df is None or grid is None:
    st.warning(
        "Model, dataset, or grid not found. "
        "Please run the full pipeline (Phases A–G) before using this page."
    )
    st.stop()

ml_df["timestamp"] = pd.to_datetime(ml_df["timestamp"])
available_timestamps = sorted(ml_df["timestamp"].unique())

st.sidebar.header("Controls")
selected_ts = st.sidebar.select_slider(
    "Select timestamp",
    options=available_timestamps,
    value=available_timestamps[-1],
    format_func=lambda t: pd.Timestamp(t).strftime("%Y-%m-%d %H:%M"),
)

snapshot = ml_df[ml_df["timestamp"] == selected_ts].copy()

if snapshot.empty:
    st.warning("No data available for the selected timestamp.")
    st.stop()

X_snap = snapshot[feature_cols].values
snapshot["flood_prob"] = model.predict(X_snap)

# Merge with grid geometry
snap_geo = grid.merge(snapshot[["grid_cell_id", "flood_prob"]], on="grid_cell_id", how="left")
snap_geo["flood_prob"] = snap_geo["flood_prob"].fillna(0.0)

# Build folium map
m = folium.Map(location=[1.3521, 103.8198], zoom_start=11, tiles="CartoDB positron")

def color_scale(prob: float) -> str:
    r = int(255 * min(prob * 3, 1.0))
    g = int(255 * max(1 - prob * 3, 0.0))
    return f"#{r:02x}{g:02x}00"

for _, row in snap_geo.iterrows():
    prob = row["flood_prob"]
    if prob < 0.01:
        continue  # skip near-zero cells for performance
    folium.GeoJson(
        row["geometry"].__geo_interface__,
        style_function=lambda _, p=prob: {
            "fillColor": color_scale(p),
            "color": "none",
            "fillOpacity": min(0.2 + p * 0.6, 0.85),
        },
        tooltip=f"Cell: {row['grid_cell_id']}<br>Prob: {prob:.3f}",
    ).add_to(m)

col1, col2 = st.columns([3, 1])
with col1:
    st.subheader(f"Flood probability at {pd.Timestamp(selected_ts).strftime('%Y-%m-%d %H:%M')}")
    st_folium(m, width=900, height=600)

with col2:
    st.subheader("Top risk cells")
    top_cells = snap_geo.nlargest(10, "flood_prob")[["grid_cell_id", "flood_prob", "lat_centroid", "lon_centroid"]]
    st.dataframe(top_cells.rename(columns={"flood_prob": "Prob", "lat_centroid": "Lat", "lon_centroid": "Lon"}),
                 use_container_width=True)
    st.markdown("**Colour scale:**")
    st.markdown("🟢 Low → 🔴 High")
