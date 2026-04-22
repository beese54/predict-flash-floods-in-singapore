"""
Page: Historical verified flood event browser.
"""
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

ROOT = Path(__file__).parent.parent.parent

st.set_page_config(page_title="Event Browser", layout="wide")
st.title("📋 Flood Event Browser")


@st.cache_data
def load_events():
    p = ROOT / "data" / "processed" / "flood_events.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


@st.cache_data
def load_grid():
    p = ROOT / "data" / "processed" / "singapore_grid.geojson"
    if not p.exists():
        return None
    return gpd.read_file(p)


events = load_events()
grid = load_grid()

if events is None:
    st.warning("flood_events.parquet not found. Complete Phase E first.")
    st.stop()

events["date"] = pd.to_datetime(events["date"])

# Sidebar filters
st.sidebar.header("Filters")
sources = ["All"] + sorted(events["source"].unique().tolist())
selected_source = st.sidebar.selectbox("Source", sources)
date_range = st.sidebar.date_input(
    "Date range",
    value=(events["date"].min().date(), events["date"].max().date()),
)

filtered = events.copy()
if selected_source != "All":
    filtered = filtered[filtered["source"] == selected_source]
if len(date_range) == 2:
    filtered = filtered[
        (filtered["date"] >= pd.Timestamp(date_range[0])) &
        (filtered["date"] <= pd.Timestamp(date_range[1]))
    ]

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader(f"{len(filtered)} events")
    display = filtered[["date", "source", "location_str", "grid_cell_id"]].copy()
    display["date"] = display["date"].dt.strftime("%Y-%m-%d")
    st.dataframe(display, use_container_width=True, height=500)

with col2:
    st.subheader("Event locations")
    m = folium.Map(location=[1.3521, 103.8198], zoom_start=11, tiles="CartoDB positron")
    for _, row in filtered.iterrows():
        if pd.notna(row["lat"]) and pd.notna(row["lon"]):
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=6,
                color="#e74c3c" if row["source"] == "pub_telegram" else "#3498db",
                fill=True,
                fill_opacity=0.7,
                tooltip=(
                    f"{row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else row['date']}<br>"
                    f"Source: {row['source']}<br>"
                    f"{row['location_str']}"
                ),
            ).add_to(m)
    st_folium(m, width=750, height=500)
    st.caption("🔴 PUB Telegram   🔵 Straits Times")
