"""
Page: NEA rainfall explorer — time-series per station or grid cell with flood overlays.
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent.parent

st.set_page_config(page_title="Rainfall Explorer", layout="wide")
st.title("🌧️ NEA Rainfall Explorer")


@st.cache_data
def load_stations():
    p = ROOT / "data" / "raw" / "rainfall" / "stations.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_data
def load_rainfall_year(year: int) -> pd.DataFrame | None:
    p = ROOT / "data" / "raw" / "rainfall" / f"{year}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


@st.cache_data
def load_flood_events():
    p = ROOT / "data" / "processed" / "flood_events.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


stations_list = load_stations()
flood_events = load_flood_events()

if stations_list is None:
    st.warning("stations.json not found. Run src/collect/nea_rainfall.py first.")
    st.stop()

stations = {s["station_id"]: s for s in stations_list}
station_options = {f"{s['name']} ({sid})": sid for sid, s in stations.items()}

st.sidebar.header("Controls")
selected_label = st.sidebar.selectbox("Select station", sorted(station_options.keys()))
selected_station = station_options[selected_label]

current_year = 2026
years = list(range(2016, current_year + 1))
selected_year = st.sidebar.selectbox("Year", years, index=len(years) - 1)

rain_df = load_rainfall_year(selected_year)

if rain_df is None:
    st.warning(f"No rainfall data for year {selected_year}. Run nea_rainfall.py --year {selected_year}.")
    st.stop()

rain_df["timestamp"] = pd.to_datetime(rain_df["timestamp"])
station_rain = rain_df[rain_df["station_id"] == selected_station].sort_values("timestamp")

if station_rain.empty:
    st.warning(f"No data for station {selected_station} in {selected_year}.")
    st.stop()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=station_rain["timestamp"],
    y=station_rain["rainfall_mm"],
    mode="lines",
    name="Rainfall (mm/5min)",
    line={"color": "#3498db", "width": 1},
))

# Overlay flood events as vertical lines
if flood_events is not None:
    flood_events["date"] = pd.to_datetime(flood_events["date"])
    year_events = flood_events[flood_events["date"].dt.year == selected_year]
    for _, ev in year_events.iterrows():
        fig.add_vline(
            x=ev["date"].isoformat(),
            line_dash="dash",
            line_color="#e74c3c",
            annotation_text=ev["location_str"][:20],
            annotation_position="top right",
        )

fig.update_layout(
    xaxis_title="Timestamp",
    yaxis_title="Rainfall (mm per 5 min)",
    height=450,
    hovermode="x unified",
    legend={"orientation": "h"},
    margin={"l": 50, "r": 20, "t": 30, "b": 50},
)

station_info = stations.get(selected_station, {})
st.subheader(f"{station_info.get('name', selected_station)} — {selected_year}")
st.caption(f"Lat: {station_info.get('lat', 'N/A')}, Lon: {station_info.get('lon', 'N/A')}")
st.plotly_chart(fig, use_container_width=True)

if flood_events is not None and not year_events.empty:
    st.markdown("**Verified flood events in this year (red dashed lines):**")
    st.dataframe(year_events[["date", "source", "location_str"]].assign(
        date=year_events["date"].dt.strftime("%Y-%m-%d")
    ), use_container_width=True)
