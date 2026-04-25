"""
Page: Interactive Singapore flood probability map.
Shows predicted flood class per 1km grid cell for historical timestamps.
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

# (model_file, target_column, is_multiclass)
MODEL_OPTIONS = {
    "30-min model (v2)": ("lgbm_30min_v2.pkl", "flood_class_30min", True),
    "6-hour model (v2)": ("lgbm_6h_v2.pkl",    "flood_class_6h",   True),
    "Legacy binary":     ("lgbm_flood_v1.pkl",  "flood",            False),
}
CLASS_COLORS = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
CLASS_NAMES  = {0: "Normal",  1: "Flood Risk", 2: "Flash Flood"}


@st.cache_resource
def load_model(model_file: str):
    mp = ROOT / "models" / model_file
    fp = ROOT / "models" / "feature_list.json"
    if not mp.exists() or not fp.exists():
        return None, None
    with open(mp, "rb") as f:
        model = pickle.load(f)
    with open(fp) as f:
        features = json.load(f)
    return model, features


@st.cache_data
def load_thresholds() -> dict:
    p = ROOT / "models" / "thresholds.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


@st.cache_data
def load_ml_dataset():
    p = ROOT / "data" / "processed" / "ml_dataset.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


@st.cache_data
def load_grid():
    p = ROOT / "data" / "processed" / "singapore_grid.geojson"
    if not p.exists():
        return None
    return gpd.read_file(p)


ml_df      = load_ml_dataset()
grid       = load_grid()
all_thresholds = load_thresholds()

if ml_df is None or grid is None:
    st.warning("Dataset or grid not found. Run the full pipeline first.")
    st.stop()

ml_df["timestamp"] = pd.to_datetime(ml_df["timestamp"])
available_timestamps = sorted(ml_df["timestamp"].unique())

# ── Sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.header("Controls")
selected_model_label = st.sidebar.radio("Model", list(MODEL_OPTIONS.keys()), index=0)
model_file, target_col, is_multiclass = MODEL_OPTIONS[selected_model_label]
model, feature_cols = load_model(model_file)

if model is None:
    st.warning(f"`{model_file}` not found. Run the training pipeline first.")
    st.stop()

if target_col not in ml_df.columns:
    st.warning(
        f"Column `{target_col}` missing from ml_dataset.parquet. "
        "Re-run generate_labels + feature_engineering."
    )
    st.stop()

selected_ts = st.sidebar.select_slider(
    "Select timestamp",
    options=available_timestamps,
    value=available_timestamps[-1],
    format_func=lambda t: pd.Timestamp(t).strftime("%Y-%m-%d %H:%M"),
)

# ── Prediction ────────────────────────────────────────────────────────────────
snapshot = ml_df[ml_df["timestamp"] == selected_ts].copy()

if snapshot.empty:
    st.warning("No data for the selected timestamp.")
    st.stop()

X_snap = snapshot[feature_cols].values

if is_multiclass:
    probs  = model.predict(X_snap)           # (n, 3)
    thresh = all_thresholds.get(model_file.replace(".pkl", ""))
    if thresh:
        y_pred = np.zeros(len(probs), dtype=int)
        y_pred[probs[:, 1] >= thresh["flood_risk"]]   = 1
        y_pred[probs[:, 2] >= thresh["flash_flood"]]  = 2
        snapshot["pred_class"] = y_pred
    else:
        snapshot["pred_class"] = np.argmax(probs, axis=1)
    snapshot["p0"] = probs[:, 0]
    snapshot["p1"] = probs[:, 1]
    snapshot["p2"] = probs[:, 2]
    snapshot["risk_score"] = probs[:, 1] + probs[:, 2]
else:
    raw = model.predict(X_snap)             # (n,)
    snapshot["pred_class"] = (raw >= 0.5).astype(int)
    snapshot["p0"] = 1 - raw
    snapshot["p1"] = np.zeros(len(raw))
    snapshot["p2"] = raw
    snapshot["risk_score"] = raw

# ── Merge with grid geometry ──────────────────────────────────────────────────
snap_geo = grid.merge(
    snapshot[["grid_cell_id", "pred_class", "p0", "p1", "p2", "risk_score"]],
    on="grid_cell_id", how="left",
)
snap_geo["pred_class"] = snap_geo["pred_class"].fillna(0).astype(int)
snap_geo["risk_score"] = snap_geo["risk_score"].fillna(0.0)
snap_geo[["p0", "p1", "p2"]] = snap_geo[["p0", "p1", "p2"]].fillna(0.0)

# ── Build folium map ──────────────────────────────────────────────────────────
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

# ── Layout ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    st.subheader(
        f"Flood risk — {pd.Timestamp(selected_ts).strftime('%Y-%m-%d %H:%M')} "
        f"({selected_model_label})"
    )
    st_folium(m, width=900, height=600)

with col2:
    st.subheader("Top risk cells")
    top = (
        snap_geo.nlargest(10, "risk_score")
        [["grid_cell_id", "pred_class", "p0", "p1", "p2", "lat_centroid", "lon_centroid"]]
        .copy()
    )
    top["pred_class"] = top["pred_class"].map(CLASS_NAMES)
    st.dataframe(
        top.rename(columns={
            "pred_class": "Class", "p0": "P(Normal)", "p1": "P(Risk)", "p2": "P(Flood)",
            "lat_centroid": "Lat", "lon_centroid": "Lon",
        }),
        use_container_width=True,
    )
    st.markdown("**Colour scale:**")
    st.markdown("🟢 Normal &nbsp;&nbsp; 🟡 Flood Risk &nbsp;&nbsp; 🔴 Flash Flood")

# ── Historical Flood Frequency Map ───────────────────────────────────────────
st.markdown("---")
st.subheader("📍 Historical Flood Frequency (2016–2026)")
st.caption(
    "Each 1 km² cell coloured by how many times it was labelled as a flood event "
    "across the full dataset. Darker = more frequent. "
    "Amber = flood risk only (CCTV and drain level sensors triggered a risk warning — no confirmed flooding). "
    "Red = confirmed flash flood events."
)


@st.cache_data
def load_flood_frequency():
    p = ROOT / "data" / "processed" / "ml_dataset.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p, columns=["grid_cell_id", "flood_class_6h"])
    flash = df[df["flood_class_6h"] == 2].groupby("grid_cell_id").size().rename("flash_count")
    risk  = df[df["flood_class_6h"] == 1].groupby("grid_cell_id").size().rename("risk_count")
    return pd.concat([flash, risk], axis=1).fillna(0).astype(int).reset_index()


def _freq_color(flash: int, risk: int, max_flash: int, max_risk: int) -> tuple[str, float]:
    """Return (hex_color, opacity) for a frequency cell."""
    if flash > 0:
        t = flash / max_flash                           # 0 → 1
        # #FECACA (254,202,202) → #EF4444 (239,68,68) → #7F1D1D (127,29,29)
        if t < 0.5:
            s = t * 2
            r = int(254 + (239 - 254) * s)
            g = int(202 + (68  - 202) * s)
            b = int(202 + (68  - 202) * s)
        else:
            s = (t - 0.5) * 2
            r = int(239 + (127 - 239) * s)
            g = int(68  + (29  - 68 ) * s)
            b = int(68  + (29  - 68 ) * s)
        return f"#{r:02x}{g:02x}{b:02x}", 0.45 + 0.50 * t
    else:
        t = risk / max(max_risk, 1)                     # amber gradient
        # #FEF3C7 (254,243,199) → #B45309 (180,83,9)
        r = int(254 + (180 - 254) * t)
        g = int(243 + (83  - 243) * t)
        b = int(199 + (9   - 199) * t)
        return f"#{r:02x}{g:02x}{b:02x}", 0.35 + 0.45 * t


freq_df = load_flood_frequency()

if freq_df is not None and not freq_df.empty:
    freq_geo    = grid.merge(freq_df, on="grid_cell_id", how="left")
    freq_geo[["flash_count", "risk_count"]] = (
        freq_geo[["flash_count", "risk_count"]].fillna(0).astype(int)
    )
    max_flash = int(freq_geo["flash_count"].max())
    max_risk  = int(freq_geo["risk_count"].max())

    m_freq = folium.Map(location=[1.3521, 103.8198], zoom_start=11, tiles="CartoDB positron")

    for _, row in freq_geo.iterrows():
        flash = int(row["flash_count"])
        risk  = int(row["risk_count"])
        if flash == 0 and risk == 0:
            continue
        color, opacity = _freq_color(flash, risk, max_flash, max_risk)
        tooltip = (
            f"<b>{row['grid_cell_id']}</b><br>"
            f"Flash Flood events: {flash}<br>"
            f"Flood Risk events: {risk}"
        )
        folium.GeoJson(
            row["geometry"].__geo_interface__,
            style_function=lambda _, c=color, o=opacity: {
                "fillColor": c, "color": "none", "fillOpacity": o,
            },
            tooltip=folium.Tooltip(tooltip),
        ).add_to(m_freq)

    st_folium(m_freq, width=None, height=620, key="freq_map", returned_objects=[])

    # ── Colour scale legend ───────────────────────────────────────────────────
    st.markdown(
        f"""
<div style="
    background:#1e1e1e; border-radius:10px; padding:16px 24px;
    max-width:820px; margin-top:4px; font-family:sans-serif;">

  <div style="margin-bottom:10px">
    <span style="color:#aaa; font-size:12px; letter-spacing:0.06em; text-transform:uppercase">
      Flash Flood — confirmed events
    </span>
    <div style="display:flex; align-items:center; margin-top:4px; gap:10px">
      <span style="color:#ccc; font-size:12px; width:60px">Rare</span>
      <div style="flex:1; height:20px; border-radius:4px;
           background:linear-gradient(to right,#FECACA,#EF4444,#7F1D1D)"></div>
      <span style="color:#ccc; font-size:12px; width:80px; text-align:right">
        Frequent<br><span style="font-size:10px; color:#888">(max {max_flash} events)</span>
      </span>
    </div>
  </div>

  <div>
    <span style="color:#aaa; font-size:12px; letter-spacing:0.06em; text-transform:uppercase">
      Flood Risk — CCTV and drain level sensors triggered a risk warning (no confirmed flood)
    </span>
    <div style="display:flex; align-items:center; margin-top:4px; gap:10px">
      <span style="color:#ccc; font-size:12px; width:60px">Rare</span>
      <div style="flex:1; height:20px; border-radius:4px;
           background:linear-gradient(to right,#FEF3C7,#F59E0B,#B45309)"></div>
      <span style="color:#ccc; font-size:12px; width:80px; text-align:right">
        Frequent<br><span style="font-size:10px; color:#888">(max {max_risk} events)</span>
      </span>
    </div>
  </div>

</div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info("Flood frequency data not available. Run the full pipeline first.")
