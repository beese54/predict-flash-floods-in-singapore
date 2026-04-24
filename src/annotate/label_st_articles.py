"""
Streamlit annotation tool: label ST articles with flood event dates and polygons.
Run: streamlit run src/annotate/label_st_articles.py
Output: data/processed/st_flood_labels.csv  +  data/processed/manual_annotations.json
"""
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from folium.plugins import Draw
from shapely.geometry import shape
from streamlit_folium import st_folium

_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import project_root

ROOT          = project_root()
EXCEL_PATH    = ROOT / "data/raw/pub_flash_flood_straits_times_dataset_2010_onwards.xlsx"
ARTICLES_JSON = ROOT / "data/raw/straits_times/articles.json"
ANNOTATIONS   = ROOT / "data/processed/manual_annotations.json"
LABELS_CSV    = ROOT / "data/processed/st_flood_labels.csv"

FLOOD_CATEGORIES = {
    "Incident report",
    "Incident follow-up",
    "Incident / flood impact",
    "Incident follow-up / enforcement",
}

SG_CENTER = [1.3521, 103.8198]


def _valid_time(s: str) -> bool:
    return bool(re.fullmatch(r"\d{2}:\d{2}", s.strip()))


@st.cache_data
def load_articles() -> list[dict]:
    df = pd.read_excel(EXCEL_PATH)
    df["Article URL"] = df["Article URL"].astype(str).str.strip()

    scraped: dict[str, dict] = {}
    if ARTICLES_JSON.exists():
        with open(ARTICLES_JSON, encoding="utf-8") as f:
            for a in json.load(f):
                scraped[a["url"].strip()] = a

    records = []
    for excel_idx, row in df.iterrows():
        if row.get("Category") not in FLOOD_CATEGORIES:
            continue
        url = str(row["Article URL"]).strip()
        s = scraped.get(url, {})
        records.append({
            "excel_row_id": int(excel_idx),
            "url": url,
            "title": str(row.get("Article Title", "")),
            "published_date": pd.to_datetime(row["Published Date"]).date(),
            "category": str(row.get("Category", "") or ""),
            "pub_notes": str(row.get("PUB / Flood Notes", "") or ""),
            "locations": str(row.get("Locations", "") or ""),
            "text": s.get("text"),
            "scrape_status": s.get("scrape_status", "not_scraped"),
        })
    return records


@st.cache_data
def load_annotations() -> dict[int, list[dict]]:
    if not ANNOTATIONS.exists():
        return {}
    with open(ANNOTATIONS, encoding="utf-8") as f:
        raw = json.load(f)
    grouped: dict[int, list[dict]] = {}
    for ann in raw:
        if ann.get("source") != "straits_times":
            continue
        grouped.setdefault(ann["source_row_id"], []).append(ann)
    return grouped


def load_labels() -> pd.DataFrame:
    if LABELS_CSV.exists():
        return pd.read_csv(LABELS_CSV, dtype=str)
    return pd.DataFrame(columns=["article_url", "is_flood", "flood_event_date", "flood_event_time", "annotated_at"])


def save_label(url: str, is_flood: bool, flood_date: str | None, flood_time: str | None) -> None:
    df = load_labels()
    df = df[df["article_url"] != url]
    df = pd.concat([df, pd.DataFrame([{
        "article_url": url,
        "is_flood": str(is_flood),
        "flood_event_date": flood_date or "",
        "flood_event_time": flood_time or "",
        "annotated_at": datetime.now().isoformat(timespec="seconds"),
    }])], ignore_index=True)
    LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LABELS_CSV, index=False)


def save_new_annotations(
    excel_row_id: int,
    drawn_features: list[dict],
    location_names: dict[int, str],
    named_only: list[str] | None = None,
) -> int:
    """Append new annotations to manual_annotations.json. Returns count saved.

    drawn_features: GeoJSON feature dicts from the map Draw plugin.
    named_only: plain location name strings with no geometry (geocoded later by pipeline).
    """
    existing: list[dict] = []
    if ANNOTATIONS.exists():
        with open(ANNOTATIONS, encoding="utf-8") as f:
            existing = json.load(f)
    existing_ids = {a["ann_id"] for a in existing}

    saved = 0

    # Drawn polygons
    for i, feat in enumerate(drawn_features):
        loc = location_names.get(i, "").strip() or f"Manual polygon {i + 1}"
        ann_id = f"straits_times|{excel_row_id}|{loc}"
        if ann_id in existing_ids:
            continue
        try:
            centroid = shape(feat["geometry"]).centroid
            lat, lon = centroid.y, centroid.x
        except Exception:
            lat, lon = SG_CENTER[0], SG_CENTER[1]
        existing.append({
            "ann_id": ann_id,
            "source": "straits_times",
            "source_row_id": excel_row_id,
            "original_location_str": loc,
            "location_str": loc,
            "event_type": "FLASH_FLOOD",
            "lat": lat,
            "lon": lon,
            "annotation_type": feat["geometry"]["type"].lower(),
            "geojson": {"type": "Feature", "properties": {}, "geometry": feat["geometry"]},
            "is_extra": False,
        })
        saved += 1

    # Named-only locations (no polygon — lat/lon resolved by geocode_events.py)
    for loc in (named_only or []):
        loc = loc.strip()
        if not loc:
            continue
        ann_id = f"straits_times|{excel_row_id}|{loc}"
        if ann_id in existing_ids:
            continue
        existing.append({
            "ann_id": ann_id,
            "source": "straits_times",
            "source_row_id": excel_row_id,
            "original_location_str": loc,
            "location_str": loc,
            "event_type": "FLASH_FLOOD",
            "lat": None,
            "lon": None,
            "annotation_type": "text",
            "geojson": None,
            "is_extra": False,
        })
        saved += 1

    with open(ANNOTATIONS, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    return saved


def build_map(ann_list: list[dict]) -> folium.Map:
    m = folium.Map(location=SG_CENTER, zoom_start=12, tiles="CartoDB positron")

    for ann in ann_list:
        geojson = ann.get("geojson")
        label   = ann.get("location_str", "")
        if geojson:
            folium.GeoJson(
                geojson,
                style_function=lambda _: {"color": "#e74c3c", "weight": 2, "fillOpacity": 0.25},
                tooltip=folium.Tooltip(f"Existing: {label}"),
            ).add_to(m)
        lat, lon = ann.get("lat"), ann.get("lon")
        if lat and lon:
            folium.Marker(
                [lat, lon],
                tooltip=label,
                icon=folium.Icon(color="red", icon="tint", prefix="fa"),
            ).add_to(m)

    Draw(
        export=False,
        draw_options={
            "polygon":      {"shapeOptions": {"color": "#2980b9", "fillOpacity": 0.2}},
            "rectangle":    {"shapeOptions": {"color": "#2980b9", "fillOpacity": 0.2}},
            "polyline":     False,
            "circle":       False,
            "marker":       False,
            "circlemarker": False,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(m)

    if ann_list:
        lats = [a["lat"] for a in ann_list if a.get("lat")]
        lons = [a["lon"] for a in ann_list if a.get("lon")]
        if lats:
            m.fit_bounds([[min(lats) - 0.01, min(lons) - 0.01], [max(lats) + 0.01, max(lons) + 0.01]])

    return m


# ── App ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="ST Article Labeller", layout="wide")
st.title("Straits Times Flash Flood Labeller")

articles   = load_articles()
ann_by_row = load_annotations()
labels_df  = load_labels()
annotated  = set(labels_df["article_url"].tolist())
total      = len(articles)
n_done     = len(annotated)

if "idx" not in st.session_state:
    first_pending = next((i for i, a in enumerate(articles) if a["url"] not in annotated), 0)
    st.session_state.idx = first_pending

idx     = st.session_state.idx
article = articles[idx]
rid     = article["excel_row_id"]

# ── Progress & navigation ─────────────────────────────────────────────────────
st.progress(n_done / total, text=f"Annotated {n_done} / {total} articles")

c1, c2, c3 = st.columns([1, 6, 1])
with c1:
    if st.button("← Prev", disabled=idx == 0):
        st.session_state.idx -= 1
        st.rerun()
with c2:
    new_idx = int(st.number_input(
        "Jump", min_value=1, max_value=total, value=idx + 1, step=1, label_visibility="collapsed"
    )) - 1
    if new_idx != idx:
        st.session_state.idx = new_idx
        st.rerun()
    st.caption(f"Article {idx + 1} / {total}")
with c3:
    if st.button("Next →", disabled=idx == total - 1):
        st.session_state.idx += 1
        st.rerun()

st.divider()

# ── Two-column layout ─────────────────────────────────────────────────────────
left, right = st.columns([3, 2])

with left:
    badge = "✅ Annotated" if article["url"] in annotated else "⏳ Pending"
    st.markdown(f"### {article['title']}  &nbsp; `{badge}`")

    m1, m2, m3 = st.columns(3)
    m1.metric("Published", str(article["published_date"]))
    m2.metric("Category", article["category"] or "—")
    m3.metric("Scrape", article["scrape_status"])

    if article["locations"] and article["locations"] not in ("", "nan"):
        st.markdown(f"**Locations in article:** {article['locations']}")

    st.markdown(f"[Open article in browser ↗]({article['url']})")

    with st.expander("Article text / PUB notes", expanded=True):
        if article["text"]:
            st.text_area("text", value=article["text"], height=220, disabled=True, label_visibility="collapsed")
        else:
            st.info("Paywalled / not scraped — showing PUB notes.")
            st.write(article["pub_notes"] if article["pub_notes"] not in ("", "nan") else "*(no notes)*")

with right:
    existing_anns = ann_by_row.get(rid, [])

    if existing_anns:
        st.markdown(f"**Existing polygons** ({len(existing_anns)})")
        for ann in existing_anns:
            st.caption(f"• {ann['location_str']}  `{ann['annotation_type']}`")
    else:
        st.info("No polygons yet for this article.")

    st.markdown("Draw a polygon or rectangle on the map to add a new flood area:")

    map_data = st_folium(
        build_map(existing_anns),
        width=450,
        height=360,
        returned_objects=["all_drawings"],
        key=f"map_{rid}",
    )

    # Capture drawn features into session state
    # st_folium returns all_drawings as a list of GeoJSON features directly
    drawn_key = f"drawn_{rid}"
    if map_data:
        raw = map_data.get("all_drawings") or []
        if isinstance(raw, dict):
            raw = raw.get("features", [])
        if raw:
            st.session_state[drawn_key] = raw

    drawn = st.session_state.get(drawn_key, [])
    if drawn:
        st.success(f"{len(drawn)} new polygon(s) drawn — name them below, then click Save.")
        for i, _ in enumerate(drawn):
            st.text_input(
                f"Location name for polygon {i + 1}",
                key=f"loc_{rid}_{i}",
                placeholder="e.g. Orchard Road",
            )
        if st.button("Clear drawn polygons", key="clear_drawn"):
            st.session_state[drawn_key] = []
            st.rerun()

    # ── Named locations (no polygon required — geocoded by pipeline) ──────────
    st.markdown("---")
    st.markdown("**Add location by name** — type a name and click Add (no drawing needed):")
    named_key = f"named_{rid}"
    if named_key not in st.session_state:
        st.session_state[named_key] = []

    named_locs: list[str] = st.session_state[named_key]

    # Show existing annotations as replication templates
    if existing_anns:
        st.caption("Quick-replicate an existing location with a new name:")
        for ann in existing_anns:
            if st.button(f"Replicate '{ann['location_str']}'", key=f"rep_{ann['ann_id']}"):
                st.session_state[named_key].append("")   # blank slot; user fills in name
                st.rerun()

    # List queued named locations with editable fields and remove buttons
    for i in range(len(named_locs)):
        nc1, nc2 = st.columns([5, 1])
        with nc1:
            named_locs[i] = st.text_input(
                f"Named location {i + 1}",
                value=named_locs[i],
                key=f"named_edit_{rid}_{i}",
                placeholder="e.g. Bishan, Ang Mo Kio Ave 3",
                label_visibility="collapsed",
            )
        with nc2:
            if st.button("✕", key=f"rm_named_{rid}_{i}"):
                st.session_state[named_key].pop(i)
                st.rerun()

    nl1, nl2 = st.columns([5, 1])
    with nl1:
        new_name = st.text_input("New location name", placeholder="e.g. Bishan", key=f"new_loc_{rid}", label_visibility="collapsed")
    with nl2:
        if st.button("Add", key="add_named_loc"):
            if new_name.strip():
                st.session_state[named_key].append(new_name.strip())
                st.rerun()

st.divider()

# ── Annotation form ──────────────────────────────────────────────────────────
st.subheader("Annotation")

existing_label = labels_df[labels_df["article_url"] == article["url"]]
if not existing_label.empty:
    row         = existing_label.iloc[0]
    ex_is_flood = row["is_flood"] == "True"
    ex_is_no    = row["is_flood"] == "False"
    ex_date_str = row["flood_event_date"] if row["flood_event_date"] not in ("", "nan") else None
    ex_time_str = row["flood_event_time"] if row["flood_event_time"] not in ("", "nan") else ""
    default_radio = 0 if ex_is_flood else (1 if ex_is_no else 2)
else:
    ex_date_str   = None
    ex_time_str   = ""
    default_radio = 2

flood_choice = st.radio(
    "Is this a flash flood event?",
    options=["Yes — flood occurred", "No — not a flood event", "Skip for now"],
    index=default_radio,
    horizontal=True,
)

flood_date_out: str | None = None
flood_time_out: str | None = None

if flood_choice.startswith("Yes"):
    dc1, dc2 = st.columns(2)
    with dc1:
        try:
            default_date = date.fromisoformat(ex_date_str) if ex_date_str else article["published_date"]
        except ValueError:
            default_date = article["published_date"]
        flood_date_out = str(st.date_input("Flood event date", value=default_date))
    with dc2:
        flood_time_raw = st.text_input(
            "Flood time — optional (HH:MM, 24h)",
            value=ex_time_str,
            placeholder="e.g. 19:00",
        )
        if flood_time_raw.strip():
            if _valid_time(flood_time_raw):
                flood_time_out = flood_time_raw.strip()
            else:
                st.warning("Use HH:MM format, e.g. 07:30")

b1, b2 = st.columns([3, 1])
with b1:
    if st.button("Save & Next →", type="primary", disabled=flood_choice.startswith("Skip")):
        # Save flood label
        if flood_choice.startswith("Yes"):
            save_label(article["url"], True, flood_date_out, flood_time_out)
        else:
            save_label(article["url"], False, None, None)

        # Save drawn polygons
        drawn = st.session_state.get(drawn_key, [])
        if drawn:
            loc_names = {i: st.session_state.get(f"loc_{rid}_{i}", "") for i in range(len(drawn))}
            save_new_annotations(rid, drawn, loc_names)
            st.session_state.pop(drawn_key, None)

        # Save named locations (no polygon — geocoded later)
        named = [n for n in st.session_state.get(named_key, []) if n.strip()]
        if named:
            save_new_annotations(rid, [], {}, named_only=named)
            st.session_state.pop(named_key, None)

        if drawn or named:
            load_annotations.clear()

        load_articles.clear()
        st.session_state.idx = min(total - 1, idx + 1)
        st.rerun()

with b2:
    if st.button("Skip for now"):
        st.session_state.idx = min(total - 1, idx + 1)
        st.rerun()
